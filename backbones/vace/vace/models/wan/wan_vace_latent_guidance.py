# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import gc
import math
import time
import random
import types
import logging
import traceback
from contextlib import contextmanager
from functools import partial

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
from .modules.model import VaceWanModel
from ..utils.preprocessor import VaceVideoProcessor


class WanVace(WanT2V):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward,
                                                            usp_dit_forward_vace)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            # Resize the reference image to the canvas (aspect ratio not preserved).
                            ref_img = F.interpolate(
                                ref_img.squeeze(1).unsqueeze(0), 
                                size=(canvas_height, canvas_width), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0).unsqueeze(1)
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return vae.decode(trimed_zs)



    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 init_noise=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_noise (`torch.Tensor`, *optional*, defaults to None):
                Pre-computed initial latent noise of shape (C, T, H, W).
                When provided (e.g. optical-flow-warped noise), this replaces the
                standard random Gaussian initialisation.  The tensor is cast to
                float32 on the model device automatically.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        #
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # vace context encode
        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        if init_noise is not None:
            # Use caller-supplied initial noise
            assert init_noise.shape == torch.Size(target_shape), (
                f"init_noise shape {init_noise.shape} != expected {target_shape}"
            )
            noise = [init_noise.to(device=self.device, dtype=torch.float32)]
        else:
            noise = [
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g)
            ]
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,**arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


class WanVaceLatentGuidance(WanVace):
    """WanVace extended with simulation-guided latent-space guidance.

    In addition to the standard VACE conditioning (src_video / src_mask /
    src_ref_images), this class accepts a *motion-signal* video (sim_video)
    together with a binary mask (sim_mask) that marks which spatial regions
    should follow the reference motion.  Region-aware latent wrapping blends the
    clean denoising trajectory with a progressively-denoised version of the
    motion-signal latent inside the masked region, giving strong structural
    guidance in the early timesteps and letting the model refine freely later.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flow_add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t) -> torch.Tensor:
        """Flow-matching forward process: x_t = (1 - sigma)*x0 + sigma*noise.

        Args:
            x0:    clean latent  [C, T', H', W']
            noise: Gaussian noise with same shape as x0
            t:     scheduler timestep (scalar tensor or float in [0, num_train_timesteps])
        """
        sigma = float(t) / float(self.num_train_timesteps)
        return (1.0 - sigma) * x0 + sigma * noise

    def _convert_sim_mask_to_latent(self, sim_mask: torch.Tensor) -> torch.Tensor:
        """Downsample a pixel-space binary mask to the latent grid.

        Args:
            sim_mask: [1, T, H, W]  float mask in [0, 1]
        Returns:
            [1, T', H', W']  binary mask in latent space, same device as input
        """
        t_stride, h_stride, w_stride = self.vae_stride
        mask = (sim_mask[:1] > 0.5).float()  # [1, T, H, W]

        # Temporal: first frame then every t_stride-th frame
        m0 = mask[:, 0:1, :, :]
        m_rest = mask[:, 1::t_stride, :, :]
        mask_t = torch.cat([m0, m_rest], dim=1)  # [1, T', H, W]

        # Spatial: nearest-neighbour downsample to latent resolution
        H, W = mask_t.shape[2], mask_t.shape[3]
        mask_lat = F.interpolate(
            mask_t.permute(1, 0, 2, 3),          # [T', 1, H, W]
            size=(H // h_stride, W // w_stride),
            mode='nearest',
        ).permute(1, 0, 2, 3)                    # [1, T', H', W']

        return mask_lat

    @staticmethod
    def _soften_mask_pixel(mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply a separable Gaussian blur to a [1, T, H, W] float mask.

        Blurring in pixel space before latent downsampling creates soft gradient
        boundaries instead of hard binary edges.  This is critical for fluid
        regions whose shape is inherently amorphous.

        Args:
            mask:  [1, T, H, W] float mask in [0, 1]
            sigma: standard deviation of the Gaussian kernel in pixels.
                   0 or negative → returns mask unchanged.
        Returns:
            [1, T, H, W] soft mask in [0, 1], same device/dtype as input
        """
        if sigma <= 0.0:
            return mask
        ksize = 2 * int(math.ceil(3.0 * sigma)) + 1
        x = torch.arange(ksize, dtype=torch.float32, device=mask.device) - ksize // 2
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        k2d = k1d[None, :] * k1d[:, None]               # [ksize, ksize]
        k2d = k2d.unsqueeze(0).unsqueeze(0)              # [1, 1, ksize, ksize]
        pad = ksize // 2
        _, T, H, W = mask.shape
        mask_flat = mask.permute(1, 0, 2, 3)             # [T, 1, H, W]
        blurred = F.conv2d(mask_flat.float(), k2d, padding=pad)  # [T, 1, H, W]
        return blurred.permute(1, 0, 2, 3).clamp(0.0, 1.0).to(mask.dtype)

    def _downsample_soft_mask_to_latent(self, mask: torch.Tensor) -> torch.Tensor:
        """Downsample a soft pixel-space mask [1, T, H, W] to latent resolution.

        Unlike _convert_sim_mask_to_latent (which uses nearest-exact and assumes
        a binary mask), this method uses bilinear interpolation to preserve the
        soft gradient values produced by _soften_mask_pixel.

        Args:
            mask: [1, T, H, W] soft mask in [0, 1]
        Returns:
            [1, T', H', W'] soft mask in [0, 1] at latent resolution
        """
        t_stride, h_stride, w_stride = self.vae_stride
        m0 = mask[:, 0:1, :, :]
        m_rest = mask[:, 1::t_stride, :, :]
        mask_t = torch.cat([m0, m_rest], dim=1)         # [1, T', H, W]
        H, W = mask_t.shape[2], mask_t.shape[3]
        mask_lat = F.interpolate(
            mask_t.permute(1, 0, 2, 3),                 # [T', 1, H, W]
            size=(H // h_stride, W // w_stride),
            mode='bilinear',
            align_corners=False,
        ).permute(1, 0, 2, 3).clamp(0.0, 1.0)          # [1, T', H', W']
        return mask_lat

    # ------------------------------------------------------------------
    # Source preparation
    # ------------------------------------------------------------------

    def prepare_simulation_reference(self, sim_video_path, sim_mask_path, num_frames, image_size, device):
        """Load and preprocess the warped reference video V^sim and its mask.

        Args:
            sim_video_path: path to the motion-signal video, or None
            sim_mask_path:  path to the binary motion mask video, or None
            num_frames:     target number of frames
            image_size:     (W, H) tuple (same convention as prepare_source)
            device:         target device

        Returns:
            (sim_video, sim_mask) tensors ready to pass to generate(), or
            (None, None) when sim_video_path is None.
        """
        if sim_video_path is None:
            return None, None

        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720 * 1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480 * 832:
            self.vid_proc.set_seq_len(32760)

        if sim_mask_path is not None:
            sim_video, sim_mask, _, _, _ = self.vid_proc.load_video_pair(sim_video_path, sim_mask_path)
            sim_video = sim_video.to(device)
            sim_mask = sim_mask.to(device)
            sim_mask = torch.clamp((sim_mask[:1, :, :, :] + 1) / 2, min=0, max=1)
        else:
            sim_video, _, _, _ = self.vid_proc.load_video(sim_video_path)
            sim_video = sim_video.to(device)
            T = sim_video.shape[1]
            sim_mask = torch.ones((1, T, sim_video.shape[2], sim_video.shape[3]),
                                  device=device, dtype=sim_video.dtype)

        return sim_video, sim_mask

    def prepare_fluid_mask(self, fluid_mask_path, image_size, device):
        """Load and preprocess a fluid mask video (independent of sim_video).

        The fluid mask encodes which spatial regions belong to fluid.  Unlike
        the object mask (sim_mask), this is stored as a standalone video file
        where bright pixels indicate fluid presence.

        Args:
            fluid_mask_path: path to the fluid mask video, or None
            image_size:      (W, H) tuple (same convention as prepare_source)
            device:          target device

        Returns:
            [1, T, H, W] float tensor in [0, 1] on *device*, or None.
        """
        if fluid_mask_path is None:
            return None

        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720 * 1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480 * 832:
            self.vid_proc.set_seq_len(32760)

        fluid_video, _, _, _ = self.vid_proc.load_video(fluid_mask_path)
        fluid_video = fluid_video.to(device)
        # vid_proc normalises pixel values to [-1, 1]; convert first channel to [0, 1]
        fluid_mask = torch.clamp((fluid_video[:1] + 1.0) / 2.0, 0.0, 1.0)
        return fluid_mask

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 sim_video=None,
                 sim_mask=None,
                 fluid_mask=None,
                 guidance_start_step=3,
                 object_wrap_end_step=6,
                 fluid_wrap_end_step=None,
                 fluid_wrap_strength=0.8,
                 fluid_mask_sigma=16.0,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 init_noise=None):
        r"""Generate a video with the native VACE condition plus latent-space guidance.

        All arguments match WanVace.generate(); the new ones are:

        Args:
            sim_video:           motion-signal video tensor [C, T, H, W], or None.
                                 Typically the optical-flow video that contains both
                                 object and fluid motion.
            sim_mask:            binary object mask [1, T, H, W] (1 = object region).
                                 Object region receives binary (hard) latent wrapping.
            fluid_mask:          fluid region mask [1, T, H, W] (1 = fluid pixel),
                                 or None.  Unlike the object mask this is softened
                                 with a Gaussian blur before latent conversion so
                                 that amorphous fluid boundaries are handled
                                 gracefully.
            guidance_start_step:         scheduler step index at which denoising starts.
                                 Latents are initialised from noisy sim_video at
                                 this noise level.
            object_wrap_end_step:       step index up to which the *object* masked region
                                 is blended with progressively-denoised sim_video
                                 latents (full replacement, alpha = 1.0).
            fluid_wrap_end_step: step index up to which the *fluid* masked region
                                 receives latent wrapping.  Defaults to object_wrap_end_step.
                                 Setting this smaller than object_wrap_end_step releases
                                 fluid guidance earlier so the model can generate
                                 realistic fluid appearance more freely.
            fluid_wrap_strength:   peak blend weight for the fluid region in [0, 1].
                                 1.0 = hard replace (same as object); 0.5 = 50 %
                                 blend.  The weight decays linearly from this value
                                 to 0 over the fluid wrapping window.
            fluid_mask_sigma:    standard deviation (pixels) of the Gaussian kernel
                                 applied to the fluid mask before latent downsampling.
                                 Creates soft gradient boundaries that match the
                                 amorphous nature of fluid.  Set to 0 to disable.
        """
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # ---- text encoding ------------------------------------------------
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [c.to(self.device) for c in context]
            context_null = [c.to(self.device) for c in context_null]

        # ---- VACE context encode ------------------------------------------
        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)   # z_dim (without mask channels)

        # ---- latent wrapping: determine which regions are active ----------------------
        # Object wrapping: hard 1:1 replacement in object mask region
        # Fluid wrapping: soft partial replacement with progressive decay
        object_wrap_enabled = sim_video is not None and sim_mask is not None
        fluid_wrap_enabled = sim_video is not None and fluid_mask is not None
        guidance_enabled = object_wrap_enabled or fluid_wrap_enabled

        if guidance_enabled:
            ref_latents_list = self.vae.encode([sim_video])
            ref_latents = ref_latents_list[0].to(device=self.device, dtype=torch.float32)
            # Fixed noise tied to the reference for consistent blending
            fixed_noise = torch.randn(
                ref_latents.shape,
                dtype=torch.float32, device=self.device, generator=seed_g)

        if object_wrap_enabled:
            motion_mask = self._convert_sim_mask_to_latent(sim_mask)
            motion_mask = motion_mask.to(device=self.device, dtype=torch.float32)
            background_mask = 1.0 - motion_mask
        else:
            # Dummy all-background mask (only used for fluid-only wrapping)
            motion_mask = None
            background_mask = None

        if fluid_wrap_enabled:
            # Gaussian-blur the fluid mask in pixel space → soft latent mask
            soft_fm = self._soften_mask_pixel(
                fluid_mask.to(device=self.device, dtype=torch.float32),
                fluid_mask_sigma)
            fluid_lat_mask = self._downsample_soft_mask_to_latent(soft_fm)
            # Fluid wrapping ends at k_fl (falls back to k_obj when not given)
            fluid_wrap_end_eff = (
                fluid_wrap_end_step if fluid_wrap_end_step is not None else object_wrap_end_step)
        else:
            fluid_lat_mask = None
            fluid_wrap_end_eff = 0

        # ---- initial Gaussian noise ---------------------------------------
        if init_noise is not None:
            noise = [init_noise.to(device=self.device, dtype=torch.float32)]
        else:
            noise = [torch.randn(
                target_shape[0], target_shape[1], target_shape[2], target_shape[3],
                dtype=torch.float32, device=self.device, generator=seed_g)]

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.patch_size[1] * self.patch_size[2]) *
            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            # ---- build scheduler ----------------------------------------
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1, use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # ---- simulation-reference initialisation ----------------------
            if guidance_enabled:
                guidance_start_step = max(0, min(guidance_start_step, len(timesteps) - 1))
                object_wrap_end_step = max(guidance_start_step, min(object_wrap_end_step, len(timesteps) - 1))
                fluid_wrap_end_eff = max(guidance_start_step, min(fluid_wrap_end_eff, len(timesteps) - 1))
                guidance_start_t = timesteps[guidance_start_step]
                noisy_ref = self._flow_add_noise(ref_latents, fixed_noise, guidance_start_t)
                latents = [noisy_ref.to(dtype=torch.float32)]
                start_idx = guidance_start_step
            else:
                latents = noise
                start_idx = 0

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # ---- denoising loop -----------------------------------------
            for step_i, t in enumerate(tqdm(timesteps[start_idx:])):
                latent_model_input = latents
                timestep = torch.stack([t])

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep,
                    vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep,
                    vace_context=z, vace_context_scale=context_scale, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                # ---- region-aware latent wrapping -------------------------
                abs_step = step_i + start_idx
                do_obj_blend = object_wrap_enabled and abs_step < object_wrap_end_step
                do_fluid_blend = fluid_wrap_enabled and abs_step < fluid_wrap_end_eff

                if do_obj_blend or do_fluid_blend:
                    next_idx = abs_step + 1
                    if next_idx < len(timesteps):
                        noisy_ref_next = self._flow_add_noise(
                            ref_latents, fixed_noise, timesteps[next_idx])
                    else:
                        noisy_ref_next = ref_latents
                    noisy_ref_next = noisy_ref_next.to(dtype=latents[0].dtype)

                    result = latents[0]

                    # Object region: hard 1:1 replacement (binary schedule)
                    if do_obj_blend:
                        result = (result * background_mask
                                  + noisy_ref_next * motion_mask)

                    # Fluid region: soft partial replacement with linear decay.
                    # Blend weight starts at fluid_wrap_strength and decays to 0 so
                    # that early steps provide strong structural guidance while
                    # later steps let the model refine fluid appearance freely.
                    # Fluid pixels that overlap with the object mask are excluded
                    # to avoid double-blending.
                    if do_fluid_blend:
                        fluid_window = max(1, fluid_wrap_end_eff - guidance_start_step)
                        step_in_window = abs_step - guidance_start_step
                        alpha = fluid_wrap_strength * max(
                            0.0, 1.0 - step_in_window / fluid_window)

                        # Exclude object region from fluid blend
                        obj_excl = motion_mask if object_wrap_enabled else torch.zeros_like(
                            fluid_lat_mask)
                        fluid_exclusive = fluid_lat_mask * (1.0 - obj_excl)

                        result = (result * (1.0 - fluid_exclusive * alpha)
                                  + noisy_ref_next * (fluid_exclusive * alpha))

                    latents = [result]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


class WanVaceMP(WanVace):
    def __init__(
            self,
            config,
            checkpoint_dir,
            use_usp=False,
            ulysses_size=None,
            ring_size=None
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_usp = use_usp
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        self.in_q_list = None
        self.out_q = None
        self.inference_pids = None
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.dynamic_load()

        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple([x * y for x, y in zip(config.vae_stride, config.patch_size)]),
            min_area=720 * 1280,
            max_area=720 * 1280,
            min_fps=config.sample_fps,
            max_fps=config.sample_fps,
            zero_start=True,
            seq_len=75600,
            keep_last=True)


    def dynamic_load(self):
        if hasattr(self, 'inference_pids') and self.inference_pids is not None:
            return
        gpu_infer = os.environ.get('LOCAL_WORLD_SIZE') or torch.cuda.device_count()
        pmi_rank = int(os.environ['RANK'])
        pmi_world_size = int(os.environ['WORLD_SIZE'])
        in_q_list = [torch.multiprocessing.Manager().Queue() for _ in range(gpu_infer)]
        out_q = torch.multiprocessing.Manager().Queue()
        initialized_events = [torch.multiprocessing.Manager().Event() for _ in range(gpu_infer)]
        context = mp.spawn(self.mp_worker, nprocs=gpu_infer, args=(gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, self), join=False)
        all_initialized = False
        while not all_initialized:
            all_initialized = all(event.is_set() for event in initialized_events)
            if not all_initialized:
                time.sleep(0.1)
        print('Inference model is initialized', flush=True)
        self.in_q_list = in_q_list
        self.out_q = out_q
        self.inference_pids = context.pids()
        self.initialized_events = initialized_events

    def transfer_data_to_cuda(self, data, device):
        if data is None:
            return None
        else:
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            elif isinstance(data, list):
                data = [self.transfer_data_to_cuda(subdata, device) for subdata in data]
            elif isinstance(data, dict):
                data = {key: self.transfer_data_to_cuda(val, device) for key, val in data.items()}
        return data

    def mp_worker(self, gpu, gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, work_env):
        try:
            world_size = pmi_world_size * gpu_infer
            rank = pmi_rank * gpu_infer + gpu
            print("world_size", world_size, "rank", rank, flush=True)

            torch.cuda.set_device(gpu)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )

            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.ring_size or 1,
                ulysses_degree=self.ulysses_size or 1
            )

            num_train_timesteps = self.config.num_train_timesteps
            param_dtype = self.config.param_dtype
            shard_fn = partial(shard_model, device_id=gpu)
            text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
                tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
                shard_fn=shard_fn if True else None)
            text_encoder.model.to(gpu)
            vae_stride = self.config.vae_stride
            patch_size = self.config.patch_size
            vae = WanVAE(
                vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                device=gpu)
            logging.info(f"Creating VaceWanModel from {self.checkpoint_dir}")
            model = VaceWanModel.from_pretrained(self.checkpoint_dir)
            model.eval().requires_grad_(False)

            if self.use_usp:
                from xfuser.core.distributed import get_sequence_parallel_world_size
                from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                                usp_dit_forward,
                                                                usp_dit_forward_vace)
                for block in model.blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                for block in model.vace_blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                model.forward = types.MethodType(usp_dit_forward, model)
                model.forward_vace = types.MethodType(usp_dit_forward_vace, model)
                sp_size = get_sequence_parallel_world_size()
            else:
                sp_size = 1

            dist.barrier()
            model = shard_fn(model)
            sample_neg_prompt = self.config.sample_neg_prompt

            torch.cuda.empty_cache()
            event = initialized_events[gpu]
            in_q = in_q_list[gpu]
            event.set()

            while True:
                item = in_q.get()
                input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale, \
                shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model, init_noise = item
                input_frames = self.transfer_data_to_cuda(input_frames, gpu)
                input_masks = self.transfer_data_to_cuda(input_masks, gpu)
                input_ref_images = self.transfer_data_to_cuda(input_ref_images, gpu)
                if init_noise is not None:
                    init_noise = init_noise.to(device=gpu, dtype=torch.float32)

                if n_prompt == "":
                    n_prompt = sample_neg_prompt
                seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
                seed_g = torch.Generator(device=gpu)
                seed_g.manual_seed(seed)

                context = text_encoder([input_prompt], gpu)
                context_null = text_encoder([n_prompt], gpu)

                # vace context encode
                z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, vae=vae)
                m0 = self.vace_encode_masks(input_masks, input_ref_images, vae_stride=vae_stride)
                z = self.vace_latent(z0, m0)

                target_shape = list(z0[0].shape)
                target_shape[0] = int(target_shape[0] / 2)
                if init_noise is not None:
                    noise = [init_noise]
                else:
                    noise = [
                        torch.randn(
                            target_shape[0],
                            target_shape[1],
                            target_shape[2],
                            target_shape[3],
                            dtype=torch.float32,
                            device=gpu,
                            generator=seed_g)
                    ]
                seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                    (patch_size[1] * patch_size[2]) *
                                    target_shape[1] / sp_size) * sp_size

                @contextmanager
                def noop_no_sync():
                    yield

                no_sync = getattr(model, 'no_sync', noop_no_sync)

                # evaluation mode
                with amp.autocast(dtype=param_dtype), torch.no_grad(), no_sync():

                    if sample_solver == 'unipc':
                        sample_scheduler = FlowUniPCMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=gpu, shift=shift)
                        timesteps = sample_scheduler.timesteps
                    elif sample_solver == 'dpm++':
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=gpu,
                            sigmas=sampling_sigmas)
                    else:
                        raise NotImplementedError("Unsupported solver.")

                    # sample videos
                    latents = noise

                    arg_c = {'context': context, 'seq_len': seq_len}
                    arg_null = {'context': context_null, 'seq_len': seq_len}

                    for _, t in enumerate(tqdm(timesteps)):
                        latent_model_input = latents
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        model.to(gpu)
                        noise_pred_cond = model(
                            latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[
                            0]
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,
                            **arg_null)[0]

                        noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond)

                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

                    torch.cuda.empty_cache()
                    x0 = latents
                    if rank == 0:
                        videos = self.decode_latent(x0, input_ref_images, vae=vae)

                del noise, latents
                del sample_scheduler
                if offload_model:
                    gc.collect()
                    torch.cuda.synchronize()
                if dist.is_initialized():
                    dist.barrier()

                if rank == 0:
                    out_q.put(videos[0].cpu())

        except Exception as e:
            trace_info = traceback.format_exc()
            print(trace_info, flush=True)
            print(e, flush=True)



    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 init_noise=None):

        input_data = (input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale,
                      shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model, init_noise)
        for in_q in self.in_q_list:
            in_q.put(input_data)
        value_output = self.out_q.get()

        return value_output
