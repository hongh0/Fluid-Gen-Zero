# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .modules.trajectory import replace_feature, get_video_track_video, create_pos_feature_map

from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanMove:

    @staticmethod
    def _flow_add_noise(x0, noise, timestep, num_train_timesteps):
        sigma = timestep.to(device=x0.device, dtype=x0.dtype) / num_train_timesteps
        while sigma.ndim < x0.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * x0 + sigma * noise

    def _prepare_reference_video(self, sim_video, frame_num, height, width):
        if not torch.is_tensor(sim_video):
            sim_video = torch.as_tensor(sim_video)
        sim_video = sim_video.to(device=self.device, dtype=torch.float32)
        if sim_video.ndim == 4 and sim_video.shape[0] in (1, 3):
            video = sim_video
        elif sim_video.ndim == 4 and sim_video.shape[-1] in (1, 3):
            video = sim_video.permute(3, 0, 1, 2)
        else:
            raise ValueError(
                "sim_video must have shape [C,T,H,W] or [T,H,W,C], "
                f"got {tuple(sim_video.shape)}")
        if video.max() > 1.5:
            video = video / 127.5 - 1.0
        elif video.min() >= 0:
            video = video * 2.0 - 1.0
        video = torch.nn.functional.interpolate(
            video.permute(1, 0, 2, 3), size=(height, width),
            mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        if video.shape[1] < frame_num:
            video = torch.cat([
                video, video[:, -1:].repeat(1, frame_num - video.shape[1], 1, 1)
            ], dim=1)
        return video[:, :frame_num]

    def _mask_to_latent(self, mask, frame_num, latent_t, latent_h, latent_w, soft=False, blur_sigma=16.0):
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=self.device, dtype=torch.float32)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.ndim != 4:
            raise ValueError(f"mask must be [T,H,W] or [1,T,H,W], got {tuple(mask.shape)}")
        mask = mask[:1]
        # Resize space and time separately, so clips of any length work.
        target_h = latent_h * self.vae_stride[1]
        target_w = latent_w * self.vae_stride[2]
        mask = torch.nn.functional.interpolate(
            mask, size=(target_h, target_w), mode="nearest").clamp(0, 1)
        if mask.shape[1] != frame_num:
            # Nearest-neighbour along T by gathering frames (memory-light), with
            # the index tensor on the mask's own device.
            src = torch.arange(frame_num, device=mask.device, dtype=torch.float32)
            src = (src * mask.shape[1] / float(frame_num)).long().clamp_(0, mask.shape[1] - 1)
            mask = mask.index_select(1, src).clamp(0, 1)
        if soft:
            blurred = []
            kernel = max(3, int(round(blur_sigma * 2)) * 2 + 1)
            for idx in range(mask.shape[1]):
                blurred.append(TF.gaussian_blur(mask[:, idx:idx + 1], [kernel, kernel], [blur_sigma, blur_sigma]))
            mask = torch.cat(blurred, dim=1)
        pad = (-mask.shape[1]) % self.vae_stride[0]
        if pad:
            mask = torch.cat([mask, mask[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        mask = mask.view(1, mask.shape[1] // self.vae_stride[0], self.vae_stride[0], mask.shape[2], mask.shape[3])
        mask = mask.amax(dim=2)
        return torch.nn.functional.interpolate(mask, size=(latent_h, latent_w), mode="nearest")[:, :latent_t]

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
        init_on_cpu=True,
    ):
        r"""
        Initializes the motion-controllable video generation model.

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
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
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
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir, torch_dtype=self.param_dtype)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 track,
                 track_visibility,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 eval_bench=True,
                 sim_video=None,
                 sim_mask=None,
                 fluid_mask=None,
                 guidance_start_step=3,
                 object_wrap_end_step=6,
                 fluid_wrap_end_step=None,
                 fluid_wrap_strength=0.8,
                 fluid_mask_sigma=16.0,
                 init_noise=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            track (numpy.array):
                Input track tensor. Shape: [b, F, track_num, 2]
            track_visibility (numpy.array):
                Input image tensor. Shape: [b, F, track_num]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            eval_bench (`bool`, *optional*, defaults to True):
                If use the data from MoveBench, set it to True

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        track = torch.from_numpy(track).to(self.device)
        track_visibility = torch.from_numpy(track_visibility).to(self.device)

        track = track[0] if track.ndim == 4 else track
        track_visibility = track_visibility[0] if track_visibility.ndim == 3 else track_visibility

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w

        if eval_bench:
            lat_h = h // self.vae_stride[1]
            lat_w = w // self.vae_stride[2]
        else:
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                self.patch_size[1] * self.patch_size[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                self.patch_size[2] * self.patch_size[2])
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]

        # Scale tracks to match the target video dimensions
        scale_h = h / img.shape[1]
        scale_w = w / img.shape[2]
        track[..., 0] *= scale_w
        track[..., 1] *= scale_h

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        # Keep the conditioning mask aligned with the requested clip length.
        # For frame_num=81 this yields latent_t=21; for frame_num=41 it yields latent_t=11.
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, latent_t, lat_h, lat_w]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
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

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]]) # torch.Size([1, 257, 1280])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
 
        # Edit y to enable motion control
        track_feats, track_pos = create_pos_feature_map(track, track_visibility, self.vae_stride, h, w, y.size(0), track_num=track_visibility.size(-1), device=y.device)
        track_feats = track_feats.permute(3, 0, 1, 2)
        edited_y = replace_feature(y.unsqueeze(0), track_pos.unsqueeze(0))[0]

        y_cond = torch.concat([msk, edited_y], dim=0)
        y_uncond = torch.concat([msk, y])

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
                timesteps = sample_scheduler.timesteps # torch.Size([40])
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

            # Optional simulation-guided latent-space guidance. The original trajectory
            # condition remains active through y_cond/y_uncond below.
            guidance_enabled = sim_video is not None
            ref_latents = None
            motion_mask = None
            soft_fluid_mask = None
            fixed_noise = None
            if guidance_enabled:
                sim_video_tensor = self._prepare_reference_video(sim_video, F, h, w)
                ref_latents = self.vae.encode([sim_video_tensor])[0].to(self.device)
                fixed_noise = noise if init_noise is None else init_noise.to(self.device)
                if fixed_noise.shape != ref_latents.shape:
                    raise ValueError(
                        f"init_noise shape {tuple(fixed_noise.shape)} does not match "
                        f"reference latent shape {tuple(ref_latents.shape)}")
                latent_t = ref_latents.shape[1]
                motion_mask = self._mask_to_latent(
                    sim_mask, F, latent_t, ref_latents.shape[2], ref_latents.shape[3])
                if motion_mask is None:
                    raise ValueError("sim_mask is required when sim_video is provided")
                if fluid_mask is not None:
                    soft_fluid_mask = self._mask_to_latent(
                        fluid_mask, F, latent_t, ref_latents.shape[2], ref_latents.shape[3],
                        soft=True, blur_sigma=fluid_mask_sigma)
                    soft_fluid_mask = soft_fluid_mask * (1.0 - motion_mask)
                guidance_start_step = max(0, min(int(guidance_start_step), len(timesteps) - 1))
                object_wrap_end_step = max(guidance_start_step, min(int(object_wrap_end_step), len(timesteps)))
                if fluid_wrap_end_step is None:
                    fluid_wrap_end_step = object_wrap_end_step
                fluid_wrap_end_step = max(guidance_start_step, min(int(fluid_wrap_end_step), len(timesteps)))
                latent = self._flow_add_noise(
                    ref_latents, fixed_noise, timesteps[guidance_start_step], self.num_train_timesteps)
                sampling_timesteps = timesteps[guidance_start_step:]
            else:
                latent = noise
                sampling_timesteps = timesteps

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y_cond],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y_uncond],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for step_offset, t in enumerate(tqdm(sampling_timesteps)):
                absolute_step = guidance_start_step + step_offset if guidance_enabled else step_offset
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                if guidance_enabled:
                    next_t = (
                        sampling_timesteps[step_offset + 1]
                        if step_offset + 1 < len(sampling_timesteps) else t
                    )
                    noisy_ref = self._flow_add_noise(
                        ref_latents, fixed_noise, next_t, self.num_train_timesteps)
                    if absolute_step < object_wrap_end_step:
                        # With offload_model on, the sampler keeps `latent` on the
                        # CPU between steps while the masks live on self.device.
                        mm = motion_mask.to(latent.device)
                        noisy_ref = noisy_ref.to(latent.device)
                        latent = mm * noisy_ref + (1.0 - mm) * latent
                    if soft_fluid_mask is not None and absolute_step < fluid_wrap_end_step:
                        soft_fluid_mask = soft_fluid_mask.to(latent.device)
                        duration = max(1, fluid_wrap_end_step - guidance_start_step)
                        alpha = float(fluid_wrap_strength) * max(
                            0.0, 1.0 - (absolute_step - guidance_start_step) / duration)
                        latent = (
                            alpha * soft_fluid_mask * noisy_ref
                            + (1.0 - alpha * soft_fluid_mask) * latent
                        )

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if eval_bench:
                videos = self.vae.decode(x0)
            else:
                if self.rank == 0:
                    videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if eval_bench:
            return videos[0]
        else:
            return videos[0] if self.rank == 0 else None
