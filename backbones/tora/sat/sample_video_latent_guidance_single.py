import argparse
import gc
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import List, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from einops import rearrange, repeat
from omegaconf import ListConfig
from PIL import Image
from torchvision.io import write_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import tqdm
from utils.flow_utils import process_traj, get_flow
from utils.flow_utils import blur_kernel
from utils.misc import vis_tensor

from sat import mpu
from sat.arguments import set_random_seed
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def read_from_json(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        data = [data]

    for cnt, item in enumerate(data):
        if cnt % world_size != rank:
            continue
        yield item, cnt


def resolve_json_path(path, base_dir):
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def is_enabled(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def _sample_indices(num_frames, target_frames):
    if num_frames <= 0:
        raise ValueError("Input video has no frames.")
    if num_frames >= target_frames:
        return np.linspace(0, num_frames - 1, target_frames).round().astype(np.int64)
    pad = np.full(target_frames - num_frames, num_frames - 1, dtype=np.int64)
    return np.concatenate([np.arange(num_frames, dtype=np.int64), pad])


def _read_video_frames(path, total_num_frames, image_size, rgb=True):
    path = str(path)
    if path.endswith((".pt", ".pth")):
        frames = torch.load(path, map_location="cpu", weights_only=True)
        if frames.dim() == 5:
            frames = frames[0]
        if frames.dim() != 4:
            raise ValueError(f"Expected video tensor with 4 or 5 dims, got {tuple(frames.shape)} from {path}")
        if frames.shape[0] in (1, 3):
            frames = frames.permute(1, 2, 3, 0)
        elif frames.shape[1] in (1, 3):
            frames = frames.permute(0, 2, 3, 1)
        frames = frames.detach().cpu().float()
        if frames.max() <= 1.5 and frames.min() >= -1.5:
            frames = (frames + 1.0) * 127.5 if frames.min() < 0 else frames * 255.0
        frames = frames.clamp(0, 255).to(torch.uint8).numpy()
    else:
        frames = np.asarray(imageio.mimread(path, memtest=False))

    if frames.ndim == 3:
        frames = frames[..., None]
    if rgb and frames.shape[-1] == 4:
        frames = frames[..., :3]
    if rgb and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if not rgb and frames.shape[-1] > 1:
        frames = frames[..., :3].mean(axis=-1, keepdims=True)

    frames = frames[_sample_indices(frames.shape[0], total_num_frames)]
    tensor = torch.from_numpy(frames).float()
    tensor = rearrange(tensor, "t h w c -> t c h w")
    mode = InterpolationMode.BILINEAR if rgb else InterpolationMode.NEAREST
    tensor = resize(tensor, image_size, interpolation=mode, antialias=True if rgb else False)
    tensor = rearrange(tensor, "t c h w -> t h w c").contiguous()
    return tensor


def load_sim_video_condition(path, total_num_frames, image_size, device):
    video = _read_video_frames(path, total_num_frames, image_size, rgb=True)
    return video.to(device).unsqueeze(0)


def load_mask_video(path, total_num_frames, image_size, device="cpu", threshold=0.5):
    mask = load_gray_video(path, total_num_frames, image_size)
    mask = (mask >= threshold).to(torch.float32)
    return mask.to(device)


def load_gray_video(path, total_num_frames, image_size):
    return _read_video_frames(path, total_num_frames, image_size, rgb=False)[..., 0].float() / 255.0


def load_first_frame(path, image_size, transform):
    frames = imageio.mimread(path, memtest=False)
    if len(frames) == 0:
        raise ValueError(f"Input video has no frames: {path}")
    frame = np.asarray(frames[0])
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    image = image.resize(tuple(reversed(image_size)))
    return transform(image).unsqueeze(0).to("cuda")


def _points_from_mask(mask):
    points = []
    last_point = None
    h, w = mask.shape[-2:]
    fallback = [w // 2, h // 2]
    for frame in mask:
        ys, xs = torch.where(frame > 0)
        if xs.numel() > 0:
            point = [int(torch.round(xs.float().mean()).item()), int(torch.round(ys.float().mean()).item())]
            last_point = point
        else:
            point = last_point if last_point is not None else fallback
        points.append(point)
    return points


def flow_from_mask_videos(mask_paths, total_num_frames, image_size, device, threshold=0.5):
    optical_flow = np.zeros((total_num_frames, image_size[0], image_size[1], 2), dtype=np.float32)
    processed_points = []
    loaded_masks = [
        load_mask_video(mask_path, total_num_frames, image_size, device="cpu", threshold=threshold)
        for mask_path in mask_paths
    ]

    for mask in loaded_masks:
        points = _points_from_mask(mask)
        optical_flow = get_flow(points, optical_flow, video_len=total_num_frames)
        processed_points.append(points)

    for i in range(1, total_num_frames):
        optical_flow[i] = cv2.filter2D(optical_flow[i], -1, blur_kernel)

    return torch.tensor(optical_flow).to(device), processed_points


def _blur_mask_video(mask, sigma):
    if sigma is None or sigma <= 0:
        return mask
    blurred = []
    for frame in mask.cpu().numpy():
        blurred.append(cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma))
    return torch.from_numpy(np.stack(blurred, axis=0)).to(mask.device, dtype=mask.dtype)


def _mask_to_latent(mask, latent_shape, hard=True):
    # mask: [T, H, W], latent_shape: [B, T_latent, C, H_latent, W_latent]
    target_t, target_h, target_w = latent_shape[1], latent_shape[3], latent_shape[4]
    mask = mask[None, None].float()
    if hard:
        mask = F.interpolate(mask, size=(target_t, target_h, target_w), mode="nearest")
    else:
        mask = F.interpolate(mask, size=(target_t, target_h, target_w), mode="trilinear", align_corners=False)
    return mask.permute(0, 2, 1, 3, 4).contiguous()


def prepare_guidance_state(args, model, total_num_frames, image_size, device):
    if args.sim_video is None:
        return None
    if args.sim_mask is None and args.fluid_mask is None:
        raise ValueError("--sim_video simulation reference requires --sim_mask or --fluid_mask.")
    if args.object_wrap_end_step < args.guidance_start_step:
        raise ValueError("--object_wrap_end_step must be >= --guidance_start_step.")
    if args.fluid_wrap_end_step is not None and args.fluid_wrap_end_step < args.guidance_start_step:
        raise ValueError("--fluid_wrap_end_step must be >= --guidance_start_step.")

    print(f"simulation reference sim_video: {args.sim_video}")
    sim_video = load_sim_video_condition(args.sim_video, total_num_frames, image_size, device=device)
    sim_video = rearrange(sim_video / 255.0 * 2 - 1, "b t h w c -> b c t h w").contiguous().to(torch.bfloat16)

    model.to("cpu")
    model.first_stage_model.to(device)
    torch.cuda.empty_cache()
    ref_latents = model.encode_first_stage(sim_video, None)
    ref_latents = ref_latents.permute(0, 2, 1, 3, 4).contiguous().float()

    motion_mask = None
    if args.sim_mask is not None:
        sim_mask = load_mask_video(
            args.sim_mask,
            total_num_frames,
            image_size,
            device=device,
            threshold=args.sim_mask_threshold,
        )
        motion_mask = _mask_to_latent(sim_mask, ref_latents.shape, hard=True)

    fluid_mask = None
    if args.fluid_mask is not None:
        fluid = load_gray_video(args.fluid_mask, total_num_frames, image_size).to(device)
        fluid = _blur_mask_video(fluid, args.fluid_mask_sigma).clamp(0, 1)
        fluid_mask = _mask_to_latent(fluid, ref_latents.shape, hard=False)

    fixed_noise = torch.randn_like(ref_latents)
    fluid_wrap_end = args.fluid_wrap_end_step if args.fluid_wrap_end_step is not None else args.object_wrap_end_step
    return {
        "ref_latents": ref_latents,
        "fixed_noise": fixed_noise,
        "motion_mask": motion_mask,
        "fluid_mask": fluid_mask,
        "guidance_start_step": args.guidance_start_step,
        "object_wrap_end_step": args.object_wrap_end_step,
        "fluid_wrap_end_step": fluid_wrap_end,
        "fluid_wrap_strength": args.fluid_wrap_strength,
    }


def draw_points(video, points):
    """
    Draw points onto video frames.

    Parameters:
        video (torch.tensor): Video tensor with shape [T, H, W, C], where T is the number of frames,
                            H is the height, W is the width, and C is the number of channels.
        points (list): Positions of points to be drawn as a tensor with shape [N, T, 2],
                            each point contains x and y coordinates.

    Returns:
        torch.tensor: The video tensor after drawing points, maintaining the same shape [T, H, W, C].
    """

    T = video.shape[0]
    N = len(points)
    device = video.device
    dtype = video.dtype
    video = video.cpu().numpy().copy()
    traj = np.zeros(video.shape[-3:], dtype=np.uint8)  # [H, W, C]
    for n in range(N):
        for t in range(1, T):
            cv2.line(traj, tuple(points[n][t - 1]), tuple(points[n][t]), (255, 1, 1), 2)
    for t in range(T):
        mask = traj[..., -1] > 0
        mask = repeat(mask, "h w -> h w c", c=3)
        alpha = 0.7
        video[t][mask] = video[t][mask] * (1 - alpha) + traj[mask] * alpha
        for n in range(N):
            cv2.circle(video[t], tuple(points[n][t]), 3, (160, 230, 100), -1)
    video = torch.from_numpy(video).to(device, dtype)
    return video


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor,
    save_path: str,
    name: str,
    fps: int = 5,
    args=None,
    key=None,
    traj_points=None,
    prompt="",
):
    os.makedirs(save_path, exist_ok=True)
    p = Path(save_path)
    saved_paths = []

    for i, vid in enumerate(video_batch):
        x = rearrange(vid, "t c h w -> t h w c")
        x = x.mul(255).add(0.5).clamp(0, 255).to("cpu", torch.uint8)  # [T H W C]
        os.makedirs(p / "video", exist_ok=True)
        os.makedirs(p / "prompt", exist_ok=True)
        if traj_points is not None:
            os.makedirs(p / "traj", exist_ok=True)
            os.makedirs(p / "traj_video", exist_ok=True)
            write_video(
                p / "video" / f"{name}_{i:06d}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
            saved_paths.append(f"video/{name}_{i:06d}.mp4")
            with open(p / "traj" / f"{name}_{i:06d}.pkl", "wb") as f:
                pickle.dump(traj_points, f)
            x = draw_points(x, traj_points)
            write_video(
                p / "traj_video" / f"{name}_{i:06d}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
        else:
            write_video(
                p / "video" / f"{name}_{i:06d}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
            saved_paths.append(f"video/{name}_{i:06d}.mp4")
        with open(p / "prompt" / f"{name}_{i:06d}.txt", "w") as f:
            f.write(prompt)
    return saved_paths[0] if len(saved_paths) == 1 else saved_paths


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
        use_json_input = False
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
        use_json_input = False
    elif args.input_type == "json":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_json(args.input_file, rank=rank, world_size=world_size)
        use_json_input = True
    else:
        raise NotImplementedError

    if use_json_input:
        json_base_dir = getattr(args, "video_base_dir", None) or os.path.dirname(os.path.abspath(args.input_file))
        output_json_path = getattr(args, "output_json_path", None)
        if output_json_path is None:
            output_json_path = os.path.join(args.output_dir, "tora_latent_results.json")
        with open(args.input_file, "r") as f:
            original_json_data = json.load(f)
        if not isinstance(original_json_data, list):
            original_json_data = [original_json_data]
        output_json_data = [item.copy() for item in original_json_data]
        os.makedirs(args.output_dir, exist_ok=True)

    image_size = [480, 720]

    if args.image2video or use_json_input:
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device
    with torch.no_grad():
        for item_or_text, cnt in tqdm(data_iter):
            if use_json_input:
                item = item_or_text
                text = item["caption"]
                video_path = resolve_json_path(item["video_path"], json_base_dir)
                args.sim_video = resolve_json_path(item["warped_video_path"], json_base_dir)
                args.sim_mask = resolve_json_path(item["warped_mask"], json_base_dir)
                args.fluid_mask = resolve_json_path(item.get("sim_fluid_mask"), json_base_dir)
                print(f"Processing JSON item {cnt}: {item['video_path']}")
                print(f"  sim_video={args.sim_video}")
                print(f"  sim_mask={args.sim_mask}")
                print(f"  fluid_mask={args.fluid_mask}")
            else:
                text = item_or_text
                video_path = None
            print(text)
            set_random_seed(args.seed)
            if use_json_input:
                image = load_first_frame(video_path, image_size, transform)
                image = image * 2.0 - 1.0
                image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None)
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                pad_shape = (image.shape[0], T - 1, C, H // F, W // F)
                image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
                flow_from_prompt = False
            elif args.image2video:
                flow_from_prompt = is_enabled(args.flow_from_prompt)
                if flow_from_prompt:
                    text, image_path, flow_files = text.strip().split("@@@")
                    print(flow_files, image_path)
                else:
                    text, image_path = text.split("@@")
                image_path = os.path.join(args.img_dir, image_path)
                assert os.path.exists(image_path), image_path
                image = Image.open(image_path).convert("RGB")
                image = image.resize(tuple(reversed(image_size)))
                image = transform(image).unsqueeze(0).to("cuda")
                image = image * 2.0 - 1.0
                image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None)
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                pad_shape = (image.shape[0], T - 1, C, H // F, W // F)
                image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
            else:
                image = None
                flow_from_prompt = is_enabled(args.flow_from_prompt)
                if flow_from_prompt:
                    text, flow_files = text.split("\t")
            total_num_frames = (T - 1) * 4 + 1  # T is the video latent size, 13 * 4 = 52
            points = None
            if args.no_flow_injection:
                video_flow = None
            else:
                mask_paths = []
                if args.sim_mask:
                    print(f"Deriving object trajectory flow from sim_mask center: {args.sim_mask}")
                    mask_paths.append(args.sim_mask)
                if getattr(args, "use_fluid_mask_center_trajectory", False) and args.fluid_mask:
                    print(f"Deriving fluid trajectory flow from fluid_mask center: {args.fluid_mask}")
                    mask_paths.append(args.fluid_mask)
            if not args.no_flow_injection and locals().get("mask_paths"):
                video_flow, points = flow_from_mask_videos(
                    mask_paths,
                    total_num_frames,
                    image_size,
                    device=device,
                    threshold=args.sim_mask_threshold,
                )
                video_flow = video_flow.unsqueeze_(0)
            elif flow_from_prompt:
                assert args.flow_path is not None, "Flow path must be provided if flow_from_prompt is True"
                p = os.path.join(args.flow_path, flow_files)
                print(f"Flow path: {p}")
                video_flow = (
                    torch.load(p, map_location="cpu", weights_only=True)[:total_num_frames].unsqueeze_(0).cuda()
                )
            elif args.flow_path:
                print(f"Flow path: {args.flow_path}")
                video_flow = torch.load(args.flow_path, map_location=device, weights_only=True)[
                    :total_num_frames
                ].unsqueeze_(0)
            elif args.point_path:
                if type(args.point_path) == str:
                    args.point_path = json.loads(args.point_path)
                video_flow, points = process_traj(args.point_path, total_num_frames, image_size, device=device)

                video_flow = video_flow.unsqueeze_(0)
            else:
                print("No flow injection")
                video_flow = None

            if video_flow is not None:
                model.to("cpu")  # move model to cpu, run vae on gpu only.
                tmp = rearrange(video_flow[0], "T H W C -> T C H W")
                if tmp.shape[1] == 3:
                    video_flow = resize(
                        tmp.float(),
                        image_size,
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    ).unsqueeze(0).to("cuda")
                elif tmp.shape[1] == 1:
                    video_flow = resize(
                        tmp.float().repeat(1, 3, 1, 1),
                        image_size,
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    ).unsqueeze(0).to("cuda")
                else:
                    video_flow = flow_to_image(tmp).unsqueeze_(0).to("cuda")  # [1 T C H W]
                if args.vis_traj_features:
                    os.makedirs("samples_flow/flow", exist_ok=True)
                    vis_tensor(tmp, *tmp.shape[-2:], "samples_flow/flow/flow1_vis.gif")
                    imageio.mimwrite(
                        "samples_flow/flow/flow2_vis.gif",
                        rearrange(video_flow[0], "T C H W -> T H W C").cpu(),
                        fps=8,
                        loop=0,
                    )
                del tmp
                video_flow = (
                    rearrange(video_flow / 255.0 * 2 - 1, "B T C H W -> B C T H W").contiguous().to(torch.bfloat16)
                )
                video_flow = video_flow.repeat(2, 1, 1, 1, 1).contiguous()  # for uncondition
                model.first_stage_model.to(device)
                torch.cuda.empty_cache()
                video_flow = model.encode_first_stage(video_flow, None)
                video_flow = video_flow.permute(0, 2, 1, 3, 4).contiguous()
            torch.cuda.empty_cache()
            guidance_state = prepare_guidance_state(args, model, total_num_frames, image_size, device)
            if guidance_state is not None:
                expected_shape = (T, C, H // F, W // F)
                if tuple(guidance_state["ref_latents"].shape[1:]) != expected_shape:
                    raise ValueError(
                        f"simulation-reference latent shape {tuple(guidance_state['ref_latents'].shape[1:])} "
                        f"does not match sampling shape {expected_shape}."
                    )
            model.to(device)

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            if image is not None:
                c["concat"] = image
                uc["concat"] = image

            for index in range(args.num_samples_per_prompt):
                if cnt > 0:
                    args.seed = np.random.randint(1e6)
                    set_random_seed(args.seed)
                # reload model on GPU
                model.to(device)
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    video_flow=video_flow,
                    guidance_state=guidance_state,
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                # Unload the model from GPU to save GPU memory
                model.to("cpu")
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                save_path = args.output_dir
                if use_json_input:
                    video_stem = Path(item["video_path"]).stem
                    output_idx = item.get("_tora_input_index", cnt)
                    name = f"{output_idx}_{video_stem}_{index}_seed{args.seed}"
                else:
                    name = str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:60] + f"_{index}_seed{args.seed}"
                if (not use_json_input) and flow_from_prompt:
                    name = Path(flow_files).stem
                if mpu.get_model_parallel_rank() == 0:
                    generated_rel_path = save_video_as_grid_and_mp4(
                        samples,
                        save_path,
                        name,
                        fps=args.sampling_fps,
                        traj_points=locals().get("points", None),
                        prompt=text,
                    )
                    if use_json_input and index == 0:
                        if isinstance(generated_rel_path, list):
                            generated_rel_path = generated_rel_path[0]
                        generated_abs_path = os.path.abspath(os.path.join(args.output_dir, generated_rel_path))
                        output_json_data[cnt]["tora_latent_video_path"] = generated_abs_path
                        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                        with open(output_json_path, "w") as f:
                            json.dump(output_json_data, f, indent=2, ensure_ascii=False)
                        print(f"Saved JSON update: {output_json_path}")
            del samples_z, samples_x, samples, video_flow, latent, recon, recons, c, uc, batch, batch_uc, guidance_state
            gc.collect()


def _cleanup_runtime() -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            try:
                mpu.destroy_model_parallel()
            except Exception:
                pass
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--use_fluid_mask_center_trajectory", action="store_true")
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    args.model_config.en_and_decode_n_samples_a_time = 1

    try:
        sampling_main(args, model_cls=SATVideoDiffusionEngine)
    finally:
        _cleanup_runtime()
