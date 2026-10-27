import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as TT
from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from einops import rearrange
from PIL import Image
from torchvision.io import write_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import tqdm

from sat import mpu
from sat.arguments import set_random_seed
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint

from sample_video_latent_guidance_single import (
    flow_from_mask_videos,
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    prepare_guidance_state,
)


_GUIDANCE_PARAM_KEYS = [
    "guidance_start_step",
    "object_wrap_end_step",
    "fluid_wrap_end_step",
    "fluid_wrap_strength",
    "fluid_mask_sigma",
]


def _check_frame_num(frame_num: int) -> None:
    if frame_num < 5 or (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must be 4n+1 and >=5, got {frame_num}.")


def _latent_frames(frame_num: int) -> int:
    _check_frame_num(frame_num)
    return (frame_num - 1) // 4 + 1


def _read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames were read from video: {path}")
    return frames, fps


def _write_video_cv2(path: str | Path, frames: Sequence[np.ndarray], fps: float) -> str:
    if not frames:
        raise ValueError(f"Cannot write empty video: {path}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()
    return str(out)


def _slice_and_pad(
    frames: list[np.ndarray],
    start: int,
    actual_frame_num: int,
    frame_num: int,
    stream_name: str,
) -> list[np.ndarray]:
    if start < 0:
        raise ValueError(f"Invalid negative start frame for {stream_name}: {start}")
    if start >= len(frames):
        raise RuntimeError(f"Start frame {start} is outside {stream_name} length {len(frames)}")
    sliced = [frame.copy() for frame in frames[start : min(start + actual_frame_num, len(frames))]]
    if not sliced:
        raise RuntimeError(f"Clip slice for {stream_name} at {start} produced no frames.")
    while len(sliced) < actual_frame_num:
        sliced.append(sliced[-1].copy())
    while len(sliced) < frame_num:
        sliced.append(sliced[-1].copy())
    return sliced[:frame_num]


def _resize_frames_like(
    frames: list[np.ndarray],
    target_hw: tuple[int, int],
    interpolation: int,
) -> list[np.ndarray]:
    target_h, target_w = target_hw
    resized: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
        resized.append(frame)
    return resized


def _resolve_job_path(job: Dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = job.get(key)
        if value:
            return str(value)
    return None


def _build_clip_input_paths(
    job: Dict[str, Any],
    video_cache: Dict[str, tuple[list[np.ndarray], float]],
) -> tuple[Dict[str, str], float]:
    start = int(job["start_frame"])
    actual = int(job["actual_frame_num"])
    frame_num = int(job["frame_num"])
    save_dir = Path(job["save_dir"])
    input_dir = save_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    path_specs = [
        ("warped_video_path", ("warped_video_path", "sim_video"), "warped_video.mp4"),
        ("warped_mask", ("warped_mask", "sim_mask"), "warped_mask.mp4"),
        ("sim_fluid_mask", ("sim_fluid_mask", "fluid_mask"), "sim_fluid_mask.mp4"),
    ]
    paths: Dict[str, str] = {}
    sliced_frames: Dict[str, list[np.ndarray]] = {}
    fps = float(job.get("sampling_fps") or 8)
    for out_key, source_keys, filename in path_specs:
        source = _resolve_job_path(job, *source_keys)
        if not source:
            continue
        if not Path(source).exists():
            raise FileNotFoundError(f"{out_key} does not exist: {source}")
        if source not in video_cache:
            video_cache[source] = _read_video(source)
        frames, source_fps = video_cache[source]
        fps = source_fps or fps
        sliced = _slice_and_pad(frames, start, actual, frame_num, out_key)
        sliced_frames[out_key] = sliced
    warped_video_frames = sliced_frames.get("warped_video_path")
    warped_mask_frames = sliced_frames.get("warped_mask")
    if warped_video_frames and warped_mask_frames:
        target_hw = warped_video_frames[0].shape[:2]
        sliced_frames["warped_mask"] = _resize_frames_like(warped_mask_frames, target_hw, cv2.INTER_NEAREST)
    for out_key, _, filename in path_specs:
        sliced = sliced_frames.get(out_key)
        if sliced:
            paths[out_key] = _write_video_cv2(input_dir / filename, sliced, fps)
    if "warped_video_path" not in paths:
        raise ValueError(f"Clip {job.get('clip_id')} is missing warped_video_path/sim_video.")
    if "warped_mask" not in paths:
        raise ValueError(f"Clip {job.get('clip_id')} is missing warped_mask/sim_mask.")
    return paths, fps


def _read_first_frame_tensor(path: str | Path, image_size: list[int], transform, device: str = "cuda") -> torch.Tensor:
    frames = imageio.mimread(str(path), memtest=False)
    if len(frames) == 0:
        raise ValueError(f"Input image/video has no frames: {path}")
    frame = np.asarray(frames[0])
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    image = image.resize(tuple(reversed(image_size)))
    return transform(image).unsqueeze(0).to(device)


def _rgb_frame_tensor(frame: np.ndarray, image_size: list[int], transform, device: str = "cuda") -> torch.Tensor:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    image = image.resize(tuple(reversed(image_size)))
    return transform(image).unsqueeze(0).to(device)


def _anchor_latent(
    args,
    model,
    job: Dict[str, Any],
    previous_anchor_rgb: np.ndarray | None,
    image_size: list[int],
    transform,
    latent_frames: int,
    latent_channels: int,
    vae_stride: int,
) -> torch.Tensor:
    if previous_anchor_rgb is not None and not args.no_previous_anchor:
        image = _rgb_frame_tensor(previous_anchor_rgb, image_size, transform)
    else:
        video_path = job.get("video_path")
        if not video_path:
            raise ValueError(f"Clip {job.get('clip_id')} is missing video_path for the first-frame anchor.")
        if not Path(video_path).exists():
            raise FileNotFoundError(f"video_path does not exist: {video_path}")
        image = _read_first_frame_tensor(video_path, image_size, transform)

    image = image * 2.0 - 1.0
    image = image.unsqueeze(2).to(torch.bfloat16)
    model.first_stage_model.to("cuda")
    image = model.encode_first_stage(image, None)
    image = image.permute(0, 2, 1, 3, 4).contiguous()
    pad_shape = (
        image.shape[0],
        latent_frames - 1,
        latent_channels,
        image_size[0] // vae_stride,
        image_size[1] // vae_stride,
    )
    return torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)


def _prepare_video_flow(args, model, sim_mask_path: str | None, total_num_frames: int, image_size: list[int], device):
    points = None
    if args.no_flow_injection:
        return None, points
    mask_paths = []
    if sim_mask_path:
        print(f"Deriving object trajectory flow from sim_mask center: {sim_mask_path}")
        mask_paths.append(sim_mask_path)
    if getattr(args, "use_fluid_mask_center_trajectory", False) and getattr(args, "fluid_mask", None):
        print(f"Deriving fluid trajectory flow from fluid_mask center: {args.fluid_mask}")
        mask_paths.append(args.fluid_mask)
    if mask_paths:
        video_flow, points = flow_from_mask_videos(
            mask_paths,
            total_num_frames,
            image_size,
            device=device,
            threshold=args.sim_mask_threshold,
        )
        video_flow = video_flow.unsqueeze_(0)
    elif args.flow_path:
        print(f"Flow path: {args.flow_path}")
        video_flow = torch.load(args.flow_path, map_location=device, weights_only=True)[:total_num_frames].unsqueeze_(0)
    else:
        print("No flow injection")
        return None, points

    model.to("cpu")
    tmp = rearrange(video_flow[0], "T H W C -> T C H W")
    if tmp.shape[1] == 3:
        video_flow = resize(tmp.float(), image_size, interpolation=InterpolationMode.BILINEAR, antialias=True).unsqueeze(0).to("cuda")
    elif tmp.shape[1] == 1:
        video_flow = resize(tmp.float().repeat(1, 3, 1, 1), image_size, interpolation=InterpolationMode.BILINEAR, antialias=True).unsqueeze(0).to("cuda")
    else:
        video_flow = flow_to_image(tmp).unsqueeze_(0).to("cuda")
    del tmp
    video_flow = rearrange(video_flow / 255.0 * 2 - 1, "B T C H W -> B C T H W").contiguous().to(torch.bfloat16)
    video_flow = video_flow.repeat(2, 1, 1, 1, 1).contiguous()
    model.first_stage_model.to(device)
    torch.cuda.empty_cache()
    video_flow = model.encode_first_stage(video_flow, None)
    video_flow = video_flow.permute(0, 2, 1, 3, 4).contiguous()
    return video_flow, points


def _set_job_guidance_args(args, job: Dict[str, Any], paths: Dict[str, str]) -> None:
    args.sim_video = paths.get("warped_video_path")
    args.sim_mask = paths.get("warped_mask")
    args.fluid_mask = paths.get("sim_fluid_mask")
    for key in _GUIDANCE_PARAM_KEYS:
        value = (job.get("guidance_params") or {}).get(key)
        if value is not None:
            setattr(args, key, value)


def _save_clip_samples(samples: torch.Tensor, job: Dict[str, Any], fps: int, prompt: str) -> np.ndarray:
    save_file = Path(job["save_file"])
    save_file.parent.mkdir(parents=True, exist_ok=True)
    actual = int(job["actual_frame_num"])
    clip = samples[0]
    if clip.shape[0] < actual:
        pad = clip[-1:].repeat(actual - clip.shape[0], 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)
    clip = clip[:actual]
    x = rearrange(clip, "t c h w -> t h w c")
    x = x.mul(255).add(0.5).clamp(0, 255).to("cpu", torch.uint8)
    write_video(
        save_file,
        x,
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    prompt_dir = save_file.parent / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    return x[-1].numpy()


def _read_rgb_video(path: str | Path) -> list[np.ndarray]:
    frames = imageio.mimread(str(path), memtest=False)
    out = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        out.append(arr.astype(np.uint8))
    return out


def _write_rgb_video(path: str | Path, frames: Sequence[np.ndarray], fps: int) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return str(out)


def _stitch_clip_outputs(jobs: Sequence[Dict[str, Any]], output_path: str | Path, fps: int) -> str:
    stitched: list[np.ndarray] = []
    total_frame_num = max(int(job.get("end_frame", 0)) for job in jobs) + 1
    for idx, job in enumerate(jobs):
        path = job.get("save_file")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Missing clip output for stitching: {path}")
        frames = _read_rgb_video(path)
        expected = int(job.get("actual_frame_num") or len(frames))
        if len(frames) < expected:
            frames = frames + [frames[-1].copy() for _ in range(expected - len(frames))]
        elif len(frames) > expected:
            frames = frames[:expected]
        stitched.extend(frames if idx == 0 else frames[1:])
    if len(stitched) < total_frame_num and stitched:
        stitched.extend([stitched[-1].copy() for _ in range(total_frame_num - len(stitched))])
    stitched = stitched[:total_frame_num]
    return _write_rgb_video(output_path, stitched, fps)


def _load_jobs_from_manifest(path: str | Path) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    jobs = [dict(job) for job in manifest.get("tora_clip_jobs", [])]
    if not jobs:
        raise ValueError(f"No tora_clip_jobs found in {manifest_path}")
    for idx, job in enumerate(jobs):
        job.setdefault("clip_id", idx)
        job.setdefault("save_dir", str(manifest_path.parent / "tora_clips" / f"clip_{idx:03d}"))
        job.setdefault("save_file", str(Path(job["save_dir"]) / "out_video.mp4"))
        job["start_frame"] = int(job["start_frame"])
        if "actual_frame_num" not in job or job.get("actual_frame_num") is None:
            job["actual_frame_num"] = int(job.get("end_frame", job["start_frame"]) - job["start_frame"] + 1)
        job["actual_frame_num"] = int(job["actual_frame_num"])
        if "frame_num" not in job or job.get("frame_num") is None:
            job["frame_num"] = job["actual_frame_num"]
        job["frame_num"] = int(job["frame_num"])
        _check_frame_num(job["frame_num"])
        if job["frame_num"] < job["actual_frame_num"]:
            raise ValueError(f"Clip {idx} frame_num={job['frame_num']} is shorter than actual_frame_num={job['actual_frame_num']}.")
        job.setdefault("end_frame", job["start_frame"] + job["actual_frame_num"] - 1)
    return jobs, manifest


def run_multiclip(args, model_cls=SATVideoDiffusionEngine) -> Dict[str, Any]:
    if not args.manifest:
        raise ValueError("--manifest is required for Tora multiclip generation.")
    jobs, _ = _load_jobs_from_manifest(args.manifest)
    final_save_file = Path(args.final_save_file) if args.final_save_file else Path(args.manifest).parent / "out_video_dynamic_tora_clips.mp4"
    if args.dry_run:
        plan = {
            "mode": "tora_single_load_multiclip",
            "manifest": str(args.manifest),
            "final_save_file": str(final_save_file),
            "num_clips": len(jobs),
            "clips": [
                {
                    "clip_id": job.get("clip_id"),
                    "start_frame": job.get("start_frame"),
                    "end_frame": job.get("end_frame"),
                    "actual_frame_num": job.get("actual_frame_num"),
                    "frame_num": job.get("frame_num"),
                    "latent_num_frames": _latent_frames(int(job["frame_num"])),
                    "save_file": job.get("save_file"),
                    "prompt": job.get("prompt"),
                }
                for job in jobs
            ],
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan

    model = get_model(args, model_cls) if isinstance(model_cls, type) else model_cls
    load_checkpoint(model, args)
    model.eval()

    image_size = [480, 720]
    transform = TT.Compose([TT.ToTensor()])
    force_uc_zero_embeddings = ["txt"]
    num_samples = [1]
    device = model.device
    latent_channels = args.latent_channels
    vae_stride = 8
    previous_anchor_rgb: np.ndarray | None = None
    video_cache: Dict[str, tuple[list[np.ndarray], float]] = {}
    executed: list[Dict[str, Any]] = []

    with torch.no_grad():
        for idx, job in enumerate(tqdm(jobs)):
            frame_num = int(job["frame_num"])
            actual_frame_num = int(job["actual_frame_num"])
            latent_t = _latent_frames(frame_num)
            total_num_frames = frame_num
            prompt = str(job.get("prompt") or "")
            seed = int(job.get("seed", args.seed)) + idx
            set_random_seed(seed)
            print(f"[tora multiclip {idx:03d}/{len(jobs):03d}] frames {job['start_frame']}..{job['end_frame']} actual={actual_frame_num} frame_num={frame_num}")
            print(prompt)

            paths, _ = _build_clip_input_paths(job, video_cache)
            _set_job_guidance_args(args, job, paths)
            image = _anchor_latent(
                args,
                model,
                job,
                previous_anchor_rgb,
                image_size,
                transform,
                latent_t,
                latent_channels,
                vae_stride,
            )

            video_flow, points = _prepare_video_flow(args, model, args.sim_mask, total_num_frames, image_size, device)
            torch.cuda.empty_cache()
            guidance_state = prepare_guidance_state(args, model, total_num_frames, image_size, device)
            if guidance_state is not None:
                expected_shape = (latent_t, latent_channels, image_size[0] // vae_stride, image_size[1] // vae_stride)
                if tuple(guidance_state["ref_latents"].shape[1:]) != expected_shape:
                    raise ValueError(
                        f"simulation-reference latent shape {tuple(guidance_state['ref_latents'].shape[1:])} "
                        f"does not match sampling shape {expected_shape}."
                    )
            model.to(device)

            value_dict = {
                "prompt": prompt,
                "negative_prompt": "",
                "num_frames": torch.tensor(latent_t).unsqueeze(0),
            }
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                num_samples,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            for key in c:
                if key != "crossattn":
                    c[key], uc[key] = map(lambda y: y[key][: math.prod(num_samples)].to("cuda"), (c, uc))
            c["concat"] = image
            uc["concat"] = image

            model.to(device)
            samples_z = model.sample(
                c,
                uc=uc,
                batch_size=1,
                shape=(latent_t, latent_channels, image_size[0] // vae_stride, image_size[1] // vae_stride),
                video_flow=video_flow,
                guidance_state=guidance_state,
            )
            samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

            model.to("cpu")
            torch.cuda.empty_cache()
            first_stage_model = model.first_stage_model.to(device)
            latent = 1.0 / model.scale_factor * samples_z
            recons = []
            loop_num = (latent_t - 1) // 2
            for i in range(loop_num):
                if i == 0:
                    start_frame, end_frame = 0, 3
                else:
                    start_frame, end_frame = i * 2 + 1, i * 2 + 3
                clear_fake_cp_cache = i == loop_num - 1
                recon = first_stage_model.decode(
                    latent[:, :, start_frame:end_frame].contiguous(),
                    clear_fake_cp_cache=clear_fake_cp_cache,
                )
                recons.append(recon)
            if not recons:
                raise ValueError(f"Cannot decode latent_t={latent_t}; use a larger frame_num for Tora.")
            recon = torch.cat(recons, dim=2).to(torch.float32)
            samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

            if mpu.get_model_parallel_rank() == 0:
                previous_anchor_rgb = _save_clip_samples(samples, job, fps=args.sampling_fps, prompt=prompt)
            executed.append(
                {
                    "clip_id": job.get("clip_id"),
                    "start_frame": job.get("start_frame"),
                    "end_frame": job.get("end_frame"),
                    "actual_frame_num": actual_frame_num,
                    "frame_num": frame_num,
                    "latent_num_frames": latent_t,
                    "input_paths": paths,
                    "save_file": job.get("save_file"),
                    "seed": seed,
                    "trajectory_points": points,
                }
            )

            del samples_z, samples_x, samples, video_flow, latent, recon, recons, c, uc, batch, batch_uc, guidance_state, image
            gc.collect()
            torch.cuda.empty_cache()

    stitched = None
    if mpu.get_model_parallel_rank() == 0:
        stitched = _stitch_clip_outputs(jobs, final_save_file, fps=args.sampling_fps)
        print(f"Saved stitched Tora multiclip video to {stitched}")

    report = {
        "mode": "tora_single_load_multiclip",
        "manifest": str(args.manifest),
        "num_clips": len(jobs),
        "final_save_file": str(final_save_file),
        "stitched_output": stitched,
        "executed_jobs": executed,
    }
    if mpu.get_model_parallel_rank() == 0:
        report_path = Path(args.manifest).parent / "tora_multiclip_generation_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved Tora multiclip generation report to {report_path}")
    return report


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
    py_parser.add_argument("--manifest", default=None)
    py_parser.add_argument("--final_save_file", default=None)
    py_parser.add_argument("--dry_run", action="store_true")
    py_parser.add_argument("--no_previous_anchor", action="store_true")
    py_parser.add_argument("--use_fluid_mask_center_trajectory", action="store_true")
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    if hasattr(args, "deepspeed_config"):
        del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    args.model_config.en_and_decode_n_samples_a_time = 1

    try:
        run_multiclip(args, model_cls=SATVideoDiffusionEngine)
    finally:
        _cleanup_runtime()
