# -*- coding: utf-8 -*-
"""Generate all planned clips with VACE + latent-space guidance, loading WanVaceLatentGuidance once.

This script reads long condition videos, slices each planned clip, keeps the
generated last frame as the next clip anchor, and calls WanVaceLatentGuidance.generate
multiple times with one model load.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_VIDEO_KEYS = ["src_video", "src_mask", "sim_video", "sim_mask", "fluid_mask"]
_GUIDANCE_PARAM_KEYS = [
    "guidance_start_step",
    "object_wrap_end_step",
    "fluid_wrap_end_step",
    "fluid_wrap_strength",
    "fluid_mask_sigma",
]
_DEFAULT_GUIDANCE_PARAMS = {
    "guidance_start_step": 8,
    "object_wrap_end_step": 11,
    "fluid_wrap_end_step": 10,
    "fluid_wrap_strength": 0.8,
    "fluid_mask_sigma": 16.0,
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _check_frame_num(frame_num: int) -> None:
    if frame_num < 5:
        raise ValueError(f"frame_num must be at least 5, got {frame_num}.")
    if (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must satisfy 4n+1, got {frame_num}.")


def _next_4n_plus_1(length: int) -> int:
    if length < 2:
        raise ValueError(f"clip length must be at least 2, got {length}.")
    return max(5, ((length - 1 + 3) // 4) * 4 + 1)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
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


def _read_optional_video(path: str | Path | None) -> tuple[list[np.ndarray] | None, float | None]:
    if path is None:
        return None, None
    return _read_video(path)


def _write_video(path: str | Path, frames: Sequence[np.ndarray], fps: float, fourcc: str = "mp4v") -> str:
    if not frames:
        raise ValueError(f"Cannot write empty video: {path}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
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


def _cache_video_from_bgr_frames(path: str | Path, frames: Sequence[np.ndarray], fps: float) -> str:
    import torch
    from wan.utils.utils import cache_video

    if not frames:
        raise ValueError(f"Cannot write empty video: {path}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    tensor = torch.from_numpy(np.stack(rgb_frames)).permute(3, 0, 1, 2)
    tensor = tensor.float().div_(127.5).sub_(1.0)
    cache_video(
        tensor=tensor[None],
        save_file=str(out),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return str(out)


def _black_like(frame: np.ndarray) -> np.ndarray:
    return np.zeros(frame.shape[:2], dtype=np.uint8)


def _white_like(frame: np.ndarray) -> np.ndarray:
    return np.full(frame.shape[:2], 255, dtype=np.uint8)


def _resize_like(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == ref.shape[:2]:
        return frame
    return cv2.resize(frame, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)


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
    if len(sliced) > frame_num:
        sliced = sliced[:frame_num]
    return sliced


def _load_json_value(value: str | None) -> Any:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def _load_clip_prompts(value: str | None, num_clips: int, fallback: str) -> list[str]:
    data = _load_json_value(value)
    if data is None:
        return [fallback for _ in range(num_clips)]
    if isinstance(data, dict):
        prompts = [str(data.get(str(i), data.get(i, fallback))) for i in range(num_clips)]
    elif isinstance(data, list):
        prompts = [str(data[i]) if i < len(data) else fallback for i in range(num_clips)]
    else:
        raise ValueError("--clip_prompts_json must be a JSON list or dict.")
    return prompts


def _build_raw_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.src_video:
        raise ValueError("Raw multiclip mode requires --src_video.")
    if not args.save_dir:
        raise ValueError("Raw multiclip mode requires --save_dir.")
    if args.total_frame_num is None:
        raise ValueError("Raw multiclip mode requires --total_frame_num.")

    clip_frame_nums = _parse_int_list(args.clip_frame_nums)
    clip_start_frames = _parse_int_list(args.clip_start_frames)
    if clip_frame_nums is None and clip_start_frames is None:
        raise ValueError("Provide --clip_frame_nums or --clip_start_frames in raw mode.")

    specs: list[tuple[int, int, int]] = []
    total = int(args.total_frame_num)
    if clip_start_frames is not None:
        if sorted(clip_start_frames) != clip_start_frames or clip_start_frames[0] != 0:
            raise ValueError("--clip_start_frames must be sorted and start with 0.")
        for idx, start in enumerate(clip_start_frames):
            next_start = clip_start_frames[idx + 1] if idx + 1 < len(clip_start_frames) else total
            actual = next_start - start + (0 if idx + 1 == len(clip_start_frames) else 1)
            if actual < 2:
                raise ValueError(f"Clip {idx} inferred invalid actual length {actual}.")
            frame_num = clip_frame_nums[idx] if clip_frame_nums and idx < len(clip_frame_nums) else _next_4n_plus_1(actual)
            specs.append((start, actual, frame_num))
    else:
        assert clip_frame_nums is not None
        start = 0
        for frame_num in clip_frame_nums:
            actual = min(frame_num, total - start)
            if actual < 2:
                break
            specs.append((start, actual, frame_num))
            start += actual - 1
            if start >= total - 1:
                break

    prompts = _load_clip_prompts(args.clip_prompts_json, len(specs), args.prompt or "")
    output_root = Path(args.save_dir)
    jobs: list[dict[str, Any]] = []
    for idx, (start, actual, frame_num) in enumerate(specs):
        _check_frame_num(frame_num)
        if frame_num < actual:
            raise ValueError(f"Clip {idx} frame_num={frame_num} is shorter than actual_frame_num={actual}.")
        save_dir = output_root / f"clip_{idx:03d}"
        job = {
            "clip_id": idx,
            "start_frame": start,
            "end_frame": start + actual - 1,
            "actual_frame_num": actual,
            "frame_num": frame_num,
            "prompt": prompts[idx],
            "save_dir": str(save_dir),
            "save_file": str(save_dir / "out_video.mp4"),
            "src_video": args.src_video,
            "src_mask": args.src_mask,
            "sim_video": args.sim_video,
            "sim_mask": args.sim_mask,
            "fluid_mask": args.fluid_mask,
            "src_ref_images": getattr(args, "src_ref_images", None),
            "guidance_params": {key: getattr(args, key) for key in _GUIDANCE_PARAM_KEYS},
        }
        jobs.append(job)
    return jobs


def _load_jobs_from_manifest(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    jobs = [dict(job) for job in manifest.get("vace_clip_jobs", [])]
    if not jobs:
        raise ValueError(f"No vace_clip_jobs found in {manifest_path}")
    for idx, job in enumerate(jobs):
        job.setdefault("clip_id", idx)
        job.setdefault("save_dir", str(manifest_path.parent / "vace_clips" / f"clip_{idx:03d}"))
        job.setdefault("save_file", str(Path(job["save_dir"]) / "out_video.mp4"))
    return jobs, manifest


def _jobs_total_frame_num(jobs: Sequence[dict[str, Any]], explicit_total: int | None = None) -> int:
    if explicit_total is not None:
        return int(explicit_total)
    last_end = max(int(job.get("end_frame", 0)) for job in jobs)
    return last_end + 1


def _init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _setup_runtime(args: argparse.Namespace):
    import torch
    import torch.distributed as dist
    from annotators.utils import get_annotator
    from models.wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    from models.wan.wan_vace_latent_guidance import WanVaceLatentGuidance

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.model_name not in WAN_CONFIGS:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    if args.size not in SUPPORTED_SIZES[args.model_name]:
        raise ValueError(f"Unsupported size {args.size} for {args.model_name}")
    if args.sample_steps is None:
        args.sample_steps = 50
    if args.sample_shift is None:
        args.sample_shift = 16
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if args.object_wrap_end_step < args.guidance_start_step:
        raise ValueError("--object_wrap_end_step must be >= --guidance_start_step")
    if args.fluid_wrap_end_step is not None and args.fluid_wrap_end_step < args.guidance_start_step:
        raise ValueError("--fluid_wrap_end_step must be >= --guidance_start_step")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "FSDP is not supported in non-distributed mode."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "Context parallel is not supported in non-distributed mode."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, "ulysses_size * ring_size must equal world_size."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(), ring_degree=args.ring_size, ulysses_degree=args.ulysses_size)

    prompt_expander = None
    if args.use_prompt_extend and args.use_prompt_extend != "plain":
        prompt_expander = get_annotator(config_type="prompt", config_task=args.use_prompt_extend, return_dict=False)

    cfg = WAN_CONFIGS[args.model_name]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, "num_heads must be divisible by ulysses_size."

    logging.info("Creating WanVaceLatentGuidance pipeline once for all clips.")
    wan_vace = WanVaceLatentGuidance(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    return wan_vace, cfg, SIZE_CONFIGS, device, rank, dist, torch, prompt_expander


def _extend_prompt(prompt: str, prompt_expander: Any, rank: int, dist: Any) -> str:
    if prompt_expander is None:
        return prompt
    if rank == 0:
        extended = prompt_expander.forward(prompt)
        input_prompt = [extended]
        logging.info(f"Prompt extended from '{prompt}' to '{extended}'")
    else:
        input_prompt = [None]
    if dist.is_initialized():
        dist.broadcast_object_list(input_prompt, src=0)
    return input_prompt[0]


def _get_guidance_params(args: argparse.Namespace, job: dict[str, Any]) -> dict[str, Any]:
    params = dict(_DEFAULT_GUIDANCE_PARAMS)
    params.update({key: getattr(args, key) for key in _GUIDANCE_PARAM_KEYS if getattr(args, key) is not None})
    params.update({key: value for key, value in (job.get("guidance_params") or {}).items() if value is not None})
    return params


def _build_clip_input_paths(
    job: dict[str, Any],
    video_cache: dict[str, tuple[list[np.ndarray], float]],
    previous_anchor: np.ndarray | None,
    use_previous_anchor: bool,
    black_anchor_mask: bool,
) -> tuple[dict[str, str], float]:
    start = int(job["start_frame"])
    actual = int(job["actual_frame_num"])
    frame_num = int(job["frame_num"])
    save_dir = Path(job["save_dir"])
    input_dir = save_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    src_path = job.get("src_video")
    if not src_path:
        raise ValueError(f"Clip {job.get('clip_id')} is missing src_video.")
    if not Path(src_path).exists():
        raise FileNotFoundError(f"src_video does not exist: {src_path}")
    if src_path not in video_cache:
        video_cache[src_path] = _read_video(src_path)
    src_frames, src_fps = video_cache[src_path]
    seg_src = _slice_and_pad(src_frames, start, actual, frame_num, "src_video")

    if previous_anchor is not None and use_previous_anchor:
        seg_src[0] = _resize_like(previous_anchor, seg_src[0]).copy()

    paths: dict[str, str] = {"src_video": _write_video(input_dir / "src_video.mp4", seg_src, src_fps)}

    src_mask_path = job.get("src_mask")
    if src_mask_path:
        if not Path(src_mask_path).exists():
            raise FileNotFoundError(f"src_mask does not exist: {src_mask_path}")
        if src_mask_path not in video_cache:
            video_cache[src_mask_path] = _read_video(src_mask_path)
        src_mask_frames, _ = video_cache[src_mask_path]
        seg_mask = _slice_and_pad(src_mask_frames, start, actual, frame_num, "src_mask")
        if previous_anchor is not None and use_previous_anchor and black_anchor_mask:
            seg_mask[0] = _black_like(seg_mask[0])
    else:
        seg_mask = [_black_like(seg_src[0])] + [_white_like(frame) for frame in seg_src[1:]]
    paths["src_mask"] = _write_video(input_dir / "src_mask.mp4", seg_mask, src_fps)

    for key in ["sim_video", "sim_mask", "fluid_mask"]:
        source = job.get(key)
        if not source:
            continue
        if not Path(source).exists():
            raise FileNotFoundError(f"{key} does not exist: {source}")
        if source not in video_cache:
            video_cache[source] = _read_video(source)
        frames, _ = video_cache[source]
        seg = _slice_and_pad(frames, start, actual, frame_num, key)
        paths[key] = _write_video(input_dir / f"{key}.mp4", seg, src_fps)

    return paths, src_fps


def _prepare_conditions(
    wan_vace: Any,
    args: argparse.Namespace,
    job: dict[str, Any],
    paths: dict[str, str],
    frame_num: int,
    size_configs: dict[str, Any],
    device: int,
):
    import torch.nn.functional as F

    src_ref_images_value = job.get("src_ref_images") or getattr(args, "src_ref_images", None)
    if src_ref_images_value is None:
        src_ref_images = None
    elif isinstance(src_ref_images_value, list):
        src_ref_images = src_ref_images_value
    else:
        src_ref_images = str(src_ref_images_value).split(",")
    src_video, src_mask, src_ref_images = wan_vace.prepare_source(
        [paths["src_video"]],
        [paths.get("src_mask")],
        [src_ref_images],
        frame_num,
        size_configs[args.size],
        device,
    )

    sim_video, sim_mask = wan_vace.prepare_simulation_reference(
        paths.get("sim_video"),
        paths.get("sim_mask"),
        frame_num,
        size_configs[args.size],
        device,
    )

    if sim_video is not None:
        target_hw = src_video[0].shape[2:]
        sim_hw = sim_video.shape[2:]
        if sim_hw != target_hw:
            logging.info(f"Resizing sim_video/sim_mask from {tuple(sim_hw)} to {tuple(target_hw)}.")
            sim_video = F.interpolate(
                sim_video.permute(1, 0, 2, 3),
                size=target_hw,
                mode="bicubic",
                align_corners=False,
            ).permute(1, 0, 2, 3)
            if sim_mask is not None:
                sim_mask = F.interpolate(
                    sim_mask.permute(1, 0, 2, 3),
                    size=target_hw,
                    mode="nearest",
                ).permute(1, 0, 2, 3)

    fluid_mask = wan_vace.prepare_fluid_mask(paths.get("fluid_mask"), size_configs[args.size], device)
    if fluid_mask is not None and sim_video is not None:
        target_hw = src_video[0].shape[2:]
        fluid_hw = fluid_mask.shape[2:]
        if fluid_hw != target_hw:
            logging.info(f"Resizing fluid_mask from {tuple(fluid_hw)} to {tuple(target_hw)}.")
            fluid_mask = F.interpolate(
                fluid_mask.permute(1, 0, 2, 3),
                size=target_hw,
                mode="nearest",
            ).permute(1, 0, 2, 3)

    return src_video, src_mask, src_ref_images, sim_video, sim_mask, fluid_mask


def _save_clip_outputs(
    cfg: Any,
    job: dict[str, Any],
    video: Any,
    src_video: Any,
    src_mask: Any,
    src_ref_images: Any,
    sim_video: Any,
    sim_mask: Any,
    fluid_mask: Any,
    actual_frame_num: int,
    rank: int,
    save_condition_previews: bool = False,
) -> dict[str, str]:
    from wan.utils.utils import cache_image, cache_video

    ret: dict[str, str] = {}
    if rank != 0:
        return ret
    save_dir = Path(job["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = Path(job.get("save_file") or save_dir / "out_video.mp4")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    cache_video(
        tensor=video[:, :actual_frame_num][None],
        save_file=str(save_file),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    ret["out_video"] = str(save_file)
    logging.info(f"Saved generated clip to {save_file}")

    if save_condition_previews:
        condition_outputs = [
            ("src_video", src_video[0], (-1, 1)),
            ("src_mask", src_mask[0], (0, 1)),
            ("sim_video", sim_video, (-1, 1)),
            ("sim_mask", sim_mask, (0, 1)),
            ("fluid_mask", fluid_mask, (0, 1)),
        ]
        for name, tensor, value_range in condition_outputs:
            if tensor is None:
                continue
            path = save_dir / f"{name}.mp4"
            cache_video(
                tensor=tensor[None],
                save_file=str(path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=value_range,
            )
            ret[name] = str(path)

    if save_condition_previews and src_ref_images[0] is not None:
        for i, ref_img in enumerate(src_ref_images[0]):
            path = save_dir / f"src_ref_image_{i}.png"
            cache_image(tensor=ref_img[:, 0, ...], save_file=str(path), nrow=1, normalize=True, value_range=(-1, 1))
            ret[f"src_ref_image_{i}"] = str(path)
    return ret


def _last_frame(path: str | Path) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    frames, _ = _read_video(path)
    return frames[-1] if frames else None


def _stitch_clip_outputs(jobs: Sequence[dict[str, Any]], output_path: str | Path, total_frame_num: int | None = None, fps: float = 16) -> str:
    stitched: list[np.ndarray] = []
    for idx, job in enumerate(jobs):
        path = job.get("save_file")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Missing clip output for stitching: {path}")
        frames, fps = _read_video(path)
        expected = int(job.get("actual_frame_num") or len(frames))
        if len(frames) < expected:
            frames = frames + [frames[-1].copy() for _ in range(expected - len(frames))]
        elif len(frames) > expected:
            frames = frames[:expected]
        stitched.extend(frames if idx == 0 else frames[1:])
    if total_frame_num is not None:
        if len(stitched) < total_frame_num and stitched:
            stitched.extend([stitched[-1].copy() for _ in range(total_frame_num - len(stitched))])
        stitched = stitched[:total_frame_num]
    return _cache_video_from_bgr_frames(output_path, stitched, fps)


def run_multiclip(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest:
        jobs, manifest = _load_jobs_from_manifest(args.manifest)
        manifest_path = Path(args.manifest)
        output_root = manifest_path.parent
    else:
        jobs = _build_raw_jobs(args)
        manifest = {}
        output_root = Path(args.save_dir)
        manifest_path = None

    total_frame_num = _jobs_total_frame_num(jobs, args.total_frame_num)
    final_save_file = Path(args.final_save_file) if args.final_save_file else output_root / "out_video_dynamic_clips.mp4"

    for idx, job in enumerate(jobs):
        job["clip_id"] = int(job.get("clip_id") if job.get("clip_id") is not None else idx)
        job["start_frame"] = int(job["start_frame"])
        if "actual_frame_num" not in job or job.get("actual_frame_num") is None:
            job["actual_frame_num"] = int(job.get("end_frame", job["start_frame"]) - job["start_frame"] + 1)
        job["actual_frame_num"] = int(job["actual_frame_num"])
        if "frame_num" not in job or job.get("frame_num") is None:
            job["frame_num"] = _next_4n_plus_1(job["actual_frame_num"])
        job["frame_num"] = int(job["frame_num"])
        _check_frame_num(job["frame_num"])
        if job["frame_num"] < job["actual_frame_num"]:
            raise ValueError(f"Clip {idx} frame_num={job['frame_num']} is shorter than actual_frame_num={job['actual_frame_num']}.")
        job.setdefault("end_frame", job["start_frame"] + job["actual_frame_num"] - 1)
        job.setdefault("save_dir", str(output_root / "vace_clips" / f"clip_{idx:03d}"))
        job.setdefault("save_file", str(Path(job["save_dir"]) / "out_video.mp4"))

    if args.dry_run:
        plan = {
            "mode": "single_load_multiclip",
            "manifest": str(manifest_path) if manifest_path else None,
            "total_frame_num": total_frame_num,
            "final_save_file": str(final_save_file),
            "num_clips": len(jobs),
            "clips": [
                {
                    "clip_id": job.get("clip_id"),
                    "start_frame": job.get("start_frame"),
                    "end_frame": job.get("end_frame"),
                    "actual_frame_num": job.get("actual_frame_num"),
                    "frame_num": job.get("frame_num"),
                    "save_file": job.get("save_file"),
                    "prompt": job.get("prompt"),
                }
                for job in jobs
            ],
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan

    wan_vace, cfg, size_configs, device, rank, dist, torch, prompt_expander = _setup_runtime(args)
    video_cache: dict[str, tuple[list[np.ndarray], float]] = {}
    previous_anchor: np.ndarray | None = None
    executed: list[dict[str, Any]] = []
    last_fps = 16.0

    for idx, job in enumerate(jobs):
        frame_num = int(job["frame_num"])
        actual_frame_num = int(job["actual_frame_num"])
        prompt = str(job.get("prompt") or args.prompt or "")
        prompt = _extend_prompt(prompt, prompt_expander, rank, dist)
        guidance_params = _get_guidance_params(args, job)

        logging.info(
            f"[multiclip {idx:03d}/{len(jobs):03d}] frames "
            f"{job['start_frame']}..{job['end_frame']} actual={actual_frame_num} frame_num={frame_num}"
        )
        paths, last_fps = _build_clip_input_paths(
            job,
            video_cache=video_cache,
            previous_anchor=previous_anchor,
            use_previous_anchor=not args.no_previous_anchor,
            black_anchor_mask=not args.keep_anchor_mask,
        )
        src_video, src_mask, src_ref_images, sim_video, sim_mask, fluid_mask = _prepare_conditions(
            wan_vace,
            args,
            job,
            paths,
            frame_num,
            size_configs,
            device,
        )
        video = wan_vace.generate(
            prompt,
            src_video,
            src_mask,
            src_ref_images,
            sim_video=sim_video,
            sim_mask=sim_mask,
            fluid_mask=fluid_mask,
            guidance_start_step=guidance_params["guidance_start_step"],
            object_wrap_end_step=guidance_params["object_wrap_end_step"],
            fluid_wrap_end_step=guidance_params["fluid_wrap_end_step"],
            fluid_wrap_strength=guidance_params["fluid_wrap_strength"],
            fluid_mask_sigma=guidance_params["fluid_mask_sigma"],
            size=size_configs[args.size],
            frame_num=frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

        ret = _save_clip_outputs(
            cfg,
            job,
            video,
            src_video,
            src_mask,
            src_ref_images,
            sim_video,
            sim_mask,
            fluid_mask,
            actual_frame_num,
            rank,
            save_condition_previews=args.save_condition_previews,
        )
        if rank == 0:
            previous_anchor = _last_frame(job["save_file"])
        executed.append(
            {
                "clip_id": job.get("clip_id"),
                "start_frame": job.get("start_frame"),
                "end_frame": job.get("end_frame"),
                "actual_frame_num": actual_frame_num,
                "frame_num": frame_num,
                "input_paths": paths,
                "save_file": job.get("save_file"),
                "outputs": ret,
            }
        )

        del video, src_video, src_mask, src_ref_images, sim_video, sim_mask, fluid_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stitched = None
    if rank == 0:
        stitched = _stitch_clip_outputs(jobs, final_save_file, total_frame_num=total_frame_num, fps=last_fps)
        logging.info(f"Saved stitched multiclip video to {stitched}")

    report = {
        "mode": "single_load_multiclip",
        "manifest": str(manifest_path) if manifest_path else None,
        "total_frame_num": total_frame_num,
        "num_clips": len(jobs),
        "final_save_file": str(final_save_file),
        "stitched_output": stitched,
        "executed_jobs": executed,
    }
    if rank == 0:
        report_path = output_root / "multiclip_generation_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        logging.info(f"Saved multiclip generation report to {report_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-load multi-clip VACE generation with simulation-guided latent-space guidance.")
    parser.add_argument("--manifest", default=None, help="agent_manifest.json with vace_clip_jobs. If set, clip specs and paths are read from the manifest.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved multi-clip plan without loading the model.")

    parser.add_argument("--model_name", type=str, default="vace-1.3B")
    parser.add_argument("--size", type=str, default="480p")
    parser.add_argument("--frame_num", type=int, default=None, help="Unused compatibility alias; per-clip frame_num comes from the clip plan.")
    parser.add_argument("--ckpt_dir", type=str, default="models/Wan2.1-VACE-1.3B/")
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)

    parser.add_argument("--src_video", default=None, help="Long VACE src_video condition for raw mode.")
    parser.add_argument("--src_mask", default=None, help="Long VACE src_mask condition for raw mode.")
    parser.add_argument("--src_ref_images", default=None, help="Comma-separated reference images.")
    parser.add_argument("--sim_video", default=None, help="Long warped reference video (motion_signal) for raw mode.")
    parser.add_argument("--sim_mask", default=None, help="Long object mask for raw mode.")
    parser.add_argument("--fluid_mask", default=None, help="Long fluid mask condition for raw mode.")
    parser.add_argument("--save_dir", default=None, help="Raw mode output root.")
    parser.add_argument("--final_save_file", default=None, help="Final stitched video path.")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Final stitched frame count. Required in raw mode; inferred from manifest otherwise.")
    parser.add_argument("--clip_frame_nums", default=None, help="Comma-separated per-clip VACE frame_num list, e.g. 41,21,81.")
    parser.add_argument("--clip_start_frames", default=None, help="Comma-separated inclusive clip start frames, e.g. 0,40,60.")
    parser.add_argument("--clip_prompts_json", default=None, help="JSON list/dict or path for per-clip prompts in raw mode.")

    parser.add_argument("--guidance_start_step", type=int, default=8)
    parser.add_argument("--object_wrap_end_step", type=int, default=11)
    parser.add_argument("--fluid_wrap_end_step", type=int, default=10)
    parser.add_argument("--fluid_wrap_strength", type=float, default=0.8)
    parser.add_argument("--fluid_mask_sigma", type=float, default=16.0)

    parser.add_argument("--prompt", type=str, default=None, help="Global fallback prompt.")
    parser.add_argument("--use_prompt_extend", default="plain", choices=["plain", "wan_zh", "wan_en", "wan_zh_ds", "wan_en_ds"])
    parser.add_argument("--base_seed", type=int, default=2025)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)

    parser.add_argument("--no_previous_anchor", action="store_true", help="Do not replace each later clip's first source frame with the previous generated last frame.")
    parser.add_argument("--keep_anchor_mask", action="store_true", help="Do not black out the first src_mask frame when previous-anchor is used.")
    parser.add_argument("--save_condition_previews", action="store_true", help="Also save processed src/sim/mask preview videos in each clip directory. By default only inputs/ slices and out_video.mp4 are saved.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_multiclip(args)


if __name__ == "__main__":
    main()
