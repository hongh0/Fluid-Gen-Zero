# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _write_json(path: str | Path, data: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, output_path)


def _slugify(text: str, max_len: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (value or "item")[:max_len]


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _resolve_data_path(path: str | None, base_dir: str = "") -> str | None:
    if not path:
        return None
    if os.path.isabs(path) or not base_dir:
        return path
    return os.path.join(base_dir, path)


def _resolve_manifest_path(item: dict[str, Any]) -> str:
    manifest_path = item.get("agent_manifest_path")
    if manifest_path:
        return str(manifest_path)
    for key in ("generation_video_path", "vace_video_path", "tora_video_path", "wanmove_video_path"):
        candidate = item.get(key)
        if not candidate:
            continue
        inferred = Path(candidate).resolve().parent / "agent_manifest.json"
        if inferred.exists():
            return str(inferred)
    raise KeyError("JSON item must contain agent_manifest_path, or generation_video_path must sit next to agent_manifest.json.")


def _find_manifest_value(obj: Any, target_key: str, found: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key and value:
                found.append(str(value))
            _find_manifest_value(value, target_key, found)
    elif isinstance(obj, list):
        for value in obj:
            _find_manifest_value(value, target_key, found)


def _manifest_path_value(manifest: dict[str, Any], manifest_path: str | Path, key: str) -> str:
    found: list[str] = []
    _find_manifest_value(manifest, key, found)
    unique = sorted(set(found))
    if not unique:
        raise KeyError(f"Could not find '{key}' in manifest: {manifest_path}")
    if len(unique) > 1:
        raise ValueError(f"Found multiple values for '{key}' in manifest {manifest_path}: {unique}")
    value = unique[0]
    if os.path.isabs(value):
        return value
    return str((Path(manifest_path).resolve().parent / value).resolve())


def _check_frame_num(frame_num: int) -> None:
    if frame_num < 5 or (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must satisfy 4n+1 and be >= 5, got {frame_num}.")


def _plan_clips(total_frames: int, clip_frame_num: int) -> list[dict[str, int]]:
    if total_frames < 2:
        raise ValueError(f"At least 2 frames are required for multi-clip generation, got {total_frames}.")
    _check_frame_num(clip_frame_num)
    clips: list[dict[str, int]] = []
    start = 0
    clip_id = 0
    while start < total_frames:
        actual = min(clip_frame_num, total_frames - start)
        if actual < 2:
            break
        clips.append(
            {
                "clip_id": clip_id,
                "start_frame": start,
                "actual_frame_num": actual,
                "frame_num": clip_frame_num,
                "end_frame": start + actual - 1,
            }
        )
        if start + actual >= total_frames:
            break
        start += clip_frame_num - 1
        clip_id += 1
    if not clips:
        raise ValueError(f"Could not build valid clips for total_frames={total_frames}, clip_frame_num={clip_frame_num}.")
    return clips


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


def _extract_first_frame_from_video(video_path: str) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _last_frame(path: str | Path) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    frames, _ = _read_video(path)
    return frames[-1] if frames else None


def _fill_mask_centers(points: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_t = np.arange(len(points), dtype=np.float32)
    valid_t = src_t[visibility]
    if len(valid_t) == 0:
        return np.zeros((len(points), 2), dtype=np.float32), np.zeros((len(points),), dtype=np.float32)
    valid_points = points[visibility]
    x = np.interp(src_t, valid_t, valid_points[:, 0])
    y = np.interp(src_t, valid_t, valid_points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32), visibility.astype(np.float32)


def _mask_frames_to_centers(mask_frames: list[np.ndarray], threshold: int = 1) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    visibility: list[bool] = []
    for frame in mask_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        ys, xs = np.nonzero(gray > threshold)
        if len(xs) == 0:
            points.append([np.nan, np.nan])
            visibility.append(False)
        else:
            points.append([float(xs.mean()), float(ys.mean())])
            visibility.append(True)
    return _fill_mask_centers(np.asarray(points, dtype=np.float32), np.asarray(visibility, dtype=bool))


def _video_frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """Convert OpenCV BGR frames to Wan's [C,T,H,W] uint8-like tensor."""
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return torch.from_numpy(np.stack(rgb)).permute(3, 0, 1, 2).contiguous()


def _mask_frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    masks = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        masks.append((gray > 0).astype(np.float32))
    return torch.from_numpy(np.stack(masks))


def _save_generated_clip(video, actual_frame_num: int, save_file: str | Path, fps: int) -> str:
    save_path = Path(save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cache_video(
        tensor=video[:, :actual_frame_num][None],
        save_file=str(save_path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return str(save_path)


def _stitch_clip_outputs(jobs: list[dict[str, Any]], output_path: str | Path, total_frame_num: int, fps: int) -> str:
    stitched: list[np.ndarray] = []
    for idx, job in enumerate(jobs):
        frames, _ = _read_video(job["save_file"])
        expected = int(job["actual_frame_num"])
        if len(frames) < expected:
            frames.extend([frames[-1].copy() for _ in range(expected - len(frames))])
        elif len(frames) > expected:
            frames = frames[:expected]
        stitched.extend(frames if idx == 0 else frames[1:])
    if len(stitched) < total_frame_num and stitched:
        stitched.extend([stitched[-1].copy() for _ in range(total_frame_num - len(stitched))])
    stitched = stitched[:total_frame_num]
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in stitched]
    tensor = torch.from_numpy(np.stack(rgb_frames)).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_video(
        tensor=tensor[None],
        save_file=str(output_path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return str(output_path)


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50
    if args.sample_shift is None:
        args.sample_shift = 3.0 if "i2v" in args.task and args.size in ["832*480", "480*832"] else 5.0
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupported size {args.size} for {args.task}"
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch Wan-Move multi-clip generation from manifest paths in JSON.")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--video_base_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json_path", type=str, required=True)
    parser.add_argument("--clip_frame_num", type=int, default=81)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_previous_anchor", action="store_true")
    parser.add_argument(
        "--latent_guidance", action="store_true",
        help="Enable warped-video latent reference control with object/fluid masks.")
    parser.add_argument("--guidance_start_step", type=int, default=8)
    parser.add_argument("--object_wrap_end_step", type=int, default=11)
    parser.add_argument("--fluid_wrap_end_step", type=int, default=10)
    parser.add_argument("--fluid_wrap_strength", type=float, default=0.8)
    parser.add_argument("--fluid_mask_sigma", type=float, default=16.0)

    parser.add_argument("--task", type=str, default="wan-move-i2v", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="480*832", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, default="./Wan-Move-14B-480P")
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    return parser


def run_batch(args) -> list[dict[str, Any]]:
    args = _validate_args(args)
    _init_logging()
    if args.offload_model is None:
        args.offload_model = True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    cfg = WAN_CONFIGS[args.task]
    logging.info("Creating WanMove pipeline once for all items.")
    wan_move = wan.WanMove(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )

    items = _load_json(args.input_json)
    output_items = [dict(item) for item in items]

    for item_idx, item in enumerate(items):
        video_stem = Path(str(item.get("video_path") or f"item_{item_idx:06d}")).stem
        item_name = f"{item_idx:06d}_{_slugify(video_stem)}"
        item_dir = Path(args.output_dir) / "items" / item_name
        final_save_file = item_dir / "out_video_dynamic_wanmove_clips.mp4"

        try:
            if args.resume and final_save_file.exists():
                output_items[item_idx]["generation_video_path"] = str(final_save_file.resolve())
                _write_json(args.output_json_path, output_items)
                continue

            prompt = item.get("caption") or item.get("prompt")
            if not prompt:
                raise KeyError("JSON item must contain 'caption' or 'prompt'.")

            video_path = _resolve_data_path(item.get("video_path"), args.video_base_dir)
            if not video_path or not Path(video_path).exists():
                raise FileNotFoundError(f"Input video_path does not exist: {video_path}")
            base_anchor = _extract_first_frame_from_video(video_path)
            anchor_size = base_anchor.size

            manifest_path = _resolve_manifest_path(item)
            manifest = _load_json(manifest_path)[0]
            warped_video_path = _manifest_path_value(manifest, manifest_path, "warped_reference_video_path") if args.latent_guidance else None
            warped_mask_path = _manifest_path_value(manifest, manifest_path, "warped_reference_mask_path")
            fg_mask_path = _manifest_path_value(manifest, manifest_path, "fg_mask_video_path")
            fluid_mask_path = _manifest_path_value(manifest, manifest_path, "fluid_mask_video_path") if args.latent_guidance else None

            warped_video_frames = _read_video(warped_video_path)[0] if warped_video_path else None
            warped_frames, _ = _read_video(warped_mask_path)
            fg_frames, _ = _read_video(fg_mask_path)
            fluid_frames = _read_video(fluid_mask_path)[0] if fluid_mask_path else None
            frame_lengths = [len(warped_frames), len(fg_frames)]
            if warped_video_frames is not None:
                frame_lengths.append(len(warped_video_frames))
            if fluid_frames is not None:
                frame_lengths.append(len(fluid_frames))
            total_frames = min(frame_lengths)
            # Use the agent's clip plan when the manifest carries one. The
            # planner sizes each clip from the simulation's complexity, so the
            # boundaries are a property of the scene rather than of a backend,
            # and WanMove's 81-frame limit is the same one the plan was built
            # under. Without a plan, clips are uniform.
            plan = manifest.get("clip_plan")
            if plan:
                clips = [dict(c) for c in plan]
                print(f"  using the agent clip plan: {len(clips)} clips "
                      f"{[c['actual_frame_num'] for c in clips]}", flush=True)
            else:
                clips = _plan_clips(total_frames, args.clip_frame_num)
            previous_anchor = None
            executed_jobs: list[dict[str, Any]] = []

            for clip in clips:
                clip_id = int(clip["clip_id"])
                clip_dir = item_dir / "clips" / f"clip_{clip_id:03d}"
                clip_dir.mkdir(parents=True, exist_ok=True)

                warped_clip = _slice_and_pad(
                    warped_frames,
                    int(clip["start_frame"]),
                    int(clip["actual_frame_num"]),
                    int(clip["frame_num"]),
                    "warped_reference_mask_path",
                )
                fg_clip = _slice_and_pad(
                    fg_frames,
                    int(clip["start_frame"]),
                    int(clip["actual_frame_num"]),
                    int(clip["frame_num"]),
                    "fg_mask_video_path",
                )
                warped_video_clip = None
                fluid_clip = None
                if warped_video_frames is not None:
                    warped_video_clip = _slice_and_pad(
                        warped_video_frames,
                        int(clip["start_frame"]),
                        int(clip["actual_frame_num"]),
                        int(clip["frame_num"]),
                        "warped_reference_video_path",
                    )
                if fluid_frames is not None:
                    fluid_clip = _slice_and_pad(
                        fluid_frames,
                        int(clip["start_frame"]),
                        int(clip["actual_frame_num"]),
                        int(clip["frame_num"]),
                        "fluid_mask_video_path",
                    )

                if previous_anchor is None or args.no_previous_anchor:
                    img = base_anchor.copy()
                else:
                    anchor_rgb = cv2.cvtColor(previous_anchor, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(anchor_rgb).resize(anchor_size, Image.BILINEAR)

                mask_specs = [warped_clip, fg_clip]
                track = np.zeros((int(clip["frame_num"]), len(mask_specs), 2), dtype=np.float32)
                track_visibility = np.zeros((int(clip["frame_num"]), len(mask_specs)), dtype=np.float32)
                for mask_idx, mask_frames in enumerate(mask_specs):
                    centers, visibility = _mask_frames_to_centers(mask_frames)
                    track[:, mask_idx, :] = centers
                    track_visibility[:, mask_idx] = visibility

                clip_seed = int(args.base_seed) + item_idx * 1000 + clip_id
                # Clip-level prompt p_i from the plan; the input prompt otherwise.
                video = wan_move.generate(
                    str(clip.get("prompt") or prompt),
                    img,
                    track,
                    track_visibility,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=int(clip["frame_num"]),
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=clip_seed,
                    offload_model=args.offload_model,
                    eval_bench=False,
                    sim_video=_video_frames_to_tensor(warped_video_clip) if warped_video_clip is not None else None,
                    sim_mask=_mask_frames_to_tensor(warped_clip) if warped_video_clip is not None else None,
                    fluid_mask=_mask_frames_to_tensor(fluid_clip) if fluid_clip is not None else None,
                    guidance_start_step=args.guidance_start_step,
                    object_wrap_end_step=args.object_wrap_end_step,
                    fluid_wrap_end_step=args.fluid_wrap_end_step,
                    fluid_wrap_strength=args.fluid_wrap_strength,
                    fluid_mask_sigma=args.fluid_mask_sigma,
                )
                clip_save_file = clip_dir / "out_video.mp4"
                _save_generated_clip(video, int(clip["actual_frame_num"]), clip_save_file, cfg.sample_fps)
                previous_anchor = _last_frame(clip_save_file)
                executed_jobs.append(
                    {
                        **clip,
                        "save_file": str(clip_save_file),
                    }
                )
                del video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            stitched_output = _stitch_clip_outputs(executed_jobs, final_save_file, total_frames, cfg.sample_fps)
            output_items[item_idx]["generation_video_path"] = str(Path(stitched_output).resolve())
            output_items[item_idx]["agent_manifest_path"] = str(Path(manifest_path).resolve())
            output_items[item_idx].pop("_multiclip_error", None)
        except Exception as exc:
            # Print the traceback as well as storing the message.
            import traceback
            print(f"!! item {item_idx} failed: {exc}", flush=True)
            traceback.print_exc()
            output_items[item_idx]["_multiclip_error"] = str(exc)
            output_items[item_idx]["_multiclip_traceback"] = traceback.format_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _write_json(args.output_json_path, output_items)

    return output_items


if __name__ == "__main__":
    args = build_parser().parse_args()
    _check_frame_num(args.clip_frame_num)
    run_batch(args)
