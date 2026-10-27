"""Execute dynamic VACE clip jobs from an agent manifest.

Each planned clip can have a different VACE-compatible frame_num, e.g. 41, 21,
81. The default executor calls vace_wan_inference_latent_guidance.py so
Wan/VACE is loaded once for all clips.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required for dynamic clip slicing/stitching.") from exc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fluidgen import paths  # noqa: E402

_VIDEO_KEYS = ["src_video", "src_mask", "sim_video", "sim_mask", "fluid_mask"]
_GUIDANCE_PARAM_KEYS = [
    "guidance_start_step",
    "object_wrap_end_step",
    "fluid_wrap_end_step",
    "fluid_wrap_strength",
    "fluid_mask_sigma",
]


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


def _write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"Cannot write empty video: {out}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out}")
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()
    return str(out)


def _cache_video_like_vace_direct(path: str | Path, frames: list[np.ndarray], fps: float) -> str:
    import imageio

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
    try:
        for frame in rgb_frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return str(out)


def _write_video_like_vace(path: str | Path, frames: list[np.ndarray], fps: float, wan_conda_env: str | None = "wan") -> str:
    """Write stitched output through the same Wan cache_video path as VACE clips when possible."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"Cannot write empty stitched video: {out}")
    try:
        return _cache_video_like_vace_direct(out, frames, fps)
    except Exception as direct_exc:
        if wan_conda_env:
            helper = Path(__file__).with_name("vace_video_writer.py")
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f"{out.stem}_",
                    suffix=".frames.npz",
                    dir=str(out.parent),
                    delete=False,
                ) as tmp:
                    tmp_path = tmp.name
                np.savez(tmp_path, frames=np.asarray(frames, dtype=np.uint8), fps=np.asarray([fps], dtype=np.float32))
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    str(wan_conda_env),
                    "--no-capture-output",
                    "python",
                    str(helper),
                    "--frames_npz",
                    tmp_path,
                    "--output",
                    str(out),
                    "--fps",
                    str(fps),
                ]
                result = subprocess.run(cmd, check=False)
                if result.returncode == 0 and out.exists():
                    return str(out)
                print(
                    f"[stitch warning] VACE cache_video external writer failed with returncode={result.returncode}; "
                    f"falling back to OpenCV mp4v. Direct error was: {direct_exc}",
                    file=sys.stderr,
                )
            except Exception as external_exc:
                print(
                    f"[stitch warning] VACE cache_video unavailable ({direct_exc}); external writer failed ({external_exc}); "
                    "falling back to OpenCV mp4v.",
                    file=sys.stderr,
                )
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)
        else:
            print(
                f"[stitch warning] VACE cache_video unavailable ({direct_exc}); falling back to OpenCV mp4v.",
                file=sys.stderr,
            )
        return _write_video(out, frames, fps)


def _black_like(frame: np.ndarray) -> np.ndarray:
    return np.zeros(frame.shape[:2], dtype=np.uint8)


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


def _last_frame(path: str | Path) -> Optional[np.ndarray]:
    if not Path(path).exists():
        return None
    frames, _ = _read_video(path)
    return frames[-1]


def _slice_frames(frames: list[np.ndarray], start: int, end: int) -> list[np.ndarray]:
    # end is inclusive in ClipPlan.
    if start < 0 or end < start:
        raise ValueError(f"Invalid clip slice {start}:{end}")
    sliced = [frame.copy() for frame in frames[start : end + 1]]
    if not sliced:
        raise RuntimeError(f"Clip slice {start}:{end} produced no frames.")
    return sliced


def _slice_job_inputs(
    job: Dict[str, Any],
    video_cache: Dict[str, tuple[list[np.ndarray], float]],
    previous_output: Optional[str],
    use_previous_anchor: bool,
    black_anchor_mask: bool,
) -> Dict[str, str]:
    start = int(job["start_frame"])
    end = int(job["end_frame"])
    save_dir = Path(job["save_dir"])
    input_dir = save_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    sliced_paths: Dict[str, str] = {}
    sliced_frames: Dict[str, list[np.ndarray]] = {}
    sliced_fps: Dict[str, float] = {}
    previous_anchor = _last_frame(previous_output) if previous_output and use_previous_anchor else None
    for key in _VIDEO_KEYS:
        source = job.get(key)
        if not source:
            continue
        if source not in video_cache:
            video_cache[source] = _read_video(source)
        frames, fps = video_cache[source]
        sliced = _slice_frames(frames, start, min(end, len(frames) - 1))

        if key == "src_video" and previous_anchor is not None:
            if previous_anchor.shape[:2] != sliced[0].shape[:2]:
                previous_anchor = cv2.resize(previous_anchor, (sliced[0].shape[1], sliced[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            sliced[0] = previous_anchor.copy()
        if key == "src_mask" and previous_anchor is not None and black_anchor_mask:
            sliced[0] = _black_like(sliced[0])

        sliced_frames[key] = sliced
        sliced_fps[key] = fps

    sim_video_frames = sliced_frames.get("sim_video")
    sim_mask_frames = sliced_frames.get("sim_mask")
    if sim_video_frames and sim_mask_frames:
        target_hw = sim_video_frames[0].shape[:2]
        sliced_frames["sim_mask"] = _resize_frames_like(sim_mask_frames, target_hw, cv2.INTER_NEAREST)

    for key, sliced in sliced_frames.items():
        sliced_paths[key] = _write_video(input_dir / f"{key}.mp4", sliced, sliced_fps[key])
    return sliced_paths


def _build_command(job: Dict[str, Any], sliced_paths: Dict[str, str]) -> list[str]:
    inference_script = job["inference_script"]
    cmd: list[str] = []
    wan_conda_env = job.get("wan_conda_env", "wan")
    if wan_conda_env:
        cmd.extend(["conda", "run", "-n", str(wan_conda_env), "--no-capture-output"])
    else:
        cmd.append(sys.executable)
    cmd.extend([
        "python" if wan_conda_env else inference_script,
    ])
    if wan_conda_env:
        cmd.append(inference_script)
    cmd.extend([
        "--frame_num",
        str(job["frame_num"]),
        "--total_frame_num",
        str(job["actual_frame_num"]),
        "--save_dir",
        str(job["save_dir"]),
        "--final_save_file",
        str(job["save_file"]),
        "--prompt",
        str(job["prompt"]),
    ])
    if "src_video" not in sliced_paths:
        raise ValueError(f"Clip {job.get('clip_id', job.get('save_dir'))} has no src_video to pass to VACE.")
    for key, flag in [
        ("src_video", "--src_video"),
        ("src_mask", "--src_mask"),
        ("sim_video", "--sim_video"),
        ("sim_mask", "--sim_mask"),
        ("fluid_mask", "--fluid_mask"),
    ]:
        if key in sliced_paths:
            cmd.extend([flag, sliced_paths[key]])
    guidance_params = job.get("guidance_params", {})
    for key in _GUIDANCE_PARAM_KEYS:
        value = guidance_params.get(key, job.get(key))
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def _build_single_load_command(
    jobs: list[Dict[str, Any]],
    manifest_path: str | Path,
    stitched_output: str | Path,
    dry_run: bool,
    use_previous_anchor: bool,
    black_anchor_mask: bool,
) -> list[str]:
    if not jobs:
        raise ValueError("Cannot build a multiclip command without jobs.")
    first_job = jobs[0]
    inference_script = first_job.get("inference_script")
    if inference_script and Path(str(inference_script)).name == "vace_wan_inference_latent_guidance.py":
        multiclip_script = str(inference_script)
    elif inference_script:
        multiclip_script = str(Path(inference_script).with_name("vace_wan_inference_latent_guidance.py"))
    else:
        multiclip_script = str(Path(paths.VACE_DIR) / "vace" / "vace_wan_inference_latent_guidance.py")

    cmd: list[str] = []
    wan_conda_env = first_job.get("wan_conda_env", "wan")
    if wan_conda_env:
        cmd.extend(["conda", "run", "-n", str(wan_conda_env), "--no-capture-output"])
    else:
        cmd.append(sys.executable)
    cmd.extend([
        "python" if wan_conda_env else multiclip_script,
    ])
    if wan_conda_env:
        cmd.append(multiclip_script)
    cmd.extend([
        "--manifest",
        str(manifest_path),
        "--final_save_file",
        str(stitched_output),
        "--ckpt_dir",
        str(first_job.get("ckpt_dir") or paths.VACE_CKPT),
    ])
    if dry_run:
        cmd.append("--dry_run")
    if not use_previous_anchor:
        cmd.append("--no_previous_anchor")
    if not black_anchor_mask:
        cmd.append("--keep_anchor_mask")
    return cmd


def _stitch_outputs(jobs: list[Dict[str, Any]], output_path: str | Path) -> Optional[str]:
    if len(jobs) == 1:
        path = jobs[0].get("save_file")
        if not path or not Path(path).exists():
            return None
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        return str(out)

    stitched: list[np.ndarray] = []
    fps = 16
    wan_conda_env = jobs[0].get("wan_conda_env", "wan") if jobs else "wan"
    for idx, job in enumerate(jobs):
        path = job.get("save_file")
        if not path or not Path(path).exists():
            return None
        frames, fps = _read_video(path)
        expected = int(job.get("actual_frame_num") or 0)
        if expected > 0 and len(frames) < expected:
            missing = expected - len(frames)
            print(
                f"[stitch warning] clip {idx:03d} has {len(frames)} frame(s), expected {expected}; "
                f"padding {missing} trailing frame(s) before stitching.",
                file=sys.stderr,
            )
            frames = frames + [frames[-1].copy() for _ in range(missing)]
        elif expected > 0 and len(frames) > expected:
            frames = frames[:expected]
        stitched.extend(frames if idx == 0 else frames[1:])
    return _write_video_like_vace(output_path, stitched, fps, wan_conda_env=wan_conda_env)


def _vace_preprocess_outputs_exist(stage: Dict[str, Any]) -> bool:
    outputs = stage.get("outputs", {})
    required = [outputs.get("src_video"), outputs.get("src_mask")]
    return all(path and Path(path).exists() for path in required)


def _run_manifest_vace_preprocess(manifest: Dict[str, Any]) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for stage in manifest.get("stage_commands", []):
        if stage.get("name") != "vace_preprocess":
            continue
        record = {
            "name": stage.get("name"),
            "command": stage.get("command"),
            "cwd": stage.get("cwd"),
            "skipped": False,
            "returncode": None,
        }
        if _vace_preprocess_outputs_exist(stage):
            print("[vace_preprocess] outputs already exist; skipping")
            record["skipped"] = True
            records.append(record)
            continue
        command = stage.get("command")
        if not command:
            raise ValueError("vace_preprocess stage is missing its command.")
        print("[vace_preprocess]", " ".join(map(str, command)))
        result = subprocess.run(
            command,
            cwd=stage.get("cwd") or None,
            env={**os.environ, **stage.get("env", {})},
            check=False,
        )
        record["returncode"] = result.returncode
        records.append(record)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
    return records


def execute_manifest(
    manifest_path: str | Path,
    dry_run: bool = False,
    stitch_only: bool = False,
    run_preprocess: bool = False,
    single_load: bool = True,
    use_previous_anchor: bool = True,
    black_anchor_mask: bool = True,
    stitched_output: str | Path | None = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest.get("vace_clip_jobs", [])
    if not jobs:
        raise ValueError(f"No vace_clip_jobs found in {manifest_path}")
    preprocess_records = _run_manifest_vace_preprocess(manifest) if run_preprocess else []

    executed: list[Dict[str, Any]] = []
    if stitch_only:
        for index, job in enumerate(jobs):
            executed.append(
                {
                    "clip_id": index,
                    "frame_num": job.get("frame_num"),
                    "actual_frame_num": job.get("actual_frame_num"),
                    "start_frame": job.get("start_frame"),
                    "end_frame": job.get("end_frame"),
                    "save_file": job.get("save_file"),
                    "returncode": "stitch_only",
                }
            )
    else:
        if stitched_output is None:
            stitched_output = manifest_path.parent / "out_video_dynamic_clips.mp4"
        cmd = _build_single_load_command(
            jobs,
            manifest_path=manifest_path,
            stitched_output=stitched_output,
            dry_run=dry_run,
            use_previous_anchor=use_previous_anchor,
            black_anchor_mask=black_anchor_mask,
        )
        record = {
            "mode": "single_load_multiclip",
            "command": cmd,
            "returncode": None,
            "num_clips": len(jobs),
        }
        if dry_run:
            print("[dry-run multiclip]", " ".join(map(str, cmd)))
        else:
            print("[multiclip]", " ".join(map(str, cmd)))
            result = subprocess.run(cmd, check=False)
            record["returncode"] = result.returncode
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
        executed.append(record)

    if stitched_output is None:
        stitched_output = manifest_path.parent / "out_video_dynamic_clips.mp4"
    if dry_run:
        stitched = None
    elif not stitch_only:
        stitched = str(stitched_output) if Path(stitched_output).exists() else None
    else:
        stitched = _stitch_outputs(jobs, stitched_output)
    report = {
        "manifest": str(manifest_path),
        "dry_run": dry_run,
        "stitch_only": stitch_only,
        "single_load": True,
        "vace_preprocess": preprocess_records,
        "num_clips": len(jobs),
        "stitched_output": stitched,
        "executed_jobs": executed,
    }
    report_name = "dynamic_clip_execution_dry_run_report.json" if dry_run else "dynamic_clip_execution_report.json"
    report_path = manifest_path.parent / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved dynamic clip execution report to {report_path}")
    if stitched:
        print(f"Saved stitched dynamic clip video to {stitched}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-clip VACE generation with latent-space guidance from agent_manifest.json.")
    parser.add_argument("--manifest", required=True, help="Path to agent_manifest.json.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved multi-clip command without running VACE.")
    parser.add_argument("--stitch_only", action="store_true", help="Do not run VACE; only restitch existing per-clip out_video.mp4 files.")
    parser.add_argument("--run_preprocess", action="store_true", help="Run the manifest's vace_preprocess stage before slicing/generation.")
    parser.add_argument("--no_previous_anchor", action="store_true", help="Do not replace each later clip's first src frame with previous generated last frame.")
    parser.add_argument("--keep_anchor_mask", action="store_true", help="Do not black out src_mask first frame when previous-anchor is used.")
    parser.add_argument("--stitched_output", default=None, help="Optional path for the final stitched video.")
    return parser


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    return execute_manifest(
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        stitch_only=args.stitch_only,
        run_preprocess=args.run_preprocess,
        single_load=True,
        use_previous_anchor=not args.no_previous_anchor,
        black_anchor_mask=not args.keep_anchor_mask,
        stitched_output=args.stitched_output,
    )


if __name__ == "__main__":
    main()
