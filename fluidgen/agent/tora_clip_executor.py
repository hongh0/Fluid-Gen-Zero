"""Execute Tora multi-clip jobs with latent-space guidance from an agent manifest."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fluidgen import paths  # noqa: E402


def _build_command(
    manifest: Dict[str, Any],
    manifest_path: str | Path,
    dry_run: bool,
    use_previous_anchor: bool,
    stitched_output: str | Path | None,
) -> tuple[list[str], Path]:
    jobs = manifest.get("tora_clip_jobs", [])
    if not jobs:
        raise ValueError(f"No tora_clip_jobs found in {manifest_path}")
    first_job = jobs[0]
    tora_dir = Path(first_job.get("tora_dir") or paths.TORA_DIR)
    cwd = tora_dir / "sat"
    inference_script = str(
        first_job.get("inference_script")
        or cwd / "sample_video_latent_guidance.py"
    )
    base_configs = list(first_job.get("base_configs") or [
        "configs/tora/model/cogvideox_5b_tora_i2v.yaml",
        "configs/tora/inference_sparse_json.yaml",
    ])
    ckpt = str(first_job.get("ckpt") or paths.TORA_CKPT)
    output_dir = str(Path(first_job.get("save_dir", Path(manifest_path).parent / "tora_clips")).parent)

    cmd: list[str] = []
    conda_env = first_job.get("tora_conda_env", "tora")
    if conda_env:
        cmd.extend(["conda", "run", "-n", str(conda_env), "--no-capture-output"])
    else:
        cmd.append(sys.executable)
    cmd.extend(["python" if conda_env else inference_script])
    if conda_env:
        cmd.append(inference_script)
    cmd.extend([
        "--manifest",
        str(manifest_path),
        "--base",
        *base_configs,
        "--load",
        ckpt,
        "--output-dir",
        output_dir,
        "--sampling-fps",
        str(first_job.get("sampling_fps", 8)),
        "--seed",
        str(first_job.get("seed", 1234)),
    ])
    if stitched_output is not None:
        cmd.extend(["--final_save_file", str(stitched_output)])
    if dry_run:
        cmd.append("--dry_run")
    if not use_previous_anchor:
        cmd.append("--no_previous_anchor")
    if first_job.get("use_fluid_mask_center_trajectory"):
        cmd.append("--use_fluid_mask_center_trajectory")
    return cmd, cwd


def execute_manifest(
    manifest_path: str | Path,
    dry_run: bool = False,
    use_previous_anchor: bool = True,
    stitched_output: str | Path | None = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest.get("tora_clip_jobs", [])
    if not jobs:
        raise ValueError(f"No tora_clip_jobs found in {manifest_path}")
    if stitched_output is None:
        stitched_output = manifest_path.parent / "out_video_dynamic_tora_clips.mp4"

    cmd, cwd = _build_command(
        manifest,
        manifest_path=manifest_path,
        dry_run=dry_run,
        use_previous_anchor=use_previous_anchor,
        stitched_output=stitched_output,
    )
    record = {
        "mode": "tora_single_load_multiclip",
        "command": cmd,
        "cwd": str(cwd),
        "returncode": None,
        "num_clips": len(jobs),
    }
    clip_report_path = manifest_path.parent / "tora_multiclip_generation_report.json"

    if dry_run:
        print("[dry-run tora multiclip]", " ".join(map(str, cmd)))
    else:
        print("[tora multiclip]", " ".join(map(str, cmd)))
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            start_new_session=True,
        )
        assumed_success = False
        while True:
            returncode = proc.poll()
            if returncode is not None:
                record["returncode"] = returncode
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)
                break

            stitched_ready = Path(stitched_output).exists()
            report_ready = clip_report_path.exists()
            if stitched_ready and report_ready:
                latest_mtime = max(
                    Path(stitched_output).stat().st_mtime,
                    clip_report_path.stat().st_mtime,
                )
                if time.time() - latest_mtime > 20:
                    print(
                        "[tora multiclip] outputs are present but the sampler process is still alive; "
                        "terminating the lingering process group and continuing."
                    )
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait(timeout=15)
                    assumed_success = True
                    record["returncode"] = 0
                    record["assumed_success_after_output_stall"] = True
                    break

            time.sleep(5)

        if not assumed_success:
            record.setdefault("assumed_success_after_output_stall", False)

    report = {
        "manifest": str(manifest_path),
        "dry_run": dry_run,
        "num_clips": len(jobs),
        "stitched_output": None if dry_run else str(stitched_output),
        "executed_jobs": [record],
    }
    report_name = "dynamic_tora_clip_execution_dry_run_report.json" if dry_run else "dynamic_tora_clip_execution_report.json"
    report_path = manifest_path.parent / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved Tora dynamic clip execution report to {report_path}")
    if not dry_run:
        print(f"Saved stitched dynamic Tora clip video to {stitched_output}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-clip Tora generation with latent-space guidance from agent_manifest.json.")
    parser.add_argument("--manifest", required=True, help="Path to agent_manifest.json.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved Tora multi-clip command without running the model.")
    parser.add_argument("--no_previous_anchor", action="store_true", help="Do not use each generated clip's final frame as the next clip anchor.")
    parser.add_argument("--stitched_output", default=None, help="Optional path for the final stitched video.")
    return parser


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    return execute_manifest(
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        use_previous_anchor=not args.no_previous_anchor,
        stitched_output=args.stitched_output,
    )


if __name__ == "__main__":
    main()
