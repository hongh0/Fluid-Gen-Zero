"""Command wrappers for the condition-building stages run before generation:
the warped reference video and the VACE native condition."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from fluidgen import paths

from .schema import SimulationBundle, StageCommand

WARPED_REFERENCE_DIR = paths.REPO_ROOT / "fluidgen" / "warped_reference"


def conda_python_command(conda_env: str | None, script: str | Path, *args: str) -> list[str]:
    script_str = str(script)
    if conda_env:
        return ["conda", "run", "-n", conda_env, "--no-capture-output", "python", script_str, *args]
    return [sys.executable, script_str, *args]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in command)


def build_warped_reference_stage(
    image_path: str,
    bundle: SimulationBundle,
    output_dir: str | None = None,
    conda_env: str | None = None,
    size: str = "480x832",
    fps: float = 24.0,
    sam2_checkpoint: str = paths.SAM2_CKPT,
    sam2_config: str = "/configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_device: str = "cuda",
    inpaint_prompt: str = "background, nothing",
    inpaint_steps: int = 24,
    inpaint_seed: int = 42,
    inpaint_device: str = "cuda",
) -> StageCommand:
    if not bundle.fg_mask_video_path:
        raise ValueError("Warped reference generation requires simulation fg_mask_video_path.")
    out_dir = Path(output_dir) if output_dir else Path(bundle.root_dir) / "warped_reference"
    # The first-frame object mask reconstruction already produced, so the warp
    # does not have to guess it from a single SAM2 point prompt. Empty when the
    # case layout is unexpected, and get_warped_video falls back to SAM2 then.
    recon_mask = Path(bundle.root_dir).parent / "s1_6_recon" / "s1_segment" / "object_mask.png"
    cmd = conda_python_command(
        conda_env,
        WARPED_REFERENCE_DIR / "get_warped_video.py",
        "--image",
        str(Path(image_path).resolve()),
        "--object-mask",
        str(recon_mask) if recon_mask.exists() else "",
        "--mask-video",
        str(Path(bundle.fg_mask_video_path).resolve()),
        "--output-dir",
        str(out_dir),
        "--fps",
        str(fps),
        "--size",
        size,
        "--sam2-checkpoint",
        sam2_checkpoint,
        "--sam2-config",
        sam2_config,
        "--sam2-device",
        sam2_device,
        "--inpaint-prompt",
        inpaint_prompt,
        "--inpaint-steps",
        str(inpaint_steps),
        "--inpaint-seed",
        str(inpaint_seed),
        "--inpaint-device",
        inpaint_device,
    )
    return StageCommand(
        name="warped_reference",
        command=cmd,
        cwd=str(WARPED_REFERENCE_DIR),
        outputs={
            "warped_reference_dir": str(out_dir),
            "warped_reference_video": str(out_dir / "motion_signal.mp4"),
            "warped_reference_mask": str(out_dir / "mask.mp4"),
            "displacement_vis": str(out_dir / "disp_vis.mp4"),
            "first_frame": str(out_dir / "first_frame.png"),
            "sam2_mask0": str(out_dir / "mask0.png"),
            "background": str(out_dir / "background.png"),
        },
        notes=[
            "Runs fluidgen/warped_reference/get_warped_video.py after simulation.",
            "Uses simulation segmap/fg_mask.mp4 as --mask-video.",
            "Outputs motion_signal.mp4 as warped reference video and mask.mp4 as its corresponding mask video.",
        ],
    )


def build_vace_preprocess_stage(
    image_path: str,
    bundle: SimulationBundle,
    output_dir: str | None = None,
    vace_dir: str = paths.VACE_DIR,
    conda_env: str | None = "wan",
    save_fps: int = 16,
) -> StageCommand:
    if not bundle.flow_video_path:
        raise ValueError("VACE preprocess requires simulation optical_flow.mp4 as flow_video_path.")
    out_dir = Path(output_dir) if output_dir else Path(bundle.root_dir) / "vace_condition"
    cmd = conda_python_command(
        conda_env,
        Path(vace_dir) / "vace" / "vace_preproccess.py",
        "--task",
        "firstframe_ctrl",
        "--video",
        str(Path(bundle.flow_video_path).resolve()),
        "--image",
        str(Path(image_path).resolve()),
        "--pre_save_dir",
        str(out_dir),
        "--save_fps",
        str(save_fps),
    )
    return StageCommand(
        name="vace_preprocess",
        command=cmd,
        cwd=str(Path(vace_dir)),
        outputs={
            "vace_condition_dir": str(out_dir),
            "src_video": str(out_dir / "src_video-firstframe_ctrl.mp4"),
            "src_mask": str(out_dir / "src_mask-firstframe_ctrl.mp4"),
        },
        notes=[
            "Runs the patched VACE vace/vace_preproccess.py before multiclip VACE generation.",
            "Only task firstframe_ctrl is used.",
            "Input video is simulation optical_flow.mp4, not warped_reference/motion_signal.mp4.",
            "Output src_video/src_mask become the VACE source condition; warped reference remains the latent-guidance sim_video/sim_mask.",
        ],
    )


def run_stage(command: StageCommand, check: bool = True) -> StageCommand:
    result = subprocess.run(command.command, cwd=command.cwd, env={**os.environ, **command.env}, check=False)
    command.returncode = result.returncode
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command.command)
    return command


def attach_warped_reference(bundle: SimulationBundle, command: StageCommand) -> SimulationBundle:
    bundle.warped_reference_dir = command.outputs.get("warped_reference_dir")
    video = command.outputs.get("warped_reference_video")
    mask = command.outputs.get("warped_reference_mask")
    if video and Path(video).exists():
        bundle.warped_reference_video_path = video
    elif video:
        bundle.warped_reference_video_path = video
    if mask and Path(mask).exists():
        bundle.warped_reference_mask_path = mask
    elif mask:
        bundle.warped_reference_mask_path = mask
    return bundle


def attach_vace_preprocess_condition(bundle: SimulationBundle, command: StageCommand) -> SimulationBundle:
    bundle.vace_condition_dir = command.outputs.get("vace_condition_dir")
    src_video = command.outputs.get("src_video")
    src_mask = command.outputs.get("src_mask")
    if src_video:
        bundle.vace_src_video_path = src_video
    if src_mask:
        bundle.vace_src_mask_path = src_mask
    return bundle
