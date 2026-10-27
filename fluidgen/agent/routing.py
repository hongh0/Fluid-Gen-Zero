"""Route planned clips to per-backbone generation job specs (VACE or Tora).

Latent-space guidance parameters (paper notation):

    guidance_start_step   k_0         sampling starts from the noised simulation reference
    object_wrap_end_step  k_obj       binary object-region wrapping for k_0 <= k < k_obj
    fluid_wrap_end_step   k_fl        decaying fluid-region wrapping for k_0 <= k < k_fl
    fluid_wrap_strength   beta^f_max  initial fluid-region wrapping strength
    fluid_mask_sigma      Gaussian smoothing of the fluid mask m^f
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fluidgen import paths

from .schema import ClipPlan, SimulationBundle

DEFAULT_VACE_GUIDANCE_SCRIPT = str(Path(paths.VACE_DIR) / "vace" / "vace_wan_inference_latent_guidance.py")
DEFAULT_TORA_GUIDANCE_SCRIPT = str(Path(paths.TORA_DIR) / "sat" / "sample_video_latent_guidance.py")
DEFAULT_TORA_BASE_CONFIGS = [
    "configs/tora/model/cogvideox_5b_tora_i2v.yaml",
    "configs/tora/inference_sparse_json.yaml",
]
DEFAULT_TORA_CKPT = paths.TORA_CKPT
DEFAULT_GUIDANCE_PARAMS = {
    "guidance_start_step": 8,
    "object_wrap_end_step": 11,
    "fluid_wrap_end_step": 10,
    "fluid_wrap_strength": 0.8,
    "fluid_mask_sigma": 16.0,
}


def _append_guidance_params(command: list[str], params: Dict[str, Any]) -> None:
    for key in [
        "guidance_start_step",
        "object_wrap_end_step",
        "fluid_wrap_end_step",
        "fluid_wrap_strength",
        "fluid_mask_sigma",
    ]:
        value = params.get(key)
        if value is not None:
            command.extend([f"--{key}", str(value)])


def _first_existing(paths: list[Optional[str]]) -> Optional[str]:
    for path in paths:
        if path and Path(path).exists():
            return path
    return None


def _prefer_expected(preferred: Optional[str], fallbacks: list[Optional[str]]) -> Optional[str]:
    if preferred:
        return preferred
    return _first_existing(fallbacks)


def _ceil_4n1(length: int) -> int:
    return ((max(1, int(length)) - 1 + 3) // 4) * 4 + 1


def build_vace_clip_jobs(
    bundle: SimulationBundle,
    clips: List[ClipPlan],
    output_dir: str,
    inference_script: str = DEFAULT_VACE_GUIDANCE_SCRIPT,
    wan_conda_env: str | None = paths.WAN_CONDA_ENV,
    guidance_params: Optional[Dict[str, Any]] = None,
    use_simulation_fg_mask_as_sim_mask: bool = False,
) -> List[Dict[str, Any]]:
    """Build VACE clip job specs without running model inference.

    The target generation entrypoint is vace_wan_inference_latent_guidance.py,
    which loads Wan/VACE once and consumes the full list of clip specs from the
    agent manifest. Per-clip job records still store the clip interval, prompt,
    and long condition video paths for routing/reporting.
    """
    segmap_dir = Path(bundle.segmap_dir) if bundle.segmap_dir else None
    guidance_params = {**DEFAULT_GUIDANCE_PARAMS, **(guidance_params or {})}

    sim_video = _prefer_expected(bundle.warped_reference_video_path, [
        bundle.sim_video_path,
        bundle.segmap_video_path,
        bundle.flow_video_path,
    ])
    if use_simulation_fg_mask_as_sim_mask:
        sim_mask = _prefer_expected(bundle.fg_mask_video_path, [
            str(segmap_dir / "fg_mask.mp4") if segmap_dir else None,
            bundle.segmap_video_path,
            str(segmap_dir / "segmap.mp4") if segmap_dir else None,
            bundle.warped_reference_mask_path,
        ])
    else:
        sim_mask = _prefer_expected(bundle.warped_reference_mask_path, [
            bundle.segmap_video_path,
            bundle.fg_mask_video_path,
            str(segmap_dir / "segmap.mp4") if segmap_dir else None,
            str(segmap_dir / "fg_mask.mp4") if segmap_dir else None,
        ])
    fluid_mask = _first_existing([
        bundle.fluid_mask_video_path,
        str(segmap_dir / "fluid_mask.mp4") if segmap_dir else None,
    ])

    # VACE source conditioning should come from vace_preproccess.py
    # firstframe_ctrl, which combines simulation optical_flow.mp4 with the
    # input first frame. Warped reference is reserved for latent-space guidance (sim_video/sim_mask).
    src_video = _prefer_expected(bundle.vace_src_video_path, [
        bundle.sim_video_path,
        bundle.segmap_video_path,
        bundle.flow_video_path,
    ])
    src_mask = _prefer_expected(bundle.vace_src_mask_path, [
        bundle.segmap_video_path,
        bundle.fluid_mask_video_path,
        bundle.fg_mask_video_path,
        str(segmap_dir / "segmap.mp4") if segmap_dir else None,
        str(segmap_dir / "fluid_mask.mp4") if segmap_dir else None,
        str(segmap_dir / "fg_mask.mp4") if segmap_dir else None,
    ])

    jobs: List[Dict[str, Any]] = []
    for clip in clips:
        save_dir = str(Path(output_dir) / f"clip_{clip.clip_id:03d}")
        save_file = str(Path(save_dir) / "out_video.mp4")
        input_dir = Path(save_dir) / "inputs"
        sliced_inputs = {
            "src_video": str(input_dir / "src_video.mp4"),
            "src_mask": str(input_dir / "src_mask.mp4"),
            "sim_video": str(input_dir / "sim_video.mp4"),
            "sim_mask": str(input_dir / "sim_mask.mp4"),
            "fluid_mask": str(input_dir / "fluid_mask.mp4"),
        }
        args: Dict[str, Any] = {
            "inference_script": inference_script,
            "wan_conda_env": wan_conda_env,
            "clip_id": clip.clip_id,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "actual_frame_num": clip.actual_frame_num,
            "frame_num": clip.frame_num,
            "segment_frame_num": clip.frame_num,
            "prompt": clip.prompt,
            "risk_tags": clip.risk_tags,
            "anchor_strategy": clip.anchor_strategy,
            "save_dir": save_dir,
            "save_file": save_file,
            "final_save_file": save_file,
            "sliced_inputs": sliced_inputs,
            "guidance_params": dict(guidance_params),
        }
        if src_video:
            args["src_video"] = src_video
        if src_mask:
            args["src_mask"] = src_mask
        if sim_video:
            args["sim_video"] = sim_video
        if sim_mask:
            args["sim_mask"] = sim_mask
        if fluid_mask:
            args["fluid_mask"] = fluid_mask
        if bundle.flow_npy_path:
            args["flow_path_metadata"] = bundle.flow_npy_path
        missing_required = []
        if not src_video:
            missing_required.append("src_video")
        if missing_required:
            args["missing_required_inputs"] = missing_required

        command = []
        if wan_conda_env:
            command.extend(["conda", "run", "-n", wan_conda_env, "--no-capture-output"])
        command.extend([
            "python",
            inference_script,
            "--manifest",
            "<agent_manifest.json>",
        ])

        args["command"] = command
        args["notes"] = [
            "Generation uses vace_wan_inference_latent_guidance.py as the default script.",
            "src_video/src_mask should come from vace_preproccess.py --task firstframe_ctrl using simulation optical_flow.mp4 plus the input first frame.",
            (
                "sim_video uses the warped reference motion_signal.mp4 when available; sim_mask uses simulation fg_mask.mp4 with per-clip resolution alignment."
                if use_simulation_fg_mask_as_sim_mask
                else "sim_video/sim_mask use the warped reference motion_signal.mp4/mask.mp4 when available."
            ),
            "Default latent-guidance params: guidance_start_step=8, object_wrap_end_step=11, fluid_wrap_end_step=10, fluid_wrap_strength=0.8, fluid_mask_sigma=16.0.",
            "Run run_dynamic_vace_clips.py with the manifest; it now defaults to the multi-clip path.",
            "frame_num is per-clip and must satisfy 4n+1; actual_frame_num is trimmed after internal padding.",
        ]
        jobs.append(args)
    return jobs


def build_tora_clip_jobs(
    bundle: SimulationBundle,
    clips: List[ClipPlan],
    output_dir: str,
    input_image_or_video_path: str | None,
    inference_script: str = DEFAULT_TORA_GUIDANCE_SCRIPT,
    tora_dir: str = paths.TORA_DIR,
    tora_conda_env: str | None = paths.TORA_CONDA_ENV,
    base_configs: Optional[List[str]] = None,
    ckpt: str = DEFAULT_TORA_CKPT,
    fixed_frame_num: int | None = 49,
    sampling_fps: int = 8,
    seed: int = 1234,
    guidance_params: Optional[Dict[str, Any]] = None,
    use_fluid_mask_center_trajectory: bool = False,
    use_simulation_fg_mask_as_sim_mask: bool = False,
) -> List[Dict[str, Any]]:
    """Build Tora latent-guidance clip job specs without running inference.

    Tora's current CogVideoX checkpoint is trained around 49-frame clips.  By
    default each planned segment is therefore padded to 49 generated frames and
    trimmed back to ``actual_frame_num`` before final stitching.
    """
    segmap_dir = Path(bundle.segmap_dir) if bundle.segmap_dir else None
    guidance_params = {**DEFAULT_GUIDANCE_PARAMS, **(guidance_params or {})}
    base_configs = list(base_configs or DEFAULT_TORA_BASE_CONFIGS)

    warped_video = _prefer_expected(bundle.warped_reference_video_path, [
        bundle.sim_video_path,
        bundle.segmap_video_path,
        bundle.flow_video_path,
    ])
    if use_simulation_fg_mask_as_sim_mask:
        warped_mask = _prefer_expected(bundle.fg_mask_video_path, [
            str(segmap_dir / "fg_mask.mp4") if segmap_dir else None,
            bundle.segmap_video_path,
            str(segmap_dir / "segmap.mp4") if segmap_dir else None,
            bundle.warped_reference_mask_path,
        ])
    else:
        warped_mask = _prefer_expected(bundle.warped_reference_mask_path, [
            bundle.fg_mask_video_path,
            bundle.segmap_video_path,
            str(segmap_dir / "fg_mask.mp4") if segmap_dir else None,
            str(segmap_dir / "segmap.mp4") if segmap_dir else None,
        ])
    fluid_mask = _first_existing([
        bundle.fluid_mask_video_path,
        str(segmap_dir / "fluid_mask.mp4") if segmap_dir else None,
    ])

    jobs: List[Dict[str, Any]] = []
    for clip in clips:
        if fixed_frame_num is None:
            frame_num = clip.frame_num
        else:
            frame_num = int(fixed_frame_num)
        if frame_num < clip.actual_frame_num:
            frame_num = _ceil_4n1(clip.actual_frame_num)
        latent_num_frames = (frame_num - 1) // 4 + 1

        save_dir = str(Path(output_dir) / f"clip_{clip.clip_id:03d}")
        save_file = str(Path(save_dir) / "out_video.mp4")
        input_dir = Path(save_dir) / "inputs"
        sliced_inputs = {
            "warped_video_path": str(input_dir / "warped_video.mp4"),
            "warped_mask": str(input_dir / "warped_mask.mp4"),
            "sim_fluid_mask": str(input_dir / "sim_fluid_mask.mp4"),
        }
        args: Dict[str, Any] = {
            "backend": "tora",
            "inference_script": inference_script,
            "tora_dir": tora_dir,
            "tora_conda_env": tora_conda_env,
            "base_configs": base_configs,
            "ckpt": ckpt,
            "seed": seed,
            "sampling_fps": sampling_fps,
            "clip_id": clip.clip_id,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "actual_frame_num": clip.actual_frame_num,
            "frame_num": frame_num,
            "latent_num_frames": latent_num_frames,
            "prompt": clip.prompt,
            "risk_tags": clip.risk_tags,
            "anchor_strategy": clip.anchor_strategy,
            "save_dir": save_dir,
            "save_file": save_file,
            "final_save_file": save_file,
            "video_path": input_image_or_video_path,
            "sliced_inputs": sliced_inputs,
            "guidance_params": dict(guidance_params),
            "use_fluid_mask_center_trajectory": bool(use_fluid_mask_center_trajectory),
        }
        if warped_video:
            args["warped_video_path"] = warped_video
            args["sim_video"] = warped_video
        if warped_mask:
            args["warped_mask"] = warped_mask
            args["sim_mask"] = warped_mask
        if fluid_mask:
            args["sim_fluid_mask"] = fluid_mask
            args["fluid_mask"] = fluid_mask
        if bundle.flow_npy_path:
            args["flow_path_metadata"] = bundle.flow_npy_path

        missing_required = []
        if not input_image_or_video_path:
            missing_required.append("video_path")
        if not warped_video:
            missing_required.append("warped_video_path")
        if not warped_mask:
            missing_required.append("warped_mask")
        if missing_required:
            args["missing_required_inputs"] = missing_required

        command = []
        if tora_conda_env:
            command.extend(["conda", "run", "-n", tora_conda_env, "--no-capture-output"])
        command.extend([
            "python",
            inference_script,
            "--manifest",
            "<agent_manifest.json>",
            "--base",
            *base_configs,
            "--load",
            ckpt,
        ])

        args["command"] = command
        args["notes"] = [
            "Generation uses Tora sample_video_latent_guidance.py.",
            (
                "warped_reference motion_signal.mp4 is routed to Tora as sim_video, while sim_mask uses simulation fg_mask.mp4 with per-clip resolution alignment."
                if use_simulation_fg_mask_as_sim_mask
                else "warped_reference motion_signal.mp4/mask.mp4 are routed to Tora as sim_video/sim_mask."
            ),
            "Tora derives trajectory flow from the sliced warped_mask center and uses fluid_mask for fluid-region latent wrapping.",
            "When use_fluid_mask_center_trajectory=true, Tora also derives trajectory flow from the sliced fluid_mask center.",
            "Later clips use the previous generated last frame as their first-frame anchor by default.",
            f"frame_num is padded to {frame_num} for Tora and trimmed to actual_frame_num before stitching.",
        ]
        jobs.append(args)
    return jobs
