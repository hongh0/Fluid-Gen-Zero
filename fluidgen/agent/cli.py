"""Generation-time planning CLI.

Takes the image, the prompt and a finished simulation bundle (from
`python -m fluidgen.pipeline`), and writes `agent_manifest.json`: the parsed
intent record, the temporal complexity curve, the clip plan with clip-level
prompts, and one generation job per clip for the chosen backbone. The jobs are
executed afterwards by `scripts/run_clips.py --manifest ...`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from fluidgen import paths

from .complexity import ComplexityConfig, SimulationComplexityAnalyzer
from .intent_parser import IntentPhysicsParser
from .openai_adapter import DEFAULT_OPENAI_MODEL, OpenAIAPIAdapter
from .prompt_refiner import PromptRefiner
from .routing import (
    DEFAULT_TORA_BASE_CONFIGS,
    DEFAULT_TORA_CKPT,
    DEFAULT_TORA_GUIDANCE_SCRIPT,
    DEFAULT_GUIDANCE_PARAMS,
    build_tora_clip_jobs,
    build_vace_clip_jobs,
)
from .schema import SimulationBundle, to_dict
from .stage_executor import (
    attach_vace_preprocess_condition,
    attach_warped_reference,
    build_vace_preprocess_stage,
    build_warped_reference_stage,
    run_stage,
    shell_join,
)
from .vlm_judge import VLMComplexityJudge


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fluid-Gen-Zero generation-time planning: simulation bundle -> clip plan + per-clip generation jobs.")
    parser.add_argument("--prompt", required=True, help="Input prompt p describing appearance, dynamics, and key events.")
    parser.add_argument("--image", default=None, help="Input image I_0 (the first frame).")
    parser.add_argument("--output_dir", default="agent_outputs", help="Where to write agent_manifest.json and the per-clip outputs.")

    # Backbones.
    parser.add_argument("--generation_backend", choices=["vace", "tora"], default="vace", help="Backbone that the clip jobs target. WanMove reuses the VACE plan (scripts/make_wanmove_input.py).")
    parser.add_argument("--vace_dir", default=paths.VACE_DIR, help="Patched VACE checkout (created by backbones/setup.sh).")
    parser.add_argument("--wan_conda_env", default=paths.WAN_CONDA_ENV, help="Conda env for VACE preprocessing/generation; empty = current interpreter.")
    parser.add_argument("--tora_dir", default=paths.TORA_DIR, help="Patched Tora checkout used when --generation_backend tora.")
    parser.add_argument("--tora_conda_env", default=paths.TORA_CONDA_ENV, help="Conda env for Tora generation; empty = current interpreter.")
    parser.add_argument("--tora_inference_script", default=DEFAULT_TORA_GUIDANCE_SCRIPT, help="Tora multi-clip sampling script with latent-space guidance.")
    parser.add_argument("--tora_base_configs", nargs="+", default=DEFAULT_TORA_BASE_CONFIGS, help="Config files passed to Tora --base.")
    parser.add_argument("--tora_ckpt", default=DEFAULT_TORA_CKPT, help="Checkpoint directory passed to Tora --load.")
    parser.add_argument("--tora_clip_frame_num", type=int, default=49, help="Generated/padded frame_num for each Tora clip; clips are trimmed to actual_frame_num before stitching.")
    parser.add_argument("--tora_sampling_fps", type=int, default=8, help="FPS for Tora clip and stitched outputs.")
    parser.add_argument("--tora_seed", type=int, default=1234, help="Base seed for Tora multiclip generation.")
    parser.add_argument("--tora_use_fluid_mask_center_trajectory", action="store_true", help="Also derive Tora trajectory flow from the simulation fluid-mask center, in addition to the object/warped mask center.")

    # Simulation-derived generation conditions that are built on demand.
    parser.add_argument("--run_warped_reference", action=argparse.BooleanOptionalAction, default=True, help="Build the appearance-preserving warped reference video V^sim (fluidgen/warped_reference) if the bundle lacks it. On by default.")
    parser.add_argument("--run_vace_preprocess", action=argparse.BooleanOptionalAction, default=True, help="Build the VACE native condition (vace_preproccess.py --task firstframe_ctrl on the simulation optical flow) if the bundle lacks it. On by default.")
    parser.add_argument("--warp_conda_env", default="", help="Conda env for the warped-reference stage (needs SAM2 + diffusers); empty = current interpreter.")
    parser.add_argument("--warp_size", default="480x832", help="HxW for get_warped_video.py --size.")
    parser.add_argument("--warp_fps", type=float, default=24.0, help="FPS for warped reference output.")
    parser.add_argument("--warp_sam2_checkpoint", default=paths.SAM2_CKPT)
    parser.add_argument("--warp_sam2_config", default="/configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--warp_sam2_device", default="cuda")
    parser.add_argument("--warp_inpaint_prompt", default="background, nothing")
    parser.add_argument("--warp_inpaint_steps", type=int, default=24)
    parser.add_argument("--warp_inpaint_seed", type=int, default=42)
    parser.add_argument("--warp_inpaint_device", default="cuda")
    parser.add_argument("--vace_preprocess_dir", default=None, help="Output directory for vace_preproccess.py firstframe_ctrl condition files.")
    parser.add_argument("--vace_preprocess_save_fps", type=int, default=16, help="Fallback --save_fps for vace_preproccess.py; input video FPS is used when available.")

    # Existing simulation result inputs.
    parser.add_argument("--simulation_dir", default=None, help="Directory containing fg_traj.npz/fluid_traj.npz/optical_flow_flow.npy/segmap.")
    parser.add_argument("--fg_traj", default=None, help="Override foreground trajectory npz path.")
    parser.add_argument("--fluid_traj", default=None, help="Override fluid trajectory npz path.")
    parser.add_argument("--flow_npy", default=None, help="Override raw optical-flow npy path.")
    parser.add_argument("--fluid_mask_video", default=None, help="Override fluid mask video path.")
    parser.add_argument("--fg_mask_video", default=None, help="Override foreground mask video path.")
    parser.add_argument("--warped_reference_video", default=None, help="Override warped reference motion_signal.mp4 path.")
    parser.add_argument("--warped_reference_mask", default=None, help="Override warped reference mask.mp4 path.")
    parser.add_argument("--vace_condition_video", default=None, help="Override vace_preproccess.py src_video-firstframe_ctrl.mp4 path.")
    parser.add_argument("--vace_condition_mask", default=None, help="Override vace_preproccess.py src_mask-firstframe_ctrl.mp4 path.")
    parser.add_argument(
        "--use_simulation_fg_mask_as_sim_mask",
        action="store_true",
        help="Route simulation fg_mask.mp4 as clip sim_mask instead of warped_reference/mask.mp4; clip slicing will resize the mask to match the routed sim video resolution.",
    )

    # Parser / VLM adapters.
    parser.add_argument("--llm_override_json", default=None, help="Optional JSON to override parser output.")
    parser.add_argument("--vlm_judgement_json", default=None, help="Optional VLM judgement JSON with risk intervals/no-cut hints.")
    parser.add_argument("--write_vlm_request", action="store_true", help="Write a structured VLM request JSON alongside the manifest.")
    parser.add_argument("--openai_model", default=DEFAULT_OPENAI_MODEL, help="VLM used for intent parsing, risk annotation and clip-level prompts (env FGZ_VLM_MODEL).")
    parser.add_argument("--openai_api_key", default="", help="OpenAI API key. Empty string means use OPENAI_API_KEY; if still empty, write requests and fall back.")
    parser.add_argument("--api_frame_interval", type=int, default=10, help="Sample every N frames for VLM API calls.")
    parser.add_argument("--api_max_frames_per_video", type=int, default=0, help="Optional cap per video after stride sampling; 0 = no cap.")
    parser.add_argument("--disable_openai_parser", action="store_true", help="Use only the rule-based intent parser (no VLM call).")
    parser.add_argument("--disable_openai_prompt_refinement", action="store_true", help="Keep the template clip-level prompts instead of VLM-refined ones.")
    parser.add_argument("--run_vlm_judge_api", action=argparse.BooleanOptionalAction, default=True, help="VLM risk annotation (risk intervals, no-cut flags, suggested cuts) from sampled simulation/flow/mask frames. On by default; --no-run_vlm_judge_api gives the metric-only planner.")

    # Complexity / planner knobs.
    parser.add_argument("--min_clip_frames", type=int, default=13, help="Shortest clip length, must be 4n+1.")
    parser.add_argument("--default_clip_frames", type=int, default=41, help="Default clip length, must be 4n+1.")
    parser.add_argument("--max_clip_frames", type=int, default=81, help="Longest clip length, must be 4n+1.")
    parser.add_argument("--high_threshold", type=float, default=0.72)
    parser.add_argument("--medium_threshold", type=float, default=0.45)
    parser.add_argument("--alpha_physical", type=float, default=0.40)
    parser.add_argument("--beta_visual", type=float, default=0.40)
    parser.add_argument("--gamma_semantic", type=float, default=0.20)
    parser.add_argument("--guidance_start_step", type=int, default=DEFAULT_GUIDANCE_PARAMS["guidance_start_step"], help="k_0: denoising step where sampling starts from the noised simulation reference.")
    parser.add_argument("--object_wrap_end_step", type=int, default=DEFAULT_GUIDANCE_PARAMS["object_wrap_end_step"], help="k_obj: object-region (binary) latent wrapping ends at this step.")
    parser.add_argument("--fluid_wrap_end_step", type=int, default=DEFAULT_GUIDANCE_PARAMS["fluid_wrap_end_step"], help="k_fl: fluid-region (decaying) latent wrapping ends at this step.")
    parser.add_argument("--fluid_wrap_strength", type=float, default=DEFAULT_GUIDANCE_PARAMS["fluid_wrap_strength"], help="beta^f_max: initial fluid-region wrapping strength.")
    parser.add_argument("--fluid_mask_sigma", type=float, default=DEFAULT_GUIDANCE_PARAMS["fluid_mask_sigma"], help="Gaussian smoothing (pixels) of the fluid mask m^f.")
    return parser


def _make_bundle(args: argparse.Namespace) -> SimulationBundle | None:
    if args.simulation_dir:
        bundle = SimulationBundle.from_dir(args.simulation_dir)
    elif (
        args.fg_traj
        or args.fluid_traj
        or args.flow_npy
        or args.fluid_mask_video
        or args.fg_mask_video
        or args.warped_reference_video
        or args.warped_reference_mask
        or args.vace_condition_video
        or args.vace_condition_mask
    ):
        bundle = SimulationBundle(root_dir=str(Path(args.output_dir).resolve()))
    else:
        return None
    if args.fg_traj:
        bundle.fg_traj_path = args.fg_traj
    if args.fluid_traj:
        bundle.fluid_traj_path = args.fluid_traj
    if args.flow_npy:
        bundle.flow_npy_path = args.flow_npy
    if args.fluid_mask_video:
        bundle.fluid_mask_video_path = args.fluid_mask_video
    if args.fg_mask_video:
        bundle.fg_mask_video_path = args.fg_mask_video
    if args.warped_reference_video:
        bundle.warped_reference_video_path = args.warped_reference_video
    if args.warped_reference_mask:
        bundle.warped_reference_mask_path = args.warped_reference_mask
    if args.vace_condition_video:
        bundle.vace_src_video_path = args.vace_condition_video
    if args.vace_condition_mask:
        bundle.vace_src_mask_path = args.vace_condition_mask
    return bundle


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    openai_adapter = OpenAIAPIAdapter(
        api_key=args.openai_api_key,
        model=args.openai_model,
        frame_interval=args.api_frame_interval,
        max_frames_per_video=args.api_max_frames_per_video,
        request_dir=out_dir / "api_requests",
    )

    if not openai_adapter.enabled:
        raise SystemExit("OPENAI_API_KEY is not set; Fluid-Gen-Zero requires the VLM.")
    parser = IntentPhysicsParser()
    if args.disable_openai_parser:
        intent = parser.parse(args.prompt, image_path=args.image, llm_override_json=args.llm_override_json)
    else:
        intent = parser.parse_with_openai(
            args.prompt,
            image_path=args.image,
            openai_adapter=openai_adapter,
            llm_override_json=args.llm_override_json,
            require=True,
        )
    refiner = PromptRefiner()
    manifest: Dict[str, Any] = {
        "intent": to_dict(intent),
        "sim_config": parser.export_sim_config(intent),
        "stage_commands": [],
        "openai": {
            "model": args.openai_model,
            "api_key_configured": openai_adapter.enabled,
            "api_frame_interval": args.api_frame_interval,
            "api_max_frames_per_video": args.api_max_frames_per_video,
            "request_dir": str(out_dir / "api_requests"),
        },
    }

    bundle = _make_bundle(args)
    if bundle is not None:
        if args.image and bundle.fg_mask_video_path and not (
            bundle.warped_reference_video_path and bundle.warped_reference_mask_path
        ):
            warped_cmd = build_warped_reference_stage(
                image_path=args.image,
                bundle=bundle,
                output_dir=str(Path(bundle.root_dir) / "warped_reference"),
                conda_env=args.warp_conda_env,
                size=args.warp_size,
                fps=args.warp_fps,
                sam2_checkpoint=args.warp_sam2_checkpoint,
                sam2_config=args.warp_sam2_config,
                sam2_device=args.warp_sam2_device,
                inpaint_prompt=args.warp_inpaint_prompt,
                inpaint_steps=args.warp_inpaint_steps,
                inpaint_seed=args.warp_inpaint_seed,
                inpaint_device=args.warp_inpaint_device,
            )
            manifest["stage_commands"].append(
                to_dict(warped_cmd)
                | {"shell": shell_join(warped_cmd.command), "plan_only": not args.run_warped_reference}
            )
            if args.run_warped_reference:
                warped_cmd = run_stage(warped_cmd)
                manifest["stage_commands"][-1] = to_dict(warped_cmd) | {"shell": shell_join(warped_cmd.command)}
                bundle = attach_warped_reference(bundle, warped_cmd)
            else:
                bundle = attach_warped_reference(bundle, warped_cmd)
        elif args.run_warped_reference and not args.image:
            raise ValueError("--run_warped_reference requires --image: the warped reference is built from the first frame.")
        elif args.run_warped_reference and not bundle.fg_mask_video_path:
            raise ValueError("--run_warped_reference requires simulation fg_mask video (segmap/fg_mask.mp4 or --fg_mask_video).")

        if args.generation_backend == "vace" and args.image and bundle.flow_video_path and not (
            bundle.vace_src_video_path and bundle.vace_src_mask_path
        ):
            preprocess_cmd = build_vace_preprocess_stage(
                image_path=args.image,
                bundle=bundle,
                output_dir=args.vace_preprocess_dir or str(Path(bundle.root_dir) / "vace_condition"),
                vace_dir=args.vace_dir,
                conda_env=args.wan_conda_env,
                save_fps=args.vace_preprocess_save_fps,
            )
            manifest["stage_commands"].append(
                to_dict(preprocess_cmd)
                | {"shell": shell_join(preprocess_cmd.command), "plan_only": not args.run_vace_preprocess}
            )
            if args.run_vace_preprocess:
                preprocess_cmd = run_stage(preprocess_cmd)
                manifest["stage_commands"][-1] = to_dict(preprocess_cmd) | {"shell": shell_join(preprocess_cmd.command)}
                bundle = attach_vace_preprocess_condition(bundle, preprocess_cmd)
            else:
                bundle = attach_vace_preprocess_condition(bundle, preprocess_cmd)
        elif args.run_vace_preprocess and not args.image:
            raise ValueError("--run_vace_preprocess requires --image because vace_preproccess.py firstframe_ctrl needs the first frame image.")
        elif args.run_vace_preprocess and not bundle.flow_video_path:
            raise ValueError("--run_vace_preprocess requires simulation optical_flow.mp4 (or a simulation_dir that contains it).")

        if args.generation_backend == "tora":
            if args.tora_clip_frame_num < 5 or (args.tora_clip_frame_num - 1) % 4 != 0:
                raise ValueError("--tora_clip_frame_num must be 4n+1 and >=5.")
            if args.max_clip_frames > args.tora_clip_frame_num:
                args.max_clip_frames = args.tora_clip_frame_num
            if args.default_clip_frames > args.max_clip_frames:
                args.default_clip_frames = args.max_clip_frames
            if args.min_clip_frames > args.default_clip_frames:
                args.min_clip_frames = args.default_clip_frames

        cfg = ComplexityConfig(
            min_clip_frames=args.min_clip_frames,
            default_clip_frames=args.default_clip_frames,
            max_clip_frames=args.max_clip_frames,
            high_threshold=args.high_threshold,
            medium_threshold=args.medium_threshold,
            alpha_physical=args.alpha_physical,
            beta_visual=args.beta_visual,
            gamma_semantic=args.gamma_semantic,
        )
        vlm_suggestions = VLMComplexityJudge.load_suggestions(args.vlm_judgement_json)
        analyzer = SimulationComplexityAnalyzer(cfg)
        base_prompt = refiner.refine_global(intent)
        plan = analyzer.analyze(bundle, intent=intent, base_prompt=base_prompt, vlm_suggestions=vlm_suggestions)
        if args.run_vlm_judge_api:
            api_suggestions, api_payload = VLMComplexityJudge.judge_with_openai(
                args.prompt,
                bundle,
                plan,
                openai_adapter,
            )
            manifest["openai_vlm_complexity_response"] = api_payload
            if api_suggestions:
                vlm_suggestions = [*vlm_suggestions, *api_suggestions]
                plan = analyzer.analyze(bundle, intent=intent, base_prompt=base_prompt, vlm_suggestions=vlm_suggestions)
        if args.disable_openai_prompt_refinement:
            refiner.apply(intent, plan.clips, plan)
            manifest["openai_prompt_refinement_response"] = {"_api_status": "disabled"}
        else:
            manifest["openai_prompt_refinement_response"] = refiner.apply_with_openai(
                intent,
                plan.clips,
                plan,
                openai_adapter=openai_adapter,
                image_path=args.image,
            )
        manifest["simulation_bundle"] = to_dict(bundle)
        manifest["complexity_plan"] = to_dict(plan)
        run_clips = paths.REPO_ROOT / "scripts" / "run_clips.py"
        manifest_path = out_dir / "agent_manifest.json"
        guidance_params = {
            "guidance_start_step": args.guidance_start_step,
            "object_wrap_end_step": args.object_wrap_end_step,
            "fluid_wrap_end_step": args.fluid_wrap_end_step,
            "fluid_wrap_strength": args.fluid_wrap_strength,
            "fluid_mask_sigma": args.fluid_mask_sigma,
        }
        if args.generation_backend == "tora":
            tora_clip_jobs = build_tora_clip_jobs(
                bundle,
                plan.clips,
                output_dir=str(out_dir / "tora_clips"),
                input_image_or_video_path=args.image,
                inference_script=args.tora_inference_script,
                tora_dir=args.tora_dir,
                tora_conda_env=args.tora_conda_env,
                base_configs=list(args.tora_base_configs),
                ckpt=args.tora_ckpt,
                fixed_frame_num=args.tora_clip_frame_num,
                sampling_fps=args.tora_sampling_fps,
                seed=args.tora_seed,
                guidance_params=guidance_params,
                use_fluid_mask_center_trajectory=args.tora_use_fluid_mask_center_trajectory,
                use_simulation_fg_mask_as_sim_mask=args.use_simulation_fg_mask_as_sim_mask,
            )
            manifest["generation_backend"] = "tora"
            manifest["tora_clip_jobs"] = tora_clip_jobs
            manifest["dynamic_clip_generation_command"] = [
                "python",
                str(run_clips),
                "--manifest",
                str(manifest_path),
            ]
            manifest["dynamic_clip_generation_notes"] = [
                "Tora generation uses sample_video_latent_guidance.py so the Tora model is loaded once and all planned clips are generated in one process.",
                "Parser, reconstruction, simulation, warped reference, mask routing, complexity planning, prompt refinement, and optional VLM judging remain shared with the VACE backend.",
                (
                    "Warped reference motion_signal.mp4 is routed as Tora sim_video, while simulation fg_mask.mp4 is routed as sim_mask."
                    if args.use_simulation_fg_mask_as_sim_mask
                    else "Warped reference motion_signal.mp4/mask.mp4 are routed as Tora sim_video/sim_mask."
                ),
                "Each Tora clip is padded to --tora_clip_frame_num and trimmed to actual_frame_num before stitching.",
                "Use --dry_run on scripts/run_clips.py to preview the resolved Tora command without running the model.",
            ]
        else:
            vace_clip_jobs = build_vace_clip_jobs(
                bundle,
                plan.clips,
                output_dir=str(out_dir / "vace_clips"),
                wan_conda_env=args.wan_conda_env,
                guidance_params=guidance_params,
                use_simulation_fg_mask_as_sim_mask=args.use_simulation_fg_mask_as_sim_mask,
            )
            manifest["generation_backend"] = "vace"
            manifest["vace_clip_jobs"] = vace_clip_jobs
            manifest["dynamic_clip_generation_command"] = [
                "python",
                str(run_clips),
                "--manifest",
                str(manifest_path),
            ]
            manifest["dynamic_clip_generation_notes"] = [
                "The dynamic executor now defaults to vace_wan_inference_latent_guidance.py, so Wan/VACE is loaded once and all planned clips are generated in one process.",
                "Each planned clip can still have a different frame_num, e.g. 41/21/81.",
                "VACE src_video/src_mask are produced by vace_preproccess.py firstframe_ctrl from simulation optical_flow.mp4 plus the input first frame.",
                (
                    "Warped reference motion_signal.mp4 is routed as the latent-guidance reference (sim_video), while simulation fg_mask.mp4 is routed as sim_mask, not as the VACE preprocess input."
                    if args.use_simulation_fg_mask_as_sim_mask
                    else "Warped reference motion_signal.mp4/mask.mp4 are routed as the latent-guidance reference (sim_video/sim_mask), not as the VACE preprocess input."
                ),
                "If vace_condition files are only planned, pass --run_preprocess to scripts/run_clips.py before clip generation.",
                "Use --dry_run on scripts/run_clips.py to preview the resolved multiclip command without running the model.",
            ]

        if args.write_vlm_request:
            request = VLMComplexityJudge.build_request(args.prompt, bundle, plan)
            request_path = out_dir / "vlm_complexity_request.json"
            VLMComplexityJudge.write_request(request_path, request)
            manifest["vlm_request_path"] = str(request_path)

    else:
        manifest["complexity_plan"] = None
        manifest["notes"] = [
            "No simulation result passed or executed; wrote parser, stage command specs, and sim config only."
        ]

    _write_json(out_dir / "agent_manifest.json", manifest)
    print(f"Saved agent manifest to {out_dir / 'agent_manifest.json'}")
    return manifest


if __name__ == "__main__":
    main()
