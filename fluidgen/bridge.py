"""The seam between the agent and the reconstruction/simulation stages.

Translates an `IntentParseResult` into the stages' arguments, and lays their
outputs out under the names `SimulationBundle.from_dir` looks for.

Nothing here calls a model. It is argument plumbing plus a file layout, kept in
one place so the agent does not grow knowledge of stage internals and the
stages do not grow knowledge of the agent's schema.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
MPM_E_CEILING = 5e6

#: Files `SimulationBundle.from_dir` looks for, mapped to where this pipeline
#: actually writes them. Relative to a case directory.
BUNDLE_LAYOUT = {
    "fg_traj.npz": "s8_sim/fg_traj.npz",
    "fluid_traj.npz": "s8_sim/fluid_traj.npz",
    "simulation.mp4": "s8_sim/simulation.mp4",
    "optical_flow.mp4": "s9_cond/optical_flow.mp4",
    # the bundle's name for the raw flow array differs from ours
    "optical_flow_flow.npy": "s9_cond/optical_flow.npy",
    "segmap/fg_mask.mp4": "s9_cond/fg_mask.mp4",
    "segmap/fluid_mask.mp4": "s9_cond/fluid_mask.mp4",
    "segmap/segmap.mp4": "s9_cond/segmap.mp4",
    # not part of the bundle contract, but the clip planner reads 2D tracks
    "trajectory_2d.npz": "s9_cond/trajectory_2d.npz",
    "object_trajectory.txt": "s9_cond/object_trajectory.txt",
    "fluid_trajectory.txt": "s9_cond/fluid_trajectory.txt",
}


# --------------------------------------------------------------------------
# intent -> stage arguments
# --------------------------------------------------------------------------
def fluid_material(intent) -> Dict[str, Any]:
    """The liquid's physical properties, as stage 6 wants them."""
    dyn, app = intent.dynamics, intent.appearance
    return {
        "material": app.material,
        "rho": float(dyn.liquid_rho),
        "viscous": bool(dyn.liquid_viscous),
        "viscosity": float(dyn.viscosity),
    }


def object_kind_from(intent, override: str = "") -> str:
    """The object's material type: an explicit override, else what the VLM saw."""
    if override:
        return override
    g = getattr(intent, "object_grounding", None) if intent is not None else None
    return (g.object_kind if g and g.object_kind else "rigid")


def object_material(intent, kind: str = "rigid") -> Dict[str, Any]:
    """The object's physical properties.

    `rho` is absent unless the caller pins it; stage 6 then uses its default
    object density.
    """
    dyn = intent.dynamics
    mat: Dict[str, Any] = {"type": kind, "nu": float(dyn.elastic_nu)}
    if kind == "rigid":
        mat["coup_friction"] = 0.1
        return mat
    E = float(dyn.elastic_E)
    if E > MPM_E_CEILING:
        mat["notes"] = (f"E {E:.3g} clamped to {MPM_E_CEILING:.3g}; a solid that stiff "
                        f"freezes in Genesis' MPM and belongs in the rigid solver")
        E = MPM_E_CEILING
    mat["E"] = E
    return mat


def write_material_files(intent, out_dir: str | Path, object_kind: str = "rigid") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, payload in (("fluid_material.json", fluid_material(intent)),
                          ("object_material.json", object_material(intent, object_kind))):
        p = out / name
        p.write_text(json.dumps(payload, indent=2) + "\n")
        paths[name] = str(p)
    return paths


def reconstruction_argv(image: str, out_dir: str | Path, intent=None, *,
                        object_prompt: str = "", object_size_m: Optional[float] = None,
                        fluid_prompt: str = "", material_files: Optional[Dict[str, str]] = None,
                        force: bool = True, ignore_gate: bool = False,
                        extra: Optional[List[str]] = None) -> List[str]:
    """Arguments for `reconstruction/run.py`.

    `object_size_m` is the only metric anchor in the whole pipeline -- MoGe is
    defined up to scale -- so it is worth being explicit that omitting it leaves
    the scene in arbitrary units.
    """
    grounding = getattr(intent, "object_grounding", None) if intent is not None else None
    if not object_prompt and grounding is not None:
        object_prompt = grounding.object_prompt or ""
    if object_size_m is None and grounding is not None:
        object_size_m = grounding.object_size_m

    argv = ["--image", str(image), "--out_dir", str(out_dir)]
    if object_prompt:
        argv += ["--object_prompt", object_prompt]
    if fluid_prompt:
        argv += ["--fluid_prompt", fluid_prompt]
    elif intent is not None and intent.appearance.material:
        argv += ["--fluid_prompt", intent.appearance.material]
    if intent is not None and intent.raw_prompt:
        argv += ["--scene_prompt", intent.raw_prompt]
    if object_size_m:
        argv += ["--object_size_m", str(object_size_m)]
    for key, flag in (("fluid_material.json", "--fluid_material_json"),
                      ("object_material.json", "--object_material_json")):
        if material_files and key in material_files:
            argv += [flag, material_files[key]]
    if force:
        argv.append("--force")
    if ignore_gate:
        argv.append("--ignore_gate")
    return argv + list(extra or [])


def rollout_argv(scene_json: str | Path, out_dir: str | Path, intent=None, *,
                 num_frames: int = 401, force_seed: int = 0,
                 force_accel: Optional[float] = None,
                 extra: Optional[List[str]] = None) -> List[str]:
    """Arguments for `simulation/rollout.py`.
    """
    argv = ["--scene", str(scene_json), "--out_dir", str(out_dir),
            "--num_frames", str(int(num_frames)), "--force_seed", str(int(force_seed))]
    if force_accel is not None:
        argv += ["--force_accel", str(force_accel)]
    elif intent is not None and intent.dynamics.push_force is not None:
        argv += ["--force_accel", str(float(intent.dynamics.push_force))]
    return argv + list(extra or [])


# --------------------------------------------------------------------------
# outputs -> SimulationBundle layout
# --------------------------------------------------------------------------
def publish_bundle(case_dir: str | Path, bundle_dir: str | Path,
                   copy: bool = False) -> Dict[str, str]:
    """Expose a case's outputs under the names the agent already looks for.

    Symlinks by default: the trajectories are hundreds of megabytes and there is
    no reason to have two of them. `copy=True` for when the bundle has to
    outlive the case directory.
    """
    case, bundle = Path(case_dir), Path(bundle_dir)
    published, missing = {}, []
    for name, rel in BUNDLE_LAYOUT.items():
        src = case / rel
        dst = bundle / name
        if not src.exists():
            missing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
        published[name] = str(dst)
    if missing:
        (bundle / "MISSING.txt").write_text("\n".join(missing) + "\n")
    return published


def build_reconstruction_stage(image: str, output_dir: str, intent=None, *,
                               python: str, repo: str, object_prompt: str = "",
                               object_size_m: Optional[float] = None,
                               fluid_prompt: str = "", ignore_gate: bool = False):
    """A `StageCommand` running this repo's reconstruction on a single image."""
    from .agent.schema import StageCommand

    out = Path(output_dir).resolve() / "reconstruction"
    mats = write_material_files(intent, out) if intent is not None else None
    argv = reconstruction_argv(image, out, intent, object_prompt=object_prompt,
                               object_size_m=object_size_m, fluid_prompt=fluid_prompt,
                               material_files=mats, ignore_gate=ignore_gate)
    return StageCommand(
        name="reconstruction",
        command=[python, str(Path(repo) / "fluidgen/reconstruction/run.py")] + argv,
        cwd=str(repo),
        outputs={"scene_json": str(out / "s6_assemble" / "scene.json"), "root": str(out)},
        notes=["single image -> segmentation, MoGe geometry, SAM3D mesh, fluid plane, "
               "registration, Genesis scene"])


def build_simulation_stage(scene_json: str, output_dir: str, intent=None, *,
                           python: str, repo: str, num_frames: int = 401,
                           force_seed: int = 0, force_accel: Optional[float] = None):
    """A `StageCommand` rolling a reconstructed scene forward and rendering it."""
    from .agent.schema import StageCommand

    out = Path(output_dir).resolve() / "simulation"
    argv = rollout_argv(scene_json, out / "s8_sim", intent, num_frames=num_frames,
                        force_seed=force_seed, force_accel=force_accel)
    return StageCommand(
        name="simulation",
        command=[python, str(Path(repo) / "fluidgen/simulation/rollout.py")] + argv,
        cwd=str(repo),
        outputs={"sim_dir": str(out / "s8_sim"), "root": str(out)},
        notes=[f"{num_frames} frames; conditions are rendered by the following stage"])


def build_conditions_stage(scene_json: str, sim_dir: str, output_dir: str, *,
                           python: str, repo: str):
    """A `StageCommand` turning a rollout into flow / masks / 2D tracks."""
    from .agent.schema import StageCommand

    out = Path(output_dir).resolve() / "simulation" / "s9_cond"
    return StageCommand(
        name="conditions",
        command=[python, str(Path(repo) / "fluidgen/simulation/render_conditions.py"),
                 "--scene", str(scene_json), "--sim_dir", str(sim_dir), "--out_dir", str(out)],
        cwd=str(repo),
        outputs={"cond_dir": str(out)},
        notes=["projected with the same calibrated camera as the render, so flow, masks "
               "and tracks are mutually consistent"])
