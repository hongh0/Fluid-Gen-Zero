"""Reconstruction orchestrator: single image -> simulation-ready scene.

    python run.py --image apple.png --out_dir runs/apple/s1_6_recon \
                  --object_prompt apple --object_size_m 0.08
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STAGES = ["segment", "geometry", "object3d", "fluid", "register", "assemble"]
HERE = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script)] + [str(a) for a in args]
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--object_prompt", default="")
    p.add_argument("--fluid_prompt", default="",
                   help="Liquid material word; derived from --scene_prompt when omitted.")
    p.add_argument("--scene_prompt", default="",
                   help="Full scene text; the intent parser reads the liquid material from it.")
    p.add_argument("--object_size_m", type=float, default=None,
                   help="Real-world size of the object in metres; sets absolute scale.")
    p.add_argument("--grid_density", type=int, default=None,
                   help="MPM cells per metre (default 64); 0 = size it from the object.")
    p.add_argument("--fluid_depth_factor", type=float, default=0.75,
                   help="Liquid depth as a multiple of the object's size (not observable).")
    p.add_argument("--fluid_depth_m", type=float, default=None,
                   help="Liquid depth in metres, overriding --fluid_depth_factor.")
    p.add_argument("--fluid_material_json", default="")
    p.add_argument("--object_material_json", default="")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--register_iters", type=int, default=1500,
                   help="Random-restart steps for the pose search. 200 is not enough and, "
                        "worse, its variance is large: the same sample scored IoU 0.36 in one "
                        "run and 0.74 in another, so samples passed or failed the gate on "
                        "luck. 1500 costs about two more minutes and is reproducible.")
    p.add_argument("--max_faces", type=int, default=40000)
    p.add_argument("--ignore_gate", action="store_true",
                   help="Assemble a scene even if registration missed its IoU gate.")
    a = p.parse_args()

    sys.path.insert(0, str(HERE))
    from contracts import (GeometryArtifact, ObjectArtifact, RegistrationArtifact,
                           SceneArtifact, SegmentationArtifact, FluidArtifact)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip() for s in a.stages.split(",") if s.strip()]
    d = {s: out / f"s{i + 1}_{s}" for i, s in enumerate(STAGES)}
    report = {}

    def fresh(marker: Path) -> bool:
        return a.force or not marker.exists()

    # ---- 1 segment ----------------------------------------------------
    marker = d["segment"] / "segmentation.json"
    if "segment" in wanted:
        if fresh(marker):
            _run("s1_segment.py", ["--image", a.image, "--out_dir", d["segment"],
                                   "--object_prompt", a.object_prompt,
                                   "--fluid_prompt", a.fluid_prompt,
                                   "--scene_prompt", a.scene_prompt, "--device", a.device])
        else:
            _log("segment: cached")
        art = SegmentationArtifact.load(marker)
        _log(f"segment: object={art.object_pixels}px fluid={art.fluid_pixels}px "
             f"background={art.background_pixels}px ({art.fluid_prompt or 'n/a'}) "
             f"overlap={art.overlap_pixels}px")
        report["segment"] = str(marker)

    # ---- 2 geometry ---------------------------------------------------
    marker = d["geometry"] / "geometry.json"
    if "geometry" in wanted:
        if fresh(marker):
            _run("s2_geometry.py", ["--image", a.image, "--out_dir", d["geometry"],
                                    "--device", a.device])
        else:
            _log("geometry: cached")
        geo = GeometryArtifact.load(marker)
        _log(f"geometry: {geo.width}x{geo.height} valid={geo.valid_ratio:.1%} "
             f"depth=[{geo.depth_min:.3f},{geo.depth_max:.3f}] metric={geo.metric}")
        report["geometry"] = str(marker)

    # ---- 3 object3d ---------------------------------------------------
    marker = d["object3d"] / "object3d.json"
    if "object3d" in wanted:
        if fresh(marker):
            seg = SegmentationArtifact.load(d["segment"] / "segmentation.json")
            _run("s3_object3d.py", ["--image", a.image, "--object_mask", seg.object_mask_path,
                                    "--out_dir", d["object3d"],
                                    "--points", d["geometry"] / "points.npy",
                                    "--geometry", d["geometry"] / "geometry.json",
                                    "--max_faces", a.max_faces])
        else:
            _log("object3d: cached")
        obj = ObjectArtifact.load(marker)
        _log(f"object3d: {obj.num_vertices} verts, {obj.num_faces} faces")
        report["object3d"] = str(marker)

    # ---- 4 fluid ------------------------------------------------------
    marker = d["fluid"] / "fluid.json"
    if "fluid" in wanted:
        if fresh(marker):
            _run("s4_fluid.py", ["--geometry", d["geometry"] / "geometry.json",
                                 "--segmentation", d["segment"] / "segmentation.json",
                                 "--out_dir", d["fluid"]])
        else:
            _log("fluid: cached")
        flu = FluidArtifact.load(marker)
        _log(f"fluid: inliers={flu.inlier_ratio:.1%} residual_z={flu.residual_z_median:.2e} "
             f"domain={flu.domain_type} size={[round(x, 3) for x in flu.domain_size]}")
        report["fluid"] = str(marker)

    # ---- 5 register ---------------------------------------------------
    marker = d["register"] / "registration.json"
    if "register" in wanted:
        if fresh(marker):
            args = ["--object", d["object3d"] / "object3d.json",
                    "--fluid", d["fluid"] / "fluid.json",
                    "--geometry", d["geometry"] / "geometry.json",
                    "--segmentation", d["segment"] / "segmentation.json",
                    "--out_dir", d["register"], "--iters", a.register_iters]
            if a.object_size_m:
                args += ["--object_size_m", a.object_size_m]
            _run("s5_register.py", args)
        else:
            _log("register: cached")
        reg = RegistrationArtifact.load(marker)
        _log(f"register: IoU={reg.mask_iou:.3f} passed={reg.passed} "
             f"state={reg.interaction_state} submerged={reg.submerged_fraction:.1%}")
        if not reg.passed and not a.ignore_gate:
            _log(f"STOP: registration IoU {reg.mask_iou:.3f} < {reg.IOU_GATE}. "
                 "Inspect s5_register/registered_silhouette.png, or pass --ignore_gate.")
            print(json.dumps({"status": "failed_registration_gate", **report}, indent=2))
            return 1
        report["register"] = str(marker)

    # ---- 6 assemble ---------------------------------------------------
    marker = d["assemble"] / "scene.json"
    if "assemble" in wanted:
        if fresh(marker):
            obj = ObjectArtifact.load(d["object3d"] / "object3d.json")
            args = ["--registration", d["register"] / "registration.json",
                    "--fluid", d["fluid"] / "fluid.json",
                    "--geometry", d["geometry"] / "geometry.json",
                    "--object_mesh", obj.mesh_path, "--out_dir", d["assemble"]]
            if a.object_size_m:
                # stage 5 fits apparent size; stage 6 turns world units into
                # metres, and needs the same anchor to do it
                args += ["--object_size_m", a.object_size_m]
            args += ["--fluid_depth_factor", a.fluid_depth_factor]
            if a.grid_density is not None:
                args += ["--grid_density", a.grid_density]
            if a.fluid_depth_m:
                args += ["--fluid_depth_m", a.fluid_depth_m]
            if a.fluid_material_json:
                args += ["--fluid_material_json", a.fluid_material_json]
            if a.object_material_json:
                args += ["--object_material_json", a.object_material_json]
            _run("s6_assemble.py", args)
        else:
            _log("assemble: cached")
        scene = SceneArtifact.load(marker)
        _log(f"assemble: state={scene.interaction_state} "
             f"domain={[round(x, 3) for x in scene.domain_lower]}.."
             f"{[round(x, 3) for x in scene.domain_upper]} "
             f"metric_scale={scene.metric_scale:.4f} grid_density={scene.grid_density} "
             f"vessel={scene.vessel.get('kind', 'none') if scene.vessel else 'none'}")
        report["assemble"] = str(marker)

    print(json.dumps({"status": "ok", **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
