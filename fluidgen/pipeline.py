"""One image and a prompt, all the way to a bundle the agent can generate from.

    python -m fluidgen.pipeline --image examples/apple.png \
        --prompt "An apple falls into a vat of cider, sending up a spray." \
        --out_dir runs/apple

The pipeline stops at the simulation bundle; generation-time planning
(`scripts/run_agent.py --simulation_dir <out>/bundle`) continues from there.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fluidgen import bridge  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(argv, cwd=None) -> int:
    _log("$ " + " ".join(str(a) for a in argv[:6]) + (" ..." if len(argv) > 6 else ""))
    return subprocess.run([str(a) for a in argv], cwd=cwd).returncode


def parse_intent(prompt: str, image: str, override_json: str = ""):
    """Intent record from the VLM on top of the rule-based parse."""
    from fluidgen.agent.intent_parser import IntentPhysicsParser
    from fluidgen.agent.openai_adapter import OpenAIAPIAdapter

    parser = IntentPhysicsParser()
    adapter = OpenAIAPIAdapter()
    if not adapter.enabled:
        raise SystemExit("OPENAI_API_KEY is not set; Fluid-Gen-Zero requires the VLM.")
    return parser.parse_with_openai(prompt, image, adapter,
                                    llm_override_json=override_json or None, require=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--object_prompt", default="",
                   help="Text the segmenter grounds the object with; defaults to the "
                        "parser's guess from the prompt.")
    p.add_argument("--object_size_m", type=float, default=None,
                   help="The object's real size. This is the only metric anchor in the "
                        "pipeline -- MoGe is defined up to scale -- so without it the "
                        "scene is in arbitrary units and gravity means nothing.")
    p.add_argument("--fluid_prompt", default="")
    p.add_argument("--num_frames", type=int, default=401)
    p.add_argument("--force_accel", type=float, default=None)
    p.add_argument("--force_seed", type=int, default=0)
    p.add_argument("--object_kind", default="",
                   help="rigid, or one of elastic/elastoplastic/sand/snow/muscle/liquid. "
                        "Empty means take it from the parser, which sees the image.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--llm_override_json", default="",
                   help="JSON file overriding parsed fields; applied before the LLM pass.")
    p.add_argument("--stages", default="reconstruct,simulate,conditions,bundle")
    p.add_argument("--ignore_gate", action="store_true",
                   help="Assemble a scene even if registration missed its IoU gate.")
    a = p.parse_args()

    # Every stage runs as a subprocess with its own working directory.
    a.image = str(Path(a.image).resolve())
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip() for s in a.stages.split(",") if s.strip()}
    recon_dir, sim_dir, cond_dir = out / "s1_6_recon", out / "s8_sim", out / "s9_cond"
    scene_json = recon_dir / "s6_assemble" / "scene.json"

    intent = parse_intent(a.prompt, a.image,
                          override_json=a.llm_override_json)
    kind = bridge.object_kind_from(intent, a.object_kind)
    mats = bridge.write_material_files(intent, out / "materials", kind)
    g = intent.object_grounding
    _log(f"intent: fluid={intent.appearance.material} rho={intent.dynamics.liquid_rho} "
         f"viscous={intent.dynamics.liquid_viscous} object={kind}")
    _log(f"grounding: '{g.object_prompt}' size={g.object_size_m} m "
         f"({g.confidence}) -- {g.size_evidence[:80]}")
    if "reconstruct" in wanted and not (a.object_size_m or g.object_size_m):
        # The object's real size is the scene's only metric anchor; without it
        # the simulation would run in MoGe's arbitrary units.
        _log("the VLM returned no object size; pass --object_size_m explicitly")
        return 1

    if "reconstruct" in wanted:
        argv = [a.python, REPO / "fluidgen/reconstruction/run.py"] + bridge.reconstruction_argv(
            a.image, recon_dir, intent, object_prompt=a.object_prompt,
            object_size_m=a.object_size_m, fluid_prompt=a.fluid_prompt,
            material_files=mats, ignore_gate=a.ignore_gate)
        if _run(argv, cwd=str(REPO)) != 0:
            _log("reconstruction failed"); return 1
    if not scene_json.exists():
        _log(f"no scene at {scene_json}"); return 1

    if "simulate" in wanted:
        argv = [a.python, REPO / "fluidgen/simulation/rollout.py"] + bridge.rollout_argv(
            scene_json, sim_dir, intent, num_frames=a.num_frames,
            force_seed=a.force_seed, force_accel=a.force_accel)
        if _run(argv, cwd=str(REPO)) != 0:
            _log("rollout failed"); return 1

    if "conditions" in wanted:
        argv = [a.python, REPO / "fluidgen/simulation/render_conditions.py",
                "--scene", scene_json, "--sim_dir", sim_dir, "--out_dir", cond_dir]
        if _run(argv, cwd=str(REPO)) != 0:
            _log("condition rendering failed"); return 1

    result = {"scene_json": str(scene_json), "sim_dir": str(sim_dir), "cond_dir": str(cond_dir)}
    if "bundle" in wanted:
        published = bridge.publish_bundle(out, out / "bundle")
        result["bundle_dir"] = str(out / "bundle")
        result["bundle_files"] = sorted(published)
        _log(f"bundle: {len(published)} files under {out / 'bundle'}")
        _log("continue with:  python scripts/run_agent.py "
             f"--simulation_dir {out / 'bundle'} --use_simulation_fg_mask_as_sim_mask ...")
    (out / "pipeline.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
