#!/usr/bin/env python3
"""Build the WanMove input from finished VACE runs.

    python scripts/make_wanmove_input.py --runs runs/apple --out runs/apple/wanmove

WanMove reuses the clip plan made for VACE (boundaries and clip-level prompts)
and the same simulation bundle. Each run directory must contain sample.json,
bundle/ and vace/agent_manifest.json (written by scripts/run_sample.sh). One
manifest per sample goes to OUT/manifests/ and the list to OUT/input.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories (runs/<sample_id>)")
    p.add_argument("--plan_from", choices=["vace", "tora"], default="vace")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out).resolve()
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    items = []
    for run in sorted(Path(r).resolve() for r in a.runs):
        meta_p, b = run / "sample.json", run / "bundle"
        if not meta_p.exists() or not b.is_dir():
            print(f"skip {run}: no sample.json or bundle/")
            continue
        meta = json.loads(meta_p.read_text())
        man = {
            "sample_id": meta["sample_id"],
            "simulation_bundle": {
                "vace_src_video_path": str(b / "vace_condition/src_video-firstframe_ctrl.mp4"),
                "vace_src_mask_path": str(b / "vace_condition/src_mask-firstframe_ctrl.mp4"),
                "warped_reference_video_path": str(b / "warped_reference/motion_signal.mp4"),
                "warped_reference_mask_path": str(b / "warped_reference/mask.mp4"),
                "fg_mask_video_path": str(b / "segmap/fg_mask.mp4"),
                "fluid_mask_video_path": str(b / "segmap/fluid_mask.mp4"),
            },
        }
        agent = json.loads((run / a.plan_from / "agent_manifest.json").read_text())
        clips = (agent.get("complexity_plan") or {}).get("clips") or []
        man["clip_plan"] = [{
            "clip_id": i,
            "start_frame": c["start_frame"],
            "end_frame": c["end_frame"],
            "actual_frame_num": c["end_frame"] - c["start_frame"] + 1,
            "frame_num": c["end_frame"] - c["start_frame"] + 1,
            "prompt": c.get("prompt") or meta["prompt"],
        } for i, c in enumerate(clips)]
        man_p = out / "manifests" / f"{meta['sample_id']}.json"
        man_p.write_text(json.dumps(man, indent=2) + "\n")
        items.append({"video_path": meta["image"], "caption": meta["prompt"],
                      "agent_manifest_path": str(man_p), "sample_id": meta["sample_id"]})
    (out / "input.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(items)} samples -> {out / 'input.json'}")


if __name__ == "__main__":
    main()
