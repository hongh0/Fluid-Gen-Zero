#!/usr/bin/env python3
"""Generate every planned clip of one manifest and stitch them.

    python scripts/run_clips.py --manifest runs/apple/vace/agent_manifest.json

The backbone (VACE or Tora) is read from the manifest. The model is loaded once
and all clips are generated in one process, inside the backbone's own conda
environment (FGZ_WAN_ENV / FGZ_TORA_ENV). The result is written next to the
manifest as `out_video_stitched.mp4`, with the shared anchor frame between
consecutive clips removed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fluidgen.agent import clip_executor, tora_clip_executor  # noqa: E402
from fluidgen.tools.stitch import stitch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="agent_manifest.json written by scripts/run_agent.py")
    p.add_argument("--dry_run", action="store_true", help="Print the backbone command without running it.")
    p.add_argument("--run_preprocess", action="store_true",
                   help="VACE only: build the native VACE condition first if the manifest only planned it.")
    p.add_argument("--no_previous_anchor", action="store_true",
                   help="Ablation: do not anchor clip i+1 on the last generated frame of clip i.")
    p.add_argument("--output", default=None, help="Stitched video path (default: <manifest dir>/out_video_stitched.mp4).")
    p.add_argument("--fps", type=float, default=16.0, help="FPS of the stitched video.")
    a = p.parse_args()

    manifest = Path(a.manifest).resolve()
    backend = json.loads(manifest.read_text()).get("generation_backend", "vace")
    argv = ["--manifest", str(manifest)]
    if a.dry_run:
        argv.append("--dry_run")
    if a.no_previous_anchor:
        argv.append("--no_previous_anchor")
    if backend == "tora":
        tora_clip_executor.main(argv)
    else:
        if a.run_preprocess:
            argv.append("--run_preprocess")
        clip_executor.main(argv)

    if not a.dry_run:
        out = a.output or str(manifest.parent / "out_video_stitched.mp4")
        stitch(str(manifest), out, a.fps)


if __name__ == "__main__":
    main()
