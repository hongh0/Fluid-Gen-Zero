#!/usr/bin/env python3
"""Generation-time planning: simulation bundle -> agent_manifest.json.

    python scripts/run_agent.py --prompt "..." --image examples/apple.png \
        --simulation_dir runs/apple/bundle --output_dir runs/apple/vace \
        --generation_backend vace --use_simulation_fg_mask_as_sim_mask \
        --run_vlm_judge_api --run_warped_reference --run_vace_preprocess

See `python scripts/run_agent.py --help` for every option.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fluidgen.agent.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
