#!/bin/bash
# Fluid-Gen-Zero on WanMove for the samples listed in INPUT_JSON
# (built by scripts/make_wanmove_input.py from the VACE clip plans).
#
#   bash scripts/run_wanmove.sh INPUT_JSON OUTPUT_DIR
#
# Runs in the WanMove conda env (FGZ_WAN_ENV) on the visible GPU.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IN=$(realpath "$1"); OUT=$(realpath -m "$2")
mkdir -p "$OUT"
P() { (cd "$REPO" && python -c "from fluidgen import paths; print(paths.$1)"); }
WAN_ENV=${FGZ_WAN_ENV:-wan}
run_in() { local env=$1; shift; if [ -n "$env" ]; then conda run -n "$env" --no-capture-output "$@"; else "$@"; fi; }
cd "$(P WAN_MOVE_DIR)"
run_in "$WAN_ENV" python generate_multiclip.py \
  --input_json "$IN" --output_dir "$OUT" --output_json_path "$OUT/generated.json" \
  --task wan-move-i2v --size "480*832" --ckpt_dir "$(P WAN_MOVE_CKPT)" \
  --clip_frame_num 81 --base_seed 2025 --resume --latent_guidance
