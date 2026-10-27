#!/bin/bash
# Fluid-Gen-Zero on one image + prompt, end to end.
#
#   bash scripts/run_sample.sh SAMPLE_ID IMAGE "PROMPT" [BACKENDS]
#
#   BACKENDS  space-separated subset of "wanmove vace tora" (default: "wanmove")
#
# Writes everything under $FGZ_RUNS/SAMPLE_ID (default runs/SAMPLE_ID):
#   s1_6_recon/ s8_sim/ s9_cond/   reconstruction, simulation rollout, projected conditions
#   bundle/                        simulation bundle shared by every backbone
#   vace/  tora/                   agent_manifest.json, per-clip outputs, out_video_stitched.mp4
#   wanmove/                       WanMove run on the VACE clip plan
#
# Needs OPENAI_API_KEY: the VLM parses the intent, annotates risk intervals and
# writes the clip-level prompts. FGZ_NUM_FRAMES sets the rollout length (default 401).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SID=$1; IMG=$(realpath "$2"); PROMPT=$3; BACKENDS=${4:-"wanmove"}
PY=${FGZ_PY:-python}
OUT=$(realpath -m "${FGZ_RUNS:-$REPO/runs}/$SID")
mkdir -p "$OUT"
printf '{"sample_id": %s, "image": %s, "prompt": %s}\n' \
  "$($PY -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$SID")" \
  "$($PY -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$IMG")" \
  "$($PY -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")" > "$OUT/sample.json"
cd "$REPO"

# 1) Scene reconstruction + physics simulation -> simulation bundle (Sec. 3.1)
if [ ! -f "$OUT/bundle/fg_traj.npz" ]; then
  $PY -m fluidgen.pipeline --image "$IMG" --prompt "$PROMPT" --out_dir "$OUT" \
      --num_frames "${FGZ_NUM_FRAMES:-401}" --force_accel 5.0 --force_seed 0
fi

# 2) Generation-time planning (Sec. 3.2) + 3) latent-space guidance (Sec. 3.3)
for BK in vace tora; do
  [[ " $BACKENDS " == *" $BK "* || ( $BK == vace && " $BACKENDS " == *" wanmove "* ) ]] || continue
  [ -s "$OUT/$BK/out_video_stitched.mp4" ] && continue
  if [ ! -f "$OUT/$BK/agent_manifest.json" ]; then
    $PY scripts/run_agent.py --prompt "$PROMPT" --image "$IMG" \
        --output_dir "$OUT/$BK" --generation_backend $BK \
        --simulation_dir "$OUT/bundle" --use_simulation_fg_mask_as_sim_mask \
        --api_frame_interval 10
  fi
  [[ " $BACKENDS " == *" $BK "* ]] || continue
  $PY scripts/run_clips.py --manifest "$OUT/$BK/agent_manifest.json"
done

# WanMove reuses the VACE clip plan and clip-level prompts.
if [[ " $BACKENDS " == *" wanmove "* ]]; then
  $PY scripts/make_wanmove_input.py --runs "$OUT" --out "$OUT/wanmove"
  bash scripts/run_wanmove.sh "$OUT/wanmove/input.json" "$OUT/wanmove/gen"
fi
echo "done: $OUT"
