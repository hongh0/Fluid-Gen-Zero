#!/bin/bash
# The 43-sample FluidGen Benchmark, sharded over ranks.
#
#   bash scripts/run_benchmark.sh [BACKENDS]          # one process, all samples
#   RANK=0 NRANK=4 bash scripts/run_benchmark.sh ...  # sample k runs on rank k % NRANK
#
# Under SLURM, RANK/NRANK default to SLURM_PROCID/SLURM_NTASKS, so
#   srun -n 8 --gpus-per-task=1 bash scripts/run_benchmark.sh "vace tora wanmove"
# is enough.
#
# By default each sample starts from the simulation released with the benchmark
# (FGZ_BENCHMARK/samples/<id>/simulation): the dense optical flow is re-rendered
# from the particle trajectories, then planning and generation run as usual.
# FROM_SCRATCH=1 reruns reconstruction and simulation from the image instead.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
BENCH=$(realpath "${FGZ_BENCHMARK:?set FGZ_BENCHMARK to the FluidGen_Benchmark directory}")
BACKENDS=${1:-"wanmove"}
RANK=${RANK:-${SLURM_PROCID:-0}}
NRANK=${NRANK:-${SLURM_NTASKS:-1}}
PY=${FGZ_PY:-python}
RUNS=$(realpath -m "${FGZ_RUNS:-$REPO/runs}")
mapfile -t ROWS < <($PY - "$BENCH/benchmark.json" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1])):
    print("\t".join([e["sample_id"], e["prompt"].replace("\n", " ")]))
PY
)
for k in "${!ROWS[@]}"; do
  [ $((k % NRANK)) -eq "$RANK" ] || continue
  IFS=$'\t' read -r SID PROMPT <<< "${ROWS[$k]}"
  SAMPLE=$BENCH/samples/$SID
  OUT=$RUNS/$SID
  echo "[rank $RANK] $SID"
  if [ "${FROM_SCRATCH:-0}" != 1 ] && [ ! -f "$OUT/bundle/optical_flow_flow.npy" ]; then
    # real directories with symlinked files: anything the planner writes into the
    # bundle (e.g. vace_condition/) lands in the run, never in the benchmark
    mkdir -p "$OUT/bundle"
    cp -rs "$SAMPLE/simulation/." "$OUT/bundle/"
    $PY "$REPO/fluidgen/simulation/render_conditions.py" --scene "$SAMPLE/scene/scene.json" \
        --sim_dir "$SAMPLE/simulation" --out_dir "$OUT/s9_cond" > "$OUT/render_conditions.log" 2>&1 \
      && ln -sf "$OUT/s9_cond/optical_flow.npy" "$OUT/bundle/optical_flow_flow.npy" \
      || { echo "[rank $RANK] $SID flow re-render FAILED (see $OUT/render_conditions.log)"; continue; }
  fi
  bash "$REPO/scripts/run_sample.sh" "$SID" "$SAMPLE/image.png" "$PROMPT" "$BACKENDS" \
    || echo "[rank $RANK] $SID FAILED"
done
