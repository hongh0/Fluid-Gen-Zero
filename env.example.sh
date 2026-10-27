# Copy to env.sh, edit, and `source env.sh` before running anything.
# Every variable is optional: unset ones fall back to the defaults in
# fluidgen/paths.py (checkpoints/ and third_party/ inside this repository).

# ---- where things live ----------------------------------------------------
export FGZ_CKPT_DIR=$PWD/checkpoints          # perception / reconstruction / inpainting weights
export FGZ_THIRD_PARTY=$PWD/third_party       # VACE, Tora, Wan-Move, sam-3d-objects checkouts
export FGZ_RUNS=$PWD/runs                     # outputs of scripts/run_sample.sh
export FGZ_BENCHMARK=/path/to/FluidGen_Benchmark

# ---- python / conda ---------------------------------------------------------
export FGZ_PY=python        # interpreter of the main `fluidgen` env (reconstruction, simulation, planning)
export FGZ_WAN_ENV=wan      # conda env for VACE and Wan-Move ("" = use FGZ_PY)
export FGZ_TORA_ENV=tora    # conda env for Tora              ("" = use FGZ_PY)

# ---- VLM (intent parsing, risk annotation, clip-level prompts) ------------
export OPENAI_API_KEY=sk-...
# export FGZ_VLM_MODEL=gpt-5.4-mini-2026-03-17

# ---- individual overrides (defaults shown) --------------------------------
# export FGZ_SAM2_CKPT=$FGZ_CKPT_DIR/sam2.1_hiera_large.pt
# export FGZ_GDINO_CKPT=$FGZ_CKPT_DIR/grounding-dino-base
# export FGZ_MOGE2_CKPT=$FGZ_CKPT_DIR/moge-2-vitl/model.pt
# export FGZ_SAM3D_REPO=$FGZ_THIRD_PARTY/sam-3d-objects
# export FGZ_SAM3D_PIPELINE=$FGZ_CKPT_DIR/sam-3d-objects/pipeline.yaml
# export FGZ_FLUX_DIR=$FGZ_CKPT_DIR/FLUX.1-dev
# export FGZ_FLUX_INPAINT_CONTROLNET=$FGZ_CKPT_DIR/FLUX.1-dev-Controlnet-Inpainting-Beta
# export FGZ_OWLV2_CKPT=$FGZ_CKPT_DIR/owlv2-base-patch16-ensemble          # segmentation fallback
# export FGZ_CLIPSEG_CKPT=$FGZ_CKPT_DIR/clipseg-rd64-refined               # segmentation fallback
# export FGZ_GDINO_SWINT_CFG=$FGZ_CKPT_DIR/groundingdino/GroundingDINO_SwinT_OGC.py
# export FGZ_GDINO_SWINT_CKPT=$FGZ_CKPT_DIR/groundingdino/groundingdino_swint_ogc.pth
# export FGZ_VACE_CKPT=$FGZ_THIRD_PARTY/VACE/models/Wan2.1-VACE-1.3B
# export FGZ_TORA_CKPT=$FGZ_THIRD_PARTY/Tora/sat/ckpts/tora/i2v
# export FGZ_WAN_MOVE_CKPT=$FGZ_THIRD_PARTY/Wan-Move/Wan-Move-14B-480P

# SAM 3D Objects: skip its init hooks and the optional nvdiffrast hole filling.
export LIDRA_SKIP_INIT=1 MESH_POST=0
# FLUX inpainting keeps ~36 GB on the GPU; set to 1 on smaller cards.
export FLUX_CPU_OFFLOAD=0
