"""Every external path Fluid-Gen-Zero needs, in one place.

Nothing here is hard-coded to a machine. Each entry reads an environment
variable and otherwise falls back to a location under the repository, so the
default layout after following the README is:

    Fluid-Gen-Zero/
      checkpoints/     perception / reconstruction / inpainting weights (FGZ_CKPT_DIR)
      third_party/     patched backbone checkouts (FGZ_THIRD_PARTY); each backbone
                       keeps its weights where its own README puts them

Override any single entry with its own variable, e.g.
``export FGZ_WAN_MOVE_CKPT=/data/Wan-Move-14B-480P``. ``env.example.sh`` lists
them all. This module is stdlib-only so it can be imported from every conda
environment the pipeline runs in.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: Path | str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else str(default)


CKPT_DIR = Path(_env("FGZ_CKPT_DIR", REPO_ROOT / "checkpoints"))
THIRD_PARTY = Path(_env("FGZ_THIRD_PARTY", REPO_ROOT / "third_party"))

# ---- perception / reconstruction ---------------------------------------
SAM2_CKPT = _env("FGZ_SAM2_CKPT", CKPT_DIR / "sam2.1_hiera_large.pt")
SAM2_CFG = _env("FGZ_SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
GDINO_CKPT = _env("FGZ_GDINO_CKPT", CKPT_DIR / "grounding-dino-base")
# Fallback detectors, used only when Grounding DINO + SAM2 return no plausible mask.
GDINO_SWINT_CFG = _env("FGZ_GDINO_SWINT_CFG", CKPT_DIR / "groundingdino" / "GroundingDINO_SwinT_OGC.py")
GDINO_SWINT_CKPT = _env("FGZ_GDINO_SWINT_CKPT", CKPT_DIR / "groundingdino" / "groundingdino_swint_ogc.pth")
OWLV2_CKPT = _env("FGZ_OWLV2_CKPT", CKPT_DIR / "owlv2-base-patch16-ensemble")
CLIPSEG_CKPT = _env("FGZ_CLIPSEG_CKPT", CKPT_DIR / "clipseg-rd64-refined")
MOGE2_CKPT = _env("FGZ_MOGE2_CKPT", CKPT_DIR / "moge-2-vitl" / "model.pt")
SAM3D_REPO = _env("FGZ_SAM3D_REPO", THIRD_PARTY / "sam-3d-objects")
SAM3D_PIPELINE = _env("FGZ_SAM3D_PIPELINE", CKPT_DIR / "sam-3d-objects" / "pipeline.yaml")

# ---- warped reference (FLUX ControlNet inpainting) ----------------------
FLUX_DIR = _env("FGZ_FLUX_DIR", CKPT_DIR / "FLUX.1-dev")
FLUX_INPAINT_CONTROLNET = _env(
    "FGZ_FLUX_INPAINT_CONTROLNET", CKPT_DIR / "FLUX.1-dev-Controlnet-Inpainting-Beta")

# ---- video backbones -----------------------------------------------------
VACE_DIR = _env("FGZ_VACE_DIR", THIRD_PARTY / "VACE")
VACE_CKPT = _env("FGZ_VACE_CKPT", Path(VACE_DIR) / "models" / "Wan2.1-VACE-1.3B")
TORA_DIR = _env("FGZ_TORA_DIR", THIRD_PARTY / "Tora")
# Tora resolves its T5 / VAE paths relative to sat/ckpts (upstream layout), so the
# i2v transformer lives there too.
TORA_CKPT = _env("FGZ_TORA_CKPT", Path(TORA_DIR) / "sat" / "ckpts" / "tora" / "i2v")
WAN_MOVE_DIR = _env("FGZ_WAN_MOVE_DIR", THIRD_PARTY / "Wan-Move")
WAN_MOVE_CKPT = _env("FGZ_WAN_MOVE_CKPT", Path(WAN_MOVE_DIR) / "Wan-Move-14B-480P")

# Conda environments the backbones run in. Empty string = current interpreter.
WAN_CONDA_ENV = os.environ.get("FGZ_WAN_ENV", "wan")
TORA_CONDA_ENV = os.environ.get("FGZ_TORA_ENV", "tora")
