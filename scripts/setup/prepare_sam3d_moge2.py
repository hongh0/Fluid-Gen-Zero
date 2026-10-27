#!/usr/bin/env python3
"""Point SAM 3D Objects at MoGe-2 instead of MoGe-1.

SAM 3D Objects ships `checkpoints/pipeline.yaml` with a MoGe-1 depth model.
Fluid-Gen-Zero uses MoGe-2 everywhere, so that the object pose and the scene
point map come from the same geometry. This writes FGZ_CKPT_DIR/sam-3d-objects/
with symlinks to every upstream checkpoint and a pipeline.yaml whose depth model
is MoGe-2 at FGZ_MOGE2_CKPT.

    python scripts/setup/prepare_sam3d_moge2.py /path/to/sam-3d-objects/checkpoints
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fluidgen import paths  # noqa: E402

src = Path(sys.argv[1]).resolve()
dst = Path(paths.SAM3D_PIPELINE).parent
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "pipeline.yaml" and not (dst / f.name).exists():
        (dst / f.name).symlink_to(f)

text = (src / "pipeline.yaml").read_text()
lines, n = [], 0
for line in text.splitlines():
    if "moge.model.v1.MoGeModel.from_pretrained" in line:
        line = line.replace("moge.model.v1", "moge.model.v2"); n += 1
    elif line.strip().startswith("pretrained_model_name_or_path:") and "moge" in line.lower():
        indent = line[: len(line) - len(line.lstrip())]
        line = f"{indent}pretrained_model_name_or_path: {paths.MOGE2_CKPT}"; n += 1
    lines.append(line)
if n != 2:
    raise SystemExit(f"expected to patch 2 lines in {src / 'pipeline.yaml'}, patched {n}")
Path(paths.SAM3D_PIPELINE).write_text("\n".join(lines) + "\n")
print("wrote", paths.SAM3D_PIPELINE)
