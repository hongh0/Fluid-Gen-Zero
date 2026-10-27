"""Stage 2 -- monocular geometry with MoGe v2.

SAM 3D Objects bundles its own MoGe (v1) and does not return the intrinsics it
used, so we run MoGe v2 here, keep ownership of K, and hand the resulting
point map to SAM 3D Objects. Both stages then describe the same geometry, and
the camera we later hand to Genesis is the one the reconstruction was built
on rather than a second, independently guessed one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from contracts import GeometryArtifact
from conventions import Camera, principal_point_crop_window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fluidgen import paths  # noqa: E402

MOGE_V2 = paths.MOGE2_CKPT


def _bbox_fill(valid: np.ndarray) -> float:
    """How solidly the valid pixels fill their own bounding box.
    """
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return 0.0
    y0, y1 = np.percentile(ys, [1, 99])
    x0, x1 = np.percentile(xs, [1, 99])
    inside = ((ys >= y0) & (ys <= y1) & (xs >= x0) & (xs <= x1)).sum()
    box = (y1 - y0 + 1) * (x1 - x0 + 1)
    return float(inside / max(box, 1.0))


def run_geometry(image_path: str, out_dir: str, device: str = "cuda",
                 recenter_principal_point: bool = True) -> GeometryArtifact:
    import utils3d_compat  # noqa: F401  (aliases utils3d.pt for MoGe v2)
    from moge.model.v2 import MoGeModel

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crop_window = []
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    model = MoGeModel.from_pretrained(MOGE_V2).to(device).eval()
    tensor = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        # force_projection=False is what SAM 3D Objects' own MoGe wrapper uses,
        # so our point map matches the one it consumes. It also avoids
        # `utils3d.pt.depth_map_to_point_map`, which the utils3d revision this
        # environment is pinned to does not have (see utils3d_compat).
        pred = model.infer(tensor, force_projection=False)

    notes = [f"moge v2 output keys: {sorted(pred.keys())}"]
    points = pred["points"].detach().cpu().numpy().astype(np.float32)
    depth = pred["depth"].detach().cpu().numpy().astype(np.float32)
    K = pred["intrinsics"].detach().cpu().numpy().astype(np.float64).reshape(3, 3)
    valid = (pred["mask"].detach().cpu().numpy().astype(bool)
             if "mask" in pred else np.isfinite(depth))
    # MoGe reports intrinsics normalised by image size
    h, w = depth.shape
    if K[0, 0] < 5.0:
        K = K.copy()
        K[0, :] *= w
        K[1, :] *= h
        notes.append("rescaled normalised intrinsics to pixels")
    metric = bool(pred.get("metric_scale") is not None) if hasattr(pred, "get") else False

    if recenter_principal_point:
        # Genesis' camera has no principal-point parameter, so crop once here
        # and let every later stage work in the frame Genesis can reproduce.
        x0, x1, y0, y1 = principal_point_crop_window(K, w, h)
        crop_window = [int(x0), int(x1), int(y0), int(y1)]
        points, depth, valid = points[y0:y1, x0:x1], depth[y0:y1, x0:x1], valid[y0:y1, x0:x1]
        rgb = rgb[y0:y1, x0:x1]
        K = K.copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        cv2.imwrite(str(out / "image_centered.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        notes.append(f"cropped {w}x{h} -> {rgb.shape[1]}x{rgb.shape[0]} to centre the principal point "
                     f"(was offset by {x0 + (x1 - x0) / 2 - w / 2:.1f}, {y0 + (y1 - y0) / 2 - h / 2:.1f} px)")

    h, w = depth.shape
    np.save(out / "points.npy", points)
    np.save(out / "depth.npy", depth)
    np.save(out / "valid_mask.npy", valid)

    finite = depth[valid & np.isfinite(depth)]
    art = GeometryArtifact(
        points_path=str(out / "points.npy"),
        depth_path=str(out / "depth.npy"),
        valid_mask_path=str(out / "valid_mask.npy"),
        K=K.tolist(), width=int(w), height=int(h),
        valid_ratio=float(valid.mean()),
        valid_bbox_fill=_bbox_fill(valid),
        depth_min=float(np.percentile(finite, 1)) if finite.size else 0.0,
        depth_max=float(np.percentile(finite, 99)) if finite.size else 0.0,
        metric=metric, notes=notes,
        image_path=str(out / 'image_centered.png') if crop_window else str(image_path),
        crop_window=crop_window)
    art.save(out / "geometry.json")
    Camera(K, w, h).to_dict()
    return art


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep_principal_point", action="store_true")
    a = p.parse_args()
    art = run_geometry(a.image, a.out_dir, a.device, not a.keep_principal_point)
    cam = Camera(np.array(art.K), art.width, art.height)
    print(json.dumps({"size": [art.width, art.height], "fov_y_deg": round(cam.fov_y_deg, 2),
                      "valid_ratio": round(art.valid_ratio, 3),
                      "depth_p1_p99": [round(art.depth_min, 3), round(art.depth_max, 3)],
                      "metric": art.metric, "notes": art.notes}, indent=2))


if __name__ == "__main__":
    main()
