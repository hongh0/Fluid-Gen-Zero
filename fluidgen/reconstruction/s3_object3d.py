"""Stage 3 -- complete object mesh and initial layout from SAM 3D Objects.

SAM 3D Objects takes an RGBA image (the mask lives in the alpha channel) and
returns a mesh plus the sparse-structure dict that carries its layout. We feed
it the point map from stage 2 so its geometry and ours are the same geometry,
and we ask for `with_layout_postprocess=True` so it also runs its own
silhouette optimisation -- stage 5 then only has to add the fluid-specific
part rather than solve the pose from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch

from contracts import ObjectArtifact

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fluidgen import paths  # noqa: E402

SAM3D_REPO = paths.SAM3D_REPO
PIPELINE_MOGE2 = paths.SAM3D_PIPELINE

# SAM 3D Objects returns
#   rotation    (1, 4)  quaternion, w-x-y-z (pytorch3d's convention; its own
#                       helper documents "q: ... in w,x,y,z order")
#   translation (1, 3)
#   scale       (1, 3)  per-axis, not a scalar
# and applies them as  X_cam = R @ (s * X_obj) + t.
_ROT_KEYS = ("rotation", "R", "rot", "pose_rotation", "layout_rotation")
_TRA_KEYS = ("translation", "t", "trans", "pose_translation", "layout_translation")
_SCL_KEYS = ("scale", "pose_scale", "layout_scale")


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _first_key(d: Dict[str, Any], names) -> Tuple[str, Any]:
    for n in names:
        if n in d and d[n] is not None:
            return n, d[n]
    return "", None


def extract_layout(out: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Pull (R, t, per-axis scale) out of SAM 3D Objects' output dict."""
    from conventions import R_from_quat_wxyz

    notes = []
    rk, rot = _first_key(out, _ROT_KEYS)
    tk, tra = _first_key(out, _TRA_KEYS)
    sk, scl = _first_key(out, _SCL_KEYS)

    R = np.eye(3)
    if rot is not None:
        arr = _as_numpy(rot).reshape(-1)
        if arr.size == 4:
            R = R_from_quat_wxyz(arr)
            notes.append(f"rotation from key {rk!r} (quaternion w-x-y-z)")
        elif arr.size >= 9:
            R = arr[:9].reshape(3, 3)
            notes.append(f"rotation from key {rk!r} (matrix)")
        else:
            notes.append(f"rotation key {rk!r} had {arr.size} values; using identity")
    else:
        notes.append(f"no rotation key found; available keys: {sorted(out.keys())}")

    t = np.zeros(3)
    if tra is not None:
        t = _as_numpy(tra).reshape(-1)[:3]
        notes.append(f"translation from key {tk!r}")
    else:
        notes.append("no translation key found; using origin")

    s = np.ones(3)
    if scl is not None:
        arr = _as_numpy(scl).reshape(-1)
        if arr.size >= 3:
            s = arr[:3]
        elif arr.size == 1:
            s = np.full(3, float(arr[0]))
        notes.append(f"scale from key {sk!r} = {np.round(s, 4).tolist()}")
    return R, t, s, notes


def run_object3d(image_path: str, object_mask_path: str, out_dir: str,
                 points_npy: str = "", seed: int = 42,
                 config_path: str = PIPELINE_MOGE2,
                 geometry_json: str = "",
                 mesh_postprocess: bool = None,
                 max_faces: int = 40000) -> ObjectArtifact:
    os.environ.setdefault("LIDRA_SKIP_INIT", "1")
    if mesh_postprocess is None:
        mesh_postprocess = os.environ.get("MESH_POST", "0") == "1"
    import utils3d_compat  # noqa: F401  (aliases utils3d.pt for MoGe v2)
    sys.path.insert(0, SAM3D_REPO)
    sys.path.insert(0, str(Path(SAM3D_REPO) / "notebook"))
    from inference import Inference  # noqa: E402  (repo-local module)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    crop = []
    if geometry_json and Path(geometry_json).exists():
        from contracts import GeometryArtifact
        geo = GeometryArtifact.load(geometry_json)
        crop = list(geo.crop_window)
        if geo.image_path and Path(geo.image_path).exists():
            image_path = geo.image_path      # the frame the point map describes

    bgr = cv2.imread(str(image_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE) > 127
    if crop and mask.shape != rgb.shape[:2]:
        x0, x1, y0, y1 = crop
        mask = mask[y0:y1, x0:x1]
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    pointmap, pm_notes = None, []
    if points_npy and Path(points_npy).exists():
        # compute_pointmap's user-supplied branch treats the point map as
        # (H, W, 3): it permutes to (3, H, W) to resample, then back again.
        pm = np.ascontiguousarray(np.load(points_npy)).astype(np.float32)
        bad = ~np.isfinite(pm).all(axis=-1)
        if bad.any() and (~bad).any():
            # Nearest-neighbour fill, not zeros. Zeroing looks safe and is not:
            # (0, 0, 0) is a perfectly finite 3D point at the camera centre, so
            # the layout optimiser tries to fit it and comes back with
            # "Residuals are not finite in the initial point". Copying the
            # closest real value is the same thing depth completion would do,
            # and it keeps the map continuous across the object's boundary.
            _, idx = cv2.distanceTransformWithLabels(
                bad.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(~bad)
            order = np.zeros(idx.max() + 1, np.int64)
            order[idx[~bad]] = np.arange(len(ys))
            pm = pm[ys[order[idx]], xs[order[idx]]]
            pm_notes.append(f"filled {int(bad.sum())} non-finite pointmap pixels "
                            f"({100.0 * bad.mean():.1f}% of the frame) from their "
                            "nearest finite neighbour before SAM 3D")
            if not np.isfinite(pm[mask]).all():
                pm_notes.append("warning: the object's own pointmap is partly non-finite")
        pointmap = torch.from_numpy(pm)

    inference = Inference(config_path, compile=False)
    # The notebook wrapper hard-codes with_layout_postprocess=False, and the
    # layout optimisation is exactly what we want as a pose initialiser, so go
    # to the pipeline directly (the wrapper only folds the mask into alpha).
    # merge_mask_to_rgba does `mask.astype(uint8) * 255` internally, so handing
    # it an already-scaled 0/255 mask overflows uint8 (255*255 % 256 == 1) and
    # the alpha channel reads as empty downstream. Pass the boolean mask.
    rgba = inference.merge_mask_to_rgba(rgb, mask)
    output = inference._pipeline.run(
        rgba, None, seed,
        stage1_only=False,
        with_mesh_postprocess=mesh_postprocess,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=True,
        pointmap=pointmap,
    )

    keys = sorted(k for k in output.keys())
    glb = output.get("glb", None)
    if glb is None:
        raise RuntimeError(f"SAM 3D Objects returned no mesh; keys={keys}")

    import trimesh
    geom = glb if isinstance(glb, trimesh.Trimesh) else trimesh.util.concatenate(
        [g for g in glb.geometry.values()])

    R, t, s, notes = extract_layout(output)
    notes = pm_notes + notes
    notes.append(f"sam3d output keys: {keys}")
    notes.append(f"mesh_postprocess={mesh_postprocess}")

    # SAM 3D Objects' scale is per-axis, while Genesis' Mesh morph and our own
    # refinement both work with a single factor. Bake the anisotropy into the
    # vertices so everything downstream only ever sees rotation, translation
    # and one uniform scale.
    geom = geom.copy()
    geom.vertices = np.asarray(geom.vertices, dtype=np.float64) * s.reshape(1, 3)
    # SAM 3D Objects emits ~1M faces. Registration rasterises the mesh once per
    # optimiser step and Genesis only voxelises it, so carrying that many faces
    # buys nothing and makes the fit intractable.
    if len(geom.faces) > max_faces:
        before = len(geom.faces)
        try:
            geom = geom.simplify_quadric_decimation(face_count=int(max_faces))
            notes.append(f"decimated mesh {before} -> {len(geom.faces)} faces")
        except Exception as exc:
            notes.append(f"decimation unavailable ({type(exc).__name__}); keeping {before} faces")
    if not np.allclose(s, s[0], rtol=1e-3):
        notes.append(f"baked anisotropic scale {np.round(s, 4).tolist()} into the mesh vertices")

    # Put the file's origin at the mesh's centroid, and move SAM 3D Objects'
    # translation to match. Every consumer here -- registration, assembly, the
    # rollout's vertex tracking, the PLY export -- poses the mesh *about its
    # centroid* (they all start with `v -= v.mean(0)`), but Genesis' Mesh morph
    # transforms about the file's own origin and has no notion of recentring.
    # SAM 3D does not centre its output, so without this the two conventions
    # would disagree by `R @ (scale * mean)`.
    centroid = np.asarray(geom.vertices, dtype=np.float64).mean(axis=0)
    if np.linalg.norm(centroid) > 1e-9:
        geom.vertices = np.asarray(geom.vertices, dtype=np.float64) - centroid
        t = np.asarray(t, dtype=np.float64) + R @ centroid
        notes.append(f"recentred the mesh on its centroid {np.round(centroid, 4).tolist()} and "
                     f"folded that into the translation; Genesis poses a Mesh morph about the "
                     f"file origin, not the centroid")

    mesh_path = out / "object.glb"
    geom.export(str(mesh_path))
    verts, faces = np.asarray(geom.vertices), np.asarray(geom.faces)
    if len(verts) == 0:
        # Say so here rather than let `verts.min(axis=0)` below raise
        # "zero-size array to reduction operation minimum", which names numpy
        # and points at nothing.
        raise RuntimeError("SAM 3D Objects returned a mesh with no vertices; "
                           f"notes={notes}")

    if "pointmap" in output:
        np.save(out / "sam3d_pointmap.npy", _as_numpy(output["pointmap"]).astype(np.float32))

    art = ObjectArtifact(
        mesh_path=str(mesh_path), scale=1.0,   # anisotropy already in the mesh
        rotation=R.tolist(), translation=[float(x) for x in t],
        num_vertices=int(len(verts)), num_faces=int(len(faces)),
        bbox_min=verts.min(axis=0).tolist(), bbox_max=verts.max(axis=0).tolist(),
        notes=notes)
    art.save(out / "object3d.json")
    return art


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--object_mask", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--points", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=PIPELINE_MOGE2)
    p.add_argument("--geometry", default="")
    p.add_argument("--max_faces", type=int, default=40000)
    a = p.parse_args()
    art = run_object3d(a.image, a.object_mask, a.out_dir, a.points, a.seed, a.config,
                       geometry_json=a.geometry, max_faces=a.max_faces)
    print(json.dumps({"mesh": art.mesh_path, "verts": art.num_vertices, "faces": art.num_faces,
                      "scale": art.scale, "bbox_extent":
                      (np.array(art.bbox_max) - np.array(art.bbox_min)).round(4).tolist(),
                      "notes": art.notes}, indent=2))


if __name__ == "__main__":
    main()
