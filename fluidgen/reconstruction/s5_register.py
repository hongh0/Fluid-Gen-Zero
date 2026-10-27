"""Stage 5 -- refine the object pose against the fluid surface.

This is the one piece that is specific to object-fluid scenes. A generic
silhouette fit compares the whole rendered mesh to the object mask, but in
these images the liquid *hides* the submerged part of the object: the mask
stops at the waterline. So we clip the mesh by the fluid plane and compare
only what should be visible. That single change subsumes both extra terms the
usual formulation needs (waterline alignment and water-occlusion), because the
lower edge of the clipped silhouette *is* the waterline.

    L = (1 - IoU(render(clip(mesh, z>0)), object_mask)) + w_d * depth_residual

Absolute scale comes from the object's real-world size (the LLM parses it from
the prompt); MoGe is only defined up to scale, and guessing metric depth from
a monocular network is less reliable than knowing an apple is ~8 cm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from contracts import (FluidArtifact, GeometryArtifact, ObjectArtifact,
                       RegistrationArtifact, SegmentationArtifact)
from conventions import Camera, invert_rigid, quat_wxyz_from_R, transform_points


# --------------------------------------------------------------------------
def _load_mesh(path: str):
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _clip_below_plane(verts_world: np.ndarray, faces: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the faces that are entirely under the fluid surface (z < 0).

    A face straddling the surface is kept: its visible part dominates and the
    resulting silhouette edge is what we want to align with the waterline.
    """
    if len(faces) == 0:
        return verts_world, faces
    z = verts_world[faces][:, :, 2]
    keep = (z > 0.0).any(axis=1)
    return verts_world, faces[keep]


def _rasterize_mask(verts_cam: np.ndarray, faces: np.ndarray, cam: Camera) -> np.ndarray:
    """Painter's-algorithm silhouette. No shading needed, just coverage."""
    mask = np.zeros((cam.height, cam.width), np.uint8)
    if len(faces) == 0:
        return mask.astype(bool)
    in_front = verts_cam[:, 2] > 1e-6
    uv = cam.project(verts_cam)
    tri = uv[faces]
    ok = in_front[faces].all(axis=1)
    tri = tri[ok]
    if len(tri) == 0:
        return mask.astype(bool)
    polys = [t.round().astype(np.int32) for t in tri]
    cv2.fillPoly(mask, polys, 1)
    return mask.astype(bool)


def _boundary_error(target: np.ndarray, render: np.ndarray) -> float:
    """Symmetric outline distance, as a fraction of the object's size.
    """
    if not target.any() or not render.any():
        return float("inf")

    def edge(m):
        return (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~m)

    et, er = edge(target), edge(render)
    if not et.any() or not er.any():
        return float("inf")
    dt = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3)
    dr = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(target)
    diag = float(np.hypot(np.ptp(ys) + 1, np.ptp(xs) + 1))
    return float(0.5 * (dr[et].mean() + dt[er].mean()) / max(diag, 1e-6))


def _rasterize_depth(verts_cam: np.ndarray, faces: np.ndarray, cam: Camera) -> np.ndarray:
    """Nearest-surface depth via per-face z, painted back to front."""
    depth = np.full((cam.height, cam.width), np.inf, np.float32)
    if len(faces) == 0:
        return depth
    uv = cam.project(verts_cam)
    zs = verts_cam[faces][:, :, 2].mean(axis=1)
    order = np.argsort(-zs)                      # far first, near overwrites
    for fi in order:
        f = faces[fi]
        if (verts_cam[f, 2] <= 1e-6).any():
            continue
        poly = uv[f].round().astype(np.int32)
        cv2.fillPoly(depth, [poly], float(zs[fi]))
    return depth


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


# --------------------------------------------------------------------------
def register(object_json: str, fluid_json: str, geometry_json: str,
             segmentation_json: str, out_dir: str,
             object_size_m: Optional[float] = None,
             depth_weight: float = 0.5, iters: int = 200,
             seed: int = 0) -> RegistrationArtifact:
    obj = ObjectArtifact.load(object_json)
    flu = FluidArtifact.load(fluid_json)
    geo = GeometryArtifact.load(geometry_json)
    seg = SegmentationArtifact.load(segmentation_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    notes = []

    T_world_cam = np.asarray(flu.T_world_cam)
    K = np.asarray(geo.K)
    T_cam_world = invert_rigid(T_world_cam)      # world -> camera
    cam_proj = Camera(K, geo.width, geo.height)  # projection happens in camera coords

    target = cv2.imread(seg.object_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if target.shape != (geo.height, geo.width):
        target = cv2.resize(target.astype(np.uint8), (geo.width, geo.height),
                            interpolation=cv2.INTER_NEAREST) > 0
    depth_obs = np.load(geo.depth_path)
    valid = np.load(geo.valid_mask_path) & target & np.isfinite(depth_obs)

    verts_obj, faces = _load_mesh(obj.mesh_path)
    verts_obj = verts_obj - verts_obj.mean(axis=0)

    # SAM 3D Objects' layout is the starting point; convert it into the world
    R0 = np.asarray(obj.rotation)
    t0_cam = np.asarray(obj.translation, dtype=np.float64)
    s0 = float(obj.scale)
    if object_size_m and object_size_m > 0:
        extent = float(np.max(verts_obj.max(axis=0) - verts_obj.min(axis=0)))
        if extent > 1e-9:
            # the prompt's "an apple is ~8 cm" means its largest dimension, not
            # the diagonal of its bounding box
            s0 = object_size_m / extent
            notes.append(f"scale from the object's real size ({object_size_m} m / "
                         f"largest mesh extent {extent:.4f}) -> {s0:.4f}")
    R_w = T_world_cam[:3, :3] @ R0
    t_w = transform_points(T_world_cam, t0_cam.reshape(1, 3))[0]

    def render_masks(R, t, s):
        vw = (verts_obj * s) @ R.T + t                        # world
        _, f_vis = _clip_below_plane(vw, faces)
        vc = transform_points(T_cam_world, vw)                # world -> camera
        return _rasterize_mask(vc, f_vis, cam_proj), vc, f_vis

    obs_depth_median = float(np.median(depth_obs[valid])) if valid.any() else 0.0

    def cost(R, t, s):
        """Silhouette IoU plus a vertex-level depth proxy.

        Rasterising a depth buffer per step would mean one fillPoly per face
        per iteration; the proxy compares the median depth of the visible
        vertices instead, which is O(V) and captures the along-axis error the
        silhouette alone cannot see. The real depth residual is measured once,
        after the search.
        """
        m, vc, f_vis = render_masks(R, t, s)
        iou = _iou(m, target)
        loss = 1.0 - iou
        if depth_weight > 0 and obs_depth_median > 0 and len(f_vis):
            zs = vc[np.unique(f_vis)][:, 2]
            zs = zs[np.isfinite(zs) & (zs > 0)]
            if zs.size:
                loss += depth_weight * abs(float(np.median(zs)) - obs_depth_median) / obs_depth_median
        return loss, iou, m

    best = cost(R_w, t_w, s0)
    best_params = (R_w.copy(), t_w.copy(), s0)
    notes.append(f"initial IoU {best[1]:.3f}")

    # random-restart local search: cheap, derivative-free and robust to the
    # local minima a silhouette objective is full of
    step_t = 0.15 * max(1e-3, float(np.linalg.norm(verts_obj.max(axis=0) - verts_obj.min(axis=0))) * s0)
    step_r, step_s = 0.25, 0.12
    for it in range(iters):
        decay = 1.0 - it / max(1, iters)
        R, t, s = best_params
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.normal() * step_r * decay
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
        cand_R = dR @ R
        cand_t = t + rng.normal(size=3) * step_t * decay
        cand_s = float(np.clip(s * (1.0 + rng.normal() * step_s * decay), 1e-4, 1e4))
        c = cost(cand_R, cand_t, cand_s)
        if c[0] < best[0]:
            best, best_params = c, (cand_R, cand_t, cand_s)

    R_f, t_f, s_f = best_params
    final_mask = best[2]
    iou = best[1]

    # interaction state falls out of the geometry; no separate classifier
    vw = (verts_obj * s_f) @ R_f.T + t_f
    zmin, zmax = float(vw[:, 2].min()), float(vw[:, 2].max())
    submerged = float((vw[:, 2] < 0).mean())
    if zmin > 0.01:
        state = "airborne"
    elif zmax < 0.0:
        state = "submerged"
    elif submerged > 0.02:
        state = "floating"
    else:
        state = "touching"

    d = _rasterize_depth(transform_points(T_cam_world, vw),
                         _clip_below_plane(vw, faces)[1], cam_proj)
    both = valid & np.isfinite(d)
    depth_err = float(np.median(np.abs(d[both] - depth_obs[both]))) if both.sum() > 50 else float("nan")

    cv2.imwrite(str(out / "registered_silhouette.png"),
                np.dstack([final_mask.astype(np.uint8) * 255,
                           target.astype(np.uint8) * 255,
                           np.zeros_like(final_mask, np.uint8)]))

    art = RegistrationArtifact(
        scale=float(s_f), rotation=R_f.tolist(), translation=[float(x) for x in t_f],
        quat_wxyz=quat_wxyz_from_R(R_f).tolist(),
        mask_iou=float(iou), boundary_error=_boundary_error(target, final_mask),
        depth_median_error=depth_err,
        interaction_state=state, signed_distance_to_plane=zmin,
        submerged_fraction=submerged, notes=notes)
    art.evaluate_gate()
    art.save(out / "registration.json")
    return art


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True)
    p.add_argument("--fluid", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--segmentation", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--object_size_m", type=float, default=None)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()
    art = register(a.object, a.fluid, a.geometry, a.segmentation, a.out_dir,
                   a.object_size_m, iters=a.iters)
    print(json.dumps({"iou": round(art.mask_iou, 4), "passed": art.passed,
                      "depth_median_error": art.depth_median_error,
                      "state": art.interaction_state,
                      "submerged_fraction": round(art.submerged_fraction, 3),
                      "scale": round(art.scale, 5), "notes": art.notes}, indent=2))


if __name__ == "__main__":
    main()
