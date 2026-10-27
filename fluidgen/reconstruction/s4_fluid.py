"""Stage 4 -- fit the fluid surface and let it define the world frame.

Only fluid pixels well away from the object are used: specular highlights, the
object's reflection and its shadow all sit near the object and would otherwise
drag the plane. RANSAC then rejects whatever survives that.

The fitted plane *is* the world: z = 0 on the surface, +z along the normal,
gravity (0, 0, -g).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from contracts import FluidArtifact
from conventions import plane_to_world, transform_points


def _ransac_plane(points: np.ndarray, threshold: float, iterations: int = 2000,
                  seed: int = 0) -> Tuple[np.ndarray, float, np.ndarray]:
    """Plane fit returning (unit normal, offset, inlier mask)."""
    rng = np.random.default_rng(seed)
    # A few thousand samples pin a plane just as well as 300k and keep each
    # iteration cheap; the refit below still uses every inlier.
    if len(points) > 20000:
        sub = rng.choice(len(points), 20000, replace=False)
        sample_points = points[sub]
    else:
        sample_points = points
    n_pts = len(sample_points)
    best_hits = -1
    best = (np.array([0.0, 0.0, -1.0]), 0.0)
    for _ in range(iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = sample_points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -float(normal @ p0)
        dist = np.abs(sample_points @ normal + offset)
        hits = int((dist < threshold).sum())
        if hits > best_hits:
            best_hits, best = hits, (normal, offset)
    # least-squares refit over every inlier in the full set
    normal, offset = best
    best_inliers = np.abs(points @ normal + offset) < threshold
    if best_inliers.sum() >= 3:
        pts = points[best_inliers]
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vh[-1] / np.linalg.norm(vh[-1])
        offset = -float(normal @ centroid)
        best_inliers = np.abs(points @ normal + offset) < threshold
    return normal, offset, best_inliers


def _waterline(obj: np.ndarray, fluid: np.ndarray, usable: np.ndarray,
               band: int = 11) -> np.ndarray:
    """The rim where the object meets the liquid, as a stand-in for the surface.
    """
    k = np.ones((band, band), np.uint8)
    rim = obj & (cv2.erode(obj.astype(np.uint8), k) == 0)
    near_fluid = cv2.dilate(fluid.astype(np.uint8), k) > 0
    return rim & near_fluid & usable


def _footprint_on_plane(mask: np.ndarray, K, normal: np.ndarray,
                        offset: float, max_pts: int = 200_000) -> np.ndarray:
    """Where a masked region lands on the plane, by ray intersection.

    The domain is normally the extent of the fluid points that were fitted, but
    a waterline fit only knows about the object's own rim, and a pool the size
    of the object is not the scene. Rays through the liquid's pixels meet the
    plane where the liquid actually is, which needs no depth at all.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((0, 3), np.float64)
    if ys.size > max_pts:
        sel = np.linspace(0, ys.size - 1, max_pts).astype(np.int64)
        ys, xs = ys[sel], xs[sel]
    uv1 = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    dirs = (np.linalg.inv(np.asarray(K, dtype=np.float64)) @ uv1).T
    denom = dirs @ normal
    ok = np.abs(denom) > 1e-9
    t = np.zeros(len(dirs))
    t[ok] = -offset / denom[ok]
    ok &= t > 0                                   # only rays that hit in front
    return dirs[ok] * t[ok, None]


def fit_fluid(geometry_json: str, segmentation_json: str, out_dir: str,
              erode_px: int = 25, domain_margin: float = 1.5,
              seed: int = 0) -> FluidArtifact:
    from contracts import GeometryArtifact, SegmentationArtifact

    geo = GeometryArtifact.load(geometry_json)
    seg = SegmentationArtifact.load(segmentation_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    points = np.load(geo.points_path)
    valid = np.load(geo.valid_mask_path)
    h, w = points.shape[:2]

    fluid = cv2.imread(seg.fluid_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    obj = cv2.imread(seg.object_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if fluid.shape != (h, w):   # stage 2 may have cropped to centre the principal point
        fluid = cv2.resize(fluid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        obj = cv2.resize(obj.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    # stay away from the object: reflections, highlights and shadow live there
    far_from_object = cv2.dilate(obj.astype(np.uint8),
                                 np.ones((2 * erode_px + 1,) * 2, np.uint8)) == 0
    usable = valid & np.isfinite(points).all(axis=-1)
    sample = fluid & far_from_object & usable
    notes = [f"{int(sample.sum())} fluid points after excluding a {erode_px}px halo "
             f"around the object"]
    if sample.sum() < 200:
        sample = fluid & usable
        notes.append("halo left too few points; falling back to all fluid pixels")
    used_waterline = sample.sum() < 200
    if used_waterline:
        sample = _waterline(obj, fluid, usable)
        notes.append(f"the liquid itself has no usable depth; fitting the plane to "
                     f"{int(sample.sum())} waterline points on the object's rim")
        if sample.sum() < 50:
            raise ValueError("neither the liquid nor its waterline has usable geometry")

    pts = points[sample]
    scene_scale = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    threshold = max(1e-4, 0.015 * scene_scale)
    if used_waterline:
        # The rim is a proxy for the waterline, not the waterline itself: a
        # rounded object's silhouette sits at its widest cross-section, which is
        # above or below the surface by some fraction of its size. Hold the fit
        # to the object's flatness rather than to a liquid's.
        threshold = max(threshold, 0.05 * scene_scale)
    normal, offset, inliers = _ransac_plane(pts, threshold, seed=seed)

    # orient the normal towards the camera: in OpenCV +z goes into the scene,
    # so a surface we can see must have a normal with negative z
    if normal[2] > 0:
        normal, offset = -normal, -offset

    T_world_cam = plane_to_world(normal, offset)
    world_pts = transform_points(T_world_cam, pts[inliers])
    residual = float(np.median(world_pts[:, 2]))

    # domain: open water unless the fluid clearly stops inside the frame
    border_touch = (fluid[0, :].mean() + fluid[-1, :].mean()
                    + fluid[:, 0].mean() + fluid[:, -1].mean()) / 4.0
    domain_type = "open_water" if border_touch > 0.25 else "container"
    liquid_world = world_pts
    if used_waterline:
        foot = _footprint_on_plane(fluid, geo.K, normal, offset)
        if len(foot):
            liquid_world = transform_points(T_world_cam, foot)
            notes.append(f"domain and published point cloud from {len(foot)} liquid "
                         "pixels ray-cast onto the plane")
        else:
            notes.append("ray-cast footprint was empty; falling back to the fitted points")
    span = liquid_world[:, :2]
    lx = float(np.percentile(span[:, 0], 99) - np.percentile(span[:, 0], 1))
    ly = float(np.percentile(span[:, 1], 99) - np.percentile(span[:, 1], 1))
    depth_h = max(0.1 * max(lx, ly), 0.05)
    margin = domain_margin if domain_type == "open_water" else 1.1
    domain = [float(lx * margin), float(ly * margin), float(depth_h)]
    notes.append(f"border coverage {border_touch:.2f} -> {domain_type}")

    art = FluidArtifact(
        normal_cam=[float(x) for x in normal], offset_cam=float(offset),
        inlier_ratio=float(inliers.mean()), num_points=int(len(pts)),
        T_world_cam=T_world_cam.tolist(), domain_type=domain_type,
        domain_size=domain, residual_z_median=residual,
        scene_scale=float(scene_scale), notes=notes)
    art.save(out / "fluid.json")

    np.save(out / "fluid_points_world.npy", liquid_world.astype(np.float32))
    vis = np.zeros((h, w), np.uint8)
    vis[sample] = 128
    idx = np.nonzero(sample)
    vis[idx[0][inliers], idx[1][inliers]] = 255
    cv2.imwrite(str(out / "fluid_plane_inliers.png"), vis)
    return art


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", required=True)
    p.add_argument("--segmentation", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--erode_px", type=int, default=25)
    a = p.parse_args()
    art = fit_fluid(a.geometry, a.segmentation, a.out_dir, a.erode_px)
    print(json.dumps({"normal_cam": np.round(art.normal_cam, 4).tolist(),
                      "inlier_ratio": round(art.inlier_ratio, 3),
                      "residual_z_median": float(f"{art.residual_z_median:.2e}"),
                      "domain": [art.domain_type, np.round(art.domain_size, 3).tolist()],
                      "notes": art.notes}, indent=2))


if __name__ == "__main__":
    main()
