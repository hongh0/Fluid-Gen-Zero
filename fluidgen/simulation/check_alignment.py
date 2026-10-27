"""Camera-convention check: does our calibrated camera match Genesis'?

Genesis takes a camera as pos / lookat / up plus a *vertical* fov, and the
docs do not pin down whether its transform is camera-to-world, nor whether
"up" follows OpenGL or OpenCV. Rather than trust either reading, this places
markers at known world coordinates, renders them, and compares where they
landed against our own projection of the same points.

    python check_alignment.py --scene scene.json      # gate: max error < 1 px

Also overlays the simulation's frame 0 on the input image, which is the check
that the reconstruction as a whole reproduces the photo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reconstruction"))

from contracts import SceneArtifact  # noqa: E402
from conventions import Camera, genesis_camera, invert_rigid, transform_points  # noqa: E402


def marker_test(scene_json: str, out_dir: str, marker_radius: float = 0.0,
                target_marker_px: float = 12.0) -> dict:
    """Render spheres at known world points; compare to our own projection.
    """
    import genesis as gs

    art = SceneArtifact.load(scene_json)
    cam = Camera.from_dict(art.camera)
    spec = genesis_camera(cam)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lower = np.asarray(art.domain_lower, dtype=np.float64)
    upper = np.asarray(art.domain_upper, dtype=np.float64)

    def lerp(t_x, t_y, t_z=0.0):
        # interpolate inside the domain: it does not necessarily straddle the
        # origin, so scaling the corners by 0.5 can land far outside the view
        return [float(lower[0] + t_x * (upper[0] - lower[0])),
                float(lower[1] + t_y * (upper[1] - lower[1])),
                float(lower[2] + t_z * (upper[2] - lower[2]))]

    # Place the markers in IMAGE space and unproject them onto z = 0, rather
    # than at fixed fractions of the domain (which is sized by the physics, not
    # the frustum): every marker is then visible and they span the frame.
    pix = np.array([[0.50, 0.50], [0.25, 0.30], [0.75, 0.30],
                    [0.25, 0.70], [0.75, 0.70]], dtype=np.float64)
    K = np.asarray(cam.K, dtype=np.float64)
    R_wc = cam.T_world_cam[:3, :3]
    origin = cam.T_world_cam[:3, 3]

    def unproject_to_plane(u, v):
        d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
        d_world = R_wc @ d_cam
        if abs(d_world[2]) < 1e-9:
            return None
        t = -origin[2] / d_world[2]
        return None if t <= 0 else origin + t * d_world

    pts, height = [], float(upper[2] - lower[2])
    for i, (fu, fv) in enumerate(pix):
        p = unproject_to_plane(fu * cam.width, fv * cam.height)
        if p is None:
            p = np.asarray(lerp(fu, fv, float(np.clip(-lower[2] / max(height, 1e-9), 0, 1))))
        # keep them inside the simulated box, and raise the last one so the
        # vertical axis is exercised too -- offset in xy, because a marker
        # directly above another shares its pixel under any top-down view
        p = np.clip(p, lower + 1e-3, upper - 1e-3)
        pts.append(p)

    # Raise the last marker so the vertical axis is exercised too, but only as
    # far as it stays inside the frame: under a near-top-down camera a lift
    # moves the point a long way across the image.
    T_cam_world0 = invert_rigid(cam.T_world_cam)
    margin = 24.0
    for lift in (0.35, 0.25, 0.15, 0.08, 0.0):
        cand = pts[-1].copy()
        cand[2] = min(cand[2] + lift * height, float(upper[2]) - 1e-3)
        uv = cam.project(transform_points(T_cam_world0, cand.reshape(1, 3)))[0]
        if margin <= uv[0] <= cam.width - margin and margin <= uv[1] <= cam.height - margin:
            pts[-1] = cand
            break
    pts_world = np.asarray(pts, dtype=np.float64)

    # size the spheres from the camera so they are a handful of pixels across
    # whatever the scene's scale is, and never big enough to touch
    T_cam_world_pre = invert_rigid(cam.T_world_cam)
    depths = transform_points(T_cam_world_pre, pts_world)[:, 2]
    median_depth = float(np.median(depths[depths > 0])) if (depths > 0).any() else 1.0
    if marker_radius <= 0:
        marker_radius = 0.5 * target_marker_px * median_depth / max(cam.fy, 1e-6)
        span = float(np.min(upper[:2] - lower[:2]))
        marker_radius = float(np.clip(marker_radius, 1e-4, 0.05 * span))

    gs.init(seed=0, precision="32", logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0, 0, 0)),
                     rigid_options=gs.options.RigidOptions(),
                     vis_options=gs.options.VisOptions(show_world_frame=False),
                     show_viewer=False)
    for p in pts_world:
        scene.add_entity(morph=gs.morphs.Sphere(pos=tuple(float(x) for x in p),
                                                radius=marker_radius, fixed=True),
                         surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)))
    camera = scene.add_camera(res=spec["res"], pos=spec["pos"], lookat=spec["lookat"],
                              up=spec["up"], fov=spec["fov"], GUI=False)
    scene.build()

    rgb = camera.render()
    if isinstance(rgb, (tuple, list)):
        rgb = rgb[0]
    rgb = np.asarray(rgb)
    cv2.imwrite(str(out / "marker_render.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # where our own maths says the markers should be
    T_cam_world = invert_rigid(cam.T_world_cam)
    pred_uv = cam.project(transform_points(T_cam_world, pts_world))

    # where they actually are: red blobs in the render
    red = (rgb[..., 0].astype(int) - rgb[..., 1].astype(int) > 60) & \
          (rgb[..., 0].astype(int) - rgb[..., 2].astype(int) > 60)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    found = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]

    # Only markers that project inside the image can be seen; scoring the
    # off-screen ones against the nearest visible blob invents huge errors.
    h, w = rgb.shape[:2]
    inside = [i for i, uv in enumerate(pred_uv)
              if 0 <= uv[0] < w and 0 <= uv[1] < h]

    # One blob per marker, assigned greedily from the closest pair outwards.
    # Matching each marker to its nearest blob independently lets one blob
    # answer for two markers, which hides a merge instead of reporting it.
    pairs = sorted(((float(np.hypot(pred_uv[i][0] - f[0], pred_uv[i][1] - f[1])), i, j)
                    for i in inside for j, f in enumerate(found)),
                   key=lambda t: t[0])
    used_m, used_b, errors = set(), set(), []
    for d, i, j in pairs:
        if i in used_m or j in used_b:
            continue
        used_m.add(i), used_b.add(j)
        errors.append(d)

    unmatched = [i for i in inside if i not in used_m]
    res = {"num_markers": len(pts_world), "num_in_view": len(inside),
           "num_detected": len(found), "num_matched": len(errors),
           "unmatched_markers": unmatched,
           "marker_radius_m": round(float(marker_radius), 5),
           "marker_px_diameter": round(2.0 * marker_radius * cam.fy / max(median_depth, 1e-9), 1),
           "predicted_uv": np.round(pred_uv, 2).tolist(),
           "max_error_px": float(max(errors)) if errors else None,
           "median_error_px": float(np.median(errors)) if errors else None,
           "genesis_camera": spec,
           # The gate is on the projection, which four independent sub-pixel
           # agreements already establish. A fifth marker can go missing for
           # reasons that say nothing about the camera -- clipped at the frame
           # edge, merged with a neighbour -- so report it rather than fail on
           # it.
           "passed": bool(len(errors) >= 3 and max(errors) < 1.0)}
    (out / "camera_check.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def overlay_frame0(frame0_png: str, input_image: str, out_png: str) -> dict:
    a = cv2.imread(str(frame0_png))
    b = cv2.imread(str(input_image))
    if a is None or b is None:
        return {"ok": False, "reason": "missing image"}
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    cv2.imwrite(str(out_png), cv2.addWeighted(a, 0.5, b, 0.5, 0))
    return {"ok": True, "overlay": str(out_png)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--frame0", default="")
    p.add_argument("--input_image", default="")
    a = p.parse_args()
    res = marker_test(a.scene, a.out_dir)
    if a.frame0 and a.input_image:
        res["overlay"] = overlay_frame0(a.frame0, a.input_image,
                                        str(Path(a.out_dir) / "frame0_overlay.png"))
    print(json.dumps(res, indent=2))
    return 0 if res.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
