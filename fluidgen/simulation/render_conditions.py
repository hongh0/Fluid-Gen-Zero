"""Turn a rollout into the conditioning signals the video backbones consume.

Everything is projected with the *same* calibrated camera the scene was built
on (checked to sub-pixel by check_alignment), so the flow, the masks and the
2D tracks are all mutually consistent.

Outputs (per rollout):
    optical_flow.npy      (T-1, 2, H, W) float32, **pixel** displacements
    optical_flow.mp4      HSV visualisation
    fg_mask.mp4 / fluid_mask.mp4 / segmap.mp4
    object_trajectory.txt / fluid_trajectory.txt   "x,y" per frame
    trajectory_2d.npz     object/fluid centroid tracks, (T, 2)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reconstruction"))

from contracts import SceneArtifact  # noqa: E402
from conventions import Camera, invert_rigid, transform_points  # noqa: E402
try:
    from fluidgen.vendor_flow_utils import flow_to_image as _fs_flow_to_image
except Exception:                                     # pragma: no cover
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from vendor_flow_utils import flow_to_image as _fs_flow_to_image
    except Exception:
        _fs_flow_to_image = None


class _Writer:
    """Write H.264 (yuv420p + faststart) through ffmpeg, falling back to
    OpenCV's mp4v when no H.264 encoder is available."""

    @staticmethod
    def _ffmpeg_with_h264():
        """An ffmpeg that can encode H.264: prefer `imageio_ffmpeg`'s bundled
        binary and check that libx264 is really present."""
        import shutil as _sh
        cands = []
        try:
            import imageio_ffmpeg
            cands.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass
        cands += [c for c in (_sh.which("ffmpeg"), "/usr/bin/ffmpeg") if c]
        for c in cands:
            try:
                out = subprocess.run([c, "-hide_banner", "-encoders"],
                                     capture_output=True, text=True, timeout=30).stdout
                if "libx264" in out:
                    return c
            except Exception:
                continue
        return None

    def __init__(self, path, fps, size):
        self.path, self.size = str(path), size
        w, h = size
        self.proc = self.vw = None
        ff = self._ffmpeg_with_h264()
        if ff and w % 2 == 0 and h % 2 == 0:
            self.proc = subprocess.Popen(
                [ff, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
        else:
            self.vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, size)

    def write(self, frame):
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            except BrokenPipeError:
                # surface why the encoder died instead of a bare pipe error
                err = self.proc.stderr.read().decode(errors="ignore")[-400:]
                raise RuntimeError(f"ffmpeg died while encoding {self.path}: {err}") from None
        else:
            self.vw.write(frame)

    def release(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                err = self.proc.stderr.read().decode(errors="ignore")[-400:]
                raise RuntimeError(f"ffmpeg failed on {self.path}: {err}")
        else:
            self.vw.release()


def _spacing_px(uv: np.ndarray) -> float:
    """Typical gap between neighbouring projected points, in pixels.

    Splat radius and smoothing both have to follow this. A fixed 2-pixel dot and
    sigma = 1 leave the object as visible speckle: 6000 mesh vertices across a
    200-pixel silhouette sit about 2.6 px apart, so the discs barely touch and
    the nearest-neighbour fill between them shows up as Voronoi patches.
    """
    if len(uv) < 4:
        return 2.0
    area = (np.ptp(uv[:, 0]) + 1.0) * (np.ptp(uv[:, 1]) + 1.0)
    return float(np.clip(np.sqrt(area / len(uv)), 1.0, 24.0))


def _project(cam: Camera, T_cam_world: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    return cam.project(transform_points(T_cam_world, pts_world))


def _point_render(uv: np.ndarray, depth: np.ndarray, values: np.ndarray,
                  radius_px: np.ndarray, h: int, w: int):
    """Painter's-algorithm point render with a depth test.
    """
    canvas = np.zeros((h, w, values.shape[1]), np.float32)
    hit = np.zeros((h, w), np.uint8)
    order = np.argsort(-depth)                       # far first, near overwrites
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    for i in order:
        if not (0 <= u[i] < w and 0 <= v[i] < h):
            continue
        r = int(radius_px[i])
        cv2.circle(canvas, (u[i], v[i]), r, tuple(float(x) for x in values[i]), -1)
        cv2.circle(hit, (u[i], v[i]), r, 1, -1)
    return canvas, hit.astype(bool)


def _visible(uv: np.ndarray, depth: np.ndarray, radius_px: np.ndarray,
             h: int, w: int, fy: float, shell: float = 4.0) -> np.ndarray:
    """Which points lie on the surface the camera actually sees.

    The tolerance is per point rather than a fixed distance: a disc of radius r
    at depth z covers r*z/fy metres, and a slanted surface crossing it changes
    depth by a few times that, so `shell` multiples of it keep the real surface
    while still rejecting the far side.
    """
    zb = np.full((h, w), np.inf, np.float32)
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    for i in np.argsort(-depth):                     # far first; near overwrites
        if 0 <= u[i] < w and 0 <= v[i] < h:
            cv2.circle(zb, (u[i], v[i]), int(radius_px[i]), float(depth[i]), -1)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    tol = shell * radius_px * np.clip(depth, 1e-6, None) / max(fy, 1e-6)
    near = np.zeros(len(uv), bool)
    uc, vc = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    near[inside] = depth[inside] <= zb[vc[inside], uc[inside]] + tol[inside]
    return inside & near


def _apparent_radius(depth: np.ndarray, fy: float, size_m: float,
                     lo: int = 1, hi: int = 12) -> np.ndarray:
    """How many pixels across a particle of physical size `size_m` covers."""
    r = 0.5 * fy * float(size_m) / np.clip(depth, 1e-6, None)
    return np.clip(np.round(r), lo, hi).astype(np.int32)


def _close(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Bridge the gaps a discrete particle set leaves inside a solid surface."""
    m = _close_mask(mask, k)
    try:
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(m)
    except Exception:
        pass
    return np.asarray(m, dtype=bool)


def _fill(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill, so the flow is dense inside the covered region."""
    if mask.all() or not mask.any():
        return field
    _, idx = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    order = np.zeros(idx.max() + 1, np.int64)
    order[idx[mask]] = np.arange(len(ys))
    return field[ys[order[idx]], xs[order[idx]]]


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Middlebury colour wheel; falls back to a fixed-scale HSV if missing."""
    if _fs_flow_to_image is not None:
        return _fs_flow_to_image(flow, clip_flow=None, convert_to_bgr=True)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 255.0 / 20.0, 0, 255).astype(np.uint8)   # fixed scale
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _normalized_splat_smooth(flow: np.ndarray, valid: np.ndarray, sigma: float):
    """Gaussian splat normalised by coverage."""
    w = valid.astype(np.float32)
    fb = cv2.GaussianBlur(flow * w[:, :, None], (0, 0), sigmaX=sigma)
    wb = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma)
    out_mask = wb > 1e-6
    smoothed = np.zeros_like(flow)
    smoothed[out_mask] = fb[out_mask] / wb[out_mask, None]
    return smoothed, out_mask


def _close_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask.astype(bool, copy=False)
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0


def render(scene_json: str, sim_dir: str, out_dir: str, fps: int = 24,
           radius: int = 2, max_frames: int = 0,
           fg_point_size: int = 5, fluid_point_size: int = 10,
           fg_close_kernel: int = 9, fluid_close_kernel: int = 51,
           smooth_sigma: float = 1.0) -> dict:
    scene = SceneArtifact.load(scene_json)
    cam = Camera.from_dict(scene.camera)
    T_cam_world = invert_rigid(cam.T_world_cam)
    h, w = cam.height, cam.width
    sim, out = Path(sim_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fg = np.load(sim / "fg_traj.npz")["xyz"]
    fl = np.load(sim / "fluid_traj.npz")["xyz"]
    T = min(len(fg), len(fl))
    if max_frames:
        T = min(T, max_frames)

    w_flow = _Writer(out / "optical_flow.mp4", fps, (w, h))
    w_fg = _Writer(out / "fg_mask.mp4", fps, (w, h))
    w_fl = _Writer(out / "fluid_mask.mp4", fps, (w, h))
    w_seg = _Writer(out / "segmap.mp4", fps, (w, h))

    flows = np.zeros((T - 1, 2, h, w), np.float32)
    obj_xy, flu_xy = [], []

    flu_size = float(getattr(scene, "particle_size", 0.01) or 0.01)
    # A rigid object is tracked through mesh vertices, which are dense and
    # arbitrary in spacing; an MPM one through real particles laid out at
    # `particle_size`, which is what they should be drawn at.
    obj_kind = str(scene.object_material.get("type", "rigid")).lower()
    obj_size = flu_size if obj_kind != "rigid" else float(scene.object_scale) * 0.004
    uv_fg_prev = uv_fl_prev = None
    z_fg_prev = z_fl_prev = None
    r_fg_prev = r_fl_prev = None
    for t in range(T):
        cam_fg = transform_points(T_cam_world, fg[t])
        cam_fl = transform_points(T_cam_world, fl[t])
        uv_fg, z_fg = cam.project(cam_fg), cam_fg[:, 2]
        uv_fl, z_fl = cam.project(cam_fl), cam_fl[:, 2]
        obj_xy.append(uv_fg.mean(axis=0))
        flu_xy.append(uv_fl.mean(axis=0))
        sp_fg, sp_fl = _spacing_px(uv_fg), _spacing_px(uv_fl)
        r_fg = np.maximum(_apparent_radius(z_fg, cam.fy, obj_size),
                          max(fg_point_size // 2, int(np.ceil(0.8 * sp_fg))))
        r_fl = np.maximum(_apparent_radius(z_fl, cam.fy, flu_size),
                          max(fluid_point_size // 2, int(np.ceil(0.8 * sp_fl))))
        _, hit_fg = _point_render(uv_fg, z_fg, np.ones((len(uv_fg), 1), np.float32), r_fg, h, w)
        _, hit_fl = _point_render(uv_fl, z_fl, np.ones((len(uv_fl), 1), np.float32), r_fl, h, w)
        fg_mask = _close(hit_fg, fg_close_kernel).astype(np.uint8)
        fl_mask = (_close(hit_fl, fluid_close_kernel) & (fg_mask == 0)).astype(np.uint8)
        w_fg.write(cv2.cvtColor(fg_mask * 255, cv2.COLOR_GRAY2BGR))
        w_fl.write(cv2.cvtColor(fl_mask * 255, cv2.COLOR_GRAY2BGR))
        seg = np.zeros((h, w, 3), np.uint8)
        seg[fl_mask > 0] = (255, 128, 0)      # fluid  (BGR of 0,128,255)
        seg[fg_mask > 0] = (0, 128, 255)      # object (BGR of 255,128,0)
        w_seg.write(seg)

        if t > 0:
            uv = np.concatenate([uv_fg_prev, uv_fl_prev], axis=0)
            zz = np.concatenate([z_fg_prev, z_fl_prev], axis=0)
            nxt = np.concatenate([uv_fg, uv_fl], axis=0)
            rr = np.concatenate([r_fg_prev, r_fl_prev], axis=0)
            disp = (nxt - uv).astype(np.float32)
            # Object and liquid are tested for visibility together, because the
            # object occludes the liquid behind it and the camera sees one
            # surface, not two.
            vis = _visible(uv, zz, rr, h, w, cam.fy)
            n_fg = len(uv_fg_prev)
            # Discarding the hidden half doubles the gap between what is left,
            # so both the splat radius and the smoothing have to follow the
            # spacing of the *visible* points, not of all of them.
            sp_fg_v, sp_fl_v = _spacing_px(uv[:n_fg][vis[:n_fg]]), _spacing_px(uv[n_fg:][vis[n_fg:]])
            rr = rr.copy()
            rr[:n_fg] = np.maximum(rr[:n_fg], int(np.ceil(sp_fg_v)))
            rr[n_fg:] = np.maximum(rr[n_fg:], int(np.ceil(sp_fl_v)))
            field, hit = _point_render(uv[vis], zz[vis], disp[vis], rr[vis], h, w)
            covered = _close(hit, fluid_close_kernel)
            field = _fill(field, hit)
            field[~covered] = 0.0
            # Smooth at the scale the samples are actually spaced at, not a
            # fixed 1 px. Below the spacing the Gaussian cannot remove the
            # discretisation it is there to remove.
            sigma = max(smooth_sigma, 0.9 * max(sp_fg_v, sp_fl_v))
            if sigma > 0:
                field, _ = _normalized_splat_smooth(field, covered, sigma)
                field[~covered] = 0.0
            flows[t - 1, 0] = field[..., 0]
            flows[t - 1, 1] = field[..., 1]
            w_flow.write(flow_to_rgb(field))
        uv_fg_prev, uv_fl_prev = uv_fg, uv_fl
        z_fg_prev, z_fl_prev = z_fg, z_fl
        r_fg_prev, r_fl_prev = r_fg, r_fl

    for writer in (w_flow, w_fg, w_fl, w_seg):
        writer.release()

    np.save(out / "optical_flow.npy", flows)
    obj_xy = np.asarray(obj_xy, np.float32)
    flu_xy = np.asarray(flu_xy, np.float32)
    np.savez_compressed(out / "trajectory_2d.npz", object_xy=obj_xy, fluid_xy=flu_xy)
    for name, arr in (("object_trajectory.txt", obj_xy), ("fluid_trajectory.txt", flu_xy)):
        (out / name).write_text("\n".join(f"{p[0]:.3f},{p[1]:.3f}" for p in arr) + "\n")

    mag = np.sqrt(flows[:, 0] ** 2 + flows[:, 1] ** 2)
    return {
        "frames": int(T), "flow_frames": int(T - 1), "resolution": [w, h],
        "flow_units": "pixels", "flow_colour": "middlebury",
        "flow_mean_px": float(mag.mean()), "flow_p99_px": float(np.percentile(mag, 99)),
        "flow_active_ratio": float((mag > 0.5).mean()),
        "object_xy_start": obj_xy[0].tolist(), "object_xy_end": obj_xy[-1].tolist(),
        "outputs": sorted(p.name for p in out.iterdir()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--sim_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--fg_point_size", type=int, default=5)
    p.add_argument("--fluid_point_size", type=int, default=10)
    p.add_argument("--fg_close_kernel", type=int, default=9)
    p.add_argument("--fluid_close_kernel", type=int, default=51)
    p.add_argument("--smooth_sigma", type=float, default=1.0)
    a = p.parse_args()
    print(json.dumps(render(a.scene, a.sim_dir, a.out_dir, a.fps, a.radius, a.max_frames,
                            a.fg_point_size, a.fluid_point_size, a.fg_close_kernel,
                            a.fluid_close_kernel, a.smooth_sigma), indent=2))


if __name__ == "__main__":
    main()
