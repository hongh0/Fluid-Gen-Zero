"""Coordinate conventions shared by reconstruction and simulation.

Three frames are in play:

  camera frame (OpenCV)  +x right, +y down, +z forward into the scene.
                         MoGe / SAM 3D Objects both work in this frame.

  world frame            +z up, origin on the fluid surface, so the fluid
                         plane is exactly z = 0 and gravity is (0, 0, -g).
                         This is also Genesis' native convention
                         (`SimOptions.gravity == (0, 0, -9.81)`,
                          `add_camera(up=(0, 0, 1))`), so the reconstruction
                         world drops into Genesis with no transform at all.

  Genesis camera         positioned with pos / lookat / up and a *vertical*
                         fov; it has no principal-point parameter, hence
                         `center_principal_point` below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

GRAVITY = 9.81


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
@dataclass
class Camera:
    """Pinhole camera. `T_world_cam` maps camera coords to world coords."""

    K: np.ndarray                 # (3, 3) intrinsics, OpenCV convention
    width: int
    height: int
    T_world_cam: np.ndarray = None  # (4, 4); identity means "still in camera frame"

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.T_world_cam = (np.eye(4) if self.T_world_cam is None
                            else np.asarray(self.T_world_cam, dtype=np.float64).reshape(4, 4))

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def fov_y_deg(self) -> float:
        """Genesis takes a vertical fov: it derives fy = 0.5*H / tan(fov/2)."""
        return float(np.degrees(2.0 * np.arctan(0.5 * self.height / self.fy)))

    @property
    def principal_point_offset(self) -> Tuple[float, float]:
        """How far the principal point sits from the image centre, in pixels.

        Genesis cannot represent a non-zero offset, so anything larger than a
        pixel or so has to be removed by cropping before it silently biases
        every rendered frame.
        """
        return (self.cx - 0.5 * self.width, self.cy - 0.5 * self.height)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """(N, 3) camera-frame points -> (N, 2) pixels."""
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = np.clip(pts[:, 2], 1e-9, None)
        u = self.fx * pts[:, 0] / z + self.cx
        v = self.fy * pts[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def unproject(self, depth: np.ndarray) -> np.ndarray:
        """(H, W) depth -> (H, W, 3) camera-frame points."""
        h, w = depth.shape
        vs, us = np.meshgrid(np.arange(h, dtype=np.float64),
                             np.arange(w, dtype=np.float64), indexing="ij")
        x = (us - self.cx) / self.fx * depth
        y = (vs - self.cy) / self.fy * depth
        return np.stack([x, y, depth], axis=-1)

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "width": self.width, "height": self.height,
                "T_world_cam": self.T_world_cam.tolist(), "fov_y_deg": self.fov_y_deg,
                "convention": "opencv"}

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        return Camera(np.array(d["K"]), int(d["width"]), int(d["height"]),
                      np.array(d["T_world_cam"]))


def principal_point_crop_window(K: np.ndarray, width: int, height: int
                                ) -> Tuple[int, int, int, int]:
    """Largest window centred on the principal point: returns (x0, x1, y0, y1).

    Genesis' camera is a centred pinhole, so an off-centre (cx, cy) cannot be
    expressed and would show up as a constant pixel offset between the render
    and the input image -- an offset that biases trajectory error identically
    for every method and is therefore easy to miss.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    half_w = int(min(cx, width - cx))
    half_h = int(min(cy, height - cy))
    x0 = int(round(cx)) - half_w
    y0 = int(round(cy)) - half_h
    return x0, x0 + 2 * half_w, y0, y0 + 2 * half_h


def center_principal_point(image: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop `image` so its principal point lands at the centre; also fixes K."""
    h, w = image.shape[:2]
    x0, x1, y0, y1 = principal_point_crop_window(K, w, h)
    K_new = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    K_new[0, 2] -= x0
    K_new[1, 2] -= y0
    return image[y0:y1, x0:x1], K_new


# --------------------------------------------------------------------------
# world frame from the fluid plane
# --------------------------------------------------------------------------
def plane_to_world(normal: np.ndarray, offset: float) -> np.ndarray:
    """Build `T_world_cam` that sends the fluid plane to z = 0 with +z up.

    The plane is given in camera coords as ``normal . X + offset == 0`` with
    `normal` already oriented towards the camera (i.e. pointing out of the
    liquid).
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        raise ValueError("degenerate plane normal")
    n = n / norm
    offset = float(offset) / norm

    # any two in-plane axes; pick the world x from the most orthogonal cardinal
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(seed, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    R = np.stack([e1, e2, n], axis=0)          # rows: world axes in camera coords
    t = np.array([0.0, 0.0, offset])           # plane points land on z = 0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to (..., 3) points."""
    pts = np.asarray(points, dtype=np.float64)
    shape = pts.shape
    flat = pts.reshape(-1, 3)
    out = flat @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
    return out.reshape(shape)


def invert_rigid(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# --------------------------------------------------------------------------
# Genesis interop
# --------------------------------------------------------------------------
def quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    """Genesis morphs take `quat` in w-x-y-z order.

    We always pass quat rather than `euler`, because Genesis' euler is degrees
    in scipy's extrinsic x-y-z order -- an easy thing to get silently wrong.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k]))
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
        w, x, y, z = q
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def R_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Inverse of `quat_wxyz_from_R`; SAM 3D Objects reports w-x-y-z too
    (its own helper documents "q: (..., 4) tensor in w,x,y,z order")."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(1e-12, np.linalg.norm(q))
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def genesis_camera(cam: Camera) -> dict:
    """Convert a calibrated Camera into Genesis `add_camera` arguments.

    Genesis wants a look-at description in world coords. In the OpenCV camera
    frame the viewing direction is +z and "up" on screen is -y, so those two
    columns of the camera-to-world rotation give lookat and up directly.
    """
    T = cam.T_world_cam
    R, pos = T[:3, :3], T[:3, 3]
    forward = R[:, 2]
    up = -R[:, 1]
    return {
        "res": (int(cam.width), int(cam.height)),
        "pos": tuple(float(x) for x in pos),
        "lookat": tuple(float(x) for x in (pos + forward)),
        "up": tuple(float(x) for x in up),
        "fov": cam.fov_y_deg,          # Genesis' fov is vertical
    }
