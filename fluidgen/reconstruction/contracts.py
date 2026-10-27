"""Artifact contracts between reconstruction stages.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class ContractError(ValueError):
    """A stage produced or consumed an artifact that violates its contract."""


def _check(condition: bool, message: str) -> None:
    """Enforce a contract (only report it when RECON_GATES=warn)."""
    if condition:
        return
    if os.environ.get("RECON_GATES", "").lower() == "warn":
        print(f"[gate] WOULD FAIL: {message}", flush=True)
        return
    raise ContractError(message)


@dataclass
class Artifact:
    """Base: JSON round-trip plus a `validate` hook."""

    def validate(self) -> "Artifact":
        return self

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"__type__": type(self).__name__, **asdict(self)}
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Artifact":
        data = json.loads(Path(path).read_text())
        data.pop("__type__", None)
        return cls(**data).validate()


@dataclass
class SegmentationArtifact(Artifact):
    """Stage 1: object / fluid / background masks."""

    image_path: str
    object_mask_path: str
    fluid_mask_path: str
    background_mask_path: str
    width: int
    height: int
    object_pixels: int
    fluid_pixels: int
    overlap_pixels: int          # object & fluid must not overlap
    uncovered_pixels: int        # every pixel should land in exactly one class
    object_prompt: str = ""
    background_pixels: int = -1  # -1 = not computed
    fluid_prompt: str = ""       # material word the liquid was segmented with
    notes: List[str] = field(default_factory=list)

    def validate(self) -> "SegmentationArtifact":
        total = self.width * self.height
        _check(total > 0, "empty image")
        _check(self.object_pixels > 0, "object mask is empty")
        _check(self.fluid_pixels > 0, "fluid mask is empty")
        _check(self.overlap_pixels <= 0.01 * total,
               f"object and fluid masks overlap on {self.overlap_pixels} px (>1% of image)")
        _check(self.uncovered_pixels <= 0.05 * total,
               f"{self.uncovered_pixels} px belong to no class (>5% of image)")
        _check(self.fluid_pixels <= 0.995 * total,
               f"fluid covers {100.0 * self.fluid_pixels / total:.1f}% of the image; "
               "nothing else was segmented")
        _check(self.object_pixels <= 0.60 * total,
               f"object covers {100.0 * self.object_pixels / total:.1f}% of the image; "
               "the detector almost certainly returned the whole frame")
        return self


@dataclass
class GeometryArtifact(Artifact):
    """Stage 2: MoGe point map, depth and intrinsics (camera frame)."""

    points_path: str             # (H, W, 3) float32 .npy
    depth_path: str              # (H, W) float32 .npy
    valid_mask_path: str
    K: List[List[float]]
    width: int
    height: int
    valid_ratio: float
    depth_min: float
    depth_max: float
    frame: str = "camera_opencv"
    metric: bool = False         # MoGe v2 is up to scale unless proven otherwise
    image_path: str = ""
    crop_window: List[int] = field(default_factory=list)   # [x0, x1, y0, y1]
    # How solidly the valid pixels fill their own bounding box; see `_bbox_fill`.
    valid_bbox_fill: float = 1.0
    notes: List[str] = field(default_factory=list)

    def validate(self) -> "GeometryArtifact":
        _check(self.frame == "camera_opencv", f"unexpected frame {self.frame!r}")
        _check(self.valid_ratio >= 0.10,
               f"only {self.valid_ratio:.1%} of pixels have valid geometry")
        _check(self.valid_bbox_fill >= 0.70,
               f"valid geometry fills only {self.valid_bbox_fill:.1%} of its own bounding box, "
               "so it is scattered rather than a coherent region")
        _check(self.depth_min > 0, f"non-positive depth {self.depth_min}")
        _check(self.depth_max > self.depth_min, "degenerate depth range")
        fy = float(self.K[1][1])
        fov_y = float(np.degrees(2 * np.arctan(0.5 * self.height / fy)))
        # Wide enough to admit telephoto close-ups; the gate only catches a
        # degenerate intrinsic matrix.
        _check(5.0 <= fov_y <= 150.0, f"implausible vertical fov {fov_y:.1f} deg")
        return self


@dataclass
class ObjectArtifact(Artifact):
    """Stage 3: complete object mesh plus SAM 3D Objects' initial layout."""

    mesh_path: str               # canonical/object frame
    scale: float
    rotation: List[List[float]]  # (3, 3) object -> camera
    translation: List[float]     # camera frame, metres
    num_vertices: int
    num_faces: int
    bbox_min: List[float]
    bbox_max: List[float]
    frame: str = "camera_opencv"
    backend: str = "sam3d_objects"
    notes: List[str] = field(default_factory=list)

    def validate(self) -> "ObjectArtifact":
        _check(self.num_vertices > 100, f"mesh has only {self.num_vertices} vertices")
        _check(self.num_faces > 100, f"mesh has only {self.num_faces} faces")
        extent = np.asarray(self.bbox_max) - np.asarray(self.bbox_min)
        # a degenerate axis breaks Genesis' CubeBoundary
        _check(np.all(extent > 1e-4),
               f"degenerate mesh bounding box, extent={extent.tolist()}")
        _check(self.scale > 0, f"non-positive scale {self.scale}")
        R = np.asarray(self.rotation, dtype=np.float64)
        _check(R.shape == (3, 3), "rotation must be 3x3")
        _check(np.allclose(R @ R.T, np.eye(3), atol=1e-3),
               "rotation is not orthonormal")
        return self


@dataclass
class FluidArtifact(Artifact):
    """Stage 4: fluid plane and the world frame it induces."""

    normal_cam: List[float]      # unit, pointing out of the liquid
    offset_cam: float            # normal . X + offset == 0
    inlier_ratio: float
    num_points: int
    T_world_cam: List[List[float]]
    domain_type: str             # "open_water" | "container"
    domain_size: List[float]     # (Lx, Ly, H) in world metres
    residual_z_median: float     # fluid points in world coords, should be ~0
    # Diagonal of the fluid point cloud in MoGe's units. MoGe is not metric and
    # its unit varies by orders of magnitude between images, so every tolerance
    # here is relative to this.
    scene_scale: float = 1.0
    notes: List[str] = field(default_factory=list)

    def validate(self) -> "FluidArtifact":
        n = np.asarray(self.normal_cam, dtype=np.float64)
        _check(abs(np.linalg.norm(n) - 1) < 1e-3, "plane normal is not unit length")
        # OpenCV +z points into the scene, so a surface we can see must face back
        _check(n[2] < 0, f"plane normal {n.tolist()} does not face the camera")
        _check(self.inlier_ratio >= 0.30,
               f"plane fit kept only {self.inlier_ratio:.1%} of fluid points")
        _check(self.num_points >= 200, f"only {self.num_points} fluid points to fit")
        _check(self.domain_type in ("open_water", "container"),
               f"unknown domain type {self.domain_type!r}")
        tol = max(1e-2 * float(self.scene_scale), 1e-6)
        _check(abs(self.residual_z_median) < tol,
               f"fluid points sit at z={self.residual_z_median:.4g} in world coords, not 0 "
               f"(tolerance {tol:.4g} = 1% of the scene's {self.scene_scale:.4g}-unit span)")
        return self


@dataclass
class RegistrationArtifact(Artifact):
    """Stage 5: object pose refined against the fluid plane."""

    scale: float
    rotation: List[List[float]]  # object -> world
    translation: List[float]     # world frame, metres
    quat_wxyz: List[float]       # what Genesis actually consumes
    mask_iou: float
    depth_median_error: float
    interaction_state: str       # airborne | touching | floating | submerged
    signed_distance_to_plane: float
    submerged_fraction: float
    frame: str = "world_z_up"
    passed: bool = False
    #: Symmetric silhouette-outline distance over the object's bbox diagonal.
    #: Default is inf, so a missing value never passes the boundary criterion.
    boundary_error: float = float("inf")
    notes: List[str] = field(default_factory=list)
    IOU_GATE = 0.6
    BOUNDARY_GATE = 0.05

    def validate(self) -> "RegistrationArtifact":
        _check(self.frame == "world_z_up", f"unexpected frame {self.frame!r}")
        _check(self.interaction_state in
               ("airborne", "touching", "floating", "submerged"),
               f"unknown interaction state {self.interaction_state!r}")
        _check(0.0 <= self.mask_iou <= 1.0, "iou out of range")
        _check(0.0 <= self.submerged_fraction <= 1.0, "submerged fraction out of range")
        # the gate itself is reported, not enforced, so a failure is still
        # inspectable; run.py refuses to continue when passed is False
        return self

    def evaluate_gate(self) -> "RegistrationArtifact":
        by_area = self.mask_iou >= self.IOU_GATE
        by_outline = self.boundary_error <= self.BOUNDARY_GATE
        self.passed = by_area or by_outline
        if by_outline and not by_area:
            self.notes.append(
                f"registration passed on outline: IoU {self.mask_iou:.3f} < {self.IOU_GATE} "
                f"but boundary error {self.boundary_error:.4f} <= {self.BOUNDARY_GATE}")
        if not self.passed:
            self.notes.append(
                f"registration gate failed: IoU {self.mask_iou:.3f} < {self.IOU_GATE} "
                f"and boundary error {self.boundary_error:.4f} > {self.BOUNDARY_GATE}")
        return self


@dataclass
class SceneArtifact(Artifact):
    """Stage 6: everything Genesis needs, in world coords."""

    object_mesh_path: str
    object_pos: List[float]
    object_quat_wxyz: List[float]
    object_scale: float
    object_material: Dict[str, Any]
    fluid_particles_path: str
    fluid_material: Dict[str, Any]
    domain_lower: List[float]
    domain_upper: List[float]
    gravity: List[float]
    camera: Dict[str, Any]
    interaction_state: str
    frame: str = "world_z_up"
    # static background (walls, table, far water) as world-space coloured
    # points; carried for visualisation and collision context, not simulated
    background_points_path: str = ""
    particle_size: float = 0.01
    # the container standing on the observed waterline; simulated, unlike the
    # background points, because an unconfined liquid just slumps and spreads.
    # `vessel["kind"]` is "planes" (tangent half-spaces, cheap) or "mesh" (a cup
    # with a real rim, whose sampled SDF costs ~5x the rollout time), in which
    # case `vessel_mesh_path` points at the .obj.
    vessel_mesh_path: str = ""
    vessel: Dict[str, Any] = field(default_factory=dict)
    # MPM cells per metre. Derived from the object's size rather than fixed, so
    # a 5 cm scene and a 5 m scene are both resolved.
    grid_density: int = 64
    # world units per metre applied to the whole scene; MoGe is only defined up
    # to scale, so this is what makes the numbers here actually metric
    metric_scale: float = 1.0
    notes: List[str] = field(default_factory=list)

    def validate(self) -> "SceneArtifact":
        _check(self.frame == "world_z_up", f"unexpected frame {self.frame!r}")
        _check(self.grid_density > 0, f"non-positive grid_density {self.grid_density}")
        _check(self.metric_scale > 0, f"non-positive metric scale {self.metric_scale}")
        g = np.asarray(self.gravity, dtype=np.float64)
        _check(g[0] == 0 and g[1] == 0 and g[2] < 0,
               f"gravity {g.tolist()} is not straight down in the world frame")
        lower = np.asarray(self.domain_lower, dtype=np.float64)
        upper = np.asarray(self.domain_upper, dtype=np.float64)
        extent = upper - lower
        _check(np.all(extent > 0), f"degenerate simulation domain, extent={extent.tolist()}")
        _check(np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)),
               "non-finite simulation domain")
        _check(self.object_scale > 0, "non-positive object scale")
        q = np.asarray(self.object_quat_wxyz, dtype=np.float64)
        _check(abs(np.linalg.norm(q) - 1) < 1e-3, "object quaternion is not unit length")
        return self


def require_mpm_domain(lower, upper, grid_density: int = 64, margin_cells: float = 6.0):
    """Widen any axis too thin for Genesis' MPM boundary padding.

    `MPMSolver.setup_boundary` shrinks the domain by `3 * dx` on each side, so
    an axis thinner than `6 / grid_density` inverts and trips
    `assert (upper >= lower)` inside CubeBoundary -- with bounds that looked
    perfectly valid going in. Only offending axes are touched.
    """
    lower = np.asarray(lower, dtype=np.float64).copy()
    upper = np.asarray(upper, dtype=np.float64).copy()
    min_extent = margin_cells * 2.0 / float(grid_density)
    extent = upper - lower
    thin = extent < min_extent
    if np.any(thin):
        center = 0.5 * (lower + upper)
        half = np.where(thin, 0.5 * min_extent, 0.5 * extent)
        lower = center - half
        upper = center + half
    return lower, upper, thin.tolist()
