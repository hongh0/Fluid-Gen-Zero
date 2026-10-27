"""Stage 6 -- assemble everything Genesis needs, in world coordinates.

Fluid particles are sampled in the world frame directly (a slab under z = 0
spanning the domain), so no further transform is applied anywhere downstream:
the reconstruction world *is* the simulation world.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from contracts import (FluidArtifact, GeometryArtifact, RegistrationArtifact,
                       SceneArtifact, require_mpm_domain)
from conventions import GRAVITY, Camera

# Fallback fluid properties; the real ones come from the intent parser.
DEFAULT_FLUID = {"material": "water", "rho": 1000.0, "viscous": False, "viscosity": 1.0}
MPM_E_CEILING = 5e6
#: MPM cells per metre. 100 gives dx = 1 cm: an 8 cm object spans 8 cells and a
#: 6 cm liquid is 6 deep. Cost goes as dx^-3, so this is the main runtime dial.
DEFAULT_GRID_DENSITY = 100
#: Cells the grid must put across the object's thinnest axis for the rigid-MPM
#: coupling to be defined at all.
MIN_CELLS_ACROSS_OBJECT = 4.0
DEFAULT_OBJECT = {"type": "rigid", "rho": 700.0, "coup_friction": 0.1,
                  "E": 2e6, "nu": 0.27}


def _mesh_world_bounds(mesh_path: str, reg) -> tuple:
    """World-space bounding box of the registered mesh."""
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = v - v.mean(axis=0)                       # s5 centres the mesh too
    R = np.asarray(reg.rotation, dtype=np.float64)
    vw = (v * float(reg.scale)) @ R.T + np.asarray(reg.translation, dtype=np.float64)
    return vw.min(axis=0), vw.max(axis=0)


def _carve_object(particles: np.ndarray, mesh_path: str, reg, k: float,
                  spacing: float) -> tuple:
    """Drop liquid particles that start inside the object.
    """
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = v - v.mean(axis=0)
    R = np.asarray(reg.rotation, dtype=np.float64)
    vw = ((v * float(reg.scale)) @ R.T + np.asarray(reg.translation, dtype=np.float64)) * k
    world = trimesh.Trimesh(vertices=vw, faces=np.asarray(mesh.faces), process=False)
    pitch = max(spacing, float(np.max(world.extents)) / 64.0)
    try:
        vox = world.voxelized(pitch=pitch).fill()
        occupied = np.asarray(vox.points, dtype=np.float64)
    except Exception:
        return particles, 0, "object voxelisation failed; no particles carved"
    if len(occupied) == 0:
        return particles, 0, "object voxelisation was empty; no particles carved"
    from scipy.spatial import cKDTree
    # a particle is inside if it sits within half a voxel of an occupied cell
    hit = cKDTree(occupied).query(particles, k=1)[0] < 0.5 * pitch * np.sqrt(3.0)
    n = int(hit.sum())
    return particles[~hit], n, (f"carved {n} liquid particles out of the object's volume "
                                f"(voxel pitch {pitch:.4f} m)")


def sample_fluid_particles(xy_min, xy_max, depth, spacing, seed=0, z_floor=None,
                           footprint=None) -> np.ndarray:
    """Particles filling z in [-depth, 0] under the liquid's actual footprint.

    A plain rectangular slab is wrong: the liquid in the image has a silhouette
    of its own (a round pool, an irregular puddle, a container's cross-section)
    and filling the whole bounding rectangle invents fluid where there is none.
    `footprint(x, y) -> bool` is asked which columns are liquid; without it the
    slab falls back to the rectangle.
    """
    rng = np.random.default_rng(seed)
    lx = float(xy_max[0] - xy_min[0])
    ly = float(xy_max[1] - xy_min[1])
    nx = max(2, int(lx / spacing))
    ny = max(2, int(ly / spacing))
    nz = max(2, int(depth / spacing))
    xs = np.linspace(float(xy_min[0]), float(xy_max[0]), nx)
    ys = np.linspace(float(xy_min[1]), float(xy_max[1]), ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    columns = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if footprint is not None:
        keep = footprint(columns[:, 0], columns[:, 1])
        columns = columns[keep]
    if len(columns) == 0:
        columns = np.stack([gx.ravel(), gy.ravel()], axis=1)

    bottom = -abs(float(depth)) if z_floor is None else max(-abs(float(depth)), float(z_floor))
    zs = np.linspace(bottom, -spacing * 0.5, nz)
    grid = np.concatenate(
        [np.hstack([np.repeat(columns, 1, axis=0), np.full((len(columns), 1), z)])
         for z in zs], axis=0)
    grid = grid + rng.uniform(-0.25, 0.25, size=grid.shape) * spacing
    grid[:, 2] = np.minimum(grid[:, 2], -1e-4)     # never poke above the surface
    return grid.astype(np.float32)


def _extend_columns_under_object(xy: np.ndarray, tops: np.ndarray, cells: np.ndarray,
                                 spacing: float, support: np.ndarray, geo, T_world_cam):
    """Add liquid columns wherever the object hides the water.

    A column is added if its centre projects onto `support` (the fluid mask
    unioned with the object's) but no fluid pixel was ever seen there. Its
    height is copied from the nearest observed column, which is the only
    defensible guess: the point map at those pixels measures the *object*.
    """
    import cv2
    from conventions import Camera, invert_rigid, transform_points

    lo, hi = cells.min(axis=0), cells.max(axis=0)
    nx, ny = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1
    if nx * ny > 4_000_000:                       # nothing sane is this large
        return xy, tops

    seen = np.zeros((nx, ny), bool)
    zbuf = np.zeros((nx, ny), np.float32)
    idx = (cells - lo).astype(int)
    seen[idx[:, 0], idx[:, 1]] = True
    zbuf[idx[:, 0], idx[:, 1]] = tops

    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centres = np.column_stack([(gx.ravel() + lo[0] + 0.5) * spacing,
                               (gy.ravel() + lo[1] + 0.5) * spacing])

    cam = Camera(np.asarray(geo.K), geo.width, geo.height)
    T_cam_world = invert_rigid(np.asarray(T_world_cam))
    world = np.column_stack([centres, np.full(len(centres), float(np.median(tops)))])
    uv = cam.project(transform_points(T_cam_world, world))
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    on = (u >= 0) & (u < geo.width) & (v >= 0) & (v < geo.height)
    on[on] = support[v[on], u[on]]
    wanted = on.reshape(nx, ny)

    missing = wanted & ~seen
    if not missing.any():
        return xy, tops

    # nearest observed column's height, by distance transform on the empty cells
    _, labels = cv2.distanceTransformWithLabels((~seen).astype(np.uint8), cv2.DIST_L2, 3,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(seen)
    order = np.zeros(labels.max() + 1, np.int64)
    order[labels[seen]] = np.arange(len(ys))
    nearest = zbuf[ys[order[labels]], xs[order[labels]]]

    mi, mj = np.nonzero(missing)
    add_xy = np.column_stack([(mi + lo[0] + 0.5) * spacing, (mj + lo[1] + 0.5) * spacing])
    return (np.vstack([xy, add_xy]),
            np.concatenate([tops, nearest[mi, mj].astype(tops.dtype)]))


def fluid_from_pointmap(seg_json: str, geo, T_world_cam, spacing: float, depth: float,
                        seed: int = 0, metric_scale: float = 1.0):
    """Build the liquid from MoGe's measured surface instead of a flat slab.

    Every fluid-masked pixel gives a 3D point; those points *are* the observed
    liquid surface, ripples and all. Downsampling them onto a `spacing` grid in
    xy and extruding each column downwards keeps that shape rather than
    replacing it with a plane.

    Column heights are clipped to a robust band around the median. Monocular
    depth is noisy at the liquid's specular highlights and at its silhouette,
    and a handful of columns 20 cm above the rest is not a wave -- it is a bad
    pixel. Left in, those spikes make the liquid start far out of equilibrium
    and the first simulated frame is a collapse rather than the photograph.

    Returns (particles, surface_points) in world coordinates, or (None, None).
    """
    import cv2
    from contracts import SegmentationArtifact
    from conventions import transform_points

    seg = SegmentationArtifact.load(seg_json)
    pts = np.load(geo.points_path)
    ok = np.load(geo.valid_mask_path) & np.isfinite(pts).all(axis=-1)
    mask = cv2.imread(seg.fluid_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if mask.shape != pts.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (pts.shape[1], pts.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    # The object hides some of the liquid, and `binary_fill_holes` only closes
    # fully enclosed holes (not a notch where the object touches the mask's
    # boundary). The liquid continues under the object, so its support is the
    # union of the two masks.
    obj_mask = cv2.imread(seg.object_mask_path, cv2.IMREAD_GRAYSCALE)
    obj_mask = (obj_mask > 127) if obj_mask is not None else np.zeros_like(mask)
    if obj_mask.shape != mask.shape:
        obj_mask = cv2.resize(obj_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
    support = mask | obj_mask
    try:
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask)
        support = binary_fill_holes(support)
    except Exception:
        pass
    # heights may only come from pixels that really are liquid: the object's
    # pixels carry the object's depth, not the water line
    keep = mask & ok
    if keep.sum() < 200:
        return None, None

    surface = transform_points(np.asarray(T_world_cam), pts[keep] * float(metric_scale))
    # one column per xy cell, taking the median height of the points in it
    cell = np.floor(surface[:, :2] / spacing).astype(np.int64)
    order = np.lexsort((cell[:, 1], cell[:, 0]))
    surface, cell = surface[order], cell[order]
    _, starts = np.unique(cell, axis=0, return_index=True)
    starts = np.sort(starts)
    ends = np.append(starts[1:], len(surface))
    tops = np.array([np.median(surface[a:b, 2]) for a, b in zip(starts, ends)])
    xy = np.array([surface[a:b, :2].mean(axis=0) for a, b in zip(starts, ends)])

    med = float(np.median(tops))
    mad = 1.4826 * float(np.median(np.abs(tops - med)))
    band = max(3.0 * mad, 0.25 * spacing)
    tops = np.clip(tops, med - band, med + band)

    xy, tops = _extend_columns_under_object(xy, tops, cell[starts], spacing, support,
                                            geo, T_world_cam)

    rng = np.random.default_rng(seed)
    nz = max(2, int(round(depth / spacing)))
    cols = []
    for k in range(nz):
        z = tops - depth * (k / (nz - 1))
        cols.append(np.column_stack([xy, z]))
    particles = np.concatenate(cols, axis=0)
    particles += rng.uniform(-0.25, 0.25, size=particles.shape) * spacing
    return particles.astype(np.float32), surface.astype(np.float32)


def fluid_footprint(seg_json: str, geo, T_world_cam, fill_holes: bool = True):
    """Return `f(x, y) -> bool`: is this world column inside the liquid?

    Uses the calibrated camera to project each column back into the image and
    read the fluid mask, so the particle slab inherits the liquid's real
    outline instead of a bounding rectangle. Holes are filled first: an object
    resting in the middle punches a hole in the fluid mask, but the liquid
    obviously continues underneath it.
    """
    import cv2
    from contracts import SegmentationArtifact
    from conventions import Camera, invert_rigid, transform_points

    seg = SegmentationArtifact.load(seg_json)
    mask = cv2.imread(seg.fluid_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if mask.shape != (geo.height, geo.width):
        mask = cv2.resize(mask.astype(np.uint8), (geo.width, geo.height),
                          interpolation=cv2.INTER_NEAREST) > 0
    if fill_holes:
        try:
            from scipy.ndimage import binary_fill_holes
            mask = binary_fill_holes(mask)
        except Exception:
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                    np.ones((15, 15), np.uint8)) > 0

    cam = Camera(np.asarray(geo.K), geo.width, geo.height)
    T_cam_world = invert_rigid(np.asarray(T_world_cam))

    def inside(x, y):
        pts = np.stack([x, y, np.zeros_like(x)], axis=1)
        uv = cam.project(transform_points(T_cam_world, pts))
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < geo.width) & (v >= 0) & (v < geo.height)
        out = np.zeros(len(pts), dtype=bool)
        out[ok] = mask[v[ok], u[ok]]
        return out

    return inside


def _resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to `n` points spaced evenly along its length."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return poly
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def vessel_planes(footprint_xy: np.ndarray, z_floor: float, num_sides: int = 12,
                  margin: float = 0.0) -> dict:
    """The same container as `build_vessel`, but as half-spaces.

    A convex region is the intersection of half-spaces, so a ring of vertical
    planes tangent to the liquid's convex hull confines it exactly as a wall
    does -- and a plane has an analytic SDF, where a mesh needs a sampled one.

    An axis-aligned box would be cheaper still and is wrong: the liquid creeps
    into the corners and a round barrel slowly turns square. Tangent planes
    keep the observed outline.

    Normals point *into* the free region, matching Genesis' ground plane
    (`pos=(0,0,z), normal=(0,0,1)` leaves the world above it).
    """
    import cv2

    pts = np.ascontiguousarray(np.asarray(footprint_xy, dtype=np.float32).reshape(-1, 2))
    hull = cv2.convexHull(pts)[:, 0, :].astype(np.float64)
    walls = []
    for k in range(num_sides):
        a = 2.0 * np.pi * k / num_sides
        outward = np.array([np.cos(a), np.sin(a)])
        support = float((hull @ outward).max()) + float(margin)
        walls.append({"pos": [float(outward[0] * support), float(outward[1] * support),
                              float(z_floor)],
                      "normal": [float(-outward[0]), float(-outward[1]), 0.0]})
    return {"kind": "planes", "num_sides": int(num_sides), "floor_z": float(z_floor),
            "walls": walls, "margin": float(margin),
            "inner_radius_mean": float(np.linalg.norm(hull - hull.mean(axis=0), axis=1).mean())}


def build_vessel(footprint_xy: np.ndarray, z_bottom: float, z_floor: float, z_top: float,
                 thickness: float, out_path: Path, num_sides: int = 48,
                 inner_margin: float = 0.0):
    """A cup standing on the observed waterline: walls *and* a floor.

    Without one the liquid is a free slab on an infinite plane and spreads into
    a film. Whatever holds the liquid in the photograph -- a barrel, a bowl, a
    pool edge -- its silhouette is the fluid mask's boundary, so a wall erected
    there is the weakest assumption that still reproduces the image.

    The floor is needed because the scene's ground plane sits below the MPM
    solver's own cube boundary.

    Built as a closed solid rather than a zero-thickness shell: Genesis couples
    MPM to rigid bodies through a signed distance field, and an SDF is only
    meaningful for a solid with an inside.
    """
    import cv2
    import trimesh

    # cv2 rejects a non-contiguous slice of a float32 array outright
    pts = np.ascontiguousarray(np.asarray(footprint_xy, dtype=np.float32).reshape(-1, 2))
    hull = cv2.convexHull(pts)[:, 0, :].astype(np.float64)
    poly = _resample_closed(hull, num_sides)

    # Resampling the hull to a polygon cuts corners, so a few particles end up
    # a fraction of a millimetre inside the wall. Standing it back by
    # `inner_margin` keeps the liquid strictly in the pool.
    centre = poly.mean(axis=0)
    radius = np.linalg.norm(poly - centre, axis=1)
    grow = lambda d: centre + (poly - centre) * ((radius + d) / np.maximum(radius, 1e-9))[:, None]
    inner, outer = grow(inner_margin), grow(inner_margin + thickness)

    n = len(inner)
    z_floor = float(min(max(z_floor, z_bottom), z_top))
    verts = np.vstack([
        np.column_stack([outer, np.full(n, z_bottom)]),   # Ob:  0    .. n-1
        np.column_stack([outer, np.full(n, z_top)]),      # Ot:  n    .. 2n-1
        np.column_stack([inner, np.full(n, z_floor)]),    # If:  2n   .. 3n-1
        np.column_stack([inner, np.full(n, z_top)]),      # It:  3n   .. 4n-1
    ])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        Ob, Ob_, Ot, Ot_ = i, j, n + i, n + j
        If, If_, It, It_ = 2 * n + i, 2 * n + j, 3 * n + i, 3 * n + j
        for quad in ((Ob, Ot, Ot_, Ob_),    # outer wall
                     (If, If_, It_, It),    # inner wall
                     (It, It_, Ot_, Ot)):   # top rim
            a, b, c, dd = quad
            faces += [[a, b, c], [a, c, dd]]
    # the two discs that close the solid: the cup's underside and the pool floor
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])                       # outer disc @ z_bottom
        faces.append([2 * n, 2 * n + i, 2 * n + i + 1])   # inner disc @ z_floor

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    return {"path": str(out_path), "num_sides": n, "thickness": float(thickness),
            "z_bottom": float(z_bottom), "z_floor": z_floor, "z_top": float(z_top),
            "inner_radius_mean": float(radius.mean()),
            "centre": [float(centre[0]), float(centre[1])],
            "watertight": bool(mesh.is_watertight)}


def assemble(registration_json: str, fluid_json: str, geometry_json: str,
             object_mesh: str, out_dir: str,
             fluid_material: Optional[Dict[str, Any]] = None,
             object_material: Optional[Dict[str, Any]] = None,
             particle_spacing: Optional[float] = None,
             grid_density: Optional[int] = None, seed: int = 0,
             particles_across_object: int = 16, object_size_m: Optional[float] = None,
             fluid_ppc_axis: int = 2, max_fluid_particles: int = 250000,
             cells_across_object: float = 8.0, vessel: str = "planes",
             fluid_depth_factor: float = 0.75,
             fluid_depth_m: Optional[float] = None,
             sim_extent_objects: float = 8.0,
             max_grid_cells: int = 2_500_000,
             min_cells_across: int = 32) -> SceneArtifact:
    reg = RegistrationArtifact.load(registration_json)
    flu = FluidArtifact.load(fluid_json)
    geo = GeometryArtifact.load(geometry_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes = []

    # ---- absolute scale -------------------------------------------------
    verts_raw = _mesh_world_bounds(object_mesh, reg)
    obj_extent_raw = float(np.max(verts_raw[1] - verts_raw[0]))
    k = 1.0
    if object_size_m and object_size_m > 0 and obj_extent_raw > 1e-9:
        k = float(object_size_m) / obj_extent_raw
        notes.append(f"metric scale {k:.4f}: the registered object spans "
                     f"{obj_extent_raw:.3f} world units and is {object_size_m} m across, "
                     f"so the entire scene is rescaled by that ratio")

    # p_world_metric = R (k p_cam) + k t -- a rigid transform with a scaled
    # translation, applied to camera points that have themselves been scaled.
    T_world_cam = np.asarray(flu.T_world_cam, dtype=np.float64).copy()
    T_world_cam[:3, 3] *= k

    lx, ly, depth = [float(x) * k for x in flu.domain_size]
    obj_pos = np.asarray(reg.translation, dtype=np.float64) * k
    obj_scale = float(reg.scale) * k
    verts = (verts_raw[0] * k, verts_raw[1] * k)

    # Where the liquid actually is, in world coords
    fluid_pts_path = Path(fluid_json).parent / "fluid_points_world.npy"
    if fluid_pts_path.exists():
        fp = np.load(fluid_pts_path) * k
        xy_min = np.percentile(fp[:, :2], 1, axis=0)
        xy_max = np.percentile(fp[:, :2], 99, axis=0)
    else:
        xy_min, xy_max = np.array([-lx / 2, -ly / 2]), np.array([lx / 2, ly / 2])
        notes.append("no fluid point cloud found; assuming the slab is centred on the origin")

    # The object must be inside the domain or Genesis refuses to add it, and
    # the world origin is the camera axis' hit point on the plane, not the
    # centre of the scene -- so grow the box to cover both.
    xy_min = np.minimum(xy_min, verts[0][:2])
    xy_max = np.maximum(xy_max, verts[1][:2])

    # The walls should bound the *observed* scene, not just the liquid: MoGe's
    # point map is everything the camera can see, so its world-space extent is
    # the natural domain. Walls then sit at the edge of what was reconstructed
    # rather than cutting through the middle of the visible surface.
    from conventions import transform_points as _tp
    pm = np.load(geo.points_path)
    pm_ok = np.load(geo.valid_mask_path) & np.isfinite(pm).all(axis=-1)
    if pm_ok.sum() > 100:
        pm_world = _tp(T_world_cam, pm[pm_ok] * k)
        # ignore the far tail: monocular depth is unreliable at the horizon
        lo = np.percentile(pm_world[:, :2], 2, axis=0)
        hi = np.percentile(pm_world[:, :2], 98, axis=0)
        xy_min = np.minimum(xy_min, lo)
        xy_max = np.maximum(xy_max, hi)
        notes.append(f"domain grown to the observed point map: "
                     f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}]")

    pad = 0.10 * max(float(xy_max[0] - xy_min[0]), float(xy_max[1] - xy_min[1]))
    xy_min = xy_min - pad
    xy_max = xy_max + pad

    # The camera can see a whole river; we do not have to simulate one. Water
    # far from the object never interacts with it, so clip the simulated region
    obj_extent_m = float(np.max(verts[1] - verts[0]))
    half = 0.5 * sim_extent_objects * obj_extent_m
    centre = obj_pos[:2]
    clipped_min = np.maximum(xy_min, centre - half)
    clipped_max = np.minimum(xy_max, centre + half)
    if np.any(clipped_min > xy_min + 1e-9) or np.any(clipped_max < xy_max - 1e-9):
        notes.append(f"simulated region clipped from "
                     f"{np.round(xy_max - xy_min, 2).tolist()} m to "
                     f"{np.round(clipped_max - clipped_min, 2).tolist()} m: "
                     f"{sim_extent_objects:g} object-lengths around the object is as far as "
                     f"the interaction reaches")
        xy_min, xy_max = clipped_min, clipped_max
    obj_extent = float(np.max(verts[1] - verts[0]))
    particle_size = max(1e-3, obj_extent / float(particles_across_object))
    notes.append(f"object spans {obj_extent:.3f} m -> particle_size {particle_size:.4f} m")

    # A camera cannot see how deep an opaque liquid is, so the depth is a
    # modelling choice referenced to the object: it must be deep enough that the
    # object does not strike the bottom.
    depth_s4 = depth
    depth = fluid_depth_m if fluid_depth_m and fluid_depth_m > 0 else fluid_depth_factor * obj_extent
    notes.append(f"liquid depth {depth:.3f} m = {fluid_depth_factor:g} x the object's "
                 f"{obj_extent:.3f} m (stage 4's width heuristic said {depth_s4:.3f} m; "
                 f"the bottom is not observable in this image)")
    if grid_density is None:
        grid_density = DEFAULT_GRID_DENSITY
    elif int(grid_density) <= 0:
        grid_density = int(np.clip(round(cells_across_object / max(obj_extent, 1e-3)),
                                   64, 200))
    grid_density = int(grid_density)

    # The solver pads its domain by 3 cells on every side and the particles are
    # inset by 4, so a domain has to be many cells across or the padding eats
    # the scene. This is the floor; the cell cap below is the ceiling.
    smallest = float(min(xy_max[0] - xy_min[0], xy_max[1] - xy_min[1]))
    need = min_cells_across / max(smallest, 1e-9)
    if grid_density < need:
        finer = int(np.ceil(need))
        notes.append(f"grid_density {grid_density} -> {finer} cells/m: the domain's short "
                     f"side is {smallest:.3f} m, which at {grid_density} is only "
                     f"{smallest * grid_density:.0f} cells and the solver's padding would "
                     f"take most of it")
        grid_density = finer

    # MPM cost is the grid, and the grid is cells/m cubed times the volume. A
    # fixed cells/m is right for a fixed-size scene and ruinous for a big one,
    # so cap the total and coarsen if a scene blows past it.
    est_extent = np.array([xy_max[0] - xy_min[0], xy_max[1] - xy_min[1],
                           depth + max(float(verts[1][2]), 0.05)])
    cells = float(np.prod(est_extent * grid_density))
    if cells > max_grid_cells:
        coarser = int(max(16, grid_density * (max_grid_cells / cells) ** (1.0 / 3.0)))
        notes.append(f"grid_density {grid_density} -> {coarser} cells/m: at {grid_density} "
                     f"this {np.round(est_extent, 2).tolist()} m domain needs "
                     f"{cells / 1e6:.1f}M cells, over the {max_grid_cells / 1e6:.1f}M budget")
        grid_density = coarser
    notes.append(f"grid_density {grid_density} cells/m (dx={1.0 / grid_density:.4f} m): the "
                 f"object spans {obj_extent * grid_density:.1f} cells and the liquid is "
                 f"{depth * grid_density:.1f} cells deep")

    # The solver's own padding (4 cells, below) is subtracted from the domain,
    # so a floor placed exactly at -depth leaves the liquid only half as deep
    # as intended. Sink the box by that much and the liquid gets its full depth.
    floor_pad = 4.0 / float(grid_density)
    top = max(float(verts[1][2]) + 0.1 * max(lx, ly), 0.05)
    lower = np.array([xy_min[0], xy_min[1], -depth - floor_pad])
    upper = np.array([xy_max[0], xy_max[1], top])
    # a thin axis here breaks Genesis' CubeBoundary
    lower, upper, widened = require_mpm_domain(lower, upper, grid_density=grid_density)
    if any(widened):
        notes.append(f"widened thin simulation axes {widened} for the MPM boundary padding")

    # MPMSolver shrinks the usable region by 3*dx on every side, so particles
    # sampled right up to the domain edge are rejected as "outside the solver
    # boundary". Inset by a little more than that padding.
    inset = 4.0 / float(grid_density)
    p_min = lower + inset
    p_max = upper - inset
    # Tie the fluid resolution to the MPM grid, not to the scene's overall
    # size: with dx = 1/grid_density, standard MPM practice is 2 particles per
    dx = 1.0 / float(grid_density)
    spacing = particle_spacing or dx / float(max(1, fluid_ppc_axis))
    fp = None
    seg_json_p = Path(geometry_json).parent.parent / "s1_segment" / "segmentation.json"
    if seg_json_p.exists():
        # projection is scale-invariant, so the metric transform reads the same
        # pixels as the raw one
        fp = fluid_footprint(str(seg_json_p), geo, T_world_cam)
    slab_depth = min(depth, -float(p_min[2])) if p_min[2] < 0 else depth

    # Preferred: take the liquid's shape straight from the point map.
    particles, surface = (None, None)
    if seg_json_p.exists():
        particles, surface = fluid_from_pointmap(str(seg_json_p), geo, T_world_cam,
                                                 spacing, slab_depth, seed=seed,
                                                 metric_scale=k)
    if particles is not None:
        inside = ((particles[:, 0] > p_min[0]) & (particles[:, 0] < p_max[0]) &
                  (particles[:, 1] > p_min[1]) & (particles[:, 1] < p_max[1]) &
                  (particles[:, 2] > p_min[2]) & (particles[:, 2] < p_max[2]))
        particles = particles[inside]
        np.save(out / "fluid_surface_world.npy", surface)
        notes.append(f"fluid shape taken from the MoGe point map "
                     f"({len(surface)} surface points, raw z spread "
                     f"{surface[:, 2].min():.3f}..{surface[:, 2].max():.3f} m; column "
                     f"heights clipped to a robust band, giving particle tops "
                     f"{particles[:, 2].max() - slab_depth:.3f}..{particles[:, 2].max():.3f} m)")
    else:
        particles = sample_fluid_particles(p_min[:2], p_max[:2], slab_depth,
                                           spacing, seed=seed, z_floor=float(p_min[2]),
                                           footprint=fp)
        notes.append("point map unavailable; fell back to a flat slab")
    # Coarsen rather than run out of memory if the footprint is large. Resample
    # through whichever sampler produced the particles: rebuilding a shaped
    # liquid as a rectangular slab just to thin it out would throw away the
    # shape this stage exists to preserve.
    from_pointmap = surface is not None
    while len(particles) > max_fluid_particles:
        spacing *= 1.26                      # ~2x fewer particles each pass
        if from_pointmap:
            particles, _ = fluid_from_pointmap(str(seg_json_p), geo, T_world_cam,
                                               spacing, slab_depth, seed=seed,
                                               metric_scale=k)
            particles = particles[(particles[:, 0] > p_min[0]) & (particles[:, 0] < p_max[0]) &
                                  (particles[:, 1] > p_min[1]) & (particles[:, 1] < p_max[1]) &
                                  (particles[:, 2] > p_min[2]) & (particles[:, 2] < p_max[2])]
        else:
            particles = sample_fluid_particles(p_min[:2], p_max[:2], slab_depth,
                                               spacing, seed=seed, z_floor=float(p_min[2]),
                                               footprint=fp)
        notes.append(f"coarsened fluid spacing to {spacing:.4f} m to stay under "
                     f"{max_fluid_particles} particles")
    # Spacing, `particle_size` and `grid_density` have to stay consistent, and
    # the coarsening loop above breaks that: it widens the spacing to fit the
    # particle budget, and if `particle_size` keeps its old value the solver
    # charges each particle `particle_size**3` of mass while they actually sit
    # `spacing` apart, which would make the liquid too light. A spacing wider
    # than dx/ppc also means under one particle per cell, so the grid has to
    # come down with it, which moves the solver padding and the domain.
    if spacing > dx / float(max(1, fluid_ppc_axis)) * 1.01:
        grid_density = max(16, int(round(1.0 / (spacing * max(1, fluid_ppc_axis)))))
        dx = 1.0 / float(grid_density)
        inset = 4.0 / float(grid_density)
        lower, upper, widened = require_mpm_domain(
            [xy_min[0], xy_min[1], -depth - inset], [xy_max[0], xy_max[1], top],
            grid_density=grid_density)
        p_min, p_max = lower + inset, upper - inset
        particles = particles[np.all((particles > p_min) & (particles < p_max), axis=1)]
        notes.append(f"grid_density -> {grid_density} cells/m (dx={dx:.4f}) so the grid "
                     f"matches the {spacing:.4f} m spacing the particle budget forced")
    if abs(particle_size - spacing) > 1e-9:
        notes.append(f"particle_size {particle_size:.4f} -> {spacing:.4f}: it sets each "
                     f"particle's volume, so it must equal the spacing they were laid out at")
        particle_size = spacing

    particles, n_carved, carve_note = _carve_object(particles, str(object_mesh), reg, k, spacing)
    notes.append(carve_note)

    particles_path = out / "fluid_particles.npy"
    np.save(particles_path, particles)
    notes.append(f"{len(particles)} fluid particles at {spacing:.4f} m spacing "
                 f"(dx={dx:.4f}, {fluid_ppc_axis} per cell per axis), inset "
                 f"{inset:.3f} m inside the domain for the MPM padding"
                 + (", footprint taken from the fluid mask" if fp is not None
                    else ", rectangular footprint (no segmentation found)"))

    # A wall on the waterline; without it the liquid is a free slab and slumps
    # out to the domain edge over the first hundred frames.
    vessel_path, vessel_info = "", {}
    mode = str(vessel).lower() if not isinstance(vessel, bool) else ("planes" if vessel else "none")
    if mode != "none" and len(particles) > 100:
        thickness = max(3.0 * dx, 0.02 * float(np.ptp(particles[:, 0]) + np.ptp(particles[:, 1])))
        diameter = float(max(np.ptp(particles[:, 0]), np.ptp(particles[:, 1])))
        z_top = float(particles[:, 2].max()) + max(1.5 * slab_depth, 0.15 * diameter)
        z_top = min(z_top, float(p_max[2]))
        try:
            if mode == "planes":
                vessel_info = vessel_planes(particles[:, :2], float(p_min[2]),
                                            margin=spacing)
                notes.append(f"vessel as {vessel_info['num_sides']} tangent planes on the "
                             f"observed waterline (mean inner radius "
                             f"{vessel_info['inner_radius_mean']:.3f} m) with the floor at "
                             f"z={vessel_info['floor_z']:.3f} m")
            else:
                # the cup's underside goes down to the domain floor so nothing
                # can squeeze under it; its *inner* floor is the liquid's bottom
                vessel_info = build_vessel(particles[:, :2], float(lower[2]), float(p_min[2]),
                                           z_top, thickness, out / "vessel.obj",
                                           inner_margin=spacing)
                vessel_info["kind"] = "mesh"
                vessel_path = vessel_info["path"]
                notes.append(f"vessel cup on the observed waterline: mean inner radius "
                             f"{vessel_info['inner_radius_mean']:.3f} m, {thickness:.3f} m thick, "
                             f"floor at z={vessel_info['z_floor']:.3f} m, rim at z={z_top:.3f} m, "
                             f"watertight={vessel_info['watertight']}")
        except Exception as exc:                      # keep the scene usable
            notes.append(f"vessel construction failed ({exc}); the liquid is unconfined")

    # Background point cloud: everything that is neither object nor fluid,
    # lifted into world coords. Genesis does not simulate it, but it is what
    # makes the exported scene readable and gives walls/table context.
    bg_path = ""
    seg_json = Path(geometry_json).parent.parent / "s1_segment" / "segmentation.json"
    if seg_json.exists():
        import cv2
        from contracts import SegmentationArtifact
        from conventions import transform_points
        seg = SegmentationArtifact.load(seg_json)
        pts = np.load(geo.points_path)
        okm = np.load(geo.valid_mask_path)
        bg = cv2.imread(seg.background_mask_path, cv2.IMREAD_GRAYSCALE) > 127
        rgb_img = cv2.imread(geo.image_path or seg.image_path)
        if bg.shape != pts.shape[:2]:
            bg = cv2.resize(bg.astype(np.uint8), (pts.shape[1], pts.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
        keep = bg & okm & np.isfinite(pts).all(axis=-1)
        if keep.sum() > 0:
            world = transform_points(T_world_cam, pts[keep] * k)
            if rgb_img is not None and rgb_img.shape[:2] == pts.shape[:2]:
                cols = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)[keep]
            else:
                cols = np.tile([150, 150, 150], (len(world), 1))
            bg_path = str(out / "background_points.npy")
            np.save(bg_path, np.hstack([world, cols]).astype(np.float32))
            notes.append(f"{len(world)} background points saved for visualisation")

    # `Camera.T_world_cam` is camera -> world, which is what stage 4 produced
    # (its translation column is the camera's position in the world).
    cam = Camera(np.array(geo.K), geo.width, geo.height, T_world_cam=T_world_cam)

    # Density is a fixed default, not inferred from how much of the object sits
    # below the waterline.
    #
    # Archimedes only holds for a body held up by buoyancy alone, and the
    om = dict(object_material or DEFAULT_OBJECT)
    if "rho" not in om:
        om["rho"] = float(DEFAULT_OBJECT["rho"])
    notes.append(f"object density {om['rho']} kg/m3 (fixed default; not inferred from the "
                 f"{reg.submerged_fraction:.0%} submerged fraction, which is unobservable "
                 f"along the view axis)")

    scene = SceneArtifact(
        object_mesh_path=str(object_mesh),
        object_pos=[float(x) for x in obj_pos],
        object_quat_wxyz=[float(x) for x in reg.quat_wxyz],
        object_scale=float(obj_scale),
        object_material=om,
        fluid_particles_path=str(particles_path),
        fluid_material=dict(fluid_material or DEFAULT_FLUID),
        domain_lower=lower.tolist(), domain_upper=upper.tolist(),
        gravity=[0.0, 0.0, -GRAVITY],
        camera=cam.to_dict(),
        interaction_state=reg.interaction_state,
        background_points_path=bg_path,
        particle_size=float(particle_size),
        vessel_mesh_path=vessel_path,
        vessel=vessel_info,
        grid_density=int(grid_density),
        metric_scale=float(k),
        notes=notes)
    scene.save(out / "scene.json")
    return scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration", required=True)
    p.add_argument("--fluid", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--object_mesh", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fluid_material_json", default="")
    p.add_argument("--object_material_json", default="")
    p.add_argument("--object_size_m", type=float, default=None,
                   help="Real size of the object in metres; sets the scene's absolute scale.")
    p.add_argument("--grid_density", type=int, default=None,
                   help=f"MPM cells per metre (default {DEFAULT_GRID_DENSITY}). "
                        "Pass 0 to size it from the object instead. Cost goes as dx^-3.")
    p.add_argument("--fluid_depth_factor", type=float, default=0.75,
                   help="Liquid depth as a multiple of the object's size. The depth of an "
                        "opaque liquid is not observable; this is the modelling choice.")
    p.add_argument("--fluid_depth_m", type=float, default=None,
                   help="Liquid depth in metres, overriding --fluid_depth_factor.")
    p.add_argument("--sim_extent_objects", type=float, default=8.0,
                   help="Simulate at most this many object-lengths around the object; the "
                        "rest of a river or lake never touches it.")
    p.add_argument("--max_grid_cells", type=int, default=2_500_000,
                   help="Coarsen grid_density rather than exceed this many MPM cells.")
    p.add_argument("--min_cells_across", type=int, default=32,
                   help="Refine grid_density so the domain's short side spans at least this "
                        "many cells; the solver's padding is measured in cells.")
    p.add_argument("--vessel", choices=("planes", "mesh", "none"), default="planes",
                   help="How to confine the liquid: tangent half-spaces (cheap, default), "
                        "a cup mesh (a real rim, ~5x slower), or nothing.")
    a = p.parse_args()
    fm = json.loads(Path(a.fluid_material_json).read_text()) if a.fluid_material_json else None
    om = json.loads(Path(a.object_material_json).read_text()) if a.object_material_json else None
    s = assemble(a.registration, a.fluid, a.geometry, a.object_mesh, a.out_dir, fm, om,
                 grid_density=a.grid_density, object_size_m=a.object_size_m,
                 vessel=a.vessel, fluid_depth_factor=a.fluid_depth_factor,
                 fluid_depth_m=a.fluid_depth_m, sim_extent_objects=a.sim_extent_objects,
                 max_grid_cells=a.max_grid_cells, min_cells_across=a.min_cells_across)
    print(json.dumps({"state": s.interaction_state, "gravity": s.gravity,
                      "domain": [np.round(s.domain_lower, 3).tolist(),
                                 np.round(s.domain_upper, 3).tolist()],
                      "fluid": s.fluid_material, "notes": s.notes}, indent=2))


if __name__ == "__main__":
    main()
