"""Build a Genesis scene from a reconstruction `scene.json`.

The reconstruction world is already Genesis' world: z up, fluid surface at
z = 0, gravity (0, 0, -9.81). So nothing is transformed here -- positions go in
as they come out of registration. Two details are handled explicitly:

  * the object is a **mesh**, not a point cloud, so it keeps the geometry that
    is hidden from the camera;
  * orientation is passed as a quaternion, never as `euler` (Genesis' euler is
    degrees in scipy's extrinsic x-y-z order, which is easy to get wrong).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

RECON = str(Path(__file__).resolve().parent.parent / "reconstruction")
if RECON not in sys.path:
    sys.path.insert(0, RECON)

from contracts import SceneArtifact, require_mpm_domain  # noqa: E402
from conventions import Camera, genesis_camera  # noqa: E402


#: every deformable Genesis exposes, keyed by the string the intent parser
#: writes into `object_material["type"]`. Anything here is a drop-in for
#: "rigid"; the rest of the pipeline branches only on `kind == "rigid"`.
MPM_MATERIALS = ("elastic", "elastoplastic", "sand", "snow", "muscle", "liquid")

#: Young's modulus above which Genesis' MPM solids freeze instead of deforming.
MPM_E_CEILING = 5e6


def object_material(kind: str, om: dict):
    """Build the Genesis material for `object_material["type"]`.

    Unknown names fail loudly rather than silently falling back to elastic: a
    scene that asked for sand and quietly got rubber is worse than a crash.
    """
    import genesis as gs

    def f(key, default):
        return float(om.get(key, default))

    if kind == "rigid":
        return gs.materials.Rigid(rho=f("rho", 700.0),
                                  friction=om.get("friction", None),
                                  coup_friction=f("coup_friction", 0.1))
    E = f("E", 2e6)
    if E > MPM_E_CEILING:
        print(f"  [material] E={E:.3g} Pa exceeds the workable MPM ceiling "
              f"{MPM_E_CEILING:.3g}; clamping. Anything this stiff is better "
              f"modelled as type='rigid'.", flush=True)
        E = MPM_E_CEILING
    common = dict(E=E, nu=f("nu", 0.27), rho=f("rho", 700.0))
    if kind == "elastic":
        return gs.materials.MPM.Elastic(model=str(om.get("model", "corotation")), **common)
    if kind == "elastoplastic":
        return gs.materials.MPM.ElastoPlastic(
            yield_lower=f("yield_lower", 2.5e-2), yield_higher=f("yield_higher", 4.5e-3),
            use_von_mises=bool(om.get("use_von_mises", True)),
            von_mises_yield_stress=f("von_mises_yield_stress", 1e4), **common)
    if kind == "sand":
        return gs.materials.MPM.Sand(friction_angle=f("friction_angle", 45.0), **common)
    if kind == "snow":
        return gs.materials.MPM.Snow(yield_lower=f("yield_lower", 2.5e-2),
                                     yield_higher=f("yield_higher", 4.5e-3), **common)
    if kind == "muscle":
        return gs.materials.MPM.Muscle(model=str(om.get("model", "neohooken")),
                                       n_groups=int(om.get("n_groups", 1)), **common)
    if kind == "liquid":
        return gs.materials.MPM.Liquid(viscous=bool(om.get("viscous", False)), **common)
    raise ValueError(f"unknown object material type {kind!r}; "
                     f"expected 'rigid' or one of {MPM_MATERIALS}")


def object_particles(art, particle_size: float) -> np.ndarray:
    """Fill the registered mesh with particles, in world coordinates.

    `trimesh.sample.volume_mesh` needs a watertight `contains`, which SAM 3D
    Objects' output is not. A surface voxelisation followed by a per-slice fill
    does not, and it degrades gracefully: if the fill fails the surface voxels
    alone still describe the object's shape.
    """
    import trimesh
    from conventions import R_from_quat_wxyz

    mesh = trimesh.load(art.object_mesh_path, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = v - v.mean(axis=0)                      # registration works on the centred mesh
    R = R_from_quat_wxyz(np.asarray(art.object_quat_wxyz, dtype=np.float64))
    v = (v * float(art.object_scale)) @ R.T + np.asarray(art.object_pos, dtype=np.float64)
    world = trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)

    vox = world.voxelized(pitch=float(particle_size))
    try:
        vox = vox.fill()
    except Exception:
        pass
    pts = np.asarray(vox.points, dtype=np.float32)
    if len(pts) < 64:                           # pitch too coarse for a small object
        pts = np.asarray(trimesh.sample.sample_surface(world, 4096)[0], dtype=np.float32)
    return pts


def required_substeps(dt: float, dx: float, media, safety: float = 0.4) -> int:
    """Substeps needed for MPM's explicit integrator to stay stable.

    A substep must be shorter than the time an elastic wave takes to cross one
    grid cell: `dt/substeps <= safety * dx / sqrt(E/rho)`. It matters as soon as
    the object stops being rigid -- the default elastic solid (E = 5e7,
    rho = 700) has a wave speed of 267 m/s, which crosses a 1 cm cell in 37 us,
    against 33 us per substep at 30 substeps.
    """
    import math
    need = 1
    for E, rho in media:
        c = math.sqrt(max(float(E), 1.0) / max(float(rho), 1.0))
        need = max(need, int(math.ceil(dt * c / max(safety * dx, 1e-12))))
    return need


def build_scene(scene_json: str, dt: float = 1e-3, substeps: int = 30,
                grid_density: int = None, particle_size: float = None,
                show_viewer: bool = False, render_res: Tuple[int, int] = None,
                substeps_override: int = 0, fluid_vis: str = "recon",
                particle_size_scale: float = 1.0):
    """Returns (scene, entities, camera, artifact) without stepping."""
    import genesis as gs

    art = SceneArtifact.load(scene_json)
    if particle_size is None:
        particle_size = float(getattr(art, "particle_size", 0.01) or 0.01)
    if grid_density is None:
        # cells per metre, chosen in stage 6 from the object's real size
        grid_density = int(getattr(art, "grid_density", 64) or 64)
    lower, upper, widened = require_mpm_domain(art.domain_lower, art.domain_upper,
                                               grid_density=grid_density)

    om, fm = art.object_material, art.fluid_material
    kind = str(om.get("type", "rigid")).lower()
    media = [(fm.get("E", 1e6), fm.get("rho", 1000.0))]
    if kind != "rigid":
        media.append((min(float(om.get("E", 2e6)), MPM_E_CEILING), om.get("rho", 700.0)))
    substeps_cfl = required_substeps(dt, 1.0 / grid_density, media)
    substeps = max(substeps_override or substeps, substeps_cfl)
    if substeps_override and substeps > substeps_override:
        print(f"  [substeps] {substeps_override} raised to {substeps}: a substep of "
              f"{dt / substeps_override * 1e3:.3f} ms exceeds the CFL limit for this grid "
              f"and material", flush=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps,
                                          gravity=tuple(float(x) for x in art.gravity)),
        coupler_options=gs.options.LegacyCouplerOptions(),
        mpm_options=gs.options.MPMOptions(lower_bound=tuple(lower), upper_bound=tuple(upper),
                                          grid_density=grid_density,
                                          particle_size=particle_size),
        rigid_options=gs.options.RigidOptions(),
        # `particle_size_scale` shrinks the drawn spheres only; `particle_size`
        # itself is physics (`_particle_volume_real = particle_size**3`), so it
        # must never be tuned for looks.
        vis_options=gs.options.VisOptions(show_world_frame=False,
                                          particle_size_scale=float(particle_size_scale)),
        show_viewer=show_viewer,
    )

    material = object_material(kind, om)
    if kind == "rigid":
        # A rigid body keeps the mesh itself and couples to the liquid through
        # its SDF (`needs_coup` is on by default).
        morph = gs.morphs.Mesh(file=art.object_mesh_path,
                               pos=tuple(float(x) for x in art.object_pos),
                               quat=tuple(float(x) for x in art.object_quat_wxyz),
                               scale=float(art.object_scale),
                               convexify=bool(om.get("convexify", False)),
                               decompose_nonconvex=bool(om.get("decompose_nonconvex", False)))
        vis_mode = "visual"
        obj_particles = None
    else:
        obj_particles = object_particles(art, particle_size)
        morph = gs.morphs.PointCloud(points=obj_particles, pos=(0.0, 0.0, 0.0),
                                     euler=(0.0, 0.0, 0.0), scale=1.0)
        vis_mode = "particle"
    obj = scene.add_entity(material=material, morph=morph,
                           surface=gs.surfaces.Default(color=(0.85, 0.4, 0.35),
                                                       vis_mode=vis_mode))

    particles = np.load(art.fluid_particles_path).astype(np.float32)
    fluid = scene.add_entity(
        material=gs.materials.MPM.Liquid(rho=float(fm.get("rho", 1000.0)),
                                         E=float(fm.get("E", 1e6)),
                                         viscous=bool(fm.get("viscous", False))),
        morph=gs.morphs.PointCloud(points=particles, pos=(0.0, 0.0, 0.0),
                                   euler=(0.0, 0.0, 0.0), scale=1.0),
        # "particle" draws every MPM particle as a sphere; "recon" reconstructs
        # the liquid surface each frame via splashsurf (slightly slower). The
        # conditioning outputs come from render_conditions.py either way.
        surface=gs.surfaces.Water(color=(0.3, 0.4, 0.9), vis_mode=fluid_vis),
    )
    vessel, vessel_kind = None, "none"
    vspec = dict(getattr(art, "vessel", {}) or {})
    vpath = str(getattr(art, "vessel_mesh_path", "") or "")

    def _static(morph_kwargs, plane=False):
        """A fixed, invisible collider -- the near wall must not hide the liquid."""
        cls = gs.morphs.Plane if plane else gs.morphs.Mesh
        try:
            m = cls(visualization=False, **morph_kwargs)
        except TypeError:
            m = cls(**morph_kwargs)
        return scene.add_entity(material=gs.materials.Rigid(rho=2000.0, coup_friction=0.2),
                                morph=m,
                                surface=gs.surfaces.Default(color=(0.45, 0.35, 0.25),
                                                            vis_mode="visual"))

    if vspec.get("kind") == "planes" and vspec.get("walls"):
        for w in vspec["walls"]:
            vessel = _static(dict(pos=tuple(float(x) for x in w["pos"]),
                                  normal=tuple(float(x) for x in w["normal"])), plane=True)
        vessel = _static(dict(pos=(0.0, 0.0, float(vspec["floor_z"])), normal=(0, 0, 1)),
                         plane=True)
        vessel_kind = f"planes x{len(vspec['walls'])}+floor"
    elif vpath and Path(vpath).exists():
        vessel = _static(dict(file=vpath, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                              scale=1.0, fixed=True, convexify=False, collision=True))
        vessel_kind = "mesh"

    # Ground at the bottom of the domain, as a backstop under everything.
    scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, float(lower[2])), normal=(0, 0, 1)),
                     surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5), vis_mode="visual"))
    # Walls at the domain edge, but only when nothing else confines the liquid.
    # Every rigid geom is tested against every active MPM grid node each substep,
    # so four redundant half-spaces are four times that cost for nothing.
    for axis, sign in (() if vessel is not None else
                       ((0, -1), (0, 1), (1, -1), (1, 1))):
        pos = [0.0, 0.0, 0.0]
        pos[axis] = float(lower[axis] if sign < 0 else upper[axis])
        normal = [0.0, 0.0, 0.0]
        normal[axis] = -sign
        scene.add_entity(morph=gs.morphs.Plane(pos=tuple(pos), normal=tuple(normal)),
                         surface=gs.surfaces.Default(color=(0.4, 0.4, 0.4), vis_mode="visual"))

    cam_spec = genesis_camera(Camera.from_dict(art.camera))
    if render_res:
        cam_spec["res"] = tuple(int(x) for x in render_res)
    camera = scene.add_camera(res=cam_spec["res"], pos=cam_spec["pos"],
                              lookat=cam_spec["lookat"], up=cam_spec["up"],
                              fov=cam_spec["fov"], GUI=False)

    info = {"object_kind": kind, "widened_axes": widened, "domain": (lower.tolist(), upper.tolist()),
            "camera": cam_spec, "num_fluid_particles": int(len(particles)),
            "grid_density": int(grid_density), "particle_size": float(particle_size),
            "vessel": vessel_kind,
            "substeps": int(substeps), "substeps_cfl": int(substeps_cfl),
            "num_object_particles": (0 if obj_particles is None else int(len(obj_particles)))}
    return scene, {"object": obj, "fluid": fluid, "vessel": vessel}, camera, art, info
