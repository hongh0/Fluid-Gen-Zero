"""Dump the assembled scene as PLY files you can open in MeshLab.

Everything is written in the simulation world frame (fluid surface at z = 0,
+z up), so what you see in MeshLab is literally what Genesis is given:

    scene.ply              object + fluid + background, colour-coded
    object_mesh_world.ply  the registered mesh alone, as a mesh
    fluid_particles.ply    the MPM particles alone
    background.ply         the static background point cloud

Colours: object red, fluid blue, background grey.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from contracts import SceneArtifact


def _write_ply_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())


def object_mesh_world(scene: SceneArtifact):
    """The registered mesh, moved into world coordinates."""
    import trimesh
    from conventions import R_from_quat_wxyz

    mesh = trimesh.load(scene.object_mesh_path, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = v - v.mean(axis=0)          # registration works on the centred mesh
    R = R_from_quat_wxyz(np.asarray(scene.object_quat_wxyz))
    v = (v * float(scene.object_scale)) @ R.T + np.asarray(scene.object_pos)
    return trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)


def export(scene_json: str, out_dir: Optional[str] = None,
           object_samples: int = 60000) -> dict:
    scene = SceneArtifact.load(scene_json)
    out = Path(out_dir or Path(scene_json).parent)
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    mesh = object_mesh_world(scene)
    mesh.export(str(out / "object_mesh_world.ply"))
    written["object_mesh_world.ply"] = len(mesh.vertices)

    obj_pts, _ = __import__("trimesh").sample.sample_surface(mesh, object_samples)
    obj_pts = np.asarray(obj_pts)
    _write_ply_points(out / "object_points.ply", obj_pts,
                      np.tile([220, 70, 60], (len(obj_pts), 1)))
    written["object_points.ply"] = len(obj_pts)

    fluid = np.load(scene.fluid_particles_path).astype(np.float32)
    _write_ply_points(out / "fluid_particles.ply", fluid,
                      np.tile([60, 110, 230], (len(fluid), 1)))
    written["fluid_particles.ply"] = len(fluid)

    parts = [(obj_pts, np.tile([220, 70, 60], (len(obj_pts), 1))),
             (fluid, np.tile([60, 110, 230], (len(fluid), 1)))]

    # the wall that keeps the liquid in; it is simulated, so it belongs in the
    # scene dump rather than being left as an invisible surprise
    vpath = getattr(scene, "vessel_mesh_path", "")
    if vpath and Path(vpath).exists():
        import trimesh
        ves = trimesh.load(vpath, force="mesh")
        ves.export(str(out / "vessel.ply"))
        written["vessel.ply"] = len(ves.vertices)
        vp, _ = trimesh.sample.sample_surface(ves, 40000)
        vp = np.asarray(vp)
        parts.append((vp, np.tile([120, 90, 60], (len(vp), 1))))

    bg_path = getattr(scene, "background_points_path", "")
    if bg_path and Path(bg_path).exists():
        bg = np.load(bg_path)
        xyz, rgb = (bg[:, :3], bg[:, 3:6].astype(np.uint8)) if bg.shape[1] >= 6 else \
                   (bg[:, :3], np.tile([150, 150, 150], (len(bg), 1)))
        _write_ply_points(out / "background.ply", xyz, rgb)
        written["background.ply"] = len(xyz)
        parts.append((xyz, rgb))

    allp = np.concatenate([p for p, _ in parts], axis=0)
    allc = np.concatenate([c for _, c in parts], axis=0)
    _write_ply_points(out / "scene.ply", allp, allc)
    written["scene.ply"] = len(allp)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out_dir", default="")
    a = p.parse_args()
    for name, n in export(a.scene, a.out_dir or None).items():
        print(f"  {name:26s} {n:>9,} points/verts")


if __name__ == "__main__":
    main()
