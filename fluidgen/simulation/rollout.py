"""Roll a reconstructed scene forward in Genesis and record the trajectories.

Outputs `fg_traj.npz`, `fluid_traj.npz` and `simulation.mp4`, rendered with the
*calibrated* camera so the render lines up with the input image.

Frame 0 is rendered before any stepping, which is what the alignment check
compares against the input image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reconstruction"))
from scene_builder import build_scene  # noqa: E402


def screen_right_in_world(art) -> np.ndarray:
    """The camera's "right" direction, flattened onto the fluid surface.

    The world x/y axes are an arbitrary in-plane basis picked when the plane
    was fitted, so "left" and "right" only mean something relative to the
    camera. Projecting the camera's +x onto z = 0 gives the direction that
    actually reads as rightwards on screen.
    """
    from conventions import Camera
    R = Camera.from_dict(art.camera).T_world_cam[:3, :3]
    right = np.asarray(R[:, 0], dtype=np.float64).copy()
    right[2] = 0.0
    n = np.linalg.norm(right)
    return right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def screen_back_in_world(art) -> np.ndarray:
    """The direction pointing away from the camera, flattened onto the surface.

    Camera frame is OpenCV, so column 2 of the camera-to-world rotation looks
    into the scene. Dropping z and renormalising gives the in-plane direction
    that reads as "backward" on screen regardless of camera tilt.
    """
    from conventions import Camera
    R = Camera.from_dict(art.camera).T_world_cam[:3, :3]
    back = np.asarray(R[:, 2], dtype=np.float64).copy()
    back[2] = 0.0
    n = np.linalg.norm(back)
    return back / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def random_tangent(rng, prev=None, min_turn_dot: float = 0.85) -> np.ndarray:
    """A random horizontal direction, meaningfully different from `prev`.

    The fluid surface is z = 0, so "tangent" is just the xy plane. Rejecting
    candidates within `min_turn_dot` of the previous direction (|cos| >= 0.85,
    i.e. under ~32 degrees, either sign) keeps a re-randomisation from handing
    back nearly the same push.
    """
    for _ in range(16):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cand = np.array([np.cos(theta), np.sin(theta), 0.0])
        if prev is None or abs(float(cand @ prev)) < min_turn_dot:
            return cand
    return -prev if prev is not None else np.array([1.0, 0.0, 0.0])


def rollout_with_retry(*args, max_attempts: int = 3, **kwargs):
    """Run a rollout, halving the substep until the rigid solver stops diverging.

    Genesis' rigid solver raises `Invalid accelerations causing 'nan'` /
    `Invalid constraint forces causing 'nan'` and asks, in the message itself,
    for a smaller timestep. The CFL guard in `scene_builder` cannot predict
    this: it sizes substeps from the MPM wave speed, and contact-constraint
    stiffness is not in that formula; small, thin objects are the usual case.
    The value that worked is recorded in `rollout.json`.
    """
    import genesis as gs
    substeps = int(kwargs.pop("substeps", 0)) or 0
    last = None
    for attempt in range(max_attempts):
        try:
            return rollout(*args, substeps=substeps, **kwargs)
        except gs.GenesisException as exc:
            if "nan" not in str(exc).lower():
                raise
            last = exc
            base = substeps or 30
            substeps = base * 2
            try:
                gs.destroy()
            except Exception as cleanup:                 # pragma: no cover
                print(f"  [rigid] gs.destroy() failed: {cleanup}", flush=True)
            print(f"  [rigid] {str(exc).strip()[:90]}\n  retrying with substeps={substeps} "
                  f"(attempt {attempt + 2}/{max_attempts})", flush=True)
    raise last


def rollout(scene_json: str, out_dir: str, num_frames: int = 401,
            substeps_per_frame: int = 10, dt: float = 1e-3,
            grid_density: int = None, seed: int = 0, fps: int = 30,
            substeps: int = 0, fluid_vis: str = "recon",
            particle_size_scale: float = 1.0,
            force_accel: float = 5.0, force_mode: str = "random",
            force_seed: int = 0, stuck_eps: float = 2e-4,
            stuck_patience: int = 40,
            push_switch: float = 0.5,
            force_deg: float = 0.0, force_ref: str = "right",
            fluid_mode: str = "none", fluid_accel: float = 0.0,
            fluid_deg: float = 0.0) -> dict:
    import genesis as gs

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gs.init(seed=seed, precision="32", logging_level="warning")

    scene, ents, camera, art, info = build_scene(scene_json, dt=dt,
                                                 grid_density=grid_density,
                                                 substeps_override=substeps,
                                                 fluid_vis=fluid_vis,
                                                 particle_size_scale=particle_size_scale)
    scene.build()

    obj, fluid = ents["object"], ents["fluid"]
    traj_obj, traj_fluid = [], []

    # Rigid entities expose a pose rather than particles. Carrying a fixed set
    # of mesh vertices through that pose keeps `fg_traj.npz` in the same
    # (T, N, 3) form as the fluid particles.
    rigid = info.get("object_kind", "rigid") == "rigid"
    if rigid:
        import trimesh
        from conventions import R_from_quat_wxyz
        m = trimesh.load(art.object_mesh_path, force="mesh")
        v0 = np.asarray(m.vertices, dtype=np.float64)
        v0 = v0 - v0.mean(axis=0)
        # enough vertices that the projected set tiles the silhouette
        if len(v0) > 6000:
            v0 = v0[np.linspace(0, len(v0) - 1, 6000).astype(int)]
        v0 = v0 * float(art.object_scale)

    def object_points():
        if not rigid:
            return obj.get_particles_pos().cpu().numpy().astype(np.float32)
        pos = np.asarray(obj.get_pos().cpu().numpy(), dtype=np.float64).reshape(3)
        quat = np.asarray(obj.get_quat().cpu().numpy(), dtype=np.float64).reshape(4)
        return (v0 @ R_from_quat_wxyz(quat).T + pos).astype(np.float32)

    # Lateral push: fixed magnitude, random direction in the fluid plane,
    # re-drawn whenever the object stops making progress.
    #
    # A rigid body is driven through its dofs; an MPM body has none, so its
    # particle velocities are nudged instead (`vel += accel * dt`).
    right = screen_right_in_world(art)
    back = screen_back_in_world(art)

    def dir_from_deg(deg: float) -> np.ndarray:
        """Horizontal direction `deg` degrees from the chosen reference axis.

        With force_ref="right" the basis is (right, z x right), matching the
        deg_from_screen_right the push log reports. With "back" the zero moves
        to the camera's viewing direction so 0 means straight away from the
        viewer in any scene.
        """
        base = back if force_ref == "back" else right
        perp = np.cross([0.0, 0.0, 1.0], base)
        a = np.radians(float(deg))
        d = np.cos(a) * base + np.sin(a) * perp
        d[2] = 0.0
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    fluid_dir = dir_from_deg(fluid_deg)

    def apply_fluid(frame: int):
        if fluid_mode == "none" or fluid_accel <= 0:
            return
        v = fluid.get_particles_vel()
        as_tensor = hasattr(v, "detach")
        vn = v.detach().cpu().numpy().copy() if as_tensor else np.array(v, copy=True)
        dv = float(fluid_accel) * float(scene.dt)
        if fluid_mode == "push":
            vn[:, 0] += dv * fluid_dir[0]
            vn[:, 1] += dv * fluid_dir[1]
        elif fluid_mode == "swirl":
            p = fluid.get_particles_pos()
            pn = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
            r = pn[:, :2] - pn[:, :2].mean(axis=0)
            t = np.stack([-r[:, 1], r[:, 0]], axis=1)
            # normalise so the rim and the centre get the same speed rather
            # than the centre being left behind by a rigid-body profile
            t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
            vn[:, 0] += dv * t[:, 0]
            vn[:, 1] += dv * t[:, 1]
        if as_tensor:
            import torch
            fluid.set_particles_vel(torch.as_tensor(vn, device=v.device, dtype=v.dtype))
        else:
            fluid.set_particles_vel(vn)

    if fluid_mode != "none" and fluid_accel > 0:
        print(f"  fluid: {fluid_mode}, {fluid_accel} m/s^2, "
              f"{fluid_deg} deg from screen-right", flush=True)

    force_rng = np.random.default_rng(force_seed)
    push_log, direction = [], None
    switch_at = int(num_frames * float(push_switch))
    stuck_frames = 0
    prev_centroid = None

    if force_accel > 0:
        if rigid:
            mass = float(obj.get_mass())
            n_dofs = int(obj.n_dofs)
            how = f"dof force, mass={mass:.4f} kg, dofs={n_dofs}"
        else:
            how = f"particle velocity nudge, {info.get('num_object_particles', 0)} particles"
        if force_mode == "fixed":
            direction = dir_from_deg(force_deg)
        elif force_mode == "random":
            direction = random_tangent(force_rng)
            how += (f"; re-randomised after {stuck_patience} frames moving less than "
                    f"{stuck_eps * 1e3:.2f} mm each")
        print(f"  force: {force_mode}, {force_accel} m/s^2, seed={force_seed}; {how}",
              flush=True)

    def _log(frame: int, why: str):
        # angle relative to screen-right, so the log is readable against the video
        ang = float(np.degrees(np.arctan2(float(direction @ np.cross([0, 0, 1.0], right)),
                                          float(direction @ right))))
        push_log.append({"frame": int(frame), "reason": why,
                         "direction": np.round(direction, 4).tolist(),
                         "deg_from_screen_right": round(ang, 1)})

    def apply_push(frame: int):
        nonlocal direction, stuck_frames, prev_centroid
        if force_accel <= 0:
            return None
        if force_mode == "fixed":
            direction = dir_from_deg(force_deg)
            if frame == 0:
                _log(0, "fixed")
        elif force_mode == "lr":
            direction = -right if frame < switch_at else right
            if frame in (0, switch_at):
                _log(frame, "schedule")
        else:
            centroid = traj_obj[-1].mean(axis=0) if traj_obj else None
            if frame == 0:
                _log(0, "initial")
            elif centroid is not None and prev_centroid is not None:
                if float(np.linalg.norm(centroid - prev_centroid)) < stuck_eps:
                    stuck_frames += 1
                    if stuck_frames >= stuck_patience:
                        direction = random_tangent(force_rng, direction)
                        stuck_frames = 0
                        _log(frame, "stuck")
                        print(f"  [frame {frame}] stuck -> new direction "
                              f"{np.round(direction, 3).tolist()}", flush=True)
                else:
                    stuck_frames = 0
            prev_centroid = centroid

        if rigid:
            force = np.zeros(n_dofs, dtype=np.float32)
            force[:3] = (direction * force_accel * mass).astype(np.float32)
            obj.control_dofs_force(force)
        else:
            dv = (direction * force_accel * float(scene.dt)).astype(np.float32)
            vel = obj.get_particles_vel()
            vel[:, 0] += float(dv[0])
            vel[:, 1] += float(dv[1])
            vel[:, 2] += float(dv[2])
            obj.set_particles_vel(vel)
        return direction

    camera.start_recording()
    first = camera.render()
    if isinstance(first, (tuple, list)):
        first = first[0]
    import cv2
    cv2.imwrite(str(out / "frame0000.png"), cv2.cvtColor(np.asarray(first), cv2.COLOR_RGB2BGR))

    for frame in range(num_frames):
        apply_push(frame)
        apply_fluid(frame)
        traj_obj.append(object_points())
        traj_fluid.append(fluid.get_particles_pos().cpu().numpy().astype(np.float32))
        camera.render()
        for _ in range(substeps_per_frame):
            scene.step()

    np.savez_compressed(out / "fg_traj.npz", xyz=np.asarray(traj_obj))
    np.savez_compressed(out / "fluid_traj.npz", xyz=np.asarray(traj_fluid))
    camera.stop_recording(save_to_filename=str(out / "simulation.mp4"), fps=fps)

    summary = {
        "num_frames": num_frames,
        "seconds_per_frame": substeps_per_frame * dt,
        "playback_fps": fps,
        "slow_motion_factor": 1.0 / (fps * substeps_per_frame * dt),
        "object_particles": int(traj_obj[0].shape[0]),
        "fluid_particles": int(traj_fluid[0].shape[0]),
        "object_z_start": float(np.mean(traj_obj[0][:, 2])),
        "object_z_end": float(np.mean(traj_obj[-1][:, 2])),
        "interaction_state": art.interaction_state,
        "force_mode": force_mode,
        "force_deg": force_deg,
        "force_ref": force_ref,
        "screen_right_world": np.round(right, 4).tolist(),
        "screen_back_world": np.round(back, 4).tolist(),
        "fluid_mode": fluid_mode,
        "fluid_accel": fluid_accel,
        "fluid_deg": fluid_deg,
        "force_accel": force_accel,
        "force_seed": force_seed,
        "stuck_eps": stuck_eps,
        "stuck_patience": stuck_patience,
        "force_schedule": push_log,
        **info,
    }
    (out / "rollout.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    # what was actually applied, so a run can be reproduced or audited later
    (out / "force_realized.json").write_text(json.dumps(
        {"mode": force_mode, "accel_m_s2": force_accel, "seed": force_seed,
         "stuck_eps_m": stuck_eps, "stuck_patience_frames": stuck_patience,
         "screen_right_world": np.round(right, 4).tolist(),
         "schedule": push_log}, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_frames", type=int, default=401)
    p.add_argument("--substeps_per_frame", type=int, default=10)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--fluid_vis", choices=("particle", "recon"), default="recon",
                   help="How the liquid is drawn in simulation.mp4 only. 'recon' rebuilds the "
                        "surface each frame with splashsurf; 'particle' draws every MPM "
                        "particle as a sphere, which reads as a ball pit. Measured on 40 "
                        "frames of the apple: 103 s particle vs 117 s recon.")
    p.add_argument("--particle_size_scale", type=float, default=1.0,
                   help="Shrinks the drawn particle spheres; render-only, never physics.")
    p.add_argument("--substeps", type=int, default=0,
                   help="Override the MPM substeps per dt; 0 = derive from the CFL bound.")
    p.add_argument("--fps", type=int, default=30,
                   help="Playback fps for simulation.mp4.")
    p.add_argument("--force_accel", "--push_accel", dest="force_accel", type=float, default=5.0,
                   help="Lateral acceleration in m/s^2. Fixed magnitude; only the direction "
                        "is random.")
    p.add_argument("--force_deg", type=float, default=0.0,
                   help="For --force_mode fixed: degrees from the reference axis.")
    p.add_argument("--force_ref", choices=("right", "back"), default="right",
                   help="Zero of --force_deg: screen-right, or away from the camera.")
    p.add_argument("--fluid_mode", choices=("none", "push", "swirl"), default="none",
                   help="Force applied to the fluid: uniform push, or a stir about its centroid.")
    p.add_argument("--fluid_accel", type=float, default=0.0,
                   help="Fluid forcing magnitude in m/s^2.")
    p.add_argument("--fluid_deg", type=float, default=0.0,
                   help="For --fluid_mode push: degrees from screen-right.")
    p.add_argument("--force_mode", choices=("random", "lr", "fixed", "none"), default="random",
                   help="'random': a random horizontal direction, re-drawn when the object "
                        "stops moving. 'lr': the old fixed left-then-right schedule.")
    p.add_argument("--force_seed", type=int, default=0,
                   help="Seeds the direction draws, so a rollout is reproducible.")
    p.add_argument("--stuck_eps", type=float, default=2e-4,
                   help="Per-frame centroid displacement below which the object counts as "
                        "stuck (metres).")
    p.add_argument("--stuck_patience", type=int, default=40,
                   help="Consecutive stuck frames before the direction is re-drawn.")
    p.add_argument("--push_switch", type=float, default=0.5,
                   help="For --force_mode lr: fraction of the rollout after which it flips.")
    a = p.parse_args()
    print(json.dumps(rollout_with_retry(a.scene, a.out_dir, a.num_frames, a.substeps_per_frame,
                             a.dt, fps=a.fps, substeps=a.substeps, fluid_vis=a.fluid_vis,
                             particle_size_scale=a.particle_size_scale,
                             force_accel=(0.0 if a.force_mode == "none" else a.force_accel),
                             force_mode=a.force_mode, force_seed=a.force_seed,
                             force_deg=a.force_deg, force_ref=a.force_ref,
                             fluid_mode=a.fluid_mode,
                             fluid_accel=a.fluid_accel, fluid_deg=a.fluid_deg,
                             stuck_eps=a.stuck_eps, stuck_patience=a.stuck_patience,
                             push_switch=a.push_switch),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
