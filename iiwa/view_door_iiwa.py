"""
Interactive viewer: KUKA iiwa 14 welded to the door handle.

Applies the same joint-torque controller as run_door_iiwa_estimation.py
so you can watch the arm open the door.

Usage:
    python3.10 view_door_iiwa.py              # excited profile (default)
    python3.10 view_door_iiwa.py --mode qs    # quasi-static creep
    python3.10 view_door_iiwa.py --video      # also write mp4, no live window
    python3.10 view_door_iiwa.py --both       # live window + write mp4

Close the viewer window to exit.
"""

from __future__ import annotations

import argparse
import os

import imageio
import mujoco
import mujoco_viewer
import numpy as np

from iiwa.run_door_iiwa_estimation import (
    DT,
    N_STEPS,
    T_END,
    arm_torques_for_hinge_torque,
    door_angle_from_proprio,
    load_iiwa_door_model,
    tau_excited,
    tau_quasistatic,
)
from scenes import REPO_ROOT

RENDER_EVERY = 10
FPS = 1.0 / (DT * RENDER_EVERY)
TARGET_FPS = 60.0


def overlay_text(mode: str, t: float, theta_deg: float, tau: float) -> str:
    return f"{mode}  t={t:4.2f}s  θ={theta_deg:5.1f}°  τ={tau:5.2f} N·m"


def run_viewer(mode: str) -> None:
    model = load_iiwa_door_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    tau_fn = tau_quasistatic if mode == "qs" else tau_excited
    label = "quasi-static" if mode == "qs" else "excited"

    steps_per_render = max(1, int(round(1.0 / (TARGET_FPS * model.opt.timestep))))
    print(f"Door + iiwa viewer ({label})")
    print(f"  timestep={model.opt.timestep*1e3:.1f} ms  →  {steps_per_render} steps/frame")
    print("  Close the window to exit.\n")

    viewer = mujoco_viewer.MujocoViewer(model, data)
    step = 0
    while viewer.is_alive:
        for _ in range(steps_per_render):
            t = (step % N_STEPS) * DT
            th = door_angle_from_proprio(model, data)
            tau = float(tau_fn(t))
            tau_arm, _, _ = arm_torques_for_hinge_torque(model, data, tau, th)
            data.ctrl[:] = tau_arm
            mujoco.mj_step(model, data)
            step += 1
            # Loop the 6 s profile so you can keep watching
            if step > 0 and step % N_STEPS == 0:
                mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
                mujoco.mj_forward(model, data)
                print(f"  replay {label} …")

        # HUD via window title if available
        t_now = ((step - 1) % N_STEPS) * DT
        th_deg = np.degrees(door_angle_from_proprio(model, data))
        tau_now = float(tau_fn(t_now))
        try:
            viewer._time_per_render  # noqa: B018 — touch attr to keep linter calm
            if hasattr(viewer, "window") and viewer.window is not None:
                import glfw
                glfw.set_window_title(
                    viewer.window,
                    overlay_text(label, t_now, th_deg, tau_now),
                )
        except Exception:
            pass
        viewer.render()

    viewer.close()


def run_video(mode: str, path: str | None = None) -> str:
    model = load_iiwa_door_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    tau_fn = tau_quasistatic if mode == "qs" else tau_excited
    label = "quasistatic" if mode == "qs" else "excited"
    if path is None:
        path = f"door_iiwa_{label}.mp4"

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    for i in range(N_STEPS):
        t = i * DT
        th = door_angle_from_proprio(model, data)
        tau = float(tau_fn(t))
        tau_arm, _, _ = arm_torques_for_hinge_torque(model, data, tau, th)
        data.ctrl[:] = tau_arm
        mujoco.mj_step(model, data)
        if i % RENDER_EVERY == 0:
            renderer.update_scene(data, camera="view")
            frames.append(renderer.render().copy())

    renderer.close()
    imageio.mimsave(path, frames, fps=FPS, format="FFMPEG")
    th_end = np.degrees(door_angle_from_proprio(model, data))
    print(f"Saved {len(frames)} frames → {path}  (θ_end={th_end:.1f}°, T={T_END}s)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize door + iiwa interaction")
    parser.add_argument(
        "--mode",
        choices=("excited", "qs"),
        default="excited",
        help="Torque profile (default: excited)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Write mp4 only (no live window)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Write mp4 for both profiles, then open live viewer",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)   # keep artifact outputs landing in the repo root

    if args.both:
        run_video("qs")
        run_video("excited")
        run_viewer(args.mode)
    elif args.video:
        run_video(args.mode)
    else:
        run_viewer(args.mode)


if __name__ == "__main__":
    main()
