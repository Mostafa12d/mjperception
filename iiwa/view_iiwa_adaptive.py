"""
Live viewer — KUKA iiwa + adaptive impedance controller + online RLS.

Runs the same control loop as run_iiwa_adaptive_impedance.py inside an
interactive MuJoCo window.  The window title shows the live estimates.

Usage:
    python3.10 view_iiwa_adaptive.py              # excited (adaptive, default)
    python3.10 view_iiwa_adaptive.py --mode qs    # quasi-static creep
    python3.10 view_iiwa_adaptive.py --video      # render mp4, no window
"""
from __future__ import annotations

import argparse
import os

import mujoco
import mujoco_viewer
import numpy as np

from baseline.run_door_dynamics_validation import rls_init, rls_step
from baseline.run_door_adaptive_impedance import (
    reference_smooth,
    DITHER_AMP, DITHER_FREQ, TAU_MAX,
)
from iiwa.run_door_iiwa_estimation import (
    DT, N_STEPS, T_END,
    load_iiwa_door_model,
    door_angle_from_proprio,
    wrist_ft_hinge_torque,
    arm_torques_for_hinge_torque,
    true_hinge_inertia,
)
from iiwa.run_iiwa_adaptive_impedance import (
    adaptive_gains,
    RLS_LAM, I_HAT_INIT, MU_HAT_INIT, VEL_THRESH,
)
from scenes import REPO_ROOT

TARGET_FPS  = 60.0
RENDER_EVERY = max(1, int(round(1.0 / (TARGET_FPS * DT))))


# ---------------------------------------------------------------------------
# One simulation step (same logic as run_condition, returns display values)
# ---------------------------------------------------------------------------

class AdaptiveState:
    """Mutable carry-over state across steps."""
    __slots__ = ("rls", "th_prev", "thd_prev")

    def __init__(self) -> None:
        self.rls      = rls_init(2, delta=1e3, lam=RLS_LAM)
        self.rls.theta[:] = [I_HAT_INIT, MU_HAT_INIT]
        self.th_prev  = None
        self.thd_prev = None


def adaptive_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: AdaptiveState,
    t: float,
    excited: bool,
) -> dict:
    """One control step.  Returns display info dict."""
    # sense
    th  = door_angle_from_proprio(model, data)
    thd  = 0.0 if state.th_prev  is None else (th  - state.th_prev)  / DT
    thdd = 0.0 if state.thd_prev is None else (thd - state.thd_prev) / DT
    state.th_prev, state.thd_prev = th, thd

    I_hat  = float(state.rls.theta[0])
    mu_hat = float(state.rls.theta[1])

    # command
    th_d_ref, thd_d, thdd_d = reference_smooth(t)
    dither = DITHER_AMP * np.sin(2.0 * np.pi * DITHER_FREQ * t) if excited else 0.0

    if excited:
        kp, kd = adaptive_gains(I_hat)
        tau = (
            max(I_hat,  0.5) * thdd_d
            + max(mu_hat, 0.0) * np.sign(thd)
            + kp * (th_d_ref - th)
            + kd * (thd_d - thd)
            + dither
        )
    else:
        tau      = 3.5
        th_d_ref = th

    tau = float(np.clip(tau, -TAU_MAX, TAU_MAX))

    # actuate
    tau_arm, _, _ = arm_torques_for_hinge_torque(model, data, tau, th)
    data.ctrl[:] = tau_arm
    mujoco.mj_step(model, data)

    # observe + RLS update
    tau_h      = wrist_ft_hinge_torque(model, data)
    near_limit = th > 2.04 or th < -0.12
    if t > 0.04 and abs(thd) > VEL_THRESH and not near_limit:
        phi = np.array([thdd, np.sign(thd)])
        state.rls = rls_step(state.rls, phi, tau_h)
        state.rls.theta[0] = max(state.rls.theta[0], 0.1)

    return dict(
        t=t, th=th, th_d=th_d_ref, tau=tau,
        I_hat=I_hat, mu_hat=mu_hat,
        thd=thd,
    )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def run_viewer(mode: str) -> None:
    excited = (mode == "excited")
    label   = "excited (adaptive)" if excited else "quasi-static"

    model = load_iiwa_door_model()
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    gt     = true_hinge_inertia(model)
    I_true = gt["I_hinge"]

    state  = AdaptiveState()
    viewer = mujoco_viewer.MujocoViewer(model, data)

    print(f"Adaptive viewer ({label})  |  I_true={I_true:.2f} kg·m²")
    print("  Close the window to exit.\n")

    step = 0
    while viewer.is_alive:
        for _ in range(RENDER_EVERY):
            t   = (step % N_STEPS) * DT
            inf = adaptive_step(model, data, state, t, excited)
            step += 1

            if step > 0 and step % N_STEPS == 0:
                mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
                mujoco.mj_forward(model, data)
                state = AdaptiveState()            # reset estimator for new pass
                print(f"  replay — I_hat_final={inf['I_hat']:.3f}  "
                      f"I_err={abs(inf['I_hat']-I_true)/I_true*100:.1f}%  "
                      f"mu_hat={inf['mu_hat']:.3f}")

        inf = adaptive_step.__wrapped__ if hasattr(adaptive_step, "__wrapped__") else inf
        title = (
            f"{label}  t={inf['t']:4.2f}s  "
            f"θ={np.degrees(inf['th']):5.1f}° (ref {np.degrees(inf['th_d']):4.1f}°)  "
            f"τ={inf['tau']:5.1f}Nm  "
            f"Î={inf['I_hat']:5.2f}  μ̂={inf['mu_hat']:5.2f}"
        )
        try:
            if hasattr(viewer, "window") and viewer.window is not None:
                import glfw
                glfw.set_window_title(viewer.window, title)
        except Exception:
            pass

        viewer.render()

    viewer.close()


# ---------------------------------------------------------------------------
# Video renderer
# ---------------------------------------------------------------------------

def run_video(mode: str, path: str | None = None) -> str:
    import imageio

    excited = (mode == "excited")
    label   = "excited" if excited else "quasistatic"
    if path is None:
        path = f"iiwa_adaptive_{label}.mp4"

    model = load_iiwa_door_model()
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    gt     = true_hinge_inertia(model)
    I_true = gt["I_hinge"]
    state  = AdaptiveState()

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames   = []
    fps      = 1.0 / (DT * RENDER_EVERY)

    for i in range(N_STEPS):
        t   = i * DT
        inf = adaptive_step(model, data, state, t, excited)
        if i % RENDER_EVERY == 0:
            renderer.update_scene(data, camera="view")
            frames.append(renderer.render().copy())

    renderer.close()
    imageio.mimsave(path, frames, fps=fps, format="FFMPEG")
    I_final = float(state.rls.theta[0])
    mu_final = float(state.rls.theta[1])
    print(
        f"Saved {len(frames)} frames → {path}\n"
        f"  I_hat={I_final:.3f}  I_err={abs(I_final-I_true)/I_true*100:.1f}%  "
        f"  mu_hat={mu_final:.3f}"
    )
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize KUKA iiwa + adaptive impedance controller"
    )
    parser.add_argument(
        "--mode", choices=("excited", "qs"), default="excited",
        help="Control mode (default: excited)"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Render mp4 only (no live window)"
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)   # keep artifact outputs landing in the repo root

    mode = "qs" if args.mode == "qs" else "excited"
    if args.video:
        run_video(mode)
    else:
        run_viewer(mode)


if __name__ == "__main__":
    main()
