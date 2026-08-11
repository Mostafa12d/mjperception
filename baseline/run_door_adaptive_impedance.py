"""
Adaptive impedance + online RLS on the MuJoCo door (oracle θ, F/T τ).

Two conditions:
  A) quasi-static creep: nearly constant torque just above friction (Phase-0 style)
     → Î stays unobservable / wrong; no intentional θ̈
  B) excited impedance: smooth minimum-jerk open (monotonic task) + small
     high-frequency torque dither for observability
     τ = Î θ̈_d + μ̂ sign(θ̇) + Kp e + Kd ė + A_τ sin(ωt)
     → door opens smoothly; Î still converges

Produces:
  adaptive_quasistatic.mp4, adaptive_excited.mp4
  adaptive_impedance_results.png

Run:
    python3.10 run_door_adaptive_impedance.py
"""

from __future__ import annotations

import numpy as np
import mujoco
import imageio
import matplotlib.pyplot as plt

from baseline.run_door_dynamics_validation import (
    DT,
    HANDLE_DIST,
    DEFAULT_DENSITY,
    DEFAULT_FRICTIONLOSS,
    DEFAULT_DAMPING,
    load_model,
    true_hinge_inertia,
    hinge_torque_from_handle_force,
    tangential_direction,
    rls_init,
    rls_step,
)

T_END = 6.0
N_STEPS = int(T_END / DT)
RENDER_EVERY = 10
FPS = 1.0 / (DT * RENDER_EVERY)

# Impedance / PD gains (used in excited condition)
KP = 40.0
KD = 16.0
TAU_MAX = 25.0

# Quasi-static creep torque (just above frictionloss=3.0)
CREEP_TORQUE = 3.5

# RLS
RLS_LAM = 0.995
I_HAT_INIT = 5.0
MU_HAT_INIT = 3.0
VEL_THRESH = 0.02

# Smooth task reference (minimum-jerk open — no position wiggle)
THETA_GOAL = 1.0          # rad (~57°)
T_RAMP = 5.0

# Observability via torque dither (not position oscillation).
# Keeps net opening monotonic while injecting θ̈ for RLS.
DITHER_AMP = 2.5          # N·m  (small vs creep/impedance torques)
DITHER_FREQ = 2.0         # Hz   (fast shimmer, not a visible swing)

# trace(P)-gated dither fade: once RLS covariance shrinks below this,
# ramp the dither amplitude down to 0 (linearly) instead of running all episode.
TRACE_P_THRESH = 1.0
RAMP_DURATION = 0.2       # s, linear ramp-down once gate trips (no hard cutoff)


def reference_smooth(t: float) -> tuple[float, float, float]:
    """Minimum-jerk open to THETA_GOAL — monotonic, visually smooth."""
    if t < T_RAMP:
        s = t / T_RAMP
        th = THETA_GOAL * (10 * s**3 - 15 * s**4 + 6 * s**5)
        thd = THETA_GOAL * (30 * s**2 - 60 * s**3 + 30 * s**4) / T_RAMP
        thdd = THETA_GOAL * (60 * s - 180 * s**2 + 120 * s**3) / T_RAMP**2
    else:
        th, thd, thdd = THETA_GOAL, 0.0, 0.0
    return th, thd, thdd


def impedance_torque(
    th: float,
    thd: float,
    th_d: float,
    thd_d: float,
    thdd_d: float,
    I_hat: float,
    mu_hat: float,
    dither: float = 0.0,
) -> float:
    """τ_cmd = Î θ̈_d + μ̂ sign(θ̇) + Kp e + Kd ė + dither."""
    I_use = max(I_hat, 0.5)
    mu_use = max(mu_hat, 0.0)
    tau = (
        I_use * thdd_d
        + mu_use * np.sign(thd)
        + KP * (th_d - th)
        + KD * (thd_d - thd)
        + dither
    )
    return float(np.clip(tau, -TAU_MAX, TAU_MAX))


def run_condition(
    excited: bool,
    video_path: str,
    *,
    model_path: str = "door.xml",
    handle_dist: float = HANDLE_DIST,
    density: float = DEFAULT_DENSITY,
    frictionloss: float = DEFAULT_FRICTIONLOSS,
    damping: float = DEFAULT_DAMPING,
    i_hat_init: float = I_HAT_INIT,
    mu_hat_init: float = MU_HAT_INIT,
    log_path: str | None = None,
) -> dict:
    model = load_model(
        density=density, frictionloss=frictionloss, damping=damping, model_path=model_path,
    )
    data = mujoco.MjData(model)
    gt = true_hinge_inertia(model)
    I_true = gt["I_hinge"]

    hinge_qpos = model.joint("hinge").qposadr[0]
    hinge_dof = model.joint("hinge").dofadr[0]
    handle_sid = model.site("handle").id
    door_bid = model.body("door").id

    rls = rls_init(2, delta=1e3, lam=RLS_LAM)
    rls.theta[:] = [i_hat_init, mu_hat_init]
    mu_true = gt["frictionloss"]

    t = np.arange(N_STEPS) * DT
    theta = np.zeros(N_STEPS)
    theta_dot = np.zeros(N_STEPS)
    theta_ddot = np.zeros(N_STEPS)
    theta_d = np.zeros(N_STEPS)
    tau_cmd = np.zeros(N_STEPS)
    I_hat_log = np.zeros(N_STEPS)
    mu_hat_log = np.zeros(N_STEPS)
    track_err = np.zeros(N_STEPS)
    trace_P_log = np.zeros(N_STEPS)
    dither_amp_log = np.zeros(N_STEPS)

    # trace(P)-gated dither fade state (excited condition only)
    a_tau_current = DITHER_AMP
    gate_tripped_t: float | None = None

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []

    for i in range(N_STEPS):
        th = float(data.qpos[hinge_qpos])
        thd = float(data.qvel[hinge_dof])
        I_hat = float(rls.theta[0])
        mu_hat = float(rls.theta[1])

        if excited:
            th_d, thd_d, thdd_d = reference_smooth(t[i])
            dither = a_tau_current * np.sin(2.0 * np.pi * DITHER_FREQ * t[i])
            tau = impedance_torque(
                th, thd, th_d, thd_d, thdd_d, I_hat, mu_hat, dither=dither
            )
        else:
            # Phase-0 creep: constant torque, no PD-injected acceleration
            th_d, thd_d, thdd_d = th, 0.0, 0.0
            tau = CREEP_TORQUE

        force = (tau / handle_dist) * tangential_direction(th)
        hinge_pos = data.xpos[door_bid].copy()
        hinge_axis = data.xmat[door_bid].reshape(3, 3)[:, 2].copy()
        handle_pos = data.site_xpos[handle_sid].copy()

        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data, force, np.zeros(3), handle_pos, door_bid, data.qfrc_applied,
        )
        tau_meas = hinge_torque_from_handle_force(
            handle_pos, hinge_pos, hinge_axis, force
        )

        mujoco.mj_step(model, data)

        th_new = float(data.qpos[hinge_qpos])
        thd_new = float(data.qvel[hinge_dof])
        thdd_new = float(data.qacc[hinge_dof])

        near_limit = th_new > 2.09 - 0.05 or th_new < -0.17 + 0.05
        if i >= 20 and abs(thd_new) > VEL_THRESH and not near_limit:
            phi = np.array([thdd_new, np.sign(thd_new)])
            rls = rls_step(rls, phi, tau_meas)
            if rls.theta[0] < 0.1:
                rls.theta[0] = 0.1

        trace_P = float(np.trace(rls.P))

        theta[i] = th_new
        theta_dot[i] = thd_new
        theta_ddot[i] = thdd_new
        theta_d[i] = th_d
        tau_cmd[i] = tau
        I_hat_log[i] = float(rls.theta[0])
        mu_hat_log[i] = float(rls.theta[1])
        track_err[i] = (th_d - th_new) if excited else 0.0
        trace_P_log[i] = trace_P
        dither_amp_log[i] = a_tau_current if excited else 0.0

        # trace(P)-gated dither fade: once the gate trips, ramp A_tau to 0
        # linearly over RAMP_DURATION instead of cutting it off.
        if excited:
            if gate_tripped_t is None and trace_P < TRACE_P_THRESH:
                gate_tripped_t = t[i]
            if gate_tripped_t is not None:
                ramp_frac = (t[i] - gate_tripped_t) / RAMP_DURATION
                a_tau_current = DITHER_AMP * max(0.0, 1.0 - ramp_frac)

        if i % RENDER_EVERY == 0:
            renderer.update_scene(data, camera="view")
            frames.append(renderer.render().copy())

    renderer.close()
    imageio.mimsave(video_path, frames, fps=FPS, format="FFMPEG")

    I_err_log = I_hat_log - I_true
    mu_err_log = mu_hat_log - mu_true

    I_final = float(I_hat_log[-1])
    rel_err = abs(I_final - I_true) / I_true * 100.0
    rmse_track = float(np.sqrt(np.mean(track_err**2))) if excited else float("nan")
    thdd_rms = float(np.sqrt(np.mean(theta_ddot**2)))
    # fraction of time door velocity is negative (backtracking) during the open
    open_mask = t < T_RAMP
    n_back = int(np.sum((theta_dot < -0.01) & open_mask)) if excited else 0
    frac_back = n_back / max(int(open_mask.sum()), 1)

    conv_t = None
    for i in range(N_STEPS):
        if abs(I_hat_log[i] - I_true) / I_true < 0.10:
            w = I_hat_log[i : min(i + 100, N_STEPS)]
            if np.all(np.abs(w - I_true) / I_true < 0.15):
                conv_t = i * DT
                break

    # dither-fade gate summary (excited condition only)
    zero_mask = dither_amp_log <= 1e-9
    a_tau_zero_t = float(t[np.argmax(zero_mask)]) if (excited and zero_mask.any()) else None
    rmse_post_gate = float("nan")
    if excited and gate_tripped_t is not None:
        post_mask = t >= gate_tripped_t
        if np.any(post_mask):
            rmse_post_gate = float(np.sqrt(np.mean(track_err[post_mask] ** 2)))

    if excited:
        if log_path is None:
            log_path = "adaptive_impedance_dither_log.csv"
        header = "t,trace_P,A_tau,I_hat,mu_hat,I_err,mu_err,theta,theta_dot,tracking_error"
        np.savetxt(
            log_path,
            np.column_stack(
                [t, trace_P_log, dither_amp_log, I_hat_log, mu_hat_log,
                 I_err_log, mu_err_log, theta, theta_dot, track_err]
            ),
            delimiter=",",
            header=header,
            comments="",
        )

    label = "excited" if excited else "quasi-static"
    print(f"=== adaptive impedance: {label} ===")
    print(f"  Saved {len(frames)} frames → {video_path}")
    print(f"  I_true={I_true:.3f}  I_hat_final={I_final:.3f}  err={rel_err:.1f}%")
    print(f"  μ_hat_final={mu_hat_log[-1]:.3f}  (frictionloss={gt['frictionloss']:.3f})")
    print(f"  θ̈ RMS={thdd_rms:.4f} rad/s²  theta_end={np.degrees(theta[-1]):.1f}°")
    if excited:
        print(f"  tracking RMSE={rmse_track:.4f} rad")
        print(f"  velocity reversals during open: {100 * frac_back:.1f}% of samples")
        if gate_tripped_t is not None:
            print(f"  trace(P) gate tripped at t={gate_tripped_t:.3f}s")
        else:
            print("  trace(P) gate never tripped")
        if a_tau_zero_t is not None:
            print(f"  A_tau reached 0 at t={a_tau_zero_t:.3f}s")
        print(f"  post-gate tracking RMSE={rmse_post_gate:.4f} rad")
        print(f"  log → {log_path}")
    if conv_t is not None:
        print(f"  Î entered <10% error at t≈{conv_t:.2f}s")
    else:
        print("  Î never sustained <10% error")
    print()

    return dict(
        t=t,
        theta=theta,
        theta_d=theta_d,
        theta_dot=theta_dot,
        theta_ddot=theta_ddot,
        tau_cmd=tau_cmd,
        I_hat=I_hat_log,
        mu_hat=mu_hat_log,
        track_err=track_err,
        I_true=I_true,
        mu_true=mu_true,
        rel_err=rel_err,
        rmse_track=rmse_track,
        thdd_rms=thdd_rms,
        frac_back=frac_back,
        conv_t=conv_t,
        label=label,
        excited=excited,
        trace_P=trace_P_log,
        dither_amp=dither_amp_log,
        I_err=I_err_log,
        mu_err=mu_err_log,
        gate_tripped_t=gate_tripped_t,
        a_tau_zero_t=a_tau_zero_t,
        rmse_post_gate=rmse_post_gate,
    )


def plot_results(qs: dict, ex: dict, path: str = "adaptive_impedance_results.png") -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex="col")
    for col, r in enumerate([qs, ex]):
        if r["excited"]:
            axes[0, col].plot(r["t"], np.degrees(r["theta_d"]), "k--", lw=1, label="θ_d")
        axes[0, col].plot(r["t"], np.degrees(r["theta"]), label="θ")
        axes[0, col].set_ylabel("angle [deg]")
        axes[0, col].set_title(r["label"])
        axes[0, col].legend(loc="lower right", fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(r["t"], r["I_hat"], label="Î")
        axes[1, col].axhline(r["I_true"], color="k", ls="--", label="I_true")
        axes[1, col].set_ylabel("I_hinge [kg·m²]")
        axes[1, col].legend(loc="best", fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        if r["conv_t"] is not None:
            axes[1, col].axvline(r["conv_t"], color="C2", ls=":")

        axes[2, col].plot(r["t"], r["tau_cmd"], label="τ_cmd")
        axes[2, col].set_ylabel("τ [N·m]")
        axes[2, col].set_xlabel("time [s]")
        axes[2, col].legend(loc="best", fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

    fig.suptitle(
        "Creep vs smooth open + torque dither  |  "
        f"qs I_err={qs['rel_err']:.1f}%  ex I_err={ex['rel_err']:.1f}%  "
        f"ex backtrack={100 * ex['frac_back']:.1f}%",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot → {path}")


def main() -> None:
    print("Adaptive impedance door sim (oracle θ, F/T τ, online RLS)\n")
    print("  QS  = constant creep torque (Phase-0 non-exciting interaction)")
    print("  EX  = smooth min-jerk open + small torque dither (task stays monotonic)\n")
    qs = run_condition(excited=False, video_path="adaptive_quasistatic.mp4")
    ex = run_condition(excited=True, video_path="adaptive_excited.mp4")
    plot_results(qs, ex)

    print("--- Summary ---")
    print(f"  Quasi-static: I_err={qs['rel_err']:.1f}%, θ̈_rms={qs['thdd_rms']:.4f}")
    print(f"  Excited:      I_err={ex['rel_err']:.1f}%, θ̈_rms={ex['thdd_rms']:.4f}, "
          f"track_RMSE={ex['rmse_track']:.4f} rad, "
          f"backtrack={100 * ex['frac_back']:.1f}%")
    if ex["rel_err"] < qs["rel_err"]:
        print("  Excited recovers inertia better — Phase-0 effect with a smooth task.")
    else:
        print("  WARN: excited did not beat quasi-static on final I_err.")


if __name__ == "__main__":
    main()
