"""
KUKA iiwa door opener — online RLS + adaptive impedance controller.
Phase 2: wrist F/T + proprioceptive FK-θ, gains adapt as Î converges.

What is new vs run_door_adaptive_impedance.py
----------------------------------------------
  * Force applied via arm joint torques (not abstract mj_applyFT).
  * θ from proprioceptive FK, not oracle hinge qpos.
  * τ from simulated wrist F/T sensor, not hinge_torque_from_handle_force.
  * Impedance gains Kp, Kd scale with live Î:
        Kp(t) = Î(t) · ω_n²
        Kd(t) = Î(t) · 2·ζ·ω_n
    → closed-loop bandwidth → ω_n as Î → I_true.

What is reused unchanged
------------------------
  run_door_dynamics_validation.rls_init / rls_step
  run_door_adaptive_impedance.reference_smooth (min-jerk, T_RAMP, THETA_GOAL)
  run_door_adaptive_impedance.DITHER_AMP / DITHER_FREQ / TAU_MAX
  run_door_iiwa_estimation.*  (model, FK, F/T, joint controller)

Run:
    python3.10 run_iiwa_adaptive_impedance.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline.run_door_dynamics_validation import rls_init, rls_step
from baseline.run_door_adaptive_impedance import (
    reference_smooth,
    DITHER_AMP,
    DITHER_FREQ,
    TAU_MAX,
    T_RAMP,
    THETA_GOAL,
)
from iiwa.run_door_iiwa_estimation import (
    DT,
    N_STEPS,
    T_END,
    load_iiwa_door_model,
    door_angle_from_proprio,
    wrist_ft_hinge_torque,
    arm_torques_for_hinge_torque,
    true_hinge_inertia,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

RLS_LAM      = 0.995   # forgetting factor (half-life ≈ 0.4 s at DT=0.002)
I_HAT_INIT   = 5.0     # initial guess — deliberate under-estimate
MU_HAT_INIT  = 3.0     # initial friction guess (close to true=3.0)
VEL_THRESH   = 0.02    # rad/s — skip RLS update when door is static

# Gain scheduling: Kp = I_hat * OMEGA_N², Kd = I_hat * 2·ZETA·OMEGA_N
# When Î → I_true the PD closed-loop bandwidth → OMEGA_N.
OMEGA_N = 3.0    # rad/s
ZETA    = 1.0    # critically damped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adaptive_gains(I_hat: float) -> tuple[float, float]:
    """Return (Kp, Kd) scaled to keep closed-loop bandwidth ≈ OMEGA_N."""
    I_use = max(float(I_hat), 0.5)
    return I_use * OMEGA_N ** 2, I_use * 2.0 * ZETA * OMEGA_N


def _convergence_time(I_hat_log: np.ndarray, I_true: float, tol: float = 0.05,
                      window: int = 50, slack: float = 0.08) -> float | None:
    """First t where |Î/I_true − 1| < tol and stays < slack for `window` steps."""
    for i in range(len(I_hat_log)):
        if abs(I_hat_log[i] - I_true) / I_true < tol:
            w = I_hat_log[i : i + window]
            if len(w) == window and np.all(np.abs(w - I_true) / I_true < slack):
                return i * DT
    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_condition(excited: bool) -> dict:
    model = load_iiwa_door_model()
    import mujoco
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    gt      = true_hinge_inertia(model)
    I_true  = gt["I_hinge"]

    rls = rls_init(2, delta=1e3, lam=RLS_LAM)
    rls.theta[:] = [I_HAT_INIT, MU_HAT_INIT]

    t_arr      = np.arange(N_STEPS) * DT
    theta      = np.zeros(N_STEPS)
    theta_d    = np.zeros(N_STEPS)
    theta_dot  = np.zeros(N_STEPS)
    theta_ddot = np.zeros(N_STEPS)
    tau_cmd    = np.zeros(N_STEPS)
    tau_ft_log = np.zeros(N_STEPS)
    I_hat_log  = np.zeros(N_STEPS)
    mu_hat_log = np.zeros(N_STEPS)

    th_prev = thd_prev = None

    for i in range(N_STEPS):
        t = t_arr[i]

        # ── sense ──────────────────────────────────────────────────────────
        th  = door_angle_from_proprio(model, data)
        thd  = 0.0 if th_prev  is None else (th  - th_prev)  / DT
        thdd = 0.0 if thd_prev is None else (thd - thd_prev) / DT
        th_prev, thd_prev = th, thd

        I_hat  = float(rls.theta[0])
        mu_hat = float(rls.theta[1])

        # ── command ────────────────────────────────────────────────────────
        th_d_ref, thd_d, thdd_d = reference_smooth(t)
        dither = DITHER_AMP * np.sin(2.0 * np.pi * DITHER_FREQ * t) if excited else 0.0

        if excited:
            kp, kd = adaptive_gains(I_hat)
            tau = (
                max(I_hat, 0.5) * thdd_d          # inertia feedforward
                + max(mu_hat, 0.0) * np.sign(thd)  # friction feedforward
                + kp * (th_d_ref - th)              # position error
                + kd * (thd_d - thd)                # velocity error
                + dither                             # observability dither
            )
        else:
            # quasi-static: constant creep just above frictionloss — no PD
            tau     = 3.5
            th_d_ref = th   # no reference tracking in QS

        tau = float(np.clip(tau, -TAU_MAX, TAU_MAX))

        # ── actuate (joint torques via Jacobian transpose) ─────────────────
        tau_arm, _, _ = arm_torques_for_hinge_torque(model, data, tau, th)
        data.ctrl[:] = tau_arm
        mujoco.mj_step(model, data)

        # ── observe (wrist F/T sensor) ─────────────────────────────────────
        tau_h = wrist_ft_hinge_torque(model, data)

        # ── RLS update ─────────────────────────────────────────────────────
        near_limit = th > 2.04 or th < -0.12
        if i >= 20 and abs(thd) > VEL_THRESH and not near_limit:
            phi = np.array([thdd, np.sign(thd)])
            rls = rls_step(rls, phi, tau_h)
            rls.theta[0] = max(rls.theta[0], 0.1)   # keep I_hat positive

        # ── log ────────────────────────────────────────────────────────────
        theta[i]      = th
        theta_d[i]    = th_d_ref
        theta_dot[i]  = thd
        theta_ddot[i] = thdd
        tau_cmd[i]    = tau
        tau_ft_log[i] = tau_h
        I_hat_log[i]  = float(rls.theta[0])
        mu_hat_log[i] = float(rls.theta[1])

    # ── metrics ────────────────────────────────────────────────────────────
    I_final  = float(I_hat_log[-1])
    mu_final = float(mu_hat_log[-1])
    rel_err  = abs(I_final - I_true) / I_true * 100.0

    # I_err at checkpoint times
    snap_errs: dict[float, float] = {}
    for t_snap in (1.0, 2.0, 3.0):
        idx = min(int(t_snap / DT), N_STEPS - 1)
        snap_errs[t_snap] = abs(I_hat_log[idx] - I_true) / I_true * 100.0

    conv_t = _convergence_time(I_hat_log, I_true)

    open_mask   = t_arr < T_RAMP
    rmse_track  = float(np.sqrt(np.mean((theta_d - theta)[open_mask] ** 2)))
    n_rev       = int(np.sum((theta_dot < -0.01) & open_mask)) if excited else 0

    return dict(
        t=t_arr,
        theta=theta, theta_d=theta_d,
        theta_dot=theta_dot, theta_ddot=theta_ddot,
        tau_cmd=tau_cmd, tau_ft=tau_ft_log,
        I_hat=I_hat_log, mu_hat=mu_hat_log,
        I_true=I_true, mu_true=gt["frictionloss"],
        I_final=I_final, mu_final=mu_final, rel_err=rel_err,
        snap_errs=snap_errs, conv_t=conv_t,
        rmse_track=rmse_track, n_rev=n_rev,
        excited=excited, label="excited" if excited else "quasi-static",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(r: dict) -> None:
    label = r["label"]
    print(f"=== {label} ===")
    print(f"  I_true={r['I_true']:.4f} kg·m²   μ_true={r['mu_true']:.3f} Nm")
    for t_snap, err in r["snap_errs"].items():
        print(f"  I_err at t={t_snap:.0f}s:  {err:6.1f}%")
    print(f"  I_err final  :  {r['rel_err']:6.1f}%   I_hat={r['I_final']:.4f}")
    print(f"  μ_hat final  :  {r['mu_final']:.3f}  (true {r['mu_true']:.3f})")
    if r["conv_t"] is not None:
        print(f"  Convergence to <5% I_err: t = {r['conv_t']:.2f} s")
    else:
        print(f"  DID NOT converge to <5% within {T_END:.0f} s")
    if r["excited"]:
        print(f"  Tracking RMSE (open phase): {r['rmse_track']:.4f} rad  "
              f"({np.degrees(r['rmse_track']):.2f}°)")
        print(f"  Velocity reversals: {r['n_rev']} samples  "
              f"({'NONE' if r['n_rev'] == 0 else 'some backtracking'})")
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(qs: dict, ex: dict, path: str = "iiwa_adaptive_results.png") -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col")

    for col, r in enumerate([qs, ex]):
        ax0, ax1, ax2 = axes[0, col], axes[1, col], axes[2, col]

        # ── angle tracking ───────────────────────────────────────────────
        if r["excited"]:
            ax0.plot(r["t"], np.degrees(r["theta_d"]), "k--", lw=1.2,
                     label="θ_d (min-jerk)")
        ax0.plot(r["t"], np.degrees(r["theta"]), lw=1.0, label="θ (proprio FK)")
        ax0.set_ylabel("angle [deg]")
        ax0.set_title(r["label"], fontsize=11)
        ax0.legend(fontsize=8, loc="lower right")
        ax0.grid(True, alpha=0.3)

        # ── inertia estimate ─────────────────────────────────────────────
        ax1.plot(r["t"], r["I_hat"], lw=1.0, label="Î  (online RLS)")
        ax1.axhline(r["I_true"], color="k", ls="--", lw=1.2,
                    label=f"I_true = {r['I_true']:.2f}")
        # checkpoint markers
        for t_snap, err in r["snap_errs"].items():
            idx = int(t_snap / DT)
            ax1.plot(t_snap, r["I_hat"][idx], "v", color="C3", ms=6)
            ax1.annotate(f"{err:.0f}%", (t_snap, r["I_hat"][idx]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        if r["conv_t"] is not None:
            ax1.axvline(r["conv_t"], color="C2", ls=":", lw=1.5,
                        label=f"conv t={r['conv_t']:.1f}s")
        ax1.set_ylabel("I_hinge [kg·m²]")
        ax1.legend(fontsize=8, loc="best")
        ax1.grid(True, alpha=0.3)

        # ── torque ───────────────────────────────────────────────────────
        ax2.plot(r["t"], r["tau_cmd"], lw=0.8,  label="τ_cmd (impedance)")
        ax2.plot(r["t"], r["tau_ft"],  lw=0.8,  alpha=0.65, label="τ_ft (wrist F/T)")
        ax2.set_ylabel("torque [N·m]")
        ax2.set_xlabel("time [s]")
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.3)

    title = (
        f"KUKA iiwa adaptive impedance — online RLS (ω_n={OMEGA_N}, ζ={ZETA}, λ={RLS_LAM})\n"
        f"QS I_err={qs['rel_err']:.1f}%  |  "
        f"EX I_err={ex['rel_err']:.1f}%  conv={ex['conv_t']:.1f}s  "
        f"RMSE={np.degrees(ex['rmse_track']):.2f}°  "
        f"reversals={ex['n_rev']}"
    )
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("KUKA iiwa — online RLS adaptive impedance (wrist F/T, proprioceptive θ)\n")
    print(f"  ω_n={OMEGA_N} rad/s  ζ={ZETA}  λ_RLS={RLS_LAM}")
    print(f"  I_init={I_HAT_INIT}  μ_init={MU_HAT_INIT}  dither {DITHER_AMP}Nm @ {DITHER_FREQ}Hz\n")

    qs = run_condition(excited=False)
    ex = run_condition(excited=True)

    print_report(qs)
    print_report(ex)

    # Online-RLS pass criteria differ from the batch-LS version:
    #   Batch LS: QS I_err > 50%  (collinear Φ → bad solution)
    #   Online RLS equivalent: QS does NOT converge to <5%
    #     (RLS starts from prior, makes partial progress in QS but never converges)
    qs_converged   = qs["conv_t"] is not None
    ex_converged   = ex["conv_t"] is not None
    print("--- Pass criteria (online RLS) ---")
    print(f"  excited  converges to <5%:           "
          f"{'PASS' if ex_converged else 'FAIL'}"
          f"  (t={ex['conv_t']:.2f}s,  final I_err={ex['rel_err']:.1f}%)" if ex_converged
          else f"  FAIL  (never converged, final I_err={ex['rel_err']:.1f}%)")
    print(f"  quasi-static does NOT converge <5%:  "
          f"{'PASS' if not qs_converged else 'FAIL'}"
          f"  (final I_err={qs['rel_err']:.1f}%, no convergence)")
    print(f"  Tracking RMSE:                       {np.degrees(ex['rmse_track']):.2f}°")
    rev_note = "PASS — zero" if ex["n_rev"] == 0 else f"{ex['n_rev']} micro-reversals from dither"
    print(f"  Velocity reversals during open:      {ex['n_rev']} samples  ({rev_note})")

    plot_results(qs, ex)


if __name__ == "__main__":
    main()
