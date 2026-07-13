"""
Friction robustness sweep — online RLS adaptive impedance on the KUKA iiwa door.

Hypothesis: the estimator jointly recovers I_hinge and μ (Coulomb friction)
regardless of the true friction value, starting from a fixed wrong initial guess.

Sweep:  μ_true ∈ {0.5, 1.0, 2.0, 3.0, 5.0, 7.0} Nm  (6 door types)
Fixed:  μ_init = 3.0 Nm  (may be wrong for most cases)
        I_init = 5.0 kg·m²  (deliberate 2× under-estimate)

Outputs:
    friction_sweep_traces.png   — time traces of Î(t) and μ̂(t) for all doors
    friction_sweep_summary.png  — final I_err and μ_err vs μ_true

Run:
    python3.10 run_friction_sweep.py
"""
from __future__ import annotations

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_door_dynamics_validation import rls_init, rls_step
from run_door_adaptive_impedance import (
    reference_smooth,
    DITHER_AMP, DITHER_FREQ, TAU_MAX, T_RAMP,
)
from run_door_iiwa_estimation import (
    DT, N_STEPS, T_END,
    load_iiwa_door_model,
    door_angle_from_proprio,
    wrist_ft_hinge_torque,
    arm_torques_for_hinge_torque,
    true_hinge_inertia,
)
from run_iiwa_adaptive_impedance import (
    adaptive_gains,
    _convergence_time,
    RLS_LAM, OMEGA_N, ZETA, VEL_THRESH,
)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

MU_TRUE_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]   # Nm — door friction range
MU_INIT        = 3.0    # fixed initial guess for all runs
I_INIT         = 5.0    # fixed initial guess (2× under-estimate of I_true ≈ 11.67)

# colour palette — one colour per μ_true value
COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#795548"]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(mu_true: float, mu_init: float = MU_INIT, I_init: float = I_INIT) -> dict:
    """
    Excited adaptive impedance run with given true/initial friction.
    Returns full time histories + final metrics.
    """
    model = load_iiwa_door_model()
    # Override the door hinge's Coulomb frictionloss parameter
    model.dof_frictionloss[model.joint("hinge").dofadr[0]] = mu_true

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    gt     = true_hinge_inertia(model)
    I_true = gt["I_hinge"]       # ≈ 11.67 kg·m² regardless of friction

    rls = rls_init(2, delta=1e3, lam=RLS_LAM)
    rls.theta[:] = [I_init, mu_init]

    t_arr      = np.arange(N_STEPS) * DT
    theta      = np.zeros(N_STEPS)
    theta_d    = np.zeros(N_STEPS)
    theta_dot  = np.zeros(N_STEPS)
    I_hat_log  = np.zeros(N_STEPS)
    mu_hat_log = np.zeros(N_STEPS)

    th_prev = thd_prev = None

    for i in range(N_STEPS):
        t = t_arr[i]

        # sense
        th  = door_angle_from_proprio(model, data)
        thd  = 0.0 if th_prev  is None else (th  - th_prev)  / DT
        thdd = 0.0 if thd_prev is None else (thd - thd_prev) / DT
        th_prev, thd_prev = th, thd

        I_hat  = float(rls.theta[0])
        mu_hat = float(rls.theta[1])

        # command — excited impedance with adaptive gains
        th_d_ref, thd_d, thdd_d = reference_smooth(t)
        dither = DITHER_AMP * np.sin(2.0 * np.pi * DITHER_FREQ * t)
        kp, kd = adaptive_gains(I_hat)
        tau = (
            max(I_hat,  0.5) * thdd_d
            + max(mu_hat, 0.0) * np.sign(thd)
            + kp * (th_d_ref - th)
            + kd * (thd_d - thd)
            + dither
        )
        tau = float(np.clip(tau, -TAU_MAX, TAU_MAX))

        # actuate
        tau_arm, _, _ = arm_torques_for_hinge_torque(model, data, tau, th)
        data.ctrl[:] = tau_arm
        mujoco.mj_step(model, data)

        # observe + RLS update
        tau_h      = wrist_ft_hinge_torque(model, data)
        near_limit = th > 2.04 or th < -0.12
        if i >= 20 and abs(thd) > VEL_THRESH and not near_limit:
            phi = np.array([thdd, np.sign(thd)])
            rls = rls_step(rls, phi, tau_h)
            rls.theta[0] = max(rls.theta[0], 0.1)

        theta[i]      = th
        theta_d[i]    = th_d_ref
        I_hat_log[i]  = float(rls.theta[0])
        mu_hat_log[i] = float(rls.theta[1])

    I_final   = float(I_hat_log[-1])
    mu_final  = float(mu_hat_log[-1])
    I_err     = abs(I_final - I_true) / I_true * 100.0
    mu_err    = abs(mu_final - mu_true) / max(mu_true, 0.1) * 100.0
    conv_I    = _convergence_time(I_hat_log, I_true, tol=0.05)
    conv_mu   = _convergence_time(mu_hat_log, mu_true, tol=0.10, slack=0.15)

    return dict(
        mu_true=mu_true, mu_init=mu_init, I_init=I_init,
        I_true=I_true,
        t=t_arr, theta=theta, theta_d=theta_d,
        I_hat=I_hat_log, mu_hat=mu_hat_log,
        I_final=I_final, mu_final=mu_final,
        I_err=I_err, mu_err=mu_err,
        conv_I=conv_I, conv_mu=conv_mu,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_traces(results: list[dict], path: str = "friction_sweep_traces.png") -> None:
    """
    Two-panel figure: Î(t) and μ̂(t) for all μ_true values.
    Î(t) should all converge to the same I_true; μ̂(t) each to their own μ_true.
    """
    fig, (ax_I, ax_mu) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    I_true = results[0]["I_true"]

    for r, col in zip(results, COLOURS):
        mu_t = r["mu_true"]
        label = f"μ_true={mu_t:.1f}"

        ax_I.plot(r["t"], r["I_hat"], color=col, lw=1.2, label=label)
        ax_mu.plot(r["t"], r["mu_hat"], color=col, lw=1.2, label=label)

        # horizontal target lines for μ̂
        ax_mu.axhline(mu_t, color=col, ls="--", lw=0.8, alpha=0.6)

        # convergence markers on Î plot
        if r["conv_I"] is not None:
            ax_I.plot(r["conv_I"], r["I_hat"][int(r["conv_I"] / DT)],
                      "o", color=col, ms=6, zorder=5)

    # I_true reference line
    ax_I.axhline(I_true, color="k", ls="--", lw=1.4, label=f"I_true={I_true:.2f}")
    ax_I.set_ylabel("Î  [kg·m²]", fontsize=11)
    ax_I.set_ylim(0, max(25, I_true * 2.5))
    ax_I.legend(fontsize=8, loc="upper right", ncol=2)
    ax_I.grid(True, alpha=0.3)
    ax_I.set_title(
        f"Online RLS — inertia estimate Î(t)\n"
        f"μ_init={MU_INIT}  I_init={I_INIT}  ω_n={OMEGA_N}  λ={RLS_LAM}  "
        f"dither={DITHER_AMP}Nm@{DITHER_FREQ}Hz",
        fontsize=10,
    )

    ax_mu.axhline(MU_INIT, color="gray", ls=":", lw=1.2,
                  label=f"μ_init={MU_INIT} (starting guess)")
    ax_mu.set_ylabel("μ̂  [N·m]", fontsize=11)
    ax_mu.set_xlabel("time  [s]", fontsize=11)
    ax_mu.legend(fontsize=8, loc="upper right", ncol=2)
    ax_mu.grid(True, alpha=0.3)
    ax_mu.set_title("Online RLS — friction estimate μ̂(t)", fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")


def plot_summary(results: list[dict], path: str = "friction_sweep_summary.png") -> None:
    """
    Three-panel summary: final I_err, final μ_err, and convergence time vs μ_true.
    """
    mu_vals   = [r["mu_true"]  for r in results]
    I_errs    = [r["I_err"]    for r in results]
    mu_errs   = [r["mu_err"]   for r in results]
    conv_Is   = [r["conv_I"]  if r["conv_I"]  is not None else float("nan") for r in results]
    conv_mus  = [r["conv_mu"] if r["conv_mu"] is not None else float("nan") for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # ── I_err ────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(range(len(mu_vals)), I_errs, color=COLOURS[:len(mu_vals)], width=0.6)
    ax.axhline(5.0, color="k", ls="--", lw=1.2, label="5% target")
    ax.set_xticks(range(len(mu_vals)))
    ax.set_xticklabels([f"{m:.1f}" for m in mu_vals])
    ax.set_xlabel("μ_true  [N·m]")
    ax.set_ylabel("Final I_err  [%]")
    ax.set_title("Inertia estimation error\n(lower is better)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, I_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # ── μ_err ────────────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(range(len(mu_vals)), mu_errs, color=COLOURS[:len(mu_vals)], width=0.6)
    ax.axhline(10.0, color="k", ls="--", lw=1.2, label="10% target")
    ax.set_xticks(range(len(mu_vals)))
    ax.set_xticklabels([f"{m:.1f}" for m in mu_vals])
    ax.set_xlabel("μ_true  [N·m]")
    ax.set_ylabel("Final μ_err  [%]")
    ax.set_title("Friction estimation error\n(lower is better)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, mu_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # ── convergence times ────────────────────────────────────────────────
    ax = axes[2]
    w = 0.3
    xs = np.arange(len(mu_vals))
    b1 = ax.bar(xs - w / 2, conv_Is,  width=w, color=COLOURS[:len(mu_vals)],
                label="Î conv", alpha=0.9)
    b2 = ax.bar(xs + w / 2, conv_mus, width=w, color=COLOURS[:len(mu_vals)],
                label="μ̂ conv", alpha=0.5)
    ax.axhline(T_END, color="gray", ls=":", lw=1, label="sim end")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{m:.1f}" for m in mu_vals])
    ax.set_xlabel("μ_true  [N·m]")
    ax.set_ylabel("Convergence time  [s]")
    ax.set_title("Time to <5% I_err / <10% μ_err\n(NaN = did not converge)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Friction robustness sweep  |  μ_init={MU_INIT} Nm (fixed)  "
        f"I_init={I_INIT} kg·m²  |  excited adaptive impedance",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Friction robustness sweep — KUKA iiwa adaptive impedance\n")
    print(f"  μ_init = {MU_INIT} Nm (fixed wrong guess)")
    print(f"  I_init = {I_INIT} kg·m² (2× under-estimate)")
    print(f"  μ_true = {MU_TRUE_VALUES}\n")
    print(f"  {'μ_true':>8}  {'I_err%':>8}  {'μ_err%':>8}  {'conv_I':>8}  {'conv_μ':>8}  {'I_final':>9}  {'μ_final':>9}")
    print("  " + "-" * 67)

    results = []
    for mu_true in MU_TRUE_VALUES:
        r = run_one(mu_true)
        results.append(r)
        conv_I_str  = f"{r['conv_I']:.2f}s"  if r["conv_I"]  is not None else "  >6s  "
        conv_mu_str = f"{r['conv_mu']:.2f}s" if r["conv_mu"] is not None else "  >6s  "
        print(f"  {mu_true:>8.1f}  {r['I_err']:>8.1f}  {r['mu_err']:>8.1f}  "
              f"{conv_I_str:>8}  {conv_mu_str:>8}  "
              f"{r['I_final']:>9.4f}  {r['mu_final']:>9.4f}")

    print()
    print(f"  I_true = {results[0]['I_true']:.4f} kg·m² (same for all runs)")
    print(f"  μ_init = {MU_INIT} Nm  →  all μ̂ start from wrong value\n")

    # pass/fail summary
    all_I_pass  = all(r["I_err"]  < 5.0  for r in results)
    all_mu_pass = all(r["mu_err"] < 15.0 for r in results)
    print(f"  All I_err  < 5%  across μ_true range: {'PASS' if all_I_pass  else 'FAIL'}")
    print(f"  All μ_err  < 15% across μ_true range: {'PASS' if all_mu_pass else 'FAIL'}")
    print()

    plot_traces(results)
    plot_summary(results)


if __name__ == "__main__":
    main()
