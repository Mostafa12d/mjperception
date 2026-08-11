"""
Phase 0c: sensor-noise sensitivity of online inertia estimation.

Same plant / controllers as run_door_adaptive_impedance.py:
  QS  = constant creep torque
  EX  = smooth min-jerk open + torque dither

Sensors (NO filtering / NO Kalman):
  θ_meas = θ_true + N(0, σ_θ)          # vision-like angle noise
  τ_meas = τ_ft   + N(0, σ_τ)          # wrist F/T noise
  θ̇_meas, θ̈_meas from raw finite differences of θ_meas

RLS and (for EX) PD feedback both use the noisy signals.

Sweeps low / medium / high noise on θ and τ, reports I_err %,
convergence time, tracking RMSE, and whether the QS-vs-EX effect holds.

Run:
    python3.10 run_door_noise_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from baseline.run_door_dynamics_validation import (
    DT,
    HANDLE_DIST,
    load_model,
    true_hinge_inertia,
    hinge_torque_from_handle_force,
    tangential_direction,
    rls_init,
    rls_step,
)
from baseline.run_door_adaptive_impedance import (
    CREEP_TORQUE,
    RLS_LAM,
    I_HAT_INIT,
    MU_HAT_INIT,
    VEL_THRESH,
    DITHER_AMP,
    DITHER_FREQ,
    reference_smooth,
    impedance_torque,
)

T_END = 6.0
N_STEPS = int(T_END / DT)

# Noise grids (σ). θ in radians; τ in N·m.
# "med" matches the requested realistic band (0.5–1.0 deg, 0.05–0.1 N·m).
THETA_NOISE = {
    "none": 0.0,
    "low": np.deg2rad(0.5),
    "med": np.deg2rad(1.0),
    "high": np.deg2rad(2.0),
}
TAU_NOISE = {
    "none": 0.0,
    "low": 0.05,
    "med": 0.10,
    "high": 0.20,
}

# Effect "holds" if excited I_err < 20% AND excited clearly beats QS
EX_ERR_OK = 20.0  # percent
EX_BEATS_QS_MARGIN = 10.0  # percentage points


def finite_diff(prev: float | None, curr: float, dt: float) -> float:
    if prev is None:
        return 0.0
    return (curr - prev) / dt


def run_once(
    excited: bool,
    sigma_theta: float,
    sigma_tau: float,
    seed: int,
) -> dict:
    """One closed-loop trial with raw noisy θ / τ into control + RLS."""
    rng = np.random.default_rng(seed)
    model = load_model()
    data = mujoco.MjData(model)
    gt = true_hinge_inertia(model)
    I_true = gt["I_hinge"]

    hinge_qpos = model.joint("hinge").qposadr[0]
    handle_sid = model.site("handle").id
    door_bid = model.body("door").id

    rls = rls_init(2, delta=1e3, lam=RLS_LAM)
    rls.theta[:] = [I_HAT_INIT, MU_HAT_INIT]

    t = np.arange(N_STEPS) * DT
    theta_true = np.zeros(N_STEPS)
    I_hat_log = np.zeros(N_STEPS)
    track_err = np.zeros(N_STEPS)

    th_m_prev = None
    thd_m_prev = None

    for i in range(N_STEPS):
        th_true = float(data.qpos[hinge_qpos])

        # --- sensors (raw) ---
        th_m = th_true + rng.normal(0.0, sigma_theta)
        thd_m = finite_diff(th_m_prev, th_m, DT)
        thdd_m = finite_diff(thd_m_prev, thd_m, DT)

        I_hat = float(rls.theta[0])
        mu_hat = float(rls.theta[1])

        if excited:
            th_d, thd_d, thdd_d = reference_smooth(t[i])
            dither = DITHER_AMP * np.sin(2.0 * np.pi * DITHER_FREQ * t[i])
            # PD uses noisy angle / finite-diff velocity (no filter)
            tau = impedance_torque(
                th_m, thd_m, th_d, thd_d, thdd_d, I_hat, mu_hat, dither=dither
            )
        else:
            th_d = th_true
            tau = CREEP_TORQUE

        force = (tau / HANDLE_DIST) * tangential_direction(th_true)
        hinge_pos = data.xpos[door_bid].copy()
        hinge_axis = data.xmat[door_bid].reshape(3, 3)[:, 2].copy()
        handle_pos = data.site_xpos[handle_sid].copy()

        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data, force, np.zeros(3), handle_pos, door_bid, data.qfrc_applied,
        )
        tau_ft = hinge_torque_from_handle_force(
            handle_pos, hinge_pos, hinge_axis, force
        )
        tau_m = tau_ft + rng.normal(0.0, sigma_tau)

        mujoco.mj_step(model, data)

        near_limit = th_true > 2.09 - 0.05 or th_true < -0.17 + 0.05
        if i >= 20 and abs(thd_m) > VEL_THRESH and not near_limit:
            phi = np.array([thdd_m, np.sign(thd_m)])
            rls = rls_step(rls, phi, float(tau_m))
            if rls.theta[0] < 0.1:
                rls.theta[0] = 0.1
            if rls.theta[0] > 50.0:
                rls.theta[0] = 50.0  # hard cap against FD blow-ups

        th_m_prev = th_m
        thd_m_prev = thd_m

        theta_true[i] = float(data.qpos[hinge_qpos])
        I_hat_log[i] = float(rls.theta[0])
        track_err[i] = (th_d - theta_true[i]) if excited else 0.0

    I_final = float(I_hat_log[-1])
    rel_err = abs(I_final - I_true) / I_true * 100.0
    rmse_track = float(np.sqrt(np.mean(track_err**2))) if excited else float("nan")

    conv_t = None
    for i in range(N_STEPS):
        if abs(I_hat_log[i] - I_true) / I_true < 0.10:
            w = I_hat_log[i : min(i + 100, N_STEPS)]
            if len(w) >= 50 and np.all(np.abs(w - I_true) / I_true < 0.20):
                conv_t = i * DT
                break

    return dict(
        I_true=I_true,
        I_hat=I_final,
        rel_err=rel_err,
        conv_t=conv_t,
        rmse_track=rmse_track,
        I_hist=I_hat_log,
        theta=theta_true,
        excited=excited,
        sigma_theta=sigma_theta,
        sigma_tau=sigma_tau,
    )


def fmt_conv(conv_t: float | None) -> str:
    return f"{conv_t:.2f}s" if conv_t is not None else "never"


def main() -> None:
    print("Phase 0c: raw sensor-noise sensitivity (no Kalman / no denoising)\n")
    print("  θ_meas = θ + N(0, σ_θ);  θ̇, θ̈ = finite differences of θ_meas")
    print("  τ_meas = τ_ft + N(0, σ_τ)")
    print("  RLS + PD consume these raw signals directly.\n")

    # --- Baseline + realistic mid levels (detailed) ---
    print("=" * 72)
    print("1) Detailed runs: clean vs realistic (σ_θ=1.0°, σ_τ=0.10 N·m)")
    print("=" * 72)
    cases = [
        ("clean", 0.0, 0.0),
        ("noisy", THETA_NOISE["med"], TAU_NOISE["med"]),
    ]
    detailed = {}
    for tag, s_th, s_tau in cases:
        print(f"\n--- {tag}: σ_θ={np.degrees(s_th):.2f}°, σ_τ={s_tau:.3f} N·m ---")
        for excited, name in [(False, "quasi-static"), (True, "excited")]:
            r = run_once(excited, s_th, s_tau, seed=0)
            detailed[(tag, name)] = r
            track = f"{r['rmse_track']:.4f} rad" if excited else "n/a"
            print(
                f"  {name:13s}  I_err={r['rel_err']:7.1f}%  "
                f"I_hat={r['I_hat']:7.3f}  conv={fmt_conv(r['conv_t']):>6s}  "
                f"track_RMSE={track}"
            )

    # --- Full sweep table ---
    print("\n" + "=" * 72)
    print("2) Sweep table: excited I_err% / conv / track_RMSE  (QS I_err% in paren)")
    print("=" * 72)
    header = "sig_th\\sig_tau"
    print(f"  {header:>14}", *[f"{k:>22}" for k in TAU_NOISE], sep="")

    sweep = {}
    for th_name, s_th in THETA_NOISE.items():
        row = f"  {th_name:>6}({np.degrees(s_th):4.1f}°)"
        for tau_name, s_tau in TAU_NOISE.items():
            r_ex = run_once(True, s_th, s_tau, seed=0)
            r_qs = run_once(False, s_th, s_tau, seed=0)
            sweep[(th_name, tau_name)] = (r_ex, r_qs)
            cell = (
                f"{r_ex['rel_err']:5.1f}%/{fmt_conv(r_ex['conv_t'])}"
                f"/{r_ex['rmse_track']:.3f}"
                f" (qs{r_qs['rel_err']:.0f}%)"
            )
            row += f"  {cell:>22}"
        print(row)

    # --- Breakdown diagnosis ---
    print("\n" + "=" * 72)
    print("3) Where does the observability claim break?")
    print("=" * 72)
    print(
        f"  Criterion: excited I_err < {EX_ERR_OK:.0f}% AND "
        f"qs_err - ex_err > {EX_BEATS_QS_MARGIN:.0f} pp\n"
    )
    holds = []
    breaks = []
    for (th_name, tau_name), (r_ex, r_qs) in sweep.items():
        ok = (
            r_ex["rel_err"] < EX_ERR_OK
            and (r_qs["rel_err"] - r_ex["rel_err"]) > EX_BEATS_QS_MARGIN
        )
        label = f"θ={th_name}, τ={tau_name}"
        if ok:
            holds.append(label)
        else:
            breaks.append(
                f"{label}: ex_err={r_ex['rel_err']:.1f}%, qs_err={r_qs['rel_err']:.1f}%, "
                f"conv={fmt_conv(r_ex['conv_t'])}"
            )

    print(f"  HOLDS ({len(holds)}/{len(sweep)}):")
    for h in holds:
        print(f"    ✓ {h}")
    print(f"  BREAKS ({len(breaks)}/{len(sweep)}):")
    for b in breaks:
        print(f"    ✗ {b}")

    # Find approximate threshold: lowest noise where it breaks along axes
    print("\n  Threshold notes:")
    # fix τ=none, vary θ
    for th_name, s_th in THETA_NOISE.items():
        r_ex, r_qs = sweep[(th_name, "none")]
        ok = r_ex["rel_err"] < EX_ERR_OK and (r_qs["rel_err"] - r_ex["rel_err"]) > EX_BEATS_QS_MARGIN
        if not ok:
            print(
                f"    With σ_τ=0, claim fails at σ_θ ≥ {th_name} "
                f"({np.degrees(s_th):.1f}°): ex_err={r_ex['rel_err']:.1f}%"
            )
            break
    else:
        print("    With σ_τ=0, claim holds across all tested σ_θ.")

    for tau_name, s_tau in TAU_NOISE.items():
        r_ex, r_qs = sweep[("none", tau_name)]
        ok = r_ex["rel_err"] < EX_ERR_OK and (r_qs["rel_err"] - r_ex["rel_err"]) > EX_BEATS_QS_MARGIN
        if not ok:
            print(
                f"    With σ_θ=0, claim fails at σ_τ ≥ {tau_name} "
                f"({s_tau:.2f} N·m): ex_err={r_ex['rel_err']:.1f}%"
            )
            break
    else:
        print("    With σ_θ=0, claim holds across all tested σ_τ.")

    # realistic cell
    r_ex, r_qs = sweep[("med", "med")]
    print(
        f"\n  Realistic cell (1.0°, 0.10 N·m): "
        f"ex I_err={r_ex['rel_err']:.1f}%, qs I_err={r_qs['rel_err']:.1f}%, "
        f"conv={fmt_conv(r_ex['conv_t'])}, track_RMSE={r_ex['rmse_track']:.4f} rad"
    )
    if r_ex["rel_err"] >= EX_ERR_OK:
        print(
            "  → Raw finite-differenced vision noise at 500 Hz destroys θ̈; "
            "filtering (or slower vision-rate diffs) will be needed before real sensing."
        )
    else:
        print("  → Claim survives raw noise at this level (still no filter).")

    # --- Plot: Î(t) clean vs noisy for excited ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, tag in zip(axes, ["clean", "noisy"]):
        r = detailed[(tag, "excited")]
        ax.plot(np.arange(N_STEPS) * DT, r["I_hist"], label="Î")
        ax.axhline(r["I_true"], color="k", ls="--", label="I_true")
        if r["conv_t"] is not None:
            ax.axvline(r["conv_t"], color="C2", ls=":")
        ax.set_ylabel("I_hinge [kg·m²]")
        ax.set_title(
            f"excited / {tag}: I_err={r['rel_err']:.1f}%, conv={fmt_conv(r['conv_t'])}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Phase 0c: RLS with raw noisy θ (FD) + noisy τ — no filter")
    plt.tight_layout()
    out = "noise_sensitivity_results.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot → {out}")


if __name__ == "__main__":
    main()
