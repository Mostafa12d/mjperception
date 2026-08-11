"""
Phase 0d: same noise sweep as Phase 0c, but θ̇ / θ̈ from Savitzky–Golay
on noisy θ_meas instead of raw finite differences.

Design:
  - θ sampled at VISION_HZ=30 Hz with Gaussian noise (vision-like)
  - Centered Savitzky–Golay on that stream → θ̇, θ̈ (half-window delay)
  - Physics-rate noisy θ for position feedback (same idea as Phase 0c)
  - KD reduced 16→8: full KD with ~0.3 s delayed velocity stalls the door
    (documented; not a Kalman filter)
  - RLS / door / noise grid otherwise match Phase 0c

Run:
    python3.10 run_door_noise_savgol.py
"""

from __future__ import annotations

from collections import deque

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from scipy.signal import savgol_coeffs

from run_door_dynamics_validation import (
    DT,
    HANDLE_DIST,
    load_model,
    true_hinge_inertia,
    hinge_torque_from_handle_force,
    tangential_direction,
    rls_init,
    rls_step,
)
from run_door_adaptive_impedance import (
    CREEP_TORQUE,
    RLS_LAM,
    I_HAT_INIT,
    MU_HAT_INIT,
    VEL_THRESH,
    T_RAMP,
    DITHER_AMP,
    DITHER_FREQ,
    reference_smooth,
    TAU_MAX,
)

# Position gain unchanged; damping reduced for delayed SG velocity
KP = 40.0
KD_SAVGOL = 8.0

T_END = 6.0
N_STEPS = int(T_END / DT)

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

EX_ERR_OK = 20.0
EX_BEATS_QS_MARGIN = 10.0

VISION_HZ = 30.0
VISION_DT = 1.0 / VISION_HZ
VISION_EVERY = max(1, int(round(VISION_DT / DT)))
SG_WINDOW = 21  # 0.7 s at 30 Hz; delay ≈ 0.33 s
SG_POLY = 3
SG_HALF = (SG_WINDOW - 1) // 2


class VisionRateSavGol:
    def __init__(self, window: int, poly: int, dt_vis: float):
        assert window % 2 == 1 and window > poly
        self.window = window
        self.buf: deque[float] = deque(maxlen=window)
        self.c0 = savgol_coeffs(window, poly, deriv=0, delta=dt_vis)
        self.c1 = savgol_coeffs(window, poly, deriv=1, delta=dt_vis)
        self.c2 = savgol_coeffs(window, poly, deriv=2, delta=dt_vis)

    def push(self, theta_meas: float) -> tuple[float, float, float, bool]:
        self.buf.append(float(theta_meas))
        if len(self.buf) < self.window:
            return float(theta_meas), 0.0, 0.0, False
        x = np.asarray(self.buf, dtype=float)
        return float(self.c0 @ x), float(self.c1 @ x), float(self.c2 @ x), True


def impedance_torque_savgol(
    th, thd, th_d, thd_d, thdd_d, I_hat, mu_hat, dither=0.0
) -> float:
    I_use = max(I_hat, 0.5)
    mu_use = max(mu_hat, 0.0)
    tau = (
        I_use * thdd_d
        + mu_use * np.sign(thd)
        + KP * (th_d - th)
        + KD_SAVGOL * (thd_d - thd)
        + dither
    )
    return float(np.clip(tau, -TAU_MAX, TAU_MAX))


def run_once(
    excited: bool,
    sigma_theta: float,
    sigma_tau: float,
    seed: int,
) -> dict:
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
    sg = VisionRateSavGol(SG_WINDOW, SG_POLY, VISION_DT)
    tau_vis_buf: deque[float] = deque(maxlen=SG_HALF + 1)

    t = np.arange(N_STEPS) * DT
    I_hat_log = np.zeros(N_STEPS)
    track_err = np.zeros(N_STEPS)

    thd_m, thdd_m = 0.0, 0.0
    sg_ready = False

    for i in range(N_STEPS):
        th_true = float(data.qpos[hinge_qpos])
        # physics-rate noisy angle for position loop (Phase 0c style)
        th_phys = th_true + rng.normal(0.0, sigma_theta)

        if i % VISION_EVERY == 0:
            th_vis = th_true + rng.normal(0.0, sigma_theta)
            _, thd_c, thdd_c, ready = sg.push(th_vis)
            if ready:
                thd_m, thdd_m = thd_c, thdd_c
                sg_ready = True

        I_hat = float(rls.theta[0])
        mu_hat = float(rls.theta[1])

        if excited:
            th_d, thd_d, thdd_d = reference_smooth(t[i])
            dither = DITHER_AMP * np.sin(2.0 * np.pi * DITHER_FREQ * t[i])
            thd_fb = thd_m if sg_ready else 0.0
            tau = impedance_torque_savgol(
                th_phys, thd_fb, th_d, thd_d, thdd_d, I_hat, mu_hat, dither
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

        if i % VISION_EVERY == 0:
            tau_vis_buf.append(tau_m)
            near_limit = th_true > 2.09 - 0.05 or th_true < -0.17 + 0.05
            moving = abs(thd_m) > VEL_THRESH and t[i] < T_RAMP + 0.3
            if (
                sg_ready
                and len(tau_vis_buf) == tau_vis_buf.maxlen
                and moving
                and not near_limit
                and abs(thdd_m) < 15.0
            ):
                phi = np.array([thdd_m, np.sign(thd_m)])
                rls = rls_step(rls, phi, float(tau_vis_buf[0]))
                rls.theta[0] = float(np.clip(rls.theta[0], 0.1, 50.0))

        I_hat_log[i] = float(rls.theta[0])
        track_err[i] = (th_d - float(data.qpos[hinge_qpos])) if excited else 0.0

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
        excited=excited,
        sigma_theta=sigma_theta,
        sigma_tau=sigma_tau,
    )


def fmt_conv(conv_t: float | None) -> str:
    return f"{conv_t:.2f}s" if conv_t is not None else "never"


def main() -> None:
    print("Phase 0d: Savitzky–Golay θ̇/θ̈ (vision-rate) vs Phase 0c raw FD\n")
    print(
        f"  Vision {VISION_HZ:.0f} Hz, SG window={SG_WINDOW} "
        f"({SG_WINDOW / VISION_HZ:.2f} s), poly={SG_POLY}, "
        f"delay={SG_HALF / VISION_HZ:.2f} s"
    )
    print(f"  Impedance: KP={KP}, KD={KD_SAVGOL} (was 16; lowered for SG delay)")
    print("  No Kalman. Same RLS / noise grid as Phase 0c.\n")

    print("=" * 72)
    print("1) Detailed: clean vs realistic (σ_θ=1.0°, σ_τ=0.10 N·m)")
    print("=" * 72)
    detailed = {}
    for tag, s_th, s_tau in [
        ("clean", 0.0, 0.0),
        ("noisy", THETA_NOISE["med"], TAU_NOISE["med"]),
    ]:
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

    print("\n" + "=" * 72)
    print("2) Sweep: excited I_err%/conv/track_RMSE  (QS I_err% in paren)")
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

    print("\n" + "=" * 72)
    print("3) Operating envelope (claim holds vs breaks)")
    print("=" * 72)
    print(
        f"  Criterion: excited I_err < {EX_ERR_OK:.0f}% AND "
        f"qs_err - ex_err > {EX_BEATS_QS_MARGIN:.0f} pp\n"
    )
    holds, breaks = [], []
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
        print(f"    + {h}")
    print(f"  BREAKS ({len(breaks)}/{len(sweep)}):")
    for b in breaks:
        print(f"    - {b}")

    print("\n  Threshold notes:")
    for th_name, s_th in THETA_NOISE.items():
        # claim must hold for ALL tau levels at this theta to call it safe
        ok_all = all(
            sweep[(th_name, tn)][0]["rel_err"] < EX_ERR_OK
            and (sweep[(th_name, tn)][1]["rel_err"] - sweep[(th_name, tn)][0]["rel_err"])
            > EX_BEATS_QS_MARGIN
            for tn in TAU_NOISE
        )
        r_ex, _ = sweep[(th_name, "med")]
        status = "HOLDS" if ok_all else "BREAKS"
        print(
            f"    σ_θ={th_name} ({np.degrees(s_th):.1f}°) across τ grid: {status} "
            f"(at τ=med, ex_err={r_ex['rel_err']:.1f}%)"
        )

    for cell_name, th_k, tau_k in [
        ("Realistic (1.0°, 0.10 N·m)", "med", "med"),
        ("Low vision (0.5°, 0.05 N·m)", "low", "low"),
        ("High vision (2.0°, 0.10 N·m)", "high", "med"),
    ]:
        r_ex, r_qs = sweep[(th_k, tau_k)]
        ok = (
            r_ex["rel_err"] < EX_ERR_OK
            and (r_qs["rel_err"] - r_ex["rel_err"]) > EX_BEATS_QS_MARGIN
        )
        print(
            f"\n  {cell_name}: ex I_err={r_ex['rel_err']:.1f}%, "
            f"qs I_err={r_qs['rel_err']:.1f}%, conv={fmt_conv(r_ex['conv_t'])}, "
            f"track_RMSE={r_ex['rmse_track']:.4f} rad  "
            f"[{'HOLDS' if ok else 'BREAKS'}]"
        )

    print("\n  vs Phase 0c (raw FD @ 500 Hz): claim broke at σ_θ ≥ 0.5°.")
    print("  Phase 0d (SavGol @ 30 Hz): see HOLDS/BREAKS above.")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, tag in zip(axes, ["clean", "noisy"]):
        r = detailed[(tag, "excited")]
        ax.plot(np.arange(N_STEPS) * DT, r["I_hist"], label="Î")
        ax.axhline(r["I_true"], color="k", ls="--", label="I_true")
        if r["conv_t"] is not None:
            ax.axvline(r["conv_t"], color="C2", ls=":")
        ax.set_ylabel("I_hinge [kg·m²]")
        ax.set_title(
            f"excited / {tag} + SavGol: I_err={r['rel_err']:.1f}%, "
            f"conv={fmt_conv(r['conv_t'])}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        f"Phase 0d: SavGol @ {VISION_HZ:.0f} Hz (win={SG_WINDOW}, KD={KD_SAVGOL})"
    )
    plt.tight_layout()
    out = "noise_savgol_results.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot → {out}")


if __name__ == "__main__":
    main()
