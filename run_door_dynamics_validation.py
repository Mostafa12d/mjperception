"""
Dynamics validation for the MuJoCo door (vision deferred).

Builds on run_door_estimation.py with:
  1. Handle F/T → hinge torque reconstruction vs oracle qfrc_applied
  2. Ablation table: Phase-0 2-param regressor vs 3-param (+ viscous damping)
  3. Parameter sweeps (density, friction, excitation) for structural claim
  4. Online RLS (forgetting-factor) with oracle kinematics

Vision is intentionally NOT used here. See vision_theta_interface.py for the
parked drop-in θ̂ API to wire later.

Run:
    python3.10 run_door_dynamics_validation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

MODEL_PATH = "door.xml"
DT = 0.002
T_END = 6.0
N_STEPS = int(T_END / DT)
HANDLE_DIST = 0.85
DEFAULT_STICTION_GUESS = 3.5
DEFAULT_DENSITY = 600.0
DEFAULT_FRICTIONLOSS = 3.0
DEFAULT_DAMPING = 0.2
DEFAULT_EXCITATION_AMP = 6.0
DEFAULT_EXCITATION_FREQ = 0.5  # Hz


# ---------------------------------------------------------------------------
# Torque profiles
# ---------------------------------------------------------------------------

def make_tau_quasistatic(stiction_guess: float) -> Callable[[float], float]:
    def tau(t: float) -> float:
        if t < 0.3:
            return stiction_guess * (t / 0.3)
        return stiction_guess
    return tau


def make_tau_excited(
    stiction_guess: float, amp: float, freq_hz: float
) -> Callable[[float], float]:
    def tau(t: float) -> float:
        return stiction_guess + amp * np.sin(2 * np.pi * freq_hz * t)
    return tau


def tangential_direction(theta: float) -> np.ndarray:
    return np.array([-np.sin(theta), np.cos(theta), 0.0])


# ---------------------------------------------------------------------------
# Ground truth from MuJoCo mass properties
# ---------------------------------------------------------------------------

def true_hinge_inertia(model: mujoco.MjModel) -> dict:
    bid = model.body("door").id
    m = float(model.body_mass[bid])
    I_prin = np.array(model.body_inertia[bid], dtype=float)
    com = np.array(model.body_ipos[bid], dtype=float)
    iquat = np.array(model.body_iquat[bid], dtype=float)
    R = np.zeros((3, 3))
    mujoco.mju_quat2Mat(R.ravel(), iquat)
    I_com_body = R @ np.diag(I_prin) @ R.T
    I_zz_com = float(I_com_body[2, 2])
    I_hinge = I_zz_com + m * (com[0] ** 2 + com[1] ** 2)
    return {
        "mass": m,
        "com_body": com,
        "I_zz_com": I_zz_com,
        "I_hinge": I_hinge,
        "frictionloss": float(model.dof_frictionloss[0]),
        "damping": float(model.dof_damping[0]),
    }


def load_model(
    density: float = DEFAULT_DENSITY,
    frictionloss: float = DEFAULT_FRICTIONLOSS,
    damping: float = DEFAULT_DAMPING,
) -> mujoco.MjModel:
    """Load door.xml and override mass-scale / joint friction / damping."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    bid = model.body("door").id
    scale = density / DEFAULT_DENSITY
    model.body_mass[bid] *= scale
    model.body_inertia[bid] *= scale
    model.dof_frictionloss[0] = frictionloss
    model.dof_damping[0] = damping
    # Recompute derived quantities (M, etc.) consistent with new mass props
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    return model


# ---------------------------------------------------------------------------
# Simulation: oracle τ and F/T-reconstructed τ
# ---------------------------------------------------------------------------

def hinge_torque_from_handle_force(
    handle_pos_world: np.ndarray,
    hinge_pos_world: np.ndarray,
    hinge_axis_world: np.ndarray,
    force_world: np.ndarray,
) -> float:
    """τ_hinge = (r_handle × F) · â_hinge  — wrist/handle F/T path."""
    r = handle_pos_world - hinge_pos_world
    tau_vec = np.cross(r, force_world)
    axis = hinge_axis_world / (np.linalg.norm(hinge_axis_world) + 1e-12)
    return float(np.dot(tau_vec, axis))


def simulate(tau_fn: Callable[[float], float], model: mujoco.MjModel | None = None) -> dict:
    """Forward-sim with oracle kinematics; log oracle and F/T hinge torques."""
    if model is None:
        model = load_model()

    data = mujoco.MjData(model)
    assert abs(model.opt.timestep - DT) < 1e-12

    hinge_qpos = model.joint("hinge").qposadr[0]
    hinge_dof = model.joint("hinge").dofadr[0]
    handle_sid = model.site("handle").id
    door_bid = model.body("door").id
    gt = true_hinge_inertia(model)

    t = np.arange(N_STEPS) * DT
    theta = np.zeros(N_STEPS)
    theta_dot = np.zeros(N_STEPS)
    theta_ddot = np.zeros(N_STEPS)
    tau_oracle = np.zeros(N_STEPS)
    tau_ft = np.zeros(N_STEPS)
    force_log = np.zeros((N_STEPS, 3))
    ncon = np.zeros(N_STEPS, dtype=int)

    for i in range(N_STEPS):
        th = float(data.qpos[hinge_qpos])
        tau_des = float(tau_fn(t[i]))
        force = (tau_des / HANDLE_DIST) * tangential_direction(th)

        # World hinge frame: body origin is the hinge; axis is body z
        hinge_pos = data.xpos[door_bid].copy()
        # body z-axis in world from xmat (row-major 3x3)
        xmat = data.xmat[door_bid].reshape(3, 3)
        hinge_axis = xmat[:, 2].copy()
        handle_pos = data.site_xpos[handle_sid].copy()

        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data, force, np.zeros(3), handle_pos, door_bid, data.qfrc_applied,
        )
        tau_h_oracle = float(data.qfrc_applied[hinge_dof])
        tau_h_ft = hinge_torque_from_handle_force(handle_pos, hinge_pos, hinge_axis, force)

        mujoco.mj_step(model, data)

        theta[i] = float(data.qpos[hinge_qpos])
        theta_dot[i] = float(data.qvel[hinge_dof])
        theta_ddot[i] = float(data.qacc[hinge_dof])
        tau_oracle[i] = tau_h_oracle
        tau_ft[i] = tau_h_ft
        force_log[i] = force
        ncon[i] = int(data.ncon)

    return dict(
        t=t,
        theta=theta,
        theta_dot=theta_dot,
        theta_ddot=theta_ddot,
        tau_oracle=tau_oracle,
        tau_ft=tau_ft,
        force=force_log,
        ncon=ncon,
        gt=gt,
    )


def moving_mask(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    thresh: float = 0.02,
    joint_range: tuple[float, float] = (-0.17, 2.09),
    limit_margin: float = 0.05,
) -> np.ndarray:
    """Moving samples away from joint limits (limit contact breaks the free 1-DOF model)."""
    mask = np.abs(theta_dot) > thresh
    mask[:20] = False
    lo, hi = joint_range
    near_limit = (theta < lo + limit_margin) | (theta > hi - limit_margin)
    mask &= ~near_limit
    return mask


# ---------------------------------------------------------------------------
# Batch least squares
# ---------------------------------------------------------------------------

def fit_params(
    theta_dot: np.ndarray,
    theta_ddot: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    include_damping: bool = False,
) -> dict:
    """
    Phase-0:  tau = I * thdd + mu * sign(thd)
    +damping: tau = I * thdd + mu * sign(thd) + b * thd
    """
    cols = [theta_ddot[mask], np.sign(theta_dot[mask])]
    names = ["I_hinge", "mu_k"]
    if include_damping:
        cols.append(theta_dot[mask])
        names.append("b")
    Phi = np.stack(cols, axis=1) if mask.any() else np.zeros((0, len(names)))
    Y = tau[mask]
    if Phi.shape[0] < Phi.shape[1] + 2:
        return dict(
            params=np.full(len(names), np.nan),
            names=names,
            cond=np.inf,
            rmse=np.nan,
            n=int(mask.sum()),
        )
    params, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    cond = float(np.linalg.cond(Phi))
    resid = Y - Phi @ params
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return dict(params=params, names=names, cond=cond, rmse=rmse, n=int(mask.sum()), Phi=Phi, Y=Y)


def evaluate_fit(log: dict, tau_key: str, include_damping: bool = False) -> dict:
    gt = log["gt"]
    I_true = gt["I_hinge"]
    mask = moving_mask(log["theta"], log["theta_dot"])
    n_limit = int(
        (
            (log["theta"] < -0.17 + 0.05) | (log["theta"] > 2.09 - 0.05)
        ).sum()
    )
    fit = fit_params(
        log["theta_dot"], log["theta_ddot"], log[tau_key], mask, include_damping
    )
    I_hat = float(fit["params"][0]) if np.isfinite(fit["params"][0]) else np.nan
    rel_err = abs(I_hat - I_true) / I_true * 100.0 if np.isfinite(I_hat) else np.nan
    thdd_rms = (
        float(np.sqrt(np.mean(log["theta_ddot"][mask] ** 2))) if mask.any() else 0.0
    )
    out = {
        "I_true": I_true,
        "I_hat": I_hat,
        "mu_hat": float(fit["params"][1]) if len(fit["params"]) > 1 and np.isfinite(fit["params"][1]) else np.nan,
        "b_hat": float(fit["params"][2]) if include_damping and len(fit["params"]) > 2 and np.isfinite(fit["params"][2]) else np.nan,
        "mu_true": gt["frictionloss"],
        "b_true": gt["damping"],
        "rel_err": rel_err,
        "cond": fit["cond"],
        "rmse": fit["rmse"],
        "n_moving": fit["n"],
        "n_near_limit": n_limit,
        "thdd_rms": thdd_rms,
        "theta_end_deg": float(np.degrees(log["theta"][-1])),
        "theta_max_deg": float(np.degrees(np.max(log["theta"]))),
        "tau_ft_vs_oracle_rmse": float(
            np.sqrt(np.mean((log["tau_ft"] - log["tau_oracle"]) ** 2))
        ),
        "max_ncon": int(log["ncon"].max()),
        "hit_joint_limit": bool(np.max(log["theta"]) > 2.09 - 0.05),
    }
    return out


# ---------------------------------------------------------------------------
# Online RLS
# ---------------------------------------------------------------------------

@dataclass
class RLSState:
    theta: np.ndarray  # parameter vector
    P: np.ndarray      # covariance
    lam: float         # forgetting factor


def rls_init(n_params: int, delta: float = 1e3, lam: float = 0.995) -> RLSState:
    return RLSState(
        theta=np.zeros(n_params),
        P=delta * np.eye(n_params),
        lam=lam,
    )


def rls_step(state: RLSState, phi: np.ndarray, y: float) -> RLSState:
    lam = state.lam
    P = state.P
    th = state.theta
    phi = np.asarray(phi, dtype=float).reshape(-1)
    # gain: K = P φ / (λ + φᵀ P φ)
    Pphi = P @ phi
    denom = lam + float(phi @ Pphi)
    K = Pphi / denom
    err = float(y) - float(phi @ th)
    th_new = th + K * err
    # P ← (P - K φᵀ P) / λ
    P_new = (P - np.outer(K, phi) @ P) / lam
    P_new = 0.5 * (P_new + P_new.T)
    return RLSState(theta=th_new, P=P_new, lam=lam)


def run_rls(
    log: dict,
    tau_key: str = "tau_ft",
    include_damping: bool = False,
    lam: float = 0.995,
) -> dict:
    """Online RLS over the trajectory; returns final params + history of I_hat."""
    n_params = 3 if include_damping else 2
    state = rls_init(n_params, lam=lam)
    mask = moving_mask(log["theta"], log["theta_dot"])
    I_hist = np.full(N_STEPS, np.nan)
    I_true = log["gt"]["I_hinge"]

    for i in range(N_STEPS):
        if not mask[i]:
            I_hist[i] = state.theta[0]
            continue
        phi_list = [log["theta_ddot"][i], np.sign(log["theta_dot"][i])]
        if include_damping:
            phi_list.append(log["theta_dot"][i])
        phi = np.array(phi_list)
        state = rls_step(state, phi, float(log[tau_key][i]))
        I_hist[i] = state.theta[0]

    I_hat = float(state.theta[0])
    rel_err = abs(I_hat - I_true) / I_true * 100.0
    return dict(
        I_hat=I_hat,
        mu_hat=float(state.theta[1]),
        b_hat=float(state.theta[2]) if include_damping else np.nan,
        rel_err=rel_err,
        I_hist=I_hist,
        I_true=I_true,
        final_theta=state.theta.copy(),
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_row(label: str, r: dict, with_damping: bool = False) -> None:
    damp_str = ""
    if with_damping:
        damp_str = f"  b_hat={r['b_hat']:.3f} (true {r['b_true']:.3f})"
    print(
        f"  {label:28s}  I_err={r['rel_err']:7.1f}%  "
        f"I_hat={r['I_hat']:8.3f}  true={r['I_true']:8.3f}  "
        f"cond={r['cond']:8.1f}  thdd_rms={r['thdd_rms']:.4f}  "
        f"rmse={r['rmse']:.4f}{damp_str}"
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_ft_vs_oracle() -> None:
    print("=" * 72)
    print("1) Handle F/T → hinge torque vs oracle qfrc_applied")
    print("=" * 72)
    print("  τ_ft = ((p_handle - p_hinge) × F) · â_hinge")
    print("  τ_oracle = qfrc_applied[hinge] after mj_applyFT\n")

    for label, tau_fn in [
        ("quasi-static", make_tau_quasistatic(DEFAULT_STICTION_GUESS)),
        ("excited", make_tau_excited(DEFAULT_STICTION_GUESS, DEFAULT_EXCITATION_AMP, DEFAULT_EXCITATION_FREQ)),
    ]:
        log = simulate(tau_fn)
        err = log["tau_ft"] - log["tau_oracle"]
        print(f"  {label}:")
        print(f"    max |τ_ft - τ_oracle| = {np.max(np.abs(err)):.3e} N·m")
        print(f"    RMSE(τ_ft, τ_oracle)  = {np.sqrt(np.mean(err**2)):.3e} N·m")
        r_oracle = evaluate_fit(log, "tau_oracle", include_damping=False)
        r_ft = evaluate_fit(log, "tau_ft", include_damping=False)
        print(f"    LS on oracle τ: I_err={r_oracle['rel_err']:.1f}%, cond={r_oracle['cond']:.1f}")
        print(f"    LS on F/T τ:    I_err={r_ft['rel_err']:.1f}%, cond={r_ft['cond']:.1f}")
        print()


def section_ablation_table() -> None:
    print("=" * 72)
    print("2) Ablation table (oracle kinematics; F/T-reconstructed τ)")
    print("=" * 72)
    print("  Regressor A: τ = I·θ̈ + μ·sign(θ̇)")
    print("  Regressor B: τ = I·θ̈ + μ·sign(θ̇) + b·θ̇   (includes viscous damping)\n")

    rows = []
    for label, tau_fn in [
        ("quasi-static", make_tau_quasistatic(DEFAULT_STICTION_GUESS)),
        ("excited", make_tau_excited(DEFAULT_STICTION_GUESS, DEFAULT_EXCITATION_AMP, DEFAULT_EXCITATION_FREQ)),
    ]:
        log = simulate(tau_fn)
        for damp in (False, True):
            r = evaluate_fit(log, "tau_ft", include_damping=damp)
            tag = f"{label} / {'3-param' if damp else '2-param'}"
            print_row(tag, r, with_damping=damp)
            rows.append((tag, r))

    print("  Compact table:")
    print(f"  {'condition':28s} {'I_true':>8} {'I_hat':>8} {'err%':>7} {'cond':>10} {'thdd_rms':>8}")
    for tag, r in rows:
        print(
            f"  {tag:28s} {r['I_true']:8.3f} {r['I_hat']:8.3f} "
            f"{r['rel_err']:7.1f} {r['cond']:10.1f} {r['thdd_rms']:8.4f}"
        )
    print()
    print("  NOTE: 3-param quasi-static can show ~0% I error with enormous cond(Φ);")
    print("        that is numerical coincidence / collinearity, not good observability.")
    print("        Prefer cond(Φ) + excited 2-param as the structural evidence.\n")


def section_param_sweep() -> None:
    print("=" * 72)
    print("3) Parameter sweep (structural claim, not one-door luck)")
    print("=" * 72)
    print("  Each cell: excited I_err% / quasi-static I_err%  (2-param, τ_ft)\n")

    densities = [300.0, 600.0, 900.0]
    frictions = [1.5, 3.0, 5.0]
    amps = [3.0, 6.0, 9.0]
    freqs = [0.25, 0.5, 1.0]

    print("  --- density x frictionloss (amp=6, freq=0.5) ---")
    header = "dens/fric"
    print(f"  {header:>10}", end="")
    for f in frictions:
        print(f"  fric={f:<5}", end="")
    print()
    for dens in densities:
        print(f"  dens={dens:<5.0f}", end="")
        for fric in frictions:
            stiction = fric + 0.5
            log_qs = simulate(
                make_tau_quasistatic(stiction),
                model=load_model(density=dens, frictionloss=fric),
            )
            log_ex = simulate(
                make_tau_excited(stiction, DEFAULT_EXCITATION_AMP, DEFAULT_EXCITATION_FREQ),
                model=load_model(density=dens, frictionloss=fric),
            )
            r_qs = evaluate_fit(log_qs, "tau_ft")
            r_ex = evaluate_fit(log_ex, "tau_ft")
            print(f"  {r_ex['rel_err']:5.1f}/{r_qs['rel_err']:5.1f}", end="")
            if r_ex["hit_joint_limit"]:
                print("*", end="")
            else:
                print(" ", end="")
        print()
    print("  (* = excited run contacted joint limit; near-limit samples masked)\n")

    print("  --- excitation amplitude (density=600, fric=3, freq=0.5) ---")
    for amp in amps:
        stiction = DEFAULT_STICTION_GUESS
        log_ex = simulate(make_tau_excited(stiction, amp, DEFAULT_EXCITATION_FREQ))
        log_qs = simulate(make_tau_quasistatic(stiction))
        r_ex = evaluate_fit(log_ex, "tau_ft")
        r_qs = evaluate_fit(log_qs, "tau_ft")
        print(
            f"  amp={amp:.1f}: excited err={r_ex['rel_err']:.1f}%  "
            f"qs err={r_qs['rel_err']:.1f}%  "
            f"ex thdd_rms={r_ex['thdd_rms']:.4f}  cond_ex={r_ex['cond']:.1f}"
            f"{'  [HIT JOINT LIMIT]' if r_ex['hit_joint_limit'] else ''}"
        )
    print()

    print("  --- excitation frequency (density=600, fric=3, amp=6) ---")
    for freq in freqs:
        stiction = DEFAULT_STICTION_GUESS
        log_ex = simulate(make_tau_excited(stiction, DEFAULT_EXCITATION_AMP, freq))
        r_ex = evaluate_fit(log_ex, "tau_ft")
        print(
            f"  freq={freq:.2f} Hz: I_err={r_ex['rel_err']:.1f}%  "
            f"thdd_rms={r_ex['thdd_rms']:.4f}  cond={r_ex['cond']:.1f}  "
            f"theta_end={r_ex['theta_end_deg']:.1f}°"
            f"{'  [HIT JOINT LIMIT]' if r_ex['hit_joint_limit'] else ''}"
        )
    print()

    # sanity: excited should beat qs across density×friction grid
    failures = 0
    limit_hits = 0
    checks = 0
    for dens in densities:
        for fric in frictions:
            stiction = fric + 0.5
            r_qs = evaluate_fit(
                simulate(make_tau_quasistatic(stiction), model=load_model(density=dens, frictionloss=fric)),
                "tau_ft",
            )
            r_ex = evaluate_fit(
                simulate(
                    make_tau_excited(stiction, DEFAULT_EXCITATION_AMP, DEFAULT_EXCITATION_FREQ),
                    model=load_model(density=dens, frictionloss=fric),
                ),
                "tau_ft",
            )
            checks += 1
            if r_ex["hit_joint_limit"] or r_qs["hit_joint_limit"]:
                limit_hits += 1
            if not (np.isfinite(r_ex["rel_err"]) and np.isfinite(r_qs["rel_err"]) and r_ex["rel_err"] < r_qs["rel_err"]):
                failures += 1
                print(
                    f"  WARN: excited did not beat qs at dens={dens}, fric={fric} "
                    f"(ex={r_ex['rel_err']:.1f}%, qs={r_qs['rel_err']:.1f}%, "
                    f"limit={r_ex['hit_joint_limit']})"
                )
    print(
        f"  Sweep check: excited better than qs in {checks - failures}/{checks} cells "
        f"({limit_hits} cells contacted joint limit; those samples are masked but "
        f"short free-motion windows can still leave Φ poorly excited)."
    )
    print(
        "  NOTE: failures at low density / high amp / low freq are typically joint-limit "
        "contact (unmodeled constraint torque), not a refutation of the observability claim.\n"
    )


def section_online_rls() -> None:
    print("=" * 72)
    print("4) Online RLS (forgetting λ=0.995, oracle θ / F/T τ)")
    print("=" * 72)

    for label, tau_fn in [
        ("quasi-static", make_tau_quasistatic(DEFAULT_STICTION_GUESS)),
        ("excited", make_tau_excited(DEFAULT_STICTION_GUESS, DEFAULT_EXCITATION_AMP, DEFAULT_EXCITATION_FREQ)),
    ]:
        log = simulate(tau_fn)
        batch = evaluate_fit(log, "tau_ft", include_damping=False)
        online = run_rls(log, tau_key="tau_ft", include_damping=False, lam=0.995)
        online3 = run_rls(log, tau_key="tau_ft", include_damping=True, lam=0.995)

        # convergence: first time |I_hat - I_true|/I_true < 5% after moving starts
        I_true = online["I_true"]
        hist = online["I_hist"]
        mask = moving_mask(log["theta"], log["theta_dot"])
        conv_t = None
        for i in range(N_STEPS):
            if not mask[i] or not np.isfinite(hist[i]):
                continue
            if abs(hist[i] - I_true) / I_true < 0.05:
                # require it stays roughly good for a bit
                window = hist[i : min(i + 50, N_STEPS)]
                if np.all(np.isfinite(window)) and np.all(np.abs(window - I_true) / I_true < 0.10):
                    conv_t = i * DT
                    break

        print(f"  {label}:")
        print(f"    batch 2-param:  I_err={batch['rel_err']:.1f}%  I_hat={batch['I_hat']:.3f}")
        print(f"    RLS  2-param:  I_err={online['rel_err']:.1f}%  I_hat={online['I_hat']:.3f}  "
              f"μ_hat={online['mu_hat']:.3f}")
        print(f"    RLS  3-param:  I_err={online3['rel_err']:.1f}%  I_hat={online3['I_hat']:.3f}  "
              f"μ_hat={online3['mu_hat']:.3f}  b_hat={online3['b_hat']:.3f} "
              f"(true b={log['gt']['damping']:.3f})")
        if conv_t is not None:
            print(f"    RLS 2-param entered <5% I error at t≈{conv_t:.2f}s")
        else:
            print("    RLS 2-param never sustained <5% I error")
        print()


def section_vision_deferred() -> None:
    print("=" * 72)
    print("5) Vision: deferred")
    print("=" * 72)
    print("  Vision is parked until F/T + oracle-θ dynamics validation is solid.")
    print("  Drop-in interface for later: vision_theta_interface.py")
    print("  Planned order: ArUco in MuJoCo → compare oracle θ | θ̂ | filtered θ̂")
    print("  on the same trial, then RealSense / markerless.\n")


def main() -> None:
    print("\nMuJoCo door dynamics validation (vision deferred)\n")
    section_ft_vs_oracle()
    section_ablation_table()
    section_param_sweep()
    section_online_rls()
    section_vision_deferred()
    print("Done.")


if __name__ == "__main__":
    main()
