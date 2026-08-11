"""Per-family control of the recorded timestep, and the harness to evaluate it.

The audit found one family badly under-resolved at the 20 ms model step and a
second marginal at its fast tail:

    family            tau_v = I/b            model steps per tau_v(min)
    laptop            1.99 ms .. 36 ms       0.10      <- below one 2 ms integrator step
    nonlinear_hinge   Stribeck band 27.5 ms  1.37      <- marginal
    everything else                          >= 7.9

MuJoCo integrates correctly at 2 ms (the laptop trajectory matches an independent
stiff integration to 5.5e-4); the problem is that recording every 20 ms throws
that away. Within one recorded laptop transition the velocity has already relaxed
to terminal velocity, so inertia is close to unidentifiable from (s, a, s').

Two independent knobs, because reducing frame_skip alone bottoms out: at
frame_skip=1 the recorded step IS the integrator step, giving the laptop only
~3 samples per tau_v.

    frame_skip   recorded interval, in units of dyn.DT -> dt_model = DT * frame_skip
    substeps     integrator subdivision                -> mj_dt   = DT / substeps

``TimeResolution(frame_skip=10, substeps=1)`` is the current setting exactly, and
``test_time_resolution_matches_baseline`` asserts bit-identical output for it.

IMPORTANT, and the reason this is a measurement harness rather than a switch: the
excitation is a zero-order hold on the *recorded* grid, so shrinking dt_model also
shortens the action hold. A finer-dt dataset therefore differs from the baseline
in two ways at once -- sampling rate and action bandwidth -- and the ZOH-per-
transition invariant is what forces that coupling. Reported alongside the results.

    python3.10 -m latent_mechanics.mechanisms.time_resolution --help
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.config import ExperimentConfig
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.rollout import (
    MechanismEpisodes,
    limit_margin_for,
    near_limit_mask,
)


@dataclass(frozen=True)
class TimeResolution:
    """How finely one family is integrated and recorded."""

    frame_skip: int = 10
    substeps: int = 1

    @property
    def dt_model(self) -> float:
        return dyn.DT * self.frame_skip

    @property
    def mj_dt(self) -> float:
        return dyn.DT / self.substeps

    @property
    def mj_steps_per_transition(self) -> int:
        return self.frame_skip * self.substeps

    def label(self) -> str:
        return (f"dt_model={1000*self.dt_model:g}ms "
                f"mj_dt={1000*self.mj_dt:g}ms "
                f"({self.mj_steps_per_transition} steps/transition)")


BASELINE = TimeResolution(frame_skip=10, substeps=1)

# Per-family overrides. EMPTY BY DEFAULT: importing this module changes nothing.
# Populate it (or pass overrides explicitly) to regenerate a family finer.
FRAME_SKIP_OVERRIDES: dict[str, TimeResolution] = {}


def resolution_for(family: str, default: TimeResolution = BASELINE,
                   overrides: dict[str, TimeResolution] | None = None) -> TimeResolution:
    return (overrides if overrides is not None else FRAME_SKIP_OVERRIDES).get(
        family, default)


# ---------------------------------------------------------------------------
# Rollout at an arbitrary resolution
# ---------------------------------------------------------------------------

def rollout_at_resolution(
    params: lib.MechanismParams,
    cfg: ExperimentConfig,
    n_episodes: int,
    episode_seconds: float,
    res: TimeResolution,
    seed: int = 0,
    episode_offset: int = 0,
    exclude_near_limit: bool = True,
) -> MechanismEpisodes:
    """One instance, integrated at ``res.mj_dt`` and recorded every ``res.dt_model``.

    Deliberately does not reuse ``simulate_mechanism`` + ``transitions_from_log``:
    both read ``dyn.DT`` from module scope, and the honest way to vary the
    timestep is to make it explicit here rather than to monkeypatch a global that
    every other stage also reads. The recorded quantities are identical in
    meaning -- state after the interval, one constant action across it.
    """
    n_steps_original = int(round(episode_seconds / dyn.DT))
    # Block k spans [k*dt_model, (k+1)*dt_model). transitions_from_log starts at
    # logged index frame_skip-1, i.e. the state AFTER block 0, and strides by one
    # block -- so the first recorded transition is block 0's end to block 1's end,
    # carrying block 1's action, and there are n_blocks - 1 of them. Matching that
    # exactly is what makes TimeResolution(10, 1) a drop-in for the current path.
    n_blocks = n_steps_original // res.frame_skip
    n_transitions = n_blocks - 1
    gt = lib.ground_truth(lib.build_model(params), params)
    _, _, jid0 = lib.joint_info(lib.build_model(params))
    m0 = lib.build_model(params)
    lo = float(m0.jnt_range[jid0][0]); hi = float(m0.jnt_range[jid0][1])

    S, A, N, E = [], [], [], []
    for ep in range(episode_offset, episode_offset + n_episodes):
        rng = np.random.default_rng(seed * 7717 + (params.mechanism_id + 1) * 1009 + ep)
        # control_dt = frame_skip * dyn.DT = dt_model, so exactly one action per
        # recorded transition -- the same ZOH invariant Stage 1 relies on.
        profile = lib.scaled_profile(cfg.excitation, rng, n_steps_original,
                                     res.frame_skip, params)
        model = lib.build_model(params)
        model.opt.timestep = res.mj_dt
        data = mujoco.MjData(model)
        qadr, dof, _ = lib.joint_info(model)
        perts = lib.perturbations_for(params)
        for p in perts:
            p.reset(model)

        block_end = np.zeros((n_blocks, 2))    # state after each block
        block_tau = np.zeros(n_blocks)
        t = 0.0
        for k in range(n_blocks):
            tau = float(profile.values[min(k, len(profile.values) - 1)])
            for _ in range(res.mj_steps_per_transition):
                qi, vi = float(data.qpos[qadr]), float(data.qvel[dof])
                extra = 0.0
                for p in perts:
                    p.update_model(t, model)
                    extra += p.extra_torque(t, qi, vi)
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dof] = tau + extra
                mujoco.mj_step(model, data)
                t += res.mj_dt
            block_end[k] = (float(data.qpos[qadr]), float(data.qvel[dof]))
            block_tau[k] = tau

        s, ns = block_end[:-1], block_end[1:]
        actions = block_tau[1:].reshape(-1, 1)
        keep = (~near_limit_mask(s, ns, lo, hi) if exclude_near_limit
                else np.ones(len(s), bool))
        if not keep.any():
            continue
        S.append(s[keep].astype(np.float32))
        A.append(actions[keep].astype(np.float32))
        N.append(ns[keep].astype(np.float32))
        E.append(np.full(int(keep.sum()), ep, dtype=np.int32))

    empty = np.zeros((0, 2), np.float32)
    return MechanismEpisodes(
        params=params,
        state=np.concatenate(S) if S else empty,
        action=np.concatenate(A) if A else np.zeros((0, 1), np.float32),
        next_state=np.concatenate(N) if N else empty,
        episode_id=np.concatenate(E) if E else np.zeros(0, np.int32),
        gt=gt,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def timescale_table(
    families: list[str], overrides: dict[str, TimeResolution], n: int = 300,
    seed: int = 12345,
) -> list[dict]:
    """tau_v-to-recorded-step ratio per family, at baseline and at the override."""
    rows = []
    for fam in families:
        rng = np.random.default_rng(seed)
        tv = []
        for i in range(n):
            p = lib.sample_params(fam, rng, i)
            gt = lib.ground_truth(lib.build_model(p), p)
            if gt["damping"] > 0:
                tv.append(gt["inertia"] / gt["damping"])
        tv = np.array(tv)
        res = resolution_for(fam, BASELINE, overrides)
        extra = {}
        if fam == "nonlinear_hinge":
            # The binding timescale here is the Stribeck band, not I/b.
            rng2 = np.random.default_rng(seed)
            band = []
            for i in range(n):
                p = lib.sample_params(fam, rng2, i)
                gt = lib.ground_truth(lib.build_model(p), p)
                a = max(gt["frictionloss"], 1e-9) / gt["inertia"]
                band.append(p.extra["stribeck_v"] / a)
            extra["binding_timescale_min"] = float(np.min(band))
            extra["binding_name"] = "Stribeck band"
        else:
            extra["binding_timescale_min"] = float(tv.min())
            extra["binding_name"] = "tau_v = I/b"
        rows.append({
            "family": fam,
            "tau_v_min": float(tv.min()), "tau_v_median": float(np.median(tv)),
            **extra,
            "baseline_dt_model": BASELINE.dt_model,
            "new_dt_model": res.dt_model,
            "new_mj_dt": res.mj_dt,
            "steps_per_binding_baseline":
                extra["binding_timescale_min"] / BASELINE.dt_model,
            "steps_per_binding_new":
                extra["binding_timescale_min"] / res.dt_model,
            "steps_per_median_tau_v_baseline":
                float(np.median(tv)) / BASELINE.dt_model,
            "steps_per_median_tau_v_new": float(np.median(tv)) / res.dt_model,
            "resolution": res.label(),
        })
    return rows


RESOLVED_STEPS = 5.0   # recorded samples per timescale below which it is unresolved


def print_timescale_table(rows: list[dict]) -> None:
    print(f"{'family':17s} {'binding timescale':>18s} {'fastest':>9s} "
          f"{'base':>7s} {'new':>7s} | {'median tau_v':>12s} {'base':>7s} {'new':>7s}  "
          f"{'dt_model':>9s}")
    print("-" * 112)
    for r in rows:
        flag = "" if r["steps_per_binding_new"] >= RESOLVED_STEPS else "  UNRESOLVED"
        print(f"{r['family']:17s} {r['binding_name']:>18s} "
              f"{1000*r['binding_timescale_min']:>7.2f}ms "
              f"{r['steps_per_binding_baseline']:>7.2f} "
              f"{r['steps_per_binding_new']:>7.2f} | "
              f"{1000*r['tau_v_median']:>10.1f}ms "
              f"{r['steps_per_median_tau_v_baseline']:>7.2f} "
              f"{r['steps_per_median_tau_v_new']:>7.2f}  "
              f"{1000*r['new_dt_model']:>7.1f}ms{flag}")
    print(f"\n  Columns are RECORDED samples per timescale; >= {RESOLVED_STEPS:g} is "
          f"treated as resolved.")
    print("  'fastest' is the worst instance in the family; 'median tau_v' is the "
          "typical one.")
    print("  dt_model cannot go below the integrator step (dyn.DT = 2 ms) by "
          "reducing frame_skip alone:")
    print("  a family whose fastest timescale is under 2 ms needs a finer "
          "integrator AND an")
    print("  excitation grid to match, since the action is a zero-order hold on "
          "the recorded grid.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="laptop,nonlinear_hinge")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--substeps", type=int, default=1)
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    res = TimeResolution(frame_skip=a.frame_skip, substeps=a.substeps)
    ov = {f: res for f in fams}
    print(f"Proposed override for {fams}: {res.label()}\n")
    print_timescale_table(timescale_table(list(lib.FAMILY_ORDER), ov))


if __name__ == "__main__":
    main()
