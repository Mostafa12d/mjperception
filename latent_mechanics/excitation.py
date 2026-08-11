"""
Torque excitation profiles for data collection.

Every profile is generated as a **zero-order hold on the model timestep grid**
(``frame_skip`` MuJoCo steps per hold). That matters: a learned transition
``(s_t, a_t) -> s_{t+1}`` is only well defined if the torque is constant across
the whole interval. MuJoCo still integrates at 500 Hz underneath.

The profiles deliberately mirror the excitation vocabulary the RLS baseline
already relies on (a bias above stiction plus an oscillation for observability),
so the two methods see comparable data.

Every episode starts from the closed door. Randomising the starting angle by
writing ``model.qpos0`` does *not* work: MuJoCo treats ``qpos0`` as the
reference configuration at which the body sits in its XML pose, so shifting it
decouples the joint coordinate from the door's geometric angle and silently
corrupts the handle moment arm. State-space coverage therefore has to come from
the torque signal, which is what the 'swing' profile is for -- a slow, large
oscillation that carries the door open and then back closed, so the model sees
both signs of velocity (where Coulomb friction flips).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baseline import run_door_dynamics_validation as dyn
from latent_mechanics.config import ExcitationConfig


@dataclass(frozen=True)
class TorqueProfile:
    """A zero-order-hold torque signal plus a description of how it was made."""

    values: np.ndarray  # (n_ctrl,) commanded hinge torque per control interval
    frame_skip: int
    kind: str
    meta: dict

    @property
    def control_dt(self) -> float:
        return self.frame_skip * dyn.DT

    def as_fn(self):
        """Callable ``t -> tau`` for ``run_door_dynamics_validation.simulate``.

        The step index is recovered from ``t`` exactly (``t`` is always an integer
        multiple of the MuJoCo timestep), so hold boundaries land where intended
        instead of drifting with floating-point error.
        """
        values = self.values
        skip = self.frame_skip
        n = len(values)

        def tau(t: float) -> float:
            step = int(round(t / dyn.DT))
            return float(values[min(step // skip, n - 1)])

        return tau


def _control_times(n_ctrl: int, control_dt: float) -> np.ndarray:
    return np.arange(n_ctrl) * control_dt


def _multisine(
    n_ctrl: int, control_dt: float, cfg: ExcitationConfig, bias: float,
    frictionloss: float, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    t = _control_times(n_ctrl, control_dt)
    n_sines = int(rng.integers(cfg.n_sines_range[0], cfg.n_sines_range[1] + 1))
    freqs = rng.uniform(*cfg.freq_range, size=n_sines)
    amps = rng.uniform(*cfg.amp_range, size=n_sines) / n_sines
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sines)
    sig = bias + sum(
        a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases)
    )
    return sig, {"n_sines": n_sines, "freqs": freqs.tolist(), "amps": amps.tolist()}


def _steps(
    n_ctrl: int, control_dt: float, cfg: ExcitationConfig, bias: float,
    frictionloss: float, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    amp = float(rng.uniform(*cfg.amp_range))
    sig = np.empty(n_ctrl)
    i = 0
    n_holds = 0
    while i < n_ctrl:
        hold_s = float(rng.uniform(*cfg.step_hold_range))
        n_hold = max(1, int(round(hold_s / control_dt)))
        sig[i : i + n_hold] = bias + rng.uniform(-amp, amp)
        i += n_hold
        n_holds += 1
    return sig, {"amp": amp, "n_holds": n_holds}


def _chirp(
    n_ctrl: int, control_dt: float, cfg: ExcitationConfig, bias: float,
    frictionloss: float, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    t = _control_times(n_ctrl, control_dt)
    duration = max(t[-1], control_dt)
    f0, f1 = sorted(rng.uniform(*cfg.freq_range, size=2))
    amp = float(rng.uniform(*cfg.amp_range))
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / duration)
    return bias + amp * np.sin(phase), {"f0": float(f0), "f1": float(f1), "amp": amp}


def _swing(
    n_ctrl: int, control_dt: float, cfg: ExcitationConfig, bias: float,
    frictionloss: float, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Slow, large oscillation that opens the door and pulls it back shut.

    ``bias`` is ignored: a nonzero mean would bias the swing toward one
    direction, and the point of this profile is symmetric coverage.
    """
    t = _control_times(n_ctrl, control_dt)
    freq = float(rng.uniform(*cfg.swing_freq_range))
    amp = float(rng.uniform(*cfg.swing_over_friction_range)) * max(frictionloss, 0.5)
    return amp * np.sin(2 * np.pi * freq * t), {"amp": amp, "freq": freq}


_PROFILES = {
    "multisine": _multisine, "steps": _steps, "chirp": _chirp, "swing": _swing,
}


def sample_profile(
    cfg: ExcitationConfig,
    rng: np.random.Generator,
    n_steps: int,
    frame_skip: int,
    frictionloss: float,
) -> TorqueProfile:
    """Draw one episode's torque signal.

    ``frictionloss`` scales the profile so that heavily sticking doors still
    break away -- a door that never moves contributes no information about its
    own mechanics.
    """
    kinds = list(cfg.profile_weights)
    weights = np.array([cfg.profile_weights[k] for k in kinds], dtype=float)
    kind = str(rng.choice(kinds, p=weights / weights.sum()))

    n_ctrl = int(np.ceil(n_steps / frame_skip))
    control_dt = frame_skip * dyn.DT
    bias = float(rng.uniform(*cfg.bias_over_friction_range)) * max(frictionloss, 0.5)

    values, meta = _PROFILES[kind](n_ctrl, control_dt, cfg, bias, frictionloss, rng)
    values = np.clip(values, -cfg.tau_clip, cfg.tau_clip)
    meta["bias"] = bias
    return TorqueProfile(
        values=values.astype(np.float64),
        frame_skip=frame_skip,
        kind=kind,
        meta=meta,
    )
