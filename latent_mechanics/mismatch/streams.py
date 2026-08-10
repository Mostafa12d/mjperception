"""
Building the interaction streams each experiment feeds to the estimators.

One stream = one unseen door, simulated under a given plant perturbation, then
passed through a given sensor pipeline. The important invariant is that for a
fixed door and severity level, *every method sees byte-identical input*: the
rollout is generated once and the same transition list is handed to the
no-adaptation control, the latent adaptor and both RLS variants.

Two arrays come out of each stream:

``transitions``  what the estimator observes -- possibly noisy, quantised,
                 dropped or delayed.
``clean_next``   what actually happened -- the true next state from the
                 simulator, used only for scoring.

Keeping those separate is what lets Experiment 1 measure whether an estimator
still recovers the right dynamics *despite* bad sensing, instead of just
measuring the sensor. Scoring against the noisy reading would put a floor of
sigma under every method and hide the effect entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

import run_door_dynamics_validation as dyn
from latent_mechanics import door_sampler
from latent_mechanics.config import ExperimentConfig
from latent_mechanics.data_gen import JOINT_RANGE, LIMIT_MARGIN, transitions_from_log
from latent_mechanics.door_sampler import DoorParams
from latent_mechanics.excitation import sample_profile
from latent_mechanics.mismatch.perturbations import PlantPerturbation
from latent_mechanics.mismatch.sensors import SensorPipeline
from latent_mechanics.mismatch.simulate import simulate_perturbed

Transition = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class DoorStream:
    """One door's worth of interaction, as the estimators will consume it."""

    door_id: int
    params: dict[str, float]
    transitions: list[Transition]
    clean_state: np.ndarray  # (T, 2) true states, for the motion scale
    clean_next: np.ndarray  # (T, 2) true next states, for scoring
    observed_next: np.ndarray  # (T, 2) what was fed in as the target
    boundaries: list[int]
    perturb_torque_rms: float = 0.0
    extras: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    def motion_scale(self) -> np.ndarray:
        """RMS true one-step change per dimension.

        Errors are normalised by this. Without it the study is confounded: a
        perturbation that slows the door down (more friction, say) shrinks every
        absolute error and masquerades as an accuracy improvement. Normalised
        error is the fraction of the actual motion left unexplained, so 1.0 means
        no better than predicting "nothing changes".
        """
        d = self.clean_next - self.clean_state
        return np.sqrt(np.mean(d**2, axis=0)) if len(d) else np.ones(2)


def heldout_doors(cfg: ExperimentConfig, n_doors: int) -> list[DoorParams]:
    """The same held-out population Stage 1 reserved, regenerated deterministically."""
    _, held = door_sampler.sample_door_population(cfg.doors, cfg.sim.seed)
    return held[:n_doors]


def _near_limit(state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
    lo, hi = JOINT_RANGE
    at = lambda th: (th < lo + LIMIT_MARGIN) | (th > hi - LIMIT_MARGIN)
    return at(state[:, 0]) | at(next_state[:, 0])


def build_door_stream(
    params: DoorParams,
    cfg: ExperimentConfig,
    n_episodes: int,
    episode_seconds: float,
    frame_skip: int,
    perturbations: Sequence[PlantPerturbation] = (),
    sensors: SensorPipeline | None = None,
    seed: int = 0,
    exclude_near_limit: bool = True,
    episode_offset: int = 0,
) -> DoorStream:
    """Simulate one door and turn it into an observed transition stream.

    Excitation is drawn from a seed that depends only on the door and episode
    index, never on the perturbation, so the *same* torque profile is replayed
    at every severity level. Without that, a change in error could come from a
    different trajectory rather than from the mismatch.
    """
    n_steps = int(round(episode_seconds / dyn.DT))
    sensors = sensors or SensorPipeline()

    transitions: list[Transition] = []
    clean_chunks, clean_state_chunks, obs_chunks, boundaries = [], [], [], []
    perturb_sq, perturb_n = 0.0, 0
    total = 0

    for ep in range(episode_offset, episode_offset + n_episodes):
        # Excitation seed: door + episode only.
        prof_rng = np.random.default_rng((params.door_id + 1) * 100_003 + ep)
        profile = sample_profile(
            cfg.excitation, prof_rng, n_steps, frame_skip, params.frictionloss
        )
        model = door_sampler.build_model(params)
        log = simulate_perturbed(profile.as_fn(), model, n_steps, perturbations)
        perturb_sq += float((log.tau_perturb**2).sum())
        perturb_n += len(log.tau_perturb)

        tr = transitions_from_log(log.as_dict(), frame_skip)
        state, action, nxt = tr["state"], tr["action"], tr["next_state"]

        keep = ~_near_limit(state, nxt) if exclude_near_limit else np.ones(len(state), bool)
        if not keep.any():
            continue

        # Corrupt the STATE SEQUENCE once, then rebuild transitions from it, so a
        # shared state is not measured twice with independent noise.
        seq = np.concatenate([state, nxt[-1:]], axis=0)
        sensor_rng = np.random.default_rng(seed * 7919 + params.door_id * 101 + ep)
        obs_seq = sensors.apply(seq, frame_skip * dyn.DT, sensor_rng)
        obs_state, obs_next = obs_seq[:-1], obs_seq[1:]

        idx = np.nonzero(keep)[0]
        boundaries.append(total)
        for i in idx:
            transitions.append((obs_state[i], action[i], obs_next[i]))
        clean_chunks.append(nxt[idx])
        clean_state_chunks.append(state[idx])
        obs_chunks.append(obs_next[idx])
        total += len(idx)

    return DoorStream(
        door_id=params.door_id,
        params=door_sampler.ground_truth(door_sampler.build_model(params), params),
        transitions=transitions,
        clean_state=np.concatenate(clean_state_chunks) if clean_state_chunks else np.zeros((0, 2)),
        clean_next=np.concatenate(clean_chunks) if clean_chunks else np.zeros((0, 2)),
        observed_next=np.concatenate(obs_chunks) if obs_chunks else np.zeros((0, 2)),
        boundaries=boundaries,
        perturb_torque_rms=float(np.sqrt(perturb_sq / max(perturb_n, 1))),
    )


def clean_errors(log, stream: DoorStream) -> np.ndarray:
    """Re-score an ``AdaptationLog`` against the true next states.

    ``AdaptationLog`` stores ``error = prediction - observed_target``, and the
    prediction is fixed once made, so

        clean_error = error + (observed_target - clean_target)

    recovers the error against ground truth exactly, with no change to the
    Stage-2 adaptor or driver. Under an identity sensor the correction is zero
    and this returns ``log.error`` unchanged.
    """
    n = len(log.error)
    return log.error + (stream.observed_next[:n] - stream.clean_next[:n])


__all__ = ["DoorStream", "build_door_stream", "heldout_doors", "clean_errors"]


def frozen_predict_errors(adaptor, stream: DoorStream) -> np.ndarray:
    """Error of a *frozen* belief on a stream, updating nothing.

    Calls only ``predict``, never ``observe``, so the belief the adaptor ended
    adaptation with is evaluated as a fixed model. Works for any
    ``OnlineAdaptor`` -- latent or RLS -- which is what keeps the comparison
    symmetric.

    This is the metric that separates the two things Experiment 1 conflates. A
    stale or dropped reading makes the *instantaneous* prediction wrong no
    matter how good your model is: you are being asked to predict forward from a
    state that is already out of date, which costs every method the same
    bookkeeping offset (~k steps of motion for latency, ~sqrt(p) for dropout).
    Evaluating the learned belief afterwards on clean data asks the question we
    actually care about: did the corrupted stream poison what you learned?
    """
    errs = []
    for (s_obs, a, _), truth in zip(stream.transitions, stream.clean_next):
        errs.append(adaptor.predict(s_obs, a) - truth)
    return np.asarray(errs)
