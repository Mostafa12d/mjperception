"""
Configuration for the latent-mechanics experiments.

Everything is a plain dataclass so the config is introspectable, type-checked at
construction, and serialisable into a checkpoint. YAML files only need to list
the fields they want to override; anything absent keeps the dataclass default.

    from latent_mechanics.config import load_config
    cfg = load_config("configs/latent_mechanics.yaml")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class DoorSamplingConfig:
    """Ranges for the randomised mechanics of one door instance.

    Each *door* is one draw from these ranges. A door keeps its physical
    parameters across all of its episodes -- that is what makes a single
    embedding vector able to describe it.
    """

    n_train_doors: int = 48
    # Doors generated but deliberately given no embedding row. They exist so the
    # stage-2 online-adaptation experiment has genuinely unseen mechanics ready.
    n_heldout_doors: int = 8

    model_paths: list[str] = field(default_factory=lambda: ["door.xml"])

    # Panel density [kg/m^3] -> scales mass and hinge inertia. Sampled log-uniform.
    density_range: tuple[float, float] = (200.0, 1400.0)
    # Coulomb friction at the hinge [N*m].
    frictionloss_range: tuple[float, float] = (0.5, 6.0)
    # Viscous damping [N*m*s/rad].
    damping_range: tuple[float, float] = (0.02, 1.5)
    # Torsional spring stiffness [N*m/rad]; 0 means a door with no self-closer.
    stiffness_range: tuple[float, float] = (0.0, 8.0)
    # Rest angle of that spring [rad].
    springref_range: tuple[float, float] = (-0.1, 0.6)
    # Fraction of doors forced to stiffness exactly 0 (plain doors, no closer).
    frac_no_spring: float = 0.3


@dataclass
class ExcitationConfig:
    """Torque profiles used to excite each episode.

    The commanded torque is a zero-order hold on the *model* timestep grid
    (``sim.frame_skip`` MuJoCo steps), so every recorded transition has exactly
    one constant action -- see ``data_gen.py``.
    """

    # Every episode starts from the closed door (see excitation.py for why the
    # starting angle cannot simply be randomised). The 'swing' profile is what
    # supplies closing-direction data: a slow large-amplitude oscillation that
    # drives the door open and then back shut.
    profile_weights: dict[str, float] = field(
        default_factory=lambda: {
            "multisine": 0.3, "steps": 0.2, "chirp": 0.2, "swing": 0.3,
        }
    )
    # Constant push, sampled relative to the door's own friction so the door
    # actually breaks away: bias ~ U(bias_over_friction) * frictionloss.
    bias_over_friction_range: tuple[float, float] = (0.6, 2.0)
    amp_range: tuple[float, float] = (1.0, 7.0)
    freq_range: tuple[float, float] = (0.2, 3.0)  # Hz
    n_sines_range: tuple[int, int] = (2, 4)
    step_hold_range: tuple[float, float] = (0.2, 1.2)  # s
    # 'swing' profile: amplitude relative to friction (must exceed 1 to break
    # away in both directions) and a deliberately low frequency.
    swing_over_friction_range: tuple[float, float] = (1.4, 3.5)
    swing_freq_range: tuple[float, float] = (0.15, 0.5)  # Hz
    tau_clip: float = 30.0  # N*m, matches the magnitudes the RLS baseline uses


@dataclass
class SimConfig:
    """How each episode is simulated and turned into transitions."""

    episode_seconds: float = 6.0
    episodes_per_door: int = 8
    # MuJoCo runs at 500 Hz (dt=0.002). The learned model predicts every
    # ``frame_skip`` steps, i.e. dt_model = 0.002 * frame_skip. At 500 Hz the
    # next state is nearly identical to the current one and the task collapses
    # to the identity map, so a coarser model rate is essential.
    frame_skip: int = 10
    # Episodes per door reserved for validation (same doors, unseen episodes).
    val_episodes_per_door: int = 2
    seed: int = 0
    out_path: str = "data/door_mechanics.npz"
    # Load-time filter (the .npz always stores every transition, so flipping
    # this needs no regeneration). Transitions touching a joint limit carry a
    # constraint torque that is not part of the action, so they are close to
    # unpredictable from (state, action, z) alone -- and being ~50x larger than
    # a typical step, they otherwise contribute ~90% of the squared error and
    # drown out the mechanics signal entirely. The RLS baseline masks the same
    # samples in ``moving_mask``.
    exclude_near_limit: bool = True


@dataclass
class ModelConfig:
    embed_dim: int = 16
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "silu"  # silu | relu | tanh | gelu
    # Predict next_state = state + delta instead of next_state directly. Keeps
    # the network away from having to re-learn the identity map.
    predict_delta: bool = True
    dropout: float = 0.0
    embedding_init_std: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 1024
    lr: float = 1e-3
    embedding_lr: float = 1e-2  # embeddings see far fewer gradients per step
    weight_decay: float = 1e-5
    embedding_weight_decay: float = 1e-4  # keeps the latent space compact
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"  # cosine | none
    warmup_epochs: int = 1
    num_workers: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps
    seed: int = 0
    # MSE is computed on the normalised delta so that angle [rad] and velocity
    # [rad/s] contribute comparably. Mathematically this is still MSE between
    # predicted and ground-truth next state, just with per-dimension scaling.
    loss_space: str = "normalized"  # normalized | raw
    log_every: int = 50  # optimiser steps between TensorBoard scalar writes
    rollout_eval_every: int = 10  # epochs between validation rollout metrics
    rollout_eval_horizon: int = 50
    rollout_eval_episodes: int = 16
    early_stop_patience: int = 0  # 0 disables
    run_dir: str = "runs/latent_mechanics"
    run_name: str = "base"


@dataclass
class EvalConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])
    # Number of episodes drawn for the rollout-overlay figure.
    n_plot_episodes: int = 6
    out_dir: str = "runs/latent_mechanics/base/eval"


@dataclass
class ExperimentConfig:
    doors: DoorSamplingConfig = field(default_factory=DoorSamplingConfig)
    excitation: ExcitationConfig = field(default_factory=ExcitationConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # -- convenience ------------------------------------------------------
    @property
    def run_path(self) -> Path:
        return Path(self.train.run_dir) / self.train.run_name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SECTIONS = {f.name: f.type for f in dataclasses.fields(ExperimentConfig)}


def _build_section(cls: Any, values: dict[str, Any], section_name: str) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in config section '{section_name}'; "
            f"valid keys: {sorted(known)}"
        )
    kwargs = {}
    for key, val in values.items():
        # YAML gives lists where the dataclass declares a tuple range.
        if isinstance(val, list) and "tuple" in str(known[key].type):
            val = tuple(val)
        kwargs[key] = val
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any] | None) -> ExperimentConfig:
    """Build a config from a (possibly partial) nested dict."""
    raw = raw or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(
            f"unknown config section(s) {sorted(unknown)}; valid: {sorted(_SECTIONS)}"
        )
    sections = {}
    for f in dataclasses.fields(ExperimentConfig):
        sections[f.name] = _build_section(
            f.default_factory().__class__, raw.get(f.name, {}) or {}, f.name
        )
    return ExperimentConfig(**sections)


def load_config(path: str | Path | None) -> ExperimentConfig:
    """Load a YAML config. ``None`` returns the all-defaults config."""
    if path is None:
        return ExperimentConfig()
    raw = yaml.safe_load(Path(path).read_text())
    return config_from_dict(raw)
