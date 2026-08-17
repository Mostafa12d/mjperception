"""Configuration for the latent-mechanics experiments.

Plain dataclasses, so the config serialises into a checkpoint. YAML need only
list the fields it overrides.

    cfg = load_config("configs/latent_mechanics.yaml")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DoorSamplingConfig:
    """Ranges for the randomised mechanics of one door. Each door is one draw and
    keeps its parameters across all of its episodes."""

    n_train_doors: int = 48
    n_heldout_doors: int = 8    # generated but given no embedding row

    model_paths: list[str] = field(default_factory=lambda: ["door.xml"])

    density_range: tuple[float, float] = (200.0, 1400.0)   # kg/m^3, log-uniform
    frictionloss_range: tuple[float, float] = (0.5, 6.0)   # N*m
    damping_range: tuple[float, float] = (0.02, 1.5)       # N*m*s/rad
    stiffness_range: tuple[float, float] = (0.0, 8.0)      # N*m/rad; 0 = no self-closer
    springref_range: tuple[float, float] = (-0.1, 0.6)     # rad
    frac_no_spring: float = 0.3


@dataclass
class ExcitationConfig:
    """Torque profiles used to excite each episode.

    The commanded torque is a zero-order hold on the model timestep grid, so
    every recorded transition has exactly one constant action.
    """

    # 'swing' is what supplies closing-direction data: a slow, large oscillation
    profile_weights: dict[str, float] = field(
        default_factory=lambda: {
            "multisine": 0.3, "steps": 0.2, "chirp": 0.2, "swing": 0.3,
        }
    )
    # bias ~ U(range) * frictionloss, so the door actually breaks away
    bias_over_friction_range: tuple[float, float] = (0.6, 2.0)
    amp_range: tuple[float, float] = (1.0, 7.0)
    freq_range: tuple[float, float] = (0.2, 3.0)  # Hz
    n_sines_range: tuple[int, int] = (2, 4)
    step_hold_range: tuple[float, float] = (0.2, 1.2)  # s
    # must exceed 1 to break away in both directions
    swing_over_friction_range: tuple[float, float] = (1.4, 3.5)
    swing_freq_range: tuple[float, float] = (0.15, 0.5)  # Hz
    tau_clip: float = 30.0  # N*m


@dataclass
class SimConfig:
    """How each episode is simulated and turned into transitions."""

    episode_seconds: float = 6.0
    episodes_per_door: int = 8
    # MuJoCo runs at 500 Hz; dt_model = 0.002 * frame_skip. At 500 Hz the task
    # collapses to the identity map, so a coarser model rate is essential.
    frame_skip: int = 10
    val_episodes_per_door: int = 2   # same doors, unseen episodes
    seed: int = 0
    out_path: str = "data/door_mechanics.npz"
    # Load-time filter; the .npz always stores every transition. Limit-touching
    # transitions carry a constraint torque outside the action and would
    # otherwise contribute ~90% of the squared error.
    exclude_near_limit: bool = True


@dataclass
class ModelConfig:
    embed_dim: int = 16
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "silu"  # silu | relu | tanh | gelu
    predict_delta: bool = True   # avoids re-learning the identity map
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
    # normalised: MSE on the delta, so rad and rad/s contribute comparably
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
    n_plot_episodes: int = 6   # episodes in the rollout-overlay figure
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
        # YAML gives lists where the dataclass declares a tuple range
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
