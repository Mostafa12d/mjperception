"""
Configuration for Stage-2 online adaptation.

Kept in its own module rather than extending ``latent_mechanics.config``, so
Stage 1 stays untouched. Same conventions: nested dataclasses, partial YAML
overrides, unknown keys raise.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AdaptorConfig:
    """Settings for ``GradientLatentAdaptor`` (the first ``_update`` rule)."""

    lr: float = 0.03
    optimizer: str = "adam"  # adam | sgd
    n_inner_steps: int = 1
    # Bounded sliding window of recent transitions per update. 1 = pure
    # single-sample online SGD. The cost per update stays constant either way.
    window: int = 32
    prior_weight: float = 0.0
    loss_space: str = "normalized"
    max_grad_norm: float = 0.0
    # Robbins-Monro step decay: lr_t = lr / (1 + lr_decay * t). Required for
    # the belief to settle instead of jittering around the optimum forever.
    lr_decay: float = 3.0e-3


@dataclass
class RLSConfig:
    """Settings for the RLS baseline. Defaults mirror the Stage-1 scripts."""

    lam: float = 0.995
    delta: float = 1e3
    vel_thresh: float = 0.02
    n_substeps: int = 10
    # Both regressors are run: 5 = spring-aware (fair), 3 = baseline's own.
    n_params: tuple[int, ...] = (5, 3)


@dataclass
class ExperimentsConfig:
    checkpoint: str = "runs/latent_mechanics/base/best.pt"
    # Optional pin on the frozen predictor's sha256 (full hash or any leading
    # prefix). Set it to make a substituted checkpoint fail loudly instead of
    # silently changing what the reported numbers mean.
    expected_sha256: str | None = None
    data: str = "data/door_mechanics.npz"
    split: str = "heldout_door"
    # Doors to run. Empty means every door in the split.
    # Latent init used by Experiment 1 and the belief animation. "zero" is
    # the honest no-prior-knowledge start and shows the clearest learning
    # curve; Experiment 2 compares every strategy against it.
    default_init: str = "zero"
    door_ids: tuple[int, ...] = ()
    max_episodes: int | None = None
    # Door used for the per-door figures and the latent-trajectory animation.
    focus_door: int | None = None
    rolling_window: int = 200
    seed: int = 0
    out_dir: str = "runs/latent_mechanics/base/online"
    device: str = "cpu"
    make_animation: bool = True
    animation_stride: int = 20
    animation_fps: int = 20


@dataclass
class OnlineConfig:
    adaptor: AdaptorConfig = field(default_factory=AdaptorConfig)
    rls: RLSConfig = field(default_factory=RLSConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls: Any, values: dict[str, Any], section: str) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in section '{section}'; "
            f"valid: {sorted(known)}"
        )
    kwargs = {}
    for k, v in values.items():
        if isinstance(v, list) and "tuple" in str(known[k].type):
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any] | None) -> OnlineConfig:
    raw = raw or {}
    fields_ = {f.name: f for f in dataclasses.fields(OnlineConfig)}
    unknown = set(raw) - set(fields_)
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}; valid: {sorted(fields_)}")
    return OnlineConfig(
        **{
            name: _build(f.default_factory().__class__, raw.get(name, {}) or {}, name)
            for name, f in fields_.items()
        }
    )


def load_config(path: str | Path | None) -> OnlineConfig:
    if path is None:
        return OnlineConfig()
    return config_from_dict(yaml.safe_load(Path(path).read_text()))


__all__ = ["OnlineConfig", "AdaptorConfig", "RLSConfig", "ExperimentsConfig", "load_config"]
