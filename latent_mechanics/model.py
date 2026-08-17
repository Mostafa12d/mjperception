"""The learned dynamics model and the door embedding table.

``MechanicsDynamicsModel`` is an MLP ``(state, action, z) -> next_state`` that
takes ``z`` as a plain tensor and knows nothing about door ids, so stage 2 can
freeze it and optimise a fresh ``z``. ``DoorEmbeddingTable`` is the stage-1
``door_id -> z`` lookup. Normalisation lives in the model as buffers, so a
checkpoint takes and returns raw SI units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from latent_mechanics import provenance
from latent_mechanics.config import ExperimentConfig, ModelConfig, config_from_dict

STATE_DIM = 2  # [door_angle (rad), door_velocity (rad/s)]
ACTION_DIM = 1  # applied hinge torque (N*m)

_ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


class NormStats(dict):
    """Per-dimension mean/std for state, action and state-delta."""

    KEYS = ("state_mean", "state_std", "action_mean", "action_std",
            "delta_mean", "delta_std")

    def validate(self) -> "NormStats":
        missing = set(self.KEYS) - set(self)
        if missing:
            raise ValueError(f"NormStats missing {sorted(missing)}")
        return self


def _mlp(in_dim: int, hidden: list[int], out_dim: int, act: str, dropout: float) -> nn.Sequential:
    if act not in _ACTIVATIONS:
        raise ValueError(f"unknown activation '{act}'; choose from {sorted(_ACTIVATIONS)}")
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), _ACTIVATIONS[act]()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MechanicsDynamicsModel(nn.Module):
    """One-step dynamics ``next_state = f(state, action, z)``.

    ``predict_delta`` is strongly recommended: over one 20 ms step the state
    barely changes, so predicting it directly makes the identity near-optimal.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_sizes: list[int] | None = None,
        activation: str = "silu",
        predict_delta: bool = True,
        dropout: float = 0.0,
        norm_stats: dict[str, torch.Tensor] | None = None,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes or [256, 256])
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.predict_delta = predict_delta

        self.net = _mlp(
            in_dim=state_dim + action_dim + embed_dim,
            hidden=hidden_sizes,
            out_dim=state_dim,
            act=activation,
            dropout=dropout,
        )

        ones_s, zeros_s = torch.ones(state_dim), torch.zeros(state_dim)
        ones_a, zeros_a = torch.ones(action_dim), torch.zeros(action_dim)
        defaults = {
            "state_mean": zeros_s, "state_std": ones_s,
            "action_mean": zeros_a, "action_std": ones_a,
            "delta_mean": zeros_s, "delta_std": ones_s,
        }
        for key, default in defaults.items():
            self.register_buffer(key, default.clone())
        if norm_stats is not None:
            self.set_norm_stats(norm_stats)

    # -- normalisation ----------------------------------------------------
    def set_norm_stats(self, stats: dict[str, torch.Tensor]) -> None:
        """Install dataset statistics. Stds are floored against constant dimensions."""
        for key in NormStats.KEYS:
            if key not in stats:
                raise ValueError(f"norm_stats missing '{key}'")
            val = torch.as_tensor(stats[key], dtype=torch.float32).reshape(-1)
            if key.endswith("_std"):
                val = val.clamp_min(1e-8)
            getattr(self, key).copy_(val)

    def get_norm_stats(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k).detach().clone() for k in NormStats.KEYS}

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self.delta_mean) / self.delta_std

    def denormalize_delta(self, delta_n: torch.Tensor) -> torch.Tensor:
        return delta_n * self.delta_std + self.delta_mean

    # -- forward ----------------------------------------------------------
    def raw_output(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Network output in normalised units: the state delta, or the next state
        if ``predict_delta`` is off. Training uses this directly."""
        state, action, z = _match_batch(state, action, z)
        x = torch.cat(
            [self.normalize_state(state), self.normalize_action(action), z], dim=-1
        )
        return self.net(x)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Next state in raw SI units, from (..., 2) state, (..., 1) torque and
        (..., embed_dim) latent."""
        out_n = self.raw_output(state, action, z)
        if self.predict_delta:
            state_b = torch.broadcast_to(state, out_n.shape)
            return state_b + self.denormalize_delta(out_n)
        return out_n * self.state_std + self.state_mean

    def target(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Normalised regression target matching ``raw_output``."""
        if self.predict_delta:
            return self.normalize_delta(next_state - state)
        return self.normalize_state(next_state)

    # -- stage-2 helpers --------------------------------------------------
    def freeze(self) -> "MechanicsDynamicsModel":
        """Freeze every network weight, leaving only a standalone ``z`` optimisable."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def new_latent(
        self, n: int = 1, init: torch.Tensor | None = None, device=None
    ) -> nn.Parameter:
        """Create a fresh optimisable ``z`` for unseen doors.

        Pass ``init`` explicitly. Zeros is a hole in the trained latent cloud
        (rows have norm >= 1.8) and predicts ~8x worse than any real latent;
        prefer the training table's medoid.
        """
        device = device or self.state_mean.device
        if init is None:
            init = torch.zeros(n, self.embed_dim, device=device)
        else:
            init = torch.as_tensor(init, dtype=torch.float32, device=device)
            init = init.reshape(n, self.embed_dim).clone()
        return nn.Parameter(init)


def _match_batch(
    state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast a single shared ``z`` across a batch of states."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.shape[0] == 1 and state.shape[0] != 1:
        z = z.expand(state.shape[0], -1)
    if action.dim() == 1:
        action = action.unsqueeze(-1)
    return state, action, z


class DoorEmbeddingTable(nn.Module):
    """Learnable ``door_id -> z`` lookup, used only while training on known doors.

    Initialised near zero, but training spreads the rows onto a shell whose
    centroid contains no door -- see ``MechanicsDynamicsModel.new_latent``.
    """

    def __init__(self, num_doors: int, embed_dim: int, init_std: float = 0.1) -> None:
        super().__init__()
        self.num_doors = num_doors
        self.embed_dim = embed_dim
        self.table = nn.Embedding(num_doors, embed_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=init_std)

    def forward(self, door_ids: torch.Tensor) -> torch.Tensor:
        return self.table(door_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.table.weight

    def as_numpy(self):
        return self.table.weight.detach().cpu().numpy()


def save_checkpoint(
    path: str | Path,
    model: MechanicsDynamicsModel,
    embeddings: DoorEmbeddingTable | None,
    cfg: ExperimentConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained checkpoint; model and table live under separate keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_kwargs": {
            "embed_dim": model.embed_dim,
            "predict_delta": model.predict_delta,
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_sizes": list(cfg.model.hidden_sizes),
            "activation": cfg.model.activation,
            "dropout": cfg.model.dropout,
        },
        "embedding_state": embeddings.state_dict() if embeddings is not None else None,
        "embedding_kwargs": (
            {"num_doors": embeddings.num_doors, "embed_dim": embeddings.embed_dim}
            if embeddings is not None
            else None
        ),
        "config": cfg.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    with_embeddings: bool = True,
    stage: str = "unlabelled",
    expected_sha256: str | None = None,
) -> tuple[MechanicsDynamicsModel, DoorEmbeddingTable | None, ExperimentConfig, dict]:
    """Load a checkpoint and record its sha256 via ``provenance.log_checkpoint``.

    ``stage`` names the caller; ``expected_sha256`` (full hash or prefix) turns a
    silent substitution into a hard failure.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model = MechanicsDynamicsModel(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state"])
    model.to(device)

    table = None
    if with_embeddings and payload.get("embedding_state") is not None:
        table = DoorEmbeddingTable(**payload["embedding_kwargs"])
        table.load_state_dict(payload["embedding_state"])
        table.to(device)

    kwargs = payload.get("embedding_kwargs") or {}
    provenance.log_checkpoint(path, stage=stage, expected_sha256=expected_sha256,
                              table_rows=kwargs.get("num_doors"))

    cfg = config_from_dict(payload["config"])
    return model, table, cfg, payload.get("extra", {})


def build_model_from_config(
    cfg: ModelConfig, norm_stats: dict[str, torch.Tensor] | None = None
) -> MechanicsDynamicsModel:
    return MechanicsDynamicsModel(
        embed_dim=cfg.embed_dim,
        hidden_sizes=list(cfg.hidden_sizes),
        activation=cfg.activation,
        predict_delta=cfg.predict_delta,
        dropout=cfg.dropout,
        norm_stats=norm_stats,
    )


__all__ = [
    "MechanicsDynamicsModel",
    "DoorEmbeddingTable",
    "NormStats",
    "build_model_from_config",
    "save_checkpoint",
    "load_checkpoint",
    "STATE_DIM",
    "ACTION_DIM",
]
