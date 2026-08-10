"""
Figures for the stage-1 model.

Four views, each answering a specific question:

  rollout overlays       does the model track a real trajectory open-loop?
  horizon error curves   how fast does error grow with prediction horizon?
  per-door error         is accuracy uniform, or carried by a few easy doors?
  latent space           did the embeddings organise by physical mechanics?

The latent-space plot is the one that speaks to the research claim. If latents
line up with true inertia / friction / damping without ever having been shown
them, the embedding really is encoding mechanics -- which is the premise stage-2
online adaptation rests on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_mechanics.dataset import DoorTransitionDataset, Episode
from latent_mechanics.model import MechanicsDynamicsModel
from latent_mechanics.rollout import rollout_episode


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


def plot_rollouts(
    model: MechanicsDynamicsModel,
    latents: dict[int, torch.Tensor],
    dataset: DoorTransitionDataset,
    episodes: list[Episode],
    path: Path,
    dt: float,
    horizon: int | None = None,
    device: str = "cpu",
) -> Path:
    """Open-loop predicted vs true trajectory for a handful of episodes."""
    n = len(episodes)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.4), squeeze=False, sharex="col")

    for col, ep in enumerate(episodes):
        pred, truth = rollout_episode(
            model, latents[ep.door_id], ep, horizon=horizon, device=device
        )
        t = np.arange(len(pred)) * dt
        p = dataset.params_for_door(ep.door_id)

        axes[0][col].plot(t, np.degrees(truth[:, 0]), "k-", lw=1.6, label="MuJoCo")
        axes[0][col].plot(t, np.degrees(pred[:, 0]), "C1--", lw=1.4, label="model")
        axes[0][col].set_title(
            f"door {ep.door_id} ({ep.kind})\n"
            f"I={p['I_hinge']:.1f}  $\\mu$={p['frictionloss']:.1f}  "
            f"b={p['damping']:.2f}  k={p['stiffness']:.1f}",
            fontsize=8,
        )
        axes[0][col].grid(alpha=0.3)

        axes[1][col].plot(t, truth[:, 1], "k-", lw=1.6)
        axes[1][col].plot(t, pred[:, 1], "C1--", lw=1.4)
        axes[1][col].set_xlabel("time [s]")
        axes[1][col].grid(alpha=0.3)

    axes[0][0].set_ylabel("angle [deg]")
    axes[1][0].set_ylabel("velocity [rad/s]")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Open-loop rollout from the first state (actions replayed, states fed back)",
        fontsize=11,
    )
    return _save(fig, path)


def plot_horizon_curve(
    agg: dict[int, dict[str, float]], path: Path, dt: float, title: str = ""
) -> Path:
    """RMSE vs prediction horizon, averaged over all start indices."""
    hs = sorted(agg)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, unit in (
        (axes[0], "rmse_angle", "rad"),
        (axes[1], "rmse_velocity", "rad/s"),
    ):
        ax.plot([h * dt for h in hs], [agg[h][key] for h in hs], "o-")
        ax.set_xlabel("horizon [s]")
        ax.set_ylabel(f"RMSE [{unit}]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title("angle")
    axes[1].set_title("velocity")
    fig.suptitle(title or "Multi-step prediction error", fontsize=11)
    return _save(fig, path)


def plot_per_door_error(
    per_door: dict[int, dict[str, float]],
    dataset: DoorTransitionDataset,
    path: Path,
    metric: str = "rmse_angle",
) -> Path:
    """One-step error per door, and the same error against each true parameter.

    A strong trend against a parameter means that regime is systematically
    harder -- useful for knowing where stage-2 adaptation will struggle.
    """
    ids = sorted(per_door)
    vals = np.array([per_door[i][metric] for i in ids])
    params = np.array([[dataset.params_for_door(i)[c] for c in
                        dataset.door_params_columns] for i in ids])

    cols = ["I_hinge", "frictionloss", "damping", "stiffness"]
    fig, axes = plt.subplots(1, 1 + len(cols), figsize=(3.0 * (1 + len(cols)), 3.2))

    axes[0].bar(range(len(ids)), vals, color="C0")
    axes[0].set_xlabel("door id")
    axes[0].set_ylabel(metric)
    axes[0].set_title("per-door one-step error", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    for ax, col in zip(axes[1:], cols):
        x = params[:, dataset.door_params_columns.index(col)]
        ax.scatter(x, vals, s=18, color="C1")
        ax.set_xlabel(col)
        ax.set_title(f"{metric} vs {col}", fontsize=9)
        ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_latent_space(
    latents: np.ndarray,
    dataset: DoorTransitionDataset,
    path: Path,
    color_by: tuple[str, ...] = ("I_hinge", "frictionloss", "damping", "stiffness"),
) -> Path:
    """First two principal components of the embedding table, coloured by the
    true physical parameters the model was never given."""
    ids = np.arange(len(latents))
    z = latents - latents.mean(0, keepdims=True)
    # SVD rather than a PCA dependency; components are the right singular vectors.
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    explained = s**2 / max(float((s**2).sum()), 1e-12)

    fig, axes = plt.subplots(1, len(color_by) + 1, figsize=(3.1 * (len(color_by) + 1), 3.3))
    for ax, col in zip(axes, color_by):
        c = np.array([dataset.params_for_door(i)[col] for i in ids])
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, cmap="viridis", s=34)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(f"PC1 ({100 * explained[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * explained[1]:.0f}%)")
        ax.grid(alpha=0.3)

    axes[-1].plot(np.arange(1, len(explained) + 1), np.cumsum(explained), "o-")
    axes[-1].set_xlabel("component")
    axes[-1].set_ylabel("cumulative variance")
    axes[-1].set_title("latent spectrum", fontsize=9)
    axes[-1].grid(alpha=0.3)

    fig.suptitle(
        "Learned mechanics embeddings (PCA), coloured by the true parameters "
        "the model never sees",
        fontsize=11,
    )
    return _save(fig, path)


__all__ = [
    "plot_rollouts",
    "plot_horizon_curve",
    "plot_per_door_error",
    "plot_latent_space",
]
