"""Figures for Stage-2 online adaptation.

The centrepiece is the latent-trajectory view: the unseen door's belief projected
into a PCA frame fitted once on the training embeddings, so the axes match the
Stage-1 figure and do not drift between frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_mechanics.online.loop import AdaptationLog


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


@dataclass
class LatentPCA:
    """A fixed 2-D projection fitted on the training embeddings."""

    mean: np.ndarray
    components: np.ndarray  # (2, d)
    explained: np.ndarray
    train_xy: np.ndarray  # (n_train, 2)

    @classmethod
    def fit(cls, train_latents: np.ndarray) -> "LatentPCA":
        mean = train_latents.mean(axis=0)
        centred = train_latents - mean
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        explained = s**2 / max(float((s**2).sum()), 1e-12)
        comp = vt[:2]
        return cls(mean, comp, explained, centred @ comp.T)

    def project(self, z: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(z) - self.mean) @ self.components.T

    def label(self, i: int) -> str:
        return f"PC{i + 1} ({100 * self.explained[i]:.0f}%)"


def plot_error_curve(
    logs: list[AdaptationLog],
    path: Path,
    window: int = 200,
    title: str = "",
    dim_names: tuple[str, str] = ("angle [rad]", "velocity [rad/s]"),
    reference: dict[str, float] | None = None,
) -> Path:
    """Rolling one-step-ahead prediction error vs interaction number."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for dim, ax in enumerate(axes):
        for lg in logs:
            r = lg.rolling_rmse(dim, window)
            ax.plot(np.arange(len(r)), r, lw=1.4, label=lg.init_name or lg.name)
        if reference and dim == 0 and "angle" in reference:
            ax.axhline(reference["angle"], color="k", ls=":", lw=1,
                       label="trained door (stage 1)")
        if reference and dim == 1 and "velocity" in reference:
            ax.axhline(reference["velocity"], color="k", ls=":", lw=1,
                       label="trained door (stage 1)")
        if logs and logs[0].boundaries:
            for b in logs[0].boundaries[1:]:
                ax.axvline(b, color="0.85", lw=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("interaction number")
        ax.set_ylabel(f"rolling RMSE, {dim_names[dim]}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(title or f"Online adaptation: prediction error (window={window})",
                 fontsize=11)
    return _save(fig, path)


def plot_latent_trajectory(
    pca: LatentPCA,
    latents: np.ndarray,
    path: Path,
    train_color: np.ndarray | None = None,
    color_label: str = "",
    title: str = "",
    stride: int = 1,
) -> Path:
    """Static view: the whole belief trajectory in the training PCA frame."""
    xy = pca.project(latents)[::stride]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    if train_color is not None:
        sc = ax.scatter(pca.train_xy[:, 0], pca.train_xy[:, 1], c=train_color,
                        cmap="viridis", s=60, alpha=0.85, edgecolor="none")
        fig.colorbar(sc, ax=ax, fraction=0.046, label=color_label)
    else:
        ax.scatter(pca.train_xy[:, 0], pca.train_xy[:, 1], c="0.75", s=50)

    steps = np.arange(len(xy))
    ax.plot(xy[:, 0], xy[:, 1], "-", color="crimson", lw=1.0, alpha=0.6, zorder=3)
    ax.scatter(xy[:, 0], xy[:, 1], c=steps, cmap="autumn_r", s=12, zorder=4)
    ax.plot(xy[0, 0], xy[0, 1], "ko", ms=11, mfc="white", mew=2, zorder=5, label="$z_0$ (init)")
    ax.plot(xy[-1, 0], xy[-1, 1], "k*", ms=20, zorder=5, label="$z_T$ (converged)")
    ax.set_xlabel(pca.label(0))
    ax.set_ylabel(pca.label(1))
    ax.set_title("belief trajectory through the training latent space", fontsize=10)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    disp = np.linalg.norm(latents - latents[0], axis=1)
    move = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    ax.plot(disp, label="$\\|z_t - z_0\\|$", lw=1.4)
    ax.plot(np.arange(1, len(latents)), move, label="$\\|z_t - z_{t-1}\\|$ (step size)",
            lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("interaction number")
    ax.set_ylabel("latent distance")
    ax.set_title("how far the belief has travelled, and how fast", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(title or "Online latent adaptation on an unseen door", fontsize=12)
    return _save(fig, path)


def plot_latent_snapshots(
    pca: LatentPCA,
    latents: np.ndarray,
    path: Path,
    n_panels: int = 6,
    train_color: np.ndarray | None = None,
    color_label: str = "",
) -> Path:
    """Sequential still frames -- the animation as a printable figure."""
    xy = pca.project(latents)
    idx = np.unique(np.linspace(0, len(xy) - 1, n_panels).astype(int))
    fig, axes = plt.subplots(1, len(idx), figsize=(2.7 * len(idx), 3.1),
                             sharex=True, sharey=True)
    lim = np.array([
        [min(pca.train_xy[:, 0].min(), xy[:, 0].min()),
         max(pca.train_xy[:, 0].max(), xy[:, 0].max())],
        [min(pca.train_xy[:, 1].min(), xy[:, 1].min()),
         max(pca.train_xy[:, 1].max(), xy[:, 1].max())],
    ])
    pad = 0.12 * (lim[:, 1] - lim[:, 0])

    for ax, k in zip(np.atleast_1d(axes), idx):
        c = "0.8" if train_color is None else train_color
        ax.scatter(pca.train_xy[:, 0], pca.train_xy[:, 1], c=c, cmap="viridis",
                   s=30, alpha=0.8, edgecolor="none")
        ax.plot(xy[: k + 1, 0], xy[: k + 1, 1], "-", color="crimson", lw=1.2)
        ax.plot(xy[k, 0], xy[k, 1], "*", color="crimson", ms=16, mec="k", mew=0.6)
        ax.set_title(f"t = {k}", fontsize=9)
        ax.set_xlim(lim[0, 0] - pad[0], lim[0, 1] + pad[0])
        ax.set_ylim(lim[1, 0] - pad[1], lim[1, 1] + pad[1])
        ax.grid(alpha=0.3)
        ax.set_xlabel(pca.label(0), fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel(pca.label(1), fontsize=8)
    fig.suptitle("Belief evolving as the robot interacts "
                 f"({color_label or 'training doors in grey'})", fontsize=11)
    return _save(fig, path)


def animate_latent_trajectory(
    pca: LatentPCA,
    latents: np.ndarray,
    path: Path,
    rolling_error: np.ndarray | None = None,
    train_color: np.ndarray | None = None,
    color_label: str = "",
    stride: int = 20,
    fps: int = 20,
) -> Path | None:
    """Render the belief trajectory as a video. Returns ``None`` with a warning if
    no encoder is available, rather than failing the run."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        print(f"  [skip] animation: imageio unavailable ({exc})")
        return None

    xy = pca.project(latents)
    frames_idx = list(range(0, len(xy), max(1, stride)))
    if frames_idx[-1] != len(xy) - 1:
        frames_idx.append(len(xy) - 1)

    lim_x = (min(pca.train_xy[:, 0].min(), xy[:, 0].min()),
             max(pca.train_xy[:, 0].max(), xy[:, 0].max()))
    lim_y = (min(pca.train_xy[:, 1].min(), xy[:, 1].min()),
             max(pca.train_xy[:, 1].max(), xy[:, 1].max()))
    pad_x, pad_y = 0.12 * (lim_x[1] - lim_x[0]), 0.12 * (lim_y[1] - lim_y[0])

    ncols = 2 if rolling_error is not None else 1
    frames = []
    for k in frames_idx:
        fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 5.0), squeeze=False)
        ax = axes[0][0]
        if train_color is not None:
            sc = ax.scatter(pca.train_xy[:, 0], pca.train_xy[:, 1], c=train_color,
                            cmap="viridis", s=60, alpha=0.85, edgecolor="none")
            fig.colorbar(sc, ax=ax, fraction=0.046, label=color_label)
        else:
            ax.scatter(pca.train_xy[:, 0], pca.train_xy[:, 1], c="0.75", s=55)
        ax.plot(xy[: k + 1, 0], xy[: k + 1, 1], "-", color="crimson", lw=1.4, alpha=0.85)
        ax.plot(xy[0, 0], xy[0, 1], "o", color="k", ms=10, mfc="white", mew=2)
        ax.plot(xy[k, 0], xy[k, 1], "*", color="crimson", ms=22, mec="k", mew=0.8)
        ax.set_xlim(lim_x[0] - pad_x, lim_x[1] + pad_x)
        ax.set_ylim(lim_y[0] - pad_y, lim_y[1] + pad_y)
        ax.set_xlabel(pca.label(0))
        ax.set_ylabel(pca.label(1))
        ax.set_title(f"mechanics belief after {k} interactions", fontsize=11)
        ax.grid(alpha=0.3)

        if rolling_error is not None:
            ax2 = axes[0][1]
            ax2.plot(rolling_error[: k + 1], color="C0", lw=1.4)
            ax2.set_xlim(0, len(rolling_error))
            pos = rolling_error[np.isfinite(rolling_error) & (rolling_error > 0)]
            if len(pos):
                ax2.set_ylim(pos.min() * 0.7, pos.max() * 1.4)
            ax2.set_yscale("log")
            ax2.set_xlabel("interaction number")
            ax2.set_ylabel("rolling angle RMSE [rad]")
            ax2.set_title("prediction error", fontsize=11)
            ax2.grid(alpha=0.3, which="both")

        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception as exc:  # pragma: no cover
        gif = path.with_suffix(".gif")
        print(f"  [warn] {path.suffix} encode failed ({exc}); writing {gif.name}")
        imageio.mimsave(gif, frames, duration=1.0 / fps)
        path = gif
    print(f"  video  -> {path}  ({len(frames)} frames)")
    return path


def plot_init_comparison(
    logs: list[AdaptationLog], path: Path, window: int = 200,
    reference_angle: float | None = None,
) -> Path:
    """Experiment 2: learning curves and final accuracy per initialisation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for lg in logs:
        axes[0].plot(lg.rolling_rmse(0, window), lw=1.4, label=lg.init_name)
    if reference_angle:
        axes[0].axhline(reference_angle, color="k", ls=":", lw=1, label="trained door")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("interaction number")
    axes[0].set_ylabel("rolling angle RMSE [rad]")
    axes[0].set_title("convergence", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    names = [lg.init_name for lg in logs]
    finals = [lg.final_rmse(0) for lg in logs]
    axes[1].bar(range(len(logs)), finals, color="C0")
    axes[1].set_xticks(range(len(logs)))
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("final angle RMSE [rad]")
    axes[1].set_yscale("log")
    axes[1].set_title("converged accuracy (last 25%)", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")

    for lg in logs:
        d = np.linalg.norm(lg.latents - lg.latents[0], axis=1)
        axes[2].plot(d, lw=1.3, label=lg.init_name)
    axes[2].set_xlabel("interaction number")
    axes[2].set_ylabel("$\\|z_t - z_0\\|$")
    axes[2].set_title("distance travelled in latent space", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Experiment 2: latent initialisation", fontsize=12)
    return _save(fig, path)


def plot_method_comparison(
    logs: list[AdaptationLog], path: Path, window: int = 200,
    reference_angle: float | None = None,
) -> Path:
    """Experiment 3: latent adaptation vs RLS on identical streams."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for lg in logs:
        axes[0].plot(lg.rolling_rmse(0, window), lw=1.4, label=lg.name)
    if reference_angle:
        axes[0].axhline(reference_angle, color="k", ls=":", lw=1, label="trained door")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("interaction number")
    axes[0].set_ylabel("rolling angle RMSE [rad]")
    axes[0].set_title("angle prediction", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    for lg in logs:
        axes[1].plot(lg.rolling_rmse(1, window), lw=1.4, label=lg.name)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("interaction number")
    axes[1].set_ylabel("rolling velocity RMSE [rad/s]")
    axes[1].set_title("velocity prediction", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    names = [lg.name for lg in logs]
    us = [1e6 * lg.seconds_per_update for lg in logs]
    axes[2].bar(range(len(logs)), us, color="C1")
    axes[2].set_xticks(range(len(logs)))
    axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[2].set_ylabel("microseconds per update")
    axes[2].set_yscale("log")
    axes[2].set_title("compute per interaction", fontsize=10)
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment 3: latent adaptation vs RLS, identical streams", fontsize=12)
    return _save(fig, path)


__all__ = [
    "LatentPCA",
    "plot_error_curve",
    "plot_latent_trajectory",
    "plot_latent_snapshots",
    "animate_latent_trajectory",
    "plot_init_comparison",
    "plot_method_comparison",
]
