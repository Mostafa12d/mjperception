"""Multi-step prediction, where a wrong latent actually shows up (one-step MSE
flatters any smooth function over 20 ms).

``rollout`` / ``rollout_episode`` are open-loop from the first state -- what you
plot. ``horizon_errors`` rolls every valid start forward H steps and averages,
which is unbiased across the trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from latent_mechanics.dataset import Episode
from latent_mechanics.model import MechanicsDynamicsModel


@torch.no_grad()
def rollout(
    model: MechanicsDynamicsModel,
    z: torch.Tensor,
    init_state: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Open-loop rollout feeding predictions back in. Returns (T+1, 2) states,
    starting with ``init_state``."""
    state = init_state.reshape(1, -1)
    z = z.reshape(1, -1)
    traj = [state]
    for k in range(actions.shape[0]):
        state = model(state, actions[k].reshape(1, -1), z)
        traj.append(state)
    return torch.cat(traj, dim=0)


@torch.no_grad()
def rollout_episode(
    model: MechanicsDynamicsModel,
    z: torch.Tensor,
    episode: Episode,
    horizon: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out one episode from its first state -> ``(pred, truth)``, both (H+1, 2)."""
    n = len(episode) if horizon is None else min(horizon, len(episode))
    actions = torch.as_tensor(episode.action[:n], dtype=torch.float32, device=device)
    init = torch.as_tensor(episode.state[0], dtype=torch.float32, device=device)
    pred = rollout(model, z.to(device), init, actions).cpu().numpy()
    truth = np.concatenate([episode.state[:n], episode.next_state[n - 1 : n]], axis=0)
    return pred, truth


def _free_window_mask(episode: Episode, horizon: int, n_starts: int) -> np.ndarray:
    """Starts whose whole ``horizon``-step window stays clear of a joint limit;
    limit contact adds a constraint torque outside the action."""
    hit = np.concatenate([[0], np.cumsum(episode.near_limit.astype(np.int64))])
    in_window = hit[np.arange(n_starts) + horizon] - hit[np.arange(n_starts)]
    return in_window == 0


@torch.no_grad()
def multistart_rollout(
    model: MechanicsDynamicsModel,
    z: torch.Tensor,
    episode: Episode,
    horizon: int,
    device: str | torch.device = "cpu",
    exclude_near_limit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll every valid start forward ``horizon`` steps in one batch ->
    ``(pred_final, true_final)``, both (n_kept, 2)."""
    T = len(episode)
    n_starts = T - horizon + 1
    if n_starts <= 0:
        return np.empty((0, 2)), np.empty((0, 2))

    keep = (
        _free_window_mask(episode, horizon, n_starts)
        if exclude_near_limit
        else np.ones(n_starts, dtype=bool)
    )
    if not keep.any():
        return np.empty((0, 2)), np.empty((0, 2))
    start_idx = np.nonzero(keep)[0]

    state = torch.as_tensor(
        episode.state[start_idx], dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(episode.action, dtype=torch.float32, device=device)
    z = z.reshape(1, -1).to(device).expand(len(start_idx), -1)

    starts = torch.as_tensor(start_idx, device=device)
    for h in range(horizon):
        state = model(state, actions[starts + h], z)

    true_final = episode.next_state[start_idx + horizon - 1]
    return state.cpu().numpy(), true_final


def horizon_errors(
    model: MechanicsDynamicsModel,
    z: torch.Tensor,
    episode: Episode,
    horizons: list[int],
    device: str | torch.device = "cpu",
    exclude_near_limit: bool = True,
) -> dict[int, dict[str, float]]:
    """Per-horizon RMSE for one episode, averaged over all valid start indices."""
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        pred, truth = multistart_rollout(
            model, z, episode, h, device, exclude_near_limit
        )
        if len(pred) == 0:
            continue
        err = pred - truth
        out[h] = {
            "rmse_angle": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "rmse_velocity": float(np.sqrt(np.mean(err[:, 1] ** 2))),
            "mae_angle": float(np.mean(np.abs(err[:, 0]))),
            "mae_velocity": float(np.mean(np.abs(err[:, 1]))),
            "n_starts": len(pred),
        }
    return out


def aggregate_horizon_errors(
    per_episode: list[dict[int, dict[str, float]]]
) -> dict[int, dict[str, float]]:
    """Combine per-episode horizon errors, weighted by start count. RMSEs pool
    through their squares, so the result is the RMSE over all starts."""
    horizons = sorted({h for pe in per_episode for h in pe})
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        rows = [pe[h] for pe in per_episode if h in pe]
        n = sum(r["n_starts"] for r in rows)
        if n == 0:
            continue
        out[h] = {
            "rmse_angle": float(
                np.sqrt(sum(r["rmse_angle"] ** 2 * r["n_starts"] for r in rows) / n)
            ),
            "rmse_velocity": float(
                np.sqrt(sum(r["rmse_velocity"] ** 2 * r["n_starts"] for r in rows) / n)
            ),
            "mae_angle": float(sum(r["mae_angle"] * r["n_starts"] for r in rows) / n),
            "mae_velocity": float(
                sum(r["mae_velocity"] * r["n_starts"] for r in rows) / n
            ),
            "n_starts": int(n),
            "n_episodes": len(rows),
        }
    return out


__all__ = [
    "rollout",
    "rollout_episode",
    "multistart_rollout",
    "horizon_errors",
    "aggregate_horizon_errors",
]
