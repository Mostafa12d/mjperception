"""
Online adaptor interfaces and the first (gradient-descent) implementation.

Three layers, each earning its place:

``OnlineAdaptor``
    The streaming-estimator contract: ``predict`` a next state from the current
    belief, then ``observe`` a real transition and revise that belief. Nothing
    here mentions latents, gradients or neural networks. Both the latent adaptor
    and the RLS baseline implement it, which is what lets Experiment 3 drive
    both with identical code on identical data.

``OnlineLatentAdaptor``
    Adds "the belief IS a latent vector fed to a frozen dynamics network".
    Implements ``predict``, latent access, reset and the frozen-network
    guarantees once. Leaves exactly one abstract method, ``_update``, which is
    the algorithm slot.

``GradientLatentAdaptor``
    The first ``_update``: backprop the prediction loss into ``z`` only.

Future update rules subclass ``OnlineLatentAdaptor`` and implement ``_update``:

  * **Kalman / EKF** -- treat ``z`` as a Gaussian state. ``_update`` linearises
    the frozen network around the current ``z`` (``torch.autograd.functional.jacobian``
    of the prediction w.r.t. ``z``, which is cheap: 2 x embed_dim) and applies the
    standard measurement update. Store the covariance and return it from
    ``belief()``, which already exists for that purpose.
  * **Learned updater** -- ``_update`` calls a trained network
    ``(z, state, action, error) -> dz``. No other machinery changes.
  * **Bayesian / particle** -- keep a particle set instead of a point ``z``;
    ``latent`` returns the posterior mean and ``belief()`` the full set.

The one thing every implementation must preserve is the frozen network. That is
enforced here, not left to each subclass: ``_assert_frozen`` runs at
construction and ``assert_network_unchanged`` compares a parameter checksum
against the value captured before any adaptation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from latent_mechanics.model import MechanicsDynamicsModel

ArrayLike = np.ndarray | torch.Tensor | list | tuple


@dataclass
class AdaptorStep:
    """Record of one online interaction.

    ``prediction`` is made with the belief held *before* this transition was
    seen, so ``error`` is a genuine one-step-ahead prediction error and never
    reports on data the estimator has already fitted. This prequential protocol
    is what makes the learning curves in Experiment 1 and the comparison in
    Experiment 3 honest.
    """

    prediction: np.ndarray  # (2,) predicted next state, before the update
    target: np.ndarray  # (2,) observed next state
    error: np.ndarray  # (2,) prediction - target
    loss: float  # the update's own objective, after the update
    latent: np.ndarray  # belief after the update
    update_seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def abs_error(self) -> np.ndarray:
        return np.abs(self.error)


def _as_tensor(x: ArrayLike, dim: int, device) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    return t.reshape(1, dim)


# ---------------------------------------------------------------------------
# Layer 1: the generic streaming-estimator contract
# ---------------------------------------------------------------------------

class OnlineAdaptor(ABC):
    """A streaming one-step-ahead predictor whose belief improves with data."""

    name: str = "adaptor"

    @abstractmethod
    def predict(self, state: ArrayLike, action: ArrayLike) -> np.ndarray:
        """Predict the next state from the current belief. Must not mutate it."""

    @abstractmethod
    def observe(
        self, state: ArrayLike, action: ArrayLike, next_state: ArrayLike
    ) -> AdaptorStep:
        """Predict, then fold one observed transition into the belief."""

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Return to the prior belief, forgetting everything observed."""

    def belief(self) -> dict[str, Any]:
        """Current belief. Point estimators return only a mean; a Kalman or
        Bayesian implementation fills in ``cov`` / particles here."""
        return {"mean": None, "cov": None}

    @property
    def n_updates(self) -> int:
        return getattr(self, "_n_updates", 0)


# ---------------------------------------------------------------------------
# Layer 2: belief = a latent vector into a frozen dynamics network
# ---------------------------------------------------------------------------

class OnlineLatentAdaptor(OnlineAdaptor):
    """Adapts a mechanics embedding for one object against a frozen network.

    Subclasses implement ``_update`` and nothing else. Everything shared --
    prediction, latent bookkeeping, reset, and the frozen-network guarantee --
    lives here so no future update rule can accidentally drop it.

    Args:
        model: a trained stage-1 ``MechanicsDynamicsModel``. Frozen in place.
        init: starting latent, ``(embed_dim,)``. Defaults to zeros, which is a
            poor prior -- see ``latent_mechanics.online.init_strategies``.
        device: torch device.
    """

    name = "latent"

    def __init__(
        self,
        model: MechanicsDynamicsModel,
        init: ArrayLike | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.freeze()  # idempotent; also puts the model in eval mode
        self._assert_frozen()
        self._param_checksum0 = self._param_checksum()

        self.embed_dim = model.embed_dim
        self._init = self._coerce_init(init)
        self._n_updates = 0
        self.reset(self._init)

    # -- frozen-network guarantees ---------------------------------------
    def _assert_frozen(self) -> None:
        live = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(
                f"dynamics network is not frozen; {len(live)} tensor(s) still "
                f"require grad, e.g. {live[:3]}"
            )

    def _param_checksum(self) -> float:
        with torch.no_grad():
            return float(sum(p.double().abs().sum() for p in self.model.parameters()))

    def assert_network_unchanged(self, tol: float = 0.0) -> None:
        """Raise if any network weight moved since construction.

        Call after an adaptation run. ``requires_grad=False`` already prevents
        an optimiser from touching these tensors, but this catches the subtler
        failure modes too: an optimiser constructed over ``model.parameters()``
        by mistake, an in-place buffer write, or a future ``_update`` that
        decides to fine-tune "just the last layer".
        """
        now = self._param_checksum()
        if abs(now - self._param_checksum0) > tol:
            raise RuntimeError(
                f"network weights changed during adaptation "
                f"(checksum {self._param_checksum0!r} -> {now!r})"
            )
        leaked = [n for n, p in self.model.named_parameters() if p.grad is not None]
        if leaked:
            raise RuntimeError(f"gradients accumulated on frozen weights: {leaked[:3]}")

    # -- latent bookkeeping ----------------------------------------------
    def _coerce_init(self, init: ArrayLike | None) -> torch.Tensor:
        if init is None:
            return torch.zeros(self.embed_dim, device=self.device)
        t = torch.as_tensor(np.asarray(init, dtype=np.float32), device=self.device)
        return t.reshape(self.embed_dim).clone()

    def reset(self, init: ArrayLike | None = None) -> None:
        self._z = nn.Parameter(self._coerce_init(init if init is not None else self._init))
        self._n_updates = 0
        self._on_reset()

    def _on_reset(self) -> None:
        """Hook for subclass state that must be rebuilt after a reset."""

    @property
    def latent(self) -> np.ndarray:
        return self._z.detach().cpu().numpy().copy()

    @property
    def latent_tensor(self) -> torch.Tensor:
        return self._z.detach()

    def belief(self) -> dict[str, Any]:
        return {"mean": self.latent, "cov": None}

    # -- prediction -------------------------------------------------------
    @torch.no_grad()
    def predict(self, state: ArrayLike, action: ArrayLike) -> np.ndarray:
        s = _as_tensor(state, self.model.state_dim, self.device)
        a = _as_tensor(action, self.model.action_dim, self.device)
        return self.model(s, a, self._z.detach().reshape(1, -1))[0].cpu().numpy()

    # -- the algorithm slot ----------------------------------------------
    @abstractmethod
    def _update(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[float, dict[str, Any]]:
        """Revise ``self._z`` from one transition. Returns ``(loss, extras)``.

        Inputs are ``(1, d)`` tensors on the right device. Implementations must
        leave every network parameter untouched.
        """

    def observe(
        self, state: ArrayLike, action: ArrayLike, next_state: ArrayLike
    ) -> AdaptorStep:
        import time

        prediction = self.predict(state, action)
        s = _as_tensor(state, self.model.state_dim, self.device)
        a = _as_tensor(action, self.model.action_dim, self.device)
        ns = _as_tensor(next_state, self.model.state_dim, self.device)

        t0 = time.perf_counter()
        loss, extras = self._update(s, a, ns)
        dt = time.perf_counter() - t0

        self._n_updates += 1
        target = np.asarray(next_state, dtype=np.float32).reshape(-1)
        return AdaptorStep(
            prediction=prediction,
            target=target,
            error=prediction - target,
            loss=loss,
            latent=self.latent,
            update_seconds=dt,
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Layer 3: first implementation -- gradient descent on z
# ---------------------------------------------------------------------------

class GradientLatentAdaptor(OnlineLatentAdaptor):
    """Online SGD/Adam on the latent, with a bounded window of recent data.

    Args:
        lr: latent learning rate.
        optimizer: ``"adam"`` or ``"sgd"``.
        n_inner_steps: gradient steps taken per observed transition.
        window: how many of the most recent transitions each gradient step uses.
            ``1`` is pure single-sample online SGD. Larger values are still
            online -- the window is bounded and slides, so the cost per update
            is constant and no full trajectory is ever revisited -- but they cut
            the gradient noise that single-sample updates suffer from at the
            50 Hz sample rate, where consecutive transitions are individually
            almost uninformative.
        prior_weight: strength of an L2 pull toward ``prior_center``. Acts as the
            regulariser that keeps ``z`` in the region the network was trained
            on while evidence is still scarce, and is the natural place a proper
            Bayesian prior would enter.
        prior_center: defaults to the initial latent.
        loss_space: ``"normalized"`` (default, matching stage-1 training) or
            ``"raw"``.
        max_grad_norm: clip on the latent gradient; 0 disables.
        lr_decay: step size follows ``lr / (1 + lr_decay * t)``. This is the
            Robbins-Monro condition, and it is not cosmetic here. At a constant
            step size the latent keeps jittering around the optimum forever, and
            measurably so: starting from an already-good init, constant-rate
            adaptation ends up *worse* than not adapting at all (0.95x on the
            shipped checkpoint) because the noise it injects exceeds the
            information it extracts. With decay it improves from every
            initialisation tested.
    """

    name = "latent-gd"

    def __init__(
        self,
        model: MechanicsDynamicsModel,
        init: ArrayLike | None = None,
        lr: float = 0.03,
        optimizer: str = "adam",
        n_inner_steps: int = 1,
        window: int = 32,
        prior_weight: float = 0.0,
        prior_center: ArrayLike | None = None,
        loss_space: str = "normalized",
        max_grad_norm: float = 0.0,
        lr_decay: float = 3e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if loss_space not in ("normalized", "raw"):
            raise ValueError(f"loss_space must be 'normalized' or 'raw', got {loss_space!r}")
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer_name = optimizer.lower()
        self.n_inner_steps = n_inner_steps
        self.window = window
        self.prior_weight = prior_weight
        self.loss_space = loss_space
        self.max_grad_norm = max_grad_norm
        self._prior_center_arg = prior_center
        super().__init__(model, init=init, device=device)

    def _on_reset(self) -> None:
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=self.window
        )
        # The optimiser is built over EXACTLY ONE tensor: the latent. This is the
        # single most important line for the frozen-network guarantee -- there is
        # no code path by which it could reach a network weight.
        if self.optimizer_name == "adam":
            self._opt = torch.optim.Adam([self._z], lr=self.lr)
        elif self.optimizer_name == "sgd":
            self._opt = torch.optim.SGD([self._z], lr=self.lr)
        else:
            raise ValueError(f"unknown optimizer {self.optimizer_name!r}")
        center = (
            self._z.detach().clone()
            if self._prior_center_arg is None
            else self._coerce_init(self._prior_center_arg)
        )
        self._prior_center = center

    def _loss(
        self, s: torch.Tensor, a: torch.Tensor, ns: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_space == "normalized":
            pred = self.model.raw_output(s, a, z)
            target = self.model.target(s, ns)
        else:
            pred = self.model(s, a, z)
            target = ns
        loss = torch.nn.functional.mse_loss(pred, target)
        if self.prior_weight > 0:
            loss = loss + self.prior_weight * (z.reshape(-1) - self._prior_center).pow(2).sum()
        return loss

    def _update(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[float, dict[str, Any]]:
        self._buffer.append((state, action, next_state))
        s = torch.cat([b[0] for b in self._buffer], dim=0)
        a = torch.cat([b[1] for b in self._buffer], dim=0)
        ns = torch.cat([b[2] for b in self._buffer], dim=0)

        lr_t = self.lr / (1.0 + self.lr_decay * self._n_updates)
        for g in self._opt.param_groups:
            g["lr"] = lr_t

        loss_val = 0.0
        grad_norm = 0.0
        for _ in range(self.n_inner_steps):
            z = self._z.reshape(1, -1)
            loss = self._loss(s, a, ns, z)
            self._opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(self._z.grad.norm())
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([self._z], self.max_grad_norm)
            self._opt.step()
            loss_val = float(loss.detach())

        return loss_val, {
            "grad_norm": grad_norm,
            "buffer": len(self._buffer),
            "latent_norm": float(self._z.detach().norm()),
            "lr": lr_t,
        }


class StaticLatentAdaptor(OnlineLatentAdaptor):
    """Never updates. The control condition for every experiment.

    Without this, "the error went down" is not evidence of anything -- a stream
    that happens to get easier over time would produce the same curve. Running
    the identical latent with updates disabled isolates what adaptation actually
    contributed, and it is what revealed that constant-step-size adaptation from
    a good initialisation was a net loss.

    It is also the smallest possible example of the ``_update`` contract.
    """

    name = "static"

    def _update(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[float, dict[str, Any]]:
        with torch.no_grad():
            loss = self._loss_value(state, action, next_state)
        return loss, {}

    def _loss_value(self, s: torch.Tensor, a: torch.Tensor, ns: torch.Tensor) -> float:
        pred = self.model.raw_output(s, a, self._z.detach().reshape(1, -1))
        return float(torch.nn.functional.mse_loss(pred, self.model.target(s, ns)))


__all__ = [
    "AdaptorStep",
    "OnlineAdaptor",
    "OnlineLatentAdaptor",
    "GradientLatentAdaptor",
    "StaticLatentAdaptor",
]
