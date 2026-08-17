"""What the mechanics belief IS -- the swappable coordinate system.

The estimator works in ``x`` (representation coordinates); the predictor consumes
``to_predictor(x)``. Keeping those separate is what lets a UKF filter a 6-D chart
of a 16-D latent, and what lets RLS filter five physical parameters, without
either estimator knowing about the other's geometry.

    FullLatent           x = z,  16-D, identity map
    ReducedLatent        x = V(z - z_mean),  6-D  -- wraps the existing LatentBasis
    PhysicalParameters   x = [I, mu, b, k, c],  5-D
    Hybrid               concatenation of two of the above

``ReducedLatent`` is not new mathematics. It is ``belief.basis.LatentBasis``,
which was already exactly a mechanics representation and was only ever described
as a UKF implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from latent_mechanics.belief.basis import LatentBasis


@runtime_checkable
class MechanicsRepresentation(Protocol):
    """A coordinate system for the mechanics belief."""

    name: str

    @property
    def dim(self) -> int:
        """Dimension of the estimator's state vector."""

    def initial(self) -> np.ndarray:
        """Prior mean, in representation coordinates."""

    def prior_covariance(self) -> np.ndarray | None:
        """Prior covariance, or None for representations with no natural prior."""

    def to_predictor(self, x: np.ndarray) -> np.ndarray:
        """Representation coordinates -> whatever the predictor consumes.
        Accepts ``(d,)`` or ``(N, d)`` and preserves that shape convention."""

    def from_predictor(self, z: np.ndarray) -> np.ndarray:
        """Inverse of ``to_predictor``, for reporting and initialisation."""


@dataclass
class FullLatent:
    """The 16-D embedding itself. What Stage 2's gradient adaptor estimates."""

    init: np.ndarray
    prior_latents: np.ndarray | None = None
    name: str = "full_latent"

    def __post_init__(self) -> None:
        self.init = np.asarray(self.init, dtype=np.float64).reshape(-1)

    @property
    def dim(self) -> int:
        return int(self.init.shape[0])

    def initial(self) -> np.ndarray:
        return self.init.copy()

    def prior_covariance(self) -> np.ndarray | None:
        if self.prior_latents is None:
            return None
        return np.cov(np.atleast_2d(np.asarray(self.prior_latents, float)), rowvar=False)

    def to_predictor(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def from_predictor(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64)


@dataclass
class ReducedLatent:
    """A frozen affine chart on the latent space -- the existing ``LatentBasis``.

    The chart is never refit online: a moving chart would make a covariance
    carried across timesteps meaningless.
    """

    basis: LatentBasis
    init: np.ndarray                       # full-latent coordinates
    prior_latents: np.ndarray | None = None
    p0_scale: float = 1.0
    name: str = "reduced_latent"

    def __post_init__(self) -> None:
        self.init = np.asarray(self.init, dtype=np.float64).reshape(-1)

    @classmethod
    def from_path(cls, path: str | Path, dim: int, init: np.ndarray,
                  prior_latents: np.ndarray | None = None,
                  p0_scale: float = 1.0,
                  expect_source: str | None = None) -> "ReducedLatent":
        """Load a persisted chart.

        ``expect_source`` is the checkpoint whose latent space this chart is
        supposed to describe. Passing it is strongly recommended: a chart fitted
        on one embedding table silently mangles beliefs from another, because the
        two tables span different subspaces. This is not hypothetical -- the
        repository ships a basis fitted on the 120-object all-families table, and
        projecting a 48-door latent through it loses most of the vector.
        """
        b = LatentBasis.load(path)
        if expect_source is not None and str(b.source_table) != str(expect_source):
            raise ValueError(
                f"basis at {path} was fitted on\n    {b.source_table}\n"
                f"but is being used with\n    {expect_source}\n"
                "These are different latent spaces; the projection would be "
                "meaningless. Compute a basis for this checkpoint instead "
                "(Workspace.basis() does this automatically).")
        if b.dim > dim:
            b = b.truncate(dim)
        if b.dim != dim:
            raise ValueError(
                f"basis at {path} has {b.dim} components but dim={dim} requested; "
                "recompute the basis with at least that many components")
        return cls(basis=b, init=init, prior_latents=prior_latents, p0_scale=p0_scale)

    @property
    def dim(self) -> int:
        return int(self.basis.dim)

    def initial(self) -> np.ndarray:
        return np.atleast_1d(self.basis.encode(self.init)).reshape(-1)

    def prior_covariance(self) -> np.ndarray | None:
        """Empirical covariance of the training population, in chart coordinates."""
        from latent_mechanics.belief.ukf import nearest_pd

        if self.prior_latents is None:
            return self.p0_scale * np.eye(self.dim)
        xr = self.basis.encode(np.asarray(self.prior_latents, dtype=np.float64))
        P = np.atleast_2d(np.cov(np.atleast_2d(xr), rowvar=False)) * self.p0_scale
        return nearest_pd(P, floor=1e-9)

    def to_predictor(self, x: np.ndarray) -> np.ndarray:
        return self.basis.decode(np.asarray(x, dtype=np.float64))

    def from_predictor(self, z: np.ndarray) -> np.ndarray:
        return self.basis.encode(np.asarray(z, dtype=np.float64))

    def decode_covariance(self, P: np.ndarray) -> np.ndarray:
        """Chart covariance -> full 16-D. Rank-``dim``, therefore singular."""
        return self.basis.decode_covariance(P)


@dataclass
class PhysicalParameters:
    """Interpretable hinge parameters ``[I, mu, b, k, c]`` -- what RLS estimates.

    ``to_predictor`` is the identity: an analytical predictor consumes these
    directly. Included so "explicit physical parameters" is a representation you
    can select rather than a separate estimator hierarchy.
    """

    init: np.ndarray
    names: tuple[str, ...] = ("I", "mu", "b", "k", "c")
    name: str = "physical"

    def __post_init__(self) -> None:
        self.init = np.asarray(self.init, dtype=np.float64).reshape(-1)
        self.names = tuple(self.names)[: len(self.init)]

    @property
    def dim(self) -> int:
        return int(self.init.shape[0])

    def initial(self) -> np.ndarray:
        return self.init.copy()

    def prior_covariance(self) -> np.ndarray | None:
        return None

    def to_predictor(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def from_predictor(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64)


@dataclass
class Hybrid:
    """Two representations side by side, e.g. a latent plus an explicit inertia.

    ``to_predictor`` returns the pair; the predictor decides what to do with it.
    Present because the brief asks for hybrid representations and, once the two
    halves exist, it is fifteen lines.
    """

    first: MechanicsRepresentation
    second: MechanicsRepresentation
    name: str = "hybrid"

    @property
    def dim(self) -> int:
        return self.first.dim + self.second.dim

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        return x[..., : self.first.dim], x[..., self.first.dim :]

    def initial(self) -> np.ndarray:
        return np.concatenate([self.first.initial(), self.second.initial()])

    def prior_covariance(self) -> np.ndarray | None:
        a, b = self.first.prior_covariance(), self.second.prior_covariance()
        if a is None or b is None:
            return None
        out = np.zeros((self.dim, self.dim))
        out[: self.first.dim, : self.first.dim] = a
        out[self.first.dim :, self.first.dim :] = b
        return out

    def to_predictor(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self._split(x)
        return self.first.to_predictor(a), self.second.to_predictor(b)

    def from_predictor(self, z) -> np.ndarray:
        a, b = z
        return np.concatenate([self.first.from_predictor(a),
                               self.second.from_predictor(b)])


__all__ = ["MechanicsRepresentation", "FullLatent", "ReducedLatent",
           "PhysicalParameters", "Hybrid"]
