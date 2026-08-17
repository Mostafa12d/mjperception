"""One place that builds methods. Eight copies of this logic existed.

The audit found "build an adaptor by name" reimplemented, with drift, in
``online/experiments.py``, ``mismatch/study.py``, ``mechanisms/study.py``,
``curriculum/study.py``, ``belief/sweep.py``, ``belief/drift_check.py``,
``belief/ablation.py`` and ``geometry/report.py`` -- and notably, *no single site
built both RLS and the UKF*, so the baseline and the current best method had never
been run by the same code path. They are both here.

A ``Method`` is a (predictor, estimator) pair, because the two are chosen together:
a UKF over a reduced latent chart needs the network predictor, RLS needs the
analytical one. Bundling them is what stops an experiment silently pairing an
estimator with a predictor it cannot drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_mechanics.belief.adaptor import UKFConfig
from latent_mechanics.model import MechanicsDynamicsModel, load_checkpoint
from latent_mechanics.online.loop import init_strategies
from mechanics.estimator import Estimator
from mechanics.estimators import (
    GradientEstimator,
    RLSEstimator,
    StaticEstimator,
    UKFEstimator,
)
from mechanics.predictor import AnalyticalPredictor, LatentNetworkPredictor, Predictor
from mechanics.representation import FullLatent, PhysicalParameters, ReducedLatent

# There is deliberately no DEFAULT_BASIS constant. A latent chart belongs to one
# checkpoint's embedding table; a shared default is what let the shipped
# all-families chart be applied silently to the 48-door predictor. Use
# ``Workspace.basis()``, which computes the chart for its own checkpoint.

# Priors matching baseline/run_door_adaptive_impedance, preserved exactly.
RLS_PRIOR = np.array([5.0, 3.0, 0.2, 0.0, 0.0])


@dataclass
class Method:
    """A predictor and the estimator that drives it."""

    name: str
    predictor: Predictor
    estimator: Estimator


@dataclass
class Workspace:
    """A loaded frozen predictor plus everything derived from its training set.

    Loading the checkpoint once per experiment (rather than once per method, as
    every old driver did) also means the provenance log records one load, not four.
    """

    model: MechanicsDynamicsModel
    train_latents: np.ndarray
    stage1_cfg: Any
    extra: dict
    device: torch.device
    checkpoint: str
    seed: int = 0

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        stage: str = "mechanics",
        expected_sha256: str | None = None,
        seed: int = 0,
    ) -> "Workspace":
        device = torch.device(device)
        model, table, cfg, extra = load_checkpoint(
            checkpoint, device=device, stage=stage, expected_sha256=expected_sha256)
        if table is None:
            raise ValueError(
                f"{checkpoint} has no embedding table; the training latents are "
                "needed for every initialisation strategy")
        return cls(
            model=model.freeze(),
            train_latents=table.weight.detach().cpu().numpy(),
            stage1_cfg=cfg, extra=extra, device=device,
            checkpoint=str(checkpoint), seed=seed,
        )

    # -- initialisation ----------------------------------------------------
    def init_latent(self, name: str = "medoid") -> np.ndarray:
        """Starting latent. ``zero`` is a hole in the trained cloud and predicts
        ~8x worse than any real latent; ``medoid`` is a real, central object."""
        strategies = init_strategies(self.train_latents, self.seed)
        if name not in strategies:
            raise ValueError(
                f"unknown init {name!r}; choose from {sorted(strategies)}")
        return strategies[name].astype(np.float64)

    def wrong_init(self, name: str = "medoid", scale: float = 3.0) -> np.ndarray:
        """A deliberately wrong prior: the medoid pushed far along PC1.

        For the sanity experiment, which needs a belief that is visibly wrong at
        step 0 so that the adaptation has something to undo.
        """
        z = self.init_latent(name)
        centred = self.train_latents - self.train_latents.mean(axis=0)
        pc1 = np.linalg.svd(centred, full_matrices=False)[2][0]
        spread = float(np.std(centred @ pc1))
        return z + scale * spread * pc1

    # -- the latent chart --------------------------------------------------
    def basis(self, n_components: int = 8, path: str | Path | None = None):
        """A PCA chart of THIS checkpoint's latent space.

        A chart fitted on a different embedding table spans a different subspace,
        so projecting through it destroys the belief -- silently. The audit flagged
        the shipped default (``belief/basis.py:DEFAULT_TABLE``) as pointing three
        stages deep into a results directory for exactly this reason.

        Passing ``path`` reuses a persisted chart, but only if it was fitted on
        this checkpoint; otherwise it raises rather than projecting nonsense.
        Passing nothing computes the chart from this checkpoint's own table and
        caches it beside the checkpoint.
        """
        from latent_mechanics.belief.basis import LatentBasis, compute_basis

        if path is not None:
            b = LatentBasis.load(path)
            if str(b.source_table) != str(self.checkpoint):
                raise ValueError(
                    f"basis at {path} was fitted on\n    {b.source_table}\n"
                    f"but this Workspace holds\n    {self.checkpoint}\n"
                    "Different latent spaces. Pass path=None to compute the "
                    "right chart for this checkpoint.")
            return b

        cache = Path(self.checkpoint).parent / f"latent_basis_{n_components}.npz"
        if cache.exists():
            b = LatentBasis.load(cache)
            if str(b.source_table) == str(self.checkpoint) and b.dim >= n_components:
                return b
        b = compute_basis(self.checkpoint, n_components=n_components)
        b.save(cache)
        return b

    # -- predictors --------------------------------------------------------
    def latent_predictor(self, representation) -> LatentNetworkPredictor:
        return LatentNetworkPredictor(
            model=self.model, representation=representation, device=self.device)

    @property
    def reference_rmse(self) -> dict[str, float]:
        """Stage-1 validation error: the floor adaptation is chasing."""
        return {"angle": float(self.extra.get("val_rmse_angle", np.nan)),
                "velocity": float(self.extra.get("val_rmse_velocity", np.nan))}


@dataclass
class MethodConfig:
    """Everything a method needs that is not the checkpoint. One block, one place."""

    init: str = "medoid"
    init_vector: np.ndarray | None = None      # overrides ``init`` when given

    # gradient
    lr: float = 0.03
    optimizer: str = "adam"
    n_inner_steps: int = 1
    window: int = 32
    prior_weight: float = 0.0
    loss_space: str = "normalized"
    max_grad_norm: float = 0.0
    lr_decay: float = 3.0e-3

    # ukf. basis_path=None computes the chart from the experiment's OWN
    # checkpoint, which is almost always what you want; an explicit path is
    # checked against that checkpoint and rejected if it came from another.
    basis_path: str | None = None
    ukf: UKFConfig = field(default_factory=UKFConfig)
    p0_scale: float = 1.0

    # rls
    lam: float = 0.995
    delta: float = 1e3
    vel_thresh: float = 0.02
    n_substeps: int = 10


def build_method(name: str, ws: Workspace, dt: float,
                 cfg: MethodConfig | None = None) -> Method:
    """Build one named method. The only ``if name ==`` chain in the codebase.

    Names: ``no-adaptation``, ``gradient``, ``ukf``, ``rls-3p``, ``rls-5p``.
    """
    cfg = cfg or MethodConfig()
    z0 = cfg.init_vector if cfg.init_vector is not None else ws.init_latent(cfg.init)
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)

    if name in ("no-adaptation", "static"):
        pred = ws.latent_predictor(FullLatent(init=z0, prior_latents=ws.train_latents))
        return Method(name="no-adaptation", predictor=pred,
                      estimator=StaticEstimator(pred))

    if name in ("gradient", "latent-gd"):
        pred = ws.latent_predictor(FullLatent(init=z0, prior_latents=ws.train_latents))
        est = GradientEstimator(
            pred, lr=cfg.lr, optimizer=cfg.optimizer,
            n_inner_steps=cfg.n_inner_steps, window=cfg.window,
            prior_weight=cfg.prior_weight, loss_space=cfg.loss_space,
            max_grad_norm=cfg.max_grad_norm, lr_decay=cfg.lr_decay)
        est.name = "gradient"
        return Method(name="gradient", predictor=pred, estimator=est)

    if name == "ukf":
        basis = ws.basis(n_components=max(cfg.ukf.dim, 8), path=cfg.basis_path)
        if basis.dim > cfg.ukf.dim:
            basis = basis.truncate(cfg.ukf.dim)
        rep = ReducedLatent(basis=basis, init=z0,
                            prior_latents=ws.train_latents, p0_scale=cfg.p0_scale)
        pred = ws.latent_predictor(rep)
        return Method(name="ukf", predictor=pred,
                      estimator=UKFEstimator(pred, cfg=cfg.ukf))

    if name.startswith("rls"):
        n_params = int(name.split("-")[1].rstrip("p")) if "-" in name else 5
        rep = PhysicalParameters(init=RLS_PRIOR[:n_params].copy())
        pred = AnalyticalPredictor(dt=dt, representation=rep,
                                   n_substeps=cfg.n_substeps)
        est = RLSEstimator(pred, n_params=n_params, lam=cfg.lam, delta=cfg.delta,
                           vel_thresh=cfg.vel_thresh, n_substeps=cfg.n_substeps)
        return Method(name=est.name, predictor=pred, estimator=est)

    raise ValueError(
        f"unknown method {name!r}; choose from no-adaptation, gradient, ukf, "
        "rls-3p, rls-5p")


__all__ = ["Workspace", "MethodConfig", "Method", "build_method"]
