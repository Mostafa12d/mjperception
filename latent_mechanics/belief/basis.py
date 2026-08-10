"""
Step 2: the fixed affine map between the full 16-D latent and a reduced d-D one.

    z   = z_mean + V_d @ z_r          (decode)
    z_r = V_d.T @ (z - z_mean)        (encode)

``V_d`` holds the top-d principal directions of a *frozen* embedding table, with
orthonormal columns. The basis is computed once, written to disk, and reloaded
thereafter. It is never refit online and never refit per episode -- it is as
fixed as the predictor, and for the same reason: if the coordinate system moved,
a covariance carried across timesteps would be meaningless.

Why reduce at all: the geometry investigation measured an effective
dimensionality of 2.36 out of 16 on the all-families table (PC1 = 63% of
variance, 6 components = 94%). A UKF maintaining a full 16x16 covariance would
be estimating 136 free parameters to describe a roughly 2.4-dimensional object,
and would need 33 sigma points per update instead of 9-13.

WHICH TABLE THE BASIS COMES FROM -- CONFIRMED: the all-families table.

The original brief said "the Stage-1 embedding table", but the statistics it
quoted (effective dim ~2.4, PC1 = 63%) belong to the *all-families* table from
the geometry report, not the Stage-1 door-only table:

    Stage-1 (48 doors)      PC1 = 40%  PC2 = 19%  effective dim 4.35
    all_families (120 objs) PC1 = 63%  PC2 = 10%  effective dim 2.36

The discrepancy was raised with the user and the all-families table was
confirmed, for three reasons:

  1. It is the table whose geometry the report actually characterised, so the
     effective-dimensionality argument for reducing to d = 4..6 applies to it
     and not to the Stage-1 table (whose effective dimension is 4.35, nearly
     twice as high -- a 6-D reduction would discard materially more there).
  2. It is the embedding table of the predictor this branch filters. The basis
     and the predictor must come from the same model or the reduced coordinates
     describe directions the network never learned to use.
  3. The per-family oracle ceilings the sweep scores against were measured on
     this model's held-out objects, so any other basis would make the headline
     ratios incomparable.

``DEFAULT_TABLE`` therefore points at the all-families checkpoint. Passing a
different ``table_ckpt`` switches it, and ``load_or_create`` refuses to reuse a
persisted basis that was fit on a different table rather than silently mixing
them. Both tables recompute deterministically: the 2-component basis persisted
during Stage 2 reproduces from the Stage-1 table with |cos| = 1.0 on both
components, and SVD signs are pinned so the artifact is bit-stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_TABLE = "runs/latent_mechanics/geometry/runs/all_families/best.pt"
DEFAULT_BASIS = "runs/latent_mechanics/belief/latent_basis.npz"


@dataclass(frozen=True)
class LatentBasis:
    """A frozen affine chart on the latent space."""

    mean: np.ndarray           # (D,)
    components: np.ndarray     # (d, D) orthonormal rows = principal directions
    explained_variance: np.ndarray  # (D,) full spectrum, for reporting
    source_table: str
    n_objects: int

    @property
    def full_dim(self) -> int:
        return len(self.mean)

    @property
    def dim(self) -> int:
        return self.components.shape[0]

    # -- the map ----------------------------------------------------------
    def encode(self, z: np.ndarray) -> np.ndarray:
        """Full 16-D -> reduced d-D. Accepts (D,) or (N, D)."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        out = (z - self.mean) @ self.components.T
        return out[0] if out.shape[0] == 1 and np.ndim(z) == 2 and z.shape[0] == 1 else out

    def decode(self, z_r: np.ndarray) -> np.ndarray:
        """Reduced d-D -> full 16-D. Accepts (d,) or (N, d)."""
        z_r = np.atleast_2d(np.asarray(z_r, dtype=np.float64))
        out = self.mean + z_r @ self.components
        return out[0] if out.shape[0] == 1 else out

    def decode_covariance(self, P_r: np.ndarray) -> np.ndarray:
        """Reduced covariance -> full 16-D covariance: ``V.T P_r V``.

        The result is rank-d and therefore singular in 16-D. That is correct and
        not a defect: the filter asserts zero uncertainty in the 16-d directions
        it does not model, which is exactly the modelling assumption made by
        reducing the dimension in the first place. Anything downstream that
        needs to invert it should work in the reduced coordinates instead.
        """
        return self.components.T @ np.asarray(P_r, dtype=np.float64) @ self.components

    def reconstruction_error(self, z: np.ndarray) -> np.ndarray:
        """Per-object L2 error from projecting onto this basis and back.

        The floor on everything this branch can achieve: no reduced-dimension
        filter can represent a latent better than its own projection.
        """
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return np.linalg.norm(z - self.decode(self.encode(z)).reshape(z.shape), axis=1)

    def truncate(self, d: int) -> "LatentBasis":
        """A nested basis with the first ``d`` components. Nested by construction,
        so d=4, 5 and 6 share their leading directions and the sweep varies only
        how many are kept."""
        if d > self.dim:
            raise ValueError(f"cannot expand a {self.dim}-component basis to {d}")
        return LatentBasis(self.mean, self.components[:d], self.explained_variance,
                           self.source_table, self.n_objects)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components,
                 explained_variance=self.explained_variance,
                 source_table=np.array(self.source_table),
                 n_objects=np.int64(self.n_objects))
        return path

    @staticmethod
    def load(path: str | Path) -> "LatentBasis":
        with np.load(path, allow_pickle=False) as a:
            return LatentBasis(a["mean"], a["components"], a["explained_variance"],
                               str(a["source_table"]), int(a["n_objects"]))

    def summary(self) -> str:
        cum = np.cumsum(self.explained_variance)
        return (f"  basis from {Path(self.source_table).parent.name} "
                f"({self.n_objects} objects, {self.full_dim}-D)\n"
                f"    kept {self.dim} components, "
                f"{100 * cum[self.dim - 1]:.1f}% of variance\n"
                f"    spectrum: " +
                " ".join(f"{100 * v:.0f}%" for v in self.explained_variance[:8]))


def compute_basis(table_ckpt: str = DEFAULT_TABLE, n_components: int = 8) -> LatentBasis:
    """Deterministic PCA on a frozen embedding table.

    SVD sign is fixed by convention (largest-magnitude entry of each component
    made positive) so repeated runs give a bit-identical artifact -- otherwise
    LAPACK sign flips would silently change the reduced coordinates between
    sessions and invalidate any stored covariance.
    """
    from latent_mechanics.model import load_checkpoint

    _, table, _, _ = load_checkpoint(table_ckpt, device="cpu")
    if table is None:
        raise ValueError(f"{table_ckpt} has no embedding table")
    z = table.weight.detach().cpu().numpy().astype(np.float64)

    mean = z.mean(axis=0)
    centred = z - mean
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    ev = s**2 / max(float((s**2).sum()), 1e-30)

    for i in range(vt.shape[0]):
        if vt[i, np.argmax(np.abs(vt[i]))] < 0:
            vt[i] = -vt[i]

    n_components = min(n_components, vt.shape[0])
    return LatentBasis(mean=mean, components=vt[:n_components],
                       explained_variance=ev, source_table=str(table_ckpt),
                       n_objects=len(z))


def load_or_create(path: str | Path = DEFAULT_BASIS,
                   table_ckpt: str = DEFAULT_TABLE,
                   n_components: int = 8) -> LatentBasis:
    """Reuse the persisted basis if present; otherwise compute and persist it."""
    path = Path(path)
    if path.exists():
        b = LatentBasis.load(path)
        if b.source_table != str(table_ckpt):
            raise ValueError(
                f"persisted basis was fit on {b.source_table} but {table_ckpt} was "
                f"requested; delete {path} or pass the matching table")
        return b
    b = compute_basis(table_ckpt, n_components)
    b.save(path)
    return b


__all__ = ["LatentBasis", "compute_basis", "load_or_create",
           "DEFAULT_TABLE", "DEFAULT_BASIS"]
