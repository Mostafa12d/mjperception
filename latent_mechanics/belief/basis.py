"""Fixed affine map between the full 16-D latent and a reduced d-D one.

    z   = z_mean + V_d @ z_r          (decode)
    z_r = V_d.T @ (z - z_mean)        (encode)

``V_d`` is the top-d PCA of a frozen embedding table (all-families: PC1 = 63%,
effective dim 2.36). Computed once and persisted; never refit online, since a
moving chart would make a covariance carried across timesteps meaningless.
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
        """Reduced covariance -> full 16-D ``V.T P_r V``. Rank-d, so singular; invert
        in reduced coordinates instead."""
        return self.components.T @ np.asarray(P_r, dtype=np.float64) @ self.components

    def reconstruction_error(self, z: np.ndarray) -> np.ndarray:
        """Per-object L2 error from projecting onto this basis and back."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return np.linalg.norm(z - self.decode(self.encode(z)).reshape(z.shape), axis=1)

    def truncate(self, d: int) -> "LatentBasis":
        """A nested basis with the first ``d`` components."""
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
    """Deterministic PCA on a frozen embedding table. SVD signs are pinned so the
    artifact is bit-stable across runs."""
    from latent_mechanics.model import load_checkpoint

    _, table, _, _ = load_checkpoint(table_ckpt, device="cpu", stage="belief_ukf:basis")
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
