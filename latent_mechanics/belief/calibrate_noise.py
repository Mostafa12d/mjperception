"""Regenerate ``noise.IRREDUCIBLE_R``.

Measures the predictor's residual at the offline-optimal latent (for a training
object, its own row of the embedding table) in the normalised measurement space.
No online estimator can beat this, so R must never fall below it.

    python3.10 -m latent_mechanics.belief.calibrate_noise
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from latent_mechanics.belief.basis import DEFAULT_TABLE
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import load_checkpoint

DATA = "runs/latent_mechanics/geometry/data_all_families.npz"


def measure(data: str = DATA, table_ckpt: str = DEFAULT_TABLE,
            split: str = "train") -> tuple[np.ndarray, int, int]:
    model, table, _, _ = load_checkpoint(table_ckpt, device="cpu",
                                         stage="belief:calibrate_noise")
    if table is None:
        raise ValueError(f"{table_ckpt} has no embedding table")
    model.freeze()
    z = table.weight.detach()

    ds = DoorTransitionDataset(data, split, exclude_near_limit=False)
    ids = np.unique(ds.door_id.numpy())
    chunks = []
    for did in ids:
        i = np.nonzero(ds.door_id.numpy() == did)[0]
        with torch.no_grad():
            eps = (model.raw_output(ds.state[i], ds.action[i],
                                    z[int(did)].reshape(1, -1))
                   - model.target(ds.state[i], ds.next_state[i]))
        chunks.append(eps.numpy().astype(np.float64))

    E = np.concatenate(chunks)
    return (E.T @ E) / len(E), len(ids), len(E)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    C, n_obj, n_tr = measure(args.data, args.table, args.split)
    w = np.linalg.eigvalsh(C)
    corr = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])

    print(f"irreducible measurement noise over {n_obj} objects, {n_tr} transitions")
    print(f"  E[eps eps^T] =\n{np.array2string(C, precision=6, prefix='    ')}")
    print(f"  eigenvalues      {w}")
    print(f"  RMS per channel  d_theta {np.sqrt(C[0, 0]):.4f}   d_omega {np.sqrt(C[1, 1]):.4f}")
    print(f"  cross-correlation {corr:.3f}  -- a scalar floor cannot represent this")
    print("\n  paste into noise.IRREDUCIBLE_R:")
    print("    " + np.array2string(C, precision=6, separator=", ", prefix="    "))

    from latent_mechanics.belief.noise import IRREDUCIBLE_R
    drift = np.abs(C - IRREDUCIBLE_R).max()
    print(f"\n  max abs drift from the committed constant: {drift:.2e}"
          + ("  OK" if drift < 1e-4 else "  STALE -- update the constant"))


if __name__ == "__main__":
    main()
