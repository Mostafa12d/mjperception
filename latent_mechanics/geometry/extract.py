"""
Step 1: extract every learned latent into a reusable analysis dataset.

No model weights are modified anywhere in this package. Checkpoints are opened
read-only, embedding tables are copied out, and each row is joined against the
mechanism metadata stored alongside the dataset it was trained on.

One gap had to be filled first. Stages 1-5 left seventeen checkpoints, but none
of them contains all six mechanism families at once: Stage 4's largest tables
hold 100 instances of five families (each ``exp3_no_*`` variant withholds one),
and Stage 5's hold 48 across six but only eight per family. A multimodality test
wants the widest table available, so ``build_all_families_checkpoint`` trains one
additional model on all six families using the *unchanged* Stage-1 pipeline and
Stage-4's cached rollouts. That is a new run of existing code, not a change to it.

The resulting dataset is saved as an ``.npz`` so every later step -- geometry,
mixture fitting, interpolation, Jacobians -- reads the same fixed table.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.mechanisms import library as lib
from latent_mechanics.mechanisms.data_gen import build_dataset_npz
from latent_mechanics.model import load_checkpoint

STAGE4_CACHE = Path("runs/latent_mechanics/mechanisms/suite_cache.pkl")
STAGE5_EVAL = Path("runs/latent_mechanics/curriculum/eval_suite.pkl")


@dataclass
class LatentDataset:
    """Every latent from one checkpoint, joined to its object metadata."""

    z: np.ndarray                # (N, d) the embeddings themselves
    family: np.ndarray           # (N,) mechanism category
    instance_id: np.ndarray      # (N,) row index = object instance
    split: np.ndarray            # (N,) "train" for every table row, by construction
    params: np.ndarray           # (N, P) true physical parameters (analysis only)
    param_names: list[str]
    checkpoint: str
    npz_path: str

    def __len__(self) -> int:
        return len(self.z)

    @property
    def dim(self) -> int:
        return self.z.shape[1]

    def summary(self) -> str:
        lines = [f"  {len(self)} latents of dimension {self.dim} "
                 f"from {Path(self.checkpoint).parent.name}"]
        for f in dict.fromkeys(self.family):
            lines.append(f"    {f:17s} {int((self.family == f).sum()):3d} instances")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, z=self.z, family=self.family, instance_id=self.instance_id,
            split=self.split, params=self.params,
            param_names=np.array(self.param_names),
            checkpoint=np.array(self.checkpoint), npz_path=np.array(self.npz_path),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "LatentDataset":
        with np.load(path, allow_pickle=False) as a:
            return LatentDataset(
                z=a["z"], family=np.array([str(x) for x in a["family"]]),
                instance_id=a["instance_id"],
                split=np.array([str(x) for x in a["split"]]),
                params=a["params"],
                param_names=[str(x) for x in a["param_names"]],
                checkpoint=str(a["checkpoint"]), npz_path=str(a["npz_path"]),
            )


def extract_from_checkpoint(ckpt: str | Path, data_npz: str | Path) -> LatentDataset:
    """Copy one checkpoint's embedding table out, with metadata attached."""
    _, table, _, _ = load_checkpoint(ckpt, device="cpu", stage="geometry_report:extract")
    if table is None:
        raise ValueError(f"{ckpt} has no embedding table")
    z = table.weight.detach().cpu().numpy().copy()

    with np.load(data_npz, allow_pickle=False) as a:
        fams = np.array([str(x) for x in a["mechanism_family"]])
        params = a["door_params"]
        names = [str(c) for c in a["door_params_columns"]]

    n = len(z)  # table rows exist only for training instances
    return LatentDataset(
        z=z, family=fams[:n], instance_id=np.arange(n),
        split=np.array(["train"] * n), params=params[:n], param_names=names,
        checkpoint=str(ckpt), npz_path=str(data_npz),
    )


def build_all_families_checkpoint(
    out_dir: Path, epochs: int = 40, force: bool = False
) -> tuple[Path, Path]:
    """Train one model on all six families. Uses the Stage-1 pipeline unchanged.

    Training instances come from Stage 4's cached rollouts (20 per family) and
    the held-out set is Stage 5's fixed evaluation suite (10 per family, a
    disjoint seed), so later steps have genuinely unseen objects to interpolate
    toward and to attribute error on.
    """
    ckpt = out_dir / "runs" / "all_families" / "best.pt"
    npz = out_dir / "data_all_families.npz"
    if ckpt.exists() and npz.exists() and not force:
        print(f"  reusing {ckpt}")
        return ckpt, npz

    if not STAGE4_CACHE.exists():
        raise FileNotFoundError(
            f"{STAGE4_CACHE} missing -- run latent_mechanics.mechanisms.study first")
    with open(STAGE4_CACHE, "rb") as f:
        train_pops = pickle.load(f)
    heldout = []
    if STAGE5_EVAL.exists():
        with open(STAGE5_EVAL, "rb") as f:
            heldout = pickle.load(f)

    stage1_cfg = load_stage1_config("configs/latent_mechanics.yaml")
    train_fams = list(dict.fromkeys(p.params.family for p in train_pops))
    print(f"  training on {len(train_pops)} instances of {len(train_fams)} families; "
          f"{len(heldout)} unseen instances held out")
    build_dataset_npz(train_pops, train_fams, npz, stage1_cfg, stage1_cfg.sim.frame_skip,
                      heldout_pops=heldout or None)

    from latent_mechanics.train import train as train_stage1
    cfg = load_stage1_config(None)
    cfg.model, cfg.train, cfg.sim = stage1_cfg.model, stage1_cfg.train, stage1_cfg.sim
    cfg.train.epochs = epochs
    cfg.train.run_dir = str(out_dir / "runs")
    cfg.train.run_name = "all_families"
    cfg.sim.exclude_near_limit = False
    ckpt = train_stage1(cfg, data_path=str(npz))
    return Path(ckpt), npz


def discover_checkpoints() -> list[tuple[str, Path, Path]]:
    """Every (label, checkpoint, dataset) triple Stages 1-5 left behind."""
    out = []
    base = Path("runs/latent_mechanics")
    s1 = base / "base" / "best.pt"
    if s1.exists():
        out.append(("stage1_doors", s1, Path("data/door_mechanics.npz")))
    for p in sorted((base / "mechanisms" / "runs").glob("*/best.pt")):
        npz = base / "mechanisms" / f"data_{p.parent.name}.npz"
        if npz.exists():
            out.append((f"stage4_{p.parent.name}", p, npz))
    for p in sorted((base / "curriculum" / "runs").glob("*/best.pt")):
        lvl = p.parent.name.split("_")[0]
        npz = base / "curriculum" / f"data_{lvl}.npz"
        if npz.exists():
            out.append((f"stage5_{p.parent.name}", p, npz))
    return out


__all__ = ["LatentDataset", "extract_from_checkpoint", "build_all_families_checkpoint",
           "discover_checkpoints"]
