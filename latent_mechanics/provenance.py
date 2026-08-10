"""Checkpoint provenance: which frozen predictor did this result actually use?

Stages 1-6 accumulated seventeen checkpoints, and the audit found four *different*
predictors in use across Stage 3, Stage 5 and the geometry/UKF branch. That is
defensible -- no single earlier stage trained on all six families -- but it was
only discoverable by reading four modules and hashing the files by hand, and
nothing would have complained if a fifth had appeared.

This module makes the loaded checkpoint's identity part of the output. Every load
goes through ``log_checkpoint`` and prints one line naming the stage, the sha256
and the embedding-table shape. Where a config pins ``expected_sha256``, a
mismatch raises instead of printing.

It deliberately does NOT fix the divergence; it makes any future divergence loud.

    python3.10 -m latent_mechanics.provenance      # table of every stage -> hash
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

# Where each stage's predictor path is defined, so the table below stays honest
# about *why* a stage loads what it loads rather than just recording the path.
STAGE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("stage1_base", "train.py run_dir/run_name (cfg.train)",
     "runs/latent_mechanics/base/best.pt"),
    ("stage1_evaluate", "evaluate.py --checkpoint (default)",
     "runs/latent_mechanics/base/best.pt"),
    ("stage3_online", "configs/online_adaptation.yaml -> experiments.checkpoint",
     "runs/latent_mechanics/base/best.pt"),
    ("mismatch_study", "mismatch/config.py MismatchConfig.checkpoint (default)",
     "runs/latent_mechanics/base/best.pt"),
    ("stage4_mechanisms", "mechanisms/study.py per-experiment run dir",
     "runs/latent_mechanics/mechanisms/runs/*/best.pt"),
    ("stage5_curriculum", "curriculum/study.py per-level run dir",
     "runs/latent_mechanics/curriculum/runs/*/best.pt"),
    ("geometry_report", "geometry/extract.build_all_families_checkpoint",
     "runs/latent_mechanics/geometry/runs/all_families/best.pt"),
    ("belief_ukf", "belief/basis.DEFAULT_TABLE",
     "runs/latent_mechanics/geometry/runs/all_families/best.pt"),
)

_quiet = False
_seen: dict[str, tuple[str, str]] = {}       # stage -> (sha256, path)


def set_quiet(quiet: bool = True) -> None:
    """Silence the per-load line (tests and sweeps that load in a loop)."""
    global _quiet
    _quiet = quiet


def file_sha256(path: str | Path) -> str:
    """sha256 of a file, streamed so a large checkpoint is not read into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_checkpoint(
    path: str | Path,
    stage: str = "unlabelled",
    expected_sha256: str | None = None,
    table_rows: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Record (and optionally enforce) the identity of a predictor checkpoint.

    ``expected_sha256`` may be a full hash or any leading prefix, so a config can
    pin the first 16 hex characters and stay readable. Raises on mismatch.
    """
    path = Path(path)
    digest = file_sha256(path)
    _seen[stage] = (digest, str(path))

    if expected_sha256:
        want = str(expected_sha256).strip().lower()
        if not digest.startswith(want):
            raise ValueError(
                f"checkpoint hash mismatch for stage '{stage}':\n"
                f"  path     {path}\n"
                f"  expected {want}\n"
                f"  actual   {digest}\n"
                "The frozen predictor is not the one this result was configured "
                "against. Either point the config at the intended checkpoint or "
                "update expected_sha256 deliberately."
            )

    if not _quiet:
        bits = [f"stage={stage}", f"sha256={digest[:16]}"]
        if table_rows is not None:
            bits.append(f"table_rows={table_rows}")
        if expected_sha256:
            bits.append("pinned=ok")
        bits.append(f"path={path}")
        print("  [provenance] " + "  ".join(bits))
    return digest


def loaded() -> dict[str, tuple[str, str]]:
    """Every (stage -> (sha256, path)) recorded in this process."""
    return dict(_seen)


def resolve(pattern: str) -> list[Path]:
    """Expand a STAGE_SOURCES path, which may contain a glob."""
    p = Path(pattern)
    if "*" not in pattern:
        return [p] if p.exists() else []
    base = Path(pattern.split("*")[0]).parent
    return sorted(base.glob(pattern[len(str(base)) + 1:])) if base.exists() else []


def report() -> list[dict[str, Any]]:
    """Table of every stage's predictor and its hash, from the files on disk."""
    import torch

    rows: list[dict[str, Any]] = []
    for stage, source, pattern in STAGE_SOURCES:
        paths = resolve(pattern)
        if not paths:
            rows.append({"stage": stage, "source": source, "path": pattern,
                         "sha256": None, "table_rows": None, "present": False})
            continue
        for p in paths:
            rows_n = None
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
                es = payload.get("embedding_state")
                if es:
                    two_d = [v for v in es.values() if getattr(v, "ndim", 0) == 2]
                    if two_d:
                        rows_n = int(two_d[0].shape[0])
            except Exception:
                pass
            rows.append({"stage": stage, "source": source, "path": str(p),
                         "sha256": file_sha256(p), "table_rows": rows_n,
                         "present": True})
    return rows


def print_report() -> list[dict[str, Any]]:
    rows = report()
    print(f"{'stage':20s} {'sha256[:16]':18s} {'rows':>5s}  path")
    print("-" * 108)
    for r in rows:
        h = r["sha256"][:16] if r["sha256"] else "MISSING"
        n = r["table_rows"] if r["table_rows"] is not None else "-"
        print(f"{r['stage']:20s} {h:18s} {str(n):>5s}  {r['path']}")

    print("\nDistinct predictors per stage group:")
    groups: dict[str, set[str]] = {}
    for r in rows:
        if r["sha256"]:
            groups.setdefault(r["stage"], set()).add(r["sha256"])
    for stage, hs in groups.items():
        print(f"  {stage:20s} {len(hs)} distinct")
    cross = {r["sha256"] for r in rows
             if r["sha256"] and r["stage"] in
             ("stage3_online", "stage5_curriculum", "geometry_report", "belief_ukf")}
    print(f"\n  Stage 3 / Stage 5 / geometry / belief span {len(cross)} distinct "
          f"checkpoints.")
    return rows


def main() -> None:
    print(__doc__.strip().splitlines()[0] + "\n")
    print_report()


if __name__ == "__main__":
    main()
