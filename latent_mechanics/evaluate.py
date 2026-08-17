"""Evaluation of a trained stage-1 model on the doors it was trained on.

Reports one-step accuracy, multi-step rollout error, and a latent ablation. The
ablation is the load-bearing check: if shuffling the embeddings barely hurts, the
network is ignoring the latent and stage 2 has nothing to optimise.

    python3.10 -m latent_mechanics.evaluate \\
        --checkpoint runs/latent_mechanics/base/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from latent_mechanics import visualize
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import load_checkpoint
from latent_mechanics.rollout import aggregate_horizon_errors, horizon_errors
from latent_mechanics.train import resolve_device


@torch.no_grad()
def one_step_errors(
    model, latents: torch.Tensor, ds: DoorTransitionDataset, device, batch: int = 8192
) -> np.ndarray:
    """Signed one-step error for every transition: (N, 2). ``latents`` is indexed
    by door id, so a shuffled or zeroed table is all the ablation needs."""
    errs = []
    for lo in range(0, len(ds), batch):
        hi = min(lo + batch, len(ds))
        s = ds.state[lo:hi].to(device)
        a = ds.action[lo:hi].to(device)
        ns = ds.next_state[lo:hi].to(device)
        z = latents[ds.door_id[lo:hi]].to(device)
        errs.append((model(s, a, z) - ns).cpu().numpy())
    return np.concatenate(errs, axis=0)


def _metrics(err: np.ndarray) -> dict[str, float]:
    return {
        "rmse_angle": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_velocity": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "mae_angle": float(np.mean(np.abs(err[:, 0]))),
        "mae_velocity": float(np.mean(np.abs(err[:, 1]))),
        "max_abs_angle": float(np.max(np.abs(err[:, 0]))),
        "n": int(len(err)),
    }


def per_door_metrics(
    err: np.ndarray, ds: DoorTransitionDataset
) -> dict[int, dict[str, float]]:
    ids = ds.door_id.numpy()
    return {int(d): _metrics(err[ids == d]) for d in np.unique(ids)}


def latent_ablation(
    model, table_weight: torch.Tensor, ds: DoorTransitionDataset, device, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Correct vs zero vs shuffled embeddings, all else identical."""
    rng = np.random.default_rng(seed)
    n = table_weight.shape[0]
    # roll by a random nonzero offset, so no door keeps its own
    shuffled = table_weight[torch.from_numpy((np.arange(n) + rng.integers(1, n)) % n)]
    variants = {
        "correct": table_weight,
        "zero": torch.zeros_like(table_weight),
        "shuffled": shuffled,
    }
    return {
        name: _metrics(one_step_errors(model, w, ds, device))
        for name, w in variants.items()
    }


def latent_probe(
    latents: np.ndarray, ds: DoorTransitionDataset, alpha: float = 1.0
) -> dict[str, dict[str, float]]:
    """Leave-one-door-out ridge regression from ``z`` to each true parameter.

    Out-of-fold R^2 says the embedding encodes that quantity linearly readably.
    In-sample would be near-perfect by construction with 48 doors and 16 dims.
    """
    x = latents - latents.mean(0, keepdims=True)
    x = x / (x.std(0, keepdims=True) + 1e-8)
    n, d = x.shape
    out: dict[str, dict[str, float]] = {}

    for col in ds.door_params_columns:
        y = np.array([ds.params_for_door(i)[col] for i in range(n)], dtype=float)
        if y.std() < 1e-12:
            continue
        pred = np.empty(n)
        for k in range(n):
            m = np.ones(n, dtype=bool)
            m[k] = False
            xk, yk = x[m], y[m]
            mu = yk.mean()
            w = np.linalg.solve(xk.T @ xk + alpha * np.eye(d), xk.T @ (yk - mu))
            pred[k] = x[k] @ w + mu
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[col] = {
            "r2_loo": 1.0 - ss_res / ss_tot,
            "corr": float(np.corrcoef(y, pred)[0, 1]),
        }
    return out


def rollout_report(
    model, table_weight, ds: DoorTransitionDataset, horizons: list[int], device,
    limit: int | None = None,
) -> dict[int, dict[str, float]]:
    per_ep = []
    for ep in ds.episodes(limit=limit, seed=0):
        z = table_weight[ep.door_id]
        per_ep.append(horizon_errors(model, z, ep, horizons, device=device))
    return aggregate_horizon_errors(per_ep)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  table  -> {path}")


def evaluate(
    checkpoint: str | Path,
    data_path: str | None = None,
    split: str = "val",
    out_dir: str | None = None,
    device_spec: str | None = None,
) -> dict:
    model, table, cfg, extra = load_checkpoint(checkpoint, device="cpu",
                                               stage="stage1_evaluate")
    if table is None:
        raise ValueError(
            f"{checkpoint} has no embedding table; stage-1 evaluation needs the "
            "per-door latents"
        )
    device = resolve_device(device_spec or cfg.train.device)
    model.to(device).eval()
    weight = table.weight.detach().to(device)

    data_path = data_path or extra.get("data_path") or cfg.sim.out_path
    out_dir = Path(out_dir or cfg.eval.out_dir)
    ds = DoorTransitionDataset(
        data_path, split=split, exclude_near_limit=cfg.sim.exclude_near_limit
    )
    dt = ds.dt_model

    print(f"Evaluating {checkpoint}")
    print(f"  {ds.summary()}")
    print(f"  trained {extra.get('epoch', '?')} epochs, "
          f"val loss {extra.get('val_loss', float('nan')):.6f}\n")

    # -- 1. one-step accuracy -------------------------------------------
    err = one_step_errors(model, weight, ds, device)
    overall = _metrics(err)
    per_door = per_door_metrics(err, ds)

    print("1) One-step prediction accuracy "
          f"(dt = {dt:.3f} s, {split} split)")
    print(f"   angle    RMSE = {overall['rmse_angle']:.3e} rad "
          f"({np.degrees(overall['rmse_angle']):.4f} deg)")
    print(f"   velocity RMSE = {overall['rmse_velocity']:.3e} rad/s")
    # scale reference: RMSE well below a typical one-step change means the model
    # explains the motion, not just its smallness
    delta = ds.next_state.numpy() - ds.state.numpy()
    ref = np.sqrt(np.mean(delta**2, axis=0))
    print(f"   for scale: RMS one-step change = {ref[0]:.3e} rad, "
          f"{ref[1]:.3e} rad/s")
    print(f"   normalised error = {overall['rmse_angle'] / ref[0]:.4f} (angle), "
          f"{overall['rmse_velocity'] / ref[1]:.4f} (velocity)")

    worst = sorted(per_door.items(), key=lambda kv: -kv[1]["rmse_angle"])[:5]
    print("   worst doors by angle RMSE:")
    for did, m in worst:
        p = ds.params_for_door(did)
        print(f"     door {did:3d}: {m['rmse_angle']:.3e} rad  "
              f"(I={p['I_hinge']:.1f} mu={p['frictionloss']:.2f} "
              f"b={p['damping']:.2f} k={p['stiffness']:.2f})")

    # -- 2. rollout ------------------------------------------------------
    horizons = list(cfg.eval.horizons)
    agg = rollout_report(model, weight, ds, horizons, device)
    print(f"\n2) Multi-step rollout (averaged over every start index)")
    print(f"   {'horizon':>9} {'time':>7} {'RMSE angle':>13} {'RMSE vel':>13}")
    for h in sorted(agg):
        print(f"   {h:>9} {h * dt:>6.2f}s {agg[h]['rmse_angle']:>13.3e} "
              f"{agg[h]['rmse_velocity']:>13.3e}")

    # -- 3. latent ablation ----------------------------------------------
    abl = latent_ablation(model, weight, ds, device)
    print("\n3) Latent ablation (identical network, different embedding)")
    base = abl["correct"]["rmse_angle"]
    for name in ("correct", "zero", "shuffled"):
        m = abl[name]
        print(f"   {name:9s}: angle RMSE = {m['rmse_angle']:.3e} rad  "
              f"({m['rmse_angle'] / base:6.2f}x correct)  "
              f"vel RMSE = {m['rmse_velocity']:.3e}")
    ratio = abl["shuffled"]["rmse_angle"] / base
    if ratio < 1.5:
        print("   WARNING: shuffling the embeddings barely changes the error. The "
              "network is\n            ignoring the latent -- stage-2 adaptation "
              "would have nothing to optimise.")
    else:
        print(f"   Embedding carries real information: a wrong latent costs "
              f"{ratio:.1f}x the error.")

    # -- 4. what the latent encodes ---------------------------------------
    latents = weight.cpu().numpy()
    probe = latent_probe(latents, ds)
    print("\n4) Linear probe: embedding -> true physics (leave-one-door-out R^2)")
    print("   These parameters are never an input or a target during training.")
    for col, m in sorted(probe.items(), key=lambda kv: -kv[1]["r2_loo"]):
        bar = "#" * max(0, int(round(20 * max(m["r2_loo"], 0.0))))
        print(f"   {col:14s} R^2 = {m['r2_loo']:+.3f}  r = {m['corr']:+.3f}  {bar}")

    # -- outputs ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nWriting artefacts")
    _write_csv(
        out_dir / "per_door_metrics.csv",
        [
            {"door_id": d, **m, **{k: ds.params_for_door(d)[k]
                                   for k in ds.door_params_columns}}
            for d, m in sorted(per_door.items())
        ],
    )
    _write_csv(
        out_dir / "horizon_metrics.csv",
        [{"horizon": h, "seconds": h * dt, **agg[h]} for h in sorted(agg)],
    )

    np.save(out_dir / "embeddings.npy", latents)
    print(f"  latents-> {out_dir / 'embeddings.npy'}")

    eps = list(ds.episodes(limit=cfg.eval.n_plot_episodes, seed=1))
    latent_map = {int(i): weight[i] for i in range(weight.shape[0])}
    visualize.plot_rollouts(
        model, latent_map, ds, eps, out_dir / "rollouts.png", dt, device=device
    )
    visualize.plot_horizon_curve(
        agg, out_dir / "horizon_error.png", dt,
        title=f"Multi-step prediction error ({split} split)",
    )
    visualize.plot_per_door_error(per_door, ds, out_dir / "per_door_error.png")
    visualize.plot_latent_space(latents, ds, out_dir / "latent_space.png")

    summary = {
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "split": split,
        "dt_model": dt,
        "one_step": overall,
        "one_step_reference_rms_delta": {"angle": float(ref[0]), "velocity": float(ref[1])},
        "rollout": {str(h): agg[h] for h in sorted(agg)},
        "latent_ablation": abl,
        "latent_probe": probe,
        "n_doors": len(per_door),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  summary-> {out_dir / 'summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/latent_mechanics/base/best.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument(
        "--split", default="val",
        help="'val' = unseen episodes from training doors (default); 'train' = "
             "fit quality; 'heldout_door' needs stage-2 and will fail here",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.split == "heldout_door":
        raise SystemExit(
            "held-out doors have no embedding row -- evaluating them is the "
            "stage-2 online-adaptation experiment, not stage 1"
        )
    evaluate(args.checkpoint, args.data, args.split, args.out_dir, args.device)


if __name__ == "__main__":
    main()
