"""Stage-1 training: the dynamics network and one embedding per training door,
optimised jointly against MSE on the next state.

Embeddings see far fewer gradients each, so they run on their own larger LR.

    python3.10 -m latent_mechanics.train --config configs/latent_mechanics.yaml
    tensorboard --logdir runs/latent_mechanics
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from latent_mechanics import provenance
from latent_mechanics.config import ExperimentConfig, load_config
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import (
    DoorEmbeddingTable,
    build_model_from_config,
    save_checkpoint,
)
from latent_mechanics.rollout import aggregate_horizon_errors, horizon_errors


def resolve_device(spec: str) -> torch.device:
    """'auto' prefers CUDA, else CPU. MPS is never chosen automatically: for an MLP
    this small its launch overhead usually loses to CPU."""
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_optimizer(
    model: nn.Module, embeddings: DoorEmbeddingTable, cfg: ExperimentConfig
) -> torch.optim.Optimizer:
    """Two groups: the shared network, and the per-door latents on a larger LR and
    stronger decay (which keeps the latent cloud compact)."""
    return torch.optim.AdamW(
        [
            {
                "params": list(model.parameters()),
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "name": "network",
            },
            {
                "params": list(embeddings.parameters()),
                "lr": cfg.train.embedding_lr,
                "weight_decay": cfg.train.embedding_weight_decay,
                "name": "embeddings",
            },
        ]
    )


def lr_scale(epoch: int, cfg: ExperimentConfig) -> float:
    """Warmup then cosine decay, as a multiplier on each group's base LR."""
    warmup = cfg.train.warmup_epochs
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / warmup
    if cfg.train.lr_schedule == "none":
        return 1.0
    progress = (epoch - warmup) / max(1, cfg.train.epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def batch_losses(
    model, embeddings, batch, device, loss_space: str
) -> tuple[torch.Tensor, dict[str, float]]:
    """Training loss plus raw-unit metrics. ``loss_space='normalized'`` divides
    each dimension by its std, so velocity does not dominate the gradient."""
    state = batch["state"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    next_state = batch["next_state"].to(device, non_blocking=True)
    z = embeddings(batch["door_id"].to(device, non_blocking=True))

    if loss_space == "normalized":
        loss = nn.functional.mse_loss(
            model.raw_output(state, action, z), model.target(state, next_state)
        )
    elif loss_space == "raw":
        loss = nn.functional.mse_loss(model(state, action, z), next_state)
    else:
        raise ValueError(f"loss_space must be 'normalized' or 'raw', got '{loss_space}'")

    with torch.no_grad():
        err = model(state, action, z) - next_state
        metrics = {
            "mse_raw": float((err**2).mean()),
            "rmse_angle": float(err[:, 0].pow(2).mean().sqrt()),
            "rmse_velocity": float(err[:, 1].pow(2).mean().sqrt()),
        }
    return loss, metrics


@torch.no_grad()
def evaluate_split(model, embeddings, loader, device, loss_space) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_total = 0
    for batch in loader:
        n = len(batch["state"])
        loss, metrics = batch_losses(model, embeddings, batch, device, loss_space)
        totals["loss"] = totals.get("loss", 0.0) + float(loss) * n
        # pooled as sums of squares: RMSE over the split, not a mean of RMSEs
        for k in ("rmse_angle", "rmse_velocity"):
            totals[k] = totals.get(k, 0.0) + metrics[k] ** 2 * n
        totals["mse_raw"] = totals.get("mse_raw", 0.0) + metrics["mse_raw"] * n
        n_total += n
    out = {"loss": totals["loss"] / n_total, "mse_raw": totals["mse_raw"] / n_total}
    for k in ("rmse_angle", "rmse_velocity"):
        out[k] = math.sqrt(totals[k] / n_total)
    model.train()
    return out


@torch.no_grad()
def rollout_metrics(
    model, embeddings, dataset: DoorTransitionDataset, cfg: ExperimentConfig, device
) -> dict[str, float]:
    """Multi-step error on a sample of episodes, using each door's learned latent."""
    model.eval()
    horizon = cfg.train.rollout_eval_horizon
    per_ep = []
    for ep in dataset.episodes(limit=cfg.train.rollout_eval_episodes, seed=0):
        z = embeddings(torch.tensor([ep.door_id], device=device))[0]
        per_ep.append(horizon_errors(model, z, ep, [1, horizon], device=device))
    model.train()
    agg = aggregate_horizon_errors(per_ep)
    return {
        f"h{h}_{k}": v
        for h, row in agg.items()
        for k, v in row.items()
        if k in ("rmse_angle", "rmse_velocity")
    }


def train(cfg: ExperimentConfig, data_path: str | None = None) -> Path:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    data_path = data_path or cfg.sim.out_path

    excl = cfg.sim.exclude_near_limit
    train_ds = DoorTransitionDataset(data_path, split="train", exclude_near_limit=excl)
    val_ds = DoorTransitionDataset(data_path, split="val", exclude_near_limit=excl)
    print(train_ds.summary())
    print(val_ds.summary())

    # train-split statistics, frozen into the model so every later stage shares a scale
    norm_stats = train_ds.norm_stats()
    model = build_model_from_config(cfg.model, norm_stats).to(device)
    embeddings = DoorEmbeddingTable(
        num_doors=train_ds.num_embedding_rows,
        embed_dim=cfg.model.embed_dim,
        init_std=cfg.model.embedding_init_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nmodel: {n_params} weights, embed_dim={cfg.model.embed_dim}, "
        f"table={train_ds.num_embedding_rows} doors x {cfg.model.embed_dim} "
        f"= {train_ds.num_embedding_rows * cfg.model.embed_dim} latent params"
    )
    print(f"device: {device}   loss space: {cfg.train.loss_space}\n")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    optimizer = make_optimizer(model, embeddings, cfg)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    run_dir = cfg.run_path
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    writer = SummaryWriter(str(run_dir / "tb"))

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    global_step = 0
    history = []
    t_start = time.time()

    for epoch in range(cfg.train.epochs):
        scale = lr_scale(epoch, cfg)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        model.train()
        running, n_seen = 0.0, 0
        for batch in train_loader:
            loss, metrics = batch_losses(
                model, embeddings, batch, device, cfg.train.loss_space
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(embeddings.parameters()),
                    cfg.train.grad_clip,
                )
            optimizer.step()

            loss_value = loss.detach().item()
            running += loss_value * len(batch["state"])
            n_seen += len(batch["state"])
            if global_step % cfg.train.log_every == 0:
                writer.add_scalar("train/loss_step", loss_value, global_step)
                writer.add_scalar("train/rmse_angle_step", metrics["rmse_angle"], global_step)
            global_step += 1

        train_loss = running / max(n_seen, 1)
        val = evaluate_split(model, embeddings, val_loader, device, cfg.train.loss_space)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val["loss"], epoch)
        writer.add_scalar("val/mse_raw", val["mse_raw"], epoch)
        writer.add_scalar("val/rmse_angle", val["rmse_angle"], epoch)
        writer.add_scalar("val/rmse_velocity", val["rmse_velocity"], epoch)
        writer.add_scalar("lr/network", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("lr/embeddings", optimizer.param_groups[1]["lr"], epoch)

        # latent geometry: collapsing norms mean the doors are not distinguished
        with torch.no_grad():
            w = embeddings.weight
            writer.add_scalar("embed/norm_mean", float(w.norm(dim=1).mean()), epoch)
            writer.add_scalar("embed/norm_max", float(w.norm(dim=1).max()), epoch)
            writer.add_scalar("embed/spread", float(w.std(dim=0).mean()), epoch)
            writer.add_histogram("embed/values", w, epoch)

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}

        if (epoch + 1) % cfg.train.rollout_eval_every == 0 or epoch == cfg.train.epochs - 1:
            rm = rollout_metrics(model, embeddings, val_ds, cfg, device)
            for k, v in rm.items():
                writer.add_scalar(f"val_rollout/{k}", v, epoch)
            row.update({f"rollout_{k}": v for k, v in rm.items()})

        history.append(row)

        improved = val["loss"] < best_val - 1e-12
        if improved:
            best_val, best_epoch, epochs_since_best = val["loss"], epoch, 0
            save_checkpoint(
                run_dir / "best.pt", model, embeddings, cfg,
                extra={"epoch": epoch, "val_loss": val["loss"],
                       "val_rmse_angle": val["rmse_angle"],
                       "val_rmse_velocity": val["rmse_velocity"],
                       "data_path": str(data_path)},
            )
        else:
            epochs_since_best += 1

        h_msg = ""
        if "rollout_h{}_rmse_angle".format(cfg.train.rollout_eval_horizon) in row:
            h = cfg.train.rollout_eval_horizon
            h_msg = f"  roll{h}_ang={row[f'rollout_h{h}_rmse_angle']:.4f}"
        print(
            f"epoch {epoch + 1:3d}/{cfg.train.epochs}  "
            f"train={train_loss:.5f}  val={val['loss']:.5f}  "
            f"ang={val['rmse_angle']:.2e} rad  vel={val['rmse_velocity']:.2e} rad/s"
            f"{h_msg}{'  *' if improved else ''}"
        )

        if cfg.train.early_stop_patience and epochs_since_best >= cfg.train.early_stop_patience:
            print(f"early stop: no improvement for {epochs_since_best} epochs")
            break

    save_checkpoint(
        run_dir / "last.pt", model, embeddings, cfg,
        extra={"epoch": epoch, "val_loss": val["loss"], "data_path": str(data_path)},
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    writer.close()

    print(
        f"\ndone in {time.time() - t_start:.1f}s  "
        f"best val loss {best_val:.6f} at epoch {best_epoch + 1}"
    )
    print(f"checkpoints: {run_dir / 'best.pt'} , {run_dir / 'last.pt'}")
    # record what was produced, so a downstream provenance line matches back here
    provenance.log_checkpoint(run_dir / "best.pt",
                              stage=f"trained:{cfg.train.run_name}",
                              table_rows=getattr(embeddings, "num_doors", None))
    print(f"tensorboard: tensorboard --logdir {cfg.train.run_dir}")
    return run_dir / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/latent_mechanics.yaml")
    ap.add_argument("--data", default=None, help="override sim.out_path")
    ap.add_argument("--run-name", default=None, help="override train.run_name")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--embed-dim", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg.train.run_name = args.run_name
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.embed_dim is not None:
        cfg.model.embed_dim = args.embed_dim
    if args.device:
        cfg.train.device = args.device
    train(cfg, data_path=args.data)


if __name__ == "__main__":
    main()
