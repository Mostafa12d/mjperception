"""Stage-2 experiments, all sharing one driver and protocol (see ``loop.py``):

  1. Prediction error vs interaction number on an unseen door.
  2. Latent initialisation: zero / random trained / mean / medoid.
  3. Latent adaptation vs the RLS baseline on identical doors and streams.

    python3.10 -m latent_mechanics.online.experiments --config configs/online_adaptation.yaml
    python3.10 -m latent_mechanics.online.experiments --only 3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online import viz
from latent_mechanics.online.adaptor import (
    GradientLatentAdaptor,
    StaticLatentAdaptor,
)
from latent_mechanics.online.config import OnlineConfig, load_config
from latent_mechanics.online.loop import (
    episode_boundaries,
    episode_stream,
    init_strategies,
    run_online_adaptation,
)
from latent_mechanics.online.rls_adaptor import RLSAdaptor


class Context:
    """Everything the experiments share: the frozen model, data, references."""

    def __init__(self, cfg: OnlineConfig) -> None:
        exp = cfg.experiments
        self.cfg = cfg
        self.device = torch.device(exp.device)

        model, table, stage1_cfg, extra = load_checkpoint(
            exp.checkpoint, device=self.device, stage="stage3_online",
            expected_sha256=exp.expected_sha256)
        if table is None:
            raise ValueError("checkpoint has no embedding table; needed for init strategies")
        self.model = model.freeze()
        self.stage1_cfg = stage1_cfg
        self.train_latents = table.weight.detach().cpu().numpy()

        excl = stage1_cfg.sim.exclude_near_limit
        self.ds = DoorTransitionDataset(exp.data, split=exp.split, exclude_near_limit=excl)
        self.train_ds = DoorTransitionDataset(exp.data, split="train", exclude_near_limit=excl)
        self.dt = self.ds.dt_model

        self.door_ids = list(exp.door_ids) or [int(d) for d in self.ds.door_ids]
        self.focus_door = exp.focus_door if exp.focus_door is not None else self.door_ids[0]

        # Stage-1 reference: a door the network was trained on, with its own
        # learned latent. This is the floor that adaptation is chasing.
        self.reference = {
            "angle": float(extra.get("val_rmse_angle", np.nan)),
            "velocity": float(extra.get("val_rmse_velocity", np.nan)),
        }
        self.out = Path(exp.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def stream(self, door_id: int):
        exp = self.cfg.experiments
        tr = episode_stream(self.ds, door_id, exp.max_episodes)
        bounds = episode_boundaries(self.ds, door_id, exp.max_episodes)
        return tr, bounds

    def make_latent_adaptor(self, init: np.ndarray) -> GradientLatentAdaptor:
        a = self.cfg.adaptor
        return GradientLatentAdaptor(
            self.model, init=init, lr=a.lr, optimizer=a.optimizer,
            n_inner_steps=a.n_inner_steps, window=a.window,
            prior_weight=a.prior_weight, loss_space=a.loss_space,
            max_grad_norm=a.max_grad_norm, lr_decay=a.lr_decay, device=self.device,
        )

    def make_static_adaptor(self, init: np.ndarray) -> StaticLatentAdaptor:
        """The no-adaptation control: same latent, updates disabled."""
        return StaticLatentAdaptor(self.model, init=init, device=self.device)

    def make_rls_adaptor(self, n_params: int) -> RLSAdaptor:
        r = self.cfg.rls
        return RLSAdaptor(dt=self.dt, n_substeps=r.n_substeps, n_params=n_params,
                          lam=r.lam, delta=r.delta, vel_thresh=r.vel_thresh)

    def door_label(self, door_id: int) -> str:
        p = self.ds.params_for_door(door_id)
        return (f"door {door_id} (I={p['I_hinge']:.1f}, mu={p['frictionloss']:.2f}, "
                f"b={p['damping']:.2f}, k={p['stiffness']:.2f})")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k: None for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  table  -> {path}")


def experiment_1(ctx: Context) -> dict:
    print("\n" + "=" * 74)
    print("EXPERIMENT 1  prediction error vs interaction number")
    print("=" * 74)
    init_name = ctx.cfg.experiments.default_init
    init = init_strategies(ctx.train_latents, ctx.cfg.experiments.seed)[init_name]
    window = ctx.cfg.experiments.rolling_window
    print(f"  latent initialised from: {init_name}")

    # the control: identical latent, updates disabled. The only honest "before"
    # number, since the adapted run's early steps have already started learning.
    rows, logs, static_logs = [], [], []
    for did in ctx.door_ids:
        tr, bounds = ctx.stream(did)
        lg = run_online_adaptation(ctx.make_latent_adaptor(init), tr, door_id=did,
                                   boundaries=bounds, init_name=f"{init_name} init")
        ctrl = run_online_adaptation(ctx.make_static_adaptor(init), tr, door_id=did,
                                     boundaries=bounds, init_name="no adaptation")
        logs.append(lg)
        static_logs.append(ctrl)
        before, final = ctrl.rmse(0), lg.final_rmse(0)
        rows.append({
            "door_id": did, "n_interactions": len(lg),
            "angle_rmse_no_adaptation": before,
            "angle_rmse_final25pct": final,
            "improvement_x": before / final if final > 0 else np.nan,
            "vel_rmse_no_adaptation": ctrl.rmse(1),
            "vel_rmse_final25pct": lg.final_rmse(1),
            **{k: ctx.ds.params_for_door(did)[k] for k in ("I_hinge", "frictionloss",
                                                           "damping", "stiffness")},
        })
        print(f"  {ctx.door_label(did):58s} {len(lg):5d} steps  "
              f"angle RMSE {before:.3e} -> {final:.3e}  ({before / final:.1f}x better)")

    med = float(np.median([r["improvement_x"] for r in rows]))
    print(f"\n  median improvement over the no-adaptation control: {med:.1f}x")
    print(f"  stage-1 reference (trained door): {ctx.reference['angle']:.3e} rad")

    _write_csv(ctx.out / "exp1_per_door.csv", rows)
    focus = [lg for lg in logs if lg.door_id == ctx.focus_door] or logs[:1]
    focus_ctrl = [lg for lg in static_logs if lg.door_id == focus[0].door_id]
    viz.plot_error_curve(
        focus + focus_ctrl, ctx.out / "exp1_error_curve.png", window,
        title=f"Experiment 1: online adaptation on {ctx.door_label(focus[0].door_id)}",
        reference=ctx.reference,
    )
    viz.plot_error_curve(
        logs, ctx.out / "exp1_error_curve_all_doors.png", window,
        title="Experiment 1: all unseen doors", reference=ctx.reference,
    )
    for lg in logs:
        lg.init_name = f"door {lg.door_id}"
    return {"per_door": rows, "median_improvement_x": med, "logs": logs}


def experiment_2(ctx: Context) -> dict:
    print("\n" + "=" * 74)
    print("EXPERIMENT 2  latent initialisation")
    print("=" * 74)
    inits = init_strategies(ctx.train_latents, ctx.cfg.experiments.seed)
    window = ctx.cfg.experiments.rolling_window

    # Convergence threshold shared by every strategy, so "steps to converge" is
    # comparable: 1.5x the best final error any strategy reaches on this door.
    per_door_logs: dict[int, list] = {}
    rows = []
    for did in ctx.door_ids:
        tr, bounds = ctx.stream(did)
        logs = []
        for name, z0 in inits.items():
            ad = ctx.make_latent_adaptor(z0)
            lg = run_online_adaptation(ad, tr, door_id=did, boundaries=bounds,
                                       init_name=name)
            logs.append(lg)
        per_door_logs[did] = logs
        thresh = 1.5 * min(lg.final_rmse(0) for lg in logs)
        for lg in logs:
            steps = lg.steps_to(thresh, 0, window)
            rows.append({
                "door_id": did, "init": lg.init_name,
                "start_rmse": lg.rmse(0, last=max(1, len(lg) // 20)),
                "final_rmse": lg.final_rmse(0),
                "steps_to_threshold": steps if steps is not None else -1,
                "threshold": thresh,
                "latent_travel": float(np.linalg.norm(lg.latents[-1] - lg.latents[0])),
            })

    print(f"  {'init':16s} {'start RMSE':>12} {'final RMSE':>12} "
          f"{'steps to conv':>14} {'travel':>8}")
    summary = {}
    for name in inits:
        sub = [r for r in rows if r["init"] == name]
        conv = [r["steps_to_threshold"] for r in sub if r["steps_to_threshold"] >= 0]
        summary[name] = {
            "start_rmse": float(np.mean([r["start_rmse"] for r in sub])),
            "final_rmse": float(np.mean([r["final_rmse"] for r in sub])),
            "median_steps_to_threshold": float(np.median(conv)) if conv else None,
            "n_converged": len(conv), "n_doors": len(sub),
            "latent_travel": float(np.mean([r["latent_travel"] for r in sub])),
        }
        s = summary[name]
        steps_txt = (f"{s['median_steps_to_threshold']:.0f} ({s['n_converged']}/{s['n_doors']})"
                     if s["median_steps_to_threshold"] is not None else "never")
        print(f"  {name:16s} {s['start_rmse']:>12.3e} {s['final_rmse']:>12.3e} "
              f"{steps_txt:>14} {s['latent_travel']:>8.2f}")

    _write_csv(ctx.out / "exp2_init_comparison.csv", rows)
    focus_logs = per_door_logs.get(ctx.focus_door, next(iter(per_door_logs.values())))
    viz.plot_init_comparison(focus_logs, ctx.out / "exp2_init_comparison.png",
                             window, ctx.reference["angle"])
    return {"rows": rows, "summary": summary}


def experiment_3(ctx: Context) -> dict:
    print("\n" + "=" * 74)
    print("EXPERIMENT 3  latent adaptation vs RLS (identical doors and streams)")
    print("=" * 74)
    print("  Both estimators see the same 50 Hz transitions in the same order and")
    print("  are scored on one-step-ahead prediction made before each update.\n")
    inits = init_strategies(ctx.train_latents, ctx.cfg.experiments.seed)
    window = ctx.cfg.experiments.rolling_window

    rows, per_door_logs = [], {}
    for did in ctx.door_ids:
        tr, bounds = ctx.stream(did)
        methods = {
            "no-adaptation": ctx.make_static_adaptor(inits[ctx.cfg.experiments.default_init]),
            "latent-gd": ctx.make_latent_adaptor(inits[ctx.cfg.experiments.default_init]),
        }
        for n_params in ctx.cfg.rls.n_params:
            methods[f"rls-{n_params}p"] = ctx.make_rls_adaptor(n_params)

        logs = []
        for name, ad in methods.items():
            lg = run_online_adaptation(ad, tr, door_id=did, boundaries=bounds)
            lg.name = name
            logs.append(lg)
        per_door_logs[did] = logs

        for lg in logs:
            # "own" = speed to 1.5x this method's own final error; "shared" =
            # speed to 1.5x the best any method reaches, a bar some never clear
            own = lg.steps_to(1.5 * lg.final_rmse(0), 0, window)
            shared_thresh = 1.5 * min(x.final_rmse(0) for x in logs)
            shared = lg.steps_to(shared_thresh, 0, window)
            rows.append({
                "door_id": did, "method": lg.name,
                "angle_rmse_final": lg.final_rmse(0),
                "vel_rmse_final": lg.final_rmse(1),
                "angle_rmse_all": lg.rmse(0),
                "steps_to_own": own if own is not None else -1,
                "steps_to_shared": shared if shared is not None else -1,
                "shared_threshold": shared_thresh,
                "us_per_update": 1e6 * lg.seconds_per_update,
                "total_seconds": lg.total_seconds,
            })

    names = list(dict.fromkeys(r["method"] for r in rows))
    print(f"  {'method':14s} {'angle RMSE':>12} {'vel RMSE':>12} "
          f"{'own conv':>10} {'shared conv':>13} {'us/update':>11}")
    summary = {}
    for name in names:
        sub = [r for r in rows if r["method"] == name]
        own = [r["steps_to_own"] for r in sub if r["steps_to_own"] >= 0]
        shared = [r["steps_to_shared"] for r in sub if r["steps_to_shared"] >= 0]
        summary[name] = {
            "angle_rmse_final": float(np.mean([r["angle_rmse_final"] for r in sub])),
            "vel_rmse_final": float(np.mean([r["vel_rmse_final"] for r in sub])),
            "median_steps_to_own": float(np.median(own)) if own else None,
            "median_steps_to_shared": float(np.median(shared)) if shared else None,
            "n_reached_shared": len(shared), "n_doors": len(sub),
            "us_per_update": float(np.mean([r["us_per_update"] for r in sub])),
        }
        s = summary[name]
        own_txt = f"{s['median_steps_to_own']:.0f}" if own else "n/a"
        sh_txt = (f"{s['median_steps_to_shared']:.0f} ({len(shared)}/{s['n_doors']})"
                  if shared else f"never (0/{s['n_doors']})")
        print(f"  {name:14s} {s['angle_rmse_final']:>12.3e} {s['vel_rmse_final']:>12.3e} "
              f"{own_txt:>10} {sh_txt:>13} {s['us_per_update']:>11.1f}")
    print(f"\n  stage-1 reference (trained door, learned latent): "
          f"{ctx.reference['angle']:.3e} rad")

    _write_csv(ctx.out / "exp3_method_comparison.csv", rows)
    focus_logs = per_door_logs.get(ctx.focus_door, next(iter(per_door_logs.values())))
    viz.plot_method_comparison(focus_logs, ctx.out / "exp3_method_comparison.png",
                               window, ctx.reference["angle"])
    return {"rows": rows, "summary": summary}


def visualise_belief(ctx: Context) -> dict:
    print("\n" + "=" * 74)
    print("VISUALISATION  belief trajectory through latent space")
    print("=" * 74)
    exp = ctx.cfg.experiments
    did = ctx.focus_door
    print(f"  focus: {ctx.door_label(did)}")

    tr, bounds = ctx.stream(did)
    init = init_strategies(ctx.train_latents, exp.seed)[exp.default_init]
    ad = ctx.make_latent_adaptor(init)
    lg = run_online_adaptation(ad, tr, door_id=did, boundaries=bounds,
                               init_name=f"{exp.default_init} init")

    # PCA is fitted on the TRAINING latents only, so the axes match the stage-1
    # latent-space figure and stay fixed across every frame.
    pca = viz.LatentPCA.fit(ctx.train_latents)
    train_I = np.array([ctx.train_ds.params_for_door(i)["I_hinge"]
                        for i in range(len(ctx.train_latents))])

    # Prepend z0 so the very first frame is the initialisation itself.
    traj = np.concatenate([init.reshape(1, -1), lg.latents], axis=0)

    viz.plot_latent_trajectory(
        pca, traj, ctx.out / "belief_trajectory.png",
        train_color=train_I, color_label="training door $I_{hinge}$",
        title=f"Online belief on unseen {ctx.door_label(did)}",
    )
    viz.plot_latent_snapshots(
        pca, traj, ctx.out / "belief_snapshots.png", n_panels=6,
        train_color=train_I, color_label="coloured by training-door inertia",
    )
    video = None
    if exp.make_animation:
        video = viz.animate_latent_trajectory(
            pca, traj, ctx.out / "belief_trajectory.mp4",
            rolling_error=lg.rolling_rmse(0, exp.rolling_window),
            train_color=train_I, color_label="training door $I_{hinge}$",
            stride=exp.animation_stride, fps=exp.animation_fps,
        )

    np.savez_compressed(
        ctx.out / "belief_trajectory.npz",
        latents=traj, train_latents=ctx.train_latents,
        pca_mean=pca.mean, pca_components=pca.components,
        error=lg.error, door_id=did,
    )
    print(f"  arrays -> {ctx.out / 'belief_trajectory.npz'}")

    # Where did the belief end up, in physical terms? Report the nearest
    # training doors and their true parameters against the unseen door's.
    d = np.linalg.norm(ctx.train_latents - lg.latents[-1], axis=1)
    near = np.argsort(d)[:3]
    truth = ctx.ds.params_for_door(did)
    print(f"\n  converged belief is closest to these training doors:")
    print(f"    {'door':>6} {'dist':>6} {'I':>7} {'mu':>6} {'b':>6} {'k':>6}")
    for i in near:
        p = ctx.train_ds.params_for_door(int(i))
        print(f"    {int(i):>6} {d[i]:>6.2f} {p['I_hinge']:>7.1f} "
              f"{p['frictionloss']:>6.2f} {p['damping']:>6.2f} {p['stiffness']:>6.2f}")
    print(f"    {'TRUE':>6} {'--':>6} {truth['I_hinge']:>7.1f} "
          f"{truth['frictionloss']:>6.2f} {truth['damping']:>6.2f} "
          f"{truth['stiffness']:>6.2f}")
    return {"door_id": did, "video": str(video) if video else None,
            "nearest_train_doors": [int(i) for i in near]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/online_adaptation.yaml")
    ap.add_argument("--only", default=None,
                    help="run a subset: comma-separated from 1,2,3,viz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-animation", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else load_config(None)
    if args.checkpoint:
        cfg.experiments.checkpoint = args.checkpoint
    if args.out_dir:
        cfg.experiments.out_dir = args.out_dir
    if args.no_animation:
        cfg.experiments.make_animation = False

    which = {s.strip() for s in (args.only or "1,2,3,viz").split(",")}
    t0 = time.perf_counter()
    ctx = Context(cfg)
    cfg.save(ctx.out / "config.yaml")

    print(f"Stage-2 online adaptation")
    print(f"  checkpoint : {cfg.experiments.checkpoint}")
    print(f"  unseen doors: {ctx.door_ids}   focus: {ctx.focus_door}")
    print(f"  model dt   : {ctx.dt:.3f} s ({1 / ctx.dt:.0f} Hz)")
    print(f"  adaptor    : lr={cfg.adaptor.lr} {cfg.adaptor.optimizer} "
          f"window={cfg.adaptor.window} inner_steps={cfg.adaptor.n_inner_steps}")

    results: dict = {"config": cfg.to_dict()}
    if "1" in which:
        r = experiment_1(ctx)
        results["experiment_1"] = {k: v for k, v in r.items() if k != "logs"}
    if "2" in which:
        results["experiment_2"] = experiment_2(ctx)
    if "3" in which:
        results["experiment_3"] = experiment_3(ctx)
    if "viz" in which:
        results["visualisation"] = visualise_belief(ctx)

    (ctx.out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsummary -> {ctx.out / 'summary.json'}")
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
