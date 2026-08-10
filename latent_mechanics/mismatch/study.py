"""
Stage 3: robustness to model mismatch.

Runs eight single-mechanism severity sweeps grouped into four experiments, and
for every (sweep, level, door) evaluates the same four estimators on byte-identical
input:

    no-adaptation   frozen latent, never updated -- the control
    latent-gd       Stage-2 online latent adaptation, unchanged
    rls-5p          RLS with the spring-aware regressor -- the strong baseline
    rls-3p          RLS with the Stage-1 baseline's own regressor

Nothing in Stages 1 or 2 is modified. The learned model is *not* retrained on the
perturbed plants, and RLS keeps its regressor, which is the point: both methods
hold their original assumptions while the world stops obeying them. Absolute
error therefore rises for everyone, and the question is entirely about *relative*
degradation and where the ranking changes.

Run:
    python3.10 -m latent_mechanics.mismatch.study
    python3.10 -m latent_mechanics.mismatch.study --only stribeck,drift
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_mechanics.config import load_config as load_stage1_config
from latent_mechanics.mismatch import figures
from latent_mechanics.mismatch.config import (
    EXPERIMENT_TITLES,
    StudyConfig,
    Sweep,
    default_sweeps,
    load_config,
)
from latent_mechanics.mismatch.perturbations import build_perturbation
from latent_mechanics.mismatch.sensors import SensorPipeline
from latent_mechanics.mismatch.streams import (
    DoorStream,
    build_door_stream,
    clean_errors,
    frozen_predict_errors,
    heldout_doors,
)
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.adaptor import GradientLatentAdaptor, StaticLatentAdaptor
from latent_mechanics.online.config import load_config as load_online_config
from latent_mechanics.online.loop import init_strategies, run_online_adaptation
from latent_mechanics.online.rls_adaptor import RLSAdaptor

METHODS = ("no-adaptation", "latent-gd", "rls-5p", "rls-3p")
PRIMARY_METHODS = ("no-adaptation", "latent-gd", "rls-5p")


class Runner:
    """Holds the frozen model and produces estimators on demand."""

    def __init__(self, cfg: StudyConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        model, table, stage1_cfg, extra = load_checkpoint(
            cfg.checkpoint, device=self.device, stage="mismatch_study")
        if table is None:
            raise ValueError("checkpoint has no embedding table")
        self.model = model.freeze()
        self.train_latents = table.weight.detach().cpu().numpy()
        self.init = init_strategies(self.train_latents, cfg.seed)[cfg.latent_init]
        self.reference = {
            "angle": float(extra.get("val_rmse_angle", np.nan)),
            "velocity": float(extra.get("val_rmse_velocity", np.nan)),
        }
        # The reference drawn on every figure must be in the SAME units as the
        # plotted metric. Stage 1 reports a raw RMSE, so convert it to the
        # normalised scale using the validation split's own motion magnitude.
        try:
            from latent_mechanics.dataset import DoorTransitionDataset
            _val = DoorTransitionDataset(
                stage1_cfg.sim.out_path, "val",
                exclude_near_limit=stage1_cfg.sim.exclude_near_limit)
            _d = (_val.next_state - _val.state).numpy()
            _scale = float(np.sqrt(np.mean(_d[:, 0] ** 2)))
            self.reference["angle_normalised"] = self.reference["angle"] / _scale
        except Exception:
            self.reference["angle_normalised"] = float("nan")

        self.stage1_cfg = load_stage1_config(cfg.stage1_config)
        self.online_cfg = load_online_config(cfg.online_config)
        self.dt = stage1_cfg.sim.frame_skip * 0.002
        self.doors = heldout_doors(self.stage1_cfg, cfg.n_doors)

    def make(self, name: str):
        a = self.online_cfg.adaptor
        r = self.online_cfg.rls
        if name == "no-adaptation":
            return StaticLatentAdaptor(self.model, init=self.init, device=self.device)
        if name == "latent-gd":
            return GradientLatentAdaptor(
                self.model, init=self.init, lr=a.lr, optimizer=a.optimizer,
                n_inner_steps=a.n_inner_steps, window=a.window,
                prior_weight=a.prior_weight, loss_space=a.loss_space,
                max_grad_norm=a.max_grad_norm, lr_decay=a.lr_decay, device=self.device,
            )
        if name.startswith("rls-"):
            n_params = int(name.split("-")[1].rstrip("p"))
            return RLSAdaptor(dt=self.dt, n_substeps=r.n_substeps, n_params=n_params,
                              lam=r.lam, delta=r.delta, vel_thresh=r.vel_thresh)
        raise ValueError(f"unknown method {name!r}")


def build_level(sweep: Sweep, value) -> tuple[list, SensorPipeline]:
    """Turn one severity level into (plant perturbations, sensor pipeline)."""
    if sweep.kind == "plant":
        disabled = value in (0.0, 0, None)
        if disabled:
            return [], SensorPipeline()
        return [build_perturbation(sweep.target, **{sweep.param: value}, **sweep.fixed)], SensorPipeline()
    if sweep.kind == "sensor":
        return [], SensorPipeline(**{sweep.param: value}, **sweep.fixed)
    raise ValueError(f"unknown sweep kind {sweep.kind!r}")


def run_level(
    runner: Runner, sweep: Sweep, value, streams: list[DoorStream],
    holdouts: list[DoorStream] | None = None,
) -> tuple[list[dict], dict[str, list]]:
    """Every method on every door at one severity level."""
    rows: list[dict] = []
    logs_by_method: dict[str, list] = {m: [] for m in METHODS}

    holdouts = holdouts or [None] * len(streams)
    for stream, holdout in zip(streams, holdouts):
        if len(stream) == 0:
            continue
        for name in METHODS:
            adaptor = runner.make(name)
            log = run_online_adaptation(
                adaptor, stream.transitions, door_id=stream.door_id,
                boundaries=stream.boundaries, verify_frozen=False,
            )
            log.name = name
            # Score against ground truth, not the (possibly corrupted) reading.
            err = clean_errors(log, stream)
            n = len(err)
            tail = max(1, n // 4)
            # Normalise by how much the door ACTUALLY moved, otherwise a
            # perturbation that slows it down looks like an improvement.
            scale = stream.motion_scale()
            tail_scale = np.sqrt(np.mean(
                (stream.clean_next[-tail:] - stream.clean_state[-tail:]) ** 2, axis=0))
            tail_scale = np.maximum(tail_scale, 1e-12)
            rows.append({
                "sweep": sweep.name, "experiment": sweep.experiment,
                "level": value if value is not None else 0,
                "door_id": stream.door_id, "method": name,
                "angle_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
                "angle_rmse_final": float(np.sqrt(np.mean(err[-tail:, 0] ** 2))),
                "vel_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
                "vel_rmse_final": float(np.sqrt(np.mean(err[-tail:, 1] ** 2))),
                "angle_nrmse_final": float(np.sqrt(np.mean(err[-tail:, 0] ** 2)) / tail_scale[0]),
                "vel_nrmse_final": float(np.sqrt(np.mean(err[-tail:, 1] ** 2)) / tail_scale[1]),
                "motion_scale_angle": float(scale[0]),
                "us_per_update": 1e6 * log.seconds_per_update,
                "n_steps": n,
                "perturb_torque_rms": stream.perturb_torque_rms,
                # Belief stability: how much the estimate still moves once it
                # should have settled. High values mean it is chattering, not
                # converging.
                "belief_drift_tail": _belief_drift(log, tail),
                "belief_travel": float(np.linalg.norm(log.latents[-1] - log.latents[0])),
                **_holdout_metrics(adaptor, holdout),
            })
            log.clean_error = err
            logs_by_method[name].append(log)
    return rows, logs_by_method


def _holdout_metrics(adaptor, holdout) -> dict:
    """Score the belief the adaptor finished with on a clean held-out stream."""
    if holdout is None or len(holdout) == 0:
        return {"holdout_nrmse": float("nan"), "holdout_rmse": float("nan")}
    err = frozen_predict_errors(adaptor, holdout)
    scale = np.maximum(holdout.motion_scale(), 1e-12)
    return {
        "holdout_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "holdout_nrmse": float(np.sqrt(np.mean(err[:, 0] ** 2)) / scale[0]),
    }


def _belief_drift(log, tail: int) -> float:
    """Mean per-step movement of the belief over the final quarter, normalised.

    For the latent this is ``||z_t - z_{t-1}||``; for RLS the same on the
    parameter vector, divided by the parameter scale so the two are comparable
    as a *relative* jitter measure.
    """
    z = log.latents[-tail:]
    if len(z) < 2:
        return 0.0
    step = np.linalg.norm(np.diff(z, axis=0), axis=1).mean()
    scale = max(float(np.linalg.norm(z[-1])), 1e-9)
    return float(step / scale)


def run_sweep(runner: Runner, sweep: Sweep) -> tuple[list[dict], dict]:
    cfg = runner.cfg
    print(f"\n--- {sweep.name}  ({EXPERIMENT_TITLES[sweep.experiment]}) ---")
    all_rows: list[dict] = []
    curves: dict = {}
    beliefs: dict = {}

    for value in sweep.levels:
        perts, sensors = build_level(sweep, value)
        t0 = time.perf_counter()
        streams = [
            build_door_stream(
                p, runner.stage1_cfg, cfg.episodes_per_door, cfg.episode_seconds,
                cfg.frame_skip, perturbations=perts, sensors=sensors, seed=cfg.seed,
            )
            for p in runner.doors
        ]
        # Clean held-out episodes on the SAME plant: same perturbation, no
        # sensor corruption, disjoint excitation. Used to score the final belief.
        holdouts = [
            build_door_stream(
                p, runner.stage1_cfg, 2, cfg.episode_seconds, cfg.frame_skip,
                perturbations=perts, sensors=None, seed=cfg.seed,
                episode_offset=cfg.episodes_per_door,
            )
            for p in runner.doors
        ]
        rows, logs = run_level(runner, sweep, value, streams, holdouts)
        all_rows += rows
        if logs["latent-gd"]:
            beliefs[str(value)] = logs["latent-gd"][0].latents
        curves[str(value)] = {
            m: _mean_curve([lg.clean_error for lg in logs[m]], cfg.rolling_window)
            for m in METHODS
        }

        summary = {m: np.nanmean([r["holdout_nrmse"] for r in rows if r["method"] == m])
                   for m in METHODS}
        tau_rms = np.mean([s.perturb_torque_rms for s in streams])
        print(f"  {sweep.param}={str(value):>7}  "
              f"(unmodelled tau RMS {tau_rms:5.2f} N*m)  "
              + "  ".join(f"{m}={summary[m]:.2e}" for m in PRIMARY_METHODS)
              + f"   [{time.perf_counter() - t0:.0f}s]")

    return all_rows, curves, beliefs


def _mean_curve(errors: list[np.ndarray], window: int) -> np.ndarray:
    """Rolling angle RMSE averaged across doors, truncated to the shortest run."""
    if not errors:
        return np.array([])
    n = min(len(e) for e in errors)
    sq = np.stack([e[:n, 0] ** 2 for e in errors]).mean(axis=0)
    c = np.concatenate([[0.0], np.cumsum(sq)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    return np.sqrt((c[idx + 1] - c[lo]) / (idx + 1 - lo))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k: None for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  table -> {path}")


def summary_table(rows: list[dict], sweeps: list[Sweep], reference: float) -> list[dict]:
    """One row per (sweep, method): clean baseline, worst severity, degradation.

    ``crossover`` is the headline column -- the first severity level at which the
    latent adaptor's error drops below RLS-5p's, or blank if RLS keeps the lead
    throughout. That is the quantity the whole stage exists to locate.
    """
    out = []
    for sw in sweeps:
        sub = [r for r in rows if r["sweep"] == sw.name]
        if not sub:
            continue
        levels = sw.levels
        base_lvl = levels[0] if levels[0] is not None else 0
        worst_lvl = levels[-1] if levels[-1] is not None else 0

        # Locate the crossover once per sweep.
        crossover = ""
        for lv in levels:
            key = lv if lv is not None else 0
            lat = np.nanmean([r["holdout_nrmse"] for r in sub
                           if r["level"] == key and r["method"] == "latent-gd"] or [np.nan])
            rls = np.nanmean([r["holdout_nrmse"] for r in sub
                           if r["level"] == key and r["method"] == "rls-5p"] or [np.nan])
            if np.isfinite(lat) and np.isfinite(rls) and lat < rls:
                crossover = str(lv)
                break

        for m in METHODS:
            base = [r["holdout_nrmse"] for r in sub
                    if r["level"] == base_lvl and r["method"] == m]
            worst = [r["holdout_nrmse"] for r in sub
                     if r["level"] == worst_lvl and r["method"] == m]
            if not base or not worst:
                continue
            b, w = float(np.nanmean(base)), float(np.nanmean(worst))
            drift = [r["belief_drift_tail"] for r in sub
                     if r["level"] == worst_lvl and r["method"] == m]
            out.append({
                "experiment": sw.experiment,
                "experiment_name": EXPERIMENT_TITLES[sw.experiment],
                "sweep": sw.name, "method": m,
                "nrmse_clean": b, "nrmse_worst": w,
                "online_nrmse_worst": float(np.nanmean(
                    [r["angle_nrmse_final"] for r in sub
                     if r["level"] == worst_lvl and r["method"] == m] or [np.nan])),
                "degradation_x": w / b if b > 0 else np.nan,
                "belief_drift_worst": float(np.mean(drift)) if drift else np.nan,
                "latent_beats_rls_from": crossover if m == "latent-gd" else "",
                "vs_stage1_reference_x": w / reference if reference > 0 else np.nan,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mismatch.yaml")
    ap.add_argument("--only", default=None, help="comma-separated sweep names")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--doors", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.doors:
        cfg.n_doors = args.doors
    if args.episodes:
        cfg.episodes_per_door = args.episodes
    if args.only:
        cfg.only = tuple(s.strip() for s in args.only.split(","))

    sweeps = [s for s in default_sweeps() if not cfg.only or s.name in cfg.only]
    if not sweeps:
        raise SystemExit(f"no sweeps matched {cfg.only}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.yaml")

    t0 = time.perf_counter()
    runner = Runner(cfg)
    print("Stage 3: robustness to model mismatch")
    print(f"  checkpoint  : {cfg.checkpoint}  (NOT retrained on perturbed plants)")
    print(f"  doors       : {[d.door_id for d in runner.doors]}")
    print(f"  per door    : {cfg.episodes_per_door} episodes x {cfg.episode_seconds}s")
    print(f"  latent init : {cfg.latent_init}")
    print(f"  sweeps      : {[s.name for s in sweeps]}")
    print(f"  stage-1 reference (ideal plant, trained door): "
          f"{runner.reference['angle']:.3e} rad "
          f"= {runner.reference['angle_normalised']:.4f} normalised")

    # Fixed Stage-1 PCA frame, so belief plots here are directly comparable to
    # the Stage-1 latent-space figure and the Stage-2 trajectory.
    from latent_mechanics.dataset import DoorTransitionDataset
    from latent_mechanics.online.viz import LatentPCA
    pca = LatentPCA.fit(runner.train_latents)
    try:
        _tr = DoorTransitionDataset(runner.stage1_cfg.sim.out_path, "train",
                                    exclude_near_limit=True)
        train_I = np.array([_tr.params_for_door(i)["I_hinge"]
                            for i in range(len(runner.train_latents))])
    except Exception:
        train_I = None

    all_rows: list[dict] = []
    all_curves: dict = {}
    all_beliefs: dict = {}
    for sw in sweeps:
        rows, curves, beliefs = run_sweep(runner, sw)
        all_rows += rows
        all_curves[sw.name] = curves
        all_beliefs[sw.name] = beliefs
        if not args.no_figures:
            figures.belief_figure(sw, beliefs, pca, out, train_I)
            figures.sweep_figure(sw, rows, curves, out,
                                 runner.reference["angle_normalised"],
                                 runner.reference["angle"])

    write_csv(out / "raw_results.csv", all_rows)
    summary = summary_table(all_rows, sweeps, runner.reference["angle_normalised"])
    write_csv(out / "summary_table.csv", summary)

    print("\n" + "=" * 100)
    print("SUMMARY: angle RMSE [rad] on unseen doors, clean plant vs worst mismatch")
    print("=" * 100)
    figures.print_summary(summary, sweeps)

    if not args.no_figures:
        figures.overview_figure(all_rows, sweeps, out, runner.reference["angle_normalised"])
        figures.summary_latex(summary, sweeps, out / "summary_table.tex")

    (out / "curves.npz")
    np.savez_compressed(
        out / "curves.npz",
        **{f"{s}|{lv}|{m}": c
           for s, lvls in all_curves.items()
           for lv, mc in lvls.items()
           for m, c in mc.items() if len(c)},
    )
    (out / "summary.json").write_text(json.dumps(
        {"config": cfg.to_dict(), "reference": runner.reference,
         "summary": summary}, indent=2, default=str))
    print(f"\nartefacts -> {out}")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
