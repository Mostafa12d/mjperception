"""Figures and tables for the Stage-3 robustness study.

Conventions held across every panel: one fixed colour per method, log error axes,
a dotted ideal-plant reference, and shaded inter-quartile bands across doors so
the spread is visible rather than averaged away.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_mechanics.mismatch.config import EXPERIMENT_TITLES, Sweep

# One colour and style per method, everywhere.
STYLE = {
    "no-adaptation": {"color": "0.55", "marker": "s", "ls": "--", "label": "no adaptation"},
    "latent-gd": {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "latent adaptation"},
    "rls-5p": {"color": "#d62728", "marker": "^", "ls": "-", "label": "RLS (spring-aware)"},
    "rls-3p": {"color": "#ff9896", "marker": "v", "ls": ":", "label": "RLS (baseline regressor)"},
}
PRIMARY = ("no-adaptation", "latent-gd", "rls-5p")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


def _level_key(v):
    return v if v is not None else 0


def _by_level(rows, sweep: Sweep, method: str, field: str):
    """Median and IQR of ``field`` across doors, per severity level."""
    med, lo, hi = [], [], []
    for lv in sweep.levels:
        vals = [r[field] for r in rows
                if r["method"] == method and r["level"] == _level_key(lv)]
        if not vals:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(float(np.median(vals)))
        lo.append(float(np.percentile(vals, 25)))
        hi.append(float(np.percentile(vals, 75)))
    return np.array(med), np.array(lo), np.array(hi)


def _x_axis(sweep: Sweep):
    """Positions and tick labels. Levels are evenly spaced, since the severity grid
    is roughly geometric; the real values stay on the tick labels."""
    xs = np.arange(len(sweep.levels))
    labels = ["off" if v in (None, 0, 0.0) else f"{v:g}" for v in sweep.levels]
    return xs, labels


def sweep_figure(
    sweep: Sweep, rows: list[dict], curves: dict, out_dir: Path,
    reference: float, reference_raw: float = float('nan'),
) -> Path:
    """Three panels: error vs severity, learning curves, belief stability."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    xs, labels = _x_axis(sweep)

    # -- (a) error vs severity ------------------------------------------
    ax = axes[0]
    for m in STYLE:
        med, lo, hi = _by_level(rows, sweep, m, "holdout_nrmse")
        s = STYLE[m]
        ax.plot(xs, med, color=s["color"], marker=s["marker"], ls=s["ls"],
                label=s["label"], ms=4, lw=1.5)
        ax.fill_between(xs, lo, hi, color=s["color"], alpha=0.15, lw=0)
    if np.isfinite(reference):
        ax.axhline(reference, color="k", ls=":", lw=1, label="Stage-1 ideal plant")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel(sweep.axis_label())
    ax.set_ylabel("normalised angle error (clean holdout)")
    ax.set_yscale("log")
    ax.set_title("(a) quality of the learned belief")
    ax.legend(loc="best", framealpha=0.9)

    # -- (b) learning curves at the worst severity -----------------------
    ax = axes[1]
    worst = str(sweep.levels[-1])
    for m in PRIMARY:
        c = curves.get(worst, {}).get(m, np.array([]))
        if len(c):
            s = STYLE[m]
            ax.plot(c, color=s["color"], ls=s["ls"], lw=1.4, label=s["label"])
    if np.isfinite(reference_raw):
        ax.axhline(reference_raw, color="k", ls=":", lw=1)
    ax.set_xlabel("interaction number")
    ax.set_ylabel("rolling angle RMSE [rad]")
    ax.set_yscale("log")
    ax.set_title(f"(b) convergence at {sweep.param}={sweep.levels[-1]}")
    ax.legend(loc="best", framealpha=0.9)

    # -- (c) belief stability -------------------------------------------
    ax = axes[2]
    for m in ("latent-gd", "rls-5p", "rls-3p"):
        med, lo, hi = _by_level(rows, sweep, m, "belief_drift_tail")
        s = STYLE[m]
        ax.plot(xs, med, color=s["color"], marker=s["marker"], ls=s["ls"],
                label=s["label"], ms=4, lw=1.5)
        ax.fill_between(xs, lo, hi, color=s["color"], alpha=0.15, lw=0)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel(sweep.axis_label())
    ax.set_ylabel("relative belief motion (tail)")
    ax.set_yscale("log")
    ax.set_title("(c) belief stability")
    ax.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        f"Experiment {sweep.experiment} - {EXPERIMENT_TITLES[sweep.experiment]}: "
        f"{sweep.name.replace('_', ' ')}",
        fontsize=11, y=1.04,
    )
    return _save(fig, out_dir / f"sweep_{sweep.name}.png")


def overview_figure(
    rows: list[dict], sweeps: list[Sweep], out_dir: Path, reference: float
) -> Path:
    """One panel per sweep: the whole study on a page."""
    n = len(sweeps)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows),
                             squeeze=False)

    for k, sw in enumerate(sweeps):
        ax = axes[k // ncols][k % ncols]
        sub = [r for r in rows if r["sweep"] == sw.name]
        xs, labels = _x_axis(sw)
        for m in PRIMARY:
            med, lo, hi = _by_level(sub, sw, m, "holdout_nrmse")
            s = STYLE[m]
            ax.plot(xs, med, color=s["color"], marker=s["marker"], ls=s["ls"],
                    ms=4, lw=1.5, label=s["label"] if k == 0 else None)
            ax.fill_between(xs, lo, hi, color=s["color"], alpha=0.15, lw=0)
        if np.isfinite(reference):
            ax.axhline(reference, color="k", ls=":", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yscale("log")
        ax.set_title(f"E{sw.experiment}: {sw.name.replace('_', ' ')}", fontsize=9)
        ax.set_xlabel(sw.axis_label(), fontsize=7)
        if k % ncols == 0:
            ax.set_ylabel("normalised angle error")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Robustness to model mismatch: unseen doors, methods unchanged",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "overview.png")


def print_summary(summary: list[dict], sweeps: list[Sweep]) -> None:
    """Console version of the cross-experiment table."""
    hdr = (f"{'exp':>3} {'sweep':<19} {'method':<15} {'clean':>10} {'worst':>10} "
           f"{'degrade':>8} {'belief drift':>13}")
    print(hdr)
    print("-" * len(hdr))
    last_sweep = None
    for sw in sweeps:
        for r in [x for x in summary if x["sweep"] == sw.name]:
            if last_sweep and r["sweep"] != last_sweep:
                print()
            last_sweep = r["sweep"]
            print(f"{r['experiment']:>3} {r['sweep']:<19} {r['method']:<15} "
                  f"{r['nrmse_clean']:>10.3e} {r['nrmse_worst']:>10.3e} "
                  f"{r['degradation_x']:>7.1f}x {r['belief_drift_worst']:>13.2e}")
    print()
    cross = [r for r in summary if r["method"] == "latent-gd" and r["latent_beats_rls_from"]]
    if cross:
        print("Latent adaptation overtakes RLS-5p at:")
        for r in cross:
            print(f"  {r['sweep']:<20} from {r['sweep']} = {r['latent_beats_rls_from']}")
    else:
        print("RLS-5p retains the lead at every severity level tested.")


def summary_latex(summary: list[dict], sweeps: list[Sweep], path: Path) -> Path:
    """LaTeX booktabs table, ready to paste into a paper."""
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Robustness to model mismatch. Angle RMSE (rad) of one-step "
        r"prediction on unseen doors, scored against ground truth. Neither method "
        r"is re-fitted for the perturbed plants: the learned dynamics network is "
        r"the Stage-1 model and RLS keeps its original regressor. "
        r"\emph{Degrade} is the ratio of worst-case to clean error.}",
        r"\label{tab:mismatch}",
        r"\begin{tabular}{lllrrr}", r"\toprule",
        r"Experiment & Mismatch & Method & Clean & Worst & Degrade \\",
        r"\midrule",
    ]
    fmt = lambda v: f"{v:.2e}".replace("e-0", r"e{-}").replace("e+0", r"e{+}")
    for sw in sweeps:
        rows = [r for r in summary if r["sweep"] == sw.name
                and r["method"] in PRIMARY]
        for i, r in enumerate(rows):
            exp = EXPERIMENT_TITLES[sw.experiment] if i == 0 else ""
            name = sw.name.replace("_", r"\_") if i == 0 else ""
            lines.append(
                f"{exp} & {name} & {STYLE[r['method']]['label']} & "
                f"${fmt(r['nrmse_clean'])}$ & ${fmt(r['nrmse_worst'])}$ & "
                f"${r['degradation_x']:.1f}\\times$ \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  latex -> {path}")
    return path


def belief_figure(
    sweep: Sweep, latents_by_level: dict, pca, out_dir: Path, train_color=None
) -> Path:
    """Belief trajectories in the fixed Stage-1 PCA frame, one panel per severity."""
    levels = list(latents_by_level)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.0 * len(levels), 3.2),
                             squeeze=False, sharex=True, sharey=True)
    for ax, lv in zip(axes[0], levels):
        traj = latents_by_level[lv]
        if traj is None or len(traj) == 0:
            ax.axis("off"); continue
        xy = pca.project(traj)
        c = "0.8" if train_color is None else train_color
        ax.scatter(pca.train_xy[:, 0], pca.train_xy[:, 1], c=c, cmap="viridis",
                   s=26, alpha=0.75, edgecolor="none")
        ax.plot(xy[:, 0], xy[:, 1], "-", color="crimson", lw=1.1)
        ax.plot(xy[0, 0], xy[0, 1], "o", color="k", ms=7, mfc="white", mew=1.5)
        ax.plot(xy[-1, 0], xy[-1, 1], "*", color="crimson", ms=15, mec="k", mew=0.5)
        ax.set_title(f"{sweep.param} = {lv}", fontsize=9)
        ax.set_xlabel(pca.label(0), fontsize=8)
    axes[0][0].set_ylabel(pca.label(1), fontsize=8)
    fig.suptitle(f"Latent belief under increasing {sweep.name.replace('_', ' ')}",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    return _save(fig, out_dir / f"belief_{sweep.name}.png")


__all__ = ["sweep_figure", "overview_figure", "print_summary", "summary_latex",
           "belief_figure", "STYLE"]
