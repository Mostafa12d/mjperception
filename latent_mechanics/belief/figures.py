"""The UKF headline figure, from the sweep.py and drift_check.py CSVs.

(a) does it beat what it replaces, (b) everywhere or only on average, (c) does it
hold up when the mechanics change.

    python3.10 -m latent_mechanics.belief.figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("runs/latent_mechanics/belief")

# same palette as the mismatch/mechanisms figures; the defective arm is kept on purpose
STYLE = {
    "baseline:no-adaptation": ("no adaptation", "0.55"),
    "baseline:gradient-descent": ("gradient descent", "#ff7f0e"),
    "ukf:d=6:R=fixed": ("UKF, fixed R", "#9467bd"),
    "ukf:d=6:R=adaptive": ("UKF, innovation R\n(defective)", "#d62728"),
    "ukf:d=6:R=residual": ("UKF, residual R\n(fixed)", "#1f77b4"),
}
FAMILY_ORDER = ["door", "nonlinear_hinge", "soft_close", "drawer", "bifold", "laptop"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _median(vals) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def make_figure(out: Path = RESULTS / "ukf_headline.png") -> Path:
    sweep = _rows(RESULTS / "sweep_results.csv")
    drift = _rows(RESULTS / "drift_check.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # -- (a) headline --------------------------------------------------
    ax = axes[0]
    keys = list(STYLE)
    vals = [_median([float(r["ratio_to_ceiling"]) for r in sweep if r["config"] == k])
            for k in keys]
    labels = [STYLE[k][0] for k in keys]
    colors = [STYLE[k][1] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=colors, width=0.62,
                  edgecolor="k", linewidth=0.5)
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.text(0.02, 1.03, "oracle ceiling", transform=ax.get_yaxis_transform(),
            fontsize=8, color="0.3")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}x",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("error / per-family oracle ceiling")
    ax.set_title("(a) 60 unseen objects, 6 families")
    ax.set_ylim(0, max(vals) * 1.25)

    # -- (b) per family -------------------------------------------------
    ax = axes[1]
    show = ["baseline:no-adaptation", "baseline:gradient-descent", "ukf:d=6:R=adaptive"]
    per = defaultdict(dict)
    for k in show:
        for fam in FAMILY_ORDER:
            per[k][fam] = _median([float(r["ratio_to_ceiling"]) for r in sweep
                                   if r["config"] == k and r["family"] == fam])
    x = np.arange(len(FAMILY_ORDER))
    w = 0.26
    for i, k in enumerate(show):
        ax.bar(x + (i - 1) * w, [per[k][f] for f in FAMILY_ORDER], w,
               label=STYLE[k][0], color=STYLE[k][1], edgecolor="k", linewidth=0.4)
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in FAMILY_ORDER], fontsize=8)
    ax.set_ylabel("error / oracle ceiling  (log)")
    ax.set_title("(b) the UKF beats no-adaptation on every family")
    ax.legend(fontsize=8)

    # -- (c) drift ------------------------------------------------------
    ax = axes[2]
    rates = sorted({float(r["drift_rate"]) for r in drift})
    base = {rt: _median([float(r["nrmse_final"]) for r in drift
                         if r["method"] == "no-adaptation" and float(r["drift_rate"]) == rt])
            for rt in rates}
    series = [("gradient-descent", "gradient descent", "#ff7f0e", "--"),
              ("ukf:w=20", "UKF w=20", "#aec7e8", "-"),
              ("ukf:w=50", "UKF w=50", "#6baed6", "-"),
              ("ukf:w=100", "UKF w=100 (chosen)", "#1f77b4", "-")]
    for key, lab, c, ls in series:
        ys = []
        for rt in rates:
            v = _median([float(r["nrmse_final"]) for r in drift
                         if r["method"] == key and float(r["drift_rate"]) == rt])
            ys.append(v / base[rt] if base[rt] > 0 else np.nan)
        ax.plot(rates, ys, ls, marker="o", ms=5, color=c, label=lab,
                lw=2.2 if "100" in key else 1.5)
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.text(rates[0], 1.02, "worse than not adapting", fontsize=8, color="0.3")
    ax.set_xlabel("friction drift rate [1/s]")
    ax.set_ylabel("error / no-adaptation control")
    ax.set_title("(c) under time-varying mechanics")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "UKF with adaptive measurement noise replaces gradient descent as the belief update",
        fontsize=13, y=1.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS / "ukf_headline.png"))
    make_figure(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
