"""Scaling curves and per-level latent geometry.

The scaling curves are Stage 5's primary result: diversity on x, generalisation /
adaptation gain / failure rate on y. The budget is fixed, so x is diversity only.

The latent-geometry panel reports four measures per level:

  log-inertia readout   leave-one-out R^2 for mechanical scale from z
  friction readout      the same, within family, so units cannot leak the answer
  geometry correlation  Spearman rank correlation of pairwise latent distance vs
                        pairwise mechanics distance -- is the geometry smooth
                        enough for an online optimiser to travel through
  effective dimension   participation ratio of the latent covariance
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_mechanics.curriculum.levels import CURRICULUM, CurriculumConfig
from latent_mechanics.mechanisms.analysis import (
    FAMILY_COLORS,
    embed_2d,
    family_separability,
    mechanics_readout,
)
from latent_mechanics.model import load_checkpoint

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


def geometry_correlation(z: np.ndarray, mech: np.ndarray) -> float:
    """Spearman correlation of pairwise latent distance vs mechanics distance.
    Mechanics are standardised first, so no wide-range parameter dominates."""
    from scipy.stats import spearmanr

    if len(z) < 6:
        return float("nan")
    m = mech.copy()
    for j in range(m.shape[1]):
        col = m[:, j]
        if np.all(col > 0):
            col = np.log10(col)
        s = col.std()
        m[:, j] = (col - col.mean()) / (s if s > 1e-12 else 1.0)
    iu = np.triu_indices(len(z), k=1)
    dz = np.linalg.norm(z[:, None] - z[None, :], axis=-1)[iu]
    dm = np.linalg.norm(m[:, None] - m[None, :], axis=-1)[iu]
    return float(spearmanr(dz, dm).correlation)


def effective_dimension(z: np.ndarray) -> float:
    """Participation ratio of the latent covariance spectrum."""
    c = z - z.mean(0)
    ev = np.linalg.svd(c, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / max(float((ev**2).sum()), 1e-30))


def analyse_level_latents(out: Path, level_index: int, name: str) -> dict | None:
    """Load one level's embedding table and measure its geometry."""
    ckpt = out / "runs" / f"L{level_index}_{name}" / "best.pt"
    npz = out / f"data_L{level_index}.npz"
    if not ckpt.exists() or not npz.exists():
        return None
    _, table, _, _ = load_checkpoint(ckpt, stage=f"stage5_curriculum:L{level_index}_{name}")
    z = table.weight.detach().cpu().numpy()
    with np.load(npz, allow_pickle=False) as a:
        fams = np.array([str(x) for x in a["mechanism_family"]])[: len(z)]
        gt = a["door_params"][: len(z)]
        cols = [str(c) for c in a["door_params_columns"]]

    idx = {c: cols.index(c) for c in ("inertia", "frictionloss", "damping", "stiffness")}
    mech = np.stack([gt[:, idx[c]] for c in idx], axis=1)
    inertia = gt[:, idx["inertia"]]

    fr = []
    for f in dict.fromkeys(fams):
        m = fams == f
        if m.sum() >= 8:
            r = mechanics_readout(z[m], gt[m, idx["frictionloss"]])
            if np.isfinite(r):
                fr.append(r)

    return {
        "level": level_index, "name": name, "n_latents": int(len(z)),
        "log_inertia_r2": mechanics_readout(z, np.log10(np.maximum(inertia, 1e-9))),
        "friction_r2_within": float(np.mean(fr)) if fr else float("nan"),
        "geometry_corr": geometry_correlation(z, mech),
        "effective_dim": effective_dimension(z),
        "separability": family_separability(z, fams) if len(set(fams)) > 1 else None,
        "_z": z, "_fams": fams, "_inertia": inertia,
    }


def scaling_curves(summaries: dict, geo: list[dict], path: Path,
                   rls_ref: float | None = None) -> Path:
    """THE figure: diversity on x, everything that matters on y."""
    lv = sorted(summaries)
    x = np.arange(len(lv))
    labels = [f"L{i}\n{len(summaries[i]['families'])}f" for i in lv]

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5))

    ax = axes[0]
    ax.plot(x, [summaries[i]["nrmse_before"] for i in lv], "s--", color="0.55",
            label="before adaptation", ms=5)
    ax.plot(x, [summaries[i]["nrmse_after"] for i in lv], "o-", color="#1f77b4",
            label="after adaptation", ms=5)
    if rls_ref and np.isfinite(rls_ref):
        ax.axhline(rls_ref, color="#d62728", ls=":", lw=1.2, label="RLS (fixed suite)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("curriculum level (families)")
    ax.set_ylabel("normalised error on fixed suite")
    ax.set_title("(a) generalisation")
    ax.legend()

    ax = axes[1]
    ax.plot(x, [summaries[i]["gain_median"] for i in lv], "o-", color="#2ca02c", ms=5)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("curriculum level (families)")
    ax.set_ylabel("adaptation gain (before / after)")
    ax.set_title("(b) adaptation gain")
    ax.annotate("adaptation helps", xy=(0.02, 1.03), xycoords=("axes fraction", "data"),
                fontsize=7, color="0.35")

    ax = axes[2]
    fr = [100 * summaries[i]["failure_rate"] for i in lv]
    ax.plot(x, fr, "o-", color="#d62728", ms=5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("curriculum level (families)")
    ax.set_ylabel("% instances made worse")
    ax.set_title("(c) harmful adaptations")

    ax = axes[3]
    gi = {g["level"]: g for g in geo if g}
    for key, lab, c in (("log_inertia_r2", "log-inertia $R^2$", "#1f77b4"),
                        ("friction_r2_within", "friction $R^2$ (within family)", "#2ca02c"),
                        ("geometry_corr", "latent-mechanics geometry $\\rho$", "#9467bd")):
        ax.plot(x, [gi[i][key] if i in gi else np.nan for i in lv], "o-", label=lab,
                color=c, ms=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(-0.5, 1.05)
    ax.set_xlabel("curriculum level (families)")
    ax.set_ylabel("structure score")
    ax.set_title("(d) latent organisation")
    ax.legend(fontsize=7)

    fig.suptitle("Mechanics prior scaling: fixed training budget, fixed evaluation suite",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    return _save(fig, path)


def per_family_heatmap(rows: list[dict], path: Path) -> Path:
    """Where diversity helps, and where it does not."""
    levels = sorted({r["level"] for r in rows})
    fams = list(dict.fromkeys(r["family"] for r in rows))
    gain = np.full((len(fams), len(levels)), np.nan)
    fail = np.full((len(fams), len(levels)), np.nan)
    for i, f in enumerate(fams):
        for j, l in enumerate(levels):
            sub = [r for r in rows if r["family"] == f and r["level"] == l]
            if sub:
                gain[i, j] = np.nanmedian([r["gain"] for r in sub])
                fail[i, j] = np.mean([r["failed"] for r in sub])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, data, title, cmap, vlim in (
        (axes[0], np.log10(np.clip(gain, 1e-3, None)), "log10 adaptation gain",
         "RdYlGn", (-0.7, 0.7)),
        (axes[1], 100 * fail, "% harmful adaptations", "Reds", (0, 100)),
    ):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
        for i in range(len(fams)):
            for j in range(len(levels)):
                v = gain[i, j] if ax is axes[0] else 100 * fail[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}" if ax is axes[0] else f"{v:.0f}",
                            ha="center", va="center", fontsize=6.5)
    fig.suptitle("Per-family outcome across the curriculum "
                 "(green / low = adaptation helps)", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, path)


def latent_evolution(geo: list[dict], path: Path) -> Path:
    """PCA and UMAP of every level's latent table, side by side."""
    geo = [g for g in geo if g is not None]
    if not geo:
        return path
    fig, axes = plt.subplots(2, len(geo), figsize=(2.7 * len(geo), 5.6), squeeze=False)
    for col, g in enumerate(geo):
        z, fams = g["_z"], g["_fams"]
        for row, method in enumerate(("pca", "umap")):
            ax = axes[row][col]
            if len(z) < 5:
                ax.axis("off"); continue
            xy, lab = embed_2d(z, method)
            for f in dict.fromkeys(fams):
                m = fams == f
                ax.scatter(xy[m, 0], xy[m, 1], s=16, alpha=0.85, edgecolor="none",
                           color=FAMILY_COLORS.get(f, "0.5"), label=f if col == 0 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"L{g['level']}  ({len(set(fams))} fam)\n"
                             f"$\\rho$={g['geometry_corr']:.2f}  "
                             f"$d_{{eff}}$={g['effective_dim']:.1f}", fontsize=8)
            if col == 0:
                ax.set_ylabel(lab.split(" ")[0], fontsize=9)
    handles, labels = [], []
    for g in geo:
        for f in dict.fromkeys(g["_fams"]):
            if f not in labels:
                labels.append(f)
                handles.append(plt.Line2D([], [], marker="o", ls="", ms=5,
                                          color=FAMILY_COLORS.get(f, "0.5")))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.suptitle("Latent geometry as mechanical diversity increases", fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def run_all(out: Path, summaries: dict, rows: list[dict], cc: CurriculumConfig) -> dict:
    print(f"\n{'=' * 78}\nREPRESENTATION ANALYSIS\n{'=' * 78}")
    names = {lv.index: lv.name for lv in CURRICULUM}
    geo = [analyse_level_latents(out, i, names[i]) for i in sorted(summaries)]
    geo = [g for g in geo if g]

    print(f"  {'level':>5} {'n_z':>4} {'logI R2':>9} {'fric R2':>9} "
          f"{'geom rho':>9} {'d_eff':>7} {'separab':>9}")
    for g in geo:
        sep = g["separability"]["accuracy"] if g["separability"] else float("nan")
        print(f"  {'L'+str(g['level']):>5} {g['n_latents']:>4} "
              f"{g['log_inertia_r2']:>9.3f} {g['friction_r2_within']:>9.3f} "
              f"{g['geometry_corr']:>9.3f} {g['effective_dim']:>7.2f} {sep:>9.2f}")

    rls = None
    sj = out / "summary.json"
    if sj.exists():
        rls = json.loads(sj.read_text()).get("rls_reference_median")

    scaling_curves(summaries, geo, out / "scaling_curves.png", rls)
    per_family_heatmap(rows, out / "per_family_heatmap.png")
    latent_evolution(geo, out / "latent_evolution.png")

    clean = [{k: v for k, v in g.items() if not k.startswith("_")} for g in geo]
    (out / "representation_analysis.json").write_text(json.dumps(clean, indent=2, default=str))
    print(f"  analysis -> {out / 'representation_analysis.json'}")
    return {"geometry": clean}


__all__ = ["run_all", "scaling_curves", "per_family_heatmap", "latent_evolution",
           "analyse_level_latents", "geometry_correlation", "effective_dimension"]
