"""
What structure does the latent space learn?

The paper hypothesis is not "the embedding represents doors" but "the embedding
represents interaction mechanics". Those two make different predictions about
the geometry of the latent space:

  category hypothesis   embeddings cluster by mechanism FAMILY, and a drawer
                        sits far from every door regardless of how it behaves.
  mechanics hypothesis  embeddings organise by mechanical BEHAVIOUR -- inertia,
                        friction, stiffness -- and a heavy stiff drawer sits
                        near a heavy stiff door.

They are distinguishable and this module measures which one holds, rather than
inviting a reader to squint at a scatter plot. Two quantitative tests:

  *family separability* -- how well a classifier recovers the family label from
    the latent alone, via leave-one-out nearest neighbour. High means the space
    is organised by category.
  *mechanics readout* -- leave-one-out ridge R^2 predicting each physical
    parameter from the latent, computed WITHIN a family so it cannot be
    inflated by between-family scale differences.

Both are reported, because the honest answer can be "both": a space can encode
category and behaviour on different axes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_mechanics.mechanisms import library as lib
from latent_mechanics.model import load_checkpoint

FAMILY_COLORS = {
    "door": "#1f77b4", "nonlinear_hinge": "#17becf", "soft_close": "#2ca02c",
    "drawer": "#d62728", "laptop": "#ff7f0e", "bifold": "#9467bd",
}

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


# ---------------------------------------------------------------------------
# Quantitative structure tests
# ---------------------------------------------------------------------------

def family_separability(z: np.ndarray, families: np.ndarray) -> dict:
    """Leave-one-out 1-NN accuracy of family from the latent alone.

    Compared against the majority-class rate, so a suite with unbalanced
    families cannot look separable by accident.
    """
    n = len(z)
    if n < 3:
        return {"accuracy": float("nan"), "chance": float("nan")}
    d = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    pred = families[np.argmin(d, axis=1)]
    _, counts = np.unique(families, return_counts=True)
    return {
        "accuracy": float(np.mean(pred == families)),
        "chance": float(counts.max() / n),
        "n": int(n),
    }


def mechanics_readout(
    z: np.ndarray, values: np.ndarray, alpha: float = 1.0
) -> float:
    """Leave-one-out ridge R^2 predicting ``values`` from ``z``."""
    n, d = z.shape
    if n < 5 or np.std(values) < 1e-12:
        return float("nan")
    x = (z - z.mean(0)) / (z.std(0) + 1e-8)
    pred = np.empty(n)
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        xk, yk = x[m], values[m]
        mu = yk.mean()
        w = np.linalg.solve(xk.T @ xk + alpha * np.eye(d), xk.T @ (yk - mu))
        pred[k] = x[k] @ w + mu
    ss_res = float(((values - pred) ** 2).sum())
    ss_tot = float(((values - values.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def within_family_readout(
    z: np.ndarray, families: np.ndarray, params: dict[str, np.ndarray]
) -> dict:
    """Mechanics readout computed inside each family, then pooled.

    Doing this across families would be misleading: a drawer's friction is in
    newtons and a door's in newton-metres, so a probe could score well purely by
    identifying the family. Within a family the units are fixed, so any signal
    is genuinely about mechanics.
    """
    out: dict[str, dict[str, float]] = {}
    for col, vals in params.items():
        per_family = {}
        for fam in dict.fromkeys(families):
            m = families == fam
            if m.sum() >= 8:
                per_family[fam] = mechanics_readout(z[m], vals[m])
        finite = [v for v in per_family.values() if np.isfinite(v)]
        out[col] = {"per_family": per_family,
                    "mean": float(np.mean(finite)) if finite else float("nan")}
    return out


def embed_2d(z: np.ndarray, method: str, seed: int = 0) -> tuple[np.ndarray, str]:
    """2-D projection. PCA is exact and linear; UMAP is neither but preserves
    local neighbourhood structure that PCA flattens away."""
    if method == "pca":
        c = z - z.mean(0)
        u, s, _ = np.linalg.svd(c, full_matrices=False)
        var = s**2 / max(float((s**2).sum()), 1e-12)
        return u[:, :2] * s[:2], f"PCA (PC1 {100*var[0]:.0f}%, PC2 {100*var[1]:.0f}%)"
    if method == "umap":
        try:
            import umap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = umap.UMAP(n_neighbors=min(15, max(2, len(z) // 4)),
                              min_dist=0.1, random_state=seed).fit_transform(z)
            return np.asarray(r), "UMAP"
        except Exception as exc:
            print(f"  [warn] UMAP unavailable ({exc}); falling back to t-SNE")
            from sklearn.manifold import TSNE
            r = TSNE(n_components=2, perplexity=min(30, max(5, len(z) // 4)),
                     random_state=seed, init="pca").fit_transform(z)
            return np.asarray(r), "t-SNE (UMAP unavailable)"
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def latent_structure_figure(
    z: np.ndarray, families: np.ndarray, params: dict[str, np.ndarray],
    path: Path, title: str,
) -> Path:
    """Two rows -- PCA and UMAP -- coloured by family and by each parameter."""
    color_cols = ["inertia", "frictionloss", "damping", "stiffness"]
    ncols = 1 + len(color_cols)
    fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 6.2), squeeze=False)

    for row, method in enumerate(("pca", "umap")):
        xy, label = embed_2d(z, method)
        ax = axes[row][0]
        for fam in dict.fromkeys(families):
            m = families == fam
            ax.scatter(xy[m, 0], xy[m, 1], s=26, alpha=0.85, edgecolor="none",
                       color=FAMILY_COLORS.get(fam, "0.5"), label=fam)
        ax.set_title(f"{label}\ncoloured by family", fontsize=9)
        if row == 0:
            ax.legend(fontsize=6.5, loc="best", framealpha=0.9)

        for k, col in enumerate(color_cols):
            ax = axes[row][k + 1]
            v = params[col]
            # Log scale: inertia spans 0.007 to 40 across families, so a linear
            # colour map would show one bright point and 100 black ones.
            pos = v[v > 0]
            c = np.log10(np.maximum(v, pos.min() * 1e-3)) if len(pos) and v.min() >= 0 else v
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="viridis", s=26,
                            alpha=0.9, edgecolor="none")
            fig.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(f"log10 {col}", fontsize=9)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    return _save(fig, path)


def transfer_figure(rows: list[dict], path: Path) -> Path:
    """Adaptation quality per family, grouped by training mixture."""
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, axes = plt.subplots(1, len(variants), figsize=(3.4 * len(variants), 3.4),
                             squeeze=False, sharey=True)
    for ax, var in zip(axes[0], variants):
        sub = [r for r in rows if r["variant"] == var]
        fams = list(dict.fromkeys(r["family"] for r in sub))
        width = 0.38
        xs = np.arange(len(fams))
        for j, (m, hatch) in enumerate([("no-adaptation", "//"), ("latent-gd", "")]):
            vals = [np.median([r["nrmse_final"] for r in sub
                               if r["family"] == f and r["method"] == m] or [np.nan])
                    for f in fams]
            ax.bar(xs + (j - 0.5) * width, vals, width, hatch=hatch,
                   color=[FAMILY_COLORS.get(f, "0.5") for f in fams],
                   alpha=0.55 if j == 0 else 1.0,
                   edgecolor="k", linewidth=0.4,
                   label="no adaptation" if j == 0 else "latent adaptation")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(fams, rotation=35, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(var.replace("_", " "), fontsize=9)
    axes[0][0].set_ylabel("normalised error (1.0 = no better than static)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Cross-mechanism adaptation: held-out families", fontsize=12, y=1.03)
    fig.tight_layout()
    return _save(fig, path)


def structure_bar_figure(sep: dict, readout: dict, path: Path) -> Path:
    """The hypothesis test, as one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.bar([0, 1], [sep["accuracy"], sep["chance"]], color=["#1f77b4", "0.7"], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1-NN from latent", "majority class"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("family classification accuracy")
    ax.set_title("(a) is the space organised by category?", fontsize=9)

    ax = axes[1]
    cols = list(readout)
    vals = [readout[c]["mean"] for c in cols]
    ax.bar(np.arange(len(cols)), vals, color="#2ca02c", width=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("within-family LOO $R^2$")
    ax.set_ylim(min(-0.2, min(vals + [0]) - 0.05), 1.0)
    ax.set_title("(b) is it organised by mechanics?", fontsize=9)
    fig.suptitle("Category structure vs mechanics structure in the latent space",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------

def run_all(out: Path, results: dict, rows: list[dict]) -> dict:
    """Analyse the richest checkpoint: the one trained on the most families."""
    print(f"\n{'=' * 78}\nLATENT STRUCTURE ANALYSIS\n{'=' * 78}")
    variants = sorted(
        {r["variant"]: r["train_families"] for r in rows}.items(),
        key=lambda kv: -len(kv[1].split("+")),
    )
    name, train_fams = variants[0]
    ckpt = out / "runs" / name / "best.pt"
    npz = out / f"data_{name}.npz"
    if not ckpt.exists():
        print(f"  [skip] {ckpt} missing")
        return {}
    print(f"  analysing {name} (trained on {train_fams})")

    model, table, _, _ = load_checkpoint(ckpt)
    z = table.weight.detach().cpu().numpy()
    with np.load(npz, allow_pickle=False) as arr:
        fam_all = np.array([str(x) for x in arr["mechanism_family"]])
        gt = arr["door_params"]
        cols = [str(c) for c in arr["door_params_columns"]]
    families = fam_all[: len(z)]  # embeddings exist only for training instances
    params = {c: gt[: len(z), cols.index(c)]
              for c in ("inertia", "frictionloss", "damping", "stiffness")}

    sep = family_separability(z, families)
    print(f"\n  Family separability (1-NN, leave-one-out): "
          f"{sep['accuracy']:.2f}  (chance {sep['chance']:.2f})")
    readout = within_family_readout(z, families, params)
    print("  Within-family mechanics readout (LOO R^2):")
    for c, r in readout.items():
        per = "  ".join(f"{k[:8]}={v:+.2f}" for k, v in r["per_family"].items())
        print(f"    {c:14s} mean {r['mean']:+.3f}   {per}")

    latent_structure_figure(
        z, families, params, out / "latent_structure.png",
        f"Latent space of the model trained on: {train_fams.replace('+', ', ')}",
    )
    structure_bar_figure(sep, readout, out / "latent_hypothesis_test.png")
    transfer_figure(rows, out / "cross_mechanism_transfer.png")

    verdict = {
        "variant": name, "train_families": train_fams,
        "family_separability": sep,
        "within_family_readout": {k: v["mean"] for k, v in readout.items()},
        "per_family_readout": {k: v["per_family"] for k, v in readout.items()},
    }
    (out / "latent_analysis.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(f"  analysis -> {out / 'latent_analysis.json'}")
    return verdict


__all__ = ["family_separability", "mechanics_readout", "within_family_readout",
           "embed_2d", "latent_structure_figure", "transfer_figure", "run_all"]
