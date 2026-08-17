"""Runs the latent-geometry investigation and writes its figures.

Read-only w.r.t. model weights, except that it may create an all-families
checkpoint via ``extract.build_all_families_checkpoint``.

    python3.10 -m latent_mechanics.geometry.report
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.geometry import analyses as an
from latent_mechanics.geometry.extract import (
    LatentDataset,
    build_all_families_checkpoint,
    extract_from_checkpoint,
)
from latent_mechanics.mechanisms.analysis import FAMILY_COLORS
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.loop import init_strategies

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)
    print(f"  figure -> {path}")
    return path


def figure_geometry(ds: LatentDataset, out: Path) -> Path:
    """PCA / UMAP / t-SNE, coloured by category and by instance."""
    methods = ["pca", "umap", "tsne"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for j, m in enumerate(methods):
        xy, lab = an.project(ds.z, m)
        ax = axes[0][j]
        for f in dict.fromkeys(ds.family):
            k = ds.family == f
            ax.scatter(xy[k, 0], xy[k, 1], s=26, alpha=0.85, edgecolor="none",
                       color=FAMILY_COLORS.get(f, "0.5"), label=f)
        ax.set_title(f"{lab}\nby mechanism family", fontsize=9)
        if j == 0:
            ax.legend(fontsize=6.5, loc="best")

        ax = axes[1][j]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=ds.instance_id, cmap="turbo", s=26,
                        alpha=0.9, edgecolor="none")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="instance id")
        ax.set_title("by object instance", fontsize=9)
    fig.suptitle("Latent geometry of the mechanics embedding "
                 f"({len(ds)} objects, {ds.dim}-D)", fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, out / "geometry_projections.png")


def figure_multimodality(ev: dict, out: Path) -> Path:
    """Selection criteria against a matched unimodal null."""
    g = ev["gmm"]
    ks = [r.k for r in g]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))

    ax = axes[0]
    ax.plot(ks, [r.bic for r in g], "o-", label="BIC", ms=4)
    ax.plot(ks, [r.aic for r in g], "s--", label="AIC", ms=4)
    ax.axvline(ev["best_k_bic"], color="#1f77b4", ls=":", lw=1)
    ax.set_xlabel("K"); ax.set_ylabel("criterion (lower better)")
    ax.set_title("(a) BIC / AIC"); ax.legend()

    ax = axes[1]
    ll = [r.heldout_ll for r in g]
    ax.plot(ks, ll, "o-", color="#2ca02c", ms=4)
    if ev["best_k_heldout_ll"]:
        ax.axvline(ev["best_k_heldout_ll"], color="#2ca02c", ls=":", lw=1)
    ax.set_xlabel("K"); ax.set_ylabel("held-out log likelihood")
    ax.set_title("(b) cross-validated fit")

    ax = axes[2]
    idx = ev["indices"]
    ax.plot(idx["k"], idx["silhouette"], "o-", color="#d62728", ms=4,
            label="real latents")
    if np.isfinite(ev["null_silhouette_mean"]):
        ax.axhline(ev["null_silhouette_mean"], color="0.5", ls="--", lw=1,
                   label="unimodal null (mean)")
        ax.axhline(ev["null_silhouette_max"], color="0.5", ls=":", lw=1,
                   label="unimodal null (max)")
    ax.set_xlabel("K"); ax.set_ylabel("silhouette")
    ax.set_title("(c) separation vs null"); ax.legend(fontsize=7)

    ax = axes[3]
    nb = ev["null_best_k_bic_all"]
    if nb:
        ax.hist(nb, bins=np.arange(0.5, max(max(nb), ev["best_k_bic"]) + 1.5),
                color="0.7", label="null: BIC-optimal K")
    ax.axvline(ev["best_k_bic"], color="#d62728", lw=2, label="real: BIC-optimal K")
    ax.set_xlabel("K selected"); ax.set_ylabel("count")
    ax.set_title("(d) is K>1 just small-sample noise?"); ax.legend(fontsize=7)

    fig.suptitle(f"Multimodality evidence  ({ev['n_points']} points, {ev['dim']}-D, "
                 f"{ev['covariance_type']} covariance)", fontsize=12, y=1.05)
    fig.tight_layout()
    return _save(fig, out / "multimodality.png")


def figure_continuity(profiles: list[dict], out: Path) -> Path:
    """Error while interpolating z between object pairs."""
    within = [p for p in profiles if p["kind"] == "within"]
    across = [p for p in profiles if p["kind"] == "across"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for ax, group, title in ((axes[0], within, "within family"),
                             (axes[1], across, "across families")):
        for p in group[:14]:
            al = p["alphas"]
            ax.plot(al, p["err_a"], color="#1f77b4", alpha=0.35, lw=1)
            ax.plot(al, p["err_b"], color="#d62728", alpha=0.35, lw=1)
        ax.set_yscale("log")
        ax.set_xlabel(r"interpolation $\alpha$  ($z=(1-\alpha)z_A+\alpha z_B$)")
        ax.set_ylabel("normalised error")
        ax.set_title(f"{title}\nblue = scored on A, red = scored on B", fontsize=9)

    ax = axes[2]
    for group, lab, c in ((within, "within family", "#2ca02c"),
                          (across, "across families", "#d62728")):
        vals = [p["barrier_ratio"] for p in group]
        if vals:
            ax.hist(vals, bins=np.linspace(0, max(3, np.percentile(vals, 95)), 20),
                    alpha=0.6, label=f"{lab} (median {np.median(vals):.2f})", color=c)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("barrier ratio (interior error / endpoint error)")
    ax.set_ylabel("count")
    ax.set_title("(c) is there a gap between objects?", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Latent interpolation: continuous manifold or separated modes?",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    return _save(fig, out / "continuity.png")


def figure_linearity(jac: dict, lin: dict, attribution: dict, out: Path) -> Path:
    """Local linearity and error attribution."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))

    ax = axes[0]
    hs = sorted(lin)
    ax.plot(hs, [lin[h] for h in hs], "o-", ms=5, color="#1f77b4")
    ax.axhline(0.1, color="k", ls=":", lw=1)
    ax.set_xlabel(r"latent step size $\|\delta z\|$")
    ax.set_ylabel("relative first-order error")
    ax.set_title("(a) how far can you linearise?", fontsize=9)
    ax.annotate("10% error", xy=(hs[0], 0.105), fontsize=7, color="0.35")

    ax = axes[1]
    ax.bar([0, 1, 2], [jac["norm_p05"], jac["norm_median"], jac["norm_p95"]],
           color=["0.7", "#1f77b4", "0.7"], width=0.6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["p05", "median", "p95"])
    ax.set_ylabel(r"$\|\partial f/\partial z\|_F$")
    ax.set_title(f"(b) Jacobian spread\np95/p05 = {jac['norm_ratio_p95_p05']:.1f}x",
                 fontsize=9)

    ax = axes[2]
    fams = list(attribution["per_family"])
    x = np.arange(len(fams))
    w = 0.27
    for i, (key, lab, c) in enumerate((("prior", "prior z", "0.6"),
                                       ("online", "online-adapted z", "#1f77b4"),
                                       ("oracle", "oracle z (ceiling)", "#2ca02c"))):
        ax.bar(x + (i - 1) * w, [attribution["per_family"][f][key] for f in fams],
               w, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("normalised error")
    ax.set_title("(c) how much error can z explain?", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Local linearity and the reachable error floor", fontsize=12, y=1.04)
    fig.tight_layout()
    return _save(fig, out / "linearity_attribution.png")


def per_object_data_scale(ds: LatentDataset) -> tuple[np.ndarray, np.ndarray]:
    """Each object's own observed scale: log RMS step size and log RMS action.
    Unlike ground-truth inertia, these are things a filter can actually see."""
    with np.load(ds.npz_path, allow_pickle=False) as a:
        did, spl = a["door_id"], a["split"]
        st, nxt, act = a["state"], a["next_state"], a["action"]
    rms_dq, rms_a = [], []
    for i in range(len(ds)):
        m = (did == i) & np.isin(spl, [0, 1])       # train/val rows for this object
        d = nxt[m] - st[m]
        rms_dq.append(np.sqrt((d[:, 0] ** 2).mean()) if m.any() else np.nan)
        rms_a.append(np.sqrt((act[m][:, 0] ** 2).mean()) if m.any() else np.nan)
    f = lambda v: np.log10(np.maximum(np.asarray(v, float), 1e-12))
    return f(rms_dq), f(rms_a)


def scale_dominance_for(ds: LatentDataset, k_max: int = 15, n_null: int = 20) -> dict:
    """Scale-dominance statistics for one latent dataset."""
    inertia = ds.params[:, ds.param_names.index("inertia")]
    log_I = np.log10(np.maximum(inertia, 1e-12))
    log_dq, log_a = per_object_data_scale(ds)
    sd = an.scale_dominance(ds.z, ds.family, log_I, data_scale=(log_dq, log_a),
                            k_max=k_max, n_null=n_null)
    sd["log_inertia_range"] = [float(log_I.min()), float(log_I.max())]
    return sd


def _print_scale_dominance(sd: dict) -> None:
    order = [k for k in ("raw", "resid_log_inertia", "resid_data_scale",
                         "resid_inertia_and_data_scale") if k in sd]
    print(f"  {'variant':30s} {'PC1%':>6} {'d_eff':>6} {'sil':>6} {'null':>6} "
          f"{'excess':>8} {'p':>6} {'ARI':>7} {'1NN pur':>8}")
    for k in order:
        r = sd[k]
        print(f"  {k:30s} {100*r['pc1_variance']:>6.1f} {r['effective_dim']:>6.2f} "
              f"{r['silhouette']:>6.3f} {r['null_mean']:>6.3f} {r['excess']:>+8.3f} "
              f"{r['p_value']:>6.2f} {r['family_agreement']['adjusted_rand']:>+7.3f} "
              f"{r['purity']['overall']:>8.3f}")
    print(f"\n  log10(inertia) spans {sd['log_inertia_range'][0]:+.2f} to "
          f"{sd['log_inertia_range'][1]:+.2f}")
    print(f"  fraction of excess silhouette explained by log-inertia alone: "
          f"{100*sd['fraction_of_excess_explained_by_inertia']:.0f}%")
    print("  PC correlation with scale (PC1..PC4):")
    for name, rs in sd["pc_scale_correlation"].items():
        print(f"    {name:16s} " + " ".join(f"{v:+.3f}" for v in rs))
    print("  per-family 1-NN purity:")
    fams = sorted(sd["raw"]["purity"]["per_family"])
    print(f"    {'variant':30s} " + " ".join(f"{f[:9]:>10s}" for f in fams))
    for k in order:
        print(f"    {k:30s} "
              + " ".join(f"{sd[k]['purity']['per_family'][f]:>10.2f}" for f in fams))


def figure_scale_dominance(sd: dict, out: Path) -> Path:
    """Excess silhouette and purity, before and after removing scale."""
    order = [k for k in ("raw", "resid_log_inertia", "resid_data_scale",
                         "resid_inertia_and_data_scale") if k in sd]
    short = {"raw": "raw z", "resid_log_inertia": "log-inertia\nout",
             "resid_data_scale": "data scale\nout",
             "resid_inertia_and_data_scale": "both\nout"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    x = np.arange(len(order))

    ax = axes[0]
    ax.bar(x - 0.2, [sd[k]["silhouette"] for k in order], 0.4, label="real",
           color="#d62728")
    ax.bar(x + 0.2, [sd[k]["null_mean"] for k in order], 0.4,
           label="matched unimodal null", color="0.7")
    ax.set_xticks(x); ax.set_xticklabels([short[k] for k in order], fontsize=7)
    ax.set_ylabel("best silhouette")
    ax.set_title("(a) separation vs null", fontsize=9); ax.legend(fontsize=7)

    ax = axes[1]
    cols = ["#d62728" if sd[k]["p_value"] < 0.05 else "0.7" for k in order]
    ax.bar(x, [sd[k]["excess"] for k in order], 0.6, color=cols)
    for i, k in enumerate(order):
        ax.annotate(f"p={sd[k]['p_value']:.2f}",
                    xy=(i, sd[k]["excess"]), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([short[k] for k in order], fontsize=7)
    ax.set_ylabel("excess silhouette over null")
    ax.set_title("(b) how much survives removing scale?", fontsize=9)

    ax = axes[2]
    fams = sorted(sd["raw"]["purity"]["per_family"])
    w = 0.8 / len(order)
    for i, k in enumerate(order):
        ax.bar(np.arange(len(fams)) + (i - (len(order) - 1) / 2) * w,
               [sd[k]["purity"]["per_family"][f] for f in fams], w,
               label=short[k].replace("\n", " "))
    ax.axhline(1.0 / len(fams), color="k", ls=":", lw=1)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("1-NN family purity")
    ax.set_title("(c) local purity is far more scale-robust", fontsize=9)
    ax.legend(fontsize=6.5)

    fig.suptitle("Is the latent's cluster structure anything more than "
                 "mechanical scale?", fontsize=12, y=1.04)
    fig.tight_layout()
    return _save(fig, out / "scale_dominance.png")


def run(out: Path, k_max: int = 15, n_null: int = 20, epochs: int = 40) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    print("=" * 78 + "\nSTEP 1  extract the latent dataset\n" + "=" * 78)
    ckpt, npz = build_all_families_checkpoint(out, epochs=epochs)
    ds = extract_from_checkpoint(ckpt, npz)
    ds.save(out / "latents_all_families.npz")
    print(ds.summary())
    report["dataset"] = {"n": len(ds), "dim": ds.dim,
                         "families": {f: int((ds.family == f).sum())
                                      for f in dict.fromkeys(ds.family)}}

    print("\n" + "=" * 78 + "\nSTEP 2  latent geometry\n" + "=" * 78)
    spec = an.spectrum(ds.z)
    print(f"  effective dimension {spec['effective_dim']:.2f} of {ds.dim}; "
          f"{spec['dims_for_90pct']} components reach 90% variance")
    print("  variance: " + " ".join(f"{100*v:.0f}%" for v in spec["explained_variance"][:6]))
    report["spectrum"] = spec
    figure_geometry(ds, out)

    print("\n" + "=" * 78 + "\nSTEP 3  multimodality\n" + "=" * 78)
    n, d = len(ds), ds.dim
    print(f"  {n} points in {d}-D. A full-covariance component costs "
          f"{d + d*(d+1)//2} parameters, so full covariance is not identifiable "
          f"here; using diagonal.")
    ev = an.multimodality_evidence(ds.z, k_max, n_null, "diag")
    print(f"  BIC-optimal K            : {ev['best_k_bic']}")
    print(f"  held-out-LL-optimal K    : {ev['best_k_heldout_ll']}")
    print(f"  best silhouette          : {ev['best_silhouette']:.3f}")
    print(f"  --- matched unimodal null (same N, d, covariance) ---")
    print(f"  null BIC-optimal K       : {ev['null_best_k_bic_mean']:.1f} (mean)")
    print(f"  null held-out-LL K       : {ev['null_best_k_ll_mean']:.1f} (mean)")
    print(f"  null silhouette          : {ev['null_silhouette_mean']:.3f} mean, "
          f"{ev['null_silhouette_max']:.3f} max")
    print(f"  P(null >= real silhouette) = {ev['silhouette_p_value']:.3f}")
    for k in (2, 6, 8):
        fa = an.family_agreement(ds.z, ds.family, k)
        print(f"  K={k}: agreement with true family labels  ARI={fa['adjusted_rand']:+.3f} "
              f"AMI={fa['adjusted_mutual_info']:+.3f}")
        report.setdefault("family_agreement", {})[k] = fa
    report["multimodality"] = {k: v for k, v in ev.items() if k != "gmm"}
    report["multimodality"]["gmm"] = [asdict(r) for r in ev["gmm"]]
    figure_multimodality(ev, out)

    print("\n" + "=" * 78 + "\nSTEP 3b  is that structure just mechanical scale?\n"
          + "=" * 78)
    sd = scale_dominance_for(ds, k_max, n_null)
    report["scale_dominance"] = sd
    _print_scale_dominance(sd)
    figure_scale_dominance(sd, out)

    # shared model + data for the remaining steps
    model, table, _, _ = load_checkpoint(ckpt, device="cpu", stage="geometry_report")
    model.freeze()
    tr = DoorTransitionDataset(npz, "train", exclude_near_limit=False)
    held = DoorTransitionDataset(npz, "heldout_door", exclude_near_limit=False)

    def obj_tensors(dataset, did, cap=4000):
        m = (dataset.door_id.numpy() == did)
        i = np.nonzero(m)[0][:cap]
        return (dataset.state[i], dataset.action[i], dataset.next_state[i])

    print("\n" + "=" * 78 + "\nSTEP 4  is the latent continuous?\n" + "=" * 78)
    rng = np.random.default_rng(0)
    profiles = []
    for kind in ("within", "across"):
        made = 0
        while made < 12:
            i, j = rng.integers(0, len(ds), 2)
            same = ds.family[i] == ds.family[j]
            if i == j or (kind == "within") != bool(same):
                continue
            p = an.interpolation_profile(model, ds.z[i], ds.z[j],
                                         obj_tensors(tr, int(i)), obj_tensors(tr, int(j)))
            p["kind"] = kind
            p["pair"] = [str(ds.family[i]), str(ds.family[j])]
            profiles.append(p); made += 1
    for kind in ("within", "across"):
        b = [p["barrier_ratio"] for p in profiles if p["kind"] == kind]
        frac = float(np.mean(np.array(b) > 1.2))
        print(f"  {kind:8s}: median barrier ratio {np.median(b):.2f}, "
              f"{100*frac:.0f}% of pairs show a barrier > 1.2x")
    report["continuity"] = {
        k: {"median_barrier": float(np.median([p["barrier_ratio"] for p in profiles
                                               if p["kind"] == k])),
            "frac_with_barrier": float(np.mean([p["barrier_ratio"] > 1.2
                                                for p in profiles if p["kind"] == k]))}
        for k in ("within", "across")}
    figure_continuity(profiles, out)

    print("\n" + "=" * 78 + "\nSTEP 5  local linearity\n" + "=" * 78)
    S = tr.state.numpy(); A = tr.action.numpy()
    jac = an.jacobian_stats(model, ds.z, S, A, n_points=400)
    print(f"  ||df/dz||_F  median {jac['norm_median']:.3f}  "
          f"p05 {jac['norm_p05']:.3f}  p95 {jac['norm_p95']:.3f}  "
          f"(spread {jac['norm_ratio_p95_p05']:.1f}x)")
    print(f"  condition number median {jac['cond_median']:.1f}, p95 {jac['cond_p95']:.1f}")
    print(f"  relative variation of J across operating points: "
          f"{jac['relative_variation']:.2f}")
    lin = an.linearization_error(model, ds.z.mean(0), S, A)
    print("  first-order approximation error vs step size:")
    for h, e in sorted(lin.items()):
        print(f"    |dz|={h:<5} relative error {e:.3f}")
    report["jacobian"] = jac
    report["linearization_error"] = {str(k): v for k, v in lin.items()}

    print("\n" + "=" * 78 + "\nSTEP 6  where does the error come from?\n" + "=" * 78)
    init = init_strategies(ds.z, 0)["medoid"]
    per_family: dict[str, dict] = {}
    rows = []
    for did in held.door_ids:
        did = int(did)
        s, a, ns = obj_tensors(held, did)
        if len(s) < 100:
            continue
        fam = str(np.load(npz, allow_pickle=False)["mechanism_family"][did])
        e_prior = an._err_on(model, init, s, a, ns)
        z_or = an.fit_oracle_latent(model, s, a, ns, init)
        e_oracle = an._err_on(model, z_or, s, a, ns)
        rows.append({"door_id": did, "family": fam, "prior": e_prior,
                     "oracle": e_oracle,
                     "explainable_fraction": 1.0 - e_oracle / max(e_prior, 1e-12),
                     "oracle_dist_from_prior": float(np.linalg.norm(z_or - init))})
    for f in dict.fromkeys(r["family"] for r in rows):
        sub = [r for r in rows if r["family"] == f]
        per_family[f] = {
            "prior": float(np.median([r["prior"] for r in sub])),
            "oracle": float(np.median([r["oracle"] for r in sub])),
            "online": float(np.median([r["prior"] for r in sub])),  # filled below
            "explainable_fraction": float(np.median([r["explainable_fraction"] for r in sub])),
            "oracle_dist": float(np.median([r["oracle_dist_from_prior"] for r in sub])),
        }
    # online-adapted error on the same objects
    from latent_mechanics.online.adaptor import GradientLatentAdaptor
    from latent_mechanics.online.config import load_config as load_online_config
    from latent_mechanics.online.loop import episode_stream, run_online_adaptation
    oc = load_online_config("configs/online_adaptation.yaml").adaptor
    for f in per_family:
        errs = []
        for r in [x for x in rows if x["family"] == f][:6]:
            stream = episode_stream(held, r["door_id"], exclude_near_limit=False)
            if len(stream) < 100:
                continue
            lg = run_online_adaptation(
                GradientLatentAdaptor(model, init=init, lr=oc.lr, window=oc.window,
                                      lr_decay=oc.lr_decay, n_inner_steps=oc.n_inner_steps),
                stream, door_id=r["door_id"], verify_frozen=False)
            st = np.stack([x for x, _, _ in stream]); nx = np.stack([x for _, _, x in stream])
            tail = max(1, len(stream) // 4)
            d = nx[-tail:] - st[-tail:]
            sc = max(float(np.sqrt((d[:, 0] ** 2).mean())), 1e-12)
            errs.append(float(np.sqrt((lg.error[-tail:, 0] ** 2).mean())) / sc)
        if errs:
            per_family[f]["online"] = float(np.median(errs))

    print(f"  {'family':17s} {'prior':>10} {'online':>10} {'oracle':>10} "
          f"{'z-explainable':>14} {'|z*-z0|':>9}")
    for f, d_ in per_family.items():
        print(f"  {f:17s} {d_['prior']:>10.3e} {d_['online']:>10.3e} "
              f"{d_['oracle']:>10.3e} {100*d_['explainable_fraction']:>13.0f}% "
              f"{d_['oracle_dist']:>9.2f}")
    overall = float(np.median([r["explainable_fraction"] for r in rows]))
    print(f"\n  Median fraction of prior error removable by choosing a better z: "
          f"{100*overall:.0f}%")
    report["attribution"] = {"per_family": per_family, "rows": rows,
                             "median_explainable_fraction": overall}
    figure_linearity(jac, lin, report["attribution"], out)

    (out / "geometry_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  report -> {out / 'geometry_report.json'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="runs/latent_mechanics/geometry")
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    run(Path(a.out_dir), a.k_max, a.n_null, a.epochs)


if __name__ == "__main__":
    main()
