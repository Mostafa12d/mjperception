"""Do online-estimated latents have the same geometry as the offline training ones?

Three latent sets, all projected into the SAME frozen PCA basis fitted on the
training table (refitting per set would rotate the frame):

    train    embedding table rows      offline optimum, SEEN objects
    oracle   held-out, batch-fit       offline optimum, UNSEEN objects
    ukf      held-out, filtered        ONLINE estimate, UNSEEN objects

``oracle`` is the control that separates the two contrasts: train vs oracle is
the cost of being unseen, oracle vs ukf the cost of estimating online.

    python3.10 -m latent_mechanics.geometry.online_latents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_mechanics.belief.adaptor import UKFConfig, UKFLatentAdaptor
from latent_mechanics.belief.basis import DEFAULT_TABLE, load_or_create
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.geometry import analyses as an
from latent_mechanics.mechanisms.analysis import (
    FAMILY_COLORS,
    family_separability,
    mechanics_readout,
)
from latent_mechanics.model import load_checkpoint
from latent_mechanics.online.loop import (
    episode_boundaries,
    episode_stream,
    init_strategies,
    run_online_adaptation,
)

DATA = "runs/latent_mechanics/geometry/data_all_families.npz"
OUT = Path("runs/latent_mechanics/geometry/online_latents")
PARAMS = ("inertia", "frictionloss", "damping", "stiffness")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def collect(n_objects: int | None = None, oracle_steps: int = 1500,
            device: str = "cpu") -> dict:
    """Run the UKF and the batch oracle over the held-out objects."""
    model, table, _, _ = load_checkpoint(DEFAULT_TABLE, device=device,
                                         stage="geometry:online_latents")
    model.freeze()
    z_train = table.weight.detach().cpu().numpy().astype(np.float64)
    basis = load_or_create("runs/latent_mechanics/belief/latent_basis.npz",
                           DEFAULT_TABLE, n_components=8)
    init = init_strategies(z_train, 0)["medoid"]

    ds = DoorTransitionDataset(DATA, "heldout_door", exclude_near_limit=False)
    with np.load(DATA, allow_pickle=False) as a:
        fam_all = np.array([str(x) for x in a["mechanism_family"]])
        gt_all = a["door_params"]
        cols = [str(c) for c in a["door_params_columns"]]
    train_fam = fam_all[: len(z_train)]
    train_gt = gt_all[: len(z_train)]

    ids = [int(d) for d in ds.door_ids][: n_objects or len(ds.door_ids)]
    z_ukf, z_oracle, fams, gts, diag = [], [], [], [], []

    for k, did in enumerate(ids):
        stream = episode_stream(ds, did, exclude_near_limit=False)
        if len(stream) < 200:
            continue
        bounds = episode_boundaries(ds, did, exclude_near_limit=False)

        # online: causal, one pass
        ad = UKFLatentAdaptor(model, basis, UKFConfig(), init=init,
                              prior_latents=z_train, device=device)
        log = run_online_adaptation(ad, stream, door_id=did, boundaries=bounds,
                                    verify_frozen=False)
        z_ukf.append(ad.latent.astype(np.float64))
        b = ad.belief()
        diag.append({"door_id": did, "P_trace": float(np.trace(b["cov_reduced"])),
                     "R_trace": float(np.trace(b["R"])),
                     "travel": float(np.linalg.norm(ad.latent - init))})

        # offline: full-batch, restarted from the filter's answer too, so the
        # optimum is a genuine upper bound rather than a stalled fit
        m = ds.door_id.numpy() == did
        i = np.nonzero(m)[0]
        z_oracle.append(an.fit_oracle_latent(
            model, ds.state[i], ds.action[i], ds.next_state[i], init,
            steps=oracle_steps, extra_inits=[z_ukf[-1]]).astype(np.float64))

        fams.append(str(fam_all[did]))
        gts.append(gt_all[did])
        if (k + 1) % 15 == 0:
            print(f"    {k + 1}/{len(ids)} objects")

    return {
        "z_train": z_train, "fam_train": train_fam, "gt_train": train_gt,
        "z_ukf": np.array(z_ukf), "z_oracle": np.array(z_oracle),
        "fam_held": np.array(fams), "gt_held": np.array(gts),
        "cols": cols, "init": init, "basis": basis, "diag": diag,
    }


def describe_set(name: str, z: np.ndarray, fam: np.ndarray, gt: np.ndarray,
                 cols: list[str], z_train: np.ndarray) -> dict:
    """Every geometry statistic, computed identically for each latent set."""
    logI = np.log10(np.maximum(gt[:, cols.index("inertia")], 1e-9))
    out: dict = {"name": name, "n": len(z), "spectrum": an.spectrum(z)}

    # where it sits relative to the training cloud
    mu = z_train.mean(0)
    cov = np.cov(z_train, rowvar=False) + 1e-8 * np.eye(z_train.shape[1])
    Pinv = np.linalg.inv(cov)
    dif = z - mu
    out["mahalanobis_median"] = float(np.median(
        np.sqrt(np.einsum("ij,jk,ik->i", dif, Pinv, dif))))
    d_nn = np.linalg.norm(z[:, None] - z_train[None, :], axis=-1).min(axis=1)
    out["dist_to_nearest_train_median"] = float(np.median(d_nn))

    out["family_separability"] = family_separability(z, fam)
    out["nn_purity"] = an.nn_family_purity(z, fam)
    out["log_inertia_r2"] = mechanics_readout(z, logI)
    for c in PARAMS:
        out[f"probe_{c}"] = mechanics_readout(z, gt[:, cols.index(c)])
    try:
        out["scale_dominance"] = {
            k: v for k, v in an.excess_silhouette(z, k_max=8, n_null=10).items()
            if not isinstance(v, (list, np.ndarray))
        }
        z_res = an.residualise(z, [logI])
        out["scale_dominance_residualised"] = {
            k: v for k, v in an.excess_silhouette(z_res, k_max=8, n_null=10).items()
            if not isinstance(v, (list, np.ndarray))
        }
    except Exception as exc:  # keep the run alive if a null fit fails
        out["scale_dominance_error"] = str(exc)
    return out


def figure(res: dict, sets: dict, out_path: Path) -> Path:
    basis = res["basis"].truncate(2)
    xy = {k: basis.encode(v) for k, v in
          (("train", res["z_train"]), ("oracle", res["z_oracle"]),
           ("ukf", res["z_ukf"]))}
    fam_h = res["fam_held"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))

    # (a) all three sets in the fixed training frame
    ax = axes[0]
    ax.scatter(xy["train"][:, 0], xy["train"][:, 1], s=22, c="0.8",
               edgecolor="none", label="training table (offline, seen)")
    for f in dict.fromkeys(fam_h):
        m = fam_h == f
        c = FAMILY_COLORS.get(f, "0.4")
        ax.scatter(xy["oracle"][m, 0], xy["oracle"][m, 1], s=34,
                   facecolor="none", edgecolor=c, linewidth=1.2)
        ax.scatter(xy["ukf"][m, 0], xy["ukf"][m, 1], s=26, color=c,
                   edgecolor="none")
    for i in range(len(xy["ukf"])):
        ax.plot([xy["oracle"][i, 0], xy["ukf"][i, 0]],
                [xy["oracle"][i, 1], xy["ukf"][i, 1]], "-", color="0.5",
                lw=0.5, alpha=0.6, zorder=0)
    ax.set_xlabel("PC1 (training basis)"); ax.set_ylabel("PC2")
    ax.set_title("(a) same frozen frame\nopen = oracle, filled = UKF")
    ax.legend(fontsize=7, loc="best")

    # (b) how far each set sits from the training cloud
    ax = axes[1]
    mu = res["z_train"].mean(0)
    cov = np.cov(res["z_train"], rowvar=False) + 1e-8 * np.eye(res["z_train"].shape[1])
    Pinv = np.linalg.inv(cov)
    md = lambda Z: np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, Pinv, Z - mu))
    data = [md(res["z_train"]), md(res["z_oracle"]), md(res["z_ukf"])]
    ax.boxplot(data, labels=["train", "oracle", "UKF"], showfliers=False)
    ax.set_ylabel("Mahalanobis distance to training cloud")
    ax.set_title("(b) do they sit on the manifold?")

    # (c) parameter decodability
    ax = axes[2]
    keys = ["train", "oracle", "ukf"]
    x = np.arange(len(PARAMS)); w = 0.26
    for i, k in enumerate(keys):
        ax.bar(x + (i - 1) * w, [sets[k].get(f"probe_{c}", np.nan) for c in PARAMS],
               w, label=k, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([p[:9] for p in PARAMS], fontsize=8)
    ax.set_ylabel("leave-one-out probe $R^2$")
    ax.set_title("(c) do they still decode physics?")
    ax.legend(fontsize=7)

    # (d) UKF vs oracle, per object
    ax = axes[3]
    d_uo = np.linalg.norm(res["z_ukf"] - res["z_oracle"], axis=1)
    d_ui = np.linalg.norm(res["z_ukf"] - res["init"], axis=1)
    d_oi = np.linalg.norm(res["z_oracle"] - res["init"], axis=1)
    ax.scatter(d_oi, d_uo, s=26, c=[FAMILY_COLORS.get(f, "0.4") for f in fam_h])
    lim = max(d_oi.max(), d_uo.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\|z_{oracle} - z_{init}\|$  (how far the answer was)")
    ax.set_ylabel(r"$\|z_{UKF} - z_{oracle}\|$  (how far the filter missed)")
    ax.set_title(f"(d) below the diagonal = closer\nthan the prior ({100*np.mean(d_uo < d_oi):.0f}% of objects)")

    fig.suptitle("Online (UKF) vs offline (batch) latents in the frozen training frame",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"  figure -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objects", type=int, default=None)
    ap.add_argument("--oracle-steps", type=int, default=1500)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Collecting latents (UKF online + batch oracle) on held-out objects")
    res = collect(args.objects, args.oracle_steps)
    print(f"  train {len(res['z_train'])}, oracle {len(res['z_oracle'])}, "
          f"ukf {len(res['z_ukf'])}")

    sets = {
        "train": describe_set("train", res["z_train"], res["fam_train"],
                              res["gt_train"], res["cols"], res["z_train"]),
        "oracle": describe_set("oracle", res["z_oracle"], res["fam_held"],
                               res["gt_held"], res["cols"], res["z_train"]),
        "ukf": describe_set("ukf", res["z_ukf"], res["fam_held"],
                            res["gt_held"], res["cols"], res["z_train"]),
    }

    print("\n" + "=" * 88)
    print("GEOMETRY OF ONLINE vs OFFLINE LATENTS")
    print("=" * 88)
    print(f"  {'':22} {'train':>12} {'oracle':>12} {'UKF':>12}")
    print(f"  {'(offline/online)':22} {'offline':>12} {'offline':>12} {'ONLINE':>12}")
    print(f"  {'(seen/unseen)':22} {'seen':>12} {'unseen':>12} {'unseen':>12}")
    print("  " + "-" * 62)
    rows = [
        ("effective dim", lambda s: s["spectrum"]["effective_dim"]),
        ("PC1 variance %", lambda s: 100 * s["spectrum"]["explained_variance"][0]),
        ("Mahalanobis to train", lambda s: s["mahalanobis_median"]),
        ("dist to nearest train", lambda s: s["dist_to_nearest_train_median"]),
        ("family separability", lambda s: s["family_separability"]["accuracy"]),
        ("log-inertia probe R2", lambda s: s["log_inertia_r2"]),
        ("friction probe R2", lambda s: s["probe_frictionloss"]),
        ("stiffness probe R2", lambda s: s["probe_stiffness"]),
        ("damping probe R2", lambda s: s["probe_damping"]),
    ]
    for label, fn in rows:
        print(f"  {label:22}" + "".join(f"{fn(sets[k]):>12.3f}" for k in
                                        ("train", "oracle", "ukf")))

    print("\n  scale dominance (excess silhouette over a matched unimodal null)")
    for k in ("train", "oracle", "ukf"):
        s = sets[k]
        raw = s.get("scale_dominance", {}).get("excess", np.nan)
        res_ = s.get("scale_dominance_residualised", {}).get("excess", np.nan)
        print(f"    {k:8s} raw {raw:+.3f}   log-inertia removed {res_:+.3f}")

    d_uo = np.linalg.norm(res["z_ukf"] - res["z_oracle"], axis=1)
    d_oi = np.linalg.norm(res["z_oracle"] - res["init"], axis=1)
    print(f"\n  UKF vs oracle, per object:")
    print(f"    median |z_ukf - z_oracle|      = {np.median(d_uo):.3f}")
    print(f"    median |z_oracle - z_init|     = {np.median(d_oi):.3f}")
    print(f"    filter closer to oracle than the prior was: "
          f"{100 * np.mean(d_uo < d_oi):.0f}% of objects")

    figure(res, sets, out / "online_vs_offline_latents.png")
    np.savez_compressed(out / "latents.npz",
                        z_train=res["z_train"], z_oracle=res["z_oracle"],
                        z_ukf=res["z_ukf"], fam_held=res["fam_held"],
                        fam_train=res["fam_train"], gt_held=res["gt_held"],
                        gt_train=res["gt_train"], init=res["init"],
                        cols=np.array(res["cols"]))
    (out / "summary.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "nn_purity"}
         for k, v in sets.items()}, indent=2, default=str))
    print(f"  arrays  -> {out / 'latents.npz'}")
    print(f"  summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
