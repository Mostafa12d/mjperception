"""What shape is the learned latent mechanics space? All read-only w.r.t. weights.

The spine of this module is the matched null baseline: any clustering procedure
returns clusters, and BIC selects K > 1 on unimodal Gaussian data at 120 points
in 16 dimensions. Every multimodality statistic is therefore computed twice, on
the real latents and on a matched unimodal Gaussian. Only the difference counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from latent_mechanics.model import MechanicsDynamicsModel


def project(z: np.ndarray, method: str, seed: int = 0) -> tuple[np.ndarray, str]:
    """2-D projection by PCA, UMAP or t-SNE."""
    if method == "pca":
        c = z - z.mean(0)
        u, s, _ = np.linalg.svd(c, full_matrices=False)
        var = s**2 / max(float((s**2).sum()), 1e-12)
        return u[:, :2] * s[:2], f"PCA ({100*var[0]:.0f}%, {100*var[1]:.0f}%)"
    if method == "umap":
        import umap
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = umap.UMAP(n_neighbors=min(15, max(2, len(z) // 5)), min_dist=0.1,
                          random_state=seed).fit_transform(z)
        return np.asarray(r), "UMAP"
    if method == "tsne":
        from sklearn.manifold import TSNE
        r = TSNE(n_components=2, perplexity=min(30, max(5, len(z) // 5)),
                 random_state=seed, init="pca").fit_transform(z)
        return np.asarray(r), "t-SNE"
    raise ValueError(method)


def spectrum(z: np.ndarray) -> dict:
    """Variance spectrum and effective dimensionality (participation ratio)."""
    c = z - z.mean(0)
    ev = np.linalg.svd(c, compute_uv=False) ** 2
    var = ev / max(float(ev.sum()), 1e-30)
    return {
        "explained_variance": var.tolist(),
        "cumulative": np.cumsum(var).tolist(),
        "effective_dim": float(ev.sum() ** 2 / max(float((ev**2).sum()), 1e-30)),
        "dims_for_90pct": int(np.searchsorted(np.cumsum(var), 0.90) + 1),
    }


@dataclass
class GMMResult:
    k: int
    covariance_type: str
    bic: float
    aic: float
    heldout_ll: float
    mean_cluster_size: float
    min_cluster_size: int
    max_cond: float
    n_params: int


def gmm_sweep(
    z: np.ndarray, k_max: int = 15, covariance_type: str = "full",
    n_folds: int = 5, seed: int = 0,
) -> list[GMMResult]:
    """Fit K = 1..k_max with selection criteria plus held-out likelihood, which is
    the honest criterion at this sample size (BIC/AIC assume N >> n_params)."""
    rng = np.random.default_rng(seed)
    n = len(z)
    folds = np.array_split(rng.permutation(n), n_folds)
    out = []

    for k in range(1, k_max + 1):
        try:
            gm = GaussianMixture(k, covariance_type=covariance_type,
                                 random_state=seed, reg_covar=1e-6,
                                 n_init=3, max_iter=500).fit(z)
        except Exception:
            continue
        labels = gm.predict(z)
        sizes = np.bincount(labels, minlength=k)

        conds = []
        if covariance_type == "full":
            for c in gm.covariances_:
                conds.append(float(np.linalg.cond(c)))
        elif covariance_type == "diag":
            for c in gm.covariances_:
                conds.append(float(c.max() / max(c.min(), 1e-30)))
        else:
            conds = [1.0]

        lls = []
        for f in range(n_folds):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
            if len(tr) <= k:
                continue
            try:
                g2 = GaussianMixture(k, covariance_type=covariance_type,
                                     random_state=seed, reg_covar=1e-6,
                                     n_init=2, max_iter=500).fit(z[tr])
                lls.append(float(g2.score(z[te])))
            except Exception:
                pass

        out.append(GMMResult(
            k=k, covariance_type=covariance_type,
            bic=float(gm.bic(z)), aic=float(gm.aic(z)),
            heldout_ll=float(np.mean(lls)) if lls else float("nan"),
            mean_cluster_size=float(sizes.mean()), min_cluster_size=int(sizes.min()),
            max_cond=float(max(conds)), n_params=int(gm._n_parameters()),
        ))
    return out


def cluster_indices(z: np.ndarray, k_max: int = 15, seed: int = 0) -> dict:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz over K = 2..k_max."""
    out = {"k": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for k in range(2, k_max + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
        if len(set(lab)) < 2:
            continue
        out["k"].append(k)
        out["silhouette"].append(float(silhouette_score(z, lab)))
        out["davies_bouldin"].append(float(davies_bouldin_score(z, lab)))
        out["calinski_harabasz"].append(float(calinski_harabasz_score(z, lab)))
    return out


def matched_null(z: np.ndarray, n_reps: int = 20, seed: int = 0) -> list[np.ndarray]:
    """Unimodal Gaussians matched to the real data's mean, covariance, N and d --
    the control every multimodality statistic is compared against."""
    rng = np.random.default_rng(seed)
    mu = z.mean(0)
    cov = np.cov(z, rowvar=False) + 1e-8 * np.eye(z.shape[1])
    return [rng.multivariate_normal(mu, cov, size=len(z)) for _ in range(n_reps)]


def multimodality_evidence(
    z: np.ndarray, k_max: int = 15, n_null: int = 20, covariance_type: str = "diag",
    seed: int = 0,
) -> dict:
    """Real vs matched-null comparison of every multimodality statistic."""
    real_gmm = gmm_sweep(z, k_max, covariance_type, seed=seed)
    real_idx = cluster_indices(z, k_max, seed)

    nulls = matched_null(z, n_null, seed)
    null_bic_k, null_ll_k, null_sil = [], [], []
    for nz in nulls:
        g = gmm_sweep(nz, k_max, covariance_type, seed=seed)
        if not g:
            continue
        null_bic_k.append(min(g, key=lambda r: r.bic).k)
        finite = [r for r in g if np.isfinite(r.heldout_ll)]
        if finite:
            null_ll_k.append(max(finite, key=lambda r: r.heldout_ll).k)
        ci = cluster_indices(nz, min(k_max, 8), seed)
        if ci["silhouette"]:
            null_sil.append(max(ci["silhouette"]))

    best_bic = min(real_gmm, key=lambda r: r.bic)
    finite = [r for r in real_gmm if np.isfinite(r.heldout_ll)]
    best_ll = max(finite, key=lambda r: r.heldout_ll) if finite else None
    real_best_sil = max(real_idx["silhouette"]) if real_idx["silhouette"] else float("nan")

    # one-sided empirical p-value: how often the null reaches this silhouette
    p_sil = (float(np.mean(np.array(null_sil) >= real_best_sil))
             if null_sil else float("nan"))

    return {
        "gmm": real_gmm, "indices": real_idx,
        "best_k_bic": best_bic.k,
        "best_k_heldout_ll": best_ll.k if best_ll else None,
        "best_silhouette": real_best_sil,
        "null_best_k_bic_mean": float(np.mean(null_bic_k)) if null_bic_k else float("nan"),
        "null_best_k_bic_all": null_bic_k,
        "null_best_k_ll_mean": float(np.mean(null_ll_k)) if null_ll_k else float("nan"),
        "null_silhouette_mean": float(np.mean(null_sil)) if null_sil else float("nan"),
        "null_silhouette_max": float(np.max(null_sil)) if null_sil else float("nan"),
        "silhouette_p_value": p_sil,
        "n_points": len(z), "dim": z.shape[1], "covariance_type": covariance_type,
    }


def residualise(z: np.ndarray, covariates: Sequence[np.ndarray]) -> np.ndarray:
    """Residual of ``z`` after OLS on ``[1, *covariates]``. Used to ask whether the
    apparent cluster structure is just mechanical scale re-expressed."""
    X = np.column_stack([np.ones(len(z))] + [np.asarray(c, float) for c in covariates])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    return z - X @ beta


def best_silhouette(z: np.ndarray, k_max: int = 15, seed: int = 0) -> tuple[float, int]:
    """Max silhouette over K = 2..k_max, and the K attaining it."""
    best, best_k = -1.0, 0
    for k in range(2, k_max + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(z, lab))
        if s > best:
            best, best_k = s, k
    return best, best_k


def nn_family_purity(z: np.ndarray, family: np.ndarray) -> dict:
    """Fraction of objects whose nearest latent neighbour is the same family.

    A local statistic, reported alongside silhouette (which is global). Where the
    two disagree is the interesting part.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2).fit(z)
    _, idx = nn.kneighbors(z)
    same = family[idx[:, 1]] == family
    return {
        "overall": float(same.mean()),
        "per_family": {str(f): float(same[family == f].mean())
                       for f in sorted(set(family.tolist()))},
    }


def excess_silhouette(
    z: np.ndarray, k_max: int = 15, n_null: int = 20, seed: int = 0
) -> dict:
    """Best silhouette minus the matched null's, plus a p-value. The null is
    searched over the same K range, so both maxima have equally many candidates."""
    real, real_k = best_silhouette(z, k_max, seed)
    nulls = [best_silhouette(nz, k_max, seed)[0]
             for nz in matched_null(z, n_null, seed)]
    return {
        "silhouette": real, "silhouette_k": real_k,
        "null_mean": float(np.mean(nulls)), "null_max": float(np.max(nulls)),
        "excess": float(real - np.mean(nulls)),
        "p_value": float(np.mean(np.array(nulls) >= real)),
        "n_null": n_null, "k_max": k_max,
    }


def scale_dominance(
    z: np.ndarray,
    family: np.ndarray,
    log_inertia: np.ndarray,
    data_scale: Sequence[np.ndarray] = (),
    k_max: int = 15,
    n_null: int = 20,
    seed: int = 0,
) -> dict:
    """How much of the latent's cluster structure is just mechanical scale?

    Runs the same statistics on the raw latent and on scale-residualised versions.
    If the excess silhouette collapses once a scalar is regressed out, the
    apparent discrete structure was that scalar. ``data_scale`` adds observed
    covariates (what a filter sees) as distinct from ground-truth inertia.
    """
    variants: dict[str, np.ndarray] = {
        "raw": z,
        "resid_log_inertia": residualise(z, [log_inertia]),
    }
    if len(data_scale):
        variants["resid_data_scale"] = residualise(z, list(data_scale))
        variants["resid_inertia_and_data_scale"] = residualise(
            z, [log_inertia, *data_scale]
        )

    out: dict[str, dict] = {}
    for name, zv in variants.items():
        spec = spectrum(zv)
        ex = excess_silhouette(zv, k_max, n_null, seed)
        pur = nn_family_purity(zv, family)
        n_fam = len(set(family.tolist()))
        out[name] = {
            "pc1_variance": float(spec["explained_variance"][0]),
            "effective_dim": spec["effective_dim"],
            **ex,
            "family_agreement": family_agreement(zv, family, n_fam, seed),
            "purity": pur,
        }

    # which latent directions ARE the scale axis
    zc = z - z.mean(0)
    u, s, _ = np.linalg.svd(zc, full_matrices=False)
    pcs = u * s
    covs = {"log_inertia": np.asarray(log_inertia, float)}
    for i, c in enumerate(data_scale):
        covs[f"data_scale_{i}"] = np.asarray(c, float)
    out["pc_scale_correlation"] = {
        name: [float(np.corrcoef(pcs[:, k], c)[0, 1]) for k in range(min(4, pcs.shape[1]))]
        for name, c in covs.items()
    }
    base, resid = out["raw"]["excess"], out["resid_log_inertia"]["excess"]
    out["fraction_of_excess_explained_by_inertia"] = float(
        1.0 - resid / base) if abs(base) > 1e-12 else float("nan")
    return out


def family_agreement(z: np.ndarray, family: np.ndarray, k: int, seed: int = 0) -> dict:
    """Do unsupervised clusters recover the known mechanism families?"""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
    return {
        "k": k,
        "adjusted_rand": float(adjusted_rand_score(family, lab)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(family, lab)),
    }


@torch.no_grad()
def _err_on(model, z: np.ndarray, s, a, ns) -> float:
    zt = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
    pred = model(s, a, zt)
    d = (ns - s)
    scale = torch.sqrt((d[:, 0] ** 2).mean()).clamp_min(1e-12)
    return float(torch.sqrt(((pred - ns)[:, 0] ** 2).mean()) / scale)


def interpolation_profile(
    model: MechanicsDynamicsModel, z_a: np.ndarray, z_b: np.ndarray,
    data_a: tuple, data_b: tuple, n_steps: int = 21,
) -> dict:
    """Walk z from object A to B, scoring on both objects' data. In a smooth space
    the curves cross monotonically; a barrier means separated modes."""
    alphas = np.linspace(0, 1, n_steps)
    ea, eb = [], []
    for al in alphas:
        z = (1 - al) * z_a + al * z_b
        ea.append(_err_on(model, z, *data_a))
        eb.append(_err_on(model, z, *data_b))
    ea, eb = np.array(ea), np.array(eb)
    # barrier: best interior error relative to the better endpoint
    interior = np.minimum(ea, eb)[1:-1]
    endpoint = min(ea[0], eb[-1])
    return {
        "alphas": alphas.tolist(), "err_a": ea.tolist(), "err_b": eb.tolist(),
        "barrier_ratio": float(interior.min() / max(endpoint, 1e-12)),
        "monotone_a": bool(np.all(np.diff(ea) >= -1e-9)),
        "monotone_b": bool(np.all(np.diff(eb) <= 1e-9)),
    }


def jacobian_stats(
    model: MechanicsDynamicsModel, z_samples: np.ndarray, states: np.ndarray,
    actions: np.ndarray, n_points: int = 400, seed: int = 0,
) -> dict:
    """Distribution of df/dz over operating points, and how fast it varies. An EKF
    is only reasonable while this is well conditioned and roughly constant."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(states), size=n_points)
    zid = rng.integers(0, len(z_samples), size=n_points)
    norms, conds = [], []
    jacs = []

    for i in range(n_points):
        s = torch.as_tensor(states[idx[i]], dtype=torch.float32).reshape(1, -1)
        a = torch.as_tensor(actions[idx[i]], dtype=torch.float32).reshape(1, -1)
        z = torch.as_tensor(z_samples[zid[i]], dtype=torch.float32).reshape(1, -1)
        z.requires_grad_(True)
        out = model(s, a, z)
        J = torch.zeros(out.shape[-1], z.shape[-1])
        for r in range(out.shape[-1]):
            g = torch.autograd.grad(out[0, r], z, retain_graph=(r == 0))[0]
            J[r] = g[0]
        Jn = J.detach().numpy()
        jacs.append(Jn)
        norms.append(float(np.linalg.norm(Jn)))
        sv = np.linalg.svd(Jn, compute_uv=False)
        conds.append(float(sv[0] / max(sv[-1], 1e-30)))

    jacs = np.stack(jacs)
    # how much J changes between operating points, relative to its size
    m = jacs.mean(0)
    rel_var = float(np.mean(np.linalg.norm(jacs - m, axis=(1, 2))) /
                    max(np.linalg.norm(m), 1e-30))
    return {
        "norm_median": float(np.median(norms)),
        "norm_p05": float(np.percentile(norms, 5)),
        "norm_p95": float(np.percentile(norms, 95)),
        "norm_ratio_p95_p05": float(np.percentile(norms, 95) / max(np.percentile(norms, 5), 1e-30)),
        "cond_median": float(np.median(conds)),
        "cond_p95": float(np.percentile(conds, 95)),
        "relative_variation": rel_var,
        "n_points": n_points,
    }


def linearization_error(
    model: MechanicsDynamicsModel, z0: np.ndarray, states: np.ndarray,
    actions: np.ndarray, step_sizes: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
    n_points: int = 200, n_dirs: int = 8, seed: int = 0,
) -> dict:
    """Relative error of ``f(z0 + d) ~= f(z0) + J d`` as a function of |d|, so 0.1
    means the linear model explains 90% of the true change."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(states), size=n_points)
    s = torch.as_tensor(states[idx], dtype=torch.float32)
    a = torch.as_tensor(actions[idx], dtype=torch.float32)
    # per-sample Jacobians: a grad of the summed output would scale J by the batch
    z0_t = torch.as_tensor(z0, dtype=torch.float32).reshape(-1)

    def _f_one(zv, s_one, a_one):
        return model(s_one.reshape(1, -1), a_one.reshape(1, -1), zv.reshape(1, -1))[0]

    try:
        from torch.func import jacrev, vmap
        Jd = vmap(jacrev(_f_one), in_dims=(None, 0, 0))(z0_t, s, a).detach()
    except Exception:
        rows = []
        for i in range(len(s)):
            zi = z0_t.clone().requires_grad_(True)
            o = _f_one(zi, s[i], a[i])
            rows.append(torch.stack([
                torch.autograd.grad(o[r], zi, retain_graph=(r == 0))[0]
                for r in range(o.shape[0])]))
        Jd = torch.stack(rows).detach()
    with torch.no_grad():
        base_d = model(s, a, z0_t.reshape(1, -1))

    out = {}
    for h in step_sizes:
        errs = []
        for _ in range(n_dirs):
            d = rng.normal(size=z0.shape)
            d = d / np.linalg.norm(d) * h
            dt = torch.as_tensor(d, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                true = model(s, a, torch.as_tensor(z0, dtype=torch.float32).reshape(1, -1) + dt)
            lin = base_d + torch.einsum("nod,d->no", Jd, dt.reshape(-1))
            num = torch.linalg.norm(true - lin, dim=1)
            den = torch.linalg.norm(true - base_d, dim=1).clamp_min(1e-12)
            errs.append(float((num / den).median()))
        out[h] = float(np.mean(errs))
    return out


def fit_oracle_latent(
    model: MechanicsDynamicsModel, s, a, ns, z_init: np.ndarray,
    steps: int = 1500, lr: float = 0.05,
    extra_inits: "list[np.ndarray] | None" = None,
    objective: str = "normalised_delta",
) -> np.ndarray:
    """Best latent for this object, fitted offline on all of its data.

    The ceiling every online method is measured against, so convergence matters:
    cosine-decayed lr, best-iterate tracking (undecayed Adam wanders and can
    return a worse z than it started from), and optional restarts via
    ``extra_inits`` -- passing the estimate under test makes the result a strictly
    tighter upper bound on it.

    ``objective`` must match whatever the caller then scores, or the ceiling is a
    ceiling on a different quantity and methods can legitimately dip below it:
    ``normalised_delta`` is the network's own training loss, ``angle`` is the
    angle-channel error that this project's tables quote.
    """
    if objective not in ("normalised_delta", "angle"):
        raise ValueError(f"unknown objective {objective!r}")
    target = model.target(s, ns).detach()
    best_z, best_loss = None, float("inf")

    def compute_loss(z):
        if objective == "angle":
            return torch.nn.functional.mse_loss(model(s, a, z)[:, 0], ns[:, 0])
        return torch.nn.functional.mse_loss(model.raw_output(s, a, z), target)

    for z0 in [z_init, *(extra_inits or [])]:
        z = torch.nn.Parameter(
            torch.as_tensor(z0, dtype=torch.float32).reshape(1, -1).clone())
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
        for _ in range(steps):
            loss = compute_loss(z)
            # record the iterate that produced this loss, before stepping away
            if loss.item() < best_loss:
                best_loss, best_z = loss.item(), z.detach().clone()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            final = compute_loss(z).item()
        if final < best_loss:
            best_loss, best_z = final, z.detach().clone()

    return best_z.numpy().reshape(-1)


__all__ = [
    "project", "spectrum", "gmm_sweep", "cluster_indices", "matched_null",
    "multimodality_evidence", "family_agreement", "interpolation_profile",
    "jacobian_stats", "linearization_error", "fit_oracle_latent", "GMMResult",
    "residualise", "best_silhouette", "nn_family_purity", "excess_silhouette",
    "scale_dominance",
]
