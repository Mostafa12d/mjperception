"""
Steps 2-6: what shape is the learned latent mechanics space?

Every routine here is read-only with respect to model weights.

The methodological spine of this module is the **matched null baseline**. Any
clustering procedure returns clusters. BIC will select K > 1 on perfectly
unimodal Gaussian data whenever the sample is small and the dimension is large,
which is exactly the regime here: 120 points in 16 dimensions, where a single
full-covariance component already costs 152 parameters. So every multimodality
statistic is computed twice -- once on the real latents, and once on synthetic
data drawn from a single Gaussian matched to the real data's mean, covariance,
sample size and dimension. Only the *difference* is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Step 2: geometry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 3: multimodality
# ---------------------------------------------------------------------------

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
    """Fit K = 1..k_max and report selection criteria plus held-out likelihood.

    Held-out log likelihood is the honest criterion here: BIC and AIC both
    penalise parameters with a formula that assumes the sample is large relative
    to the parameter count, which is not true at 120 points in 16 dimensions.
    Cross-validated likelihood makes no such assumption.
    """
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
    """Unimodal Gaussians matched to the real data's mean, covariance, N and d.

    This is the control every multimodality statistic is compared against.
    """
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

    # One-sided empirical p-value: how often does the null reach this silhouette?
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


def family_agreement(z: np.ndarray, family: np.ndarray, k: int, seed: int = 0) -> dict:
    """Do unsupervised clusters recover the known mechanism families?

    If the latent really is a discrete mixture over mechanism types, clusters
    found without labels should line up with the labels.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
    return {
        "k": k,
        "adjusted_rand": float(adjusted_rand_score(family, lab)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(family, lab)),
    }


# ---------------------------------------------------------------------------
# Step 4: continuity
# ---------------------------------------------------------------------------

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
    """Walk z from object A to object B, scoring on both objects' data.

    In a smooth space the two error curves cross over monotonically. A *barrier*
    -- both objects predicted worse at some interior point than at either end --
    is the signature of separated modes with nothing valid in between.
    """
    alphas = np.linspace(0, 1, n_steps)
    ea, eb = [], []
    for al in alphas:
        z = (1 - al) * z_a + al * z_b
        ea.append(_err_on(model, z, *data_a))
        eb.append(_err_on(model, z, *data_b))
    ea, eb = np.array(ea), np.array(eb)
    # Barrier: the best either object can do at an interior point, relative to
    # the better of the two endpoints.
    interior = np.minimum(ea, eb)[1:-1]
    endpoint = min(ea[0], eb[-1])
    return {
        "alphas": alphas.tolist(), "err_a": ea.tolist(), "err_b": eb.tolist(),
        "barrier_ratio": float(interior.min() / max(endpoint, 1e-12)),
        "monotone_a": bool(np.all(np.diff(ea) >= -1e-9)),
        "monotone_b": bool(np.all(np.diff(eb) <= 1e-9)),
    }


# ---------------------------------------------------------------------------
# Step 5: local linearity
# ---------------------------------------------------------------------------

def jacobian_stats(
    model: MechanicsDynamicsModel, z_samples: np.ndarray, states: np.ndarray,
    actions: np.ndarray, n_points: int = 400, seed: int = 0,
) -> dict:
    """Distribution of df/dz over operating points, and how fast it varies.

    An EKF/IMM linearises f about the current z. That is reasonable exactly when
    the Jacobian is well conditioned and roughly constant over the region the
    belief occupies; it breaks when the Jacobian swings by orders of magnitude
    over a typical update step.
    """
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
    # How much does J change between two operating points, relative to its size?
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
    """First-order prediction error as a function of how far z moves.

    ``f(z0 + d) ~= f(z0) + J d``. The reported number is the relative error of
    that approximation, so 0.1 means the linear model explains 90% of the true
    change. This is the quantity an EKF's accuracy hinges on.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(states), size=n_points)
    s = torch.as_tensor(states[idx], dtype=torch.float32)
    a = torch.as_tensor(actions[idx], dtype=torch.float32)
    # PER-SAMPLE Jacobians. Taking grad of a summed output and reusing it for
    # every sample silently multiplies J by the batch size, which inflates the
    # linearisation error by the same factor; vmap+jacrev keeps them separate.
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


# ---------------------------------------------------------------------------
# Step 6: where does the error come from?
# ---------------------------------------------------------------------------

def fit_oracle_latent(
    model: MechanicsDynamicsModel, s, a, ns, z_init: np.ndarray,
    steps: int = 400, lr: float = 0.05,
) -> np.ndarray:
    """Best latent for this object, fitted offline on all of its data.

    Defines the ceiling: whatever error remains at this z cannot be removed by
    any online belief update, however clever, because no z does better.
    """
    z = torch.nn.Parameter(torch.as_tensor(z_init, dtype=torch.float32).reshape(1, -1).clone())
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model.raw_output(s, a, z),
                                            model.target(s, ns))
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach().numpy().reshape(-1)


__all__ = [
    "project", "spectrum", "gmm_sweep", "cluster_indices", "matched_null",
    "multimodality_evidence", "family_agreement", "interpolation_profile",
    "jacobian_stats", "linearization_error", "fit_oracle_latent", "GMMResult",
]
