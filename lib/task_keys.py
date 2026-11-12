"""Task keys and routing scores for density-guided adapter routing.

Two kinds of task key are supported:

* The original per-adapter online diagonal GMM (``lib.signatures.OnlineDiagonalGMM``).
  ``gmm_mahal_min`` reproduces the published routing score exactly;
  ``gmm_log_likelihood`` scores the same GMM by its log-likelihood, which is
  the quantity the paper's retrieval theory is stated in (d(z, P) = -log p(z)).

* ``ClassGaussianKeyBank``: class-conditional Gaussian keys built from
  additive sufficient statistics (counts, feature sums, second moments).
  Like HESTIA's ridge statistics (G, g), every update is a sum, so the keys
  are invariant to batch order. Covariances are pooled across classes (per
  adapter or over the whole stream) and shrunk toward a scaled identity.

Routing scores returned by this module follow the original convention:
lower is better (a negative log-likelihood, or a Mahalanobis distance).
"""
from __future__ import annotations

import math

import torch

LOG_2PI = math.log(2.0 * math.pi)


def _keep_significant(counts, min_count):
    keep = counts >= min_count
    if not bool(keep.any()):  # guard: never route against an empty key
        keep = counts > 0
    return keep


@torch.no_grad()
def gmm_mahal_min(features, means, variances, counts, min_count=10):
    """Original HESTIA routing score: min diagonal Mahalanobis distance over the
    GMM components holding at least ``min_count`` points (no prior, no log-det)."""
    keep = _keep_significant(counts, min_count)
    m, v = means[keep], variances[keep]
    diff = features[:, None, :] - m[None, :, :]
    return (diff ** 2 / v[None, :, :]).sum(dim=2).min(dim=1).values


@torch.no_grad()
def gmm_log_likelihood(features, means, variances, counts, var_shrink=0.0, min_count=10):
    """log p(z) under the diagonal GMM: logsumexp_k [log pi_k + log N(z; m_k, diag(v_k))].

    ``var_shrink`` is a pseudo-count tau pulling each component variance toward
    the count-weighted pooled variance: v_k <- (n_k v_k + tau v_pool) / (n_k + tau).
    """
    keep = _keep_significant(counts, min_count)
    m, v, n = means[keep].double(), variances[keep].double(), counts[keep].double()
    z = features.double()
    if var_shrink > 0:
        pooled = (n[:, None] * v).sum(0) / n.sum()
        v = (n[:, None] * v + var_shrink * pooled) / (n[:, None] + var_shrink)
    log_pi = torch.log(n / n.sum())
    maha = ((z[:, None, :] - m[None]) ** 2 / v[None]).sum(-1)
    log_comp = log_pi[None] - 0.5 * (maha + torch.log(v).sum(-1)[None] + z.shape[1] * LOG_2PI)
    return torch.logsumexp(log_comp, dim=1)


def oas_shrinkage(cov, n_samples):
    """Oracle Approximating Shrinkage intensity (Chen et al., 2010), in the form
    used by scikit-learn's ``oas``: shrink ``cov`` toward (tr(cov)/p) I."""
    p = cov.shape[0]
    mu = torch.trace(cov) / p
    alpha = (cov ** 2).mean()
    num = alpha + mu ** 2
    den = (n_samples + 1.0) * (alpha - mu ** 2 / p)
    if den <= 0:
        return 1.0
    return float(min(num / den, 1.0))


class ClassGaussianKeyBank:
    """Order-invariant class-conditional Gaussian task keys, one per adapter.

    For adapter t the key is the mixture p_t(z) = sum_{c in C_t} pi_{c|t} N(z; mu_{t,c}, Sigma),
    where C_t are the classes observed while t was the active adapter.
    """

    def __init__(self, feature_dim=768, cov_type="full", cov_scope="shared", shrink_mode="oas",
                 shrink=0.1, class_prior="empirical", dtype=torch.float64):
        assert cov_type in ("full", "diag")
        assert cov_scope in ("shared", "task")
        assert shrink_mode in ("oas", "fixed")
        assert class_prior in ("empirical", "uniform")
        self.D = feature_dim
        self.cov_type = cov_type
        self.cov_scope = cov_scope
        self.shrink_mode = shrink_mode
        self.shrink = shrink
        self.class_prior = class_prior
        self.dtype = dtype
        self.stats = {}   # adapter id -> {"n": {c: float}, "s": {c: [D]}, "S": [D, D] or [D]}
        self._cache = None

    # ------------------------------------------------------------------
    # Accumulation (additive => order-invariant)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, adapter_id, features, labels):
        # Move to CPU *before* casting: MPS has no float64 and `.to("cpu", float64)`
        # on an MPS tensor silently returns garbage (observed with torch 2.14).
        z = features.detach().cpu().to(self.dtype)
        y = labels.detach().cpu()
        if not bool(torch.isfinite(z).all()):
            raise ValueError(f"non-finite embeddings passed to the task key of adapter {adapter_id}")
        if adapter_id not in self.stats:
            second = torch.zeros(self.D, self.D, dtype=self.dtype) if self.cov_type == "full" \
                else torch.zeros(self.D, dtype=self.dtype)
            self.stats[adapter_id] = {"n": {}, "s": {}, "S": second}
        st = self.stats[adapter_id]
        for c in torch.unique(y).tolist():
            zc = z[y == c]
            st["n"][c] = st["n"].get(c, 0.0) + float(zc.shape[0])
            st["s"][c] = st["s"].get(c, torch.zeros(self.D, dtype=self.dtype)) + zc.sum(0)
        st["S"] += z.T @ z if self.cov_type == "full" else (z ** 2).sum(0)
        self._cache = None

    def adapter_ids(self):
        return sorted(self.stats.keys())

    def classes(self, adapter_id):
        return sorted(self.stats[adapter_id]["n"].keys())

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------
    def _within_scatter(self, adapter_id):
        """Pooled within-class scatter W_t = S_t - sum_c s_c s_c^T / n_c and its dof."""
        st = self.stats[adapter_id]
        W = st["S"].clone()
        n_total = 0.0
        for c, n in st["n"].items():
            s = st["s"][c]
            W -= (torch.outer(s, s) if self.cov_type == "full" else s ** 2) / n
            n_total += n
        return W, n_total - len(st["n"])

    def _regularize(self, cov, dof):
        """Shrink toward (tr/p) I and return (precision, logdet)."""
        if self.cov_type == "full":
            cov = 0.5 * (cov + cov.T)
            gamma = oas_shrinkage(cov, dof) if self.shrink_mode == "oas" else self.shrink
            mu = torch.trace(cov) / self.D
            cov = (1.0 - gamma) * cov + gamma * mu * torch.eye(self.D, dtype=self.dtype)
            L = torch.linalg.cholesky(cov)
            return torch.cholesky_inverse(L), 2.0 * torch.log(torch.diagonal(L)).sum(), gamma
        var = cov.clamp_min(0)
        if self.shrink_mode == "oas":
            gamma = oas_shrinkage(torch.diag(var), dof)
        else:
            gamma = self.shrink
        var = (1.0 - gamma) * var + gamma * var.mean()
        return 1.0 / var, torch.log(var).sum(), gamma

    def _build_cache(self):
        ids = self.adapter_ids()
        scatters = {t: self._within_scatter(t) for t in ids}
        shared = None
        if self.cov_scope == "shared" or any(dof <= 1 for _, dof in scatters.values()):
            W = sum(w for w, _ in scatters.values())
            dof = sum(d for _, d in scatters.values())
            shared = self._regularize(W / max(dof, 1.0), max(dof, 1.0))
        cache = {}
        for t in ids:
            st = self.stats[t]
            cls = self.classes(t)
            n = torch.tensor([st["n"][c] for c in cls], dtype=self.dtype)
            means = torch.stack([st["s"][c] / st["n"][c] for c in cls])
            if self.cov_scope == "task" and scatters[t][1] > 1:
                W, dof = scatters[t]
                prec, logdet, gamma = self._regularize(W / dof, dof)
            else:
                prec, logdet, gamma = shared
            if self.class_prior == "empirical":
                log_prior = torch.log(n / n.sum())
            else:
                log_prior = torch.full_like(n, -math.log(len(cls)))
            if self.cov_type == "full":
                mP = means @ prec
                mPm = (mP * means).sum(1)
            else:
                mP = means * prec[None]
                mPm = (mP * means).sum(1)
            cache[t] = {"classes": cls, "means": means, "prec": prec, "logdet": logdet,
                        "log_prior": log_prior, "mP": mP, "mPm": mPm, "gamma": gamma}
        self._cache = cache

    def _ensure_cache(self):
        if self._cache is None:
            self._build_cache()
        return self._cache

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def class_log_likelihood(self, features, adapter_id):
        """[N, C_t] matrix of log pi_{c|t} + log N(z; mu_c, Sigma_t), and the class ids."""
        k = self._ensure_cache()[adapter_id]
        z = features.detach().cpu().to(self.dtype)
        if self.cov_type == "full":
            zPz = ((z @ k["prec"]) * z).sum(1)
        else:
            zPz = ((z ** 2) * k["prec"][None]).sum(1)
        maha = zPz[:, None] - 2.0 * z @ k["mP"].T + k["mPm"][None]
        ll = k["log_prior"][None] - 0.5 * (maha + k["logdet"] + self.D * LOG_2PI)
        return ll, k["classes"]

    @torch.no_grad()
    def log_likelihood(self, features, adapter_id):
        """log p_t(z) for every row of ``features``."""
        ll, _ = self.class_log_likelihood(features, adapter_id)
        return torch.logsumexp(ll, dim=1)

    @torch.no_grad()
    def log_likelihood_all(self, features):
        """[N, T] matrix of log p_t(z) for all adapters (same features for every key).

        The O(N D^2) quadratic term z^T P z is computed once per distinct
        precision matrix (once in total with ``cov_scope='shared'``).
        """
        cache = self._ensure_cache()
        z = features.detach().cpu().to(self.dtype)
        zPz_by_prec = {}
        cols = []
        for t in self.adapter_ids():
            k = cache[t]
            key = id(k["prec"])
            if key not in zPz_by_prec:
                zPz_by_prec[key] = ((z @ k["prec"]) * z).sum(1) if self.cov_type == "full" \
                    else ((z ** 2) * k["prec"][None]).sum(1)
            maha = zPz_by_prec[key][:, None] - 2.0 * z @ k["mP"].T + k["mPm"][None]
            ll = k["log_prior"][None] - 0.5 * (maha + k["logdet"] + self.D * LOG_2PI)
            cols.append(torch.logsumexp(ll, dim=1))
        return torch.stack(cols, dim=1)

    def describe(self):
        cache = self._ensure_cache()
        parts = [f"t{t}: {len(k['classes'])} cls, shrink={k['gamma']:.3f}" for t, k in cache.items()]
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self):
        return {"config": {"feature_dim": self.D, "cov_type": self.cov_type, "cov_scope": self.cov_scope,
                           "shrink_mode": self.shrink_mode, "shrink": self.shrink,
                           "class_prior": self.class_prior},
                "stats": self.stats}

    @classmethod
    def from_state_dict(cls, state):
        bank = cls(**state["config"])
        bank.stats = state["stats"]
        return bank


def build_key_bank(cfg, feature_dim=768):
    imp = cfg.improve
    if imp.routing_key != "class_gauss":
        return None
    return ClassGaussianKeyBank(feature_dim=feature_dim, cov_type=imp.cov_type, cov_scope=imp.cov_scope,
                                shrink_mode=imp.cov_shrink_mode, shrink=imp.cov_shrink,
                                class_prior=imp.class_prior)
