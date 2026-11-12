"""Correctness checks for the thesis improvements.

Run from the project root:  .venv/bin/python -m pytest -q tests/   (or python tests/test_improvements.py)
The ViT-based tests need the pretrained weights in the torch hub cache.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.signatures import OnlineDiagonalGMM
from lib.task_keys import ClassGaussianKeyBank, gmm_log_likelihood, gmm_mahal_min, oas_shrinkage


def test_woodbury_matches_dense_ridge():
    torch.manual_seed(0)
    for d in (768, 3072):
        A = torch.randn(16, d, dtype=torch.float64) * 0.05
        z = torch.rand(16, dtype=torch.float64)
        lam = 1.0
        dense = torch.linalg.solve(A.T @ A + lam * torch.eye(d, dtype=torch.float64), A.T @ z)
        K = A @ A.T + lam * torch.eye(16, dtype=torch.float64)
        fast = A.T @ torch.cholesky_solve(z[:, None], torch.linalg.cholesky(K)).squeeze(1)
        assert torch.allclose(dense, fast, atol=1e-10), (d, (dense - fast).abs().max())


def test_mahal_min_matches_original_eval_formula():
    torch.manual_seed(0)
    feats = torch.randn(16, 768)
    means, var, counts = torch.randn(8, 768), torch.rand(8, 768) + 0.1, torch.tensor([20., 3, 50, 10, 9, 11, 100, 1])
    keep = counts >= 10
    diff = feats[:, None, :] - means[keep][None]
    ref = (diff ** 2 / var[keep][None]).sum(dim=2).min(dim=1).values
    assert torch.allclose(gmm_mahal_min(feats, means, var, counts), ref)


def test_gmm_log_likelihood_matches_sklearn():
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20))
    gm = GaussianMixture(3, covariance_type="diag", random_state=0).fit(X)
    counts = torch.tensor(gm.weights_ * 1000.0)
    ll = gmm_log_likelihood(torch.tensor(X), torch.tensor(gm.means_), torch.tensor(gm.covariances_), counts)
    assert np.allclose(ll.numpy(), gm.score_samples(X), atol=1e-8)


def _stream(seed=0, n_classes=6, n=600, d=32):
    g = torch.Generator().manual_seed(seed)
    mus = torch.randn(n_classes, d, generator=g) * 2
    y = torch.randint(0, n_classes, (n,), generator=g)
    x = mus[y] + torch.randn(n, d, generator=g)
    return x, y


def test_class_gauss_is_order_invariant():
    x, y = _stream()
    probe = torch.randn(50, x.shape[1])
    results = []
    for perm_seed in (1, 2):
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(perm_seed))
        bank = ClassGaussianKeyBank(feature_dim=x.shape[1])
        for i in range(0, x.shape[0], 16):
            idx = perm[i:i + 16]
            bank.update(0 if bool((y[idx] < 3).all()) else 0, x[idx], y[idx])
        results.append(bank.log_likelihood(probe, 0))
    assert torch.allclose(results[0], results[1], atol=1e-8)


def test_class_gauss_matches_direct_lda_likelihood():
    from scipy.stats import multivariate_normal
    from scipy.special import logsumexp
    x, y = _stream(d=8)
    bank = ClassGaussianKeyBank(feature_dim=8, shrink_mode="fixed", shrink=0.0)
    for i in range(0, x.shape[0], 16):
        bank.update(0, x[i:i + 16], y[i:i + 16])
    X, Y = x.double().numpy(), y.numpy()
    classes = np.unique(Y)
    means = np.stack([X[Y == c].mean(0) for c in classes])
    W = sum((X[Y == c] - means[i]).T @ (X[Y == c] - means[i]) for i, c in enumerate(classes))
    cov = W / (len(X) - len(classes))
    priors = np.array([(Y == c).mean() for c in classes])
    probe = np.random.default_rng(1).normal(size=(20, 8))
    ref = logsumexp(np.stack([np.log(priors[i]) + multivariate_normal(means[i], cov).logpdf(probe)
                              for i in range(len(classes))], 1), axis=1)
    ours = bank.log_likelihood(torch.tensor(probe), 0).numpy()
    assert np.allclose(ours, ref, atol=1e-8)


def test_oas_matches_sklearn():
    from sklearn.covariance import oas
    X = np.random.default_rng(0).normal(size=(40, 30))
    _, ref = oas(X)
    emp = np.cov(X.T, bias=True)
    assert abs(oas_shrinkage(torch.tensor(emp), 40) - ref) < 1e-10


def test_task_scope_and_diag_run():
    x, y = _stream()
    for cov_type in ("full", "diag"):
        for scope in ("task", "shared"):
            bank = ClassGaussianKeyBank(feature_dim=x.shape[1], cov_type=cov_type, cov_scope=scope)
            for i in range(0, x.shape[0], 16):
                t = 0 if i < 300 else 1
                bank.update(t, x[i:i + 16], y[i:i + 16])
            ll = torch.stack([bank.log_likelihood(x[:10], t) for t in bank.adapter_ids()], 1)
            assert torch.isfinite(ll).all()


def test_log_likelihood_all_matches_per_adapter():
    x, y = _stream()
    for scope in ("task", "shared"):
        for cov_type in ("full", "diag"):
            bank = ClassGaussianKeyBank(feature_dim=x.shape[1], cov_type=cov_type, cov_scope=scope)
            for i in range(0, x.shape[0], 16):
                bank.update(i // 200, x[i:i + 16], y[i:i + 16])
            per = torch.stack([bank.log_likelihood(x[:20], t) for t in bank.adapter_ids()], 1)
            assert torch.allclose(bank.log_likelihood_all(x[:20]), per, atol=1e-9)


def test_class_gauss_accepts_accelerator_tensors():
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if dev is None:
        return
    x, y = _stream(d=768, n=64)
    ref, acc = ClassGaussianKeyBank(768), ClassGaussianKeyBank(768)
    ref.update(0, x, y)
    acc.update(0, x.to(dev), y.to(dev))
    assert torch.allclose(ref.log_likelihood(x[:5], 0), acc.log_likelihood(x[:5].to(dev), 0))


def _vit_setup():
    import models.model_register  # noqa: F401
    from timm.models import create_model
    from lib.config import cfg
    cfg.device = torch.device("cpu")
    torch.manual_seed(0)
    model = create_model("vit_base_patch16_224", pretrained=True, num_classes=10, drop_rate=0.0,
                         drop_path_rate=0.0, drop_block_rate=None, head_type="prompt")
    for n, p in model.named_parameters():
        p.requires_grad = "head" in n
    return cfg, model


def test_cached_head_training_is_identical():
    import copy
    from lib.train import train_one_batch
    cfg, model = _vit_setup()
    cfg.dtask.epochs = 3
    cfg.dtask.nb_classes = 10
    x, y = torch.rand(4, 3, 224, 224), torch.tensor([1, 3, 3, 7])
    heads = []
    for cache in (False, True):
        m = copy.deepcopy(model)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        train_one_batch(m, torch.nn.CrossEntropyLoss(), x, y, opt, torch.device("cpu"), max_norm=1.0,
                        class_mask=[1, 3, 7], cache_features=cache)
        heads.append(m.head.weight.detach().clone())
    assert torch.allclose(heads[0], heads[1], atol=1e-6), (heads[0] - heads[1]).abs().max()


def test_fast_solve_matches_original_tangent_update():
    import copy
    from torch.func import jacrev, functional_call
    from lib.ia3 import collect_ia3_params
    from lib.train import tangent_tuning_a_batch
    cfg, model = _vit_setup()
    cfg.dtask.nb_classes = 10

    class Net(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, X, target):
            p = self.model(X)["logits"].softmax(dim=1)
            return torch.gather(p, 1, target.reshape(-1, 1)).squeeze(1)

    x, y = torch.rand(4, 3, 224, 224), torch.tensor([1, 3, 3, 7])
    backbone = copy.deepcopy(model)
    net_g = Net(backbone).eval()
    for p in net_g.parameters():
        p.requires_grad = False
    params = collect_ia3_params(net_g)
    jac = jacrev(lambda prm, X, t: functional_call(net_g, prm, (X, t)), argnums=0)
    out = []
    for fast in (False, True):
        G = {k: torch.zeros(p.numel(), p.numel()) for k, p in params.items()}
        g = {k: torch.zeros(p.numel()) for k, p in params.items()}
        upd = tangent_tuning_a_batch(x, y, G, g, {}, net_g, jac, copy.deepcopy(model), params, backbone,
                                     eta=0.03, lambda_=1.0, fast_solve=fast)
        out.append(upd)
    for k in out[0]:
        assert torch.allclose(out[0][k], out[1][k], atol=1e-5, rtol=1e-3), (k, (out[0][k] - out[1][k]).abs().max())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
