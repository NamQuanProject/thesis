import itertools
import math

import pytest
import torch

from lib.mass_specialists import GaussianAnchor, predict_specialists, support_budget


def fixture():
    g = torch.Generator().manual_seed(31)
    scopes = [[0, 1], [1, 2, 3], [3, 4], [5], [0, 4]]
    p0 = torch.rand(43, 6, generator=g, dtype=torch.float64).softmax(1)
    qs = [torch.randn(43, len(s), generator=g, dtype=torch.float64).softmax(1) for s in scopes]
    return p0, scopes, qs


def test_overlapping_operator_normalization_floor_and_direct_formula():
    p0, scopes, qs = fixture()
    incidence, eta = support_budget(scopes, 6, .8)
    assert (incidence.T @ eta <= .8 + 1e-14).all()
    direct = p0.clone()
    for e, s in enumerate(scopes):
        direct[:, s] += eta[e] * (p0[:, s].sum(1, keepdim=True) * qs[e] - p0[:, s])
    p, _ = predict_specialists(p0, scopes, lambda e, ids: qs[e][ids])
    torch.testing.assert_close(p, direct)
    torch.testing.assert_close(p.sum(1), torch.ones(len(p), dtype=p.dtype))
    assert (p >= .2*p0 - 1e-14).all()
    assert ((p0.log() - p.log()) <= -math.log(.2) + 1e-14).all()


def test_wrong_specialist_cannot_exclude_uncovered_true_label():
    p0 = torch.tensor([[.15, .25, .6]], dtype=torch.float64)
    p, _ = predict_specialists(p0, [[0, 1]], lambda e, ids: torch.tensor([[1., 0.]]))
    assert p[0, 2] == p0[0, 2]
    assert torch.isfinite(-p.log()).all()
    # Hard selection of this specialist instead gives p(true label)=0.


def test_singletons_and_disjoint_irrelevant_growth_need_no_global_dilution():
    p0 = torch.tensor([[.3, .2, .4, .1]], dtype=torch.float64)
    qs = [torch.tensor([[.8, .2]]), torch.tensor([[.1, .9]])]
    p, _ = predict_specialists(p0, [[0, 1]], lambda e, ids: qs[e])
    expanded, _ = predict_specialists(p0, [[0, 1], [2, 3]], lambda e, ids: qs[e])
    torch.testing.assert_close(p[:, :2], expanded[:, :2])
    singleton, info = predict_specialists(p0, [[0], [1], [2], [3]],
                                          lambda *a: pytest.fail('singleton is identity'))
    torch.testing.assert_close(singleton, p0)
    assert info['expert_evaluations'].sum() == 0


def test_lazy_certificates_match_full_and_ignore_batch_composition():
    p0, scopes, qs = fixture()
    full, _ = predict_specialists(p0, scopes, lambda e, ids: qs[e][ids])
    lazy, meta = predict_specialists(p0, scopes, lambda e, ids: qs[e][ids], mode='certified')
    assert meta['certified'].all()
    assert torch.equal(lazy.argmax(1), full.argmax(1))
    for i in (0, 2, 19):
        single, info = predict_specialists(p0[i:i+1], scopes, lambda e, ids: qs[e][ids+i], mode='certified')
        torch.testing.assert_close(single[0], lazy[i])
        assert info['expert_evaluations'][0] == meta['expert_evaluations'][i]
    # Budgeted decisions are NOT advertised as exact without a certificate.
    budget, info = predict_specialists(p0, scopes, lambda e, ids: qs[e][ids], mode='budget', max_experts=1)
    assert (info['expert_evaluations'] <= 1).all()
    good = info['certified']
    assert torch.equal(budget[good].argmax(1), full[good].argmax(1))


def test_zero_call_certificate_is_valid_for_every_extreme_completion():
    p0 = torch.tensor([[.99, .005, .005]], dtype=torch.float64)
    scopes = [[0, 1], [1, 2]]
    lazy, info = predict_specialists(p0, scopes, lambda *a: pytest.fail('should certify without forward'),
                                     alpha=.1, mode='certified')
    assert info['expert_evaluations'][0] == 0 and info['certified'][0]
    for choices in itertools.product((0, 1), repeat=2):
        def extreme(e, ids):
            return torch.eye(2, dtype=torch.float64)[choices[e]:choices[e]+1]
        full, _ = predict_specialists(p0, scopes, extreme, alpha=.1)
        assert torch.equal(lazy.argmax(1), full.argmax(1))


def test_anchor_additivity_checkpoint_and_degenerate_features():
    g = torch.Generator().manual_seed(15)
    x = torch.randn(50, 8, generator=g)
    y = torch.arange(50) % 5
    a, b = GaussianAnchor(8, 6), GaussianAnchor(8, 6)
    a.update(x, y)
    for ids in torch.arange(49, -1, -1).split(7):
        b.update(x[ids], y[ids])
    torch.testing.assert_close(a.probabilities(x), b.probabilities(x))
    restored = GaussianAnchor.from_state_dict(a.state_dict())
    torch.testing.assert_close(restored.probabilities(x), a.probabilities(x))
    a = GaussianAnchor(8, 6)
    a.update(torch.ones(1, 8), torch.tensor([2]))
    assert torch.isfinite(a.probabilities(torch.ones(3, 8)).log()).all()


def test_bad_probabilities_and_scopes_rejected():
    p0 = torch.tensor([[.5, .5]])
    for scopes in ([[0, 0]], [[]], [[3]]):
        with pytest.raises(ValueError):
            predict_specialists(p0, scopes, lambda *a: None)
    with pytest.raises(ValueError):
        predict_specialists(p0, [[0, 1]], lambda *a: torch.tensor([[1., 1.]]))
