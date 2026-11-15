"""Check finite-dimensional identities in docs/research_next.md using NumPy.

No model fitting, dataset, network, or GPU is used. These deterministic numerical
checks neither prove the statements nor test statistical coverage or benchmark gains.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def brier(probabilities, targets):
    return 0.5 * np.square(probabilities - targets).sum(axis=-1)


def check(seed=15):
    rng = np.random.default_rng(seed)
    n, classes = 10000, 5
    old = rng.dirichlet(np.ones(classes), size=n)
    candidate = rng.dirichlet(np.ones(classes), size=n)
    target = np.eye(classes)[rng.integers(classes, size=n)]
    gate = rng.uniform(0, 0.99, size=n)
    mixed = (1 - gate[:, None]) * old + gate[:, None] * candidate
    harm = brier(mixed, target) - brier(old, target)
    envelope = gate / (2 - gate)
    assert np.all(harm <= envelope + 1e-12)

    # An independent extremal family attains the proposed universal envelope.
    alpha = np.linspace(0, 0.999, 1001)
    extremal_old = np.column_stack([1 / (2 - alpha), (1 - alpha) / (2 - alpha)])
    wrong = np.tile([0.0, 1.0], (len(alpha), 1))
    truth = np.tile([1.0, 0.0], (len(alpha), 1))
    extremal_mix = (1 - alpha[:, None]) * extremal_old + alpha[:, None] * wrong
    sharp_error = np.max(np.abs(brier(extremal_mix, truth) - brier(extremal_old, truth)
                               - alpha / (2 - alpha)))
    assert sharp_error < 1e-12

    true_old = (old * target).sum(1)
    true_candidate = (candidate * target).sum(1)
    true_mixed = (mixed * target).sum(1)
    log_harm = np.log(true_old) - np.log(true_mixed)
    assert np.all(log_harm <= -np.log1p(-gate) + 1e-12)

    displacement = candidate - old
    linear_gain = ((target - old) * displacement).sum(1)
    curvature = np.square(displacement).sum(1)
    gain_error = np.max(np.abs(brier(old, target) - brier(mixed, target)
                              - (gate * linear_gain - 0.5 * gate**2 * curvature)))
    assert gain_error < 1e-12

    # Check the gate gradient independently with centered differences.
    interior = rng.uniform(0.05, 0.95, size=n)
    step = 1e-6
    plus = (1 - interior - step) * true_old + (interior + step) * true_candidate
    minus = (1 - interior + step) * true_old + (interior - step) * true_candidate
    numeric_gradient = (-np.log(plus) + np.log(minus)) / (2 * step)
    exact_gradient = (true_old - true_candidate) / (
        (1 - interior) * true_old + interior * true_candidate)
    gradient_error = np.max(np.abs(numeric_gradient - exact_gradient))
    assert gradient_error < 1e-7

    # Hard-router information can cancel while a probabilistic mixture helps.
    # Two equally likely hidden regimes prefer opposing deterministic experts.
    oracle_gain, feature_only_hard_gain = 0.5, 0.0
    conditional_gaps = np.array([1.0, -1.0])
    information_gap = 0.5 * (np.abs(conditional_gaps).mean()
                             - abs(conditional_gaps.mean()))
    assert oracle_gain - feature_only_hard_gain == information_gap
    old_brier = brier(np.tile([1.0, 0.0], (2, 1)), np.eye(2)).mean()
    soft_brier = brier(np.full((2, 2), 0.5), np.eye(2)).mean()
    assert old_brier - soft_brier == 0.25

    return {
        'kind': 'numerical_identity_checks', 'seed': seed, 'random_simplex_cases': n,
        'brier_envelope_violations': int(np.sum(harm > envelope + 1e-12)),
        'brier_sharpness_max_absolute_error': float(sharp_error),
        'log_envelope_violations': int(np.sum(log_harm > -np.log1p(-gate) + 1e-12)),
        'soft_brier_gain_max_absolute_error': float(gain_error),
        'gate_gradient_max_absolute_error': float(gradient_error),
        'hidden_regime_hard_routing_gap': information_gap,
        'same_fixture_soft_brier_improvement': float(old_brier - soft_brier),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualification': 'No training or performance experiment. Numerical checks are not proofs, '
                         'statistical coverage tests, or evidence of novelty.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = json.dumps(check(), indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as handle:
            handle.write(result)
    print(result, end='')


if __name__ == '__main__':
    main()
