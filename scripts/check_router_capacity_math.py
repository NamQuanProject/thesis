"""Finite-population checks of the router-capacity argument in research_next.md.

No trained model, statistical coverage experiment, dataset, or GPU is involved.
All risks below are exact means over declared finite fixtures, not held-out
estimates of an unknown population. The convex dual-gap argument is established.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def risk(predictions, targets):
    return np.mean(0.5 * np.square(predictions - targets).sum(axis=-1))


def two_expert_optimum(first, second, targets):
    direction = second - first
    curvature = np.mean(np.square(direction).sum(axis=-1))
    alignment = np.mean(((targets - first) * direction).sum(axis=-1))
    weight = np.clip(alignment / curvature, 0, 1) if curvature > 0 else 0.0
    return (1 - weight) * first + weight * second


def gap(incumbent, experts, targets):
    return max(float(np.mean(((targets - incumbent) * (expert - incumbent)).sum(axis=-1)))
               for expert in experts)


def check():
    rng = np.random.default_rng(15)
    maximum_violation = 0.0
    for _ in range(1000):
        experts = [rng.dirichlet(np.ones(5), size=32) for _ in range(2)]
        targets = np.eye(5)[rng.integers(5, size=32)]
        weight = rng.uniform()
        incumbent = weight * experts[0] + (1 - weight) * experts[1]
        optimal = two_expert_optimum(*experts, targets)
        dual_gap = gap(incumbent, experts, targets)
        repair_gain = risk(incumbent, targets) - risk(optimal, targets)
        maximum_violation = max(maximum_violation, repair_gain - dual_gap)
        assert dual_gap >= -1e-12 and repair_gain <= dual_gap + 1e-12

        new_expert = rng.dirichlet(np.ones(5), size=32)
        gate = rng.uniform(0, 0.8)
        proposed = (1 - gate) * incumbent + gate * new_expert
        improvement = risk(incumbent, targets) - risk(proposed, targets)
        gain_beyond_best_old = risk(optimal, targets) - risk(proposed, targets)
        assert gain_beyond_best_old >= improvement - dual_gap - 1e-12

    # Repairable routing failure: a perfect expert already exists in the bank.
    target = np.array([[1.0, 0.0]])
    old_experts = [target.copy(), np.array([[0.0, 1.0]])]
    incumbent = np.array([[0.1, 0.9]])
    duplicate = old_experts[0]
    proposed = 0.5 * incumbent + 0.5 * duplicate
    duplicate_gain = risk(incumbent, target) - risk(proposed, target)
    duplicate_certificate = duplicate_gain - gap(incumbent, old_experts, target)
    assert duplicate_gain > 0 and duplicate_certificate < 0

    # Missing capacity within this router family: every old convex mixture is
    # limited to probability <= 0.5 on the correct class.
    old_experts = [np.array([[0.3, 0.7]]), np.array([[0.5, 0.5]])]
    incumbent = old_experts[1]
    proposed = 0.5 * incumbent + 0.5 * target
    capacity_gain = risk(incumbent, target) - risk(proposed, target)
    capacity_certificate = capacity_gain - gap(incumbent, old_experts, target)
    assert abs(capacity_certificate - 0.1875) < 1e-12

    return {
        'kind': 'finite_population_identity_checks', 'random_comparisons': 1000,
        'maximum_dual_bound_violation': float(maximum_violation),
        'duplicate_expert_gain_over_bad_router': float(duplicate_gain),
        'duplicate_expert_capacity_lower_bound': float(duplicate_certificate),
        'useful_new_expert_capacity_lower_bound': float(capacity_certificate),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualification': 'Checks known convex geometry on explicit finite populations. '
                         'No statistical coverage, learned-model performance, or novelty is established.',
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
