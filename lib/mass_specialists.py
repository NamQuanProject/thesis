"""Support-local probability corrections and certificates for lazy inference.

Research candidate, not a demonstrated benchmark improvement. Gaussian global
classification and generalist/specialist ensembles are established techniques.
The proposed operator constrains TOTAL outgoing mass per class across overlapping
specialists. Its guarantees concern probabilities and full-model top-1 agreement,
not accuracy against ground truth. See docs/support_specialists.md.
"""
import math
import torch

from lib.task_keys import ClassGaussianKeyBank


class _AnchorBank(ClassGaussianKeyBank):
    def _regularize(self, cov, dof):
        # A fixed numerical floor also covers a one-observation/constant stream.
        cov = cov + 1e-4 * torch.eye(self.D, dtype=cov.dtype)
        return super()._regularize(cov, dof)


class GaussianAnchor:
    """One global streaming LDA classifier; no segment IDs or stored images.

    Existing additive Gaussian machinery is a control, not a novelty claim.
    Uniform class prior, OAS covariance, temperature=1 and uniform probability
    floor=1e-8 are declared before experiments. No test-time fitting.
    """
    def __init__(self, feature_dim, num_classes, temperature=1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('anchor temperature must be positive and finite')
        self.num_classes = num_classes
        self.temperature = temperature
        self.bank = _AnchorBank(feature_dim, class_prior='uniform')

    def update(self, features, labels):
        self.bank.update(0, features, labels)

    def probabilities(self, features):
        scores, classes = self.bank.class_log_likelihood(features, 0)
        probabilities = torch.zeros(len(scores), self.num_classes, dtype=torch.float64)
        probabilities[:, classes] = torch.softmax(scores / self.temperature, dim=1)
        return (1 - 1e-8) * probabilities + 1e-8 / self.num_classes

    def state_dict(self):
        return {'version': 1, 'num_classes': self.num_classes, 'temperature': self.temperature,
                'bank': self.bank.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        if state['version'] != 1:
            raise ValueError('Unsupported anchor state version')
        obj = cls(state['bank']['config']['feature_dim'], state['num_classes'], state['temperature'])
        obj.bank.stats = state['bank']['stats']
        return obj


def support_budget(scopes, num_classes, alpha):
    """Return incidence [E,C] and fixed feasible weights, with M.T @ eta <= alpha.

    Singleton specialists implement identity and require no model execution.
    Disjoint specialists each get alpha; a global mixture would dilute them by E.
    """
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError('alpha must be finite and in [0, 1)')
    incidence = torch.zeros(len(scopes), num_classes, dtype=torch.float64)
    for e, scope in enumerate(scopes):
        if not scope or len(set(scope)) != len(scope) or any(type(c) is not int or not 0 <= c < num_classes for c in scope):
            raise ValueError('each scope must contain distinct valid class indices')
        if len(scope) > 1:
            incidence[e, scope] = 1
    degree = incidence.sum(0)
    eta = torch.zeros(len(scopes), dtype=torch.float64)
    for e, scope in enumerate(scopes):
        if len(scope) > 1:
            eta[e] = alpha / degree[scope].max()
    return incidence, eta


def _check_probabilities(p, shape):
    if p.shape != shape or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('invalid probability array')
    if not torch.allclose(p.sum(1), torch.ones(shape[0], dtype=p.dtype), atol=1e-7, rtol=1e-7):
        raise ValueError('probabilities must sum to one')


@torch.no_grad()
def predict_specialists(p0, scopes, expert, alpha=0.8, mode='full', max_experts=2):
    """Predict each row independently; ``expert(e, row_ids)`` returns q_e on scope e.

    full: evaluate all nontrivial specialists, obtaining the reference distribution.
    certified: stop only when top-1 is determined for EVERY unseen simplex output.
    budget: same certificate, but may stop uncertified at max_experts per image.
    soft_mixture: anchor + scope-mass-weighted conventional mixture (control).
    anchor: global classifier only (control).

    Lazy probabilities complete unevaluated specialists by the anchor's conditional
    distribution. Only their TOP-1 is certified equal to full evaluation; their
    probabilities, top-5, NLL, and Brier loss need not be equal to the full model.
    """
    p0 = p0.detach().cpu().double()
    if p0.ndim != 2 or p0.shape[1] < 2:
        raise ValueError('expected probabilities for at least two classes')
    _check_probabilities(p0, p0.shape)
    if mode not in ('full', 'certified', 'budget', 'anchor', 'soft_mixture'):
        raise ValueError('unknown specialist inference mode')
    if type(max_experts) is not int or max_experts < 0:
        raise ValueError('max_experts must be a nonnegative integer')
    incidence, eta = support_budget(scopes, p0.shape[1], alpha)
    n, classes = p0.shape
    calls = torch.zeros(n, dtype=torch.int64)
    if mode == 'anchor':
        return p0.clone(), {'expert_evaluations': calls, 'certified': torch.zeros(n, dtype=torch.bool)}
    mass = p0 @ incidence.T
    if mode == 'soft_mixture':
        # A common global convex mixture; it does not preserve out-of-scope mass.
        normalizer = mass.sum(1, keepdim=True)
        weights = mass / normalizer.clamp_min(torch.finfo(p0.dtype).tiny)
        result = (1 - alpha) * p0
        rows = torch.arange(n)
        for e, scope in enumerate(scopes):
            if len(scope) < 2 or alpha == 0:
                continue
            q = expert(e, rows).detach().cpu().double()
            _check_probabilities(q, (n, len(scope)))
            result[:, scope] += alpha * weights[:, e, None] * q
            calls += 1
        if alpha == 0:
            result = p0.clone()
        else:
            result[normalizer[:, 0] == 0] = p0[normalizer[:, 0] == 0]
        return result, {'expert_evaluations': calls, 'certified': torch.zeros(n, dtype=torch.bool)}

    coefficients = mass * eta[None]
    remaining = (eta[None] > 0).expand(n, -1).clone()
    lower = p0 * (1 - incidence.T @ eta)[None]
    partial = p0.clone()
    certified = torch.zeros(n, dtype=torch.bool)
    # Sums are float64. A conservative roundoff margin avoids boundary certificates.
    slack = 1e-12 * (1 + len(scopes))
    for _ in range(len(scopes) + 1):
        upper = lower + (coefficients * remaining) @ incidence
        winner = lower.argmax(1)
        other_upper = upper.clone()
        other_upper[torch.arange(n), winner] = -float('inf')
        exhausted = ~remaining.any(1)
        certified |= exhausted | (lower[torch.arange(n), winner] > other_upper.max(1).values + slack)
        active = ~exhausted
        if mode != 'full':
            active &= ~certified
        if mode == 'budget':
            active &= calls < max_experts
        if not active.any():
            break
        priority = coefficients.masked_fill(~remaining, -1)
        selected = priority.argmax(1)
        # Grouping rows changes only execution, never their choice or certificate.
        for e in selected[active].unique().tolist():
            rows = torch.nonzero(active & (selected == e), as_tuple=True)[0]
            scope = scopes[e]
            q = expert(e, rows).detach().cpu().double()
            _check_probabilities(q, (len(rows), len(scope)))
            contribution = coefficients[rows, e, None] * q
            col = torch.tensor(scope)
            lower[rows[:, None], col] += contribution
            partial[rows[:, None], col] += contribution - eta[e] * p0[rows[:, None], col]
            remaining[rows, e] = False
            calls[rows] += 1
    # A completion of the unseen simplexes; no silent renormalization is needed.
    _check_probabilities(partial, (n, classes))
    return partial, {'expert_evaluations': calls, 'certified': certified}
