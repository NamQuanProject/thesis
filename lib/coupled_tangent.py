"""Research candidate: a coupled residual tangent fit with bounded sketch state.

Gaussian row sketching and ridge regression are established tools. The research
question is whether correcting cross-block interference and the target offset
improves this task-free adapter pipeline. No novelty/performance claim is implied.
Only derivative summaries are stored; no images are retained or replayed.
"""
import math

import torch
from timm.utils import accuracy

from lib.ia3 import apply_ia3_updates
from lib.utils import log


class CoupledTangentState:
    def __init__(self, parameters, rank=128, seed=0):
        if rank < 1:
            raise ValueError('Sketch rank must be positive')
        self.shapes = {name: p.shape for name, p in parameters.items()}
        self.sizes = {name: p.numel() for name, p in parameters.items()}
        first = next(iter(parameters.values()))
        self.design = first.new_zeros(rank, sum(self.sizes.values()))
        self.target = first.new_zeros(rank)
        # A separate generator prevents the sketch from changing sample order,
        # augmentation, GMM initialization, or model initialization RNGs.
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.samples = 0

    def reset(self):
        self.design.zero_()
        self.target.zero_()
        self.samples = 0

    def flatten(self, jacobian):
        return torch.cat([jacobian[name].reshape(-1, size) for name, size in self.sizes.items()], dim=1)

    def unpack(self, vector):
        return {name: part.reshape(self.shapes[name]) for name, part in
                zip(self.sizes, vector.split(list(self.sizes.values())))}

    @torch.no_grad()
    def update(self, design, target, projection=None):
        # Supplying a projection makes fixed-design equivalence reproducible in
        # tests. Production draws fresh independent Gaussian columns per sample.
        rank = self.design.shape[0]
        if projection is None:
            projection = torch.randn(rank, design.shape[0], generator=self.generator,
                                      dtype=design.dtype).to(design.device) / math.sqrt(rank)
        if projection.shape != (rank, design.shape[0]) or target.shape != (design.shape[0],):
            raise ValueError('Incompatible design, target, or projection')
        self.design.add_(projection @ design)
        self.target.add_(projection @ target)
        self.samples += design.shape[0]

    @torch.no_grad()
    def solve(self, ridge):
        if ridge <= 0:
            raise ValueError('Ridge must be positive')
        gram = self.design @ self.design.T
        gram.diagonal().add_(ridge)
        device = gram.device
        # MPS lacks some factorizations; CPU and CUDA stay on their own device.
        local = gram.cpu() if device.type == 'mps' else gram
        chol = torch.linalg.cholesky(local)
        dual = torch.cholesky_solve(self.target.to(local.device)[:, None], chol).to(device)[:, 0]
        return self.unpack(self.design.T @ dual)


def tune_coupled_batch(input, target, probabilities, jacobian, model, state,
                       max_epochs, patience, eta, ridge, select_metric, select_tol,
                       residual_target=True):
    """Fit all IA3 blocks jointly; select targets using training observations only."""
    design = state.flatten(jacobian)
    gram = design @ design.T
    gram.diagonal().add_(ridge)
    local = gram.cpu() if gram.device.type == 'mps' else gram
    chol = torch.linalg.cholesky(local)
    z = probabilities.clone()
    endpoint = 2 - probabilities
    best_score = 0.0 if select_metric == 'acc' else -float('inf')
    best_target = torch.zeros_like(probabilities)
    stale = 0
    best_iter = -1
    for epoch in range(max_epochs):
        z = (1 - eta) * z + eta * endpoint
        response = z - probabilities if residual_target else z
        dual = torch.cholesky_solve(response.to(chol.device)[:, None], chol).to(design.device)[:, 0]
        apply_ia3_updates(model, state.unpack(design.T @ dual), design.device)
        with torch.no_grad():
            logits = model(input)['logits']
            score = float(accuracy(logits, target, topk=(1,))[0]) if select_metric == 'acc' \
                else -float(torch.nn.functional.cross_entropy(logits.float(), target))
        if score > best_score + select_tol:
            best_score, best_iter, stale = score, epoch, 0
            best_target = response.detach().clone()
        else:
            stale += 1
            if stale >= patience:
                break
    state.update(design, best_target)
    log(f'Coupled refinement: selected iterate {best_iter}, score={best_score:.4f}, '
        f'sketch_samples={state.samples}, rank={state.design.shape[0]}, residual={residual_target}')
    return state.solve(ridge)
