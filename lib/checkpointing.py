"""Helpers for saving and loading per-distribution ("task") signatures.

Each detected distribution stores: the ridge accumulators (``global_At_A``,
``global_AtA_w``), the closed-form IA3 update derived from them, and the
Gaussian-mixture task key (means/vars/counts). These were previously saved
and re-loaded with near-identical, copy-pasted code in main.py and
evaluate.py.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def atomic_torch_save(value, filename):
    def cpu_state(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        if isinstance(item, dict):
            return {k: cpu_state(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(cpu_state(v) for v in item)
        return item

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(cpu_state(value), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_tensor_state(filename):
    """Explicit safe tensor loading, independent of PyTorch's default/device."""
    return torch.load(filename, map_location='cpu', weights_only=True)


def task_checkpoint_dir(output_dir: str, task_id: int) -> str:
    """Path to the checkpoint directory for a given detected distribution."""
    return os.path.join(output_dir, f"ia3_task_{task_id}")


def save_task_checkpoint(output_dir, task_id, global_At_A, global_AtA_w, online_gauss,
                          global_updates=None, head_weight=None, head_bias=None,
                          mahal_history=None, route_gauss=None, save_ridge_stats=True):
    """Persist the ridge accumulators and the GMM task key for ``task_id``."""
    ckpt_dir = task_checkpoint_dir(output_dir, task_id)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    if save_ridge_stats:
        atomic_torch_save({k: v.detach().cpu() for k, v in global_At_A.items()}, os.path.join(ckpt_dir, "global_At_A.pth"))
        atomic_torch_save({k: v.detach().cpu() for k, v in global_AtA_w.items()}, os.path.join(ckpt_dir, "global_AtA_w.pth"))
    atomic_torch_save(online_gauss.means, os.path.join(ckpt_dir, "mean.pth"))
    atomic_torch_save(online_gauss.vars, os.path.join(ckpt_dir, "vars.pth"))
    atomic_torch_save(online_gauss.counts, os.path.join(ckpt_dir, "counts.pth"))

    if global_updates is not None:
        atomic_torch_save(global_updates, os.path.join(ckpt_dir, "global_updates.pth"))
    if head_weight is not None:
        atomic_torch_save(head_weight, os.path.join(ckpt_dir, "head_weight.pth"))
    if head_bias is not None:
        atomic_torch_save(head_bias, os.path.join(ckpt_dir, "head_bias.pth"))
    if mahal_history is not None:
        atomic_torch_save(mahal_history, os.path.join(ckpt_dir, "mahal_history.pth"))
    if route_gauss is not None:  # routing-only GMM on pretrained embeddings (improve.routing_space)
        atomic_torch_save(route_gauss.means, os.path.join(ckpt_dir, "route_mean.pth"))
        atomic_torch_save(route_gauss.vars, os.path.join(ckpt_dir, "route_vars.pth"))
        atomic_torch_save(route_gauss.counts, os.path.join(ckpt_dir, "route_counts.pth"))

    return ckpt_dir


def mask_significant_clusters(mean, var, counts, min_count: int = 10):
    """Drop GMM clusters that have fewer than ``min_count`` assigned points.

    Mirrors the original (duplicated) masking logic used everywhere the task
    key was consumed for evaluation/retrieval.
    """
    keep = counts >= min_count
    return mean[keep], var[keep]


def load_task_signature(output_dir, task_id, min_cluster_count: int = 10):
    """Load the masked GMM task key and IA3 ridge solution for ``task_id``."""
    ckpt_dir = task_checkpoint_dir(output_dir, task_id)
    means = load_tensor_state(os.path.join(ckpt_dir, "mean.pth"))
    variances = load_tensor_state(os.path.join(ckpt_dir, "vars.pth"))
    counts = load_tensor_state(os.path.join(ckpt_dir, "counts.pth"))
    mean, var = mask_significant_clusters(means, variances, counts, min_cluster_count)
    global_updates = load_tensor_state(os.path.join(ckpt_dir, "global_updates.pth"))
    return {"mean": mean, "var": var, "global_updates": global_updates}
