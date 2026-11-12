"""Device helpers so the pipeline runs on CUDA, Apple MPS, or CPU.

The original code hard-coded ``cuda`` (device, autocast, synchronize,
empty_cache). On CUDA these helpers behave exactly as before.
"""
from __future__ import annotations

import contextlib

import torch


def resolve_device(preference: str = "auto", gpu_ids: str = "0") -> torch.device:
    gpu = int(gpu_ids.split(",")[0])
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device(f"cuda:{gpu}")
    if preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device):
    """fp16 autocast on CUDA (as in the original), no-op elsewhere."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
