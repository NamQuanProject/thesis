import math
import sys

import torch
import torch.nn.functional as F

from lib.utils import *
from utils.vision import *
from lib.device import synchronize, empty_cache
from lib.ia3 import apply_ia3_updates
from timm.utils import accuracy


def _spd_solve(M, b, lambda_):
    """Solve (M + lambda I) x = b for symmetric PSD ``M`` (Cholesky, LU fallback).

    Linear algebra runs on CPU for MPS tensors (not all factorizations are
    implemented there); CUDA/CPU tensors stay on their device.
    """
    dev = M.device
    if dev.type == "mps":
        M, b = M.cpu(), b.cpu()
    R = M.clone()
    R.diagonal().add_(lambda_)
    L, info = torch.linalg.cholesky_ex(R)
    if int(info) == 0:
        x = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    else:
        x = torch.linalg.solve(R, b)
    return x.to(dev)


def tangent_tuning_a_batch(
    input, target, global_At_A, global_AtA_w, AtA_ridge,
    net_g, jacobian_fn, model, tuning_params, backbone,
    max_epochs=200, patience=3, eta=0.03, lambda_=5e-6,
    fast_solve=False, select_metric="acc", accumulate="merge", select_tol=0.0,
):
    device = cfg.device
    input = input.to(device)
    target = target.to(device)
    batch_size = target.size(0)

    # 1) Base (pretrained) target-class softmax probabilities.
    with torch.no_grad():
        logits = net_g(input, target).detach()
    m = batch_size

    # 2) Jacobian of the target-class probability w.r.t. the IA3 parameters.
    J = jacobian_fn(tuning_params, input, target)
    for k in list(J.keys()):
        J[k] = J[k].detach()

    # 3) Per-parameter-block design matrix and (ridge-regularized) normal
    # equations, reused across the soft-target refinement iterations below.
    # With `fast_solve`, the push-through identity
    #   (A^T A + lambda I)^{-1} A^T = A^T (A A^T + lambda I)^{-1}
    # replaces every d x d solve (d up to 3072) by an m x m one (m = batch size).
    A_cache = {}
    for name, param in tuning_params.items():
        A = J[name].reshape(m, -1)
        d = A.shape[1]
        ATA = A.T @ A
        A_cache[name] = {'A': A, 'ATA': ATA, 'd': d}
        if fast_solve:
            K = A @ A.T
            K.diagonal().add_(lambda_)
            A_cache[name]['K_chol'] = torch.linalg.cholesky(K.cpu() if device.type == "mps" else K)
        else:
            AtA_ridge[name] = ATA + lambda_ * torch.eye(d, device=ATA.device, dtype=ATA.dtype)

    # 4) Fixed-point soft-target refinement: iteratively move the current
    # prediction `z` toward the squared-loss target `b = 2 - p` and track
    # the update that yields the best batch score (early-stopped after
    # `patience` non-improving iterations). The score is batch accuracy in the
    # original method; `select_metric="loss"` uses the (negative) batch
    # cross-entropy instead, which does not saturate at 100% on easy batches.
    z = logits
    b = (2 - z).to(device)
    best_score = 0.0 if select_metric == "acc" else -float("inf")
    num_stop = 0
    early_stop = False
    epoch = 0
    best_updates = {k: torch.zeros(A_cache[k]['d'], device=device) for k in tuning_params.keys()}
    best_target = torch.zeros(m, device=device)
    best_iter = -1

    while not early_stop and epoch < max_epochs:
        z = (1 - eta) * z + eta * b
        vec = z.flatten()

        w_updates = {}
        for name in tuning_params.keys():
            A = A_cache[name]['A']
            if fast_solve:
                L = A_cache[name]['K_chol']
                alpha = torch.cholesky_solve(vec.to(L.device).unsqueeze(-1), L).squeeze(-1).to(device)
                w_updates[name] = A.T @ alpha
            else:
                ATB = A.T @ vec
                w_updates[name] = torch.linalg.solve(AtA_ridge[name], ATB)

        apply_ia3_updates(model, w_updates, device)

        with torch.no_grad():
            out = model(input)["logits"]
            if select_metric == "acc":
                score, _ = accuracy(out, target, topk=(1, 5))
            else:
                score = -F.cross_entropy(out.float(), target).item()

        if score > best_score + select_tol:
            best_score = score
            best_iter = epoch
            num_stop = 0
            best_target = vec.clone()
            for k in w_updates:
                best_updates[k] = w_updates[k].clone()
        else:
            num_stop += 1
            if num_stop >= patience:
                early_stop = True
        epoch += 1

    log(f"Soft-target refinement: {epoch} iterations, selected iterate {best_iter} "
        f"({select_metric} = {float(best_score):.4f})")

    # 5) Fold this batch's statistics into the running global accumulators
    # and resolve the global closed-form update. The released code merges the
    # per-batch ridge solutions, g += A^T A w_b, which equals A^T applied to a
    # ridge-shrunk target A A^T (A A^T + lambda I)^{-1} b_b. `accumulate="joint"`
    # uses the paper's Eq. (5), g += A^T b_b, i.e. the joint ridge regression
    # over every batch of the segment. Both are order-invariant.
    global_updates = {k: torch.zeros(A_cache[k]['d'], device=device) for k in tuning_params.keys()}
    for name in tuning_params.keys():
        ATA = A_cache[name]['ATA']
        global_At_A[name] += ATA
        if accumulate == "joint":
            global_AtA_w[name] += A_cache[name]['A'].T @ best_target
            global_updates[name] = _spd_solve(global_At_A[name], global_AtA_w[name], lambda_)
        elif fast_solve:
            A = A_cache[name]['A']
            global_AtA_w[name] += A.T @ (A @ best_updates[name])   # == ATA @ w, in O(md)
            global_updates[name] = _spd_solve(global_At_A[name], global_AtA_w[name], lambda_)
        else:
            global_AtA_w[name] += ATA @ best_updates[name]
            I = torch.eye(global_At_A[name].shape[1], device=global_At_A[name].device, dtype=global_At_A[name].dtype)
            global_updates[name] = torch.linalg.solve(global_At_A[name] + lambda_ * I, global_AtA_w[name])

    del J, A_cache
    empty_cache(device)
    return global_updates


def train_one_batch(model: torch.nn.Module, criterion, input: torch.Tensor, target: torch.Tensor,
                     optimizer: torch.optim.Optimizer, device: torch.device, max_norm: float = 0,
                     set_training_mode=True, task_id=-1, class_mask=None, cache_features=False):
    """Gradient-based warm start of the classification head on one batch.

    `task_id` is accepted for API/logging symmetry with the rest of the
    pipeline but does not alter training behavior here: task identity is
    unknown in the task-free setting, so no task-conditional logic is
    applied.

    With `cache_features`, the frozen backbone runs once and the E head
    epochs reuse its pre-logits (only the head is trainable and dropout /
    drop-path are 0, so every epoch saw identical features anyway).
    """
    model.train(set_training_mode)
    features = None
    if cache_features:
        with torch.no_grad():
            features = model(input)['pre_logits']

    for _ in range(cfg.dtask.epochs):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if features is not None:
            logits = model(features, fc_only=True)['logits']
        else:
            logits = model(input)['logits']

        # Mask out classes not present in this batch. We cannot mask by task
        # identity (task-free setting), only by the classes actually
        # observed in the current streaming batch.
        if class_mask is not None:
            not_mask = np.setdiff1d(np.arange(cfg.dtask.nb_classes), class_mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        synchronize(device)

        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    metric_logger.synchronize_between_processes()
    log('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
