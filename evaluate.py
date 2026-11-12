import os
import numpy as np
import torch
import random
import argparse
import pprint
import copy
import models.model_register

from lib.config import cfg, cfg_from_file
from lib.utils import *
from lib.device import resolve_device
from lib.ia3 import apply_ia3_updates, reset_ia3_to_identity
from lib.checkpointing import task_checkpoint_dir
from lib.task_keys import gmm_mahal_min, gmm_log_likelihood, ClassGaussianKeyBank
from data.dataset import *
from utils.vision import *
from timm.models import create_model
from timm.utils import accuracy


def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)


def _build_vit_model():
    model = create_model(
        cfg.dtask.model,
        pretrained=cfg.dtask.pretrained,
        num_classes=cfg.dtask.nb_classes,
        drop_rate=cfg.dtask.drop,
        drop_path_rate=cfg.dtask.drop_path,
        drop_block_rate=None,
        head_type=cfg.dtask.head_type,
    )
    model.to(cfg.device)
    return model


def _routing_scores(adapter_id, feats, adapter, key_bank, cfg):
    """Per-sample routing score of one adapter (lower = better), on CPU."""
    imp = cfg.improve
    feats = feats.float().cpu()
    if imp.routing_key == 'class_gauss':
        return -key_bank.log_likelihood(feats, adapter_id).float()
    means, variances, counts = adapter['gmm']
    if imp.gmm_score == 'nll':
        return -gmm_log_likelihood(feats, means, variances, counts, var_shrink=imp.gmm_var_shrink).float()
    return gmm_mahal_min(feats, means, variances, counts)


def _fuse_with_class_gauss(logits, z0, adapter_id, key_bank, alpha, device):
    """Product-of-experts fusion of the (already task-masked) head logits with
    the class-conditional Gaussian posterior of the same adapter."""
    ll, classes = key_bank.class_log_likelihood(z0, adapter_id)
    ll = ll.float().to(device)
    cls = torch.tensor(classes, dtype=torch.int64, device=device)
    fused = torch.full_like(logits, float('-inf'))
    head_lp = torch.log_softmax(logits[:, cls], dim=1)
    gauss_lp = torch.log_softmax(ll, dim=1)
    fused[:, cls] = head_lp + alpha * gauss_lp
    return fused


def _match_original_rng_consumption(n_adapters, current_real_task):
    """Keep the global torch RNG where the original evaluator would leave it.

    Every `iter(DataLoader)` draws one int64 base seed from the global RNG. The
    original evaluator iterated each test loader once per adapter (routing pass)
    plus once more (classification pass): (T + 1)(k + 1) iterators per
    evaluation; `evaluate_bank` creates k + 1. Making up the difference keeps
    the next tasks' training stream (sampling order, augmentation) identical to
    the original code for the same seed, so `base` reproduces original runs and
    every variant sees the same stream (paired comparisons).
    """
    for _ in range(n_adapters * (current_real_task + 1)):
        torch.empty((), dtype=torch.int64).random_()


@torch.no_grad()
def evaluate_bank(model, adapters, key_bank, data_loader, class_mask, current_real_task, acc_matrix, cfg):
    """Route every test batch of the streams seen so far and classify it.

    `adapters[t]` holds the IA3 update (`ia3`) and GMM task key (`gmm`) of
    detected distribution t. With default `cfg.improve` flags this computes
    exactly the published protocol (adapter-space min-Mahalanobis routing,
    one adapter per test batch); IA3 vectors are swapped into a single model
    instead of deep-copying one ViT per adapter.
    """
    imp = cfg.improve
    device = cfg.device
    T = len(adapters)
    need_z0 = imp.routing_space == 'pretrained' or imp.fusion_alpha > 0
    not_masks = {}
    for t in range(T):
        nm = np.setdiff1d(np.arange(cfg.dtask.nb_classes), class_mask[t])
        not_masks[t] = torch.tensor(nm, dtype=torch.int64, device=device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    stat_matrix = np.zeros((3, cfg.continual.n_tasks))
    batch_hits, batch_total, sample_hits, sample_total = [], [], [], []

    for tid in range(current_real_task + 1):
        metric_logger = MetricLogger(delimiter="  ")
        bh = bt = sh = st = 0
        for input, target in data_loader[tid]["val"]:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            B = target.shape[0]

            z0 = None
            if need_z0:
                reset_ia3_to_identity(model, device)
                z0 = model(input)['pre_logits']

            adapter_logits = {}
            if imp.routing_space == 'adapter':
                scores = []
                for t in range(T):
                    apply_ia3_updates(model, adapters[t]['ia3'], device)
                    out = model(input)
                    adapter_logits[t] = out['logits']
                    scores.append(_routing_scores(t, out['pre_logits'], adapters[t], key_bank, cfg))
            elif imp.routing_key == 'class_gauss':
                scores = list((-key_bank.log_likelihood_all(z0.float())).float().unbind(1))
            else:
                scores = [_routing_scores(t, z0, adapters[t], key_bank, cfg) for t in range(T)]
            scores = torch.stack(scores, dim=1)            # [B, T]

            sample_pred = scores.argmin(dim=1)
            batch_pred = int(scores.mean(dim=0).argmin())  # mean score over the batch (original)
            pred = sample_pred if imp.routing_unit == 'sample' else torch.full((B,), batch_pred, dtype=torch.int64)
            bh += int(batch_pred == tid); bt += 1
            sh += int((sample_pred == tid).sum()); st += B

            logits = torch.empty(B, cfg.dtask.nb_classes, device=device)
            for t in pred.unique().tolist():
                idx = (pred == t).to(device)
                if t in adapter_logits:
                    lt = adapter_logits[t][idx]
                else:
                    apply_ia3_updates(model, adapters[t]['ia3'], device)
                    lt = model(input[idx])['logits']
                lt = lt.index_fill(dim=1, index=not_masks[t], value=float('-inf'))
                if imp.fusion_alpha > 0 and key_bank is not None:
                    lt = _fuse_with_class_gauss(lt, z0[idx], t, key_bank, imp.fusion_alpha, device)
                logits[idx] = lt

            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=B)

        log(f"Task {tid}: routing acc (batch) {bh / max(bt, 1):.4f} | routing acc (sample) {sh / max(st, 1):.4f}")
        batch_hits.append(bh); batch_total.append(bt); sample_hits.append(sh); sample_total.append(st)

        metric_logger.synchronize_between_processes()
        log('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                    losses=metric_logger.meters['Loss']))
        test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stat_matrix[0, tid] = test_stats['Acc@1']
        stat_matrix[1, tid] = test_stats['Acc@5']
        stat_matrix[2, tid] = test_stats['Loss']
        acc_matrix[tid, current_real_task] = test_stats['Acc@1']

    _match_original_rng_consumption(T, current_real_task)

    task_id = current_real_task
    route_batch = sum(batch_hits) / max(sum(batch_total), 1)
    route_sample = sum(sample_hits) / max(sum(sample_total), 1)
    log(f"Accuracy of matching: {route_batch}")
    log(f"Accuracy of matching according to each task: {[h / max(n, 1) for h, n in zip(batch_hits, batch_total)]}")
    log(f"Per-sample matching accuracy: {route_sample}")

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    diagonal = np.diag(acc_matrix)
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2])
    forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
    backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
    result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    log(result_str)
    return {'acc_matrix': acc_matrix, 'avg_acc': avg_stat[0], 'forgetting': forgetting,
            'routing_batch': route_batch, 'routing_sample': route_sample}


def run_evaluation():
    """Re-evaluate a finished run from its checkpoints (all streams, final head)."""
    log('\nRunning evaluation using TANGENT FINETUNING')
    data_loader, _ = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    class_mask = torch.load(os.path.join(cfg.dtask.output_dir, 'class_mask.pth'))
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    count_save = len(class_mask)
    vit_model = _build_vit_model()

    last_ckpt = task_checkpoint_dir(cfg.dtask.output_dir, count_save - 1)
    head_weight = torch.load(os.path.join(last_ckpt, 'head_weight.pth'))
    head_bias = torch.load(os.path.join(last_ckpt, 'head_bias.pth'))
    with torch.no_grad():
        vit_model.head.weight.copy_(head_weight)
        vit_model.head.bias.copy_(head_bias)

    adapters = []
    prefix = 'route_' if (cfg.improve.routing_key == 'gmm' and cfg.improve.routing_space == 'pretrained') else ''
    for task_id in range(count_save):
        ckpt = task_checkpoint_dir(cfg.dtask.output_dir, task_id)
        adapters.append({
            'ia3': torch.load(os.path.join(ckpt, 'global_updates.pth')),
            'gmm': tuple(torch.load(os.path.join(ckpt, prefix + f)) for f in ('mean.pth', 'vars.pth', 'counts.pth')),
        })
    key_bank = None
    key_path = os.path.join(cfg.dtask.output_dir, 'key_bank.pth')
    if cfg.improve.routing_key == 'class_gauss':
        key_bank = ClassGaussianKeyBank.from_state_dict(torch.load(key_path, weights_only=False))

    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    return evaluate_bank(vit_model, adapters, key_bank, data_loader, class_mask, cfg.continual.n_tasks - 1,
                         acc_matrix, cfg)


def main():
    parser = argparse.ArgumentParser(description='TANGENT FINETUNING in Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/cifar-100.yml')
    parser.add_argument('--portion', '-p',  type=int, default=1)

    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    log(pprint.pformat(cfg))

    cfg.device = resolve_device(cfg.improve.device, cfg.gpu_ids)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    set_seed(cfg.seed)

    if cfg.continual.method.run_merlin:
        run_evaluation()

if __name__ == '__main__':
    main()
