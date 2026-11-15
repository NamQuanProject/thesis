import os
import numpy as np
import torch
import random
import argparse
import pprint
import copy
import json
from itertools import islice
from pathlib import Path
import models.model_register

from lib.config import cfg, cfg_from_file, cfg_from_list, validate_config
from lib.utils import *
from lib.device import resolve_device
from lib.ia3 import apply_ia3_updates, reset_ia3_to_identity
from lib.checkpointing import task_checkpoint_dir, load_tensor_state
from lib.experiment import loader_manifest, write_json
from lib.task_keys import gmm_mahal_min, gmm_log_likelihood, ClassGaussianKeyBank
from lib.mass_specialists import GaussianAnchor
from lib.specialist_evaluation import evaluate_specialist_bank
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


def _build_vit_model(pretrained=None):
    model = create_model(
        cfg.dtask.model,
        pretrained=cfg.dtask.pretrained if pretrained is None else pretrained,
        num_classes=cfg.dtask.nb_classes,
        drop_rate=cfg.dtask.drop,
        drop_path_rate=cfg.dtask.drop_path,
        drop_block_rate=None,
        head_type=cfg.dtask.head_type,
    )
    model.to(cfg.device)
    model.eval()
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


@torch.no_grad()
def evaluate_bank(model, adapters, key_bank, data_loader, class_mask, current_real_task, acc_matrix, cfg,
                  seen_task_ids=None, history_available=True, anchor=None, prediction_records=None):
    """Route every test batch of the streams seen so far and classify it.

    `adapters[t]` holds the IA3 update (`ia3`) and GMM task key (`gmm`) of
    detected distribution t. With default `cfg.improve` flags this computes
    exactly the published protocol (adapter-space min-Mahalanobis routing,
    one adapter per test batch); IA3 vectors are swapped into a single model
    instead of deep-copying one ViT per adapter.
    """
    imp = cfg.improve
    if imp.prediction == 'mass':
        return evaluate_specialist_bank(model, adapters, anchor, data_loader, class_mask,
                                         current_real_task, acc_matrix, cfg, seen_task_ids, history_available, prediction_records)
    model.eval()
    device = cfg.device
    T = len(adapters)
    if not T:
        raise ValueError('Cannot evaluate an empty adapter bank')
    seen_task_ids = list(range(current_real_task + 1)) if seen_task_ids is None else list(seen_task_ids)
    if len(seen_task_ids) != current_real_task + 1 or len(set(seen_task_ids)) != len(seen_task_ids):
        raise ValueError('seen_task_ids must contain each observed task once, in observation order')
    need_z0 = imp.routing_space == 'pretrained' or imp.fusion_alpha > 0
    not_masks = {}
    for t in range(T):
        nm = np.setdiff1d(np.arange(cfg.dtask.nb_classes), class_mask[t])
        not_masks[t] = torch.tensor(nm, dtype=torch.int64, device=device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    stat_matrix = np.zeros((3, cfg.continual.n_tasks))
    batch_hits, batch_total, sample_hits, sample_total = [], [], [], []

    for tid in seen_task_ids:
        metric_logger = MetricLogger(delimiter="  ")
        bh = bt = sh = st = 0
        batches = data_loader[tid]['val']
        if cfg.runtime.max_eval_batches:
            batches = islice(batches, cfg.runtime.max_eval_batches)
        for input, target in batches:
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
            # Detected segment IDs need not match ground-truth task indices.
            # Measure eligibility of the selected expert's class mask instead.
            labels = target.cpu().tolist()
            bh += int(all(y in class_mask[batch_pred] for y in labels)); bt += 1
            sh += sum(y in class_mask[t] for y, t in zip(labels, sample_pred.tolist())); st += B

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
            acc1, acc5 = accuracy(logits, target, topk=(1, min(5, logits.shape[1])))
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=B)

        if not st:
            raise ValueError(f'Task {tid} has no evaluation samples')
        log(f"Task {tid}: selected-adapter label coverage (batch) {bh / bt:.4f} | (sample) {sh / st:.4f}")
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

    task_id = current_real_task
    route_batch = sum(batch_hits) / max(sum(batch_total), 1)
    route_sample = sum(sample_hits) / max(sum(sample_total), 1)
    log(f"Accuracy of matching: {route_batch}")
    log(f"Accuracy of matching according to each task: {[h / max(n, 1) for h, n in zip(batch_hits, batch_total)]}")
    log(f"Per-sample matching accuracy: {route_sample}")

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2])
    previous = seen_task_ids[:-1]
    forgetting = float(np.mean([acc_matrix[t, :task_id + 1].max() - acc_matrix[t, task_id]
                                for t in previous])) if previous else 0.0
    backward = float(np.mean([acc_matrix[t, task_id] - acc_matrix[t, j]
                              for j, t in enumerate(previous)])) if previous else 0.0
    if not history_available:
        forgetting = backward = None
    result_str += f"\tForgetting: {forgetting}\tBackward: {backward}"
    log(result_str)
    return {'acc_matrix': acc_matrix, 'avg_acc': avg_stat[0], 'forgetting': forgetting,
            'routing_batch': route_batch, 'routing_sample': route_sample}


def run_evaluation(history_available=True, prediction_records=None):
    """Re-evaluate a finished run from its checkpoints (all streams, final head)."""
    log('\nRunning evaluation using TANGENT FINETUNING')
    set_seed(cfg.seed)
    data_loader, source_masks = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    bundle_path = Path(cfg.dtask.output_dir) / 'evaluation_checkpoint.pth'
    if bundle_path.exists():
        bundle = load_tensor_state(bundle_path)
        if bundle['version'] not in (1, 2):
            raise ValueError('Unsupported evaluation checkpoint version')
        stream = loader_manifest(data_loader, source_masks, bundle['stream']['task_order'])
        if stream != bundle['stream']:
            raise ValueError('Evaluation dataset or class order differs from the training stream')
        model = _build_vit_model(pretrained=False)
        model.load_state_dict(load_tensor_state(Path(cfg.dtask.output_dir) / 'backbone.pth'))
        with torch.no_grad():
            model.head.weight.copy_(bundle['head_weight'])
            model.head.bias.copy_(bundle['head_bias'])
        keys = None if bundle['key_bank'] is None else ClassGaussianKeyBank.from_state_dict(bundle['key_bank'])
        anchor = None if bundle.get('anchor') is None else GaussianAnchor.from_state_dict(bundle['anchor'])
        return evaluate_bank(model, bundle['adapters'], keys, data_loader, bundle['class_mask'],
                             len(bundle['seen_tasks']) - 1, bundle['acc_matrix'].numpy().copy(), cfg,
                             seen_task_ids=bundle['seen_tasks'], history_available=history_available, anchor=anchor,
                             prediction_records=prediction_records)

    class_mask = load_tensor_state(os.path.join(cfg.dtask.output_dir, 'class_mask.pth'))
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    count_save = len(class_mask)
    vit_model = _build_vit_model()

    last_ckpt = task_checkpoint_dir(cfg.dtask.output_dir, count_save - 1)
    head_weight = load_tensor_state(os.path.join(last_ckpt, 'head_weight.pth'))
    head_bias = load_tensor_state(os.path.join(last_ckpt, 'head_bias.pth'))
    with torch.no_grad():
        vit_model.head.weight.copy_(head_weight)
        vit_model.head.bias.copy_(head_bias)

    adapters = []
    prefix = 'route_' if (cfg.improve.routing_key == 'gmm' and cfg.improve.routing_space == 'pretrained') else ''
    for task_id in range(count_save):
        ckpt = task_checkpoint_dir(cfg.dtask.output_dir, task_id)
        adapters.append({
            'ia3': load_tensor_state(os.path.join(ckpt, 'global_updates.pth')),
            'gmm': tuple(load_tensor_state(os.path.join(ckpt, prefix + f)) for f in ('mean.pth', 'vars.pth', 'counts.pth')),
        })
    key_bank = None
    key_path = os.path.join(cfg.dtask.output_dir, 'key_bank.pth')
    if cfg.improve.routing_key == 'class_gauss':
        key_bank = ClassGaussianKeyBank.from_state_dict(load_tensor_state(key_path))

    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    return evaluate_bank(vit_model, adapters, key_bank, data_loader, class_mask, cfg.continual.n_tasks - 1,
                         acc_matrix, cfg, history_available=False)


def main():
    parser = argparse.ArgumentParser(description='TANGENT FINETUNING in Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default=None, help='config for legacy checkpoints')
    parser.add_argument('--run-dir', help='load the exact saved configuration and evaluation checkpoint')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')

    args = parser.parse_args()

    if args.run_dir:
        cfg_from_file(Path(args.run_dir) / 'config.json')
        cfg.dtask.output_dir = str(Path(args.run_dir).resolve())
    elif args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    else:
        parser.error('supply --run-dir (recommended) or --cfg for legacy checkpoints')
    cfg_from_list(args.set)
    validate_config()

    log(pprint.pformat(cfg))

    cfg.device = resolve_device(cfg.improve.device, cfg.gpu_ids)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_seed(cfg.seed)

    if cfg.continual.method.run_merlin:
        result = run_evaluation(history_available=not bool(args.set))
        write_json(Path(cfg.dtask.output_dir) / 'reevaluation.json',
                   {k: v for k, v in result.items() if k != 'acc_matrix'})

if __name__ == '__main__':
    main()
