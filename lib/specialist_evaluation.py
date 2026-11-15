"""Per-image specialist evaluation; source/task labels are used only for reporting."""
import time
from itertools import islice

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from lib.ia3 import apply_ia3_updates, reset_ia3_to_identity
from lib.mass_specialists import predict_specialists
from lib.utils import log


class _Tagged(Dataset):
    def __init__(self, dataset, source):
        self.dataset, self.source = dataset, source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, self.source, index


def evaluation_batches(loaders, seen, cfg):
    cap = cfg.runtime.max_eval_batches
    if not cfg.runtime.mixed_eval:
        for tid in seen:
            offset = 0
            batches = loaders[tid]['val']
            if cap:
                batches = islice(batches, cap)
            for x, y in batches:
                yield x, y, torch.full((len(y),), tid, dtype=torch.long), torch.arange(offset, offset + len(y))
                offset += len(y)
        return
    datasets = []
    for tid in seen:
        loader = loaders[tid]['val']
        data = loader.dataset
        if cap:
            data = Subset(data, range(min(len(data), cap * loader.batch_size)))
        datasets.append(_Tagged(data, tid))
    # The same examples as grouped evaluation; independent reproducible permutation.
    yield from DataLoader(ConcatDataset(datasets), batch_size=cfg.dtask.batch_size, shuffle=True,
                          num_workers=cfg.dtask.num_workers,
                          generator=torch.Generator().manual_seed(cfg.seed + 604021))


@torch.no_grad()
def evaluate_specialist_bank(model, adapters, anchor, loaders, masks, step, acc_matrix, cfg,
                             seen_task_ids=None, history_available=True, prediction_records=None):
    if anchor is None:
        raise ValueError('mass prediction requires a trained global anchor checkpoint')
    if any('head_weight' not in a or 'head_bias' not in a for a in adapters):
        raise ValueError('mass prediction requires frozen per-adapter heads; legacy heads cannot be reconstructed')
    seen = list(range(step + 1)) if seen_task_ids is None else list(seen_task_ids)
    if len(seen) != step + 1 or len(set(seen)) != len(seen):
        raise ValueError('invalid observed task order')
    scopes = [sorted(masks[e]) for e in range(len(adapters))]
    totals = {tid: dict(n=0, correct=0, top5=0, nll=0., brier=0., anchor_correct=0) for tid in seen}
    calls = certified = examples = 0
    started = time.monotonic()
    model.eval()
    for x, y, sources, indices in evaluation_batches(loaders, seen, cfg):
        x = x.to(cfg.device)
        y = y.cpu()
        reset_ia3_to_identity(model, cfg.device)
        p0 = anchor.probabilities(model(x)['pre_logits'])

        def expert(e, rows):
            state = adapters[e]
            apply_ia3_updates(model, state['ia3'], cfg.device)
            model.head.weight.copy_(state['head_weight'])
            model.head.bias.copy_(state['head_bias'])
            logits = model(x[rows.to(x.device)])['logits'][:, scopes[e]]
            # Compute softmax in float64 on CPU (including on MPS).
            return logits.cpu().double().softmax(1)

        p, info = predict_specialists(p0, scopes, expert, cfg.improve.mass_alpha,
                                      cfg.improve.mass_inference, cfg.improve.mass_max_experts)
        pred = p.argmax(1)
        if prediction_records is not None:
            prediction_records.append({'source': sources, 'index': indices, 'target': y, 'prediction': pred,
                                       'expert_evaluations': info['expert_evaluations'],
                                       'certified': info['certified']})
        true_p = p[torch.arange(len(y)), y]
        nll = -true_p.log()
        brier = (p.square().sum(1) - 2 * true_p + 1) / 2
        top5 = (p.topk(min(5, p.shape[1]), dim=1).indices == y[:, None]).any(1)
        for tid in sources.unique().tolist():
            idx = sources == tid
            t = totals[tid]
            t['n'] += int(idx.sum())
            t['correct'] += int((pred[idx] == y[idx]).sum())
            t['top5'] += int(top5[idx].sum())
            t['anchor_correct'] += int((p0[idx].argmax(1) == y[idx]).sum())
            t['nll'] += float(nll[idx].sum())
            t['brier'] += float(brier[idx].sum())
        examples += len(y)
        calls += int(info['expert_evaluations'].sum())
        certified += int(info['certified'].sum())
    for tid, t in totals.items():
        if not t['n']:
            raise ValueError(f'Task {tid} has no evaluation examples')
        acc_matrix[tid, step] = 100 * t['correct'] / t['n']
        log(f"Task {tid}: specialist acc {acc_matrix[tid, step]:.3f} | "
            f"anchor acc {100*t['anchor_correct']/t['n']:.3f} | NLL {t['nll']/t['n']:.5f}")
    forgetting = float(np.mean([acc_matrix[t, :step + 1].max() - acc_matrix[t, step]
                                for t in seen[:-1]])) if step else 0.
    # A post-hoc mode/batch-protocol override cannot reuse another model's history.
    if not history_available:
        forgetting = None
    result = {'acc_matrix': acc_matrix, 'avg_acc': float(np.mean([acc_matrix[t, step] for t in seen])),
              'forgetting': forgetting, 'routing_batch': None, 'routing_sample': None,
              'anchor_avg_acc': float(np.mean([100*t['anchor_correct']/t['n'] for t in totals.values()])),
              'nll': sum(t['nll'] for t in totals.values()) / examples,
              'brier': sum(t['brier'] for t in totals.values()) / examples,
              'mean_expert_evaluations': calls / examples,
              'mean_backbone_evaluations': 1 + calls / examples,
              'certified_fraction': certified / examples, 'eval_samples': examples,
              'eval_seconds': time.monotonic() - started,
              'prediction': 'mass', 'inference': cfg.improve.mass_inference,
              'batch_protocol': 'mixed' if cfg.runtime.mixed_eval else 'grouped',
              'probability_semantics': ('anchor_completion' if cfg.improve.mass_inference in ('budget', 'certified')
                                        else cfg.improve.mass_inference)}
    log(f"SPECIALISTS | acc {result['avg_acc']:.4f} | anchor {result['anchor_avg_acc']:.4f} | "
        f"experts/image {calls/examples:.4f} | top-1 certified {certified/examples:.4f}")
    return result
