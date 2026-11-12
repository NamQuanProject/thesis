"""Feature-level routing probe: compare task-key variants in minutes instead of full runs.

It streams the benchmark exactly like main.py (same class order, task splits,
batch size, augmentation), extracts *pretrained* ViT embeddings (IA3 = identity)
once, caches them, and then fits / scores every task-key variant offline:

  * original online diagonal GMM (lib.signatures.OnlineDiagonalGMM, same
    hyper-parameters as the config) scored by min-Mahalanobis (published) or
    by log-likelihood (+ variance shrinkage);
  * class-conditional Gaussian keys (lib.task_keys.ClassGaussianKeyBank) with
    diag/full covariance, task/shared pooling, OAS/fixed shrinkage;
  * keys fit on the augmented training view vs. the clean (test-transform) view.

Task boundaries are taken as given (oracle) so that only the key/score is
compared; routing is evaluated with the paper's batch protocol (one adapter per
test batch, mean score) and per sample. Everything lives in the pretrained
embedding space, i.e. this measures `improve.routing_space=pretrained`
variants; adapter-space routing still needs full runs.

Usage (project root, on a GPU server):
    python scripts/routing_probe.py --cfg configs/imgn-r-seed-15.yml
    python scripts/routing_probe.py --cfg configs/cifar-100-seed-15.yml --out reports/probe_cifar100.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models.model_register  # noqa: F401,E402
from lib.config import cfg, cfg_from_file, cfg_from_list  # noqa: E402
from lib.device import resolve_device  # noqa: E402
from lib.signatures import OnlineDiagonalGMM  # noqa: E402
from lib.task_keys import ClassGaussianKeyBank, gmm_log_likelihood, gmm_mahal_min  # noqa: E402


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def extract(cache_path):
    """Pretrained embeddings of the training stream (aug + clean view, in stream
    batches) and of every task's test set (clean view)."""
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    from timm.models import create_model
    from data.dataset import build_continual_dataloader

    set_seed(cfg.seed)
    model = create_model(cfg.dtask.model, pretrained=cfg.dtask.pretrained, num_classes=cfg.dtask.nb_classes,
                         drop_rate=0.0, drop_path_rate=0.0, drop_block_rate=None, head_type=cfg.dtask.head_type)
    model.to(cfg.device).eval()
    # Ask the loader for (augmented, clean) views of every training sample.
    cfg.improve.routing_key, cfg.improve.key_view = 'class_gauss', 'clean'
    loaders, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)

    feats = {'train': [], 'val': [], 'class_mask': class_mask}
    t0 = time.time()
    for t in range(cfg.continual.n_tasks):
        batches = []
        for (x, y) in loaders[t]['train']:
            x_aug, x_clean = x
            za = model(x_aug.to(cfg.device))['pre_logits'].float().cpu()
            zc = model(x_clean.to(cfg.device))['pre_logits'].float().cpu()
            batches.append((za, zc, y.clone()))
        vz, vy = [], []
        for (x, y) in loaders[t]['val']:
            vz.append(model(x.to(cfg.device))['pre_logits'].float().cpu())
            vy.append(y.clone())
        feats['train'].append(batches)
        feats['val'].append((torch.cat(vz), torch.cat(vy)))
        print(f"[extract] task {t}: {len(batches)} train batches, {sum(len(y) for y in vy)} test samples "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(feats, cache_path)
    return feats


# ----------------------------------------------------------------------
# Key fitting
# ----------------------------------------------------------------------
def fit_gmms(feats, view):
    """One OnlineDiagonalGMM per task, streamed batch by batch (as in main.py)."""
    v = 0 if view == 'aug' else 1
    keys = []
    for batches in feats['train']:
        gmm = OnlineDiagonalGMM(n_clusters=cfg.signature.n_clusters, feature_dim=768, device='cpu',
                                split_threshold=cfg.signature.split_threshold, cfg=cfg)
        for b, batch in enumerate(batches):
            gmm.update(batch[v], b)
        keys.append((gmm.means.clone(), gmm.vars.clone(), gmm.counts.clone()))
    return keys


def fit_class_bank(feats, view, **kw):
    v = 0 if view == 'aug' else 1
    bank = ClassGaussianKeyBank(feature_dim=768, **kw)
    for t, batches in enumerate(feats['train']):
        for batch in batches:
            bank.update(t, batch[v], batch[2])
    return bank


# ----------------------------------------------------------------------
# Routing evaluation (paper protocol + per sample)
# ----------------------------------------------------------------------
def routing_accuracy(score_fn, feats, n_tasks):
    """score_fn(z) -> [N, T] scores (lower = better)."""
    B = cfg.dtask.batch_size
    b_hit = b_tot = s_hit = s_tot = 0
    per_task = []
    for t in range(n_tasks):
        z, _ = feats['val'][t]
        S = score_fn(z)
        s_pred = S.argmin(1)
        hits = 0
        n_b = 0
        for i in range(0, z.shape[0], B):
            hits += int(int(S[i:i + B].mean(0).argmin()) == t)
            n_b += 1
        b_hit += hits; b_tot += n_b
        s_hit += int((s_pred == t).sum()); s_tot += z.shape[0]
        per_task.append(round(hits / n_b, 4))
    return {'batch': b_hit / b_tot, 'sample': s_hit / s_tot, 'per_task_batch': per_task}


def class_accuracy(bank, feats, n_tasks):
    """Within-task (oracle task) and all-class accuracy of the class-Gaussian posterior."""
    within = allc = n = 0
    ids = bank.adapter_ids()
    for t in range(n_tasks):
        z, y = feats['val'][t]
        ll, cls = bank.class_log_likelihood(z, t)
        within += int((torch.tensor(cls)[ll.argmax(1)] == y).sum())
        all_ll, all_cls = [], []
        for s in ids:
            l2, c2 = bank.class_log_likelihood(z, s)
            all_ll.append(l2); all_cls += c2
        allc += int((torch.tensor(all_cls)[torch.cat(all_ll, 1).argmax(1)] == y).sum())
        n += z.shape[0]
    return within / n, allc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--cache', default=None, help='feature cache file (default: features_cache/<cfg>_seed<seed>.pt)')
    ap.add_argument('--out', default=None, help='write results JSON here')
    ap.add_argument('--quick', action='store_true', help='only the headline variants')
    args = ap.parse_args()

    cfg_from_file(args.cfg)
    cfg_from_list(args.set)
    cfg.device = resolve_device(cfg.improve.device, cfg.gpu_ids)
    name = os.path.splitext(os.path.basename(args.cfg))[0]
    cache = args.cache or os.path.join('features_cache', f'{name}_seed{cfg.seed}.pt')
    feats = extract(cache)
    T = len(feats['train'])

    results = []

    def record(label, score_fn, extra=None):
        r = routing_accuracy(score_fn, feats, T)
        r['variant'] = label
        if extra:
            r.update(extra)
        results.append(r)
        print(f"{label:<58s} batch {100 * r['batch']:6.2f} | sample {100 * r['sample']:6.2f}", flush=True)

    # Original GMM keys (fit with the config's K / split hyper-parameters).
    for view in (['aug'] if args.quick else ['aug', 'clean']):
        set_seed(cfg.seed)  # KMeans init and split offsets are random
        gmms = fit_gmms(feats, view)
        record(f'gmm[{view}] mahal_min (published score)',
               lambda z: torch.stack([gmm_mahal_min(z, *k) for k in gmms], 1))
        for tau in ([0.0] if args.quick else [0.0, 10.0, 100.0]):
            record(f'gmm[{view}] nll var_shrink={tau:g}',
                   lambda z, tau=tau: torch.stack([-gmm_log_likelihood(z, *k, var_shrink=tau) for k in gmms], 1))

    # Class-conditional Gaussian keys.
    grid = []
    for view in ['aug', 'clean']:
        for cov_type in ['diag', 'full']:
            for scope in ['task', 'shared']:
                for mode, s in [('oas', 0.0), ('fixed', 0.1), ('fixed', 0.5)]:
                    if cov_type == 'diag' and mode == 'fixed' and s == 0.5:
                        continue
                    grid.append((view, cov_type, scope, mode, s))
    if args.quick:
        grid = [('aug', 'full', 'shared', 'oas', 0.0), ('clean', 'full', 'shared', 'oas', 0.0),
                ('clean', 'full', 'task', 'oas', 0.0), ('clean', 'diag', 'task', 'oas', 0.0)]
    for view, cov_type, scope, mode, s in grid:
        bank = fit_class_bank(feats, view, cov_type=cov_type, cov_scope=scope, shrink_mode=mode, shrink=s)
        within, allc = class_accuracy(bank, feats, T)
        label = f'class_gauss[{view}] {cov_type}/{scope}/{mode}' + (f'={s:g}' if mode == 'fixed' else '')
        record(label, lambda z, bank=bank: (-bank.log_likelihood_all(z)).float(),
               {'within_task_acc': within, 'all_class_acc': allc})
        print(f"{'':<58s} (class-Gaussian within-task acc {100 * within:.2f}, all-class acc {100 * allc:.2f})")

    print("\n| variant | routing acc, batch (%) | routing acc, sample (%) |\n|---|---|---|")
    for r in results:
        print(f"| {r['variant']} | {100 * r['batch']:.2f} | {100 * r['sample']:.2f} |")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'cfg': args.cfg, 'seed': cfg.seed, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
