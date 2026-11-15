import os
import json
import numpy as np
import torch
import random
import argparse
import pprint
import copy
import time
from itertools import islice
import models.model_register

from lib.config import cfg, cfg_from_file, cfg_from_list, validate_config
from lib.utils import *
from lib.device import resolve_device, autocast
from lib.ia3 import is_ia3_param, collect_ia3_params, apply_ia3_updates, reset_ia3_to_identity
from lib.checkpointing import save_task_checkpoint, atomic_torch_save
from lib.experiment import begin_run, finish_run, write_json, preserve_rng_state, loader_manifest
from lib.task_keys import build_key_bank
from lib.mass_specialists import GaussianAnchor
from data.dataset import *
from utils.vision import *
from lib.jacobians import build_ia3_jacobian
from lib.coupled_tangent import CoupledTangentState
from lib.train import tangent_tuning_a_batch, train_one_batch
from lib.signatures import OnlineDiagonalGMM, detect_convergence_stationary_streaming
from evaluate import evaluate_bank, _build_vit_model
from timm.models import create_model
from timm.utils import accuracy
from torch import optim

# Hyperparameters for the closed-form linearized adaptation (Eq. "ridge-sol").
TANGENT_ETA = 0.03
TANGENT_RIDGE_LAMBDA = 1.0
# Learning rate for the (gradient-based) classification-head warm start.
HEAD_LR = 1e-3

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

class Net(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, target):
        out = self.model(X)
        logits = out["logits"]
        logits = logits.softmax(dim=1)
        target_logits = torch.gather(logits, 1, target.reshape(target.size(0),1).to(cfg.device)).squeeze(1)
        out["target_logits"] = target_logits
        return target_logits

def log_gmm_summary(online_gauss, prefix=""):
    """Log a concise summary of the current Gaussian-mixture task key."""
    weights = online_gauss.counts / (online_gauss.counts.sum() + 1e-8)
    log(f"\n{prefix} ===== GMM task key summary =====")
    log(f"Clusters: {online_gauss.means.shape[0]} | Feature dim: {online_gauss.means.shape[1]}")
    for k in range(online_gauss.means.shape[0]):
        log(f"  cluster {k}: count={online_gauss.counts[k].item():.1f}, "
            f"weight={weights[k].item():.3f}, "
            f"mean_norm={online_gauss.means[k].norm().item():.3f}, "
            f"var_mean={online_gauss.vars[k].mean().item():.4f}")
    log("==================================\n")

def snapshot_adapter(global_updates, online_gauss, model=None):
    """Freeze the IA3 solution and GMM task key of a distribution (in memory)."""
    state = {
        'ia3': {k: v.detach().clone() for k, v in global_updates.items()},
        'gmm': (online_gauss.means.clone(), online_gauss.vars.clone(), online_gauss.counts.clone()),
    }
    if model is not None:
        # An expert is an adapter AND its head; later head updates must not alter it.
        state['head_weight'] = model.head.weight.detach().cpu().clone()
        state['head_bias'] = model.head.bias.detach().cpu().clone()
    return state

def learn_continually():
    log('\nRunning experiments using TANGENT FINETUNING')
    imp = cfg.improve
    started = time.monotonic()
    tasks = list(range(cfg.continual.n_tasks))
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()
    if cfg.runtime.max_tasks:
        tasks = tasks[:cfg.runtime.max_tasks]

    # Define ViT model.
    vit_model = create_model(
        cfg.dtask.model,
        pretrained=cfg.dtask.pretrained,
        num_classes=cfg.dtask.nb_classes,
        drop_rate=cfg.dtask.drop,
        drop_path_rate=cfg.dtask.drop_path,
        drop_block_rate=None,
        head_type=cfg.dtask.head_type,
    )
    vit_model.to(cfg.device)
    for n, p in vit_model.named_parameters():
        p.requires_grad = "head" in n

    # Load datasets.
    data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    stream = loader_manifest(data_loader, class_mask, tasks)
    write_json(os.path.join(cfg.dtask.output_dir, 'stream.json'), stream)
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    # `net_g` wraps a frozen copy of the backbone and exposes target-class
    # softmax probabilities; it is only used to compute Jacobians (via
    # `torch.func.jacrev`) for the closed-form linearized update. Its IA3
    # vectors stay at identity, so `backbone` also gives pretrained embeddings.
    backbone = copy.deepcopy(vit_model)
    net_g = Net(backbone)
    net_g.to(cfg.device)
    net_g.eval()
    atomic_torch_save(backbone.state_dict(), os.path.join(cfg.dtask.output_dir, 'backbone.pth'))

    tuning_params = collect_ia3_params(net_g)
    log("Tuning parameters are")
    for k, v in tuning_params.items():
        log(f"{k} : {v.shape}")

    head_params = sum(p.numel() for n, p in vit_model.named_parameters() if p.requires_grad and "head" in n)
    ia3_params = sum(p.numel() for n, p in net_g.named_parameters() if is_ia3_param(n))
    log(f"Head trainable params: {head_params}")
    log(f"IA3 trainable params: {ia3_params}")
    log(f"Total trainable params (IA3 + head): {head_params + ia3_params}")

    jacobian_fn = build_ia3_jacobian(net_g, cfg.runtime.jacobian_mode, cfg.runtime.jacobian_chunk_size)

    coupled_state = (CoupledTangentState(tuning_params, imp.tangent_sketch_rank, cfg.seed + 730201)
                     if imp.tangent_solver == 'coupled_sketch' else None)
    global_At_A = {} if coupled_state is not None else {
        layer: torch.zeros((p.size(-1), p.size(-1)), device=cfg.device) for layer, p in tuning_params.items()}
    global_AtA_w = {} if coupled_state is not None else {
        layer: torch.zeros(p.size(-1), device=cfg.device) for layer, p in tuning_params.items()}
    AtA_ridge = dict()

    previous_global_update = None
    count_save, batch_idx = 0, 0
    COOLDOWN_LIMIT = cfg.signature.cooldown_limit
    cooldown = COOLDOWN_LIMIT
    is_converged = False
    task_class_mask = dict()
    mahal_history = {count_save: []}
    within_seen_classes = set()
    adapter_bank = []                    # finalized distributions: {'ia3', 'gmm'}
    key_bank = build_key_bank(cfg, feature_dim=vit_model.num_features)
    anchor = (GaussianAnchor(vit_model.num_features, cfg.dtask.nb_classes, imp.anchor_temperature)
              if imp.prediction == 'mass' else None)
    frozen_head = vit_model if anchor is not None else None
    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))

    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    optimizer = optim.Adam(vit_model.parameters(), lr=HEAD_LR)
    online_gauss = OnlineDiagonalGMM(n_clusters=cfg.signature.n_clusters, feature_dim=vit_model.num_features, device="cpu",
                                      split_threshold=cfg.signature.split_threshold, cfg=cfg)
    # Ablation "original GMM key, pretrained embedding space": a second GMM with
    # the same hyper-parameters, used only for routing (detection is unchanged).
    route_gauss = None
    if imp.routing_key == 'gmm' and imp.routing_space == 'pretrained':
        route_gauss = OnlineDiagonalGMM(n_clusters=cfg.signature.n_clusters, feature_dim=vit_model.num_features, device="cpu",
                                         split_threshold=cfg.signature.split_threshold, cfg=cfg)
    log_gmm_summary(online_gauss)

    for task_id in tasks:
        log(f"Task {task_id}: {len(data_loader[task_id]['train'])} training batches")

    seen_tasks = []
    training_samples = 0
    for step, task_id in enumerate(tasks):
        seen_tasks.append(task_id)
        batches = data_loader[task_id]["train"]
        if cfg.runtime.max_train_batches:
            batches = islice(batches, cfg.runtime.max_train_batches)
        for (input, target) in batches:
            # With `key_view == 'clean'` the loader also returns the test-transform view.
            input_key = None
            if isinstance(input, (list, tuple)):
                input, input_key = input
                input_key = input_key.to(cfg.device)
            input, target = input.to(cfg.device), target.to(cfg.device)
            training_samples += len(target)
            new_classes = torch.unique(target).tolist()
            target_classes = set(new_classes)
            log(f"New class coming: {new_classes}")

            vit_model.eval()
            with torch.no_grad(), autocast(cfg.device):
                temp_features = vit_model(input)['pre_logits'].float().cpu()

            # Routing / change-point score: min Mahalanobis-plus-prior
            # distance to the current task key (Eq. for S_t(z) in the paper).
            mean_ood = online_gauss.score(temp_features).mean()
            mahal_history[count_save].append(mean_ood)
            log(f"Task {count_save}, Batch {batch_idx}, mean_ood: {mean_ood}")

            if cooldown > 0:
                cooldown -= 1
                log(f"[Cooldown active] Skipping detection... {cooldown} left")
                within_seen_classes |= target_classes  # update seen classes before converging
            else:
                if not is_converged:  # start detecting
                    within_seen_classes |= target_classes  # still update seen classes before converging
                    values_t = torch.tensor(mahal_history[count_save])
                    converged, converged_idx = detect_convergence_stationary_streaming(
                        values_t, window=cfg.signature.window_t, k=cfg.signature.k, cool_down=COOLDOWN_LIMIT)
                    if converged:
                        is_converged = True
                        converged_value = values_t[converged_idx].item() if converged_idx is not None else None
                        log("Convergence starts at index: %d" % converged_idx)
                        log("Value at convergence: %s " % converged_value)

                # After convergence, if new classes come then we mark this as a new incoming distribution.
                elif is_converged and not target_classes.issubset(within_seen_classes):
                    log(f"Distribution {count_save}-th is detected at batch {batch_idx} with mean_ood: {mean_ood}")
                    log(f"New coming classes: {new_classes}")

                    save_task_checkpoint(cfg.dtask.output_dir, count_save, global_At_A, global_AtA_w,
                                          online_gauss, global_updates=previous_global_update,
                                          route_gauss=route_gauss, save_ridge_stats=cfg.runtime.save_ridge_stats)
                    adapter_bank.append(snapshot_adapter(previous_global_update, route_gauss or online_gauss, frozen_head))
                    task_class_mask[count_save] = list(within_seen_classes)

                    # Reset everything for the next distribution.
                    within_seen_classes = set()
                    is_converged = False
                    mahal_history[count_save + 1] = []
                    cooldown = COOLDOWN_LIMIT
                    for layer in global_At_A:
                        global_At_A[layer].zero_()
                        global_AtA_w[layer].zero_()
                    if coupled_state is not None:
                        coupled_state.reset()
                    reset_ia3_to_identity(vit_model, cfg.device)
                    online_gauss.reset()
                    if route_gauss is not None:
                        route_gauss.reset()
                    optimizer = optim.Adam(vit_model.parameters(), lr=HEAD_LR)
                    count_save += 1
                    batch_idx = 0

            # The triggering batch belongs to the new segment too. Without this
            # update, its labels can be absent from the saved inference mask.
            within_seen_classes |= target_classes

            # Gradient-based warm start of the classification head for the ViT.
            log(f"Within seen class: {within_seen_classes}")
            train_one_batch(model=vit_model, criterion=criterion, input=input, target=target,
                             optimizer=optimizer, device=cfg.device, max_norm=1.0,
                             set_training_mode=True, task_id=count_save, class_mask=new_classes,
                             cache_features=imp.fast_head, efficient_loop=cfg.runtime.efficient_head_loop)
            vit_model.eval()

            # Sync the (just-updated) head into the Jacobian-computation
            # copies before solving the closed-form update.
            with torch.no_grad():
                net_g.model.head.weight.copy_(vit_model.head.weight)
                net_g.model.head.bias.copy_(vit_model.head.bias)
                backbone.head.weight.copy_(vit_model.head.weight)
                backbone.head.bias.copy_(vit_model.head.bias)
            for p in net_g.parameters():
                p.requires_grad = False

            # Closed-form linearized IA3 update (Eq. "ridge-sol" in the paper).
            global_updates = tangent_tuning_a_batch(
                input, target, global_At_A, global_AtA_w, AtA_ridge, net_g, jacobian_fn,
                vit_model, tuning_params, backbone, eta=TANGENT_ETA, lambda_=TANGENT_RIDGE_LAMBDA,
                max_epochs=imp.refine_max_iters, fast_solve=imp.fast_solve, select_metric=imp.select_metric,
                accumulate=imp.accumulate, select_tol=imp.select_tol,
                clear_cache=cfg.runtime.empty_cache_each_batch, coupled_state=coupled_state,
                residual_target=imp.tangent_residual_target)
            apply_ia3_updates(vit_model, global_updates, cfg.device)

            # Accuracy of the global update on the current batch (classes outside
            # this batch are masked out, since task identity is unknown).
            with torch.no_grad():
                out = vit_model(input)["logits"]
                not_mask = np.setdiff1d(np.arange(cfg.dtask.nb_classes), new_classes)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(cfg.device)
                out = out.index_fill(dim=1, index=not_mask, value=float('-inf'))
                acc1, acc5 = accuracy(out, target, topk=(1, min(5, out.shape[1])))
            log(f"GLOBAL UPDATES ON BATCH {batch_idx} is: {acc1}")

            with torch.no_grad():
                features = vit_model(input)['pre_logits'].cpu()
            online_gauss.update(features, batch_idx)
            if route_gauss is not None:
                with torch.no_grad():
                    pre_features = backbone(input)['pre_logits'].cpu()
                # KMeans init / split offsets draw from the global RNGs; isolate them
                # so the data stream stays identical to the baseline run.
                with preserve_rng_state():
                    route_gauss.update(pre_features, batch_idx)

            # Order-invariant class-conditional task key (additive statistics).
            clean_pre_features = None
            if anchor is not None:
                with torch.no_grad():
                    clean_pre_features = backbone(input_key if input_key is not None else input)['pre_logits']
                anchor.update(clean_pre_features, target)
            if key_bank is not None:
                with torch.no_grad():
                    if imp.routing_space == 'pretrained':
                        key_features = (clean_pre_features if clean_pre_features is not None else
                                        backbone(input_key if input_key is not None else input)['pre_logits'])
                    elif input_key is not None:
                        key_features = vit_model(input_key)['pre_logits']
                    else:
                        key_features = features
                key_bank.update(count_save, key_features, target)

            previous_global_update = global_updates
            batch_idx += 1

        # Evaluation at the end of each task.
        with torch.no_grad(), preserve_rng_state():
            temp_class_mask = dict(task_class_mask)
            temp_class_mask[count_save] = list(within_seen_classes)
            # Use the exact frozen backbone, including when pretrained=False.
            # This also avoids reloading/downloading ViT weights at every task.
            eval_model = copy.deepcopy(backbone).eval()
            eval_model.head.weight.copy_(vit_model.head.weight)
            eval_model.head.bias.copy_(vit_model.head.bias)
            adapters = adapter_bank + [snapshot_adapter(previous_global_update, route_gauss or online_gauss, frozen_head)]
            if key_bank is not None:
                log(f"Class-Gaussian keys: {key_bank.describe()}")
            task_test_stats = evaluate_bank(eval_model, adapters, key_bank, data_loader, temp_class_mask,
                                            step, acc_matrix, cfg, seen_task_ids=seen_tasks, anchor=anchor)
            atomic_torch_save(torch.from_numpy(acc_matrix), os.path.join(cfg.dtask.output_dir, 'global_acc_matrix.pth'))
            # One coherent bundle is published after each completed evaluation.
            # It supports portable re-evaluation, not training resume.
            atomic_torch_save({
                'version': 2 if anchor is not None else 1, 'adapters': adapters,
                'anchor': None if anchor is None else anchor.state_dict(),
                'key_bank': None if key_bank is None else key_bank.state_dict(),
                'head_weight': vit_model.head.weight, 'head_bias': vit_model.head.bias,
                'class_mask': temp_class_mask, 'seen_tasks': list(seen_tasks),
                'acc_matrix': torch.from_numpy(acc_matrix), 'stream': stream,
            }, os.path.join(cfg.dtask.output_dir, 'evaluation_checkpoint.pth'))
            write_json(os.path.join(cfg.dtask.output_dir, 'progress.json'), {
                'completed_tasks': step + 1, 'total_tasks': len(tasks),
                'training_samples': training_samples, 'elapsed_seconds': time.monotonic() - started,
                **{k: v for k, v in task_test_stats.items() if k != 'acc_matrix'},
            })
            log(f"Global Acc Matrix after truth (Not predicted) task {task_id} : \n {task_test_stats['acc_matrix']}")
            del eval_model

    # Final checkpoint after the last batch of the last task.
    save_task_checkpoint(cfg.dtask.output_dir, count_save, global_At_A, global_AtA_w, online_gauss,
                          global_updates=global_updates, head_weight=vit_model.head.weight,
                              head_bias=vit_model.head.bias, mahal_history=mahal_history, route_gauss=route_gauss,
                              save_ridge_stats=cfg.runtime.save_ridge_stats)
    task_class_mask[count_save] = list(within_seen_classes)
    atomic_torch_save(task_class_mask, os.path.join(cfg.dtask.output_dir, 'class_mask.pth'))
    if key_bank is not None:
        atomic_torch_save(key_bank.state_dict(), os.path.join(cfg.dtask.output_dir, 'key_bank.pth'))
    log(f"FINAL | avg acc {task_test_stats['avg_acc']:.4f} | forgetting {task_test_stats['forgetting']:.4f} | "
        f"routing (batch) {task_test_stats['routing_batch']} | routing (sample) {task_test_stats['routing_sample']} | "
        f"detected distributions {count_save + 1}")
    result = {'cfg': cfg.get('cfg_name', ''), 'tag': cfg.get('run_tag', ''), 'seed': cfg.seed,
              'avg_acc': float(task_test_stats['avg_acc']), 'forgetting': float(task_test_stats['forgetting']),
              'routing_batch': task_test_stats['routing_batch'],
              'routing_sample': task_test_stats['routing_sample'],
              'detected_distributions': count_save + 1, 'improve': dict(cfg.improve),
              'protocol_version': 2, 'routing_metric': 'selected_adapter_label_coverage',
              'kind': 'diagnostic' if (cfg.runtime.max_train_batches or cfg.runtime.max_eval_batches or cfg.runtime.max_tasks) else 'benchmark',
              'elapsed_seconds': time.monotonic() - started, 'training_samples': training_samples,
              'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(cfg.device) if cfg.device.type == 'cuda' else None}
    if anchor is not None:
        result.update({k: v for k, v in task_test_stats.items() if k != 'acc_matrix'})
        result.update(protocol_version=3, routing_metric=None)
    write_json(os.path.join(cfg.dtask.output_dir, 'result.json'), result)
    return task_test_stats

def main():
    parser = argparse.ArgumentParser(description='TANGENT FINETUNING in Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/imgn-r-seed-15.yml')
    parser.add_argument('--seed', type=int, default=None, help='override the seed in the config')
    parser.add_argument('--output_dir', default=None, help='override dtask.output_dir')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='config overrides, e.g. --set improve.routing_key=class_gauss')
    parser.add_argument('--tag', default='', help='run tag appended to the log file name')

    args = parser.parse_args()
    cfg_name = os.path.splitext(os.path.basename(args.cfg_file))[0]
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg_from_list(args.set)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.dtask.output_dir = args.output_dir
    validate_config()
    cfg.cfg_name, cfg.run_tag = cfg_name, args.tag
    cfg.device = resolve_device(cfg.improve.device, cfg.gpu_ids)
    begin_run(cfg)

    log_dir = cfg.dtask.output_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.dtask.output_dir, exist_ok=True)

    final_filename = os.path.join(
        log_dir,
        f"{cfg_name}{'_' + args.tag if args.tag else ''}_seed{cfg.seed}_nc{cfg.signature.n_clusters}"
        f"_k{cfg.signature.k}_wt{cfg.signature.window_t}_more_trial.log"
    )
    logging.basicConfig(
        filename=final_filename,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    log(pprint.pformat(cfg))

    cfg.output_dir = os.path.join(cfg.dtask.output_dir, f"{cfg_name}_seed_{cfg.seed}")
    log(f"Device: {cfg.device}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_seed(cfg.seed)

    if cfg.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(cfg.device)
    try:
        learn_continually()
    except BaseException as exc:
        finish_run(cfg.dtask.output_dir, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                   error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        finish_run(cfg.dtask.output_dir, 'completed')

if __name__ == '__main__':
    main()
