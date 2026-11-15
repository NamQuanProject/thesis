"""Compare equivalent IA3 Jacobians on one training batch; never reads test scores.

Timing includes concurrent GPU load, if any. The report records other GPU PIDs;
do not describe such measurements as an isolated end-to-end training speedup.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    import torch
    import main as training
    from data.dataset import build_continual_dataloader
    from lib.config import cfg, cfg_from_file
    from lib.experiment import environment, source_identity, write_json
    from lib.ia3 import collect_ia3_params
    from lib.jacobians import build_ia3_jacobian
    from lib.train import train_one_batch
    from timm.models import create_model

    cfg_from_file(Path(__file__).resolve().parents[1] / 'configs/imgn-r-seed-15.yml')
    cfg.device = torch.device('cuda:0')
    cfg.dtask.data_path, cfg.dtask.num_workers = args.data_path, 0
    training.set_seed(cfg.seed)
    model = create_model(cfg.dtask.model, pretrained=True, num_classes=cfg.dtask.nb_classes,
                         drop_rate=0., drop_path_rate=0., head_type=cfg.dtask.head_type).to(cfg.device)
    for name, param in model.named_parameters():
        param.requires_grad = 'head' in name
    loaders, _ = build_continual_dataloader(cfg.dtask.batch_size)
    x, y = next(iter(loaders[0]['train']))
    x, y = x.to(cfg.device), y.to(cfg.device)
    train_one_batch(model, torch.nn.CrossEntropyLoss(), x, y,
                    torch.optim.Adam(model.parameters(), lr=training.HEAD_LR), cfg.device,
                    max_norm=1., class_mask=y.unique().tolist(), cache_features=True)
    net = training.Net(model).eval()
    for p in net.parameters():
        p.requires_grad = False
    params = collect_ia3_params(net)
    modes = [('batch', 1), ('batch', 4), ('per_sample', 4), ('per_sample', 0)]
    functions = {str(pair): build_ia3_jacobian(net, *pair) for pair in modes}
    reference = {k: v.detach() for k, v in functions[str(modes[0])](params, x, y).items()}
    records = {str(pair): {'mode': pair[0], 'chunk': pair[1], 'seconds': [], 'peak_allocated_bytes': 0,
                           'max_absolute_error': 0., 'equivalent': True} for pair in modes}
    # Reverse alternate rounds to reduce a fixed timing-order effect.
    for repetition in range(args.repeats):
        for pair in modes if repetition % 2 == 0 else list(reversed(modes)):
            record = records[str(pair)]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            value = functions[str(pair)](params, x, y)
            torch.cuda.synchronize()
            record['seconds'].append(time.perf_counter() - start)
            record['peak_allocated_bytes'] = max(record['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            for name in reference:
                error = float((value[name] - reference[name]).abs().max())
                record['max_absolute_error'] = max(record['max_absolute_error'], error)
                record['equivalent'] &= torch.allclose(value[name], reference[name], rtol=1e-3, atol=1e-5)
            del value
    for record in records.values():
        record['median_seconds'] = statistics.median(record['seconds'])
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                              '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        processes = 'unavailable'
    result = {'kind': 'training_batch_jacobian_profile', **source_identity(), **environment(),
              'batch_size': len(y), 'gpu_processes_at_end': processes, 'results': list(records.values()),
              'qualification': 'One training batch; timings may include concurrent work. No accuracy benchmark.'}
    write_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)
    if not all(record['equivalent'] for record in records.values()):
        raise RuntimeError('Jacobian equivalence failed; do not deploy optimized execution')


if __name__ == '__main__':
    main()
