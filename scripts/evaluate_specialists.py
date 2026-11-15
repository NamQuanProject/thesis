"""Predeclared paired inference ablations on ONE trained specialist checkpoint.

No fitting, test-label calibration or modification of training artifacts. Counts
individual image-expert evaluations and saves per-example traces, so exact lazy
agreement and batch-composition invariance are checked rather than inferred from
equal aggregate accuracies. Prices and alpha are not selected by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ordered_trace(records):
    import torch
    merged = {k: torch.cat([r[k] for r in records]) for k in records[0]}
    order = sorted(range(len(merged['source'])), key=lambda i: (int(merged['source'][i]), int(merged['index'][i])))
    return {k: v[order] for k, v in merged.items()}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=('anchor', 'soft_mixture', 'full', 'certified', 'budget'),
                        default=['anchor', 'soft_mixture', 'full', 'certified', 'budget'])
    parser.add_argument('--protocols', nargs='+', choices=('grouped', 'mixed'), default=['grouped', 'mixed'])
    parser.add_argument('--device', default='cuda', choices=('cuda', 'mps', 'cpu'))
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or len(set(args.protocols)) != len(args.protocols):
        parser.error('duplicate modes/protocols')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a fresh output directory; evaluation results are immutable')
    import torch
    from lib.config import cfg, cfg_from_file, validate_config, reset_config
    from lib.device import resolve_device
    from lib.experiment import write_json, source_identity
    from lib.checkpointing import atomic_torch_save
    from evaluate import run_evaluation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run_dir / 'evaluation_checkpoint.pth'
    identity = digest(checkpoint)
    write_json(args.output_dir / 'plan.json', {'source': source_identity(), 'checkpoint_sha256': identity,
               'run_dir': str(args.run_dir.resolve()), 'modes': args.modes, 'protocols': args.protocols,
               'kind': 'predeclared_inference_ablation', 'hyperparameter_selection': 'none'})
    results, traces = {}, {}
    for protocol in args.protocols:
        for mode in args.modes:
            if digest(checkpoint) != identity:
                raise RuntimeError('Training checkpoint changed; evaluate a completed run')
            reset_config()
            cfg_from_file(args.run_dir / 'config.json')
            cfg.dtask.output_dir = str(args.run_dir.resolve())
            if cfg.improve.prediction != 'mass':
                raise ValueError('This protocol requires anchor statistics and frozen expert heads from a mass run')
            cfg.improve.mass_inference = mode
            cfg.runtime.mixed_eval = protocol == 'mixed'
            cfg.device = resolve_device(args.device, cfg.gpu_ids)
            validate_config()
            records = []
            result = run_evaluation(history_available=False, prediction_records=records)
            label = f'{protocol}_{mode}'
            results[label] = {k: v for k, v in result.items() if k != 'acc_matrix'}
            traces[label] = ordered_trace(records)
            atomic_torch_save(traces[label], args.output_dir / (label + '_predictions.pth'))
            write_json(args.output_dir / (label + '.json'), results[label])
    checks = {}
    for protocol in args.protocols:
        reference = traces.get(protocol + '_full')
        if reference is None:
            continue
        for mode in ('certified', 'budget'):
            candidate = traces.get(protocol + '_' + mode)
            if candidate is None:
                continue
            for k in ('source', 'index', 'target'):
                if not torch.equal(reference[k], candidate[k]):
                    raise AssertionError('Ablations evaluated different observations')
            bad = (reference['prediction'] != candidate['prediction']) & candidate['certified']
            checks[protocol + '_' + mode + '_certificate_violations'] = int(bad.sum())
            if bad.any() or (mode == 'certified' and not candidate['certified'].all()):
                raise AssertionError('Full-model top-1 certificate failed on real predictions')
    if 'grouped' in args.protocols and 'mixed' in args.protocols:
        for mode in args.modes:
            a, b = traces['grouped_' + mode], traces['mixed_' + mode]
            for k in ('source', 'index', 'target'):
                if not torch.equal(a[k], b[k]):
                    raise AssertionError('Batch protocols evaluated different observations')
            # Small GPU batching roundoff can change very close ties. Record all
            # disagreements instead of silently treating equal accuracies as proof.
            checks[mode + '_batch_prediction_disagreements'] = int((a['prediction'] != b['prediction']).sum())
            checks[mode + '_batch_work_disagreements'] = int((a['expert_evaluations'] != b['expert_evaluations']).sum())
    write_json(args.output_dir / 'summary.json', {'results': results, 'checks': checks})
    print(json.dumps({'results': results, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
