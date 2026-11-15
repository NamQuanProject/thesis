"""Print or execute a predeclared ImageNet-R comparison, one job at a time.

Printing is the default. --execute is required for training. Process termination
stops the job's process group, never the Vast.ai rental or unrelated jobs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'base': [],
    'class_gauss': ['improve.routing_key=class_gauss', 'improve.routing_space=pretrained', 'improve.key_view=clean'],
    'base_sample': ['improve.routing_unit=sample'],
    'class_gauss_sample': ['improve.routing_key=class_gauss', 'improve.routing_space=pretrained',
                           'improve.key_view=clean', 'improve.routing_unit=sample'],
    'coupled': ['improve.tangent_solver=coupled_sketch'],
    'coupled_raw': ['improve.tangent_solver=coupled_sketch', 'improve.tangent_residual_target=false'],
    'coupled_gauss': ['improve.tangent_solver=coupled_sketch', 'improve.routing_key=class_gauss',
                      'improve.routing_space=pretrained', 'improve.key_view=clean'],
    'mass': ['improve.prediction=mass', 'improve.routing_unit=sample'],
}


def commands(args):
    for seed in args.seeds:
        for variant in args.variants:
            output = args.output_root.resolve() / args.stage / variant / f'seed{seed}'
            overrides = ['improve.device=' + args.device, 'dtask.data_path=' + str(args.data_path.resolve()),
                         'dtask.num_workers=' + str(args.workers),
                         'runtime.jacobian_mode=' + args.jacobian_mode,
                         'runtime.jacobian_chunk_size=' + str(args.jacobian_chunk_size),
                         'runtime.efficient_head_loop=' + str(args.efficient_runtime).lower(),
                         'runtime.empty_cache_each_batch=' + str(not args.efficient_runtime).lower(),
                         'runtime.save_ridge_stats=false', 'dtask.download=false', *VARIANTS[variant]]
            if args.stage == 'pilot':
                overrides += ['runtime.max_tasks=2', 'runtime.max_train_batches=2',
                              'runtime.max_eval_batches=2', 'dtask.epochs=1', 'improve.refine_max_iters=4']
            command = [sys.executable, '-u', str(ROOT / 'main.py'), '--cfg',
                       str(ROOT / 'configs/imgn-r-seed-15.yml'), '--seed', str(seed), '--tag', variant,
                       '--output_dir', str(output), '--set', *overrides]
            yield output, command


def stop_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'compare'), default='pilot')
    parser.add_argument('--seeds', type=int, nargs='+', default=[15])
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=['base', 'class_gauss'])
    parser.add_argument('--data-path', type=Path, default=ROOT / 'local_datasets')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/imagenet-r')
    parser.add_argument('--device', choices=('cuda', 'cpu', 'mps'), default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--jacobian-mode', choices=('batch', 'per_sample'), default='batch')
    parser.add_argument('--jacobian-chunk-size', type=int, default=1,
                        help='0 vectorizes the whole batch; does not change training batch size')
    parser.add_argument('--efficient-runtime', action='store_true',
                        help='retain allocator cache and skip discarded head metrics/barriers')
    parser.add_argument('--max-seconds', type=float, help='optional limit across all jobs; rental billing is separate')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error('--max-seconds must be positive')
    if args.workers < 0 or args.jacobian_chunk_size < 0:
        parser.error('workers and Jacobian chunk size must be nonnegative')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error('duplicate seeds/variants would overwrite runs')
    jobs = list(commands(args))
    for _, command in jobs:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return
    for output, _ in jobs:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Preserving existing run {output}; use a fresh --output-root')
    # Heavy imports happen only when execution was explicitly requested.
    sys.path.insert(0, str(ROOT))
    from lib.experiment import write_json, source_identity
    log_dir = args.output_root.resolve() / '_launcher' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'-{os.getpid()}')
    log_dir.mkdir(parents=True)
    write_json(log_dir / 'plan.json', {'source': source_identity(), 'stage': args.stage,
               'commands': [command for _, command in jobs], 'max_seconds': args.max_seconds})
    started = time.monotonic()
    process = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    records = []
    try:
        for index, (output, command) in enumerate(jobs):
            if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
                raise TimeoutError('Overall experiment time limit reached')
            if source_identity() != json.loads((log_dir / 'plan.json').read_text())['source']:
                raise RuntimeError('Source changed during this plan; refusing to mix code versions')
            log = log_dir / f'job{index:02d}.log'
            print(f'Starting {output}; log={log}', flush=True)
            with log.open('x') as handle:
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                write_json(log_dir / 'status.json', {'status': 'running', 'pid': process.pid,
                           'output_dir': str(output), 'log': str(log), 'completed': records})
                while process.poll() is None:
                    if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
                        raise TimeoutError('Overall experiment time limit reached')
                    time.sleep(2)
            records.append({'output_dir': str(output), 'exit_code': process.returncode, 'log': str(log)})
            if process.returncode:
                raise RuntimeError(f'Training exited {process.returncode}; see {log}')
            # Verify saved checkpoint re-evaluation before spending on another run.
            with log.open('a') as handle:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'evaluate.py'), '--run-dir', str(output)],
                                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(log_dir / 'status.json', {'status': 'evaluating', 'pid': process.pid,
                           'output_dir': str(output), 'log': str(log), 'completed': records})
                while process.poll() is None:
                    if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
                        raise TimeoutError('Overall experiment time limit reached during re-evaluation')
                    time.sleep(2)
            if process.returncode:
                raise RuntimeError(f'Checkpoint evaluation failed; see {log}')
            result = json.loads((output / 'result.json').read_text())
            repeated = json.loads((output / 'reevaluation.json').read_text())
            for key in ('avg_acc', 'forgetting', 'routing_batch', 'routing_sample'):
                if ((result[key] is None or repeated[key] is None) and result[key] != repeated[key]) or (
                        result[key] is not None and repeated[key] is not None and abs(result[key] - repeated[key]) > 1e-5):
                    raise RuntimeError(f'Re-evaluation disagrees on {key}: {output}')
    except BaseException as exc:
        if process is not None:
            stop_group(process)
        write_json(log_dir / 'status.json', {'status': 'stopped', 'reason': str(exc), 'completed': records})
        raise
    write_json(log_dir / 'status.json', {'status': 'completed', 'completed': records,
               'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
