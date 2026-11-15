"""Run identity, durable metadata, and RNG isolation for reproducible experiments."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    """Atomic replacement; reject NaN/Infinity rather than writing invalid JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_identity():
    files = [*ROOT.glob('*.py'), ROOT / 'requirements.txt']
    for folder in ('lib', 'data', 'models', 'utils', 'scripts', 'configs'):
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.suffix in ('.py', '.sh', '.yml', '.yaml', '.txt') and '__pycache__' not in p.parts)
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'git_commit': commit, 'source_sha256': digest.hexdigest()}


def environment():
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy', 'scipy', 'scikit-learn', 'PyYAML', 'easydict'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'cuda_runtime': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def begin_run(cfg):
    """Reserve an empty output directory; never silently reuse failed/completed runs."""
    output = Path(cfg.dtask.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}. Choose a new --output_dir; no automatic resume.')
    # Exclusive creation also prevents concurrent starts after both empty-directory checks.
    with (output / 'run.lock').open('x') as handle:
        handle.write(str(os.getpid()) + '\n')
    write_json(output / 'config.json', {**cfg, 'device': None})
    write_json(output / 'provenance.json', {**source_identity(), **environment(), 'argv': sys.argv})
    finish_run(output, 'running')


def finish_run(output, status, **details):
    write_json(Path(output) / 'status.json', {'status': status,
               'updated_at': datetime.now(timezone.utc).isoformat(), **details})


@contextmanager
def preserve_rng_state():
    """Evaluation and auxiliary keys must not change subsequent training samples/views."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def loader_manifest(loaders, masks, task_order):
    """Hash the selected ordered sample IDs (not image bytes) for each stream."""
    records = []
    for task_id, loader_pair in enumerate(loaders):
        item = {'task_id': task_id, 'classes': masks[task_id]}
        for split, loader in loader_pair.items():
            dataset = loader.dataset
            ids = list(getattr(dataset, 'indices', range(len(dataset))))
            base = getattr(dataset, 'dataset', dataset)
            if hasattr(base, 'partition_metadata'):
                item['partition'] = base.partition_metadata
            samples = getattr(base, 'samples', None)
            if samples is not None:
                base_root = Path(getattr(base, 'root', '.'))
                names = [(str(Path(samples[i][0]).relative_to(base_root)), int(samples[i][1])) for i in ids]
            else:
                names = ids
            item[split] = {'samples': len(dataset), 'sample_ids_sha256':
                           hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest()}
        records.append(item)
    return {'task_order': list(task_order), 'tasks': records,
            'identity_scope': 'ordered sample paths/labels or indices; ImageNet-R manifest separately records file sizes'}
