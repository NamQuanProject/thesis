"""Non-destructive ImageNet-R partitions shared by every variant and stream seed.

ImageNet-R has no official training partition. We use a recorded, per-class
80/20 partition. Existing train/test folders are accepted without moving files.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import tarfile
import tempfile
import urllib.request

from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from lib.experiment import write_json

URL = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'


class IndexedImages(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.targets = [dataset.targets[i] for i in indices]
        self.samples = [dataset.samples[i] for i in indices]
        self.root = dataset.root


def download_imagenet_r(root):
    """Explicit preparation only; extract into staging and publish when complete."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'imagenet-r'
    if destination.is_dir():
        return
    archive = root / 'imagenet-r.tar'
    if not archive.exists():
        partial = root / 'imagenet-r.tar.partial'
        urllib.request.urlretrieve(URL, partial)
        os.replace(partial, archive)
    with tempfile.TemporaryDirectory(prefix='imagenet-r-extract-', dir=root) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, 'r:*') as handle:
            members = handle.getmembers()
            for member in members:
                target = (staging / member.name).resolve()
                if not target.is_relative_to(staging) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe archive entry: {member.name}')
            handle.extractall(staging, members=members)
        extracted = staging / 'imagenet-r'
        if not extracted.is_dir():
            raise ValueError('Archive did not contain imagenet-r/')
        os.replace(extracted, destination)


def build_imagenet_r(root, train_transform, test_transform, split_seed=0, download=False):
    root = Path(root).expanduser().resolve()
    folder = root / 'imagenet-r'
    if not folder.is_dir():
        if not download:
            raise FileNotFoundError(f'ImageNet-R not found at {folder}. Run python scripts/prepare_imagenet_r.py --download.')
        download_imagenet_r(root)
    has_train, has_test = (folder / 'train').is_dir(), (folder / 'test').is_dir()
    if has_train != has_test:
        raise ValueError('Incomplete legacy ImageNet-R split: both train/ and test/ are required')
    if has_train:
        train, test = ImageFolder(folder / 'train', train_transform), ImageFolder(folder / 'test', test_transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError('ImageNet-R train/test class mappings differ')
        return train, test, {'protocol': 'existing-train-test-folders', 'split_seed': None}

    train_base = ImageFolder(folder, train_transform)
    test_base = ImageFolder(folder, test_transform)
    inventory = [(str(Path(path).relative_to(folder)), label, Path(path).stat().st_size)
                 for path, label in train_base.samples]
    inventory_hash = hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
    manifest_path = root / f'imagenet-r-split-seed{split_seed}.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('version') != 1 or manifest.get('split_seed') != split_seed
                or manifest.get('inventory_sha256') != inventory_hash):
            raise ValueError(f'Dataset changed or incompatible split manifest: {manifest_path}')
    else:
        rng = random.Random(split_seed)
        train_ids, test_ids = [], []
        for label in range(len(train_base.classes)):
            ids = [i for i, target in enumerate(train_base.targets) if target == label]
            if len(ids) < 2:
                raise ValueError('Every ImageNet-R class needs at least two images to split')
            rng.shuffle(ids)
            boundary = min(len(ids) - 1, max(1, int(0.8 * len(ids))))
            train_ids.extend(ids[:boundary])
            test_ids.extend(ids[boundary:])
        manifest = {'version': 1, 'protocol': 'stratified-80-20-v1', 'split_seed': split_seed,
                    'inventory_sha256': inventory_hash, 'classes': train_base.classes,
                    'train_indices': sorted(train_ids), 'test_indices': sorted(test_ids)}
        write_json(manifest_path, manifest)
    train_ids, test_ids = manifest['train_indices'], manifest['test_indices']
    if (len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids)
            or set(train_ids) & set(test_ids)
            or set(train_ids) | set(test_ids) != set(range(len(inventory)))):
        raise ValueError(f'Invalid or overlapping ImageNet-R split: {manifest_path}')
    metadata = {k: v for k, v in manifest.items() if k not in ('train_indices', 'test_indices')}
    metadata['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return IndexedImages(train_base, train_ids), IndexedImages(test_base, test_ids), metadata
