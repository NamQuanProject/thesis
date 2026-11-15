"""Offline synthetic image fixture and small end-to-end checks (no pretrained weights).

    python tests/smoke_pipeline.py
    python tests/smoke_pipeline.py base class_gauss
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_CLASSES, TRAIN_PER_CLASS, VAL_PER_CLASS = 6, 8, 2


class SyntheticImages:
    """Class-dependent colour blobs + noise; exposes `.targets` / `.classes` like torchvision."""

    def __init__(self, n_per_class, seed, transform=None):
        rng = np.random.default_rng(seed)
        self.classes = [f"c{i}" for i in range(N_CLASSES)]
        self.targets = [c for c in range(N_CLASSES) for _ in range(n_per_class)]
        palette = np.random.default_rng(0).integers(0, 255, size=(N_CLASSES, 3))
        self.images = []
        for c in self.targets:
            img = np.clip(palette[c] + rng.normal(0, 40, size=(32, 32, 3)), 0, 255).astype(np.uint8)
            self.images.append(Image.fromarray(img))
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = self.images[i]
        return (self.transform(x) if self.transform else x), self.targets[i]


def fake_get_dataset(dataset, transform_train, transform_val):
    return SyntheticImages(TRAIN_PER_CLASS, 1, transform_train), SyntheticImages(VAL_PER_CLASS, 2, transform_val)


if __name__ == "__main__":
    import pytest
    from pathlib import Path
    test = str(Path(__file__).with_name('test_experiment_readiness.py')) + '::test_tiny_pipeline_and_checkpoint_reproduce'
    nodes = [test + f'[{name}]' for name in sys.argv[1:]] or [test]
    raise SystemExit(pytest.main(['-q', *nodes]))
