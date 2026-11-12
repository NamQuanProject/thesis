"""End-to-end smoke test of main.learn_continually on a tiny synthetic stream.

No dataset download: `data.dataset.get_dataset` is replaced by a synthetic
6-class image dataset (3 tasks x 2 classes). Needs the ViT-B/16 weights in the
torch hub cache. Usage (project root):

    .venv/bin/python tests/smoke_pipeline.py                # all variants
    .venv/bin/python tests/smoke_pipeline.py baseline       # one variant
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_CLASSES, TRAIN_PER_CLASS, VAL_PER_CLASS = 6, 40, 8


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


VARIANTS = {
    "baseline": [],
    "class_gauss_full": ["improve.routing_key=class_gauss", "improve.routing_space=pretrained",
                         "improve.key_view=clean", "improve.fusion_alpha=1.0", "improve.select_metric=loss",
                         "improve.select_tol=0.001", "improve.refine_max_iters=50"],
    "gmm_pretrained_nll": ["improve.routing_key=gmm", "improve.routing_space=pretrained",
                           "improve.gmm_score=nll", "improve.gmm_var_shrink=10"],
    "adapter_space_misc": ["improve.routing_key=class_gauss", "improve.cov_scope=task", "improve.cov_type=diag",
                           "improve.routing_unit=sample", "improve.accumulate=joint", "improve.fast_solve=false",
                           "improve.fast_head=false"],
}


def run(variant):
    import torch
    import data.dataset as dataset_mod
    import main as main_mod
    import evaluate as eval_mod
    from lib.config import cfg, cfg_from_file, cfg_from_list
    from lib.device import resolve_device

    dataset_mod.get_dataset = fake_get_dataset
    out = tempfile.mkdtemp(prefix=f"hestia_smoke_{variant}_")
    cfg_from_file("configs/cifar-10-seed-15.yml")
    cfg_from_list(["dtask.nb_classes=6", "dtask.batch_size=4", "dtask.epochs=3", "continual.n_tasks=3", "signature.cooldown_limit=2",
                   "signature.window_t=2", "signature.k=3.0", "signature.n_clusters=2",
                   "signature.split_warmup=1000", "dtask.num_workers=0", f"dtask.output_dir={out}/"]
                  + VARIANTS[variant])
    cfg.device = resolve_device(cfg.improve.device, cfg.gpu_ids)
    main_mod.set_seed(cfg.seed)
    final = main_mod.learn_continually()
    # Post-hoc evaluation from the saved checkpoints must reproduce the final
    # in-training evaluation (re-seed so the class order is rebuilt identically,
    # as evaluate.py's main() does).
    main_mod.set_seed(cfg.seed)
    post = eval_mod.run_evaluation()
    assert abs(final['avg_acc'] - post['avg_acc']) < 1e-6, (final['avg_acc'], post['avg_acc'])
    assert abs(final['routing_batch'] - post['routing_batch']) < 1e-9, (final['routing_batch'], post['routing_batch'])
    shutil.rmtree(out)
    print(f"SMOKE PASS: {variant}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        # Fresh interpreter state per variant (cfg is a module-level singleton).
        if len(names) > 1:
            code = os.system(f"{sys.executable} {__file__} {name}")
            assert code == 0, name
        else:
            run(name)
