"""Check that this codebase (default flags) reproduces the ORIGINAL HESTIA code.

Runs one implementation on the synthetic stream of tests/smoke_pipeline.py (no
downloads) and prints its final accuracy matrix and per-batch detection scores.
Run it once for each implementation, then compare:

    python tests/compare_with_original.py --impl orig --orig-root baseline/HESTIA > orig.txt
    python tests/compare_with_original.py --impl new > new.txt
    grep RESULT orig.txt new.txt          # accuracy matrices must match
    diff <(grep -o "mean_ood: .*" orig.txt) <(grep -o "mean_ood: .*" new.txt)   # equal up to ~1e-6 relative

Use PYTHONDONTWRITEBYTECODE=1 to keep __pycache__ out of the original repo.
Verified on 2026-10-26 (Apple MPS): identical matrices, scores within 1.3e-6.
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument("--impl", choices=["orig", "new"], default="new")
_ap.add_argument("--orig-root", default=os.path.join(PROJ, "baseline", "HESTIA"))
_ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
ARGS, _ = _ap.parse_known_args()
ROOT = os.path.abspath(ARGS.orig_root) if ARGS.impl == "orig" else PROJ

# Code roots for this process *and* spawned DataLoader workers (module level on purpose).
sys.path.insert(0, os.path.join(PROJ, "tests"))
from smoke_pipeline import fake_get_dataset  # noqa: E402
sys.path = [ROOT] + [p for p in sys.path if p not in (PROJ, ROOT)]


def main():
    import tempfile
    import numpy as np
    import torch
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None  # the original hard-codes CUDA calls
    import data.dataset as dm
    dm.get_dataset = fake_get_dataset
    import main as main_mod
    assert main_mod.__file__.startswith(ROOT), main_mod.__file__
    from lib.config import cfg, cfg_from_file
    cfg_from_file("configs/cifar-10-seed-15.yml")
    out = tempfile.mkdtemp(prefix=f"cmp_{ARGS.impl}_")
    over = {"dtask.nb_classes": 6, "dtask.batch_size": 4, "dtask.epochs": 3, "continual.n_tasks": 3,
            "signature.cooldown_limit": 2, "signature.window_t": 2, "signature.k": 3.0,
            "signature.n_clusters": 2, "signature.split_warmup": 1000, "dtask.output_dir": out + "/"}
    for k, v in over.items():
        a, b = k.split(".")
        cfg[a][b] = v
    dev = ARGS.device or ("cuda:0" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    cfg.device = torch.device(dev)
    main_mod.set_seed(cfg.seed)
    main_mod.learn_continually()
    acc = torch.load(os.path.join(out, "global_acc_matrix.pth"), weights_only=False)
    print("RESULT", ARGS.impl, "acc_matrix", np.round(acc, 4).tolist(), flush=True)


if __name__ == "__main__":
    main()
