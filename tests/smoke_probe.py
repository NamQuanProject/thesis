"""Run scripts/routing_probe.py on the synthetic stream of tests/smoke_pipeline.py (no downloads)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

if __name__ == "__main__":
    import data.dataset as dataset_mod
    from tests.smoke_pipeline import fake_get_dataset
    import routing_probe

    dataset_mod.get_dataset = fake_get_dataset
    cache = os.path.join(tempfile.mkdtemp(), "feats.pt")
    sys.argv = ["routing_probe.py", "--cfg", "configs/cifar-10-seed-15.yml", "--cache", cache, "--quick",
                "--set", "dtask.nb_classes=6", "continual.n_tasks=3", "dtask.num_workers=0", "signature.n_clusters=2"]
    routing_probe.main()
    print("SMOKE PASS: routing_probe")
