import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from lib.config import cfg, cfg_from_file, cfg_from_list, validate_config
from lib.experiment import begin_run, preserve_rng_state
from lib.signatures import OnlineDiagonalGMM


def test_configuration_rejects_unsupported_combinations():
    cfg_from_file('configs/imgn-r-seed-15.yml')
    validate_config()
    with pytest.raises(ValueError, match='invalid boolean'):
        cfg_from_list(['dtask.pretrained=treu'])
    cfg_from_list(['improve.fusion_alpha=1'])
    with pytest.raises(ValueError, match='fusion requires'):
        validate_config()


def test_no_overwrite_and_configuration_roundtrip(tmp_path):
    cfg_from_file('configs/imgn-r-seed-15.yml')
    cfg.dtask.output_dir = str(tmp_path / 'run')
    cfg.device = torch.device('cpu')
    begin_run(cfg)
    with pytest.raises(FileExistsError):
        begin_run(cfg)
    cfg.device = None
    cfg_from_file(tmp_path / 'run/config.json')
    validate_config()


def test_auxiliary_work_does_not_change_rngs():
    import random
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = random.random(), np.random.rand(), torch.rand(3)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with preserve_rng_state():
        random.random()
        np.random.rand(100)
        torch.rand(100)
    actual = random.random(), np.random.rand(), torch.rand(3)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_imagenet_partition_is_disjoint_stable_and_non_destructive(tmp_path):
    from data.imagenet_r import build_imagenet_r
    for cls in ('n0001', 'n0002'):
        folder = tmp_path / 'imagenet-r' / cls
        folder.mkdir(parents=True)
        for i in range(10):
            Image.new('RGB', (8, 8), (i, 0, 0)).save(folder / f'{i}.png')
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*.png'))
    a, b, meta = build_imagenet_r(tmp_path, None, None)
    assert len(a) == 16 and len(b) == 4
    assert not set(a.indices) & set(b.indices)
    torch.manual_seed(999)
    np.random.seed(999)
    x, y, again = build_imagenet_r(tmp_path, None, None)
    assert x.indices == a.indices and y.indices == b.indices and again == meta
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*.png')) == before
    Image.new('RGB', (8, 8)).save(tmp_path / 'imagenet-r/n0001/new.png')
    with pytest.raises(ValueError, match='Dataset changed'):
        build_imagenet_r(tmp_path, None, None)


def test_gmm_can_start_on_short_boundary_batch():
    model = OnlineDiagonalGMM(14, 4, cfg=cfg)
    model.update(torch.ones(1, 4), 0)
    assert model.K == 1
    assert torch.isfinite(model.score(torch.zeros(2, 4))).all()
    model.reset()
    assert model.K == 14 and model.means.shape == (14, 4)


@pytest.mark.parametrize('variant', ['base', 'class_gauss', 'class_gauss_sample', 'gmm_pretrained', 'shuffled',
                                   'coupled', 'coupled_raw', 'coupled_gauss'])
def test_tiny_pipeline_and_checkpoint_reproduce(tmp_path, monkeypatch, variant):
    import main as training
    import evaluate
    import data.dataset as data
    from models.vision_transformer import VisionTransformer
    from tests.smoke_pipeline import SyntheticImages
    from lib.checkpointing import load_tensor_state

    def tiny_model(*args, **kwargs):
        return VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=1,
                                  num_heads=2, num_classes=kwargs['num_classes'])
    monkeypatch.setattr(training, 'create_model', tiny_model)
    monkeypatch.setattr(evaluate, 'create_model', tiny_model)
    monkeypatch.setattr(data, 'build_transform', lambda is_train: transforms.ToTensor())
    monkeypatch.setattr(data, 'get_dataset', lambda name, aug, clean:
                        (SyntheticImages(8, 1, aug), SyntheticImages(2, 2, clean)))
    # This regression fixture forces convergence to exercise the boundary bookkeeping.
    monkeypatch.setattr(training, 'detect_convergence_stationary_streaming', lambda *a, **k: (True, 0))
    cfg_from_file('configs/cifar-10-seed-15.yml')
    cfg_from_list(['dtask.nb_classes=6', 'continual.n_tasks=3', 'dtask.batch_size=4',
                   'dtask.epochs=1', 'dtask.num_workers=0', 'dtask.pretrained=false',
                   'signature.n_clusters=2', 'signature.cooldown_limit=1', 'improve.refine_max_iters=2'])
    if variant.startswith('class_gauss'):
        cfg_from_list(['improve.routing_key=class_gauss', 'improve.routing_space=pretrained', 'improve.key_view=clean'])
    if variant.endswith('sample'):
        cfg.improve.routing_unit = 'sample'
    if variant == 'gmm_pretrained':
        cfg.improve.routing_space = 'pretrained'
        cfg.improve.gmm_score = 'nll'
    if variant == 'shuffled':
        cfg.continual.shuffle_task = True
    if variant.startswith('coupled'):
        cfg_from_list(['improve.tangent_solver=coupled_sketch', 'improve.tangent_sketch_rank=8',
                       'runtime.jacobian_mode=per_sample', 'runtime.jacobian_chunk_size=0',
                       'runtime.efficient_head_loop=true', 'runtime.empty_cache_each_batch=false'])
        if variant == 'coupled_raw':
            cfg.improve.tangent_residual_target = False
        if variant == 'coupled_gauss':
            cfg_from_list(['improve.routing_key=class_gauss', 'improve.routing_space=pretrained',
                           'improve.key_view=clean'])
    cfg.dtask.output_dir = str(tmp_path / variant)
    cfg.device = torch.device('cpu')
    begin_run(cfg)
    training.set_seed(cfg.seed)
    final = training.learn_continually()
    bundle = load_tensor_state(tmp_path / variant / 'evaluation_checkpoint.pth')
    assert len(bundle['adapters']) == 3
    assert sorted(c for mask in bundle['class_mask'].values() for c in mask) == list(range(6))
    assert torch.isfinite(bundle['acc_matrix']).all()
    assert not list((tmp_path / variant).rglob('global_At_A.pth'))
    repeated = evaluate.run_evaluation()
    for metric in ('avg_acc', 'forgetting', 'routing_batch', 'routing_sample'):
        assert repeated[metric] == pytest.approx(final[metric], abs=1e-6)


def test_launcher_prints_without_training(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, 'scripts/run_imagenet_r.py', '--stage', 'pilot',
                             '--output-root', str(tmp_path / 'runs')], capture_output=True, text=True, check=True)
    assert 'runtime.max_tasks=2' in result.stdout
    assert 'improve.routing_key=class_gauss' in result.stdout
    assert not (tmp_path / 'runs').exists()
