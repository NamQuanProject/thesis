import copy

import pytest
import torch

from lib.coupled_tangent import CoupledTangentState
from lib.jacobians import build_ia3_jacobian
from lib.ia3 import collect_ia3_params


def test_sketch_solves_joint_projected_objective_and_preserves_cross_blocks():
    torch.manual_seed(7)
    parameters = {'a': torch.zeros(3, dtype=torch.double), 'b': torch.zeros(2, dtype=torch.double)}
    state = CoupledTangentState(parameters, rank=4)
    a, b = torch.randn(9, 5, dtype=torch.double), torch.randn(9, dtype=torch.double)
    projection = torch.randn(4, 9, dtype=torch.double) / 2
    for ids in (slice(0, 4), slice(4, 9)):
        state.update(a[ids], b[ids], projection[:, ids])
    x, y = projection @ a, projection @ b
    expected = torch.linalg.solve(x.T @ x + 0.7 * torch.eye(5, dtype=torch.double), x.T @ y)
    actual = torch.cat(list(state.solve(0.7).values()))
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    # Fixed designs/targets and fixed sketch columns give additive state under reorder.
    reversed_state = CoupledTangentState(parameters, rank=4)
    for ids in (slice(4, 9), slice(0, 4)):
        reversed_state.update(a[ids], b[ids], projection[:, ids])
    torch.testing.assert_close(reversed_state.design, state.design)
    torch.testing.assert_close(reversed_state.target, state.target)


def test_identical_block_counterexample_matches_exact_joint_solution():
    state = CoupledTangentState({'a': torch.zeros(1), 'b': torch.zeros(1)}, rank=1)
    state.update(torch.ones(1, 2), torch.ones(1), torch.ones(1, 1))
    solution = torch.cat(list(state.solve(1.0).values()))
    torch.testing.assert_close(solution, torch.full((2,), 1 / 3))
    independent = torch.full((2,), 1 / 2)
    objective = lambda w: (w.sum() - 1).square() + w.square().sum()
    assert objective(solution) < objective(independent)


def test_sketch_uses_private_rng_and_reset_discards_previous_segment():
    torch.manual_seed(9)
    before = torch.random.get_rng_state().clone()
    state = CoupledTangentState({'a': torch.ones(3)}, rank=4, seed=4)
    state.update(torch.ones(2, 3), torch.ones(2))
    assert torch.equal(before, torch.random.get_rng_state())
    state.reset()
    assert state.samples == 0 and not state.design.any() and not state.target.any()


@pytest.mark.parametrize('chunk', [0, 1, 2])
def test_per_sample_jacobian_matches_reference_on_actual_ia3_vit(chunk):
    from models.vision_transformer import VisionTransformer
    from main import Net
    from lib.config import cfg
    cfg.device = torch.device('cpu')
    torch.manual_seed(3)
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=1,
                              num_heads=2, num_classes=6).double().eval()
    net = Net(model).eval()
    for p in net.parameters():
        p.requires_grad = False
    params = collect_ia3_params(net)
    x, y = torch.randn(5, 3, 32, 32, dtype=torch.double), torch.tensor([0, 1, 2, 3, 1])
    reference = build_ia3_jacobian(net, 'batch', 1)(params, x, y)
    actual = build_ia3_jacobian(net, 'per_sample', chunk)(params, x, y)
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-9, atol=1e-11)


def test_efficient_head_loop_preserves_parameters_optimizer_and_report():
    from models.vision_transformer import VisionTransformer
    from lib.train import train_one_batch
    from lib.config import cfg
    cfg.dtask.nb_classes, cfg.dtask.epochs = 6, 4
    torch.manual_seed(5)
    original = VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=1,
                                 num_heads=2, num_classes=6)
    for name, param in original.named_parameters():
        param.requires_grad = 'head' in name
    x, y = torch.rand(4, 3, 32, 32), torch.tensor([1, 3, 3, 5])
    models, optimizers, reports = [], [], []
    for efficient in (False, True):
        model = copy.deepcopy(original)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        reports.append(train_one_batch(model, torch.nn.CrossEntropyLoss(), x, y, optimizer,
                                       torch.device('cpu'), max_norm=1, class_mask=[1, 3, 5],
                                       cache_features=True, efficient_loop=efficient))
        models.append(model)
        optimizers.append(optimizer)
    assert reports[0] == reports[1]
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
