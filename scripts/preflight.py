"""Check the environment before renting time to a long experiment. No downloads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--data-path', default='local_datasets')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    import torch
    import torchvision
    import timm
    from lib.device import resolve_device
    from lib.experiment import environment, source_identity, write_json
    from models.vision_transformer import VisionTransformer
    from lib.ia3 import collect_ia3_params
    from torch.func import functional_call, jacrev

    device = resolve_device(args.device)
    # Exercise the actual custom attention and differentiation on a tiny model.
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=1,
                              num_heads=2, num_classes=6).to(device).eval()
    x = torch.rand(2, 3, 32, 32, device=device)
    params = collect_ia3_params(model)
    jac = jacrev(lambda p: functional_call(model, p, (x,))['logits'][:, 0], chunk_size=1)(params)
    if not all(torch.isfinite(value).all() for value in jac.values()):
        raise RuntimeError('Non-finite Jacobian in preflight')
    a = torch.randn(8, 4, device=device)
    gram = a.T @ a + torch.eye(4, device=device)
    torch.linalg.cholesky(gram.cpu() if device.type == 'mps' else gram)
    report = {**environment(), **source_identity(), 'device': str(device),
              'tiny_ia3_jacobian': 'passed', 'ridge_factorization': 'passed'}
    if args.check_data:
        from data.imagenet_r import build_imagenet_r
        train, test, partition = build_imagenet_r(args.data_path, None, None)
        if len(train.classes) != 200:
            raise ValueError('The requested benchmark requires 200 ImageNet-R classes')
        report['data'] = {'train_samples': len(train), 'test_samples': len(test), 'partition': partition}
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
