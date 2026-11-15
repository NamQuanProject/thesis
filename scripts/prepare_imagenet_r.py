"""Prepare ImageNet-R and weights explicitly, before timed training starts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='local_datasets')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--download', action='store_true', help='allow downloading the ImageNet-R archive')
    parser.add_argument('--weights', action='store_true', help='download/cache the configured pretrained ViT weights')
    args = parser.parse_args()
    from data.imagenet_r import build_imagenet_r
    train, test, metadata = build_imagenet_r(args.data_path, None, None, args.split_seed, args.download)
    if len(train.classes) != 200:
        raise ValueError(f'Expected the 200-class ImageNet-R benchmark, found {len(train.classes)} classes')
    print(json.dumps({'classes': len(train.classes), 'train_samples': len(train),
                      'test_samples': len(test), 'partition': metadata}, indent=2), flush=True)
    if args.weights:
        import models.model_register  # register the project's IA3 model
        from timm.models import create_model
        model = create_model('vit_base_patch16_224', pretrained=True, num_classes=200,
                             drop_rate=0.0, drop_path_rate=0.0, head_type='prompt')
        print('Pretrained ViT weights loaded successfully; parameters:', sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    main()
