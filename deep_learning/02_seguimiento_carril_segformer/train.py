"""Fine-tune SegFormer-B0: audit -> head -> encoder -> best validation -> test."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from segformer_utils import CLASSES, MEAN, STD, LaneDataset, full_resolution_logits, metrics_from_confusion, visible_mask

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'lane_segmentation_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'segformer_b0')
    parser.add_argument('--pretrained', default='nvidia/mit-b0')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fine-tune-lr', type=float, default=6e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=4012)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--limit-per-track', type=int, default=0)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Optional CUDA float16; default is float32 for numeric stability.')
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.head_epochs <= args.epochs or args.batch_size < 1 or args.limit_per_track < 0:
        parser.error('Invalid epochs, head epochs, batch size or limit')
    return args


def run_epoch(model, loader, device, scaler, optimizer=None, frozen=False, use_amp=False):
    training = optimizer is not None
    model.train(training)
    if training and frozen:
        model.segformer.eval()
    matrix = torch.zeros((2, 2), dtype=torch.int64, device=device)
    total_loss, count = 0.0, 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == 'cuda'):
                logits = full_resolution_logits(model, images)
                loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss. Retry without --amp; check inputs and pretrained weights.')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
        prediction = logits.detach().argmax(1)
        matrix += torch.bincount((2 * targets + prediction).flatten(), minlength=4).reshape(2, 2)
        total_loss += loss.item() * images.shape[0]
        count += images.shape[0]
    result = metrics_from_confusion(matrix.cpu().numpy())
    result['loss'] = total_loss / count
    return result


@torch.inference_mode()
def evaluate_test(model, loader, device, output):
    model.eval()
    matrix = np.zeros((2, 2), dtype=np.int64)
    rows = []
    prediction_dir = output / 'test_masks'
    prediction_dir.mkdir()
    total_loss, count = 0.0, 0
    for images, targets, indices in loader:
        logits = full_resolution_logits(model, images.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite logits in test evaluation')
        total_loss += F.cross_entropy(logits, targets.to(device)).item() * images.shape[0]
        count += images.shape[0]
        masks = logits.argmax(1).cpu().numpy()
        for mask, target, index in zip(masks, targets.numpy(), indices.tolist()):
            confusion = np.bincount((2 * target + mask).ravel(), minlength=4).reshape(2, 2)
            matrix += confusion
            sample = loader.dataset.rows[index]
            metrics = metrics_from_confusion(confusion)
            filename = Path(sample['mask']).name
            visible_mask(mask).save(prediction_dir / filename)
            rows.append({'image': sample['image'], 'ground_truth': sample['mask'],
                         'prediction': f'test_masks/{filename}', 'track_id': sample['track_id'],
                         'lane_iou': metrics['lane_iou'], 'lane_dice': metrics['lane_dice'],
                         'pixel_accuracy': metrics['pixel_accuracy'],
                         'tn': int(confusion[0,0]), 'fp': int(confusion[0,1]),
                         'fn': int(confusion[1,0]), 'tp': int(confusion[1,1])})
    with (output / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'confusion_matrix.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual/predicted', *CLASSES])
        for name, row in zip(CLASSES, matrix.tolist()):
            writer.writerow([name, *row])
    metrics = metrics_from_confusion(matrix)
    metrics.update(loss=total_loss / count, samples=count)
    return metrics


def main():
    args = parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output contains a previous experiment: {args.output}; use a new --output')
    report = json.loads((args.data / 'audit_report.json').read_text(encoding='utf-8'))
    if report['status'] != 'PASS' or not (args.data / 'capture_complete.json').exists():
        raise ValueError('A complete, audited dataset is required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if (args.device == 'cuda' or args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    datasets = {s: LaneDataset(args.data, s, augment=s == 'train', limit_per_track=args.limit_per_track)
                for s in ['train', 'val'] + ([] if args.skip_test else ['test'])}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=s == 'train',
                             num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                             generator=generator if s == 'train' else None)
               for s, ds in datasets.items()}
    print(f'loading_pretrained={args.pretrained} device={device}', flush=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained, num_labels=2, id2label=dict(enumerate(CLASSES)),
        label2id={name: i for i, name in enumerate(CLASSES)}, ignore_mismatched_sizes=True).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': 'segformer_b0', 'classes': CLASSES, 'image_size': [192, 320],
              'normalization_mean': MEAN.ravel().tolist(), 'normalization_std': STD.ravel().tolist(),
              'do_reduce_labels': False, 'device': str(device),
              'cuda_device': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'pretrained_revision': getattr(model.config, '_commit_hash', None),
              'split_sizes': {s: len(ds) for s, ds in datasets.items()},
              'selection_metric': 'val_mean_iou', 'loss': 'cross_entropy',
              'augmentation': 'paired horizontal reflection + RGB-only brightness/contrast/color',
              'mask_labels': [0, 1], 'task': 'semantic_segmentation'}
    save_json(args.output / 'config.json', config)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == 'cuda')
    history, optimizer, previous_phase = [], None, None
    best_score, best_epoch = -1.0, -1
    started = time.perf_counter()
    baseline = run_epoch(model, loaders['val'], device, scaler)
    save_json(args.output / 'baseline_validation.json', baseline)
    print(f'baseline_val_miou={baseline["mean_iou"]:.4f}', flush=True)
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.head_epochs
        phase = 'decoder' if frozen else 'fine_tuning'
        if phase != previous_phase:
            for parameter in model.parameters():
                parameter.requires_grad = True
            if frozen:
                for parameter in model.segformer.parameters():
                    parameter.requires_grad = False
            lr = args.lr if frozen else args.fine_tune_lr
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=args.weight_decay)
            previous_phase = phase
        epoch_start = time.perf_counter()
        train = run_epoch(model, loaders['train'], device, scaler, optimizer, frozen, args.amp)
        val = run_epoch(model, loaders['val'], device, scaler)
        row = {'epoch': epoch, 'phase': phase, 'learning_rate': lr,
               **{f'{s}_{k}': result[k] for s, result in [('train', train), ('val', val)]
                  for k in ['loss', 'pixel_accuracy', 'mean_iou', 'lane_iou', 'lane_dice']},
               'seconds': time.perf_counter() - epoch_start}
        history.append(row)
        if val['mean_iou'] > best_score:
            best_score, best_epoch = val['mean_iou'], epoch
            torch.save({'model_state_dict': model.state_dict(), 'model_config': model.config.to_dict(),
                        'classes': CLASSES, 'best_epoch': epoch, 'best_val_mean_iou': best_score,
                        'image_size': [192, 320], 'normalization_mean': MEAN.ravel().tolist(),
                        'normalization_std': STD.ravel().tolist(), 'do_reduce_labels': False},
                       args.output / 'best_model.pth')
        with (args.output / 'history.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f'epoch={epoch:02d}/{args.epochs} phase={phase} train_loss={train["loss"]:.4f} '
              f'val_loss={val["loss"]:.4f} train_miou={train["mean_iou"]:.4f} '
              f'val_miou={val["mean_iou"]:.4f} lane_iou={val["lane_iou"]:.4f} '
              f'seconds={row["seconds"]:.1f} best={best_epoch}', flush=True)
    checkpoint = torch.load(args.output / 'best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.save_pretrained(args.output / 'best_model', safe_serialization=True)
    results = {'best_epoch': best_epoch, 'best_val_mean_iou': best_score,
               'training_seconds': time.perf_counter() - started,
               'baseline_validation': baseline,
               'test': None if args.skip_test else evaluate_test(model, loaders['test'], device, args.output)}
    if device.type == 'cuda':
        results['peak_cuda_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
    save_json(args.output / 'metrics.json', results)
    print(f'complete best_epoch={best_epoch} best_val_mean_iou={best_score:.4f} output={args.output}', flush=True)
    if results['test']:
        print(f'test_mean_iou={results["test"]["mean_iou"]:.4f} lane_iou={results["test"]["lane_iou"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
