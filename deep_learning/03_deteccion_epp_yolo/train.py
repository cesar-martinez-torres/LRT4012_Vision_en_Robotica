"""Fine-tune COCO-pretrained YOLO11n in two stages; select using validation only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def evaluate(weights, data, output, split='val', size=640, batch=8, device='cpu'):
    model = YOLO(weights)
    metrics = model.val(data=str(Path(data).resolve()), split=split, imgsz=size,
                        batch=batch, device=device, workers=0, conf=.001, iou=.7,
                        rect=False, plots=True, project=str(output), name=split, exist_ok=True)
    report = {'split': split, 'checkpoint_sha256': sha256(weights),
              'confidence_floor': .001, 'nms_iou': .7, 'rectangular_batches': False,
              'metrics': {k: float(v) for k, v in metrics.results_dict.items()},
              'per_class': metrics.summary(), 'speed_ms': metrics.speed,
              'matrix_predicted_rows_reference_columns': metrics.confusion_matrix.matrix.tolist(),
              'note': 'AP sweeps confidence. The native confusion matrix is not the selected deployment operating point.'}
    save_json(output / f'metrics_{split}.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/ppe_four/dataset.yaml')
    parser.add_argument('--weights', default=str(ROOT/'data/pretrained/yolo11n.pt'))
    parser.add_argument('--output', type=Path, default=ROOT/'runs/yolo11n_ppe')
    parser.add_argument('--head-epochs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40, help='Maximum epochs for full fine-tuning.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=4012)
    parser.add_argument('--class-weight-power', type=float, default=.5)
    args = parser.parse_args()
    if min(args.head_epochs, args.epochs, args.batch_size) <= 0:
        parser.error('Epochs and batch size must be positive.')
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error('--imgsz must be a positive multiple of 32.')
    if not 0 <= args.class_weight_power <= 1:
        parser.error('--class-weight-power must be between 0 and 1.')
    if args.output.exists():
        raise FileExistsError(f'Experiment exists: {args.output}. Choose a new --output.')
    from prepare_dataset import audit
    audit_report = audit(args.data.parent)
    if args.weights == str(ROOT/'data/pretrained/yolo11n.pt'):
        from prepare_dataset import ensure_pretrained
        ensure_pretrained()
    args.output.mkdir(parents=True)
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(2)
    model = YOLO(args.weights)
    if model.model.model[-1].__class__.__name__ != 'Detect' or len(model.model.model) != 24:
        raise ValueError('This freezing recipe requires the YOLO11 detection architecture.')
    pretrained = Path(model.ckpt_path)
    save_json(args.output/'recipe.json', {**vars(args), 'device_selected': device,
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'python': platform.python_version(), 'pretrained_sha256': sha256(pretrained),
        'manifest_sha256': audit_report['manifest_sha256'], 'classes': audit_report['classes'],
        'stages': 'adaptation: freeze modules 0-10, train neck/head; finetune: all trainable modules',
        'precision': 'FP32', 'selection_criterion': 'Maximum validation mAP50-95 across stage checkpoints'})
    phases = [('adaptation', args.head_epochs, 11, .001), ('finetune', args.epochs, 0, .0003)]
    candidates = []
    for name, epochs, freeze, rate in phases:
        started = time.perf_counter()
        if name == 'finetune':
            model = YOLO(args.output/'adaptation/weights/best.pt')
        print(f'STAGE={name} epochs={epochs} freeze={freeze} lr={rate}', flush=True)
        model.train(data=str(args.data.resolve()), epochs=epochs, imgsz=args.imgsz,
                    batch=args.batch_size, device=device, workers=0,
                    project=str(args.output.resolve()), name=name,
                    optimizer='AdamW', lr0=rate, lrf=.1, freeze=freeze,
                    cls_pw=args.class_weight_power,
                    warmup_epochs=1, patience=8, weight_decay=.0005,
                    seed=args.seed, deterministic=True, amp=False, cache=False,
                    mosaic=.4 if name=='adaptation' else .2, close_mosaic=min(5, epochs),
                    fliplr=.5, degrees=5, translate=.08, scale=.25,
                    hsv_h=.015, hsv_s=.35, hsv_v=.3, plots=True, save=True,
                    exist_ok=False)
        run = args.output/name
        report = evaluate(run/'weights/best.pt', args.data, run, 'val', args.imgsz, args.batch_size, device)
        save_json(run/'stage.json', {'stage': name, 'seconds': time.perf_counter()-started,
                                   'freeze': freeze, 'learning_rate': rate, 'epochs_requested': epochs})
        candidates.append({'stage': name, 'run': name, 'validation': report['metrics']})
    selected = max(candidates, key=lambda item:item['validation']['metrics/mAP50-95(B)'])
    checkpoint = args.output/selected['run']/'weights/best.pt'
    save_json(args.output/'selection.json', {
        'selected_at_utc': datetime.now(timezone.utc).isoformat(),
        'criterion': 'Maximum validation mAP50-95; test is unused during training and selection.',
        'selected_run': selected['run'], 'checkpoint_sha256': sha256(checkpoint),
        'manifest_sha256': audit_report['manifest_sha256'], 'candidates': candidates})
    print(f'Selected checkpoint: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
