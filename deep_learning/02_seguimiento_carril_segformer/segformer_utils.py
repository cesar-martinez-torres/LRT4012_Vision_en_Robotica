"""Shared RGB preprocessing, paired augmentation and pixel metrics."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

CLASSES = ['background', 'lane']
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def preprocess(image):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).copy())


class LaneDataset(Dataset):
    def __init__(self, root, split, augment=False, limit_per_track=0):
        self.root, self.augment = Path(root), augment
        with (self.root / 'metadata.csv').open(encoding='utf-8', newline='') as handle:
            rows = [r for r in csv.DictReader(handle) if r['split'] == split]
        if limit_per_track:
            selected = []
            for track in sorted({r['track_id'] for r in rows}):
                members = [r for r in rows if r['track_id'] == track]
                indices = np.linspace(0, len(members) - 1, min(limit_per_track, len(members)), dtype=int)
                selected.extend(members[i] for i in indices)
            rows = selected
        self.rows = rows
        if not rows:
            raise ValueError(f'Empty dataset split: {split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['image']) as file:
            image = file.convert('RGB')
        with Image.open(self.root / row['mask']) as file:
            mask = file.copy()
        if self.augment:
            # Binary lane labels are invariant to reflection; transform both views.
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        target = np.asarray(mask, dtype=np.int64)
        if not np.isin(target, [0, 1]).all():
            raise ValueError(f'Mask must contain class IDs 0/1: {row["mask"]}')
        return preprocess(image), torch.from_numpy(target.copy()), index


def full_resolution_logits(model, images):
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix).astype(float)
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    union = actual + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(2), where=union > 0)
    dice = np.divide(2 * true_positive, actual + predicted, out=np.zeros(2), where=(actual + predicted) > 0)
    recall = np.divide(true_positive, actual, out=np.zeros(2), where=actual > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(2), where=predicted > 0)
    return {'pixel_accuracy': float(true_positive.sum() / max(matrix.sum(), 1)),
            'mean_iou': float(iou.mean()), 'lane_iou': float(iou[1]),
            'lane_dice': float(dice[1]), 'confusion_matrix': matrix.tolist(),
            'per_class': {name: {'iou': float(iou[i]), 'dice': float(dice[i]),
                                  'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'support_pixels': int(actual[i])}
                          for i, name in enumerate(CLASSES)}}


def visible_mask(mask):
    # Palette changes display, not stored integer labels.
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8)).convert('P')
    image.putpalette([0, 0, 0, 255, 255, 255] + [0] * (256 * 3 - 6))
    return image
