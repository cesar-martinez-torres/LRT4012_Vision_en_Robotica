"""Verify segmentation pairs, hashes and track splits; create visual QA panels."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lane_geometry import validate_track


def create_preview(root, rows, tracks):
    selected = []
    for split in ('train', 'val', 'test'):
        candidates = [r for r in rows if r['split'] == split]
        ids = sorted({r['track_id'] for r in candidates})
        for track_id in ids[:2]:
            samples = [r for r in candidates if r['track_id'] == track_id]
            selected.append(samples[len(samples) // 3])
    panels = []
    for row in selected:
        rgb = cv2.imread(str(root / row['image']))
        mask = cv2.imread(str(root / row['mask']), cv2.IMREAD_GRAYSCALE)
        colored = np.zeros_like(rgb)
        colored[mask == 1] = (60, 210, 60)
        overlay = rgb.copy()
        overlay[mask == 1] = (0.55 * rgb[mask == 1] + 0.45 * colored[mask == 1]).astype(np.uint8)
        panel = np.hstack([rgb, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), overlay])
        header = np.full((28, panel.shape[1], 3), 245, np.uint8)
        for x, label in ((8, f'{row["split"]}/{row["track_id"]} RGB'), (328, 'Mask: lane=white (stored ID=1)'), (648, 'Ground-truth overlay')):
            cv2.putText(header, label, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, panel]))
    cv2.imwrite(str(root / 'dataset_preview.png'), np.vstack(panels))
    cell = 260
    columns = min(8, len(tracks))
    atlas = np.full((int(np.ceil(len(tracks) / columns)) * cell, columns * cell, 3), 245, np.uint8)
    colors = {'train': (160, 100, 35), 'val': (40, 150, 40), 'test': (40, 90, 190)}
    for index, track in enumerate(tracks):
        x0, y0 = index % columns * cell, index // columns * cell
        points = np.asarray(track['centerline'])
        points = points - points.mean(axis=0)
        scale = 210 / np.ptp(points, axis=0).max()
        projected = np.column_stack([points[:, 0], -points[:, 1]]) * scale + [x0+130, y0+142]
        thickness = max(2, round(track['width_m'] * scale))
        cv2.polylines(atlas, [projected.astype(np.int32)], True, colors[track['split']], thickness, cv2.LINE_AA)
        cv2.putText(atlas, f'{track["track_id"]} / {track["split"]}', (x0+10, y0+20), cv2.FONT_HERSHEY_SIMPLEX, .47, (20,20,20), 1, cv2.LINE_AA)
    cv2.imwrite(str(root / 'tracks_overview.png'), atlas)


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    errors = []
    with (root / 'metadata.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    tracks, ids = [], {}
    for entry in manifest['tracks']:
        track = json.loads((root / entry['file']).read_text(encoding='utf-8'))
        if entry['track_id'] in ids:
            errors.append('Repeated track ID')
        ids[entry['track_id']] = entry['split']
        validate_track(track['centerline'], track['width_m'])
        tracks.append(track)
    if len({t['seed'] for t in tracks}) != len(tracks):
        errors.append('Repeated track seed')
    geometry_hashes = [hashlib.sha256(np.asarray(t['centerline']).tobytes()).hexdigest() for t in tracks]
    if len(set(geometry_hashes)) != len(tracks):
        errors.append('Repeated track geometry')
    counts, track_counts, hist = Counter(), Counter(), np.zeros(2, dtype=np.int64)
    image_hashes, observed_tracks = defaultdict(set), defaultdict(set)
    image_paths, mask_paths, fractions = set(), set(), []
    for row in rows:
        if ids.get(row['track_id']) != row['split']:
            errors.append(f'Split mismatch: {row["image"]}')
        observed_tracks[row['track_id']].add(row['split'])
        expected_shape = (manifest['height'], manifest['width'])
        image = cv2.imread(str(root / row['image']))
        mask = cv2.imread(str(root / row['mask']), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            errors.append(f'Missing/unreadable pair: {row["image"]}')
            continue
        if image.shape != (*expected_shape, 3) or mask.shape != expected_shape or mask.dtype != np.uint8:
            errors.append(f'Wrong dimensions/type: {row["image"]}')
            continue
        if not set(np.unique(mask)).issubset({0, 1}):
            errors.append(f'Invalid mask labels: {row["mask"]}')
            continue
        for key in ('image', 'mask'):
            digest = hashlib.sha256((root / row[key]).read_bytes()).hexdigest()
            if digest != row[f'{key}_sha256']:
                errors.append(f'Hash mismatch: {row[key]}')
            if key == 'image':
                image_hashes[digest].add(row['split'])
        fraction = float(mask.mean())
        if not 0.01 < fraction < 0.98:
            errors.append(f'Empty/extreme mask: {row["mask"]}')
        if abs(fraction - float(row['lane_fraction'])) > 1e-8:
            errors.append(f'RGB/mask metadata mismatch: {row["mask"]}')
        fractions.append(fraction)
        hist += np.bincount(mask.ravel(), minlength=2)
        counts[row['split']] += 1
        track_counts[row['track_id']] += 1
        image_paths.add(row['image'])
        mask_paths.add(row['mask'])
    if len(image_paths) != len(rows) or len(mask_paths) != len(rows):
        errors.append('Repeated pair path')
    if len(rows) != manifest['expected_samples']:
        errors.append('Unexpected sample total')
    if any(track_counts[t] != manifest['samples_per_track'] for t in ids):
        errors.append('Incomplete track')
    if any(len(splits) != 1 for splits in observed_tracks.values()):
        errors.append('Track leakage across splits')
    duplicate_cross_split = sum(len(splits) > 1 for splits in image_hashes.values())
    if duplicate_cross_split:
        errors.append('Exact RGB duplicate across splits')
    for split in ('train', 'val', 'test'):
        for folder, expected in (('images', image_paths), ('masks', mask_paths)):
            actual = {(p.relative_to(root)).as_posix() for p in (root / split / folder).glob('*.png')}
            wanted = {p for p in expected if p.startswith(split + '/')}
            if actual != wanted:
                errors.append(f'Unregistered files in {split}/{folder}')
    if not (root / 'capture_complete.json').is_file():
        errors.append('Capture completion marker absent')
    report = {'status': 'PASS' if not errors else 'FAIL', 'samples': len(rows),
              'counts': dict(counts), 'track_counts': manifest['track_counts'],
              'pixel_counts': hist.tolist(), 'lane_pixel_fraction': float(hist[1] / hist.sum()) if hist.sum() else 0,
              'sample_lane_fraction_min': min(fractions) if fractions else 0,
              'sample_lane_fraction_max': max(fractions) if fractions else 0,
              'cross_split_exact_duplicates': duplicate_cross_split,
              'split_track_overlap': any(len(s) > 1 for s in observed_tracks.values()),
              'errors': errors}
    (root / 'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not errors:
        create_preview(root, rows, tracks)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', type=Path, default=ROOT / 'data' / 'lane_segmentation_v1')
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.dataset.resolve())['status'] == 'PASS' else 1)
