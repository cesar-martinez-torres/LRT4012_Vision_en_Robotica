"""Audit the lane dataset and export visible masks without changing class IDs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from audit_dataset import audit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'lane_segmentation_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'dataset_analysis')
    args = parser.parse_args()
    report = audit(args.data.resolve())
    if report['status'] != 'PASS':
        raise ValueError('Dataset failed audit')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.data / 'metadata.csv')
    manifest = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    splits = ['train', 'val', 'test']
    summary = {'audit': report, 'by_split': {}, 'mask_display': 'Palette previews: indices remain 0 and 1, displayed black and white.'}
    for split in splits:
        rows = metadata[metadata['split'] == split]
        summary['by_split'][split] = {'tracks': int(rows.track_id.nunique()), 'samples': len(rows),
            'mean_lane_fraction': float(rows.lane_fraction.mean()),
            'min_lane_fraction': float(rows.lane_fraction.min()), 'max_lane_fraction': float(rows.lane_fraction.max()),
            'clockwise_samples': int((rows.direction == -1).sum()),
            'counterclockwise_samples': int((rows.direction == 1).sum())}
        destination = args.output / 'mask_previews' / split
        destination.mkdir(parents=True, exist_ok=True)
        for row in rows.itertuples():
            raw = np.asarray(Image.open(args.data / row.mask))
            view = Image.fromarray(raw).convert('P')
            view.putpalette([0,0,0,255,255,255] + [0] * (256*3-6))
            file = destination / Path(row.mask).name
            view.save(file)
            if not np.array_equal(np.asarray(Image.open(file)), raw):
                raise AssertionError('Palette preview changed raw class IDs')
    tracks = [json.loads((args.data / e['file']).read_text(encoding='utf-8')) for e in manifest['tracks']]
    summary['geometry'] = {field: {'min': float(min(t[field] for t in tracks)), 'max': float(max(t[field] for t in tracks))}
                           for field in ['width_m', 'length_m', 'min_curve_radius_m']}
    (args.output / 'dataset_analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(splits, [summary['by_split'][s]['samples'] for s in splits], color=['#496d89','#7a8b52','#a47756'])
    axes[0].set(title='Imágenes por partición', ylabel='Pares imagen–máscara')
    for split in splits:
        rows = metadata[metadata['split'] == split]
        axes[1].hist(rows.lane_fraction, bins=20, alpha=.4, label=split, density=True)
    axes[1].set(title='Cobertura del carril', xlabel='Fracción de píxeles', ylabel='Densidad')
    axes[1].legend()
    axes[2].bar(['Fondo', 'Carril'], report['pixel_counts'], color=['#777777','#496d89'])
    axes[2].set(title='Balance de píxeles', ylabel='Píxeles')
    fig.tight_layout()
    fig.savefig(args.output / 'dataset_analysis.png', dpi=160)
    plt.close(fig)
    # Select a curve from each split, displaying the stored labels at full contrast.
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5))
    for i, split in enumerate(splits):
        rows = metadata[metadata['split'] == split]
        row = rows.iloc[len(rows)//3 + 4]
        rgb = np.asarray(Image.open(args.data / row.image).convert('RGB'))
        mask = np.asarray(Image.open(args.data / row['mask']))
        axes[i,0].imshow(rgb)
        axes[i,1].imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        overlay = rgb.copy()
        overlay[mask == 1] = (.60 * rgb[mask == 1] + .40 * np.array([40,220,60])).astype(np.uint8)
        axes[i,2].imshow(overlay)
        for j, title in enumerate(['RGB', 'Ground truth: 0 negro / 1 blanco', 'Superposición']):
            axes[i,j].set_title(f'{split} / {row.track_id}\n{title}', fontsize=10)
            axes[i,j].axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'mask_verification.png', dpi=160)
    plt.close(fig)
    print(f'analysis={args.output} visible_mask_previews={len(metadata)}', flush=True)


if __name__ == '__main__':
    main()
