"""Create learning curves, pixel confusion and visual test error analysis."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segformer_utils import metrics_from_confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'segformer_b0')
    args = parser.parse_args()
    run = args.run.resolve()
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    history = pd.read_csv(run / 'history.csv')
    output = run / 'analysis'
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ['loss', 'mean_iou', 'lane_dice'], ['Pérdida', 'IoU media (dos clases)', 'Dice del carril']):
        for split in ['train', 'val']:
            ax.plot(history.epoch, history[f'{split}_{metric}'], label=split)
        ax.axvline(config['head_epochs'] + .5, ls='--', color='gray', lw=1, label='Inicio del ajuste fino')
        ax.axvline(metrics['best_epoch'], ls=':', color='black', lw=1, label='Mejor época')
        ax.set(title=title, xlabel='Época', ylabel=metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=170)
    plt.close(fig)
    if metrics['test'] is None:
        return
    predictions = pd.read_csv(run / 'test_predictions.csv')
    matrix = np.asarray(metrics['test']['confusion_matrix'])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(normalized, cmap='Greys', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i,j]:,}\n{normalized[i,j]:.2%}', ha='center', va='center',
                    color='white' if normalized[i,j] > .5 else 'black')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Fondo','Carril'], yticklabels=['Fondo','Carril'],
           xlabel='Predicción', ylabel='Etiqueta real', title='Matriz de confusión por píxel — prueba')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=170)
    plt.close(fig)
    # Per-track metrics are computed from summed pixel counts, not averages of ratios.
    per_track = []
    for track, group in predictions.groupby('track_id'):
        counts = group[['tn','fp','fn','tp']].sum().to_numpy().reshape(2,2)
        result = metrics_from_confusion(counts)
        per_track.append({'track_id': track, 'samples': len(group), **{k: result[k] for k in ['mean_iou','lane_iou','lane_dice','pixel_accuracy']}})
    pd.DataFrame(per_track).to_csv(output / 'per_track_metrics.csv', index=False)
    selected = predictions.sort_values('lane_iou').head(4)
    fig, axes = plt.subplots(len(selected), 4, figsize=(15, 2.8 * len(selected)), squeeze=False)
    for i, row in enumerate(selected.itertuples()):
        rgb = np.asarray(Image.open(Path(config['data']) / row.image).convert('RGB'))
        target = np.asarray(Image.open(Path(config['data']) / row.ground_truth))
        predicted = np.asarray(Image.open(run / row.prediction))
        error = np.zeros_like(rgb)
        error[(target == 0) & (predicted == 1)] = [235,80,30]
        error[(target == 1) & (predicted == 0)] = [40,140,255]
        for j, (image, title) in enumerate([(rgb, f'{row.track_id}: IoU={row.lane_iou:.3f}'),
                                           (target, 'Ground truth'), (predicted, 'SegFormer'),
                                           (error, 'Naranja: falso carril / Azul: carril omitido')]):
            if j in [1,2]:
                axes[i,j].imshow(image, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            else:
                axes[i,j].imshow(image)
            axes[i,j].set_title(title, fontsize=9)
            axes[i,j].axis('off')
    fig.tight_layout()
    fig.savefig(output / 'hardest_test_examples.png', dpi=170)
    plt.close(fig)
    print(json.dumps({'best_epoch': metrics['best_epoch'], 'test': metrics['test'],
                      'per_track': per_track, 'analysis': str(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
