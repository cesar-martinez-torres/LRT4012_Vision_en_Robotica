"""Plot a completed Webots lap and summarize vision-control measurements."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lane_geometry import validate_track


def load_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def analyze(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    track = json.loads((directory / 'track.json').read_text(encoding='utf-8'))
    trajectory = load_rows(directory / 'trajectory.csv')
    vision = load_rows(directory / 'vision_control.csv')
    points = np.asarray(track['centerline'])
    _, normals = validate_track(points, track['width_m'])
    path = np.array([[float(row['x_m']), float(row['y_m'])] for row in trajectory])
    times = np.array([float(row['time_s']) for row in trajectory])
    errors = np.array([float(row['lateral_error_m']) for row in trajectory])
    latency = np.array([float(row['inference_ms']) for row in vision])
    summary = {**result, 'vision_frames': len(vision),
               'target_visible_fraction': float(np.mean([int(row['target_visible']) for row in vision])),
               'median_inference_ms': float(np.median(latency)),
               'p95_inference_ms': float(np.percentile(latency, 95)),
               'note': 'Inference latency excludes image acquisition, preprocessing and motor actuation.'}
    (directory / 'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for sign in (-1, 1):
        border = points + sign * track['width_m'] / 2 * normals
        axes[0].plot(*np.vstack([border, border[0]]).T, color='0.6', linewidth=1)
    axes[0].plot(*np.vstack([points, points[0]]).T, '--', color='0.7', label='Lane centre')
    axes[0].plot(*path.T, color='tab:blue', label='Robot trajectory')
    axes[0].scatter(*path[0], color='black', s=25, label='Start')
    axes[0].set(aspect='equal', xlabel='x (m)', ylabel='y (m)', title=f'Fixed circuit: {result["status"]}')
    axes[0].legend(fontsize=8)
    axes[1].plot(times, errors * 100)
    axes[1].set(xlabel='Simulation time (s)', ylabel='Distance to lane centre (cm)', title='Lateral deviation')
    axes[1].grid(alpha=0.25)
    axes[2].hist(latency, bins=30, color='0.45')
    axes[2].set(xlabel='ONNX inference (ms)', ylabel='Frames', title='CPU inference latency')
    fig.savefig(directory / 'following_analysis.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path)
    args = parser.parse_args()
    directories = sorted((ROOT / 'runs/fixed_track').glob('*/result.json'), key=lambda path: path.stat().st_mtime)
    if args.run is None and not directories:
        parser.error('No completed run; execute scripts/run_fixed_track.ps1 first.')
    analyze(args.run or directories[-1].parent)
