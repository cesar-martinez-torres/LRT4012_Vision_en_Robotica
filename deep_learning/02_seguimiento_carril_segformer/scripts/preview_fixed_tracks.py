"""Draw the five fixed circuits and report their geometry."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixed_tracks import make_fixed_track
from lane_geometry import validate_track


def preview(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    report = []
    for number, ax in enumerate(axes, start=1):
        track = make_fixed_track(number)
        points = np.asarray(track['centerline'])
        _, normals = validate_track(points, track['width_m'])
        left, right = points + track['width_m'] / 2 * normals, points - track['width_m'] / 2 * normals
        road = np.vstack([left, left[0], right[0], right[::-1], left[0]])
        ax.fill(*road.T, color='0.75')
        for border in (left, right):
            ax.plot(*np.vstack([border, border[0]]).T, color='0.25', linewidth=0.6)
        ax.plot(*np.vstack([points, points[0]]).T, '--', color='0.5', linewidth=0.6)
        ax.scatter(*points[0], color='black', s=15)
        tangent = points[1] - points[0]
        tangent /= np.linalg.norm(tangent)
        ax.annotate('', xy=points[0] + tangent * 0.7, xytext=points[0],
                    arrowprops={'arrowstyle': '->', 'color': 'black'})
        ax.set(aspect='equal', xlim=(-4.4, 4.4), ylim=(-3.4, 3.4), xlabel='x (m)',
               title=f'{number}. {track["track_name"]}\n{track["length_m"]:.2f} m')
        ax.grid(alpha=0.15)
        report.append({key: track[key] for key in ('track_number', 'track_name', 'length_m', 'min_curve_radius_m', 'width_m')})
    axes[0].set_ylabel('y (m)')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/fixed_tracks_catalog.png')
    args = parser.parse_args()
    preview(args.output)
