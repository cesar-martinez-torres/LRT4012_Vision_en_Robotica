"""Five prescribed circuits; their geometry does not use a random generator."""

import json
from pathlib import Path

import numpy as np

from lane_geometry import validate_track

ROOT = Path(__file__).resolve().parent
TRACK_NAMES = {1: 'Original', 2: 'Ovalo', 3: 'Estadio',
               4: 'Rectangulo redondeado', 5: 'Curvas en S'}


def rounded_rectangle(width, height, radius, vertices=160):
    """Counterclockwise straight segments joined by circular corner fillets."""
    corners = [(width / 2 - radius, height / 2 - radius),
               (-width / 2 + radius, height / 2 - radius),
               (-width / 2 + radius, -height / 2 + radius),
               (width / 2 - radius, -height / 2 + radius)]
    pieces = []
    for index, (x, y) in enumerate(corners):
        angles = np.linspace(index * np.pi / 2, (index + 1) * np.pi / 2, 129)
        arc = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles)])
        pieces.extend(arc)
        next_x, next_y = corners[(index + 1) % 4]
        endpoint = np.array([next_x + radius * np.cos(angles[-1]),
                             next_y + radius * np.sin(angles[-1])])
        if np.linalg.norm(endpoint - arc[-1]) > 1e-8:
            pieces.extend(np.linspace(arc[-1], endpoint, 129)[1:])
    points = np.asarray(pieces)
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if np.linalg.norm(points[-1] - points[0]) > 1e-8:
        points = np.vstack([points, points[0]])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0, distance[-1], vertices, endpoint=False)
    return np.column_stack([np.interp(samples, distance, points[:, axis]) for axis in (0, 1)])


def make_fixed_track(number=1):
    if number not in TRACK_NAMES:
        raise ValueError('Track number must be 1, 2, 3, 4 or 5')
    # Frozen coordinates preserve the first circuit exactly, without regenerating it.
    track = json.loads((ROOT / 'fixed_tracks/original.json').read_text(encoding='utf-8'))
    theta = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    if number == 2:
        points = np.column_stack([3.2 * np.cos(theta), 2.1 * np.sin(theta)])
    elif number == 3:
        points = rounded_rectangle(7.0, 2.8, 1.4)
    elif number == 4:
        points = rounded_rectangle(6.6, 4.8, 1.1)
    elif number == 5:
        points = np.column_stack([3.8 * np.cos(theta), 2.2 * np.sin(theta) + 0.5 * np.sin(3 * theta)])
    else:
        points = np.asarray(track['centerline'])
    metrics, _ = validate_track(points, track['width_m'])
    track.update(metrics)
    track.update(track_id=f'fixed_{number:02d}', track_number=number,
                 track_name=TRACK_NAMES[number], split='deployment',
                 centerline=points.tolist(), geometry_source='prescribed fixed circuit')
    # Keep the same camera, width, colours and illumination across the five tests.
    # The retained seed only places reproducible decorative objects, off the road.
    return track
