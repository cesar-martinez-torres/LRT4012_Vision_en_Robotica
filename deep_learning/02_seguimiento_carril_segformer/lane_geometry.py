"""Seeded, smooth closed tracks and renderable road geometry (metres)."""

import math
import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def validate_track(points, width):
    points = np.asarray(points, dtype=float)
    previous, following = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    a, b = points - previous, following - points
    chord = following - previous
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    curvature = 2 * np.abs(cross) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(chord, axis=1))
    min_radius = float(1 / max(curvature.max(), 1e-9))
    if min_radius < max(0.8, width * 1.5):
        raise ValueError('Curve radius too small')
    tangent = following - previous
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    left, right = points + width / 2 * normals, points - width / 2 * normals
    # Both boundaries form one simple ribbon; reject boundary self-intersections.
    def crossings(first, second):
        p, q = first[:, None, :], np.roll(first, -1, axis=0)[:, None, :]
        r, s = second[None, :, :], np.roll(second, -1, axis=0)[None, :, :]
        def cross(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return (cross(q-p, r-p) * cross(q-p, s-p) < -1e-12) & (cross(s-r, p-r) * cross(s-r, q-r) < -1e-12)
    if any(crossings(a, b).any() for a, b in ((left, left), (right, right), (left, right))):
        raise ValueError('Crossing road boundaries')
    return {'length_m': float(np.linalg.norm(b, axis=1).sum()),
            'min_curve_radius_m': min_radius, 'width_m': width,
            'geometry_valid': True}, normals


def make_track(seed, track_id, split, vertices=160):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    width = float(rng.uniform(0.48, 0.68))
    for _ in range(100):
        radius = np.full(vertices, rng.uniform(2.5, 3.2))
        for frequency in (2, 3, 4):
            radius += rng.uniform(0.06, 0.20) * np.sin(frequency * theta + rng.uniform(0, 2 * np.pi))
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        points *= rng.uniform(0.85, 1.15, size=2)
        try:
            metrics, _ = validate_track(points, width)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError(f'Cannot construct valid track for {seed}')
    palettes = [([0.17, 0.19, 0.21], [0.28, 0.39, 0.25]),
                ([0.30, 0.29, 0.26], [0.58, 0.57, 0.53]),
                ([0.43, 0.44, 0.46], [0.20, 0.28, 0.23]),
                ([0.21, 0.23, 0.24], [0.55, 0.46, 0.34])]
    road, ground = palettes[int(rng.integers(len(palettes)))]
    return {'track_id': track_id, 'split': split, 'seed': int(seed),
            'centerline': points.tolist(), **metrics,
            'road_color': np.clip(np.array(road) + rng.uniform(-0.035, 0.035, 3), 0.06, 0.85).tolist(),
            'ground_color': np.clip(np.array(ground) + rng.uniform(-0.04, 0.04, 3), 0.06, 0.85).tolist(),
            'border_color': ([0.88, 0.85, 0.67] if rng.random() < 0.5 else [0.88, 0.90, 0.91]),
            'sun_direction': [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), -1.0],
            'sun_intensity': float(rng.uniform(0.65, 1.1)),
            'ambient_intensity': float(rng.uniform(0.3, 0.65)),
            'camera_pitch_rad': float(rng.uniform(0.28, 0.38))}


def sample_pose(track, sample_index, count):
    points = np.asarray(track['centerline'])
    rng = np.random.default_rng(track['seed'] + 1000003 + sample_index)
    # Equal numbers in each direction; positions cover the entire loop.
    direction = 1 if sample_index % 2 == 0 else -1
    t = ((sample_index // 2 + rng.uniform(0, 0.8)) / math.ceil(count / 2)) * len(points)
    i, fraction = int(t) % len(points), t % 1
    point = (1 - fraction) * points[i] + fraction * points[(i + 1) % len(points)]
    tangent = points[(i + 1) % len(points)] - points[i]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    offset = float(rng.uniform(-0.32, 0.32) * track['width_m'])
    heading_error = float(rng.uniform(-0.22, 0.22))
    point += offset * normal
    yaw = wrap(math.atan2(tangent[1], tangent[0]) + (math.pi if direction < 0 else 0) + heading_error)
    return {'x': float(point[0]), 'y': float(point[1]), 'yaw': yaw,
            'lateral_offset_m': offset, 'heading_offset_rad': heading_error,
            'direction': direction, 'centerline_segment': i}


def _quad(vertices, color, height, origin):
    coords = ', '.join(f'{p[0]-origin[0]:.6f} {p[1]-origin[1]:.6f} {height:.6f}' for p in vertices)
    rgb = ' '.join(f'{c:.4f}' for c in color)
    return f'Shape {{ appearance PBRAppearance {{ baseColor {rgb} roughness 1 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {coords} ] }} coordIndex [ 0 1 2 3 -1 ] }} }}'


def road_nodes(track):
    points = np.asarray(track['centerline'])
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    half = track['width_m'] / 2
    left, right = points + half * normals, points - half * normals
    inner_left = points + (half - 0.025) * normals
    inner_right = points - (half - 0.025) * normals
    for i in range(len(points)):
        j = (i + 1) % len(points)
        origin = (points[i] + points[j]) / 2
        shapes = [ _quad([left[i], right[i], right[j], left[j]], track['road_color'], 0.003, origin),
                   _quad([left[i], inner_left[i], inner_left[j], left[j]], track['border_color'], 0.004, origin),
                   _quad([inner_right[i], right[i], right[j], inner_right[j]], track['border_color'], 0.004, origin)]
        # Small recognized tiles avoid the large-floor limitation of Recognition.
        yield f'Solid {{ translation {origin[0]:.6f} {origin[1]:.6f} 0 name "lane tile {i}" recognitionColors [ 1 0 0 ] children [ {" ".join(shapes)} ] }}'


def scenery_nodes(track):
    rng = np.random.default_rng(track['seed'] + 77)
    points = np.asarray(track['centerline'])
    for i in range(16):
        for _ in range(100):
            x, y = rng.uniform(-5.5, 5.5, 2)
            if np.linalg.norm(points - [x, y], axis=1).min() > track['width_m'] / 2 + 0.5:
                break
        height = rng.uniform(0.15, 0.65)
        color = ' '.join(f'{v:.3f}' for v in rng.uniform(0.15, 0.65, 3))
        yield f'Solid {{ translation {x:.4f} {y:.4f} {height/2:.4f} name "scenery {i}" children [ Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }} geometry Box {{ size 0.25 0.25 {height:.4f} }} }} ] }}'
