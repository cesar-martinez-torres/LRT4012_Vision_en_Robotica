"""ONNX segmentation and image-based pure pursuit; no simulator ground truth."""

import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


class LaneSegmenter:
    def __init__(self, model):
        model = Path(model)
        self.metadata = json.loads(model.with_suffix('.json').read_text(encoding='utf-8'))
        self.width, self.height = self.metadata['width'], self.metadata['height']
        self.mean = np.asarray(self.metadata['mean'], np.float32)[None, :, None, None]
        self.std = np.asarray(self.metadata['std'], np.float32)[None, :, None, None]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])

    def predict(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.width, self.height)), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / self.metadata['scale']
        tensor = np.ascontiguousarray((tensor - self.mean) / self.std)
        started = perf_counter()
        logits = self.session.run(['logits'], {'images': tensor})[0][0]
        elapsed_ms = (perf_counter() - started) * 1000
        mask = logits.argmax(axis=0).astype(np.uint8)
        # For two classes, sigmoid(lane - background) equals lane softmax.
        probability = 1 / (1 + np.exp(-np.clip(logits[1] - logits[0], -30, 30)))
        return mask, probability, elapsed_ms


def lane_target(mask, pitch, lookahead=0.65, fov=1.1, height=0.213, offset=0.03, width=0.58):
    """Intersect mask-centre camera rays with the flat floor in robot coordinates.

    Robot coordinates: x forward, y left, z up. Prefer two visible boundaries.
    If a curve clips one boundary, offset the visible boundary along its normal
    using the width estimated from earlier complete observations.
    """
    h, w = mask.shape
    focal, cx, cy = w / (2 * math.tan(fov / 2)), (w - 1) / 2, (h - 1) / 2
    cosine, sine = math.cos(pitch), math.sin(pitch)
    candidates, left_edge, right_edge = [], [], []
    for row in range(0, h - 1, 2):
        dz = -(row - cy) / focal
        ray_x, ray_z = cosine + sine * dz, -sine + cosine * dz
        if ray_z >= -0.01:
            continue
        scale = -height / ray_z
        x = offset + scale * ray_x
        if not 0.35 < x < 1.3:
            continue
        padded = np.pad(mask[row].astype(np.int8), (1, 1))
        edges = np.diff(padded)
        runs = [(left, right - 1) for left, right in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
                if right - left > 12]
        if not runs:
            continue
        left, right = min(runs, key=lambda run: abs((run[0] + run[1]) / 2 - cx))
        left_y, right_y = -(left - cx) / focal * scale, -(right - cx) / focal * scale
        if left > 2:
            left_edge.append((x, left_y))
        if right < w - 3:
            right_edge.append((x, right_y))
        width_m = left_y - right_y
        if left <= 2 or right >= w - 3 or not 0.32 < width_m < 0.95:
            continue
        column = (left + right) / 2
        y = -(column - cx) / focal * scale
        candidates.append((x, y, column, row, width_m))
    if len(candidates) >= 4:
        # Correct the transverse width for the local lane heading.
        values = np.asarray(candidates)
        slope = np.polyfit(values[:, 0], values[:, 1], 1)[0]
        measured_width = float(np.median(values[:, 4]) / math.sqrt(1 + slope * slope))
        width = 0.95 * width + 0.05 * float(np.clip(measured_width, 0.45, 0.75))
    if not candidates or min(abs(point[0] - lookahead) for point in candidates) > 0.18:
        side, boundary = max(((1, right_edge), (-1, left_edge)), key=lambda item: len(item[1]))
        if len(boundary) < 5:
            return None, width
        values = np.asarray(boundary)
        x = float(np.clip(lookahead, values[:, 0].min(), values[:, 0].max()))
        polynomial = np.polyfit(values[:, 0], values[:, 1], 2)
        y = float(np.polyval(polynomial, x))
        slope = float(np.polyval(np.polyder(polynomial), x))
        normal_length = math.sqrt(1 + slope * slope)
        x, y = x - side * width / 2 * slope / normal_length, y + side * width / 2 / normal_length
        # Project the inferred ground point back into the camera for display.
        dx, dz = x - offset, -height
        camera_x, camera_z = cosine * dx - sine * dz, sine * dx + cosine * dz
        column, row = cx - focal * y / camera_x, cy - focal * camera_z / camera_x
        return (x, y, column, row), width
    # Median over nearby scanlines reduces individual boundary-pixel noise.
    nearest = sorted(candidates, key=lambda point: abs(point[0] - lookahead))[:5]
    return tuple(float(value) for value in np.median(nearest, axis=0)[:4]), width
