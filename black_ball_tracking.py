"""Black-ball detection and image-based differential-drive control."""

from math import pi

import cv2
import numpy as np


THRESHOLD = 127
MIN_AREA = 80.0
MIN_CIRCULARITY = 0.20
MIN_SOLIDITY = 0.65
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

DESIRED_RADIUS_RATIO = 0.17
K_LINEAR = 6.0
K_ANGULAR = 3.2
MAX_FORWARD_SPEED = 4.5
MAX_REVERSE_SPEED = 2.0
MAX_TURN_SPEED = 3.5
MAX_WHEEL_SPEED = 6.67


def segment_black(frame_bgr):
    """Convert BGR to grayscale and segment dark pixels."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return gray, mask


def detect_ball(mask):
    """Return the contour, centroid, and radius of the best candidate."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_detection = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < MIN_AREA or perimeter == 0:
            continue

        circularity = 4.0 * pi * area / (perimeter * perimeter)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if circularity < MIN_CIRCULARITY or solidity < MIN_SOLIDITY:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        _, radius = cv2.minEnclosingCircle(contour)

        if area > best_area:
            best_area = area
            best_detection = (contour, center, radius)

    return best_detection


def compute_control(detection, width, height, lost_frames, search_direction):
    """Convert visual errors into differential wheel speeds."""
    if detection is None:
        turn = 0.65 * search_direction if lost_frames >= 8 else 0.0
        return turn, -turn, 0.0, 0.0

    _, center, radius = detection
    error_x = (center[0] - width / 2.0) / (width / 2.0)
    error_radius = DESIRED_RADIUS_RATIO - radius / height

    if abs(error_x) < 0.04:
        error_x = 0.0
    if abs(error_radius) < 0.008:
        error_radius = 0.0

    forward = K_LINEAR * error_radius / DESIRED_RADIUS_RATIO
    forward = float(np.clip(forward, -MAX_REVERSE_SPEED, MAX_FORWARD_SPEED))
    forward *= max(0.0, 1.0 - abs(error_x))
    turn = float(np.clip(K_ANGULAR * error_x, -MAX_TURN_SPEED, MAX_TURN_SPEED))

    left = float(np.clip(forward + turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    right = float(np.clip(forward - turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    return left, right, error_x, error_radius


def process_frame(frame_bgr, lost_frames=0, search_direction=1.0):
    """Apply segmentation, detection, and control to one frame."""
    gray, mask = segment_black(frame_bgr)
    detection = detect_ball(mask)
    height, width = mask.shape
    control = compute_control(
        detection, width, height, lost_frames, search_direction
    )
    return gray, mask, detection, control


def draw_result(frame_bgr, detection, control):
    """Draw the detected geometry and control variables."""
    result = frame_bgr.copy()
    left, right, error_x, error_radius = control

    if detection is not None:
        contour, center, radius = detection
        cv2.drawContours(result, [contour], -1, (0, 180, 0), 3)
        cv2.circle(result, center, 5, (0, 0, 255), -1)
        cv2.circle(result, center, int(radius), (255, 0, 0), 2)

    cv2.putText(
        result,
        f"e_x={error_x:+.3f}  e_r={error_radius:+.3f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        result,
        f"w_L={left:+.2f}  w_R={right:+.2f} rad/s",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    return result
