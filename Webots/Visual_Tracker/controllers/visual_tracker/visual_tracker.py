"""Seguimiento visual de una pelota roja con un TurtleBot3 Burger."""

import os
from math import pi

import cv2
import numpy as np
from controller import Robot


# El tono rojo ocupa los dos extremos del intervalo H de OpenCV [0, 179].
RED_LOW_1 = np.array([0, 120, 70], dtype=np.uint8)
RED_HIGH_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_LOW_2 = np.array([170, 120, 70], dtype=np.uint8)
RED_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)

MIN_AREA = 80.0
MIN_CIRCULARITY = 0.20
MIN_SOLIDITY = 0.65

DESIRED_RADIUS_RATIO = 0.17
K_LINEAR = 6.0
K_ANGULAR = 3.2
MAX_FORWARD_SPEED = 4.5
MAX_REVERSE_SPEED = 2.0
MAX_TURN_SPEED = 3.5
MAX_WHEEL_SPEED = 6.67

SHOW_WINDOW = os.environ.get("VISUAL_TRACKING_GUI", "1") != "0"
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def segment_red(frame_bgra):
    """Convierte la imagen de Webots a HSV y segmenta el color rojo."""
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
    mask_2 = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return frame_bgr, mask


def detect_ball(mask):
    """Devuelve el contorno, centroide y radio del mejor candidato."""
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
    """Convierte errores visuales en velocidades de las ruedas."""
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


def draw_result(frame, mask, detection, error_x, error_radius, left, right):
    """Superpone las mediciones principales sobre la imagen."""
    height, width = frame.shape[:2]
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 1)

    if detection is None:
        label, color = "BALL LOST", (0, 0, 255)
    else:
        contour, center, radius = detection
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, center, 4, (255, 255, 255), -1)
        cv2.circle(frame, center, int(radius), (0, 255, 255), 1)
        label, color = "BALL DETECTED", (0, 220, 0)

    cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(
        frame,
        f"e_x={error_x:+.3f}  e_r={error_radius:+.3f}",
        (10, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"w_L={left:+.2f}  w_R={right:+.2f} rad/s",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )

    mask_view = cv2.resize(mask, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)
    frame[: height // 4, 3 * width // 4 :] = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)
    return frame


robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)
width, height = camera.getWidth(), camera.getHeight()

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

lost_frames = 0
search_direction = 1.0

while robot.step(timestep) != -1:
    frame_bgra = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(height, width, 4)
    frame, mask = segment_red(frame_bgra)
    detection = detect_ball(mask)

    if detection is None:
        lost_frames += 1
    else:
        lost_frames = 0
        center_x = detection[1][0]
        if abs(center_x - width / 2.0) > 0.04 * width / 2.0:
            search_direction = 1.0 if center_x > width / 2.0 else -1.0

    left, right, error_x, error_radius = compute_control(
        detection, width, height, lost_frames, search_direction
    )
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)

    if SHOW_WINDOW:
        result = draw_result(frame, mask, detection, error_x, error_radius, left, right)
        cv2.imshow("Visual tracking: perception and control", result)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
cv2.destroyAllWindows()
