"""Control a TurtleBot3 Burger from MobileNetV3 visual-command predictions."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from controller import Robot
from motion_control import MotionControl, WHEEL_RADIUS, AXLE_LENGTH, wrap_angle


CLASSES = ("forward", "left", "right", "stop")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "mobilenet_v3_small" / "best_model.onnx"
IMAGE_SIZE = 224
NORMALIZATION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZATION_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_controller_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stop-steps", type=int, default=24)
    parser.add_argument("--forward-distance", type=float, default=0.42)
    parser.add_argument("--turn-angle", type=float, default=90.0)
    arguments, _ = parser.parse_known_args()
    return arguments


def load_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {path}. Run export_onnx.py after training."
        )
    return cv2.dnn.readNetFromONNX(str(path))


def camera_image(camera) -> np.ndarray:
    raw = camera.getImage()
    if raw is None:
        raise RuntimeError("The Webots camera has no image available.")
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
        camera.getHeight(), camera.getWidth(), 4
    )
    return bgra[:, :, :3][:, :, ::-1].copy()


def preprocess(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - NORMALIZATION_MEAN) / NORMALIZATION_STD
    return np.transpose(normalized, (2, 0, 1))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def predict(model, image, classes):
    batch = np.stack([preprocess(image), preprocess(np.fliplr(image))])
    model.setInput(batch)
    start = time.perf_counter()
    logits = model.forward()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    probabilities = softmax(logits)
    mirrored = probabilities[1].copy()
    left_index = classes.index("left")
    right_index = classes.index("right")
    mirrored[left_index] = probabilities[1, right_index]
    mirrored[right_index] = probabilities[1, left_index]
    original = probabilities[0]
    result = original if original.max() >= mirrored.max() else mirrored
    index = int(np.argmax(result))
    return classes[index], float(result[index]), elapsed_ms


def main() -> None:
    arguments = parse_controller_arguments()
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    camera = robot.getDevice("camera")
    camera.enable(time_step)
    emitter = robot.getDevice("status emitter")

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    left_sensor = robot.getDevice("left wheel sensor")
    right_sensor = robot.getDevice("right wheel sensor")
    left_sensor.enable(time_step)
    right_sensor.enable(time_step)
    devices = {robot.getDeviceByIndex(i).getName() for i in range(robot.getNumberOfDevices())}
    imu = robot.getDevice("navigation imu") if "navigation imu" in devices else None
    if imu:
        imu.enable(time_step)
    receiver = robot.getDevice("command receiver") if "command receiver" in devices else None
    if receiver:
        receiver.enable(time_step)
    motion = MotionControl(
        float(os.environ.get("WEBOTS_FORWARD_DISTANCE", arguments.forward_distance)),
        float(os.environ.get("WEBOTS_TURN_ANGLE", arguments.turn_angle)),
    )
    previous_encoders = None
    yaw = 0.0
    settled = 0
    action_started = 0.0
    read_started = robot.getTime()

    threshold = float(os.environ.get("WEBOTS_CONFIDENCE_THRESHOLD", "0.80"))
    model_path = Path(os.environ.get("WEBOTS_MODEL_PATH", DEFAULT_MODEL)).resolve()
    model = load_model(model_path)
    classes = CLASSES

    required_votes = int(os.environ.get("WEBOTS_STABLE_PREDICTIONS", "3"))
    inference_period = int(os.environ.get("WEBOTS_INFERENCE_PERIOD_STEPS", "5"))
    cooldown_steps = int(os.environ.get("WEBOTS_COMMAND_COOLDOWN_STEPS", "18"))
    stop_steps = int(os.environ.get("WEBOTS_STOP_STEPS", arguments.stop_steps))
    save_views = os.environ.get("WEBOTS_TURTLEBOT_SAVE", "0") == "1"
    view_dir = PROJECT_ROOT / "data" / "turtlebot_route"
    if save_views:
        view_dir.mkdir(parents=True, exist_ok=True)
    recent = deque(maxlen=required_votes)
    state = "READ"
    active_command = "stop"
    remaining = 0
    step_index = 0
    rejected_count = 0
    action_index = 0

    print(
        f"model={model_path} backend=opencv-dnn-cpu threshold={threshold:.2f} "
        f"stable_predictions={required_votes} forward_distance={motion.distance:.3f} "
        f"turn_angle_deg={math.degrees(motion.turn):.1f}",
        flush=True,
    )

    while robot.step(time_step) != -1:
        step_index += 1
        left, right = left_sensor.getValue(), right_sensor.getValue()
        if previous_encoders is not None:
            dl, dr = left - previous_encoders[0], right - previous_encoders[1]
            yaw = wrap_angle(yaw + WHEEL_RADIUS * (dr - dl) / AXLE_LENGTH)
        previous_encoders = (left, right)
        if imu:
            yaw = imu.getRollPitchYaw()[2]
        if step_index == 1:
            motion.heading = yaw

        if receiver:
            while receiver.getQueueLength() > 0:
                message = receiver.getString()
                receiver.nextPacket()
                if message in ("abort", "finish"):
                    state = "HALT"
                elif message == "next" and state == "WAIT_NEXT":
                    state = "COOLDOWN"
                    remaining = cooldown_steps
        if state in ("HALT", "WAIT_NEXT"):
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            continue
        if state == "EXECUTE":
            vl, vr, done, distance_error, heading_error = motion.update(left, right, yaw)
            left_motor.setVelocity(vl)
            right_motor.setVelocity(vr)
            settled = settled + 1 if done else 0
            required_settle = stop_steps if active_command == "stop" else 6
            if robot.getTime() - action_started > 15.0:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                emitter.send(f"failed:{active_command}:motion_timeout".encode("utf-8"))
                print("motion_status=failed reason=motion_timeout", flush=True)
                state = "HALT"
            elif settled >= required_settle:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                emitter.send(f"done:{active_command}".encode("utf-8"))
                print(f"completed={active_command} distance_error={distance_error:.4f} "
                      f"heading_error_deg={math.degrees(heading_error):.3f}", flush=True)
                state = "WAIT_NEXT" if receiver else "COOLDOWN"
                remaining = cooldown_steps
            continue

        if state == "COOLDOWN":
            remaining -= 1
            if remaining <= 0:
                recent.clear()
                state = "READ"
                read_started = robot.getTime()
            continue

        if robot.getTime() - read_started > 30.0:
            emitter.send(b"failed:read:recognition_timeout")
            state = "HALT"
            continue

        if step_index % inference_period:
            continue

        image = camera_image(camera)
        prediction, confidence, inference_ms = predict(model, image, classes)
        accepted = confidence >= threshold
        recent.append(prediction if accepted else None)
        rejected_count = 0 if accepted else rejected_count + 1
        if accepted or rejected_count % 20 == 0:
            print(
                f"prediction={prediction} confidence={confidence:.4f} "
                f"accepted={accepted} inference_ms={inference_ms:.2f}",
                flush=True,
            )
        if len(recent) < required_votes or len(set(recent)) != 1 or recent[0] is None:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            continue

        active_command = recent[0]
        if save_views:
            output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(view_dir / f"action_{action_index:02d}_{active_command}.png"),
                output,
            )
        action_index += 1
        motion.start(active_command, left, right)
        action_started = robot.getTime()
        settled = 0
        state = "EXECUTE"
        print(
            f"execute={active_command} distance={motion.target_distance:.3f} "
            f"target_heading_deg={math.degrees(motion.heading):.2f}",
            flush=True,
        )

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)


if __name__ == "__main__":
    main()
