"""Drive from RGB -> ONNX mask -> local lane centre -> wheel velocities."""

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from controller import Robot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lane_inference import LaneSegmenter, lane_target
from vision_dashboard import VisionDashboard


def main():
    robot = Robot()
    step = int(robot.getBasicTimeStep())
    camera = robot.getDevice('camera')
    camera.enable(step * 3)
    dashboard = VisionDashboard(robot.getDevice('vision dashboard'))
    receiver, emitter = robot.getDevice('lane receiver'), robot.getDevice('lane emitter')
    receiver.enable(step)
    motors = [robot.getDevice(name + ' wheel motor') for name in ('left', 'right')]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0)
    model = Path(os.environ.get('WEBOTS_LANE_MODEL', ROOT / 'runs/segformer_b0/best_model.onnx'))
    # Each world's name selects its default log folder, without map information.
    track_number = int(robot.getCustomData() or '1')
    output = Path(os.environ.get('WEBOTS_LANE_RUN', ROOT / f'runs/fixed_track/latest_{track_number:02d}'))
    output.mkdir(parents=True, exist_ok=True)
    segmenter = LaneSegmenter(model)
    emitter.send(json.dumps({'event': 'ready'}).encode())
    active, pitch, angular, frame, lost_since = False, 0.32, 0.0, 0, None
    control_step = 0
    estimated_width = 0.58
    with (output / 'vision_control.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['time_s', 'target_x_m', 'target_y_m', 'linear_m_s', 'angular_rad_s',
                  'lane_fraction', 'lane_confidence', 'inference_ms', 'target_visible']
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        while robot.step(step) != -1:
            while receiver.getQueueLength():
                message = json.loads(receiver.getString())
                receiver.nextPacket()
                if message['event'] == 'start':
                    active, pitch = True, message['camera_pitch_rad']
                    print('lane follower started: ONNX Runtime CPU; image feedback only', flush=True)
                elif message['event'] in ('finish', 'abort'):
                    active = False
                    for motor in motors:
                        motor.setVelocity(0)
                    dashboard.stop('VUELTA COMPLETADA' if message['event'] == 'finish' else 'DETENIDO')
                    handle.flush()
            if not active:
                continue
            control_step += 1
            if control_step % 3:
                continue
            bgr = np.frombuffer(camera.getImage(), np.uint8).reshape(camera.getHeight(), camera.getWidth(), 4)[:, :, :3].copy()
            mask, probability, elapsed = segmenter.predict(bgr)
            target, estimated_width = lane_target(mask, pitch, width=estimated_width)
            now = robot.getTime()
            if target is None:
                speed, angular = 0.0, 0.0
                lost_since = now if lost_since is None else lost_since
                if now - lost_since > 2:
                    emitter.send(json.dumps({'event': 'vision_lost'}).encode())
                    active = False
            else:
                lost_since = None
                x, y, column, row = target
                curvature = 2 * y / (x * x + y * y)
                speed = 0.12 / (1 + 0.35 * abs(curvature))
                desired = float(np.clip(speed * curvature, -0.6, 0.6))
                angular = 0.6 * angular + 0.4 * desired
            wheel_speeds = np.array([speed - 0.08 * angular, speed + 0.08 * angular]) / 0.033
            wheel_speeds *= min(1, 6.67 / max(np.abs(wheel_speeds).max(), 1e-9))
            for motor, velocity in zip(motors, wheel_speeds):
                motor.setVelocity(float(velocity))
            row_data = {'time_s': now, 'target_x_m': target[0] if target else '',
                        'target_y_m': target[1] if target else '', 'linear_m_s': speed,
                        'angular_rad_s': angular, 'lane_fraction': float(mask.mean()),
                        'lane_confidence': float(probability[mask == 1].mean()) if mask.any() else 0,
                        'inference_ms': elapsed, 'target_visible': int(target is not None)}
            writer.writerow(row_data)
            canvas, overlay = dashboard.update(bgr, mask, target, speed, angular, elapsed,
                                               'SEGUIMIENTO' if target else 'SIN REFERENCIA VISUAL')
            frame += 1
            if frame % 30 == 1:
                cv2.imwrite(str(output / 'camera_overlay.png'), overlay)
                cv2.imwrite(str(output / 'predicted_mask.png'), mask * 255)
                cv2.imwrite(str(output / 'vision_dashboard.png'), canvas)
                handle.flush()
            if frame % 100 == 0:
                print(f'vision_frame={frame} target={target[:2] if target else None} inference_ms={elapsed:.1f}', flush=True)


if __name__ == '__main__':
    main()
