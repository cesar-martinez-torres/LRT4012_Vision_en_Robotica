"""Build one reproducible circuit and evaluate the lap without steering the robot."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from controller import Supervisor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fixed_tracks import make_fixed_track
from lane_geometry import road_nodes, scenery_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    supervisor = Supervisor()
    step = int(supervisor.getBasicTimeStep())
    output = Path(os.environ.get('WEBOTS_LANE_RUN', ROOT / f'runs/fixed_track/latest_{args.track:02d}'))
    output.mkdir(parents=True, exist_ok=True)
    track = make_fixed_track(args.track)
    (output / 'track.json').write_text(json.dumps(track, indent=2), encoding='utf-8')
    children = supervisor.getFromDef('TRACK_ROOT').getField('children')
    for index in range(children.getCount() - 1, -1, -1):
        children.removeMF(index)
    nodes = list(road_nodes(track)) + list(scenery_nodes(track))
    children.importMFNodeFromString(-1, 'Group { children [ ' + '\n'.join(nodes) + ' ] }')
    supervisor.getFromDef('GROUND_APPEARANCE').getField('baseColor').setSFColor(track['ground_color'])
    sun = supervisor.getFromDef('SUN')
    sun.getField('direction').setSFVec3f(track['sun_direction'])
    sun.getField('intensity').setSFFloat(track['sun_intensity'])
    sun.getField('ambientIntensity').setSFFloat(track['ambient_intensity'])
    supervisor.getFromDef('LANE_CAMERA').getField('rotation').setSFRotation([0, 1, 0, track['camera_pitch_rad']])
    points = np.asarray(track['centerline'])
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    length = float(cumulative[-1])
    yaw = math.atan2(segments[0, 1], segments[0, 0])
    burger = supervisor.getFromDef('BURGER')
    burger.getField('translation').setSFVec3f([*points[0], 0])
    burger.getField('rotation').setSFRotation([0, 0, 1, yaw])
    burger.resetPhysics()
    emitter, receiver = supervisor.getDevice('lane emitter'), supervisor.getDevice('lane receiver')
    receiver.enable(step)
    started, status, reason = None, 'waiting', ''
    previous_arc, progress, outside_since = 0.0, 0.0, None
    deviations = []
    ticks = 0
    print(f'fixed_track={track["track_number"]} name={track["track_name"]} length_m={length:.3f} width_m={track["width_m"]:.3f}', flush=True)
    with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'x_m', 'y_m', 'lateral_error_m', 'progress_m'])
        while supervisor.step(step) != -1:
            ticks += 1
            while receiver.getQueueLength():
                message = json.loads(receiver.getString())
                receiver.nextPacket()
                if message['event'] == 'ready' and started is None:
                    started = supervisor.getTime()
                    emitter.send(json.dumps({'event': 'start', 'camera_pitch_rad': track['camera_pitch_rad']}).encode())
                    status = 'running'
                elif message['event'] == 'vision_lost':
                    status, reason = 'failed', 'lane_target_not_visible'
            if started is None:
                if supervisor.getTime() > 120:
                    status, reason = 'failed', 'controller_not_ready'
                else:
                    continue
            else:
                position = np.asarray(burger.getPosition()[:2])
                fraction = np.clip(((position - points) * segments).sum(axis=1) / lengths ** 2, 0, 1)
                projections = points + fraction[:, None] * segments
                distances = np.linalg.norm(position - projections, axis=1)
                index = int(distances.argmin())
                error = float(distances[index])
                arc = float(cumulative[index] + fraction[index] * lengths[index])
                progress += (arc - previous_arc + length / 2) % length - length / 2
                previous_arc = arc
                deviations.append(error)
                elapsed = supervisor.getTime() - started
                if ticks % 3 == 0:
                    writer.writerow([elapsed, *position, error, progress])
                if error > track['width_m'] / 2 - 0.08:
                    outside_since = elapsed if outside_since is None else outside_since
                    if elapsed - outside_since > 0.6:
                        status, reason = 'failed', 'left_lane'
                else:
                    outside_since = None
                if progress >= length - 0.06:
                    status, reason = 'completed', 'one_lap'
                if elapsed > 500:
                    status, reason = 'failed', 'time_limit'
                if ticks % 1000 == 0:
                    print(f'lap_progress={100 * progress / length:.1f}% lateral_error_m={error:.4f}', flush=True)
            if ticks == 120:
                supervisor.exportImage(str(output / 'world_overview.png'), 90)
            if status in ('completed', 'failed'):
                emitter.send(json.dumps({'event': 'finish' if status == 'completed' else 'abort'}).encode())
                handle.flush()
                report = {'status': status, 'reason': reason, 'seed': track['seed'],
                          'track_number': track['track_number'], 'track_name': track['track_name'],
                          'track_length_m': length, 'track_width_m': track['width_m'],
                          'progress_m': progress, 'elapsed_simulation_s': supervisor.getTime() - (started or 0),
                          'mean_lateral_error_m': float(np.mean(deviations)) if deviations else None,
                          'max_lateral_error_m': float(max(deviations)) if deviations else None,
                          'control': 'RGB -> ONNX predicted mask -> local pure pursuit; no route or pose feedback'}
                (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                supervisor.exportImage(str(output / 'world_final.png'), 90)
                print(json.dumps(report), flush=True)
                for _ in range(5):
                    supervisor.step(step)
                if os.environ.get('WEBOTS_LANE_BATCH') == '1':
                    supervisor.simulationQuit(0 if status == 'completed' else 1)
                    return
                supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
                return


if __name__ == '__main__':
    main()
