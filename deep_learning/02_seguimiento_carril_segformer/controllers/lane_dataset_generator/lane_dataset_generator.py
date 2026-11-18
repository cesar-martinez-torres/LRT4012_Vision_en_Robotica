"""Render RGB/mask pairs from supervised Burger camera poses in Webots."""

import csv
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

import cv2
import numpy as np
from controller import Supervisor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lane_geometry import make_track, road_nodes, scenery_nodes, sample_pose

ACTIVE_SUPERVISOR = None


def rebuild(supervisor, track):
    print(f'building={track["track_id"]}', flush=True)
    children = supervisor.getFromDef('TRACK_ROOT').getField('children')
    for index in range(children.getCount() - 1, -1, -1):
        children.removeMF(index)
    nodes = list(road_nodes(track)) + list(scenery_nodes(track))
    children.importMFNodeFromString(-1, 'Group { children [ ' + '\n'.join(nodes) + ' ] }')
    supervisor.getFromDef('GROUND_APPEARANCE').getField('baseColor').setSFColor(track['ground_color'])
    light = supervisor.getFromDef('SUN')
    light.getField('direction').setSFVec3f(track['sun_direction'])
    light.getField('intensity').setSFFloat(track['sun_intensity'])
    light.getField('ambientIntensity').setSFFloat(track['ambient_intensity'])
    supervisor.getFromDef('LANE_CAMERA').getField('rotation').setSFRotation([0, 1, 0, track['camera_pitch_rad']])
    print(f'built={track["track_id"]}', flush=True)


def main():
    global ACTIVE_SUPERVISOR
    supervisor = Supervisor()
    ACTIVE_SUPERVISOR = supervisor
    step = int(supervisor.getBasicTimeStep())
    output = Path(os.environ.get('WEBOTS_LANE_DATASET', ROOT / 'data' / 'lane_segmentation_v1')).resolve()
    capture = os.environ.get('WEBOTS_LANE_CAPTURE', '0') == '1'
    if not capture:
        if (output / 'manifest.json').exists() and os.environ.get('WEBOTS_LANE_TRACK'):
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            track_id = os.environ['WEBOTS_LANE_TRACK']
            entry = next(e for e in manifest['tracks'] if e['track_id'] == track_id)
            track = json.loads((output / entry['file']).read_text(encoding='utf-8'))
        else:
            seed = int(os.environ.get('WEBOTS_LANE_SEED', secrets.randbelow(2000000000)))
            track = make_track(seed, 'preview', 'preview')
        rebuild(supervisor, track)
        pose = sample_pose(track, 0, 96)
        robot = supervisor.getSelf()
        robot.getField('translation').setSFVec3f([pose['x'], pose['y'], 0])
        robot.getField('rotation').setSFRotation([0, 0, 1, pose['yaw']])
        robot.resetPhysics()
        burger = supervisor.getFromDef('BURGER')
        burger.getField('translation').setSFVec3f([pose['x'], pose['y'], 0])
        burger.getField('rotation').setSFRotation([0, 0, 1, pose['yaw']])
        burger.resetPhysics()
        camera = supervisor.getDevice('camera')
        camera.enable(step)
        print(f'preview_track={track["track_id"]} seed={track["seed"]}; stationary robot, no trained lane controller yet', flush=True)
        snapshot = os.environ.get('WEBOTS_LANE_SNAPSHOT', '')
        if snapshot:
            for _ in range(12):
                supervisor.step(step)
            supervisor.exportImage(snapshot, 90)
            for _ in range(3):
                supervisor.step(step)
            supervisor.simulationQuit(0)
            return
        while supervisor.step(step) != -1:
            pass
        return
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    metadata_path = output / 'metadata.csv'
    if metadata_path.exists():
        raise FileExistsError(f'Capture exists: {metadata_path}; choose a new dataset directory.')
    track_filter = os.environ.get('WEBOTS_LANE_TRACK', '')
    camera = supervisor.getDevice('camera')
    camera.enable(step)
    camera.recognitionEnable(step)
    camera.enableRecognitionSegmentation()
    # RGB and labels share one kinematic camera; no unstable dynamic teleporting.
    supervisor.getFromDef('BURGER').getField('translation').setSFVec3f([0, 0, -5])
    robot = supervisor.getSelf()
    count = manifest['samples_per_track']
    fields = ['track_id', 'track_seed', 'split', 'sample_index', 'image', 'mask',
              'image_sha256', 'mask_sha256', 'lane_fraction', 'x', 'y', 'yaw',
              'lateral_offset_m', 'heading_offset_rad', 'direction', 'centerline_segment']
    total = 0
    with metadata_path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in manifest['tracks']:
            if track_filter and entry['track_id'] != track_filter:
                continue
            track = json.loads((output / entry['file']).read_text(encoding='utf-8'))
            rebuild(supervisor, track)
            for i in range(count):
                pose = sample_pose(track, i, count)
                robot.getField('translation').setSFVec3f([pose['x'], pose['y'], 0.0])
                robot.getField('rotation').setSFRotation([0, 0, 1, pose['yaw']])
                robot.resetPhysics()
                # Two render steps synchronize the RGB and segmentation buffers.
                for _ in range(2):
                    if supervisor.step(step) == -1:
                        return
                shape = (camera.getHeight(), camera.getWidth(), 4)
                image = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(shape)[:, :, :3].copy()
                segmentation = np.frombuffer(camera.getRecognitionSegmentationImage(), dtype=np.uint8).reshape(shape)
                mask = ((segmentation[:, :, 2] > 200) & (segmentation[:, :, 1] < 30) &
                        (segmentation[:, :, 0] < 30)).astype(np.uint8)
                fraction = float(mask.mean())
                if not 0.01 < fraction < 0.98:
                    debug = output / 'invalid_segmentation.png'
                    cv2.imwrite(str(debug), segmentation)
                    cv2.imwrite(str(output / 'invalid_rgb.png'), image)
                    print(f'debug_pose={pose} actual={robot.getPosition()} camera_objects={camera.getRecognitionNumberOfObjects()} colors={np.unique(segmentation.reshape(-1,4),axis=0).tolist()}', flush=True)
                    raise RuntimeError(f'Invalid lane mask {entry["track_id"]}/{i}: {fraction:.4f}; see {debug}')
                name = f'{track["track_id"]}_{i:04d}.png'
                image_rel = Path(entry['split']) / 'images' / name
                mask_rel = Path(entry['split']) / 'masks' / name
                for relative in (image_rel, mask_rel):
                    (output / relative).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output / image_rel), image) or not cv2.imwrite(str(output / mask_rel), mask):
                    raise OSError('Could not write RGB/mask pair')
                writer.writerow({'track_id': track['track_id'], 'track_seed': track['seed'],
                                 'split': entry['split'], 'sample_index': i,
                                 'image': image_rel.as_posix(), 'mask': mask_rel.as_posix(),
                                 'image_sha256': hashlib.sha256((output / image_rel).read_bytes()).hexdigest(),
                                 'mask_sha256': hashlib.sha256((output / mask_rel).read_bytes()).hexdigest(),
                                 'lane_fraction': fraction, **pose})
                total += 1
            handle.flush()
            print(f'captured={track["track_id"]} split={entry["split"]} samples={count} total={total}', flush=True)
    if total != manifest['expected_samples']:
        raise RuntimeError(f'Incomplete capture {total}/{manifest["expected_samples"]}')
    (output / 'capture_complete.json').write_text(json.dumps({'status': 'complete', 'samples': total,
         'webots_version': 'R2025a', 'width': camera.getWidth(), 'height': camera.getHeight()}, indent=2), encoding='utf-8')
    print(f'dataset_status=complete samples={total} output={output}', flush=True)
    for _ in range(3):
        supervisor.step(step)
    supervisor.simulationQuit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        if ACTIVE_SUPERVISOR is not None:
            for _ in range(3):
                if ACTIVE_SUPERVISOR.step(int(ACTIVE_SUPERVISOR.getBasicTimeStep())) == -1:
                    break
            ACTIVE_SUPERVISOR.simulationQuit(1)
        raise
