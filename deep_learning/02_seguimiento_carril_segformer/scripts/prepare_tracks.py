"""Create reproducible track specifications; never replace an existing dataset."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lane_geometry import make_track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'lane_segmentation_v1')
    parser.add_argument('--seed', type=int, default=4012)
    parser.add_argument('--samples-per-track', type=int, default=96)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}; choose a new --output.')
    if args.samples_per_track < 4:
        raise ValueError('At least four samples per track are required.')
    args.output.mkdir(parents=True)
    tracks_dir = args.output / 'tracks'
    tracks_dir.mkdir()
    sizes = {'train': 2, 'val': 1, 'test': 1} if args.smoke else {'train': 20, 'val': 6, 'test': 6}
    seed_values = np.random.default_rng(args.seed).choice(2000000000, sum(sizes.values()), replace=False)
    tracks = []
    cursor = 0
    for split, count in sizes.items():
        for _ in range(count):
            track_id = f'track_{cursor:03d}'
            track = make_track(int(seed_values[cursor]), track_id, split)
            file = tracks_dir / f'{track_id}.json'
            file.write_text(json.dumps(track, indent=2), encoding='utf-8')
            tracks.append({'track_id': track_id, 'split': split, 'seed': track['seed'],
                           'file': f'tracks/{file.name}'})
            cursor += 1
    manifest = {'version': 1, 'task': 'binary_drivable_lane_segmentation',
                'model_target': 'SegFormer-B0', 'seed': args.seed, 'width': 320, 'height': 192,
                'classes': {'0': 'background', '1': 'lane'},
                'mask_encoding': 'single-channel uint8 PNG; raw class IDs 0 and 1',
                'label_source': 'Webots Recognition segmentation; lane tiles red (255,0,0)',
                'camera': {'field_of_view_rad': 1.1, 'mount_xyz_m': [0.03, 0.0, 0.213],
                           'pitch': 'varies per track; see track specifications'},
                'capture_method': 'kinematic camera at Burger mount; supervised poses, not autonomous neural driving',
                'split_rule': 'disjoint complete tracks; geometry and appearance fixed per track',
                'samples_per_track': args.samples_per_track,
                'expected_samples': len(tracks) * args.samples_per_track,
                'track_counts': sizes, 'tracks': tracks}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {len(tracks)} tracks at {args.output}; expected samples={manifest["expected_samples"]}', flush=True)


if __name__ == '__main__':
    main()
