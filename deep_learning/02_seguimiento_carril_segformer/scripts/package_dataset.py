"""Package an audited dataset separately from Git."""

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', type=Path, default=ROOT / 'data' / 'lane_segmentation_v1')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    report = json.loads((dataset / 'audit_report.json').read_text(encoding='utf-8'))
    if report['status'] != 'PASS' or not (dataset / 'capture_complete.json').exists():
        raise RuntimeError('Only complete, audited datasets can be packaged.')
    destination = dataset.with_suffix('.zip')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for file in sorted(dataset.rglob('*')):
            if file.is_file():
                archive.write(file, arcname=file.relative_to(dataset.parent))
    print(f'archive={destination} size_mb={destination.stat().st_size / 1024**2:.1f}')


if __name__ == '__main__':
    main()
