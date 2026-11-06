"""Audit counts, dimensions, run separation, and exact duplicates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/synthetic_commands"))
    args = parser.parse_args()
    root = args.dataset.resolve()
    metadata_path = root / "metadata.csv"
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError("The metadata table is empty.")

    counts = Counter()
    runs_by_split = defaultdict(set)
    hashes = defaultdict(list)
    bad_dimensions = []
    missing = []
    for row in rows:
        counts[(row["split"], row["class"])] += 1
        runs_by_split[row["split"]].add(int(row["run_id"]))
        image_path = root / row["relative_path"]
        if not image_path.is_file():
            missing.append(row["relative_path"])
            continue
        with Image.open(image_path) as image:
            if image.size != (224, 224):
                bad_dimensions.append((row["relative_path"], image.size))
        hashes[sha256(image_path)].append(row)

    split_names = sorted(runs_by_split)
    overlaps = {}
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            common = sorted(runs_by_split[first] & runs_by_split[second])
            if common:
                overlaps[f"{first}-{second}"] = common

    cross_split_duplicates = []
    for digest, duplicate_rows in hashes.items():
        splits = sorted({row["split"] for row in duplicate_rows})
        if len(splits) > 1:
            cross_split_duplicates.append(
                {"sha256": digest, "splits": splits,
                 "files": [row["relative_path"] for row in duplicate_rows]}
            )

    report = {
        "root": str(root),
        "samples": len(rows),
        "counts": {f"{split}/{label}": value for (split, label), value in sorted(counts.items())},
        "runs_by_split": {split: sorted(values) for split, values in sorted(runs_by_split.items())},
        "missing_files": missing,
        "bad_dimensions": bad_dimensions,
        "run_overlap": overlaps,
        "cross_split_exact_duplicates": cross_split_duplicates,
        "status": "PASS" if not (missing or bad_dimensions or overlaps or cross_split_duplicates) else "FAIL",
    }
    (root / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
