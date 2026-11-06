"""Create a compact preview of the generated dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/synthetic_commands"))
    args = parser.parse_args()
    root = args.dataset.resolve()
    rows = list(csv.DictReader((root / "metadata.csv").open(encoding="utf-8")))

    classes = ("forward", "left", "right", "stop")
    splits = ("train", "val", "test")
    tile_size = 224
    label_height = 28
    canvas = Image.new("RGB", (tile_size * len(classes), (tile_size + label_height) * len(splits)), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, split in enumerate(splits):
        for column_index, class_name in enumerate(classes):
            candidates = [row for row in rows if row["split"] == split and row["class"] == class_name]
            if not candidates:
                continue
            sample = candidates[len(candidates) // 2]
            image = Image.open(root / sample["relative_path"]).convert("RGB")
            x = column_index * tile_size
            y = row_index * (tile_size + label_height)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y + tile_size, x + tile_size, y + tile_size + label_height), fill="white")
            draw.text((x + 6, y + tile_size + 6), f"{split} / {class_name}", fill="black")

    output = root / "dataset_preview.png"
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
