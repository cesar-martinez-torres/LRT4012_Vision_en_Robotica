"""Export the trained MobileNetV3-Small checkpoint for Webots inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mobilenet_v3_small" / "best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mobilenet_v3_small" / "best_model.onnx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(checkpoint["image_size"])
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported={args.output}")
    print(f"classes={classes} image_size={image_size}")


if __name__ == "__main__":
    main()
