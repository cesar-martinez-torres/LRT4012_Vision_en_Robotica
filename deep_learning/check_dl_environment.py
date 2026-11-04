"""Verify the LRT4012 deep-learning environment and its compute backend."""

from __future__ import annotations

import platform

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import transformers
import ultralytics


def main() -> None:
    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"TorchVision:  {torchvision.__version__}")
    print(f"OpenCV:       {cv2.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"Ultralytics:  {ultralytics.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"CUDA ready:   {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        tensor = torch.rand((1024, 1024), device="cuda")
        result = tensor @ tensor.T
        torch.cuda.synchronize()
        print(f"GPU test:     OK ({result.shape[0]} x {result.shape[1]})")
    else:
        tensor = torch.rand((256, 256))
        result = tensor @ tensor.T
        print(f"CPU test:     OK ({result.shape[0]} x {result.shape[1]})")


if __name__ == "__main__":
    main()
