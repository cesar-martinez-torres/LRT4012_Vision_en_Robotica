"""Export the fine-tuned SegFormer and compare real images in both runtimes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from segformer_utils import full_resolution_logits, preprocess

ROOT = Path(__file__).resolve().parent


class FullResolutionSegformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return full_resolution_logits(self.model, images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/segformer_b0')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/lane_segmentation_v1')
    parser.add_argument('--samples', type=int, default=24)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    torch.set_num_threads(2)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.run / 'best_model', local_files_only=True).eval().cpu()
    wrapper = FullResolutionSegformer(model).eval()
    output = args.run / 'best_model.onnx'
    with torch.inference_mode():
        torch.onnx.export(wrapper, torch.zeros(1, 3, 192, 320), str(output),
                          input_names=['images'], output_names=['logits'],
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    with (args.dataset / 'metadata.csv').open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['split'] == 'test']
    selected = np.linspace(0, len(rows) - 1, min(args.samples, len(rows)), dtype=int)
    errors, disagreements, pixels = [], 0, 0
    for index in selected:
        with Image.open(args.dataset / rows[index]['image']) as image:
            inputs = preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            reference = wrapper(inputs).numpy()
        prediction = session.run(['logits'], {'images': inputs.numpy()})[0]
        if prediction.shape != (1, 2, 192, 320) or not np.isfinite(prediction).all():
            raise RuntimeError(f'Invalid ONNX output: {prediction.shape}')
        errors.append(float(np.max(np.abs(reference - prediction))))
        disagreements += int(np.count_nonzero(reference.argmax(1) != prediction.argmax(1)))
        pixels += 192 * 320
    report = {'samples': len(selected), 'test_tracks': sorted({rows[i]['track_id'] for i in selected}),
              'max_absolute_logit_error': max(errors), 'disagreeing_pixels': disagreements,
              'mask_agreement': 1 - disagreements / pixels, 'opset': 17,
              'input_shape': [1, 3, 192, 320], 'output_shape': [1, 2, 192, 320],
              'provider': session.get_providers(), 'onnx_file': output.name}
    if max(errors) > 0.002 or report['mask_agreement'] < 0.9999:
        raise RuntimeError(f'ONNX comparison failed: {report}')
    metadata = {'width': 320, 'height': 192, 'color': 'RGB', 'scale': 255.0,
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'classes': ['background', 'lane'], 'output': 'full-resolution logits',
                'mask': 'argmax over class axis 1', 'camera_fov_rad': 1.1,
                'camera_height_m': 0.213, 'camera_forward_offset_m': 0.03}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.run / 'onnx_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
