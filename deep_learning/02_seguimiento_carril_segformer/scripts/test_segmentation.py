"""Checks for label preservation, paired augmentation and pixel metric formulas."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segformer_utils import LaneDataset, MEAN, STD, metrics_from_confusion, visible_mask


class SegmentationTests(unittest.TestCase):
    def test_known_pixel_counts(self):
        result = metrics_from_confusion([[8, 2], [1, 9]])
        self.assertAlmostEqual(result['pixel_accuracy'], 17 / 20)
        self.assertAlmostEqual(result['lane_iou'], 9 / 12)
        self.assertAlmostEqual(result['lane_dice'], 18 / 21)
        self.assertAlmostEqual(result['mean_iou'], (8 / 11 + 9 / 12) / 2)

    def test_palette_preserves_ids(self):
        labels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mask.png'
            visible_mask(labels).save(path)
            restored = Image.open(path)
            np.testing.assert_array_equal(np.asarray(restored), labels)
            self.assertEqual(tuple(np.asarray(restored.convert('RGB'))[0, 1]), (255, 255, 255))

    def test_paired_horizontal_reflection(self):
        labels = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.repeat((labels * 255)[:, :, None], 3, axis=2)).save(root / 'image.png')
            Image.fromarray(labels).save(root / 'mask.png')
            with (root / 'metadata.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['split', 'track_id', 'image', 'mask'])
                writer.writeheader()
                writer.writerow(dict(split='train', track_id='track_000', image='image.png', mask='mask.png'))
            with patch('random.random', return_value=0), patch('random.uniform', return_value=1):
                image, target, _ = LaneDataset(root, 'train', augment=True)[0]
            expected = np.fliplr(labels)
            np.testing.assert_array_equal(target.numpy(), expected)
            restored = image.numpy() * STD + MEAN
            np.testing.assert_allclose(restored[0], expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
