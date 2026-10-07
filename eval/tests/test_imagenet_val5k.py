"""Two offline checks for the fixed preprocessing and shared-GT runner."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from eval.imagenet_val5k import (center_square_linear, prepare_dataset,
                                 prepare_fid, run_checkpoint, validate_dataset)
from eval.tests.test_acceleration import CacheTestMetric
from src.lab import lab_to_rgb, rgb_to_lab
from src.model import PixelMeanFlowB


class Val5kTests(unittest.TestCase):
    def test_center_square_then_linear_resize_with_odd_margin(self):
        for shape, crop in (((301, 482, 3), (0, 90, 301)),
                            ((482, 301, 3), (90, 0, 301))):
            rgb = np.random.default_rng(4).integers(0, 256, shape, dtype=np.uint8)
            result, actual_crop = center_square_linear(rgb)
            top, left, side = crop
            expected = cv2.resize(rgb[top:top + side, left:left + side], (256, 256),
                                  interpolation=cv2.INTER_LINEAR)
            self.assertEqual(actual_crop, list(crop))
            np.testing.assert_array_equal(result, expected)

    def test_prepare_cache_and_checkpoint_use_identical_shared_rgb(self):
        with tempfile.TemporaryDirectory() as temporary, patch("eval.imagenet_val5k.COUNT", 2):
            root = Path(temporary)
            source, dataset, output = root / "source", root / "prepared", root / "run"
            # Deliberately reverse class-folder ordering vs validation filename ordering.
            for number, folder in ((1, "z"), (2, "a")):
                (source / folder).mkdir(parents=True)
                rgb = np.random.default_rng(number).integers(0, 256, (71, 99, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(source / folder / f"ILSVRC2012_val_{number:08d}.png")
            manifest = prepare_dataset(source, dataset, progress=None)
            self.assertEqual([row["image_id"] for row in manifest["images"]],
                             ["ILSVRC2012_val_00000001", "ILSVRC2012_val_00000002"])
            self.assertEqual(prepare_dataset(source, dataset, progress=None), manifest)
            metric = CacheTestMetric()
            with patch("eval.imagenet_val5k.FIDMetric", return_value=metric):
                cache = prepare_fid(dataset, progress=None)
                self.assertEqual(prepare_fid(dataset, progress=None), cache)
                self.assertEqual(metric.calls, 1)

            architecture = dict(resolution=256, patch_size=32, hidden_size=16,
                                depth=2, num_heads=2, aux_head_depth=1, pca_channels=4)
            model = PixelMeanFlowB(**architecture).eval()
            checkpoint = root / "tiny.ckpt"
            torch.save({"state_dict": model.state_dict(),
                        "hyper_parameters": {"model": architecture, "noise_scale": 0.25},
                        "datamodule_hyper_parameters": {"resize_strategy": "center_crop"}}, checkpoint)
            report = run_checkpoint(checkpoint, dataset, output, metrics=("psnr", "ssim"),
                                    warmup=0, inference_batch_size=2)
            self.assertEqual(report["coverage"]["evaluated_images"], 2)
            self.assertFalse((output / "ground_truth").exists())
            self.assertEqual(report["config"]["resize"], None)
            for row in report["per_sample"]:
                target = Path(row["ground_truth"])
                self.assertEqual(target.parent, dataset / "rgb")
                rgb = np.asarray(Image.open(target))
                luminance, _ = rgb_to_lab(rgb)
                with torch.inference_mode():
                    ab = model.sample(luminance.unsqueeze(0), seeds=[1],
                                      image_ids=[row["image_id"]], noise_scale=0.25)
                expected = lab_to_rgb(luminance.unsqueeze(0), ab)[0]
                np.testing.assert_array_equal(np.asarray(Image.open(row["prediction"])), expected)
            self.assertEqual(json.loads((output / "generation.json").read_text())["resize_strategy"],
                             "prepared")
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "content changed"):
                validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
