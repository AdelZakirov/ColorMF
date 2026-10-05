import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image
import torch

from eval.original_size import colorize, original_size_rgb, prepare_luminance
from src.lab import normalize_ab
from src.model import PixelMeanFlowB


class OriginalSizeTests(unittest.TestCase):
    def test_extract_l_before_resizing_and_normalize_after(self):
        image = np.random.default_rng(5).integers(0, 256, (37, 25, 3), dtype=np.uint8)
        original_L, model_L = prepare_luminance(image, (16, 16))
        encoded_L = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0]
        expected = cv2.resize(encoded_L, (16, 16), interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(((original_L[0, 0] + 1) * 127.5).round().numpy(), encoded_L)
        torch.testing.assert_close(model_L[0], torch.from_numpy(expected).float() / 127.5 - 1,
                                   rtol=0, atol=0)
        # Detect the earlier pipeline's RGB-before-L resize, which gives different pixels.
        wrong = cv2.cvtColor(cv2.resize(image, (16, 16), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2LAB)[..., 0]
        self.assertFalse(np.array_equal(expected, wrong))

    def test_chroma_is_resized_continuously_and_original_l_is_unchanged(self):
        # The center of a symmetric 2x2 chroma field interpolates to encoded
        # a=126.99 -> 127. Rounding before interpolation would lose that detail.
        L = torch.full((1, 1, 5, 5), 128 / 127.5 - 1)
        before = L.clone()
        ab = torch.full((1, 2, 2, 2), 128 / 127.5 - 1)
        ab[:, 0, :, 0] = 126.49 / 127.5 - 1
        ab[:, 0, :, 1] = 127.49 / 127.5 - 1
        actual = original_size_rgb(L, ab)
        expected_center = cv2.cvtColor(np.array([[[128, 127, 128]]], dtype=np.uint8),
                                      cv2.COLOR_LAB2RGB)[0, 0]
        self.assertEqual(actual.shape, (1, 5, 5, 3))
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_array_equal(actual[0, 2, 2], expected_center)
        torch.testing.assert_close(L, before, rtol=0, atol=0)

    def test_alpha_scales_physical_chroma_and_keeps_neutral_and_l(self):
        L = torch.full((1, 1, 3, 3), 150 / 127.5 - 1)
        physical = torch.tensor([[[[0., 20., -20.]]], [[[0., -10., 10.]]]]).reshape(1, 2, 1, 3)
        ab = normalize_ab(physical)
        before_L, before_ab = L.clone(), ab.clone()
        actual = original_size_rgb(L, ab, alpha=1.3)
        encoded = np.array([[[150, 128, 128], [150, 154, 115], [150, 102, 141]]], dtype=np.uint8)
        expected = cv2.cvtColor(np.repeat(encoded, 3, axis=0), cv2.COLOR_LAB2RGB)
        np.testing.assert_array_equal(actual[0], expected)
        np.testing.assert_array_equal(original_size_rgb(L, ab), original_size_rgb(L, ab, alpha=1.0))
        torch.testing.assert_close(L, before_L, rtol=0, atol=0)
        torch.testing.assert_close(ab, before_ab, rtol=0, atol=0)
        for alpha in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                original_size_rgb(L, ab, alpha=alpha)

    def test_actual_small_model_keeps_variable_original_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            sizes = [(37, 25), (20, 40), (19, 27)]
            for index, (height, width) in enumerate(sizes):
                image = np.random.default_rng(index).integers(0, 256, (height, width, 3), dtype=np.uint8)
                Image.fromarray(image).save(source / f"{index}.png")
            architecture = dict(resolution=16, patch_size=4, hidden_size=32, depth=2,
                                num_heads=4, aux_head_depth=1, pca_channels=8)
            model = PixelMeanFlowB(**architecture)
            checkpoint = root / "last.ckpt"
            torch.save({"hyper_parameters": {"model": architecture, "noise_scale": .25},
                        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}}, checkpoint)
            metadata = colorize(checkpoint, source, root / "output", batch_size=2, progress=None)
            threaded = colorize(checkpoint, source, root / "threaded", batch_size=2,
                                workers=2, progress=None)
            self.assertEqual((root / "threaded" / "generation.jsonl").read_text().count("\n"), 3)
            self.assertFalse((root / "threaded" / "generation_pending.json").exists())
            for original, parallel in zip(metadata["exported_predictions"], threaded["exported_predictions"]):
                self.assertEqual(Path(original["prediction"]).read_bytes(),
                                 Path(parallel["prediction"]).read_bytes())
            self.assertEqual(len(metadata["exported_predictions"]), 3)
            for row, (height, width) in zip(metadata["exported_predictions"], sizes):
                with Image.open(row["prediction"]) as image:
                    self.assertEqual(image.size, (width, height))
                    self.assertEqual(image.mode, "RGB")
                self.assertEqual((row["height"], row["width"]), (height, width))
            with self.assertRaisesRegex(ValueError, "empty"):
                colorize(checkpoint, source, root / "output", progress=None)
            continuation = colorize(checkpoint, source, root / "next", offset=1,
                                    limit=1, batch_size=2, progress=None)
            self.assertEqual(continuation["selected_image_ids"], ["1"])
            with Image.open(continuation["exported_predictions"][0]["prediction"]) as image:
                self.assertEqual(image.size, (40, 20))
            with self.assertRaisesRegex(ValueError, "offset"):
                colorize(checkpoint, source, root / "invalid", offset=-1, progress=None)
            with self.assertRaisesRegex(ValueError, "workers"):
                colorize(checkpoint, source, root / "invalid", workers=0, progress=None)


if __name__ == "__main__":
    unittest.main()
