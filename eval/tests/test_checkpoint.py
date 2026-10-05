"""Real small-model inference tests; no pretrained weights/downloads."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from eval.checkpoint import generate, load_model, prepare_rgb
from eval.config import EvalConfig
from eval.data import discover
from src.data import PaletteDataset
from src.lab import rgb_to_lab
from src.model import PixelMeanFlowB


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.architecture = dict(resolution=16, patch_size=4, hidden_size=32,
                                 depth=2, num_heads=4, aux_head_depth=1, pca_channels=8)
        torch.manual_seed(5)
        self.model = PixelMeanFlowB(**self.architecture).eval()
        torch.nn.init.normal_(self.model.u_final_layer.linear._flax_linear.weight, std=0.02)
        self.checkpoint = self.root / "last.ckpt"
        self.payload = {
            "state_dict": {f"model.{key}": value for key, value in self.model.state_dict().items()},
            "hyper_parameters": {"model": self.architecture, "noise_scale": 0.25,
                                 "ema_validation_variant": "500"},
            "datamodule_hyper_parameters": {"resize_strategy": "center_crop"},
            "epoch": 3, "global_step": 20,
        }
        self.save_checkpoint()
        self.truth = self.root / "source"
        (self.truth / "class_a").mkdir(parents=True)
        self.image = np.random.default_rng(2).integers(0, 256, (37, 25, 3), dtype=np.uint8)
        Image.fromarray(self.image).save(self.truth / "class_a/one.png")
        Image.fromarray(self.image[::-1]).save(self.truth / "two.png")

    def save_checkpoint(self):
        torch.save(self.payload, self.checkpoint)

    def test_raw_loader_preserves_buffers_noise_and_sample(self):
        loaded, metadata = load_model(self.checkpoint)
        # RoPE is not persisted in checkpoints. This catches invalid meta/to_empty loading.
        self.assertNotIn("rope_freqs", self.model.state_dict())
        torch.testing.assert_close(loaded.rope_freqs, self.model.rope_freqs)
        luminance = torch.randn(1, 1, 16, 16)
        expected = self.model.sample(luminance, seed=7, image_ids=["one"], noise_scale=0.25)
        actual = loaded.sample(luminance, seed=7, image_ids=["one"], noise_scale=metadata["noise_scale"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(metadata["weights"], "raw")
        self.assertEqual(metadata["global_step"], 20)

    def test_ema_selection_and_raw_override(self):
        shadow = {key: value.clone() for key, value in self.model.state_dict().items()}
        key = "u_final_layer.linear._flax_linear.bias"
        shadow[key].add_(0.5)
        self.payload["ema"] = {"started": True, "shadows": {"500": shadow}}
        self.save_checkpoint()
        loaded, metadata = load_model(self.checkpoint)
        torch.testing.assert_close(loaded.state_dict()[key], shadow[key])
        self.assertEqual(metadata["ema_variant"], "500")
        raw, metadata = load_model(self.checkpoint, use_ema=False)
        torch.testing.assert_close(raw.state_dict()[key], self.model.state_dict()[key])
        self.assertEqual(metadata["weights"], "raw")
        with self.assertRaisesRegex(ValueError, "unknown EMA"):
            load_model(self.checkpoint, ema_variant="1000")

    def test_legacy_ema_and_unready_failure(self):
        self.payload["hyper_parameters"]["ema_validation_variant"] = "fixed"
        self.payload["ema"] = {"num_updates": 1, "shadow": self.model.state_dict()}
        self.save_checkpoint()
        _, metadata = load_model(self.checkpoint)
        self.assertEqual(metadata["ema_variant"], "fixed")
        self.payload["ema"]["started"] = False
        self.save_checkpoint()
        with self.assertRaisesRegex(ValueError, "not ready"):
            load_model(self.checkpoint)

    def test_strict_architecture_and_missing_metadata(self):
        self.payload["state_dict"].pop("model.time_tokens")
        self.save_checkpoint()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_model(self.checkpoint)
        self.payload.pop("hyper_parameters")
        self.save_checkpoint()
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            load_model(self.checkpoint)

    def test_preprocessing_matches_training_luminance(self):
        path = self.truth / "class_a/one.png"
        for strategy in ("center_crop", "stretch"):
            expected = PaletteDataset(paths=[str(path)], size=(16, 16), resize_strategy=strategy)[0]
            prepared = prepare_rgb(path, (16, 16), strategy)
            luminance, chroma = rgb_to_lab(prepared)
            torch.testing.assert_close(luminance, expected["L"], rtol=0, atol=0)
            torch.testing.assert_close(chroma, expected["ab"], rtol=0, atol=0)

    def test_generation_is_independent_of_batch_size_and_pairs_nested_ids(self):
        reports = []
        for size in (1, 3):
            output = self.root / f"run_{size}"
            records, metadata = generate(self.checkpoint, self.truth, output,
                                         sample_seeds=(2, 1), batch_size=size, warmup=1, progress=None)
            config = EvalConfig(output / "predictions", output / "ground_truth", output=output,
                                layout="sample-dirs", sample_ids=("seed_2", "seed_1"))
            discovered, coverage = discover(config)
            self.assertEqual(discovered, records)
            self.assertEqual(coverage["predictions"], 4)
            self.assertEqual(metadata["timing"]["nfe"], 1)
            self.assertEqual(metadata["timing"]["model_calls"], 4 if size == 1 else 2)
            self.assertGreater(metadata["timing"]["throughput_images_per_second"], 0)
            reports.append([np.asarray(Image.open(sample.path)) for record in records for sample in record.samples])
            target = np.asarray(Image.open(records[0].ground_truth))
            np.testing.assert_array_equal(target, prepare_rgb(self.truth / "class_a/one.png", (16, 16), "center_crop"))
        for first, second in zip(*reports):
            np.testing.assert_array_equal(first, second)

    def test_invalid_selection_and_output_protection(self):
        for kwargs in ({"sample_seeds": (1, 1)}, {"batch_size": 0}, {"limit": 0}, {"warmup": -1}):
            with self.assertRaises(ValueError):
                generate(self.checkpoint, self.truth, self.root / "invalid", progress=None, **kwargs)
        with self.assertRaisesRegex(ValueError, "overlap"):
            generate(self.checkpoint, self.truth, self.truth / "out", progress=None)
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            generate(self.checkpoint, self.truth, existing, progress=None)
        self.assertEqual((existing / "keep.txt").read_text(), "keep")

    def test_limit_sorts_image_ids_independently_of_extensions(self):
        # File order is a.b.png, a.png; image ID order must be a, a.b.
        Image.fromarray(self.image).save(self.truth / "a.b.png")
        Image.fromarray(self.image).save(self.truth / "a.png")
        records, metadata = generate(self.checkpoint, self.truth, self.root / "limited",
                                     limit=1, warmup=0, progress=None)
        self.assertEqual([record.image_id for record in records], ["a"])
        self.assertEqual(metadata["selected_image_ids"], ["a"])

    def test_cli_real_checkpoint_report_and_saved_image_reuse(self):
        output = self.root / "cli"
        result = subprocess.run([
            sys.executable, "-m", "eval.checkpoint", "--checkpoint", str(self.checkpoint),
            "--ground-truth", str(self.truth), "--output", str(output), "--device", "cpu",
            "--sample-seeds", "2", "1", "--inference-batch-size", "3",
            "--metrics", "psnr", "ssim", "delta_e00"], capture_output=True, text=True, check=True)
        self.assertIn("Evaluated 2 images / 4 predictions", result.stdout)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["timing"]["inference"]["status"], "measured")
        self.assertEqual(report["inference"]["noise_scale"], 0.25)
        self.assertEqual(report["per_image"][0]["first_sample_id"], "seed_2")
        reuse = self.root / "reuse"
        subprocess.run([sys.executable, "-m", "eval", "--predictions", str(output / "predictions"),
                        "--ground-truth", str(output / "ground_truth"), "--output", str(reuse),
                        "--layout", "sample-dirs", "--sample-ids", "seed_2", "seed_1",
                        "--metrics", "psnr", "ssim", "delta_e00"], capture_output=True, text=True, check=True)
        reused = json.loads((reuse / "report.json").read_text())
        self.assertEqual(report["aggregate"], reused["aggregate"])


if __name__ == "__main__":
    unittest.main()
