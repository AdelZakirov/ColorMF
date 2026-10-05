from __future__ import annotations

import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import struct
import zlib
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image, ImageCms

from eval.__main__ import main, parse_config
from eval.config import METRICS, EvalConfig
from eval.data import discover, load_rgb
from eval.metrics.fid import FIDMetric
from eval.metrics.lpips import LPIPSMetric
from eval.pipeline import evaluate
from eval.reporting import save_report, summary
from eval.tests.mocks import MockDistance, MockFeatures

PIXEL_METRICS = ("psnr", "ssim", "colorfulness", "delta_colorfulness", "delta_e00")


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.gt, self.pred = self.root / "gt", self.root / "pred"
        self.config = EvalConfig(self.pred, self.gt, output=self.root / "out", metrics=PIXEL_METRICS)
        self.image = np.full((64, 64, 3), 120, dtype=np.uint8)

    def pair(self, name="one", image=None):
        image = self.image if image is None else image
        save(self.gt / f"{name}.png", image)
        save(self.pred / f"{name}.png", image)

    def stochastic_fixture(self, count=4):
        for i in range(count):
            image = np.random.default_rng(i).integers(20, 200, size=(64, 64, 3), dtype=np.uint8)
            save(self.gt / f"nested/{i}.png", image)
            save(self.pred / f"seed_1/nested/{i}.png", 255 - image)
            save(self.pred / f"seed_2/nested/{i}.png", image)

    def test_single_pairing_and_strict_json_csv_identity(self):
        self.pair("nested/one")
        report = evaluate(self.config)
        self.assertEqual(report["coverage"]["evaluated_images"], 1)
        self.assertEqual(report["per_image"][0]["image_id"], "nested/one")
        self.assertEqual(report["aggregate"]["mean_over_samples"]["delta_e00"], 0)
        self.assertIsNone(report["fid"])
        self.assertEqual(report["timing"]["inference"]["status"], "not_measured")
        self.assertIsNone(report["timing"]["inference"]["nfe"])
        save_report(report, self.config.output)
        loaded = json.loads((self.config.output / "report.json").read_text(),
                            parse_constant=lambda v: self.fail(f"invalid JSON constant {v}"))
        self.assertEqual(loaded["per_sample"][0]["metrics"]["psnr"], "+Infinity")
        with (self.config.output / "per_sample.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["metrics.psnr"], "+Infinity")
        self.assertTrue((self.config.output / "per_image.csv").is_file())
        self.assertTrue((self.config.output / "summary.csv").is_file())

    def test_all_metrics_stochastic_pipeline_with_injected_networks(self):
        self.stochastic_fixture()
        config = replace(self.config, layout="sample-dirs", metrics=METRICS)
        report = evaluate(config, lpips_metric=LPIPSMetric(network=MockDistance()),
                          fid_metric=FIDMetric(network=MockFeatures(), dims=3, batch_size=3))
        self.assertEqual(report["coverage"]["predictions"], 8)
        self.assertEqual(report["coverage"]["k_min"], 2)
        self.assertEqual(report["fid"]["real_count"], 4)
        self.assertEqual(report["fid"]["generated_count"], 4)
        self.assertGreater(report["fid"]["value"], 0)
        self.assertEqual(report["aggregate"]["stochastic"]["best_of_k_lpips"], 0)
        self.assertEqual(report["aggregate"]["stochastic"]["best_of_k_delta_e00"], 0)
        self.assertGreater(report["aggregate"]["stochastic"]["lpips_diversity"], 0)
        for row in report["per_image"]:
            self.assertEqual(row["best_of_k"]["lpips"]["sample_id"], "seed_2")
            self.assertEqual(row["best_of_k"]["delta_e00"]["sample_id"], "seed_2")
            self.assertEqual(row["diversity_pairs"], 1)
        save_report(report, config.output)
        self.assertIn("TECHNICAL TEST", summary(report))
        self.assertTrue(report["protocol"]["injected_neural_backends"])

    def test_parallel_image_metrics_match_serial_and_preserve_order(self):
        self.stochastic_fixture(7)
        config = replace(self.config, layout="sample-dirs", metrics=METRICS)
        phases = []
        ready = []
        reports = [evaluate(replace(config, workers=workers),
                            lpips_metric=LPIPSMetric(network=MockDistance(), batch_size=3),
                            fid_metric=FIDMetric(network=MockFeatures(), dims=3, batch_size=3),
                            phase_progress=lambda phase, done, total: phases.append((phase, done, total)),
                            before_image=lambda record: ready.append(record.image_id))
                   for workers in (1, 3)]
        for key in ("aggregate", "per_image", "per_sample", "fid", "coverage"):
            self.assertEqual(reports[0][key], reports[1][key])
        self.assertEqual(sorted(ready), sorted([row["image_id"] for row in reports[0]["per_image"]] * 2))
        for phase in ("image_metrics", "lpips", "fid_real", "fid_generated"):
            self.assertEqual([(done, total) for name, done, total in phases if name == phase],
                             [(i, 7) for i in range(1, 8)] * 2)

    def test_fid_first_selection_is_configurable_and_all_counts(self):
        self.stochastic_fixture()
        config = replace(self.config, layout="sample-dirs", metrics=("fid",),
                         sample_ids=("seed_2", "seed_1"))
        report = evaluate(config, fid_metric=FIDMetric(network=MockFeatures(), dims=3))
        self.assertAlmostEqual(report["fid"]["value"], 0, places=6)
        self.assertEqual(report["per_image"][0]["first_sample_id"], "seed_2")
        report = evaluate(replace(config, fid_sampling="all"),
                          fid_metric=FIDMetric(network=MockFeatures(), dims=3))
        self.assertEqual(report["fid"]["real_count"], 4)
        self.assertEqual(report["fid"]["generated_count"], 8)
        self.assertTrue(any("all generated" in warning for warning in report["warnings"]))

    def test_missing_extra_duplicate_and_subset(self):
        self.pair()
        save(self.gt / "missing.png", self.image)
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            discover(self.config)
        records, coverage = discover(replace(self.config, allow_subset=True))
        self.assertEqual(len(records), 1)
        self.assertEqual(coverage["unevaluated_ground_truth_ids"], ["missing"])
        save(self.pred / "extra.png", self.image)
        with self.assertRaisesRegex(ValueError, "without ground truth"):
            discover(replace(self.config, allow_subset=True))
        (self.pred / "extra.png").unlink()
        save(self.pred / "one.jpg", self.image)
        with self.assertRaisesRegex(ValueError, "duplicate image ID"):
            discover(self.config)

    def test_uniform_k_coverage_and_sample_selection(self):
        self.stochastic_fixture(2)
        (self.pred / "seed_2/nested/0.png").unlink()
        config = replace(self.config, layout="sample-dirs")
        with self.assertRaisesRegex(ValueError, "variable K"):
            discover(config)
        report = evaluate(replace(config, allow_variable_k=True))
        self.assertEqual(report["coverage"]["k_min"], 1)
        self.assertEqual(report["coverage"]["k_max"], 2)
        self.assertTrue(any("K varies" in warning for warning in report["warnings"]))
        with self.assertRaisesRegex(ValueError, "missing requested samples"):
            discover(replace(config, sample_ids=("seed_2",)))
        records, _ = discover(replace(config, sample_ids=("seed_1",)))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(len(record.samples) == 1 for record in records))

    def test_sample_dirs_reject_different_seeds_with_equal_k(self):
        self.stochastic_fixture(2)
        path = self.pred / "seed_2/nested/0.png"
        target = self.pred / "seed_3/nested/0.png"
        target.parent.mkdir(parents=True)
        path.rename(target)
        with self.assertRaisesRegex(ValueError, "same image IDs"):
            discover(replace(self.config, layout="sample-dirs"))

    def test_per_image_layout_and_undefined_single_diversity(self):
        save(self.gt / "nested/one.png", self.image)
        save(self.pred / "nested/one/sample_0.png", self.image)
        report = evaluate(replace(self.config, layout="per-image", metrics=("lpips",)),
                          lpips_metric=LPIPSMetric(network=MockDistance()))
        self.assertIsNone(report["per_image"][0]["lpips_diversity"])
        self.assertNotIn("lpips_diversity", report["aggregate"]["stochastic"])
        self.assertEqual(report["aggregate"]["stochastic"]["diversity_images"], 0)
        self.assertEqual(report["per_image"][0]["diversity_pairs"], 0)

    def test_macro_average_does_not_overweight_images_with_more_samples(self):
        # Delta E(black, white)=100, so equal input weight gives 50, not 25.
        black, white = np.zeros_like(self.image), np.full_like(self.image, 255)
        save(self.gt / "one.png", black)
        save(self.gt / "two.png", black)
        save(self.pred / "one/a.png", white)
        for name in ("a", "b", "c"):
            save(self.pred / f"two/{name}.png", black)
        report = evaluate(replace(self.config, layout="per-image", allow_variable_k=True,
                                  metrics=("delta_e00",)))
        self.assertAlmostEqual(report["aggregate"]["mean_over_samples"]["delta_e00"], 50, places=4)

    def test_best_of_k_selects_whole_image_instead_of_best_pixel(self):
        black = np.zeros_like(self.image)
        split = black.copy()
        split[:32] = 255
        save(self.gt / "one.png", black)
        save(self.pred / "one/a.png", split)
        save(self.pred / "one/b.png", 255 - split)
        report = evaluate(replace(self.config, layout="per-image", metrics=("delta_e00",)))
        self.assertAlmostEqual(report["aggregate"]["stochastic"]["best_of_k_delta_e00"], 50, places=4)

    def test_best_of_k_selects_independently_per_metric(self):
        save(self.gt / "one.png", self.image)
        save(self.pred / "one/a.png", self.image)
        save(self.pred / "one/b.png", 255 - self.image)
        metric = LPIPSMetric(network=MockDistance())
        with patch.object(metric, "iter_distances", return_value=iter([0.9, 0.1, 0.3])):
            report = evaluate(replace(self.config, layout="per-image", metrics=("lpips", "delta_e00")),
                              lpips_metric=metric)
        best = report["per_image"][0]["best_of_k"]
        self.assertEqual(best["lpips"]["sample_id"], "b")
        self.assertEqual(best["delta_e00"]["sample_id"], "a")

    def test_diversity_all_unordered_pairs_and_equal_image_weight(self):
        black = np.zeros_like(self.image)
        save(self.gt / "one.png", black)
        save(self.gt / "two.png", black)
        # Use 128/255 rather than exact 0.5 to exercise saved-image decoding.
        for sample, image in (("a", black), ("b", np.full_like(black, 128)),
                              ("c", np.full_like(black, 255))):
            save(self.pred / f"one/{sample}.png", image)
        for sample in ("a", "b"):
            save(self.pred / f"two/{sample}.png", black)
        report = evaluate(replace(self.config, layout="per-image", metrics=("lpips",),
                                  allow_variable_k=True), lpips_metric=LPIPSMetric(network=MockDistance()))
        row = report["per_image"][0]
        expected = (4 * (128 / 255) ** 2 + 4 + 4 * (127 / 255) ** 2) / 3
        self.assertEqual(row["diversity_pairs"], 3)
        self.assertAlmostEqual(row["lpips_diversity"], expected, places=6)
        self.assertAlmostEqual(report["aggregate"]["stochastic"]["lpips_diversity"], expected / 2, places=6)

    def test_absolute_per_image_cf_delta_differs_from_dataset_mean_gap(self):
        black, red = np.zeros_like(self.image), np.zeros_like(self.image)
        red[..., 0] = 255
        save(self.gt / "one.png", black)
        save(self.pred / "one.png", red)
        save(self.gt / "two.png", red)
        save(self.pred / "two.png", black)
        report = evaluate(replace(self.config, metrics=("delta_colorfulness",)))
        metrics = report["aggregate"]["mean_over_samples"]
        self.assertGreater(metrics["delta_colorfulness"], 80)
        self.assertEqual(metrics["delta_mean_colorfulness"], 0)

    def test_shape_mismatch_requires_explicit_resize(self):
        self.pair()
        save(self.pred / "one.png", self.image[:32, :32])
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            evaluate(self.config)
        report = evaluate(replace(self.config, resize=(16, 20)))
        self.assertEqual(report["per_sample"][0]["height"], 16)
        self.assertEqual(report["per_sample"][0]["width"], 20)

    def test_native_colormf_resize_has_no_error_for_perfect_reconstruction(self):
        # Independent reference: sample.prepare_input resizes uint8 RGB this way.
        for i in range(2):
            source = np.random.default_rng(i).integers(0, 256, (218, 178, 3), dtype=np.uint8)
            native = cv2.resize(source, (64, 64), interpolation=cv2.INTER_CUBIC)
            save(self.gt / f"{i}.png", source)
            save(self.pred / f"{i}.png", native)
        config = replace(self.config, metrics=METRICS, resize=(64, 64), resize_backend="opencv")
        report = evaluate(config, lpips_metric=LPIPSMetric(network=MockDistance()),
                          fid_metric=FIDMetric(network=MockFeatures(), dims=3))
        scores = report["aggregate"]["first_sample"]
        self.assertEqual(scores["psnr"], float("inf"))
        self.assertEqual(scores["ssim"], 1)
        self.assertEqual(scores["delta_e00"], 0)
        self.assertEqual(scores["lpips"], 0)
        self.assertAlmostEqual(report["fid"]["value"], 0, places=6)
        self.assertIn("OpenCV INTER_CUBIC", report["protocol"]["resize"])
        wrong = evaluate(replace(config, metrics=("delta_e00",), resize_backend="pillow"))
        self.assertGreater(wrong["aggregate"]["first_sample"]["delta_e00"], 1)

    def test_lpips_batches_across_inputs_and_keeps_partial_batch_and_order(self):
        class CountingDistance(MockDistance):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, first, second):
                self.batch_sizes.append(len(first))
                return super().forward(first, second)

        for i in range(40):
            self.pair(f"{i:02d}")
            save(self.pred / f"{i:02d}.png", np.full_like(self.image, i))
        network = CountingDistance()
        progress = []
        report = evaluate(replace(self.config, metrics=("lpips",), batch_size=32),
                          lpips_metric=LPIPSMetric(network=network, batch_size=32),
                          progress=lambda done, total: progress.append((done, total, len(network.batch_sizes))))
        self.assertEqual(network.batch_sizes, [32, 8])
        self.assertEqual([(done, total) for done, total, _ in progress],
                         [(done, 40) for done in range(1, 41)])
        self.assertTrue(all(calls > 0 for _, _, calls in progress))
        expected = [4 * ((i - 120) / 255) ** 2 for i in range(40)]
        np.testing.assert_allclose([row["metrics"]["lpips"] for row in report["per_sample"]],
                                   expected, atol=1e-6)
        for image, sample in zip(report["per_image"], report["per_sample"]):
            self.assertEqual(image["best_of_k"]["lpips"]["value"], sample["metrics"]["lpips"])
            self.assertIsNone(image["lpips_diversity"])

    def test_lpips_batch_size_does_not_change_variable_shape_and_k_results(self):
        for i, (size, k) in enumerate(((64, 2), (80, 3), (64, 1))):
            save(self.gt / f"{i}.png", np.full((size, size, 3), 120, dtype=np.uint8))
            for j in range(k):
                save(self.pred / f"{i}/{j}.png", np.full((size, size, 3), 30 * j, dtype=np.uint8))
        config = replace(self.config, layout="per-image", metrics=("lpips",), allow_variable_k=True)
        reports = [evaluate(config, lpips_metric=LPIPSMetric(network=MockDistance(), batch_size=n))
                   for n in (1, 32)]
        for key in ("aggregate", "per_image", "per_sample"):
            self.assertEqual(reports[0][key], reports[1][key])

    def test_fid_warns_about_ddcolor_protocol_for_production_backend(self):
        self.stochastic_fixture()
        metric = FIDMetric(network=MockFeatures(), dims=3)
        # Test the production warning without constructing/downloading Inception.
        with patch.object(metric, "injected", False):
            report = evaluate(replace(self.config, layout="sample-dirs", metrics=("fid",)),
                              fid_metric=metric)
        self.assertTrue(any("DDColor" in warning for warning in report["warnings"]))

    def test_decoder_grayscale_icc_transparency_and_high_depth(self):
        gray = self.root / "gray.png"
        save(gray, np.full((12, 12), 128, dtype=np.uint8))
        rgb = load_rgb(gray)
        np.testing.assert_allclose(rgb[..., 0], rgb[..., 1])
        self.assertEqual(rgb.dtype, np.float32)
        rgba = np.full((12, 12, 4), 255, dtype=np.uint8)
        rgba[0, 0, 3] = 0
        transparent = self.root / "transparent.png"
        save(transparent, rgba)
        with self.assertRaisesRegex(ValueError, "transparent"):
            load_rgb(transparent)
        high_depth = self.root / "depth.png"
        save(high_depth, np.full((12, 12), 1000, dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "unsupported image mode"):
            load_rgb(high_depth)
        tagged = self.root / "tagged.png"
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        Image.fromarray(self.image).save(tagged, icc_profile=profile)
        np.testing.assert_allclose(load_rgb(tagged), self.image.astype(np.float32) / 255)
        palette = self.root / "palette.png"
        Image.fromarray(self.image).quantize(colors=1).save(palette, icc_profile=profile)
        np.testing.assert_allclose(load_rgb(palette), self.image.astype(np.float32) / 255)

    def test_decoder_rejects_16bit_rgb_png_before_pillow_quantizes(self):
        def chunk(kind, payload):
            return (struct.pack(">I", len(payload)) + kind + payload +
                    struct.pack(">I", zlib.crc32(kind + payload)))
        path = self.root / "rgb16.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                         chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)) +
                         chunk(b"IDAT", zlib.compress(b"\0" + struct.pack(">HHH", 65535, 32768, 0))) +
                         chunk(b"IEND", b""))
        with self.assertRaisesRegex(ValueError, "high-bit-depth PNG"):
            load_rgb(path)

    def test_grayscale_with_rgb_icc_profile_is_expanded_before_conversion(self):
        gray = np.arange(144, dtype=np.uint8).reshape(12, 12)
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        filename = self.root / "gray_with_rgb_profile.png"
        Image.fromarray(gray).save(filename, icc_profile=profile)
        expected = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32) / 255
        np.testing.assert_array_equal(load_rgb(filename), expected)

    def test_rgb_with_incompatible_cmyk_profile_preserves_decoded_pixels(self):
        profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        profile[16:20] = b"CMYK"  # ICC header color space no longer matches the RGB pixels.
        filename = self.root / "rgb_with_cmyk_profile.png"
        pixels = np.random.default_rng(1).integers(0, 256, (12, 12, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(filename, icc_profile=bytes(profile))
        with self.assertWarnsRegex(RuntimeWarning, "incompatible CMYK"):
            actual = load_rgb(filename)
        np.testing.assert_array_equal(actual, pixels.astype(np.float32) / 255)

    def test_unreadable_icc_profile_warns_and_preserves_decoded_pixels(self):
        filename = self.root / "invalid_profile.png"
        Image.fromarray(self.image).save(filename, icc_profile=b"not an ICC profile")
        with self.assertWarnsRegex(RuntimeWarning, "unreadable ICC"):
            actual = load_rgb(filename)
        np.testing.assert_array_equal(actual, self.image.astype(np.float32) / 255)

    def test_parseable_but_unusable_icc_preserves_pixels(self):
        profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        profile[128:132] = b"\0\0\0\0"  # Valid header, no transform tags.
        filename = self.root / "unusable_profile.png"
        Image.fromarray(self.image).save(filename, icc_profile=bytes(profile))
        with self.assertWarnsRegex(RuntimeWarning, "unusable ICC"):
            actual = load_rgb(filename)
        np.testing.assert_array_equal(actual, self.image.astype(np.float32) / 255)

    def test_untagged_cmyk_uses_explicit_pillow_mapping(self):
        filename = self.root / "cmyk.jpg"
        cmyk = np.random.default_rng(2).integers(0, 256, (12, 15, 4), dtype=np.uint8)
        Image.fromarray(cmyk, mode="CMYK").save(filename)
        with Image.open(filename) as image:
            expected = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        with self.assertWarnsRegex(RuntimeWarning, "Untagged CMYK"):
            actual = load_rgb(filename)
        np.testing.assert_array_equal(actual, expected)

    def test_fid_insufficient_data_fails_before_network_loading(self):
        self.pair()
        with patch("eval.pipeline.FIDMetric", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "at least two"):
                evaluate(replace(self.config, metrics=("fid",)))

    def test_pixel_cli_subprocess_and_config_overrides(self):
        self.pair()
        config_file = self.root / "config.yaml"
        config_file.write_text("predictions: pred\nground_truth: gt\noutput: report\n"
                               "metrics: [psnr, ssim, delta_e00]\nallow_subset: true\n"
                               "resize_backend: pillow\n")
        config = parse_config(["--config", str(config_file), "--no-allow-subset", "--metrics", "psnr",
                               "--resize-backend", "opencv"])
        self.assertFalse(config.allow_subset)
        self.assertEqual(config.metrics, ("psnr",))
        self.assertEqual(config.output, self.root / "report")
        self.assertEqual(config.resize_backend, "opencv")
        result = subprocess.run([sys.executable, "-m", "eval", "--config", str(config_file)],
                                capture_output=True, text=True, check=True)
        self.assertIn("Evaluated 1 images", result.stdout)
        self.assertTrue((self.root / "report/report.json").is_file())

    def test_full_cli_with_mock_network_constructors(self):
        self.stochastic_fixture()
        # Exercise actual CLI wiring, batching, FID math, serializers and summary.
        with patch("lpips.LPIPS", side_effect=AssertionError("no real network loading")), \
                patch("pytorch_fid.inception.InceptionV3", side_effect=AssertionError("no real network loading")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            def small_fid(*args, **kwargs):
                return FIDMetric(network=MockFeatures(), dims=3, **kwargs)
            def mock_lpips(*args, **kwargs):
                return LPIPSMetric(*args, network=MockDistance(), **kwargs)
            with patch("eval.pipeline.FIDMetric", side_effect=small_fid), \
                    patch("eval.pipeline.LPIPSMetric", side_effect=mock_lpips):
                main(["--predictions", str(self.pred), "--ground-truth", str(self.gt),
                      "--output", str(self.root / "cli"), "--layout", "sample-dirs"])
            self.assertIn("fid=", output.getvalue())
            self.assertIn("best_of_k_lpips=0", output.getvalue())
        report = json.loads((self.root / "cli/report.json").read_text())
        self.assertEqual(len(report["per_sample"]), 8)
        self.assertTrue(report["protocol"]["injected_neural_backends"])

    def test_empty_unknown_config_and_unsafe_output(self):
        self.gt.mkdir()
        self.pred.mkdir()
        with self.assertRaisesRegex(ValueError, "both contain"):
            discover(self.config)
        with self.assertRaisesRegex(ValueError, "outside"):
            replace(self.config, output=self.pred / "out")
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(self.config, batch_size=0)
        with self.assertRaisesRegex(ValueError, "workers"):
            replace(self.config, workers=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(self.config, resize=(0, 64))
        with self.assertRaisesRegex(ValueError, "resize_backend"):
            replace(self.config, resize_backend="unknown")
        filename = self.root / "bad.yaml"
        filename.write_text("typo: true\n")
        with self.assertRaisesRegex(ValueError, "unknown config"):
            parse_config(["--config", str(filename)])


if __name__ == "__main__":
    unittest.main()
