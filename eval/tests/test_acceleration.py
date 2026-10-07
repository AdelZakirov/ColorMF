"""Cache provenance, ordered neural prefetch, and shared GT LAB regression tests."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from eval.__main__ import parse_config
from eval.config import EvalConfig
from eval.data import discover, load_rgb
from eval.fid_cache import cache_metadata, load_statistics
from eval.metrics.color import delta_e00, to_lab
from eval.metrics.fid import FeatureStatistics, FIDMetric
from eval.metrics.lpips import LPIPSMetric
from eval.pipeline import evaluate, _evaluate_image
from eval.tests.mocks import MockDistance, MockFeatures


class CacheTestMetric:
    """Test the production cache contract without pretrained weights/downloads."""
    dims = 3
    injected = False

    def __init__(self):
        self.calls = 0

    def protocol(self):
        return {"dims": 3, "weights": "test-only deterministic features"}

    def statistics(self, images):
        self.calls += 1
        result = FeatureStatistics()
        result.update(np.stack([image.mean(axis=(0, 1)) for image in images]))
        return result


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gt, self.pred = self.root / "gt", self.root / "pred"
        self.gt.mkdir()
        self.pred.mkdir()
        for index in range(6):
            rgb = np.random.default_rng(index).integers(0, 256, (64 + index % 2, 64, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(self.gt / f"{index}.png")
            Image.fromarray(255 - rgb).save(self.pred / f"{index}.png")
        self.config = EvalConfig(self.pred, self.gt, self.root / "out", metrics=("fid",),
                                 fid_real_stats=self.root / "shared/gt.npz")
        self.metric = CacheTestMetric()

    def test_cache_creation_and_reuse_skip_gt_extraction(self):
        computed = evaluate(self.config, fid_metric=self.metric)
        self.assertEqual(self.metric.calls, 2)
        phases = []
        cached = evaluate(replace(self.config, output=self.root / "second"), fid_metric=self.metric,
                          phase_progress=lambda phase, done, total: phases.append(phase))
        self.assertEqual(self.metric.calls, 3)
        self.assertEqual(computed["fid"]["value"], cached["fid"]["value"])
        self.assertEqual(cached["fid"]["real_statistics_cache"]["status"], "loaded")
        self.assertIn("fid_real_cached", phases)
        self.assertNotIn("fid_real", phases)

    def test_cache_rejects_preprocessing_subset_file_and_extractor_changes(self):
        evaluate(self.config, fid_metric=self.metric)
        original = self.config.fid_real_stats.read_bytes()
        for config in (replace(self.config, resize=(32, 32)),
                       replace(self.config, resize=(32, 32), resize_backend="opencv")):
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluate(config, fid_metric=self.metric)
        records, _ = discover(self.config)
        for subset in (records[:-1], list(reversed(records))):
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_statistics(self.config.fid_real_stats, cache_metadata(subset, self.config, self.metric))
        metadata = cache_metadata(records, self.config, self.metric)
        metadata["fid"]["weights"] = "different weights"
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_statistics(self.config.fid_real_stats, metadata)
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(self.gt / "0.png")
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate(self.config, fid_metric=self.metric)
        self.assertEqual(self.config.fid_real_stats.read_bytes(), original)

    def test_cache_rejects_malformed_moments_and_unprovenanced_legacy(self):
        records, _ = discover(self.config)
        metadata = cache_metadata(records, self.config, self.metric)
        path = self.root / "bad.npz"
        np.savez(path, count=6, mean=np.zeros(3), scatter=np.full((3, 3), np.nan),
                 metadata=json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "finite FP64"):
            load_statistics(path, metadata)
        np.savez(path, count=6, mean=np.zeros(3), scatter=np.eye(3))
        with self.assertRaisesRegex(ValueError, "no metadata"):
            load_statistics(path, metadata)

    def test_legacy_companion_report_validates_gt_not_predictions(self):
        config = replace(self.config, fid_real_stats=None)
        report = evaluate(config, fid_metric=self.metric)
        records, _ = discover(config)
        metadata = cache_metadata(records, config, self.metric)
        stats = self.metric.statistics(load_rgb(record.ground_truth) for record in records)
        path = self.root / "fid_real_statistics.npz"
        np.savez(path, count=stats.count, mean=stats.mean, scatter=stats.scatter)
        (self.root / "report.json").write_text(json.dumps(report))
        loaded, status = load_statistics(path, metadata)
        self.assertEqual(status, "legacy_loaded")
        np.testing.assert_array_equal(loaded.mean, stats.mean)
        report["per_sample"][0]["ground_truth"] = str(self.pred / "0.png")
        (self.root / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_statistics(path, metadata)

    def test_cli_default_workers_and_relative_cache_path(self):
        filename = self.root / "config.yaml"
        filename.write_text("predictions: pred\nground_truth: gt\noutput: out\nfid_real_stats: cache/gt.npz\n")
        config = parse_config(["--config", str(filename)])
        self.assertEqual(config.workers, 4)
        self.assertEqual(config.fid_real_stats, self.root / "cache/gt.npz")
        with self.assertRaisesRegex(ValueError, "injected network"):
            evaluate(config, fid_metric=FIDMetric(network=MockFeatures(), dims=3))

    def test_gt_lab_converted_once_for_multiple_samples(self):
        config = replace(self.config, metrics=("delta_e00",), fid_real_stats=None)
        records, _ = discover(config)
        record = replace(records[0], samples=records[0].samples * 3)
        with patch("eval.pipeline.to_lab", wraps=to_lab) as conversion:
            _, rows = _evaluate_image(record, config)
        self.assertEqual(conversion.call_count, 4)  # One GT + three predictions.
        reference = delta_e00(load_rgb(record.samples[0].path), load_rgb(record.ground_truth))
        self.assertTrue(all(row["metrics"]["delta_e00"] == reference for row in rows))

    def test_neural_decode_uses_workers_and_preserves_order_and_values(self):
        config = replace(self.config, fid_real_stats=None, metrics=("lpips", "fid"))
        def run(workers):
            seen = []
            def load(*args, **kwargs):
                seen.append(threading.current_thread().name)
                time.sleep(.002)
                return load_rgb(*args, **kwargs)
            with patch("eval.pipeline.load_rgb", side_effect=load):
                report = evaluate(replace(config, workers=workers),
                                  fid_metric=FIDMetric(network=MockFeatures(), dims=3),
                                  lpips_metric=LPIPSMetric(network=MockDistance()))
            return report, seen
        serial, _ = run(1)
        parallel, threads = run(4)
        for key in ("fid", "aggregate", "per_image", "per_sample"):
            self.assertEqual(serial[key], parallel[key])
        # CPU, LPIPS decode and both FID distributions each use their own pool.
        self.assertEqual(len({name.split("_")[0] for name in threads}), 4)
        self.assertNotIn("MainThread", threads)

    def test_prefetch_decode_errors_propagate(self):
        config = replace(self.config, fid_real_stats=None, metrics=("lpips",))
        calls = 0
        def load(*args, **kwargs):
            nonlocal calls
            calls += 1
            # CPU records decode 12 files, then LPIPS decode must fail cleanly.
            if calls > 12:
                raise ValueError("decode failed in neural prefetch")
            return load_rgb(*args, **kwargs)
        with patch("eval.pipeline.load_rgb", side_effect=load):
            with self.assertRaisesRegex(ValueError, "neural prefetch"):
                evaluate(config, lpips_metric=LPIPSMetric(network=MockDistance()))
