from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.linalg import sqrtm
from skimage.color import deltaE_ciede2000

from eval.metrics.color import colorfulness, delta_colorfulness, delta_e00, to_lab
from eval.metrics.fid import FIDMetric, FeatureStatistics, frechet_distance
from eval.metrics.lpips import LPIPSMetric
from eval.metrics.pixel import psnr, ssim
from eval.tests.mocks import MockDistance, MockFeatures


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.black = np.zeros((64, 64, 3), dtype=np.float32)
        self.white = np.ones_like(self.black)

    def test_identity_and_known_psnr(self):
        self.assertTrue(math.isinf(psnr(self.black, self.black)))
        self.assertAlmostEqual(ssim(self.white, self.white), 1.0)
        self.assertEqual(delta_e00(self.white, self.white), 0.0)
        self.assertAlmostEqual(psnr(self.black, self.white * 0.1), 20.0, places=5)
        self.assertEqual(psnr(self.black, self.white), 0.0)
        self.assertLess(ssim(self.black, self.white), 0.001)

    def test_rgb_lab_physical_reference(self):
        red = self.black.copy()
        red[..., 0] = 1
        np.testing.assert_allclose(to_lab(red)[0, 0], [53.2406, 80.0923, 67.2028], atol=0.002)
        np.testing.assert_allclose(to_lab(self.white)[0, 0], [100, 0, 0], atol=0.006)
        self.assertAlmostEqual(delta_e00(self.black, self.white), 100, places=4)
        self.assertAlmostEqual(delta_e00(red, self.white), delta_e00(self.white, red), places=10)

    def test_sharma_ciede2000_reference_pairs(self):
        # Sharma/Wu/Dalal supplementary test data, pairs 1-4 (physical LAB).
        first = np.array([[50, 2.6772, -79.7751], [50, 3.1571, -77.2803],
                          [50, 2.8361, -74.0200], [50, -1.3802, -84.2814]])
        second = np.tile([50, 0, -82.7485], (4, 1))
        np.testing.assert_allclose(deltaE_ciede2000(first, second),
                                   [2.0425, 2.8615, 3.4412, 1.0000], atol=5e-5)

    def test_colorfulness_scale_and_variants(self):
        self.assertEqual(colorfulness(self.black), 0)
        self.assertEqual(colorfulness(self.white * 0.6), 0)
        red = self.black.copy()
        red[..., 0] = 1
        self.assertAlmostEqual(colorfulness(red), 0.3 * np.hypot(255, 127.5))
        self.assertEqual(delta_colorfulness(red, red), 0)
        # Absolute opponents erase sign variation; signed opponents preserve it.
        checker = red.copy()
        checker[::2, :, 0], checker[::2, :, 1] = 0, 1
        self.assertAlmostEqual(colorfulness(checker), 0.3 * np.hypot(255, 127.5))
        self.assertAlmostEqual(colorfulness(checker, "signed"), 255 + 0.3 * 127.5)
        self.assertGreater(delta_colorfulness(red, self.black), 0)

    def test_reject_invalid_ranges_shapes_and_tiny_ssim(self):
        for invalid in (self.white * 2, self.white * -1, self.white * np.nan,
                        self.white.astype(np.uint8), self.white[:, :, 0]):
            with self.assertRaises(ValueError):
                colorfulness(invalid)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            psnr(self.white, self.white[:32])
        with self.assertRaisesRegex(ValueError, "11x11"):
            ssim(self.white[:8, :8], self.white[:8, :8])

    def test_lpips_batch_normalization_and_order(self):
        network = MockDistance()
        metric = LPIPSMetric(network=network, batch_size=2)
        values = metric.distances([(self.black, self.white), (self.white, self.white),
                                   (self.white * 0.5, self.black)])
        np.testing.assert_allclose(values, [4, 0, 1])
        np.testing.assert_allclose(network.last_first, 0)
        self.assertFalse(network.training)
        self.assertTrue(metric.protocol()["injected_network"])
        with self.assertRaisesRegex(ValueError, "31x31"):
            metric(self.white[:16, :16], self.black[:16, :16])

    def test_lpips_handles_variable_resolution(self):
        metric = LPIPSMetric(network=MockDistance(), batch_size=3)
        larger = np.ones((80, 72, 3), dtype=np.float32)
        values = metric.distances([(self.white, self.black), (larger, larger),
                                   (self.black, self.black)])
        self.assertEqual(values, [4, 0, 0])

    def test_neural_constructors_are_lazy(self):
        with patch("lpips.LPIPS", side_effect=AssertionError("weight loading forbidden")), \
                patch("pytorch_fid.inception.InceptionV3", side_effect=AssertionError("weight loading forbidden")):
            LPIPSMetric()
            FIDMetric()

    def test_invalid_images_fail_before_loading_metric_weights(self):
        with patch("lpips.LPIPS", side_effect=AssertionError("weight loading forbidden")), \
                patch("pytorch_fid.inception.InceptionV3", side_effect=AssertionError("weight loading forbidden")):
            with self.assertRaisesRegex(ValueError, "31x31"):
                LPIPSMetric()(self.white[:30], self.black[:30])
            with self.assertRaisesRegex(ValueError, "RGB values"):
                FIDMetric().statistics([self.white * 2])

    def test_production_adapters_request_official_weights_and_protocol(self):
        network = MockDistance()
        with patch("lpips.LPIPS", return_value=network) as constructor:
            metric = LPIPSMetric(net="vgg")
            self.assertIs(metric.network, network)
            constructor.assert_called_once_with(net="vgg", version="0.1", verbose=False)
        network = MockFeatures()
        with patch("pytorch_fid.inception.InceptionV3", return_value=network) as constructor:
            constructor.BLOCK_INDEX_BY_DIM = {2048: 3}
            metric = FIDMetric()
            self.assertIs(metric.network, network)
            constructor.assert_called_once_with([3], resize_input=True, normalize_input=True,
                                               requires_grad=False, use_fid_inception=True)

    def test_feature_statistics_match_numpy_across_batches(self):
        features = np.random.default_rng(42).normal(size=(19, 5)) + 1e6
        statistics = FeatureStatistics()
        for batch in np.array_split(features, [1, 7, 15]):
            statistics.update(batch)
        mean, covariance = statistics.finalize()
        self.assertEqual(statistics.count, 19)
        self.assertEqual(covariance.dtype, np.float64)
        np.testing.assert_allclose(mean, features.mean(axis=0), atol=1e-9)
        np.testing.assert_allclose(covariance, np.cov(features, rowvar=False), atol=1e-9)

    def test_frechet_known_mean_shift(self):
        features = np.random.default_rng(7).normal(size=(20, 3))
        real, generated = FeatureStatistics(), FeatureStatistics()
        real.update(features)
        generated.update(features + [1, 2, 3])
        self.assertAlmostEqual(frechet_distance(real, real), 0, places=6)
        self.assertAlmostEqual(frechet_distance(real, generated), 14, places=6)

    def test_fid_matches_pytorch_fid_reference_formula(self):
        from pytorch_fid.fid_score import calculate_frechet_distance
        real, generated = FeatureStatistics(), FeatureStatistics()
        rng = np.random.default_rng(10)
        real.update(rng.normal(size=(20, 5)))
        generated.update(rng.normal(size=(30, 5)) * 2 + 1)
        # Emulate the old SciPy return signature only for the upstream call.
        def legacy_sqrtm(matrix, disp=True):
            root = sqrtm(matrix)
            return root if disp else (root, 0)
        with patch("pytorch_fid.fid_score.linalg", SimpleNamespace(sqrtm=legacy_sqrtm)):
            expected = calculate_frechet_distance(*real.finalize(), *generated.finalize())
        self.assertAlmostEqual(frechet_distance(real, generated), expected, places=12)

    def test_fid_preprocessing_and_final_partial_batch(self):
        network = MockFeatures()
        metric = FIDMetric(network=network, dims=3, batch_size=2)
        statistics = metric.statistics([self.black, self.white, self.white * 0.5])
        mean, covariance = statistics.finalize()
        self.assertEqual(statistics.count, 3)
        np.testing.assert_allclose(mean, 0.5)
        np.testing.assert_allclose(covariance, 0.25)
        np.testing.assert_allclose(network.last_inputs, 0.5)
        self.assertTrue(metric.protocol()["injected_network"])

    def test_fid_feature_block_statistics_match_full_numpy_covariance(self):
        values = np.random.default_rng(3).random((601, 3), dtype=np.float32)
        metric = FIDMetric(network=MockFeatures(), dims=3, batch_size=7)
        statistics = metric.statistics(value[None, None, :] for value in values)
        mean, covariance = statistics.finalize()
        self.assertEqual(statistics.count, len(values))
        np.testing.assert_allclose(mean, values.astype(np.float64).mean(0), atol=1e-15)
        np.testing.assert_allclose(covariance, np.cov(values.astype(np.float64), rowvar=False), atol=1e-15)

    def test_statistics_fail_on_insufficient_or_invalid_features(self):
        statistics = FeatureStatistics()
        with self.assertRaisesRegex(ValueError, "at least two"):
            statistics.finalize()
        for invalid in (np.zeros((0, 3)), np.zeros(3), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                statistics.update(invalid)
        statistics.update(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            statistics.finalize()
        with self.assertRaisesRegex(ValueError, "dimension"):
            statistics.update(np.zeros((2, 4)))


if __name__ == "__main__":
    unittest.main()
