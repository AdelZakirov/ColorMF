"""Full-reference RGB pixel metrics using scikit-image."""

import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ..data import validate_pair


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    validate_pair(prediction, target)
    if np.array_equal(prediction, target):
        return math.inf
    return float(peak_signal_noise_ratio(target, prediction, data_range=1.0))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Wang et al.: 11x11 Gaussian window, sigma=1.5, population covariance."""
    validate_pair(prediction, target)
    if min(prediction.shape[:2]) < 11:
        raise ValueError("SSIM requires images at least 11x11 for the fixed protocol")
    return float(structural_similarity(
        target, prediction, data_range=1.0, channel_axis=-1,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        win_size=11, K1=0.01, K2=0.03))
