"""FID Inception features with bounded-memory float64 dataset statistics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import sqrtm

from ..data import validate_rgb
from ._batching import rgb_tensor, shape_batches


FID_EXTRACTOR = {
    "implementation": "pytorch-fid", "dims": 2048,
    "weights": "pt_inception-2015-12-05-6726825d.pth",
    "internal_resize": "bilinear 299x299, align_corners=False",
    "internal_normalization": "[0,1] to [-1,1]",
}


def create_fid_network(device="cpu") -> torch.nn.Module:
    """Shared evaluator/training pool3 factory with the existing FID weights."""
    from pytorch_fid.inception import InceptionV3
    return InceptionV3(
        [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True,
        normalize_input=True, requires_grad=False, use_fid_inception=True,
    ).to(device).eval().requires_grad_(False)


def extract_fid_features(network: torch.nn.Module, inputs: torch.Tensor,
                         dims: int = 2048) -> torch.Tensor:
    """N x D features; caller owns precision and grad/inference context."""
    features = network(inputs)[0]
    if features.ndim != 4 or features.shape[:2] != (len(inputs), dims):
        raise ValueError("FID backend must return N x D x H x W features")
    return torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)


@dataclass
class FeatureStatistics:
    """Merge centered batch moments; covariance uses the unbiased N-1 divisor."""
    count: int = 0
    mean: np.ndarray | None = None
    scatter: np.ndarray | None = None

    def update(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float64)
        if (features.ndim != 2 or min(features.shape) == 0 or
                not np.isfinite(features).all()):
            raise ValueError("features must be a nonempty finite N x D array")
        mean = features.mean(axis=0)
        centered = features - mean
        scatter = centered.T @ centered
        if self.count == 0:
            self.count, self.mean, self.scatter = len(features), mean, scatter
            return
        if self.mean.shape != mean.shape:
            raise ValueError("inconsistent feature dimension")
        total = self.count + len(features)
        delta = mean - self.mean
        self.scatter += scatter + np.outer(delta, delta) * self.count * len(features) / total
        self.mean += delta * len(features) / total
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("FID needs at least two images in each distribution")
        return self.mean, self.scatter / (self.count - 1)


def frechet_distance(real: FeatureStatistics, generated: FeatureStatistics) -> float:
    real_mean, real_covariance = real.finalize()
    pred_mean, pred_covariance = generated.finalize()
    if real_mean.shape != pred_mean.shape:
        raise ValueError("FID distributions must have the same feature dimension")
    # Standard FID formula with pytorch-fid's 1e-6 stabilization policy.
    # Call SciPy directly: pytorch-fid 0.3.0 still passes the removed disp=False
    # argument, which breaks on SciPy >=1.18. No global library monkey-patching.
    covariance_root = sqrtm(real_covariance @ pred_covariance)
    if not np.isfinite(covariance_root).all():
        offset = np.eye(len(real_mean)) * 1e-6
        covariance_root = sqrtm((real_covariance + offset) @ (pred_covariance + offset))
    if np.iscomplexobj(covariance_root):
        if not np.allclose(np.diag(covariance_root).imag, 0, atol=1e-3):
            raise ValueError("FID covariance square root has a significant imaginary component")
        covariance_root = covariance_root.real
    difference = real_mean - pred_mean
    score = float(difference @ difference + np.trace(real_covariance) +
                  np.trace(pred_covariance) - 2 * np.trace(covariance_root))
    if not np.isfinite(score):
        raise ValueError("FID calculation returned a nonfinite value")
    tolerance = 1e-6 * max(1.0, float(np.trace(real_covariance) + np.trace(pred_covariance)))
    if score < -tolerance:
        raise ValueError(f"FID calculation returned a negative distance: {score}")
    return max(0.0, score)


class FIDMetric:
    """Standard pytorch-fid pool3/2048 (not torchvision classification weights)."""
    def __init__(self, device: str = "cpu", batch_size: int = 32,
                 network: torch.nn.Module | None = None, dims: int = 2048):
        if batch_size <= 0 or dims <= 0:
            raise ValueError("batch_size and dims must be positive")
        if network is None and dims != 2048:
            raise ValueError("production FID uses 2048-dimensional pool3 features")
        self.device, self.batch_size, self.dims = device, batch_size, dims
        self._network = network
        self.injected = network is not None

    @property
    def network(self) -> torch.nn.Module:
        if self._network is None:
            self._network = create_fid_network(self.device)
        self._network.to(self.device).eval().requires_grad_(False)
        return self._network

    def statistics(self, images: Iterable[np.ndarray]) -> FeatureStatistics:
        def image_shape(image: np.ndarray) -> tuple:
            validate_rgb(image)
            return image.shape

        statistics = FeatureStatistics()
        pending, pending_count = [], 0
        network = None
        for batch in shape_batches(images, self.batch_size, image_shape):
            inputs = torch.stack([rgb_tensor(image) for image in batch]).to(self.device)
            if network is None:
                network = self.network
            with torch.inference_mode():
                features = extract_fid_features(network, inputs, self.dims)
            pending.append(features.flatten(1).cpu().numpy())
            pending_count += len(inputs)
            # Covariance updates over feature blocks avoid a 2048x2048 outer
            # product per image. Image features and their ordering are unchanged.
            if pending_count >= 256:
                statistics.update(np.concatenate(pending))
                pending, pending_count = [], 0
        if pending:
            statistics.update(np.concatenate(pending))
        return statistics

    def protocol(self) -> dict:
        return {"extractor": None if self.injected else FID_EXTRACTOR,
                "implementation": "pytorch-fid", "dims": self.dims,
                "weights": "injected network" if self.injected else
                           "FID InceptionV3 (TensorFlow-compatible)",
                "internal_resize": "backend-defined (injected)" if self.injected else
                                   "bilinear 299x299, align_corners=False",
                "internal_normalization": "backend-defined (injected)" if self.injected else
                                          "[0,1] to [-1,1]",
                "covariance_dtype": "float64", "covariance_ddof": 1,
                "distance": "SciPy sqrtm, pytorch-fid-compatible formula/stabilization",
                "injected_network": self.injected}
