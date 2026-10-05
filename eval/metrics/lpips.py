"""Lazy official LPIPS adapter, with injectable networks for offline tests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch

from ..data import validate_pair
from ._batching import rgb_tensor, shape_batches


class LPIPSMetric:
    def __init__(self, net: str = "alex", device: str = "cpu", batch_size: int = 32,
                 network: torch.nn.Module | None = None):
        if net not in ("alex", "vgg", "squeeze") or batch_size <= 0:
            raise ValueError("invalid LPIPS network or batch size")
        self.net, self.device, self.batch_size = net, device, batch_size
        self._network = network
        self.injected = network is not None

    @property
    def network(self) -> torch.nn.Module:
        if self._network is None:
            import lpips
            self._network = lpips.LPIPS(net=self.net, version="0.1", verbose=False)
        self._network.to(self.device).eval().requires_grad_(False)
        return self._network

    def distances(self, pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        return list(self.iter_distances(pairs))

    def iter_distances(self, pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> Iterator[float]:
        """Batch equal-shape pairs; keep variable-resolution input order."""
        def pair_shape(pair: tuple[np.ndarray, np.ndarray]) -> tuple:
            first, second = pair
            validate_pair(first, second)
            # AlexNet LPIPS stops at relu5, before the classifier's final pool.
            # Its two included max-pools need a minimum side of 31, not 64.
            minimum = {"alex": 31, "vgg": 32, "squeeze": 32}[self.net]
            if min(first.shape[:2]) < minimum:
                raise ValueError(f"LPIPS {self.net} requires at least {minimum}x{minimum} "
                                 "under this protocol; set resize explicitly")
            return first.shape

        network = None
        for batch in shape_batches(pairs, self.batch_size, pair_shape):
            first = torch.stack([rgb_tensor(pair[0]) for pair in batch]).to(self.device)
            second = torch.stack([rgb_tensor(pair[1]) for pair in batch]).to(self.device)
            if network is None:
                network = self.network
            with torch.inference_mode():
                scores = network(first * 2 - 1, second * 2 - 1).reshape(-1)
            if len(scores) != len(batch) or not torch.isfinite(scores).all():
                raise ValueError("LPIPS backend must return one finite score per pair")
            yield from scores.cpu().tolist()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.distances([(prediction, target)])[0]

    def protocol(self) -> dict:
        return {"implementation": "injected network" if self.injected else "lpips.LPIPS",
                "version": None if self.injected else "0.1", "net": self.net,
                "minimum_side": {"alex": 31, "vgg": 32, "squeeze": 32}[self.net],
                "input_range": [-1, 1], "injected_network": self.injected}
