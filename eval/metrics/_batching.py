"""Shared tensor conversion and ordered batches for variable-size images."""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import numpy as np
import torch

Item = TypeVar("Item")


def rgb_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def shape_batches(items: Iterable[Item], batch_size: int,
                  shape: Callable[[Item], tuple]) -> Iterator[list[Item]]:
    """Keep order and the final partial batch; the shape callback also validates."""
    batch = []
    batch_shape = None
    for item in items:
        item_shape = shape(item)
        if batch and (len(batch) == batch_size or item_shape != batch_shape):
            yield batch
            batch = []
        batch.append(item)
        batch_shape = item_shape
    if batch:
        yield batch
