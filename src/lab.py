"""CIELAB representation utilities used by both data and visualization code.

The legacy palette data loader uses OpenCV's 8-bit LAB representation and
albumentations Normalize(max_pixel_value=127.5).  The functions below make
that convention explicit and invertible (up to 8-bit quantization):

* L_norm = L_opencv / 127.5 - 1
* ab_norm = ab_opencv / 127.5 - 1
* L_physical = (L_norm + 1) * 50
* ab_physical = (ab_norm + 1) * 127.5 - 128

Generated values are not bounded by the model.  Clipping is performed only
when converting an arbitrary generated LAB tensor to 8-bit sRGB for display.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


_LAB_SCALE = 127.5
_AB_OFFSET = 128.0


def normalize_L(L: torch.Tensor) -> torch.Tensor:
    """Convert physical L* in [0, 100] to the dataset's normalized scale."""

    return L / 50.0 - 1.0


def denormalize_L(L: torch.Tensor) -> torch.Tensor:
    """Convert normalized L* back to physical CIELAB L*."""

    return (L + 1.0) * 50.0


def normalize_ab(ab: torch.Tensor) -> torch.Tensor:
    """Convert physical a*/b* values to the legacy normalized scale."""

    return (ab + _AB_OFFSET) / _LAB_SCALE - 1.0


def denormalize_ab(ab: torch.Tensor) -> torch.Tensor:
    """Convert normalized a*/b* values to physical CIELAB coordinates."""

    return (ab + 1.0) * _LAB_SCALE - _AB_OFFSET


def _as_uint8_lab(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Build OpenCV LAB bytes, clipping only for the visualization boundary."""

    lab = torch.cat([L, ab], dim=1)
    lab = ((lab.float() + 1.0) * _LAB_SCALE).round().clamp(0.0, 255.0)
    # ``torch==2.1`` in the legacy environment was built against NumPy 1.x.
    # Going through a Python list keeps this utility usable with NumPy 2.x too.
    lab = np.asarray(lab.detach().cpu().tolist(), dtype=np.uint8)
    return np.moveaxis(lab, 1, -1)


def rgb_to_lab(image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert an RGB uint8 image to normalized ``(L, ab)`` tensors."""

    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_to_lab expects an RGB uint8 array shaped [H, W, 3]")
    encoded = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = torch.tensor(encoded[..., :1].tolist()).permute(2, 0, 1).float()
    ab = torch.tensor(encoded[..., 1:].tolist()).permute(2, 0, 1).float()
    return L / _LAB_SCALE - 1.0, ab / _LAB_SCALE - 1.0


def lab_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Convert normalized LAB tensors to an RGB uint8 image for visualization.

    The conversion clamps the OpenCV encoding because sRGB has a finite gamut.
    Consequently this function is not a luminance-preserving training
    operation for out-of-gamut generated colors.
    """

    if L.ndim != 4 or ab.ndim != 4 or L.shape[1] != 1 or ab.shape[1] != 2:
        raise ValueError("lab_to_rgb expects L [B,1,H,W] and ab [B,2,H,W]")
    encoded = _as_uint8_lab(L, ab)
    rgb = [cv2.cvtColor(item, cv2.COLOR_LAB2RGB) for item in encoded]
    return np.stack(rgb, axis=0)


def compose_lab(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Compose LAB while preserving the input luminance tensor exactly."""

    if L.ndim != 4 or ab.ndim != 4 or L.shape[1] != 1 or ab.shape[1] != 2:
        raise ValueError("compose_lab expects L [B,1,H,W] and ab [B,2,H,W]")
    return torch.cat([L, ab], dim=1)
