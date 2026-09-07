"""Frozen perceptual losses on differentiable RGB reconstructions."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .lab import denormalize_L, denormalize_ab


def normalized_lab_to_rgb(L: Tensor, ab: Tensor) -> Tensor:
    """Convert ColorMF normalized LAB to sRGB without clipping model state.

    Kornia 0.8 exposes ``clip=False``. Requiring that interface is deliberate:
    silently using its default hard RGB clipping would discard out-of-gamut
    gradients before they reach predicted chroma.
    """
    try:
        from kornia.color import lab_to_rgb
    except ImportError as error:
        raise RuntimeError("Kornia is required for perceptual LAB-to-RGB training") from error
    lab_physical = torch.cat([denormalize_L(L), denormalize_ab(ab)], dim=1)
    try:
        return lab_to_rgb(lab_physical, clip=False)
    except TypeError as error:
        raise RuntimeError("installed Kornia lab_to_rgb must support clip=False") from error


def paired_random_resized_crop(x1: Tensor, x2: Tensor, out_size: int = 224,
                               scale: tuple[float, float] = (0.08, 1.0),
                               ratio: tuple[float, float] = (3 / 4, 4 / 3),
                               generator: Optional[torch.Generator] = None) -> tuple[Tensor, Tensor]:
    """Per-example paired crop matching official pMF auxiliary-loss geometry."""
    if x1.shape != x2.shape:
        raise ValueError("paired crops require equal shapes")
    _, _, height, width = x1.shape
    first, second = [], []
    for index in range(x1.shape[0]):
        area = height * width
        crop_h = crop_w = 0
        for _ in range(10):
            target = area * float(torch.empty((), device=x1.device).uniform_(
                scale[0], scale[1], generator=generator))
            aspect = math.exp(float(torch.empty((), device=x1.device).uniform_(
                math.log(ratio[0]), math.log(ratio[1]), generator=generator)))
            crop_w = min(width, max(1, round(math.sqrt(target * aspect))))
            crop_h = min(height, max(1, round(math.sqrt(target / aspect))))
            if crop_h <= height and crop_w <= width:
                break
        else:
            crop_h = crop_w = min(height, width)
        top = int(torch.randint(0, height - crop_h + 1, (), device=x1.device,
                                generator=generator))
        left = int(torch.randint(0, width - crop_w + 1, (), device=x1.device,
                                 generator=generator))
        slices = (slice(index, index + 1), slice(None),
                  slice(top, top + crop_h), slice(left, left + crop_w))
        first.append(F.interpolate(x1[slices], size=(out_size, out_size),
                                   mode="bicubic", align_corners=False, antialias=True))
        second.append(F.interpolate(x2[slices], size=(out_size, out_size),
                                    mode="bicubic", align_corners=False, antialias=True))
    return torch.cat(first), torch.cat(second)


class PerceptualLosses:
    """External frozen LPIPS and ConvNeXt-V2 feature loss networks.

    These objects intentionally live outside the generative model state and
    are therefore neither inference dependencies nor checkpoint payload.
    """
    def __init__(self, *, use_lpips: bool, use_convnext: bool,
                 lpips_network: Optional[nn.Module] = None,
                 convnext_network: Optional[nn.Module] = None):
        if use_lpips and lpips_network is None:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError("lpips is required when LPIPS loss is enabled") from error
            # lpips_j used by official pMF implements the VGG16 backend.
            lpips_network = lpips.LPIPS(net="vgg")
        if use_convnext and convnext_network is None:
            try:
                from transformers import ConvNextV2Model
            except ImportError as error:
                raise RuntimeError("transformers is required for ConvNeXt perceptual loss") from error
            convnext_network = ConvNextV2Model.from_pretrained(
                "facebook/convnextv2-base-22k-224")
        self.lpips = lpips_network if use_lpips else None
        self.convnext = convnext_network if use_convnext else None
        for network in (self.lpips, self.convnext):
            if network is not None:
                network.eval()
                for parameter in network.parameters():
                    parameter.requires_grad_(False)

    def to(self, device: torch.device) -> "PerceptualLosses":
        for network in (self.lpips, self.convnext):
            if network is not None:
                network.to(device)
        return self

    def __call__(self, predicted_ab: Tensor, target_ab: Tensor, L: Tensor, *,
                 generator: Optional[torch.Generator] = None) -> tuple[Tensor, Tensor]:
        predicted_rgb = normalized_lab_to_rgb(L, predicted_ab)
        target_rgb = normalized_lab_to_rgb(L, target_ab)
        predicted_rgb, target_rgb = paired_random_resized_crop(
            predicted_rgb, target_rgb, out_size=224, generator=generator)
        batch = predicted_rgb.shape[0]
        zeros = torch.zeros(batch, device=predicted_rgb.device, dtype=torch.float32)
        if self.lpips is None:
            lpips_values = zeros
        else:
            # The reference LPIPS interface expects RGB in [-1, 1]. clip=False
            # is retained so gamut handling remains at the loss-network boundary.
            lpips_values = self.lpips(predicted_rgb * 2 - 1,
                                       target_rgb * 2 - 1).reshape(batch).float()
        if self.convnext is None:
            convnext_values = zeros
        else:
            # Faithful pMF behavior: its normalized training images are sent
            # directly to the converted ConvNeXt feature extractor.
            predicted_input = predicted_rgb * 2 - 1
            target_input = target_rgb * 2 - 1
            predicted_output = self.convnext(pixel_values=predicted_input)
            target_output = self.convnext(pixel_values=target_input)
            predicted_features = getattr(predicted_output, "pooler_output", predicted_output)
            target_features = getattr(target_output, "pooler_output", target_output)
            convnext_values = (predicted_features - target_features).float().pow(2).flatten(1).sum(1)
        return lpips_values, convnext_values
