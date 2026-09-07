import numpy as np
import pytest
import torch

kornia = pytest.importorskip("kornia")

from src.lab import lab_to_rgb, rgb_to_lab
from src.perceptual import PerceptualLosses, normalized_lab_to_rgb


class IdentityFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, pixel_values):
        return self.scale * pixel_values.mean((2, 3))


def test_rgb_path_is_differentiable_preserves_l_and_does_not_clamp_ab():
    L = torch.zeros(1, 1, 8, 8)
    ab = torch.full((1, 2, 8, 8), 2.0, requires_grad=True)
    original_L = L.clone()
    rgb = normalized_lab_to_rgb(L, ab)
    rgb.sum().backward()
    assert ab.grad is not None and torch.count_nonzero(ab.grad) > 0
    assert torch.equal(L, original_L)
    assert torch.equal(ab.detach(), torch.full_like(ab, 2.0))
    assert bool(((rgb < 0) | (rgb > 1)).any())  # proves clip=False at the RGB boundary


def test_kornia_agrees_with_opencv_on_valid_colors():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 180, 120, 80
    L, ab = rgb_to_lab(image)
    kornia_rgb = normalized_lab_to_rgb(L[None], ab[None])[0].permute(1, 2, 0)
    opencv_rgb = torch.from_numpy(lab_to_rgb(L[None], ab[None])[0].copy()).float() / 255
    torch.testing.assert_close(kornia_rgb, opencv_rgb, atol=0.025, rtol=0)


def test_loss_networks_are_frozen():
    lpips, convnext = IdentityFeatures(), IdentityFeatures()
    PerceptualLosses(use_lpips=True, use_convnext=True,
                     lpips_network=lpips, convnext_network=convnext)
    assert all(not parameter.requires_grad
               for network in (lpips, convnext) for parameter in network.parameters())
