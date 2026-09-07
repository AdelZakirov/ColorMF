import unittest

import numpy as np
import torch

from src.lab import (
    compose_lab,
    denormalize_L,
    denormalize_ab,
    lab_to_rgb,
    normalize_L,
    normalize_ab,
    rgb_to_lab,
)


class LabTests(unittest.TestCase):
    def test_physical_normalization_roundtrip(self):
        L = torch.tensor([[[[0.0, 50.0, 100.0]]]])
        ab = torch.tensor([[[[-128.0, 0.0, 127.0]], [[-128.0, 0.0, 127.0]]]])
        self.assertTrue(torch.allclose(denormalize_L(normalize_L(L)), L))
        self.assertTrue(torch.allclose(denormalize_ab(normalize_ab(ab)), ab))

    def test_rgb_lab_roundtrip_for_in_gamut_image(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = 220
        image[..., 1] = 120
        image[..., 2] = 60
        L, ab = rgb_to_lab(image)
        restored = lab_to_rgb(L.unsqueeze(0), ab.unsqueeze(0))[0]
        self.assertLessEqual(np.abs(restored.astype(int) - image.astype(int)).max(), 2)

    def test_compose_preserves_luminance(self):
        L = torch.randn(2, 1, 4, 4)
        ab = torch.randn(2, 2, 4, 4)
        self.assertTrue(torch.equal(compose_lab(L, ab)[:, :1], L))


if __name__ == "__main__":
    unittest.main()

