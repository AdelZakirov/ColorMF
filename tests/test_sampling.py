import unittest

import torch

from src.model import PMFTiny


class SamplingTests(unittest.TestCase):
    @staticmethod
    def _permute_patches(value, permutation, patch_size):
        batch, channels, height, width = value.shape
        rows = height // patch_size
        columns = width // patch_size
        patches = value.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch, rows * columns, channels, patch_size, patch_size
        )
        patches = patches[:, permutation]
        return patches.reshape(
            batch, rows, columns, channels, patch_size, patch_size
        ).permute(0, 3, 1, 4, 2, 5).reshape_as(value)

    def test_joint_patch_permutation_changes_position_aware_output(self):
        model = PMFTiny(
            resolution=16, patch_size=4, hidden_size=32, depth=1, heads=4
        ).eval()
        z = torch.randn(1, 2, 16, 16)
        L = torch.randn(1, 1, 16, 16)
        permutation = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        with torch.no_grad():
            original, _ = model(z, L, torch.zeros(1), torch.ones(1))
            permuted_z = self._permute_patches(z, permutation, 4)
            permuted_L = self._permute_patches(L, permutation, 4)
            permuted, _ = model(
                permuted_z, permuted_L, torch.zeros(1), torch.ones(1)
            )
        restored = self._permute_patches(permuted, permutation, 4)
        self.assertFalse(torch.allclose(original, restored, atol=1e-6, rtol=1e-6))

    def test_one_step_and_reproducibility(self):
        model = PMFTiny(
            resolution=16, patch_size=4, hidden_size=32, depth=1, heads=4
        ).eval()
        L = torch.zeros(1, 1, 16, 16)
        first = model.sample(L, seed=42, image_ids=["image-a"])
        second = model.sample(L, seed=42, image_ids=["image-a"])
        other = model.sample(L, seed=43, image_ids=["image-a"])
        self.assertEqual(model.last_sample_nfe, 1)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))

    def test_seed_batch_matches_individual_sampling(self):
        model = PMFTiny(
            resolution=16, patch_size=4, hidden_size=32, depth=1, heads=4
        ).eval()
        L = torch.zeros(1, 1, 16, 16)
        batched = model.sample(L, seeds=[1, 2, 3, 4], image_ids=["image-a"])
        individual = torch.cat(
            [
                model.sample(L, seed=seed, image_ids=["image-a"])
                for seed in [1, 2, 3, 4]
            ],
            dim=0,
        )
        torch.testing.assert_close(batched, individual, rtol=1e-5, atol=1e-6)

    def test_sample_lab_keeps_luminance(self):
        model = PMFTiny(
            resolution=16, patch_size=4, hidden_size=32, depth=1, heads=4
        ).eval()
        L = torch.randn(1, 1, 16, 16)
        lab = model.sample_lab(L, seed=7, image_ids=["image-a"])
        self.assertTrue(torch.equal(lab[:, :1], L))


if __name__ == "__main__":
    unittest.main()
