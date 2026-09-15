import unittest

import torch

from src.model import PixelMeanFlowB


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
        model = PixelMeanFlowB(
            resolution=16, patch_size=4, hidden_size=32, depth=2, heads=4,
            aux_head_depth=1, pca_channels=8
        ).eval()
        torch.nn.init.normal_(model.u_final_layer.linear._flax_linear.weight, std=0.02)
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
        model = PixelMeanFlowB(
            resolution=16, patch_size=4, hidden_size=32, depth=2, heads=4,
            aux_head_depth=1, pca_channels=8
        ).eval()
        torch.nn.init.normal_(model.u_final_layer.linear._flax_linear.weight, std=0.02)
        L = torch.zeros(1, 1, 16, 16)
        first = model.sample(L, seed=42, image_ids=["image-a"])
        second = model.sample(L, seed=42, image_ids=["image-a"])
        other = model.sample(L, seed=43, image_ids=["image-a"])
        self.assertEqual(model.last_sample_nfe, 1)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))

    def test_noise_scale_is_applied_to_sampling_prior(self):
        model = PixelMeanFlowB(
            resolution=16, patch_size=4, hidden_size=32, depth=2, heads=4,
            aux_head_depth=1, pca_channels=8
        ).eval()
        model.forward = lambda z, L, r, t, return_velocity=False: (torch.zeros_like(z), None)
        L = torch.zeros(1, 1, 16, 16)
        sample = model.sample(L, seed=42, image_ids=["image-a"], noise_scale=0.25)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(model._stable_seed("image-a", 42))
        expected = 0.25 * torch.randn((1, 2, 16, 16), generator=generator)
        torch.testing.assert_close(sample, expected)

    def test_seed_batch_matches_individual_sampling(self):
        model = PixelMeanFlowB(
            resolution=16, patch_size=4, hidden_size=32, depth=2, heads=4,
            aux_head_depth=1, pca_channels=8
        ).eval()
        torch.nn.init.normal_(model.u_final_layer.linear._flax_linear.weight, std=0.02)
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
        model = PixelMeanFlowB(
            resolution=16, patch_size=4, hidden_size=32, depth=2, heads=4,
            aux_head_depth=1, pca_channels=8
        ).eval()
        L = torch.randn(1, 1, 16, 16)
        lab = model.sample_lab(L, seed=7, image_ids=["image-a"])
        self.assertTrue(torch.equal(lab[:, :1], L))


if __name__ == "__main__":
    unittest.main()
