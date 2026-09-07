import unittest

import torch

from src.model import PMFTiny


class SamplingTests(unittest.TestCase):
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
