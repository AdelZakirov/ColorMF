import numpy as np
import pytest

from experiments.colorization_losses.evaluate import average_ranks, pearson, spearman


def test_average_ranks_handles_ties():
    np.testing.assert_allclose(average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])


def test_correlations_are_one_for_monotonic_sequence():
    values = [1.0, 2.0, 3.0, 4.0]
    assert pearson(values, values) == pytest.approx(1.0)
    assert spearman(values, values) == pytest.approx(1.0)


def test_constant_correlation_is_undefined():
    assert np.isnan(pearson([1.0, 1.0], [1.0, 2.0]))


def test_spatial_features_detect_permutation_with_identical_pool():
    import torch
    from experiments.colorization_losses.losses import spatial_distance
    x = torch.tensor([[[[1., 0.], [0., 0.]]]])
    y = x.flip(-1)
    assert x.mean() == y.mean()
    assert spatial_distance((x,), (y,)).item() > 0


def test_gamut_projection_preserves_luminance():
    import torch
    from kornia.color import lab_to_rgb
    from experiments.colorization_losses.losses import physical_lab
    from experiments.colorization_losses.evaluate import chroma_endpoint
    gt = torch.full((1,3,8,8), .8)
    corrupt = torch.zeros_like(gt); corrupt[:,0] = 1
    lab = physical_lab(gt)
    ab = chroma_endpoint(gt, corrupt)
    rgb = lab_to_rgb(torch.cat((lab[:,:1],ab),1),clip=False)
    assert rgb.min() >= -2e-6 and rgb.max() <= 1+2e-6
    torch.testing.assert_close(physical_lab(rgb)[:,:1], lab[:,:1],atol=2e-4,rtol=0)


def test_direct_ab_gradient_has_no_rgb_roundtrip():
    import torch
    from experiments.colorization_losses.losses import LossSuite, LossConfig
    suite = LossSuite.__new__(LossSuite)
    torch.nn.Module.__init__(suite); suite.config = LossConfig()
    ab = torch.full((1,2,4,4),2.,requires_grad=True)
    value = suite.loss_ab('huber_ab',ab,torch.zeros_like(ab),torch.ones(1,1,4,4)*50)
    grad = torch.autograd.grad(value.sum(),ab)[0]
    torch.testing.assert_close(grad, torch.full_like(ab,2/5/32))


def test_dino_uses_selected_patch_tokens_excluding_cls_and_registers():
    import torch
    from types import SimpleNamespace
    from experiments.colorization_losses.losses import LossSuite, LossConfig

    class FakeDino(torch.nn.Module):
        config = SimpleNamespace(num_register_tokens=2)

        def forward(self, pixel_values, output_hidden_states):
            assert output_hidden_states
            states = []
            for layer in range(13):
                state = torch.zeros(1, 7, 3)
                state[:, :3] = 1000  # CLS and registers must never enter loss.
                state[:, 3:, layer % 3] = 1
                states.append(state)
            return SimpleNamespace(hidden_states=tuple(states))

    suite = LossSuite.__new__(LossSuite)
    torch.nn.Module.__init__(suite)
    suite.config = LossConfig()
    suite.dino = FakeDino()
    suite.dino_mean = torch.zeros(1, 3, 1, 1)
    suite.dino_std = torch.ones(1, 3, 1, 1)
    features = suite._dino_features(torch.zeros(1, 3, 224, 224))
    assert len(features) == 4
    for feature in features:
        assert feature.shape == (1, 4, 3)
        torch.testing.assert_close(feature[..., 0], torch.ones(1, 4))
        assert feature[..., 1:].count_nonzero() == 0


def test_feature_ab_gradient_matches_finite_difference_at_fixed_l():
    import torch
    from experiments.colorization_losses.losses import LossSuite, LossConfig

    class ToySuite(LossSuite):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = LossConfig()

        def loss(self, name, pred, target):
            return (pred-target).square().flatten(1).mean(1)

    suite = ToySuite()
    ab = torch.full((1, 2, 4, 4), 8., dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(ab)
    L = torch.full((1, 1, 4, 4), 50., dtype=torch.float64)
    grad = torch.autograd.grad(suite.loss_ab('toy', ab, target, L).sum(), ab)[0]
    direction = torch.randn_like(ab)
    epsilon = 1e-4
    numeric = (suite.loss_ab('toy', ab+epsilon*direction, target, L)
               -suite.loss_ab('toy', ab-epsilon*direction, target, L))/(2*epsilon)
    torch.testing.assert_close((grad*direction).sum(), numeric.squeeze(), rtol=1e-5, atol=1e-8)
