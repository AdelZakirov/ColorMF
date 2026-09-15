import torch
from torch import nn

from src.pmf import average_velocity, meanflow_terms


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_scale = nn.Parameter(torch.tensor(0.37))
        self.v_bias = nn.Parameter(torch.tensor(-0.21))

    def forward(self, z, L, r, t):
        h = t - r
        u = self.u_scale * z + 0.07 * h[:, None, None, None] + 0.11 * L
        v = self.u_scale * z + self.v_bias + 0.03 * h[:, None, None, None]
        return u, v


def reference_terms(model, x, L, noise, r, t, auxiliary_weight):
    z = torch.lerp(x, noise, t[:, None, None, None])
    clipped = t.clamp(0.05, 1)[:, None, None, None]
    target = (z - x) / clipped
    _, direction = model(z, L, t, t)
    ones, zeros = torch.ones_like(t), torch.zeros_like(r)

    def function(z_value, t_value, r_value):
        return model(z_value, L, r_value, t_value)[0]

    u, jvp = torch.func.jvp(
        function, (z, t, r), (direction.detach(), ones, zeros))
    _, v = model(z, L, r, t)
    corrected = u + (t - r)[:, None, None, None] * jvp.detach()

    def adaptive(residual):
        summed = residual.float().pow(2).flatten(1).sum(1)
        return (summed / (summed + 0.01).detach()).mean()

    main, auxiliary = adaptive(corrected - target), adaptive(v - target)
    return z, u, v, direction, jvp, corrected, main, auxiliary, main + auxiliary_weight * auxiliary


def test_analytical_jvp():
    z = torch.randn(2, 2, 3, 3)
    r, t = torch.rand(2), torch.rand(2)
    direction = torch.randn_like(z)

    def function(z_value, t_value, r_value):
        return 2 * z_value + 3 * r_value[:, None, None, None] - 5 * t_value[:, None, None, None]

    _, actual = torch.func.jvp(
        function, (z, t, r),
        (direction, torch.ones_like(t), torch.zeros_like(r)))
    torch.testing.assert_close(actual, 2 * direction - 5)


def test_jvp_matches_fp32_central_difference():
    torch.manual_seed(9)
    model = ToyModel().float()
    z, L = torch.randn(2, 2, 3, 3), torch.randn(2, 1, 3, 3)
    r, t = torch.tensor([0.2, 0.3]), torch.tensor([0.7, 0.8])
    _, direction = model(z, L, t, t)

    def function(z_value, t_value, r_value):
        return model(z_value, L, r_value, t_value)[0]

    _, actual = torch.func.jvp(
        function, (z, t, r),
        (direction.detach(), torch.ones_like(t), torch.zeros_like(r)))
    step = 1e-3
    upper = function(z + step * direction.detach(), t + step, r)
    lower = function(z - step * direction.detach(), t - step, r)
    torch.testing.assert_close(actual, (upper - lower) / (2 * step), atol=2e-4, rtol=2e-4)


def test_production_matches_independent_reference_and_gradients():
    torch.manual_seed(4)
    x, L = torch.randn(2, 2, 3, 3), torch.randn(2, 1, 3, 3)
    noise, r, t = torch.randn_like(x), torch.tensor([0.13, 0.21]), torch.tensor([0.77, 0.91])
    weight = 0.17
    production_model, reference_model = ToyModel(), ToyModel()
    reference_model.load_state_dict(production_model.state_dict())
    actual = meanflow_terms(production_model, x, L, noise=noise, r=r, t=t,
                            auxiliary_weight=weight)
    expected = reference_terms(reference_model, x, L, noise, r, t, weight)
    for left, right in zip(
        (actual.z, actual.u_prediction, actual.velocity_prediction, actual.jvp_direction,
         actual.average_velocity_jvp, actual.corrected_velocity, actual.main_loss,
         actual.auxiliary_loss, actual.total_loss), expected):
        torch.testing.assert_close(left, right)
    actual.total_loss.backward()
    expected[-1].backward()
    for left, right in zip(production_model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_generated_noise_uses_configured_scale():
    x = torch.zeros(1, 2, 2, 2)
    L = torch.zeros(1, 1, 2, 2)
    seed = 17
    terms = meanflow_terms(
        ToyModel(), x, L, noise_scale=0.25,
        generator=torch.Generator().manual_seed(seed),
        r=torch.zeros(1), t=torch.ones(1),
    )
    expected = 0.25 * torch.randn(
        x.shape, generator=torch.Generator().manual_seed(seed)
    )
    torch.testing.assert_close(terms.z, expected)


def test_diagonal_split_matches_legacy_objective_and_gradients():
    torch.manual_seed(14)
    x, L = torch.randn(3, 2, 3, 3), torch.randn(3, 1, 3, 3)
    noise = torch.randn_like(x)
    r = torch.tensor([0.2, 0.3, 0.6])
    t = torch.tensor([0.2, 0.8, 0.9])
    split_model, legacy_model = ToyModel(), ToyModel()
    legacy_model.load_state_dict(split_model.state_dict())
    split = meanflow_terms(
        split_model, x, L, noise=noise, r=r, t=t, split_diagonal_jvp=True
    )
    legacy = meanflow_terms(
        legacy_model, x, L, noise=noise, r=r, t=t, split_diagonal_jvp=False
    )

    for left, right in (
        (split.u_prediction, legacy.u_prediction),
        (split.velocity_prediction, legacy.velocity_prediction),
        (split.jvp_direction, legacy.jvp_direction),
        (split.corrected_velocity, legacy.corrected_velocity),
        (split.main_loss, legacy.main_loss),
        (split.auxiliary_loss, legacy.auxiliary_loss),
        (split.total_loss, legacy.total_loss),
    ):
        torch.testing.assert_close(left, right)
    assert torch.count_nonzero(split.average_velocity_jvp[0]) == 0
    split.total_loss.backward()
    legacy.total_loss.backward()
    for left, right in zip(split_model.parameters(), legacy_model.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_l_is_closed_over_not_a_jvp_primal():
    x = torch.randn(1, 2, 2, 2)
    L = torch.randn(1, 1, 2, 2, requires_grad=True)
    terms = meanflow_terms(ToyModel(), x, L, noise=torch.randn_like(x),
                           r=torch.tensor([0.2]), t=torch.tensor([0.8]))
    # L can receive ordinary conditioning gradients, but it is never a tangent.
    assert terms.jvp_direction.shape == x.shape
    assert terms.z.shape[1] == 2


def test_checkpointed_objective_generator_is_forwarded_to_perceptual_crop():
    generator = torch.Generator().manual_seed(123)
    observed = []

    def perceptual(predicted, target, luminance, *, generator):
        observed.append(generator)
        zeros = torch.zeros(predicted.shape[0], device=predicted.device)
        return zeros, zeros

    x = torch.randn(1, 2, 2, 2)
    meanflow_terms(ToyModel(), x, torch.randn(1, 1, 2, 2),
                   generator=generator, r=torch.tensor([0.1]), t=torch.tensor([0.2]),
                   perceptual_fn=perceptual,
                   lpips_weight=0.4)
    assert observed == [generator]


def test_perceptual_function_receives_only_samples_below_cutoff():
    observed = []
    x = torch.randn(3, 2, 2, 2)
    L = torch.randn(3, 1, 2, 2)

    def perceptual(predicted, target, luminance, *, generator):
        del generator
        observed.append((predicted.shape[0], target.shape[0], luminance.shape[0]))
        assert torch.equal(target, x[[0, 2]])
        assert torch.equal(luminance, L[[0, 2]])
        values = torch.ones(predicted.shape[0], device=predicted.device)
        return values, values

    terms = meanflow_terms(
        ToyModel(), x, L, noise=torch.randn_like(x),
        r=torch.tensor([0.1, 0.2, 0.3]), t=torch.tensor([0.2, 0.8, 0.79]),
        perceptual_fn=perceptual, lpips_weight=0.4, convnext_weight=0.1,
        perceptual_max_t=0.8)

    assert observed == [(2, 2, 2)]
    assert torch.count_nonzero(terms.perceptual_lpips_loss) == 1
    assert torch.count_nonzero(terms.perceptual_convnext_loss) == 1


def test_small_time_clean_conversion_and_endpoint_derivative():
    z, clean = torch.ones(1, 2, 2, 2), torch.zeros(1, 2, 2, 2)
    assert torch.equal(average_velocity(z, clean, torch.tensor([0.01])),
                       torch.full_like(z, 20.0))
    t = torch.tensor([0.05, 1.0])
    z2, clean2 = torch.ones(2, 2, 1, 1), torch.zeros(2, 2, 1, 1)
    _, tangent = torch.func.jvp(lambda value: average_velocity(z2, clean2, value),
                                (t,), (torch.ones_like(t),))
    expected = torch.tensor([-200.0, -0.5]).reshape(2, 1, 1, 1)
    assert torch.equal(tangent, expected.expand_as(tangent))
