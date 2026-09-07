import unittest

import torch
from torch import nn

from src.pmf import (
    average_velocity,
    interpolate,
    jvp_average_velocity,
    meanflow_terms,
)


class ToyModel(nn.Module):
    """Small differentiable model used as an independent math oracle target."""

    def __init__(self):
        super().__init__()
        self.clean_scale = nn.Parameter(torch.tensor(0.37))
        self.velocity_bias = nn.Parameter(torch.tensor(-0.21))

    def forward(self, z, L, r, t):
        clean = (
            self.clean_scale * z
            + 0.07 * r[:, None, None, None]
            - 0.13 * t[:, None, None, None]
            + 0.11 * L
        )
        velocity = self.clean_scale * z + self.velocity_bias
        return clean, velocity


def reference_average(z, clean, velocity, r, t):
    del velocity, r
    clipped = torch.minimum(torch.maximum(t, t.new_tensor(0.05)), t.new_tensor(1.0))
    return (z - clean) / clipped[:, None, None, None]


def reference_terms(model, x, L, noise, r, t, auxiliary_weight):
    z = torch.lerp(x, noise, t[:, None, None, None])
    clean, velocity = model(z, L, r, t)
    _, jvp_direction = model(z, L, t, t)
    clipped = torch.minimum(torch.maximum(t, t.new_tensor(0.05)), t.new_tensor(1.0))
    target = (z - x) / clipped[:, None, None, None]
    average = reference_average(z, clean, velocity, r, t)
    ones = torch.ones_like(t)
    zeros = torch.zeros_like(r)

    def fn(z_value, r_value, t_value):
        clean_value, velocity_value = model(z_value, L, r_value, t_value)
        return reference_average(
            z_value, clean_value, velocity_value, r_value, t_value
        )

    _, jvp = torch.func.jvp(
        fn, (z, r, t), (jvp_direction.detach(), zeros, ones)
    )
    corrected = average + (t - r)[:, None, None, None] * jvp.detach()
    main_residual = (corrected.float() - target.float()).pow(2).flatten(1).sum(dim=1)
    auxiliary_residual = (velocity.float() - target.float()).pow(2).flatten(1).sum(dim=1)
    main = (main_residual / (main_residual + 0.01).detach()).mean()
    auxiliary = (auxiliary_residual / (auxiliary_residual + 0.01).detach()).mean()
    return {
        "z": z,
        "clean": clean,
        "velocity": velocity,
        "jvp_direction": jvp_direction,
        "average": average,
        "jvp": jvp,
        "main": main,
        "auxiliary": auxiliary,
        "total": main + auxiliary_weight * auxiliary,
    }


class PmfMathTests(unittest.TestCase):
    def test_analytical_jvp(self):
        # g(z,r,t) = 2z + 3r - 5t, tangent (v,0,1) -> 2v - 5.
        z = torch.randn(2, 2, 3, 3)
        r = torch.rand(2)
        t = torch.rand(2)
        direction = torch.randn_like(z)

        def function(z_value, r_value, t_value):
            return 2.0 * z_value + 3.0 * r_value[:, None, None, None] - 5.0 * t_value[:, None, None, None]

        _, actual = torch.func.jvp(
            function,
            (z, r, t),
            (direction, torch.zeros_like(r), torch.ones_like(t)),
        )
        expected = 2.0 * direction - 5.0
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_production_matches_independent_reference_forward_jvp_loss_gradients(self):
        torch.manual_seed(4)
        x = torch.randn(2, 2, 3, 3, requires_grad=False)
        L = torch.randn(2, 1, 3, 3)
        noise = torch.randn_like(x)
        r = torch.tensor([0.13, 0.21])
        t = torch.tensor([0.77, 0.91])
        weight = 0.17

        production_model = ToyModel()
        reference_model = ToyModel()
        reference_model.load_state_dict(production_model.state_dict())
        production = meanflow_terms(
            production_model, x, L, noise=noise, r=r, t=t, auxiliary_weight=weight
        )
        expected = reference_terms(
            reference_model, x, L, noise, r, t, weight
        )
        for actual, target in (
            (production.z, expected["z"]),
            (production.clean_prediction, expected["clean"]),
            (production.velocity_prediction, expected["velocity"]),
            (production.jvp_direction, expected["jvp_direction"]),
            (production.average_velocity, expected["average"]),
            (production.average_velocity_jvp, expected["jvp"]),
            (production.main_loss, expected["main"]),
            (production.auxiliary_loss, expected["auxiliary"]),
            (production.total_loss, expected["total"]),
        ):
            self.assertTrue(torch.allclose(actual, target, atol=1e-6, rtol=1e-6))

        production_model.zero_grad()
        reference_model.zero_grad()
        production.total_loss.backward()
        expected["total"].backward()
        for actual, target in zip(production_model.parameters(), reference_model.parameters()):
            self.assertTrue(torch.allclose(actual.grad, target.grad, atol=1e-6, rtol=1e-6))

    def test_small_time_uses_clipped_denominator_without_division(self):
        z = torch.ones(1, 2, 2, 2)
        clean = torch.zeros_like(z)
        velocity = torch.full_like(z, 3.0)
        r = torch.tensor([0.0])
        t = torch.tensor([0.01])
        result = average_velocity(z, clean, t)
        self.assertTrue(torch.equal(result, torch.full_like(z, 20.0)))

    def test_clipping_ties_use_half_derivative(self):
        z = torch.ones(2, 2, 1, 1)
        clean = torch.zeros_like(z)
        t = torch.tensor([0.05, 1.0])
        _, tangent = torch.func.jvp(
            lambda value: average_velocity(z, clean, value),
            (t,),
            (torch.ones_like(t),),
        )
        expected = torch.tensor([-200.0, -0.5]).reshape(2, 1, 1, 1)
        self.assertTrue(torch.equal(tangent, expected.expand_as(tangent)))


if __name__ == "__main__":
    unittest.main()
