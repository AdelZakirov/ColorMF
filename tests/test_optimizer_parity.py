"""Direct one-step parity against optax.contrib.muon.

The matrix is transposed between frameworks because Flax stores Dense kernels
as [fan_in, fan_out], while torch.nn.Linear stores [fan_out, fan_in].
"""

import jax.numpy as jnp
import numpy as np
import optax
import torch

from src.optimizer import Muon


def test_one_step_matches_optax_for_matrix_vector_and_token_tensor():
    learning_rate = 1e-3
    torch_parameters = {
        "matrix": torch.tensor([[0.2, -0.1], [0.3, 0.5], [-0.4, 0.7]],
                               requires_grad=True),
        "vector": torch.tensor([0.1, -0.2, 0.3], requires_grad=True),
        "token": torch.tensor([[[0.2, -0.5], [0.7, 0.1], [-0.3, 0.4]]],
                              requires_grad=True),
    }
    torch_gradients = {
        "matrix": torch.tensor([[0.01, -0.03], [0.02, 0.04], [-0.05, 0.06]]),
        "vector": torch.tensor([0.03, -0.07, 0.11]),
        "token": torch.tensor([[[0.02, 0.04], [-0.06, 0.08], [0.1, -0.12]]]),
    }
    for name, parameter in torch_parameters.items():
        parameter.grad = torch_gradients[name]
    torch_optimizer = Muon(torch_parameters.values(), lr=learning_rate,
                           adam_b2=0.95)
    torch_optimizer.step()

    optax_parameters = {
        "matrix": jnp.asarray([[0.2, 0.3, -0.4], [-0.1, 0.5, 0.7]]),
        "vector": jnp.asarray([0.1, -0.2, 0.3]),
        "token": jnp.asarray([[[0.2, -0.5], [0.7, 0.1], [-0.3, 0.4]]]),
    }
    optax_gradients = {
        "matrix": jnp.asarray([[0.01, 0.02, -0.05], [-0.03, 0.04, 0.06]]),
        "vector": jnp.asarray([0.03, -0.07, 0.11]),
        "token": jnp.asarray([[[0.02, 0.04], [-0.06, 0.08], [0.1, -0.12]]]),
    }
    optax_optimizer = optax.contrib.muon(
        learning_rate=learning_rate, adam_b2=0.95)
    optax_state = optax_optimizer.init(optax_parameters)
    updates, _ = optax_optimizer.update(
        optax_gradients, optax_state, optax_parameters)
    optax_parameters = optax.apply_updates(optax_parameters, updates)

    np.testing.assert_allclose(
        torch_parameters["matrix"].detach().numpy(),
        np.asarray(optax_parameters["matrix"]).T,
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        torch_parameters["vector"].detach().numpy(),
        np.asarray(optax_parameters["vector"]),
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        torch_parameters["token"].detach().numpy(),
        np.asarray(optax_parameters["token"]),
        rtol=1e-6, atol=1e-7,
    )
