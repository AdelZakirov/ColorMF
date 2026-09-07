"""The conditional Pixel MeanFlow objective.

The stochastic state in this module is only ``ab``.  ``L`` is closed over by
the model function used for the JVP and is therefore a fixed condition, not a
JVP primal or tangent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class MeanFlowTerms:
    """All intermediate values needed to audit and test the objective."""

    z: Tensor
    r: Tensor
    t: Tensor
    clean_prediction: Tensor
    velocity_prediction: Tensor
    jvp_direction: Tensor
    velocity_target: Tensor
    average_velocity: Tensor
    average_velocity_jvp: Tensor
    corrected_velocity: Tensor
    main_loss: Tensor
    auxiliary_loss: Tensor
    total_loss: Tensor
    main_loss_per_example: Tensor
    auxiliary_loss_per_example: Tensor
    total_loss_per_example: Tensor
    main_velocity_mse_per_example: Tensor
    auxiliary_velocity_mse_per_example: Tensor


def sample_rt(
    batch_size: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    p_mean: float = 0.8,
    p_std: float = 0.8,
    data_proportion: float = 0.5,
    tr_uniform: bool = False,
    uniform_probability: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """Sample the official pMF ordered logit-normal ``(r,t)`` pairs.

    A deterministic prefix of rows is assigned ``r=t`` for the flow-matching
    component.  The optional uniform replacement is the reference's
    10-percent coverage path.
    """

    if not 0.0 <= data_proportion <= 1.0:
        raise ValueError("data_proportion must be in [0, 1]")
    if p_std <= 0.0:
        raise ValueError("p_std must be positive")
    if not 0.0 <= uniform_probability <= 1.0:
        raise ValueError("uniform_probability must be in [0, 1]")
    t = torch.sigmoid(
        p_mean
        + p_std
        * torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
    )
    r = torch.sigmoid(
        p_mean
        + p_std
        * torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
    )
    if tr_uniform:
        use_uniform = torch.rand(
            batch_size, device=device, dtype=dtype, generator=generator
        ) < uniform_probability
        uniform_t = torch.rand(
            batch_size, device=device, dtype=dtype, generator=generator
        )
        uniform_r = torch.rand(
            batch_size, device=device, dtype=dtype, generator=generator
        )
        t = torch.where(use_uniform, uniform_t, t)
        r = torch.where(use_uniform, uniform_r, r)
    flow_matching_rows = int(batch_size * data_proportion)
    if flow_matching_rows:
        r = r.clone()
        r[:flow_matching_rows] = t[:flow_matching_rows]
    return torch.minimum(r, t), torch.maximum(r, t)


def interpolate(x: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    """pMF linear interpolation ``z_t = (1-t)x + t epsilon``."""

    t_view = t.reshape(-1, *([1] * (x.ndim - 1)))
    return torch.lerp(x, noise, t_view)


def _reference_clip_time(t: Tensor) -> Tensor:
    time = t.float()
    return torch.minimum(
        torch.maximum(time, time.new_tensor(0.05)), time.new_tensor(1.0)
    )


def stabilized_velocity_target(
    x: Tensor, noise: Tensor, z: Tensor, t: Tensor
) -> Tensor:
    """Reference target with the pMF ``clip(t, 0.05, 1)`` endpoint policy."""

    denominator = _reference_clip_time(t).reshape(
        -1, *([1] * (x.ndim - 1))
    )
    return (z.float() - x.float()) / denominator


def average_velocity(
    z: Tensor,
    clean_prediction: Tensor,
    t: Tensor,
) -> Tensor:
    """Reference clean-x velocity ``(z_t - x_hat) / clip(t, 0.05, 1)``."""

    denominator = _reference_clip_time(t).reshape(
        -1, *([1] * (z.ndim - 1))
    )
    return (z.float() - clean_prediction.float()) / denominator


def jvp_average_velocity(
    model: Callable,
    z: Tensor,
    L: Tensor,
    r: Tensor,
    t: Tensor,
    velocity_direction: Tensor,
) -> Tensor:
    """Compute the pMF spatial/time JVP with fixed ``L``.

    The primals are ``(z, r, t)`` and the tangents are
    ``(stop_gradient(v_dir), 0, 1)``.  The condition is deliberately not a
    primal: changing ``L`` is never part of the stochastic tangent.
    """

    zeros = torch.zeros_like(r)
    ones = torch.ones_like(t)

    def average_velocity_fn(z_value: Tensor, r_value: Tensor, t_value: Tensor):
        clean_value, _ = model(z_value, L, r_value, t_value)
        return average_velocity(z_value, clean_value, t_value)

    _, tangent = torch.func.jvp(
        average_velocity_fn,
        (z, r, t),
        (velocity_direction.detach(), zeros, ones),
    )
    return tangent


def meanflow_terms(
    model: Callable,
    x: Tensor,
    L: Tensor,
    *,
    noise: Optional[Tensor] = None,
    r: Optional[Tensor] = None,
    t: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    auxiliary_weight: float = 1.0,
    adaptive_power: float = 1.0,
    adaptive_epsilon: float = 0.01,
    p_mean: float = 0.8,
    p_std: float = 0.8,
    data_proportion: float = 0.5,
    tr_uniform: bool = False,
    uniform_probability: float = 0.1,
) -> MeanFlowTerms:
    """Evaluate the production pMF objective and expose every audit value."""

    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError("x must be the stochastic ab state shaped [B,2,H,W]")
    if L.ndim != 4 or L.shape[1] != 1 or L.shape[0] != x.shape[0]:
        raise ValueError("L must be a fixed [B,1,H,W] condition")
    if noise is None:
        noise = torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )
    if r is None or t is None:
        sampled_r, sampled_t = sample_rt(
            x.shape[0],
            x.device,
            dtype=x.dtype,
            generator=generator,
            p_mean=p_mean,
            p_std=p_std,
            data_proportion=data_proportion,
            tr_uniform=tr_uniform,
            uniform_probability=uniform_probability,
        )
        r = sampled_r if r is None else r
        t = sampled_t if t is None else t

    z = interpolate(x, noise, t)
    clean_prediction, velocity_prediction = model(z, L, r, t)
    _, jvp_direction = model(z, L, t, t)
    target = stabilized_velocity_target(x, noise, z, t)
    average = average_velocity(z, clean_prediction, t)
    jvp = jvp_average_velocity(model, z, L, r, t, jvp_direction)
    delta = (t.float() - r.float()).reshape(-1, *([1] * (x.ndim - 1)))
    corrected = average + delta * jvp.detach()

    # Scalar math and reductions intentionally run in FP32 under BF16 AMP.
    def adaptive_squared_loss(prediction: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        squared = prediction.float().pow(2).flatten(1)
        per_example = squared.sum(dim=1)
        denominator = (
            (per_example + adaptive_epsilon).pow(adaptive_power).detach()
        )
        return (per_example / denominator).mean(), per_example / denominator, squared.mean(dim=1)

    main, main_per_example, main_mse = adaptive_squared_loss(corrected - target)
    auxiliary, auxiliary_per_example, auxiliary_mse = adaptive_squared_loss(
        velocity_prediction - target
    )
    total = main + auxiliary_weight * auxiliary
    total_per_example = main_per_example + auxiliary_weight * auxiliary_per_example
    return MeanFlowTerms(
        z=z,
        r=r,
        t=t,
        clean_prediction=clean_prediction,
        velocity_prediction=velocity_prediction,
        jvp_direction=jvp_direction,
        velocity_target=target,
        average_velocity=average,
        average_velocity_jvp=jvp,
        corrected_velocity=corrected,
        main_loss=main,
        auxiliary_loss=auxiliary,
        total_loss=total,
        main_loss_per_example=main_per_example,
        auxiliary_loss_per_example=auxiliary_per_example,
        total_loss_per_example=total_per_example,
        main_velocity_mse_per_example=main_mse,
        auxiliary_velocity_mse_per_example=auxiliary_mse,
    )
