"""Official Pixel Mean Flow training objective with fixed-L conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class MeanFlowTerms:
    z: Tensor
    r: Tensor
    t: Tensor
    u_prediction: Tensor
    velocity_prediction: Tensor
    jvp_direction: Tensor
    velocity_target: Tensor
    average_velocity_jvp: Tensor
    corrected_velocity: Tensor
    reconstructed_ab: Tensor
    main_loss: Tensor
    auxiliary_loss: Tensor
    perceptual_lpips_loss: Tensor
    perceptual_convnext_loss: Tensor
    total_loss: Tensor
    main_loss_per_example: Tensor
    auxiliary_loss_per_example: Tensor
    total_loss_per_example: Tensor
    main_velocity_mse_per_example: Tensor
    auxiliary_velocity_mse_per_example: Tensor

    @property
    def average_velocity(self) -> Tensor:
        return self.u_prediction


def sample_rt(batch_size: int, device: torch.device, *, dtype: torch.dtype = torch.float32,
              generator: Optional[torch.Generator] = None, p_mean: float = 0.8,
              p_std: float = 0.8, data_proportion: float = 0.5,
              tr_uniform: bool = False, uniform_probability: float = 0.1) -> Tuple[Tensor, Tensor]:
    if not 0.0 <= data_proportion <= 1.0:
        raise ValueError("data_proportion must be in [0,1]")
    if p_std <= 0 or not 0.0 <= uniform_probability <= 1.0:
        raise ValueError("invalid time-distribution parameters")
    t = torch.sigmoid(p_mean + p_std * torch.randn(
        batch_size, device=device, dtype=dtype, generator=generator))
    r = torch.sigmoid(p_mean + p_std * torch.randn(
        batch_size, device=device, dtype=dtype, generator=generator))
    if tr_uniform:
        mask = torch.rand(batch_size, device=device, generator=generator) < uniform_probability
        t = torch.where(mask, torch.rand(batch_size, device=device, dtype=dtype,
                                         generator=generator), t)
        r = torch.where(mask, torch.rand(batch_size, device=device, dtype=dtype,
                                         generator=generator), r)
    diagonal = int(batch_size * data_proportion)
    if diagonal:
        r = r.clone()
        r[:diagonal] = t[:diagonal]
    return torch.minimum(r, t), torch.maximum(r, t)


def interpolate(x: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    return torch.lerp(x, noise, t.reshape(-1, *([1] * (x.ndim - 1))))


def _clip_time(t: Tensor) -> Tensor:
    value = t.float()
    # minimum/maximum preserves the JAX clip boundary's half derivative.
    return torch.minimum(torch.maximum(value, value.new_tensor(0.05)),
                         value.new_tensor(1.0))


def stabilized_velocity_target(x: Tensor, noise: Tensor, z: Tensor, t: Tensor) -> Tensor:
    del noise
    return (z.float() - x.float()) / _clip_time(t).reshape(-1, *([1] * (x.ndim - 1)))


def average_velocity(z: Tensor, clean_prediction: Tensor, t: Tensor) -> Tensor:
    """Clean endpoint to average velocity conversion used by pMF heads."""
    return (z.float() - clean_prediction.float()) / _clip_time(t).reshape(
        -1, *([1] * (z.ndim - 1)))


def jvp_average_velocity(model: Callable, z: Tensor, L: Tensor, r: Tensor, t: Tensor,
                         velocity_direction: Tensor) -> Tensor:
    """JVP of u(z,t,r) along (stop_grad(v_dir), 1, 0), with L closed over."""
    def u_fn(z_value: Tensor, t_value: Tensor, r_value: Tensor) -> Tensor:
        return model(z_value, L, r_value, t_value)[0]

    _, tangent = torch.func.jvp(
        u_fn, (z, t, r),
        (velocity_direction.detach(), torch.ones_like(t), torch.zeros_like(r)))
    return tangent


def _adaptive_loss(residual: Tensor, norm_p: float, norm_eps: float):
    squared = residual.float().pow(2).flatten(1)
    summed = squared.sum(dim=1)
    normalized = summed / (summed + norm_eps).pow(norm_p).detach()
    return normalized.mean(), normalized, squared.mean(dim=1)


def _adaptive_values(values: Tensor, norm_p: float, norm_eps: float) -> Tensor:
    return values.float() / (values.float() + norm_eps).pow(norm_p).detach()


def meanflow_terms(model: Callable, x: Tensor, L: Tensor, *, noise: Optional[Tensor] = None,
                   r: Optional[Tensor] = None, t: Optional[Tensor] = None,
                   generator: Optional[torch.Generator] = None, auxiliary_weight: float = 1.0,
                   adaptive_power: float = 1.0, adaptive_epsilon: float = 0.01,
                   p_mean: float = 0.8, p_std: float = 0.8, data_proportion: float = 0.5,
                   tr_uniform: bool = False, uniform_probability: float = 0.1,
                   perceptual_fn: Optional[Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]] = None,
                   lpips_weight: float = 0.0, convnext_weight: float = 0.0,
                   perceptual_max_t: float = 0.8) -> MeanFlowTerms:
    """Evaluate official pMF training math for stochastic chroma state only."""
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError("x must be stochastic ab state [B,2,H,W]")
    if L.shape != (x.shape[0], 1, *x.shape[-2:]):
        raise ValueError("L must be fixed [B,1,H,W]")
    if noise is None:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    if r is None or t is None:
        sampled_r, sampled_t = sample_rt(
            x.shape[0], x.device, dtype=x.dtype, generator=generator, p_mean=p_mean,
            p_std=p_std, data_proportion=data_proportion, tr_uniform=tr_uniform,
            uniform_probability=uniform_probability)
        r = sampled_r if r is None else r
        t = sampled_t if t is None else t

    z = interpolate(x, noise, t)
    target = stabilized_velocity_target(x, noise, z, t).detach()

    # Official structure: v direction is the auxiliary prediction at h=0.
    _, v_direction = model(z, L, t, t)
    if v_direction is None:
        raise RuntimeError("training requires the deep v branch")

    def u_with_aux(z_value: Tensor, t_value: Tensor, r_value: Tensor):
        u_value, v_value = model(z_value, L, r_value, t_value)
        return u_value, v_value

    u, jvp, v = torch.func.jvp(
        u_with_aux, (z, t, r),
        (v_direction.detach(), torch.ones_like(t), torch.zeros_like(r)), has_aux=True)
    corrected = u + (t.float() - r.float()).reshape(-1, 1, 1, 1) * jvp.detach()
    loss_u, loss_u_examples, mse_u = _adaptive_loss(
        corrected - target, adaptive_power, adaptive_epsilon)
    loss_v, loss_v_examples, mse_v = _adaptive_loss(
        v - target, adaptive_power, adaptive_epsilon)

    reconstructed_ab = z - t.reshape(-1, 1, 1, 1) * u
    zeros = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    lpips_examples = convnext_examples = zeros
    perceptual_examples = zeros
    if perceptual_fn is not None and (lpips_weight or convnext_weight):
        lpips_raw, convnext_raw = perceptual_fn(reconstructed_ab, x, L)
        mask = t.flatten() < perceptual_max_t
        lpips_examples = torch.where(mask, lpips_raw.float(), zeros)
        convnext_examples = torch.where(mask, convnext_raw.float(), zeros)
        perceptual_examples = (
            lpips_weight * _adaptive_values(lpips_examples, adaptive_power, adaptive_epsilon)
            + convnext_weight * _adaptive_values(convnext_examples, adaptive_power, adaptive_epsilon))

    total_examples = loss_u_examples + auxiliary_weight * loss_v_examples + perceptual_examples
    return MeanFlowTerms(
        z=z, r=r, t=t, u_prediction=u, velocity_prediction=v,
        jvp_direction=v_direction, velocity_target=target, average_velocity_jvp=jvp,
        corrected_velocity=corrected, reconstructed_ab=reconstructed_ab,
        main_loss=loss_u, auxiliary_loss=loss_v,
        perceptual_lpips_loss=lpips_examples.mean(),
        perceptual_convnext_loss=convnext_examples.mean(),
        total_loss=total_examples.mean(), main_loss_per_example=loss_u_examples,
        auxiliary_loss_per_example=loss_v_examples, total_loss_per_example=total_examples,
        main_velocity_mse_per_example=mse_u, auxiliary_velocity_mse_per_example=mse_v)
