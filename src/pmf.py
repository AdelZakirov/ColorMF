"""Official Pixel Mean Flow training objective with fixed-L conditioning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


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
    chroma_edge_loss: Tensor
    chroma_edge_loss_per_example: Tensor
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


def _validate_edge_loss_parameters(
    edge_loss_weight: float,
    edge_boundary_boost: float,
    edge_tau: float,
    edge_max_t: float,
) -> tuple[float, float, float, float]:
    try:
        weight = float(edge_loss_weight)
        boundary_boost = float(edge_boundary_boost)
        tau = float(edge_tau)
        max_t = float(edge_max_t)
    except (TypeError, ValueError) as error:
        raise ValueError("edge loss parameters must be finite numbers") from error
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("edge_loss_weight must be finite and non-negative")
    if not math.isfinite(boundary_boost) or boundary_boost < 0.0:
        raise ValueError("edge_boundary_boost must be finite and non-negative")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("edge_tau must be finite and positive")
    if not math.isfinite(max_t) or not 0.0 <= max_t <= 1.0:
        raise ValueError("edge_max_t must be finite and within [0, 1]")
    return weight, boundary_boost, tau, max_t


def _chroma_edge_map(ab: Tensor, tau: float) -> Tensor:
    kernels = ab.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ]
    ).unsqueeze(1) / 8.0
    padded = F.pad(ab, (1, 1, 1, 1), mode="replicate")
    dx = F.conv2d(padded, kernels[0:1].expand(2, -1, -1, -1), groups=2)
    dy = F.conv2d(padded, kernels[1:2].expand(2, -1, -1, -1), groups=2)
    magnitude = torch.sqrt(
        dx.square().sum(dim=1, keepdim=True)
        + dy.square().sum(dim=1, keepdim=True)
        + torch.finfo(ab.dtype).eps
    )
    return magnitude / (magnitude + tau)


def _chroma_edge_loss(
    predicted_clean_ab: Tensor,
    target_ab: Tensor,
    t: Tensor,
    *,
    enabled: bool,
    boundary_boost: float,
    tau: float,
    max_t: float,
) -> tuple[Tensor, Tensor]:
    """Return per-example weighted L1 distances between soft chroma edges."""
    zero_per_example = torch.zeros(
        predicted_clean_ab.shape[0],
        device=predicted_clean_ab.device,
        dtype=torch.float32,
    )
    if not enabled:
        return zero_per_example.sum(), zero_per_example
    if predicted_clean_ab.shape[-2] < 2 or predicted_clean_ab.shape[-1] < 2:
        raise ValueError("edge loss requires spatial dimensions of at least 2x2")

    active = t.flatten() <= max_t
    active_indices = active.nonzero(as_tuple=True)[0]
    if not active_indices.numel():
        return predicted_clean_ab.float().sum() * 0.0, zero_per_example

    predicted_ab = predicted_clean_ab[active_indices].float()
    target_ab = target_ab[active_indices].float()
    predicted_edges = _chroma_edge_map(predicted_ab, tau)
    target_edges = _chroma_edge_map(target_ab, tau)
    weights = (1.0 + boundary_boost * target_edges).detach()
    active_values = (
        (weights * torch.abs(predicted_edges - target_edges)).flatten(1).sum(1)
        / weights.flatten(1).sum(1)
    )
    per_example = zero_per_example.index_copy(0, active_indices, active_values)
    return active_values.mean(), per_example


def _auxiliary_direction(model: Callable, z: Tensor, L: Tensor, t: Tensor) -> Tensor:
    direction_fn = getattr(model, "auxiliary_direction", None)
    if direction_fn is None:
        return model(z, L, t, t)[1]
    return direction_fn(z, L, t)


def _split_diagonal_predictions(
    model: Callable, z: Tensor, L: Tensor, r: Tensor, t: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    diagonal = torch.eq(r, t)
    diagonal_indices = diagonal.nonzero(as_tuple=True)[0]
    off_diagonal_indices = (~diagonal).nonzero(as_tuple=True)[0]

    u_prediction = torch.zeros_like(z)
    velocity_prediction = torch.zeros_like(z)
    jvp_direction = torch.zeros_like(z)
    average_velocity_jvp = torch.zeros_like(z)

    if diagonal_indices.numel():
        u_diagonal, v_diagonal = model(
            z[diagonal_indices],
            L[diagonal_indices],
            r[diagonal_indices],
            t[diagonal_indices],
        )
        u_prediction = u_prediction.index_copy(0, diagonal_indices, u_diagonal)
        velocity_prediction = velocity_prediction.index_copy(
            0, diagonal_indices, v_diagonal
        )
        jvp_direction = jvp_direction.index_copy(
            0, diagonal_indices, v_diagonal.detach()
        )

    if off_diagonal_indices.numel():
        z_off = z[off_diagonal_indices]
        L_off = L[off_diagonal_indices]
        r_off = r[off_diagonal_indices]
        t_off = t[off_diagonal_indices]
        with torch.no_grad():
            direction_off = _auxiliary_direction(model, z_off, L_off, t_off)

        def u_with_aux(z_value: Tensor, t_value: Tensor, r_value: Tensor):
            u_value, v_value = model(z_value, L_off, r_value, t_value)
            return u_value, v_value

        u_off, jvp_off, v_off = torch.func.jvp(
            u_with_aux,
            (z_off, t_off, r_off),
            (
                direction_off.detach(),
                torch.ones_like(t_off),
                torch.zeros_like(r_off),
            ),
            has_aux=True,
        )
        u_prediction = u_prediction.index_copy(0, off_diagonal_indices, u_off)
        velocity_prediction = velocity_prediction.index_copy(
            0, off_diagonal_indices, v_off
        )
        jvp_direction = jvp_direction.index_copy(
            0, off_diagonal_indices, direction_off
        )
        average_velocity_jvp = average_velocity_jvp.index_copy(
            0, off_diagonal_indices, jvp_off
        )

    return (
        u_prediction,
        average_velocity_jvp,
        velocity_prediction,
        jvp_direction,
    )


def meanflow_terms(model: Callable, x: Tensor, L: Tensor, *, noise: Optional[Tensor] = None,
                   r: Optional[Tensor] = None, t: Optional[Tensor] = None,
                   generator: Optional[torch.Generator] = None, auxiliary_weight: float = 1.0,
                   noise_scale: float = 1.0,
                   adaptive_power: float = 1.0, adaptive_epsilon: float = 0.01,
                   p_mean: float = 0.8, p_std: float = 0.8, data_proportion: float = 0.5,
                   tr_uniform: bool = False, uniform_probability: float = 0.1,
                   perceptual_fn: Optional[Callable[..., tuple[Tensor, Tensor]]] = None,
                   lpips_weight: float = 0.0, convnext_weight: float = 0.0,
                   perceptual_max_t: float = 0.8,
                   split_diagonal_jvp: bool = False,
                   edge_loss_enabled: bool = False,
                   edge_loss_weight: float = 0.02,
                   edge_boundary_boost: float = 4.0,
                   edge_tau: float = 0.1,
                   edge_max_t: float = 1.0) -> MeanFlowTerms:
    """Evaluate official pMF training math for stochastic chroma state only."""
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError("x must be stochastic ab state [B,2,H,W]")
    if L.shape != (x.shape[0], 1, *x.shape[-2:]):
        raise ValueError("L must be fixed [B,1,H,W]")
    edge_loss_weight, edge_boundary_boost, edge_tau, edge_max_t = _validate_edge_loss_parameters(
        edge_loss_weight, edge_boundary_boost, edge_tau, edge_max_t
    )
    if not math.isfinite(float(noise_scale)) or noise_scale <= 0:
        raise ValueError("noise_scale must be a finite positive number")
    if noise is None:
        noise = noise_scale * torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )
    if r is None or t is None:
        sampled_r, sampled_t = sample_rt(
            x.shape[0], x.device, dtype=x.dtype, generator=generator, p_mean=p_mean,
            p_std=p_std, data_proportion=data_proportion, tr_uniform=tr_uniform,
            uniform_probability=uniform_probability)
        r = sampled_r if r is None else r
        t = sampled_t if t is None else t

    z = interpolate(x, noise, t)
    target = stabilized_velocity_target(x, noise, z, t).detach()

    if split_diagonal_jvp and torch.any(torch.eq(r, t)) and torch.any(torch.ne(r, t)):
        u, jvp, v, v_direction = _split_diagonal_predictions(model, z, L, r, t)
    elif split_diagonal_jvp and torch.all(torch.eq(r, t)):
        u, v = model(z, L, r, t)
        v_direction = v.detach()
        jvp = torch.zeros_like(u)
    else:
        with torch.no_grad():
            v_direction = _auxiliary_direction(model, z, L, t)
        if v_direction is None:
            raise RuntimeError("training requires the deep v branch")

        def u_with_aux(z_value: Tensor, t_value: Tensor, r_value: Tensor):
            u_value, v_value = model(z_value, L, r_value, t_value)
            return u_value, v_value

        u, jvp, v = torch.func.jvp(
            u_with_aux, (z, t, r),
            (v_direction.detach(), torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )
    corrected = u + (t.float() - r.float()).reshape(-1, 1, 1, 1) * jvp.detach()
    loss_u, loss_u_examples, mse_u = _adaptive_loss(
        corrected - target, adaptive_power, adaptive_epsilon)
    loss_v, loss_v_examples, mse_v = _adaptive_loss(
        v - target, adaptive_power, adaptive_epsilon)

    reconstructed_ab = z - t.reshape(-1, 1, 1, 1) * u
    zeros = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    chroma_edge_loss, chroma_edge_loss_per_example = _chroma_edge_loss(
        reconstructed_ab,
        x,
        t,
        enabled=edge_loss_enabled,
        boundary_boost=edge_boundary_boost,
        tau=edge_tau,
        max_t=edge_max_t,
    )
    lpips_examples = convnext_examples = zeros
    perceptual_examples = zeros
    if perceptual_fn is not None and (lpips_weight or convnext_weight):
        mask = t.flatten() < perceptual_max_t
        active_indices = mask.nonzero(as_tuple=True)[0]
        if active_indices.numel():
            lpips_raw, convnext_raw = perceptual_fn(
                reconstructed_ab[active_indices], x[active_indices], L[active_indices],
                generator=generator)
            lpips_examples = zeros.clone()
            convnext_examples = zeros.clone()
            lpips_examples[active_indices] = lpips_raw.float()
            convnext_examples[active_indices] = convnext_raw.float()
            perceptual_examples = (
                lpips_weight * _adaptive_values(lpips_examples, adaptive_power, adaptive_epsilon)
                + convnext_weight * _adaptive_values(convnext_examples, adaptive_power, adaptive_epsilon))

    total_examples = (
        loss_u_examples
        + auxiliary_weight * loss_v_examples
        + perceptual_examples
        + edge_loss_weight * chroma_edge_loss_per_example
    )
    return MeanFlowTerms(
        z=z, r=r, t=t, u_prediction=u, velocity_prediction=v,
        jvp_direction=v_direction, velocity_target=target, average_velocity_jvp=jvp,
        corrected_velocity=corrected, reconstructed_ab=reconstructed_ab,
        main_loss=loss_u, auxiliary_loss=loss_v,
        perceptual_lpips_loss=lpips_examples.mean(),
        perceptual_convnext_loss=convnext_examples.mean(),
        chroma_edge_loss=chroma_edge_loss,
        chroma_edge_loss_per_example=chroma_edge_loss_per_example,
        total_loss=total_examples.mean(), main_loss_per_example=loss_u_examples,
        auxiliary_loss_per_example=loss_v_examples, total_loss_per_example=total_examples,
        main_velocity_mse_per_example=mse_u, auxiliary_velocity_mse_per_example=mse_v)
