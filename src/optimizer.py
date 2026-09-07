"""PyTorch port of the Optax Muon recipe used by official pMF-B."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer


def _newton_schulz(update: Tensor, steps: int = 5,
                    coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
                    eps: float = 1e-8) -> Tensor:
    """Optax default Frobenius-preconditioned Newton-Schulz iteration."""
    original_dtype = update.dtype
    x = update.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + eps)
    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(original_dtype)


class Muon(Optimizer):
    """Optax-compatible partitioned Muon.

    As in ``optax.contrib.muon``, exactly 2-D parameters use Nesterov Muon;
    biases, norm scales, token tensors, and convolution kernels use
    Nesterov-AdamW.
    PyTorch stores linear kernels transposed relative to Flax, so width scaling
    is computed from ``(fan_in=shape[1], fan_out=shape[0])``.
    """

    def __init__(self, params, lr: float = 1e-3, beta: float = 0.95,
                 adam_b1: float = 0.9, adam_b2: float = 0.95,
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 adam_weight_decay: float = 0.0, nesterov: bool = True,
                 ns_steps: int = 5):
        defaults = dict(lr=lr, beta=beta, adam_b1=adam_b1, adam_b2=adam_b2,
                        eps=eps, weight_decay=weight_decay,
                        adam_weight_decay=adam_weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]
                if parameter.ndim == 2:
                    momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                    beta = group["beta"]
                    momentum.mul_(beta).add_(gradient, alpha=1 - beta)
                    # Matches Optax's bias-corrected Nesterov form.
                    if group["nesterov"]:
                        update = beta * momentum / (1 - beta ** (step + 1))
                        update = update + (1 - beta) * gradient / (1 - beta ** step)
                    else:
                        update = momentum / (1 - beta ** step)
                    update = _newton_schulz(update, group["ns_steps"], eps=group["eps"])
                    fan_out, fan_in = parameter.shape
                    update.mul_(math.sqrt(max(1.0, fan_out / fan_in)))
                    if group["weight_decay"]:
                        parameter.mul_(1 - lr * group["weight_decay"])
                    parameter.add_(update, alpha=-lr)
                else:
                    b1, b2, eps = group["adam_b1"], group["adam_b2"], group["eps"]
                    first = state.setdefault("exp_avg", torch.zeros_like(parameter))
                    second = state.setdefault("exp_avg_sq", torch.zeros_like(parameter))
                    first.mul_(b1).add_(gradient, alpha=1 - b1)
                    second.mul_(b2).addcmul_(gradient, gradient, value=1 - b2)
                    if group["nesterov"]:
                        first_hat = b1 * first / (1 - b1 ** (step + 1))
                        first_hat = first_hat + (1 - b1) * gradient / (1 - b1 ** step)
                    else:
                        first_hat = first / (1 - b1 ** step)
                    second_hat = second / (1 - b2 ** step)
                    denominator = second_hat.sqrt().add_(eps)
                    if group["adam_weight_decay"]:
                        parameter.mul_(1 - lr * group["adam_weight_decay"])
                    parameter.addcdiv_(first_hat, denominator, value=-lr)
        return loss
