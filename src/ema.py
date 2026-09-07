"""Exponential moving average weights for model evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Optional

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    """Maintain a shadow copy of a module state without tracking gradients."""

    def __init__(
        self,
        decay: float = 0.9999,
        *,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if update_after_step < 0:
            raise ValueError("update_after_step must be non-negative")
        if update_every < 1:
            raise ValueError("update_every must be positive")
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.num_updates = 0
        self._shadow: Optional[Dict[str, Tensor]] = None
        self._backup: Optional[Dict[str, Tensor]] = None
        self._started = False

    @property
    def initialized(self) -> bool:
        return self._shadow is not None

    @property
    def ready(self) -> bool:
        return self.initialized and self._started

    def initialize(self, module: nn.Module) -> None:
        if self.initialized:
            self._validate_keys(module.state_dict())
            return
        self._shadow = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }
        self.num_updates = 0
        self._started = False

    def move_to(self, module: nn.Module) -> None:
        self.initialize(module)
        assert self._shadow is not None
        state = module.state_dict()
        self._shadow = {
            key: value.to(device=state[key].device)
            for key, value in self._shadow.items()
        }

    def update(self, module: nn.Module) -> None:
        self.initialize(module)
        assert self._shadow is not None
        self.num_updates += 1
        if self.num_updates <= self.update_after_step:
            return
        current = module.state_dict()
        self._validate_keys(current)
        if not self._started:
            self._shadow = {
                key: value.detach().clone()
                for key, value in current.items()
            }
            self._started = True
            return
        if (self.num_updates - self.update_after_step - 1) % self.update_every:
            return
        with torch.no_grad():
            for key, value in current.items():
                shadow = self._shadow[key]
                if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                    shadow.mul_(self.decay).add_(
                        value.detach(), alpha=1.0 - self.decay
                    )
                else:
                    shadow.copy_(value.detach())

    def store(self, module: nn.Module) -> None:
        if self._backup is not None:
            raise RuntimeError("EMA weights are already stored")
        self._backup = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    def copy_to(self, module: nn.Module) -> None:
        if not self.ready:
            raise RuntimeError("EMA weights are not ready")
        assert self._shadow is not None
        state = module.state_dict()
        self._validate_keys(state)
        with torch.no_grad():
            for key, value in state.items():
                value.copy_(self._shadow[key].to(device=value.device, dtype=value.dtype))

    def restore(self, module: nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA weights are not stored")
        state = module.state_dict()
        self._validate_keys(state, self._backup)
        with torch.no_grad():
            for key, value in state.items():
                value.copy_(self._backup[key].to(device=value.device, dtype=value.dtype))
        self._backup = None

    @contextmanager
    def scope(self, module: nn.Module) -> Iterator[None]:
        if not self.initialized:
            yield
            return
        self.store(module)
        try:
            self.copy_to(module)
            yield
        finally:
            self.restore(module)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "started": self._started,
            "shadow": None
            if self._shadow is None
            else {key: value.detach().clone() for key, value in self._shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("shadow") is None:
            self._shadow = None
            self.num_updates = int(state.get("num_updates", 0))
            self._started = False
            return
        saved_decay = float(state["decay"])
        if saved_decay != self.decay:
            raise ValueError(
                f"EMA decay changed from {saved_decay} to {self.decay}"
            )
        saved_update_after_step = int(state["update_after_step"])
        if saved_update_after_step != self.update_after_step:
            raise ValueError("EMA update_after_step changed")
        saved_update_every = int(state["update_every"])
        if saved_update_every != self.update_every:
            raise ValueError("EMA update_every changed")
        self._shadow = {
            key: value.detach().clone()
            for key, value in state["shadow"].items()
        }
        self.num_updates = int(state["num_updates"])
        self._started = bool(state.get("started", False))

    def _validate_keys(
        self,
        state: Mapping[str, Tensor],
        reference: Optional[Mapping[str, Tensor]] = None,
    ) -> None:
        expected = self._shadow if reference is None else reference
        if expected is None:
            return
        if state.keys() != expected.keys():
            raise ValueError("EMA state keys do not match model state keys")