"""Checkpointable fixed-decay and official EDM-style pMF EMA shadows."""

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


class EMAManager:
    """Maintain one legacy fixed EMA or multiple pMF EDM half-lives.

    EDM decay is expressed in actual global images, generalizing the official
    ``step * 1024`` implementation to arbitrary global batch sizes.
    """

    def __init__(self, *, ema_type: str = "edm",
                 half_lives_kimg: tuple[float, ...] = (500, 1000, 2000),
                 decay: float = 0.9999, update_after_step: int = 0,
                 update_every: int = 1, rampup_ratio: float = 0.05):
        if ema_type not in {"edm", "fixed"}:
            raise ValueError("ema_type must be 'edm' or 'fixed'")
        if not half_lives_kimg or any(value <= 0 for value in half_lives_kimg):
            raise ValueError("EDM half-lives must be positive")
        self.ema_type = ema_type
        self.half_lives_kimg = tuple(float(value) for value in half_lives_kimg)
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.rampup_ratio = float(rampup_ratio)
        self.num_updates = 0
        self.images_seen = 0
        self._started = False
        self._shadows: Optional[Dict[str, Dict[str, Tensor]]] = None
        self._backup: Optional[Dict[str, Tensor]] = None

    @property
    def variants(self) -> tuple[str, ...]:
        if self.ema_type == "fixed":
            return ("fixed",)
        return tuple(f"{value:g}" for value in self.half_lives_kimg)

    @property
    def initialized(self) -> bool:
        return self._shadows is not None

    @property
    def ready(self) -> bool:
        return self.initialized and self._started

    def initialize(self, module: nn.Module) -> None:
        if self.initialized:
            self._validate(module.state_dict())
            return
        state = module.state_dict()
        self._shadows = {
            variant: {key: value.detach().clone() for key, value in state.items()}
            for variant in self.variants
        }

    def move_to(self, module: nn.Module) -> None:
        self.initialize(module)
        state = module.state_dict()
        assert self._shadows is not None
        self._shadows = {
            variant: {key: value.to(device=state[key].device)
                      for key, value in shadow.items()}
            for variant, shadow in self._shadows.items()
        }

    def _decay_for(self, variant: str, global_images: int) -> float:
        if self.ema_type == "fixed":
            return self.decay
        target = float(variant) * 1000.0
        ramped = max(float(self.images_seen) * self.rampup_ratio, 1e-8)
        half_life = min(target, ramped)
        return 0.5 ** (global_images / half_life)

    def update(self, module: nn.Module, *, global_images: int) -> None:
        if global_images <= 0:
            raise ValueError("global_images must be positive")
        self.initialize(module)
        assert self._shadows is not None
        self.num_updates += 1
        self.images_seen += int(global_images)
        if self.num_updates <= self.update_after_step:
            return
        if (self.num_updates - self.update_after_step - 1) % self.update_every:
            return
        current = module.state_dict()
        self._validate(current)
        if self.ema_type == "fixed" and not self._started:
            self._shadows = {
                "fixed": {key: value.detach().clone() for key, value in current.items()}
            }
            self._started = True
            return
        with torch.no_grad():
            for variant, shadow_state in self._shadows.items():
                beta = self._decay_for(variant, global_images)
                for key, value in current.items():
                    shadow = shadow_state[key]
                    if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                        shadow.mul_(beta).add_(value.detach(), alpha=1.0 - beta)
                    else:
                        shadow.copy_(value.detach())
        self._started = True

    def store(self, module: nn.Module) -> None:
        if self._backup is not None:
            raise RuntimeError("EMA weights are already stored")
        self._backup = {key: value.detach().clone()
                        for key, value in module.state_dict().items()}

    def copy_to(self, module: nn.Module, variant: Optional[str] = None) -> None:
        if not self.ready:
            raise RuntimeError("EMA weights are not ready")
        variant = self.variants[0] if variant is None else str(variant)
        if variant not in self.variants:
            raise ValueError(f"unknown EMA variant {variant}; choose from {self.variants}")
        state = module.state_dict()
        self._validate(state)
        assert self._shadows is not None
        with torch.no_grad():
            for key, value in state.items():
                value.copy_(self._shadows[variant][key].to(value.device, value.dtype))

    def restore(self, module: nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA weights are not stored")
        with torch.no_grad():
            for key, value in module.state_dict().items():
                value.copy_(self._backup[key].to(value.device, value.dtype))
        self._backup = None

    @contextmanager
    def scope(self, module: nn.Module, variant: Optional[str] = None) -> Iterator[None]:
        if not self.ready:
            yield
            return
        self.store(module)
        try:
            self.copy_to(module, variant)
            yield
        finally:
            self.restore(module)

    def state_dict(self) -> dict:
        return {
            "ema_type": self.ema_type,
            "half_lives_kimg": self.half_lives_kimg,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "rampup_ratio": self.rampup_ratio,
            "num_updates": self.num_updates,
            "images_seen": self.images_seen,
            "started": self._started,
            "shadows": None if self._shadows is None else {
                variant: {key: value.detach().clone() for key, value in shadow.items()}
                for variant, shadow in self._shadows.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for key, current in (
            ("ema_type", self.ema_type),
            ("half_lives_kimg", self.half_lives_kimg),
            ("decay", self.decay),
            ("update_after_step", self.update_after_step),
            ("update_every", self.update_every),
        ):
            saved = state.get(key, current)
            if key == "half_lives_kimg":
                saved = tuple(float(value) for value in saved)
            if saved != current:
                raise ValueError(f"EMA setting {key} changed from {saved} to {current}")
        shadows = state.get("shadows")
        # Migration path for checkpoints written by the original single-EMA
        # ColorMF implementation.
        if shadows is None and state.get("shadow") is not None:
            if self.ema_type != "fixed":
                raise ValueError("legacy single-shadow checkpoint requires fixed EMA mode")
            shadows = {"fixed": state["shadow"]}
        self._shadows = None if shadows is None else {
            str(variant): {key: value.detach().clone() for key, value in shadow.items()}
            for variant, shadow in shadows.items()
        }
        self.num_updates = int(state.get("num_updates", 0))
        self.images_seen = int(state.get("images_seen", 0))
        self._started = bool(state.get("started", self.num_updates > self.update_after_step))

    def _validate(self, state: Mapping[str, Tensor]) -> None:
        if self._shadows is None:
            return
        for shadow in self._shadows.values():
            if state.keys() != shadow.keys():
                raise ValueError("EMA state keys do not match model state keys")
