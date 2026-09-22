"""PyTorch Lightning training and rank-safe qualitative validation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from PIL import Image, ImageDraw

from .lab import lab_to_rgb
from .ema import EMAManager
from .model import PixelMeanFlowB
from .optimizer import Muon
from .perceptual import PerceptualLosses
from .pmf import _validate_edge_loss_parameters, meanflow_terms


def _save_qualitative_row(
    filename: str, L: torch.Tensor, gt: torch.Tensor, samples: torch.Tensor
) -> None:
    L_image = np.asarray(L[0, 0].float().cpu().tolist(), dtype=np.float32)
    L_image = ((L_image + 1.0) * 127.5).clip(0, 255)
    L_image = np.repeat(L_image.astype(np.uint8)[..., None], 3, axis=-1)
    gt_rgb = lab_to_rgb(L, gt)[0]
    sample_rgb = lab_to_rgb(
        L.expand(samples.shape[0], -1, -1, -1), samples
    )
    images = [L_image, gt_rgb, *list(sample_rgb)]
    height, width = images[0].shape[:2]
    canvas = Image.new("RGB", (width * len(images), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["L", "GT"] + [f"seed{i + 1}" for i in range(len(samples))]
    for index, (image, label) in enumerate(zip(images, labels)):
        tile = Image.fromarray(image)
        canvas.paste(tile, (index * width, 24))
        draw.text((index * width + 4, 4), label, fill="black")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(filename)


def _select_validation_visuals(
    visuals: Dict[str, tuple],
    fixed_validation_ids: set[str],
    validation_image_count: int,
) -> Dict[str, tuple]:
    if fixed_validation_ids:
        return {
            image_id: visuals[image_id]
            for image_id in sorted(fixed_validation_ids)
            if image_id in visuals
        }
    return dict(sorted(visuals.items())[:validation_image_count])


def _mean_active_values(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Average a raw validation metric over eligible examples only."""
    active = active.to(device=values.device, dtype=torch.bool)
    if torch.any(active):
        return values[active].mean()
    return values.sum() * 0.0


class PMFColorizerModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model: Optional[dict] = None,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        max_steps: Optional[int] = None,
        optimizer: str = "muon",
        adam_b2: float = 0.95,
        lr_schedule: str = "constant",
        auxiliary_weight: float = 1.0,
        noise_scale: float = 1.0,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        time_p_mean: float = 0.8,
        time_p_std: float = 0.8,
        time_data_proportion: float = 0.5,
        time_tr_uniform: bool = False,
        time_uniform_probability: float = 0.1,
        split_diagonal_jvp: bool = False,
        random_seed: int = 1234,
        fixed_validation_ids: Optional[Iterable[str]] = None,
        validation_image_count: int = 4,
        validation_sample_seeds: Optional[Sequence[int]] = None,
        ema_enabled: bool = True,
        ema_type: str = "edm",
        ema_half_lives_kimg: Sequence[float] = (500, 1000, 2000),
        ema_decay: Optional[float] = 0.9999,
        ema_update_after_step: int = 0,
        ema_update_every: int = 1,
        ema_use_for_validation: bool = True,
        ema_validation_variant: Optional[str] = None,
        lpips_enabled: bool = False,
        lpips_weight: float = 0.4,
        convnext_enabled: bool = False,
        convnext_weight: float = 0.1,
        perceptual_max_t: float = 0.8,
        edge_loss_enabled: bool = False,
        edge_loss_weight: float = 0.02,
        edge_boundary_boost: float = 4.0,
        edge_tau: float = 0.1,
        edge_max_t: float = 1.0,
        sample_dir: str = "qualitative",
    ):
        edge_loss_weight, edge_boundary_boost, edge_tau, edge_max_t = _validate_edge_loss_parameters(
            edge_loss_weight, edge_boundary_boost, edge_tau, edge_max_t
        )
        super().__init__()
        model_config = model or {}
        self.save_hyperparameters()
        self.model = PixelMeanFlowB(**model_config)
        self.auxiliary_weight = auxiliary_weight
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer_name = optimizer.lower()
        self.adam_b2 = adam_b2
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_scale = float(noise_scale)
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0:
            raise ValueError("noise_scale must be a finite positive number")
        self.time_p_mean = time_p_mean
        self.time_p_std = time_p_std
        self.time_data_proportion = time_data_proportion
        self.time_tr_uniform = time_tr_uniform
        self.time_uniform_probability = time_uniform_probability
        self.split_diagonal_jvp = bool(split_diagonal_jvp)
        self.random_seed = random_seed
        self.fixed_validation_ids = set(fixed_validation_ids or [])
        self.validation_image_count = max(0, int(validation_image_count))
        sample_seeds = (
            validation_sample_seeds
            if validation_sample_seeds is not None
            else (1, 2, 3, 4)
        )
        self.validation_sample_seeds = tuple(int(seed) for seed in sample_seeds)
        if not self.validation_sample_seeds:
            raise ValueError("validation_sample_seeds must not be empty")
        self.ema_decay = ema_decay
        self.ema_type = ema_type
        self.ema_half_lives_kimg = tuple(ema_half_lives_kimg)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_update_every = int(ema_update_every)
        self.ema = (
            EMAManager(
                ema_type=ema_type,
                half_lives_kimg=self.ema_half_lives_kimg,
                decay=0.9999 if ema_decay is None else ema_decay,
                update_after_step=self.ema_update_after_step,
                update_every=self.ema_update_every,
            )
            if ema_enabled
            else None
        )
        self.ema_use_for_validation = ema_use_for_validation
        self.ema_validation_variant = (
            str(ema_validation_variant) if ema_validation_variant is not None
            else (self.ema.variants[0] if self.ema is not None else None)
        )
        self.lpips_enabled = lpips_enabled
        self.lpips_weight = lpips_weight
        self.convnext_enabled = convnext_enabled
        self.convnext_weight = convnext_weight
        self.perceptual_max_t = perceptual_max_t
        self.edge_loss_enabled = bool(edge_loss_enabled)
        self.edge_loss_weight = edge_loss_weight
        self.edge_boundary_boost = edge_boundary_boost
        self.edge_tau = edge_tau
        self.edge_max_t = edge_max_t
        self._perceptual_losses: Optional[PerceptualLosses] = None
        self._pending_ema_state: Optional[dict] = None
        self._ema_validation_applied = False
        self.sample_dir = sample_dir
        self._validation_visuals: Dict[str, tuple] = {}
        self._validation_metrics: Dict[str, tuple] = {}
        self._train_generator: Optional[torch.Generator] = None
        self._validation_generator: Optional[torch.Generator] = None
        self._pending_train_generator_state: Optional[torch.Tensor] = None
        self._train_batch_started_at: Optional[float] = None
        self._ema_images_pending = 0

    def on_fit_start(self):
        if self.lpips_enabled or self.convnext_enabled:
            self._perceptual_losses = PerceptualLosses(
                use_lpips=self.lpips_enabled,
                use_convnext=self.convnext_enabled,
            ).to(self.device)

    def forward(self, z, L, r, t, *, return_velocity: bool = True):
        return self.model(z, L, r, t, return_velocity=return_velocity)

    @torch.no_grad()
    def sample(self, L: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs.setdefault("noise_scale", self.noise_scale)
        return self.model.sample(L, **kwargs)

    @torch.no_grad()
    def sample_lab(self, L: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs.setdefault("noise_scale", self.noise_scale)
        return self.model.sample_lab(L, **kwargs)

    def training_step(self, batch: dict, batch_idx: int):
        if self._train_generator is None:
            self._train_generator = torch.Generator(device=batch["ab"].device)
            self._train_generator.manual_seed(
                self.random_seed + 1_000_003 * int(self.global_rank)
            )
        terms = meanflow_terms(
            self.model,
            batch["ab"],
            batch["L"],
            auxiliary_weight=self.auxiliary_weight,
            noise_scale=self.noise_scale,
            adaptive_power=self.norm_p,
            adaptive_epsilon=self.norm_eps,
            generator=self._train_generator,
            p_mean=self.time_p_mean,
            p_std=self.time_p_std,
            data_proportion=self.time_data_proportion,
            tr_uniform=self.time_tr_uniform,
            uniform_probability=self.time_uniform_probability,
            perceptual_fn=self._perceptual_losses,
            lpips_weight=self.lpips_weight if self.lpips_enabled else 0.0,
            convnext_weight=self.convnext_weight if self.convnext_enabled else 0.0,
            perceptual_max_t=self.perceptual_max_t,
            split_diagonal_jvp=self.split_diagonal_jvp,
            edge_loss_enabled=self.edge_loss_enabled,
            edge_loss_weight=self.edge_loss_weight,
            edge_boundary_boost=self.edge_boundary_boost,
            edge_tau=self.edge_tau,
            edge_max_t=self.edge_max_t,
        )
        batch_size = batch["ab"].shape[0]
        world_size = int(getattr(self.trainer, "world_size", 1))
        self._ema_images_pending += batch_size * world_size
        self.log(
            "train/pMF_loss",
            terms.main_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/auxiliary_velocity_loss",
            terms.auxiliary_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/total_loss",
            terms.total_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if self.lpips_enabled:
            self.log("train/lpips_loss", terms.perceptual_lpips_loss,
                     on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if self.convnext_enabled:
            self.log("train/convnext_loss", terms.perceptual_convnext_loss,
                     on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if self.edge_loss_enabled:
            self.log(
                "train/chroma_edge_loss",
                terms.chroma_edge_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            "train/raw_main_velocity_mse",
            terms.main_velocity_mse_per_example.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/raw_auxiliary_velocity_mse",
            terms.auxiliary_velocity_mse_per_example.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/learning_rate",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        return terms.total_loss

    def on_train_batch_start(self, batch: dict, batch_idx: int):
        self._train_batch_started_at = time.perf_counter()

    def on_train_batch_end(self, outputs, batch: dict, batch_idx: int):
        if self._train_batch_started_at is None:
            return
        elapsed = max(time.perf_counter() - self._train_batch_started_at, 1.0e-6)
        world_size = int(getattr(self.trainer, "world_size", 1))
        global_samples = batch["ab"].shape[0] * world_size
        self.log(
            "train/samples_per_sec",
            global_samples / elapsed,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        self._train_batch_started_at = None

    def on_train_start(self):
        self._train_generator = torch.Generator(device=self.device)
        if self._pending_train_generator_state is not None:
            self._train_generator.set_state(self._pending_train_generator_state)
            self._pending_train_generator_state = None
        else:
            self._train_generator.manual_seed(
                self.random_seed + 1_000_003 * int(self.global_rank)
            )
        if self.ema is not None:
            if self._pending_ema_state is not None:
                self.ema.load_state_dict(self._pending_ema_state)
                self._pending_ema_state = None
            else:
                self.ema.initialize(self.model)
            self.ema.move_to(self.model)
        world_size = int(getattr(self.trainer, "world_size", 1))
        datamodule_hparams = getattr(self.trainer.datamodule, "hparams", None)
        batch_size = getattr(datamodule_hparams, "batch_size", None)
        if batch_size is None:
            batch_size = self.trainer.train_dataloader.batch_size
        batch_size = int(batch_size)
        accumulation = int(self.trainer.accumulate_grad_batches)
        self.log(
            "train/gpu_count",
            float(world_size),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/global_batch_size",
            float(batch_size * world_size * accumulation),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        self._validation_visuals.clear()
        self._validation_metrics.clear()
        if (
            self.ema is not None
            and self.ema_use_for_validation
            and self.ema.ready
        ):
            self.ema.store(self.model)
            self.ema.copy_to(self.model, self.ema_validation_variant)
            self._ema_validation_applied = True
        self._validation_generator = torch.Generator(device=self.device)
        self._validation_generator.manual_seed(
            self.random_seed + 2_000_033 * (int(self.current_epoch) + 1)
        )

    def validation_step(self, batch: dict, batch_idx: int):
        terms = meanflow_terms(
            self.model,
            batch["ab"],
            batch["L"],
            auxiliary_weight=self.auxiliary_weight,
            noise_scale=self.noise_scale,
            adaptive_power=self.norm_p,
            adaptive_epsilon=self.norm_eps,
            generator=self._validation_generator,
            p_mean=self.time_p_mean,
            p_std=self.time_p_std,
            data_proportion=self.time_data_proportion,
            tr_uniform=self.time_tr_uniform,
            uniform_probability=self.time_uniform_probability,
            perceptual_fn=self._perceptual_losses,
            lpips_weight=self.lpips_weight if self.lpips_enabled else 0.0,
            convnext_weight=self.convnext_weight if self.convnext_enabled else 0.0,
            perceptual_max_t=self.perceptual_max_t,
            split_diagonal_jvp=self.split_diagonal_jvp,
            edge_loss_enabled=self.edge_loss_enabled,
            edge_loss_weight=self.edge_loss_weight,
            edge_boundary_boost=self.edge_boundary_boost,
            edge_tau=self.edge_tau,
            edge_max_t=self.edge_max_t,
        )
        for index, image_id in enumerate(batch["image_id"]):
            image_id = str(image_id)
            self._validation_metrics[image_id] = (
                float(terms.main_loss_per_example[index].detach().cpu()),
                float(terms.total_loss_per_example[index].detach().cpu()),
                float(terms.main_velocity_mse_per_example[index].detach().cpu()),
                float(
                    terms.auxiliary_velocity_mse_per_example[index]
                    .detach()
                    .cpu()
                ),
                float(terms.chroma_edge_loss_per_example[index].detach().cpu()),
                float((terms.t[index] <= self.edge_max_t).detach().cpu()),
            )
            should_capture = (
                image_id in self.fixed_validation_ids
                if self.fixed_validation_ids
                else len(self._validation_visuals) < self.validation_image_count
            )
            if should_capture:
                self._validation_visuals[image_id] = (
                    batch["L"][index : index + 1].detach().cpu(),
                    batch["ab"][index : index + 1].detach().cpu(),
                )

    def on_validation_epoch_end(self):
        visuals = self._validation_visuals
        metrics = self._validation_metrics
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(
                gathered, {"visuals": visuals, "metrics": metrics}
            )
            if self.global_rank == 0:
                visuals = {}
                metrics = {}
                for rank_payload in gathered:
                    visuals.update(rank_payload["visuals"])
                    metrics.update(rank_payload["metrics"])
        if self.global_rank != 0:
            self._restore_training_weights()
            self._validation_visuals.clear()
            self._validation_metrics.clear()
            return
        try:
            visuals = _select_validation_visuals(
                visuals,
                self.fixed_validation_ids,
                self.validation_image_count,
            )
            if metrics:
                values = torch.tensor(list(metrics.values()), device=self.device)
                self.log(
                    "val/pMF_loss",
                    values[:, 0].mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "val/total_loss",
                    values[:, 1].mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "val/raw_main_velocity_mse",
                    values[:, 2].mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "val/raw_auxiliary_velocity_mse",
                    values[:, 3].mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                if self.edge_loss_enabled:
                    self.log(
                        "val/chroma_edge_loss",
                        _mean_active_values(values[:, 4], values[:, 5]),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                        rank_zero_only=True,
                    )
            output_dir = (
                Path(self.trainer.default_root_dir) / self.sample_dir / "current"
            )
            artifact_path = f"{self.sample_dir.rstrip('/')}/current"
            for image_id, (L, gt) in visuals.items():
                device = self.device
                generated = self.sample(
                    L.to(device),
                    seeds=self.validation_sample_seeds,
                    image_ids=[image_id],
                ).cpu()
                safe_image_id = "".join(
                    character
                    if character.isalnum() or character in "._-"
                    else "_"
                    for character in image_id
                ).strip("._") or "image"
                filename = str(output_dir / f"{safe_image_id}.png")
                _save_qualitative_row(
                    filename,
                    L,
                    gt,
                    generated,
                )
                self._log_mlflow_artifact(filename, artifact_path)
        finally:
            self._restore_training_weights()
            self._validation_visuals.clear()
            self._validation_metrics.clear()

    def _restore_training_weights(self) -> None:
        if not self._ema_validation_applied:
            return
        assert self.ema is not None
        self.ema.restore(self.model)
        self._ema_validation_applied = False

    def _log_mlflow_artifact(self, filename: str, artifact_path: str) -> None:
        for logger in getattr(self.trainer, "loggers", []):
            if not isinstance(logger, MLFlowLogger):
                continue
            logger.experiment.log_artifact(
                logger.run_id, filename, artifact_path=artifact_path
            )

    def on_save_checkpoint(self, checkpoint: dict):
        if self._train_generator is not None:
            checkpoint["train_generator_state"] = self._train_generator.get_state()
        checkpoint["ema"] = (
            self.ema.state_dict()
            if self.ema is not None and self.ema.initialized
            else None
        )

    def on_load_checkpoint(self, checkpoint: dict):
        self._pending_train_generator_state = checkpoint.get("train_generator_state")
        self._pending_ema_state = checkpoint.get("ema")
        saved_hyperparameters = checkpoint.get("hyper_parameters", {})
        protected = {
            "learning_rate": "learning_rate", "weight_decay": "weight_decay",
            "warmup_steps": "warmup_steps", "max_steps": "max_steps",
            "optimizer": "optimizer_name", "adam_b2": "adam_b2",
            "lr_schedule": "lr_schedule", "auxiliary_weight": "auxiliary_weight",
            "noise_scale": "noise_scale",
            "norm_p": "norm_p", "norm_eps": "norm_eps",
            "time_p_mean": "time_p_mean", "time_p_std": "time_p_std",
            "time_data_proportion": "time_data_proportion",
            "time_tr_uniform": "time_tr_uniform",
            "time_uniform_probability": "time_uniform_probability",
            "split_diagonal_jvp": "split_diagonal_jvp",
            "edge_loss_enabled": "edge_loss_enabled",
            "edge_loss_weight": "edge_loss_weight",
            "edge_boundary_boost": "edge_boundary_boost",
            "edge_tau": "edge_tau",
            "edge_max_t": "edge_max_t",
            "random_seed": "random_seed", "ema_decay": "ema_decay",
            "ema_type": "ema_type", "ema_half_lives_kimg": "ema_half_lives_kimg",
            "ema_update_after_step": "ema_update_after_step",
            "ema_update_every": "ema_update_every",
        }
        for key, attribute in protected.items():
            if key not in saved_hyperparameters:
                continue
            saved, current = saved_hyperparameters[key], getattr(self, attribute)
            if key == "ema_half_lives_kimg":
                saved, current = tuple(saved), tuple(current)
            if key == "optimizer":
                saved = str(saved).lower()
            if saved != current:
                raise ValueError(
                    f"resume configuration changed protected training value {key}"
                )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema is not None:
            if self._ema_images_pending <= 0:
                raise RuntimeError("EMA update has no observed training images")
            self.ema.update(self.model, global_images=self._ema_images_pending)
            self._ema_images_pending = 0

    def on_before_optimizer_step(self, optimizer):
        squared_norm = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.detach().float().pow(2).sum()
        grad_norm = squared_norm.sqrt()
        self.log(
            "train/gradient_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def load_ema_state_dict(self, state: Optional[dict]) -> bool:
        if state is None or self.ema is None:
            return False
        self.ema.load_state_dict(state)
        self.ema.move_to(self.model)
        return True

    def ema_scope(self, variant: Optional[str] = None):
        if self.ema is None or not self.ema.ready:
            from contextlib import nullcontext

            return nullcontext()
        return self.ema.scope(self.model, variant)

    def configure_optimizers(self):
        if self.optimizer_name == "muon":
            optimizer = Muon(self.parameters(), lr=self.learning_rate,
                             weight_decay=self.weight_decay, adam_b2=self.adam_b2)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay,
                                          betas=(0.9, self.adam_b2))
        else:
            raise ValueError("optimizer must be 'muon' or explicit legacy 'adamw'")

        def schedule(step: int) -> float:
            if step < self.warmup_steps:
                return float(step + 1) / float(max(self.warmup_steps, 1))
            if self.lr_schedule != "constant":
                raise ValueError("faithful pMF uses constant LR after optional warmup")
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
