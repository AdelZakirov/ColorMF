"""PyTorch Lightning training and rank-safe qualitative validation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from PIL import Image, ImageDraw

from .lab import lab_to_rgb
from .model import PMFTiny
from .pmf import meanflow_terms


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


class PMFColorizerModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model: Optional[dict] = None,
        learning_rate: float = 2.0e-4,
        weight_decay: float = 1.0e-4,
        warmup_steps: int = 1_000,
        max_steps: Optional[int] = None,
        auxiliary_weight: float = 1.0,
        fixed_validation_ids: Optional[Iterable[str]] = None,
        sample_dir: str = "qualitative",
    ):
        super().__init__()
        model_config = model or {}
        self.save_hyperparameters()
        self.model = PMFTiny(**model_config)
        self.auxiliary_weight = auxiliary_weight
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fixed_validation_ids = set(fixed_validation_ids or [])
        self.sample_dir = sample_dir
        self._validation_visuals: Dict[str, tuple] = {}

    def forward(self, z, L, r, t, *, return_velocity: bool = True):
        return self.model(z, L, r, t, return_velocity=return_velocity)

    @torch.no_grad()
    def sample(self, L: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.sample(L, **kwargs)

    @torch.no_grad()
    def sample_lab(self, L: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.sample_lab(L, **kwargs)

    def training_step(self, batch: dict, batch_idx: int):
        started = time.perf_counter()
        terms = meanflow_terms(
            self.model,
            batch["ab"],
            batch["L"],
            auxiliary_weight=self.auxiliary_weight,
        )
        elapsed = max(time.perf_counter() - started, 1.0e-6)
        batch_size = batch["ab"].shape[0]
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
        self.log(
            "train/samples_per_sec",
            batch_size / elapsed,
            on_step=True,
            on_epoch=False,
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

    def on_train_start(self):
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

    def validation_step(self, batch: dict, batch_idx: int):
        terms = meanflow_terms(
            self.model,
            batch["ab"],
            batch["L"],
            auxiliary_weight=self.auxiliary_weight,
        )
        self.log(
            "val/pMF_loss",
            terms.main_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["ab"].shape[0],
        )
        self.log(
            "val/total_loss",
            terms.total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["ab"].shape[0],
        )
        if self.global_rank == 0:
            for index, image_id in enumerate(batch["image_id"]):
                if image_id in self.fixed_validation_ids:
                    self._validation_visuals[image_id] = (
                        batch["L"][index : index + 1].detach().cpu(),
                        batch["ab"][index : index + 1].detach().cpu(),
                    )

    def on_validation_epoch_end(self):
        visuals = self._validation_visuals
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, visuals)
            if self.global_rank == 0:
                visuals = {}
                for rank_visuals in gathered:
                    visuals.update(rank_visuals)
        if self.global_rank != 0:
            self._validation_visuals.clear()
            return
        output_dir = Path(self.trainer.default_root_dir) / self.sample_dir
        for image_id, (L, gt) in visuals.items():
            seeds = list(range(1, 5))
            device = self.device
            generated = self.model.sample(
                L.to(device), seeds=seeds, image_ids=[image_id]
            ).cpu()
            _save_qualitative_row(
                str(output_dir / f"epoch-{self.current_epoch:04d}-{image_id}.png"),
                L,
                gt,
                generated,
            )
        self._validation_visuals.clear()

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        max_steps = self.max_steps
        if max_steps is None and self.trainer is not None:
            max_steps = self.trainer.estimated_stepping_batches
        max_steps = max(max_steps or 1, self.warmup_steps + 1)

        def schedule(step: int) -> float:
            if step < self.warmup_steps:
                return float(step + 1) / float(max(self.warmup_steps, 1))
            progress = (step - self.warmup_steps) / float(
                max(max_steps - self.warmup_steps, 1)
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
