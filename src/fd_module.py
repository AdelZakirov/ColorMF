"""FD-only ColorMF post-training using the unmodified upstream FD core."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from eval.metrics.fid import FID_EXTRACTOR, create_fid_network, extract_fid_features
from .data import PaletteDataset, _paths_from_root
from .model import PixelMeanFlowB
from .perceptual import normalized_lab_to_rgb
from third_party.fd_loss.losses import (compute_frechet_distance_loss, diff_all_gather,
                                       precompute_sigma_ref_sqrt)
from third_party.fd_loss.queue import FeatureQueue

UPSTREAM_SHA = "5c03b8112fec8b9432631e4ce053c0d918cc24bc"
RGB_SURROGATE = "continuous LAB: clamp ab encoding to [-1,1], Kornia clip=False, clamp RGB [0,1]; no quantization"


def fd_rgb(L, ab):
    """Continuous surrogate for the evaluator's LAB-byte and RGB gamut clipping.

    Preserve source L. Saturated chroma/RGB has zero clipping gradient, as in
    a hard range adapter; uint8 rounding/quantization is deliberately omitted.
    The existing perceptual converter remains unchanged.
    """
    return normalized_lab_to_rgb(L.float(), ab.float().clamp(-1, 1)).clamp(0, 1)


def file_sha256(path, progress=None):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        total = Path(path).stat().st_size if progress is not None else 0
        completed = 0
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            completed += len(block)
            if progress is not None:
                progress(completed, total)
    return digest.hexdigest()


def train_identity(dataset, progress=None):
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        image_id, path = dataset._item(index)
        digest.update(f"{image_id}\t{Path(path).absolute()}\n".encode())
        if progress is not None and ((index + 1) % 10000 == 0 or index + 1 == len(dataset)):
            progress(index + 1, len(dataset))
    return {"split": "train", "dataset_size": len(dataset),
            "index_sha256": digest.hexdigest(), "resolution": list(dataset.size),
            "resize_strategy": dataset.resize_strategy, "horizontal_flip": False,
            "decoder": "OpenCV IMREAD_COLOR/BGR2RGB", "extractor": FID_EXTRACTOR}


def make_data(config):
    if config["data"].get("horizontal_flip", False):
        raise ValueError("FD v1 requires horizontal_flip=false for deterministic statistics/resume")
    data = config["data"]
    if data.get("train_manifest"):
        inputs = {"manifest": data["train_manifest"]}
    elif data.get("train_root"):
        inputs = {"paths": sorted(_paths_from_root(data["train_root"]))}
    else:
        raise ValueError("FD requires a train_manifest or train_root")
    return PaletteDataset(**inputs, size=tuple(data["resolution"]), train=True,
                          horizontal_flip=False,
                          resize_strategy=data.get("resize_strategy", "center_crop"))


def validate_config(config):
    training = config["training"]
    if training.get("accumulate_grad_batches", 1) != 1:
        raise ValueError("FD v1 does not support gradient accumulation")
    if training.get("precision", "bf16-mixed") not in ("32-true", "bf16-mixed"):
        raise ValueError("FD supports 32-true or bf16-mixed (no FP16 scaler)")
    steps, warmup = training["max_steps"], training["warmup_steps"]
    if not isinstance(steps, int) or not isinstance(warmup, int) or not 0 <= warmup < steps:
        raise ValueError("require integer 0 <= warmup_steps < max_steps")
    if not 0 < config["fd"]["ema_beta"] < 1:
        raise ValueError("feature EMA beta must be in (0,1)")
    if not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")


class StepBatchSampler(Sampler):
    """Deterministic, equal-size rank batches; resume at the consumed batch.

    Each epoch permutes train indices, drops the global partial batch, and
    shards each global batch. No dependency on worker RNG or prefetch position.
    """
    def __init__(self, size, batch_size, rank, world_size, seed, start=0):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed, self.start = rank, world_size, seed, start
        self.batches_per_epoch = size // (batch_size * world_size)
        if self.batches_per_epoch == 0:
            raise ValueError("train dataset must contain at least one global batch")

    def __iter__(self):
        step = self.start
        while True:
            epoch, offset = divmod(step, self.batches_per_epoch)
            indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(
                self.seed + epoch)).tolist()
            for batch in range(offset, self.batches_per_epoch):
                start = (batch * self.world_size + self.rank) * self.batch_size
                yield indices[start:start + self.batch_size]
                step += 1


def all_finite(value):
    """One synchronized validity decision on all ranks; never reduce gradients."""
    flag = value.to(dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def ddp_backward_loss(loss):
    # Upstream gather backward keeps only the local feature chunk. DDP then
    # averages parameter gradients; multiply by W to recover the global sum.
    return loss * (dist.get_world_size() if dist.is_initialized() else 1)


class FDColorizerModule(pl.LightningModule):
    def __init__(self, config, provenance):
        super().__init__()
        validate_config(config)
        self.save_hyperparameters({"config": config, "provenance": provenance,
                                   "model": provenance["model"],
                                   "noise_scale": provenance["noise_scale"]})
        self.config, self.provenance = config, provenance
        self.model = PixelMeanFlowB(**provenance["model"])
        self.model.requires_grad_(True)
        for name, parameter in self.model.named_parameters():
            if name.startswith("v_"):
                parameter.requires_grad_(False)
        self.queue = FeatureQueue(size=50000, feat_dim=2048, ema_beta=config["fd"]["ema_beta"])
        self.register_buffer("mu_ref", torch.zeros(2048, dtype=torch.float64))
        self.register_buffer("sigma_ref", torch.zeros(2048, 2048, dtype=torch.float64))
        self.register_buffer("sigma_ref_sqrt", torch.zeros(2048, 2048, dtype=torch.float64))
        self.automatic_optimization = False
        # External frozen extractor is rebuilt, not copied into each checkpoint.
        object.__setattr__(self, "extractor", None)
        self.consumed_batches = 0
        self.skipped_batches = 0
        self.pending_rng = None
        self.noise_generator = None

    def initialize(self, model, stats_dir, identity):
        self.model.load_state_dict(model.state_dict(), strict=True)
        with np.load(Path(stats_dir) / "real.npz", allow_pickle=False) as real:
            metadata = json.loads(str(real["metadata"]))
            if metadata["identity"] != identity or metadata["covariance_ddof"] != 1:
                raise ValueError("real stats train split/preprocessing/extractor mismatch")
            self.mu_ref.copy_(torch.from_numpy(real["mu"]))
            self.sigma_ref.copy_(torch.from_numpy(real["sigma"]))
        generated = torch.load(Path(stats_dir) / "generated.pt", map_location="cpu", weights_only=False)
        if generated["metadata"]["identity"] != identity:
            raise ValueError("generated stats train split/preprocessing/extractor mismatch")
        if generated["metadata"]["provenance"] != self.provenance:
            raise ValueError("generated EMA was initialized with different source weights")
        if generated["metadata"]["rgb_surrogate"] != RGB_SURROGATE:
            raise ValueError("generated RGB surrogate mismatch")
        self.queue.load_state_dict(generated["queue"], strict=True)
        if self.queue._ema_count.item() < 2:
            raise ValueError("generated EMA initialization needs at least two samples")
        self.stats_metadata = {"real": metadata, "generated": generated["metadata"]}

    def on_fit_start(self):
        object.__setattr__(self, "extractor", create_fid_network(self.device))
        # Reuse the upstream square root, once on device; moments stay FP64.
        if not torch.count_nonzero(self.sigma_ref_sqrt):
            with torch.autocast(self.device.type, enabled=False):
                self.sigma_ref_sqrt.copy_(precompute_sigma_ref_sqrt(self.sigma_ref))

    def on_train_start(self):
        self.noise_generator = torch.Generator(device=self.device).manual_seed(
            self.config["seed"] + self.global_rank)
        if self.pending_rng:
            if len(self.pending_rng) != self.trainer.world_size:
                raise ValueError("resume requires the same world size for rank RNG/batch ordering")
            state = self.pending_rng[self.global_rank]
            self.noise_generator.set_state(state["noise"].cpu())
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)
            self.pending_rng = None

    def train_dataloader(self):
        data = make_data(self.config)
        if train_identity(data) != self.stats_metadata["real"]["identity"]:
            raise ValueError("training data differs from saved statistics")
        sampler = StepBatchSampler(len(data), self.config["data"]["batch_size"],
                                   self.global_rank, self.trainer.world_size, self.config["seed"],
                                   self.consumed_batches)
        return DataLoader(data, batch_sampler=sampler,
                          num_workers=self.config["data"].get("num_workers", 0),
                          pin_memory=self.device.type == "cuda")

    def _skip(self, optimizer, zero_loss=None):
        # Complete the reducer even when forward statistics failed. These
        # gradients are discarded on every rank before any optimizer update.
        if zero_loss is not None:
            self.manual_backward(zero_loss)
        optimizer.zero_grad(set_to_none=True)
        self.skipped_batches += 1
        self.log("fd/skipped_batches", float(self.skipped_batches), on_step=True)
        if self.skipped_batches >= self.config["training"].get("max_invalid_batches", 10):
            raise RuntimeError("too many nonfinite FD batches; aborting before a long run")

    def training_step(self, batch, batch_idx):
        started = perf_counter()
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        self.consumed_batches += 1
        L = batch["L"]
        noise = torch.randn((len(L), 2, *self.model.resolution), device=L.device,
                            generator=self.noise_generator) * self.provenance["noise_scale"]
        ab = self.model.sample_from_noise(L, noise)
        # Keep converter and the *same* FID extractor in FP32, even with BF16
        # generator sampling; all moment/matrix operations outside autocast.
        with torch.autocast(self.device.type, enabled=False):
            features = extract_fid_features(self.extractor, fd_rgb(L, ab))
            if not all_finite(torch.isfinite(features).all()):
                self._skip(optimizer, features.nan_to_num().sum() * 0)
                return
            gathered = diff_all_gather(features)
            mu, sigma = self.queue.build_feats_stats(gathered)
            fd = None
            if all_finite(torch.isfinite(mu).all() & torch.isfinite(sigma).all()):
                try:
                    fd = compute_frechet_distance_loss(
                        self.mu_ref, self.sigma_ref, mu=mu, sigma=sigma,
                        sigma_ref_sqrt=self.sigma_ref_sqrt)
                except torch.linalg.LinAlgError:
                    pass
            valid = fd is not None and fd.requires_grad
            flag = torch.tensor(valid, device=L.device)
            if valid:
                flag = flag & torch.isfinite(fd) & (fd >= 0)
            if not all_finite(flag):
                self._skip(optimizer, features.sum() * 0)
                return
            loss = fd / (fd.detach() + 0.01)
        self.manual_backward(ddp_backward_loss(loss))
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        valid_grads = torch.stack([torch.isfinite(p.grad).all() if p.grad is not None
                                   else torch.tensor(False, device=L.device) for p in parameters]).all()
        if not all_finite(valid_grads):
            self._skip(optimizer)
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
        if not all_finite(torch.isfinite(grad_norm)):
            self._skip(optimizer)
            return
        optimizer.step()
        self.lr_schedulers().step()
        # Exactly one global feature EMA update per accepted optimizer step.
        # No update on invalid loss/gradient; use pre-update generated features.
        with torch.autocast(self.device.type, enabled=False):
            self.queue.enqueue(gathered.detach())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.log_dict({"fd/step_seconds": perf_counter() - started,
                       "fd/raw": fd.detach(), "fd/loss": loss.detach(),
                       "fd/grad_norm": grad_norm, "fd/lr": optimizer.param_groups[0]["lr"]},
                      on_step=True, on_epoch=False, batch_size=len(L))

    def configure_optimizers(self):
        training = self.config["training"]
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                      lr=training["learning_rate"], betas=(0.9, 0.95), weight_decay=0)
        def schedule(step):
            warmup, budget = training["warmup_steps"], training["max_steps"]
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / (budget - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return [optimizer], [scheduler]

    def on_train_end(self):
        # Save while CUDA/rank state is still active. Strategy.teardown moves
        # the model/optimizer to CPU after fit, losing self.device's CUDA RNG.
        if self.trainer.checkpoint_callback is not None:
            self.trainer.save_checkpoint(
                str(Path(self.config["training"]["checkpoint_dir"]) / "last.ckpt"),
                weights_only=False)

    def on_save_checkpoint(self, checkpoint):
        state = {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state(), "noise": self.noise_generator.get_state(),
                 "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        rng = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(rng, state)
        else:
            rng[0] = state
        checkpoint["fd_state"] = {"rng": rng, "consumed_batches": self.consumed_batches,
                                  "skipped_batches": self.skipped_batches,
                                  "stats_metadata": self.stats_metadata, "upstream_sha": UPSTREAM_SHA}
        checkpoint["datamodule_hyper_parameters"] = self.config["data"]
        # Source weight EMA is provenance only. These model.* weights are raw
        # post-trained weights; never attach the stale source EMA to this file.

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint["hyper_parameters"]
        if saved["config"] != self.config or saved["provenance"] != self.provenance:
            raise ValueError("FD resume config/provenance mismatch; use the saved recipe")
        state = checkpoint["fd_state"]
        if state["upstream_sha"] != UPSTREAM_SHA:
            raise ValueError("FD upstream commit mismatch")
        self.consumed_batches, self.skipped_batches = state["consumed_batches"], state["skipped_batches"]
        self.stats_metadata, self.pending_rng = state["stats_metadata"], state["rng"]
