"""Separate FD-only post-training: fresh source weights or exact FD resume."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import yaml

from eval.checkpoint import load_model
from src.fd_module import FDColorizerModule, file_sha256, make_data, train_identity, validate_config
from train import build_logger


def source_model(config, *, progress=None, hash_progress=None):
    source = config["source"]
    if progress is not None:
        progress(f"Loading source checkpoint: {source['checkpoint']} "
                 f"(weights={source.get('weights', 'ema')}, EMA={source.get('ema_variant')})")
    model, provenance = load_model(Path(source["checkpoint"]),
                                    use_ema=source.get("weights", "ema") == "ema",
                                    ema_variant=source.get("ema_variant"))
    if provenance["weights"] != source.get("weights", "ema"):
        raise ValueError("requested source EMA is missing; choose raw explicitly")
    if progress is not None:
        progress(f"Weights loaded: step={provenance['global_step']}, "
                 f"noise_scale={provenance['noise_scale']}; computing checkpoint SHA256")
    provenance["sha256"] = file_sha256(source["checkpoint"], progress=hash_progress)
    if list(model.resolution) != config["data"]["resolution"]:
        raise ValueError("source checkpoint and data resolutions differ")
    if provenance["resize_strategy"] != config["data"].get("resize_strategy", "center_crop"):
        raise ValueError("source and FD geometry differ")
    return model, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fd_inception_posttrain.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Absolute stop step; leaves the configured cosine budget intact")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    validate_config(config)
    stop_step = args.stop_after if args.stop_after is not None else config["training"]["max_steps"]
    if not 0 < stop_step <= config["training"]["max_steps"]:
        raise ValueError("--stop-after must be within the configured step budget")
    pl.seed_everything(config["seed"], workers=True)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if "fd_state" not in saved:
            raise ValueError("--resume accepts only FD checkpoints; use source.checkpoint for base weights")
        if saved["hyper_parameters"]["config"] != config:
            raise ValueError("resume requires the saved FD config, including the original schedule budget")
        if saved["global_step"] >= stop_step:
            print(f"FD checkpoint already reached step {saved['global_step']} (requested {stop_step})")
            return
        module = FDColorizerModule(config, saved["hyper_parameters"]["provenance"])
        del saved
    else:
        model, provenance = source_model(config)
        module = FDColorizerModule(config, provenance)
        module.initialize(model, config["fd"]["stats_dir"], train_identity(make_data(config)))
        del model
    training = config["training"]
    devices = training.get("devices", 1)
    multi = devices == -1 or (devices > 1 if isinstance(devices, int) else len(devices) > 1)
    checkpoint = ModelCheckpoint(dirpath=training["checkpoint_dir"], filename="fd-{step:06d}",
                                 every_n_train_steps=training.get("checkpoint_every_n_steps", 100),
                                 save_last=False, save_top_k=1, save_on_train_epoch_end=False)
    trainer = pl.Trainer(accelerator=training.get("accelerator", "gpu"), devices=devices,
                         strategy="ddp" if multi else "auto", precision=training.get("precision", "bf16-mixed"),
                         max_steps=stop_step, max_epochs=-1,
                         accumulate_grad_batches=1, use_distributed_sampler=False,
                         num_sanity_val_steps=0, limit_val_batches=0, callbacks=[checkpoint],
                         logger=build_logger(training), log_every_n_steps=1, enable_model_summary=False)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    trainer.fit(module, ckpt_path=args.resume, weights_only=False)
    if trainer.is_global_zero and torch.cuda.is_available():
        print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB; "
              f"reserved: {torch.cuda.max_memory_reserved()/2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
