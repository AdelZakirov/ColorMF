"""Train conditional pMF colorization with Lightning-native DDP."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from src.data import PaletteDataModule
from src.lightning_module import PMFColorizerModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mlflow-run-id", default=None)
    return parser.parse_args()


def build_logger(config: dict, run_id: str | None = None):
    logger_config = config.get("logger", {})
    logger_type = logger_config.get("type", "csv").lower()
    if logger_type == "mlflow":
        return MLFlowLogger(
            experiment_name=logger_config.get("experiment_name", "tmp"),
            run_name=logger_config.get("run_name"),
            tracking_uri=logger_config.get(
                "tracking_uri", "sqlite:///./mlflow.db"
            ),
            log_model=logger_config.get("log_model", False),
            artifact_location=logger_config.get(
                "artifact_location", "file:./mlartifacts"
            ),
            save_dir=training_log_dir(config),
            run_id=run_id if run_id is not None else logger_config.get("run_id"),
        )
    if logger_type == "csv":
        return CSVLogger(training_log_dir(config))
    raise ValueError(f"unsupported logger type: {logger_type}")


def training_log_dir(config: dict) -> str:
    return config.get("log_dir", "logs")


def validation_trainer_kwargs(training: dict) -> dict:
    interval_steps = training.get("validation_check_interval_steps")
    if interval_steps is None:
        return {}
    if isinstance(interval_steps, bool) or not isinstance(interval_steps, int):
        raise ValueError("validation_check_interval_steps must be a positive integer")
    if interval_steps <= 0:
        raise ValueError("validation_check_interval_steps must be a positive integer")
    accumulation = training.get("accumulate_grad_batches", 1)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("accumulate_grad_batches must be a positive integer")
    return {
        "val_check_interval": interval_steps * accumulation,
        "check_val_every_n_epoch": None,
    }


def logging_trainer_kwargs(training: dict) -> dict:
    log_every_n_steps = training.get("log_every_n_steps", 1)
    if (
        isinstance(log_every_n_steps, bool)
        or not isinstance(log_every_n_steps, int)
        or log_every_n_steps <= 0
    ):
        raise ValueError("log_every_n_steps must be a positive integer")
    return {"log_every_n_steps": log_every_n_steps}


def configure_model_compile(module: PMFColorizerModule, training: dict) -> None:
    compile_mode = training.get("compile_mode")
    if compile_mode is None:
        return
    if not isinstance(compile_mode, str) or not compile_mode:
        raise ValueError("compile_mode must be a non-empty string or null")
    if not hasattr(torch, "compile"):
        raise RuntimeError("the installed PyTorch does not provide torch.compile")
    module.model.forward = torch.compile(module.model.forward, mode=compile_mode)
    module.model.auxiliary_direction = torch.compile(
        module.model.auxiliary_direction, mode=compile_mode
    )


def _git_value(*arguments: str, fallback: str = "unknown") -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return fallback
    return result.stdout.strip() or fallback


def _tag_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def log_mlflow_metadata(logger, config: dict, module: PMFColorizerModule) -> None:
    if not isinstance(logger, MLFlowLogger):
        return
    data = config.get("data", {})
    training = config.get("training", {})
    tags = {
        "runtime/mlflow_version": mlflow.__version__,
        "runtime/python_version": sys.version.split()[0],
        "runtime/pytorch_version": torch.__version__,
        "runtime/lightning_version": pl.__version__,
        "runtime/hostname": platform.node(),
        "source/git_commit": _git_value("rev-parse", "HEAD"),
        "source/git_dirty": bool(
            _git_value("status", "--porcelain", fallback="")
        ),
        "model/parameter_report": module.model.parameter_report(),
        "data/train_manifest": data.get("train_manifest"),
        "data/val_manifest": data.get("val_manifest"),
        "data/resolution": data.get("resolution"),
        "data/batch_size": data.get("batch_size"),
        "data/val_batch_size": data.get("val_batch_size"),
        "training/accelerator": training.get("accelerator"),
        "training/devices": training.get("devices"),
        "training/precision": training.get("precision"),
        "training/max_epochs": training.get("max_epochs"),
        "training/max_steps": training.get("max_steps"),
        "training/accumulate_grad_batches": training.get(
            "accumulate_grad_batches"
        ),
        "training/noise_scale": training.get("noise_scale", 1.0),
        "training/ema": training.get("ema", {}),
        "sampling/time_sampling": training.get("time_sampling", {}),
    }
    for key, value in tags.items():
        logger.experiment.set_tag(logger.run_id, key, _tag_value(value))


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    pl.seed_everything(config.get("seed", 1234), workers=True)

    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    logger_config = training.get("logger", {})
    mlflow_run_id = args.mlflow_run_id or logger_config.get("run_id")
    validation = config.get("validation", {})
    ema = training.get("ema") or {}
    time_sampling = training.get("time_sampling", {})
    module = PMFColorizerModule(
        model=model_config,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_steps"],
        max_steps=training.get("max_steps"),
        optimizer=training.get("optimizer", "muon"),
        adam_b2=training.get("adam_b2", 0.95),
        lr_schedule=training.get("lr_schedule", "constant"),
        auxiliary_weight=training["auxiliary_weight"],
        noise_scale=training.get("noise_scale", 1.0),
        norm_p=training.get("norm_p", 1.0),
        norm_eps=training.get("norm_eps", 0.01),
        time_p_mean=time_sampling.get("p_mean", 0.8),
        time_p_std=time_sampling.get("p_std", 0.8),
        time_data_proportion=time_sampling.get("data_proportion", 0.5),
        time_tr_uniform=time_sampling.get("tr_uniform", False),
        time_uniform_probability=time_sampling.get("uniform_probability", 0.1),
        split_diagonal_jvp=training.get("split_diagonal_jvp", False),
        random_seed=config.get("seed", 1234),
        fixed_validation_ids=validation.get("image_ids", []),
        validation_image_count=validation.get("image_count", 4),
        validation_sample_seeds=validation.get("sample_seeds"),
        ema_enabled=ema.get("enabled", True),
        ema_type=ema.get("type", "edm"),
        ema_half_lives_kimg=ema.get("half_lives_kimg", [500, 1000, 2000]),
        ema_decay=ema.get("decay", 0.9999),
        ema_update_after_step=ema.get("update_after_step", 0),
        ema_update_every=ema.get("update_every", 1),
        ema_use_for_validation=ema.get("use_for_validation", True),
        ema_validation_variant=ema.get("validation_variant"),
        lpips_enabled=training.get("perceptual", {}).get("lpips", {}).get("enabled", False),
        lpips_weight=training.get("perceptual", {}).get("lpips", {}).get("weight", 0.4),
        convnext_enabled=training.get("perceptual", {}).get("convnext", {}).get("enabled", False),
        convnext_weight=training.get("perceptual", {}).get("convnext", {}).get("weight", 0.1),
        perceptual_max_t=training.get("perceptual", {}).get("max_t", 0.8),
        sample_dir=training.get("sample_dir", "qualitative"),
    )
    configure_model_compile(module, training)
    datamodule = PaletteDataModule(**data)
    checkpoint = ModelCheckpoint(
        dirpath=training.get("checkpoint_dir", "checkpoints"),
        filename="pmf-{epoch:04d}-{step:08d}",
        save_last=True,
        every_n_epochs=1,
    )
    accelerator = training.get("accelerator", "auto")
    devices = training.get("devices", 1)
    if isinstance(devices, int):
        multi_device = devices == -1 or devices > 1
    else:
        multi_device = len(devices) > 1
    strategy = "ddp" if multi_device else "auto"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=training.get("precision", "bf16-mixed"),
        max_epochs=training["max_epochs"],
        max_steps=training.get("max_steps") or -1,
        accumulate_grad_batches=training.get("accumulate_grad_batches", 1),
        **validation_trainer_kwargs(training),
        **logging_trainer_kwargs(training),
        gradient_clip_val=training["gradient_clip_val"],
        callbacks=[checkpoint],
        logger=build_logger(training, run_id=mlflow_run_id),
        sync_batchnorm=training.get("sync_batchnorm", False),
        use_distributed_sampler=True,
        # Forward-mode AD used by the pMF validation JVP is disabled by
        # torch.inference_mode; Lightning's no-grad validation is sufficient.
        inference_mode=False,
    )
    if trainer.logger is not None:
        if not mlflow_run_id or not isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.log_hyperparams(config)
        if trainer.is_global_zero:
            log_mlflow_metadata(trainer.logger, config, module)
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
