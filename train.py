"""Train conditional pMF colorization with Lightning-native DDP."""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.data import PaletteDataModule
from src.lightning_module import PMFColorizerModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    pl.seed_everything(config.get("seed", 1234), workers=True)

    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    module = PMFColorizerModule(
        model=model_config,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_steps"],
        max_steps=training.get("max_steps"),
        auxiliary_weight=training["auxiliary_weight"],
        fixed_validation_ids=config.get("validation", {}).get("image_ids", []),
        sample_dir=training.get("sample_dir", "qualitative"),
    )
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
        gradient_clip_val=training["gradient_clip_val"],
        callbacks=[checkpoint],
        logger=CSVLogger(training.get("log_dir", "logs")),
        sync_batchnorm=training.get("sync_batchnorm", False),
        use_distributed_sampler=True,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
