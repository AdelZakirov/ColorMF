"""One-step conditional pMF sampling from an RGB luminance input."""

from __future__ import annotations

import argparse

import cv2
import torch
import yaml

from src.lab import lab_to_rgb, rgb_to_lab
from src.lightning_module import PMFColorizerModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="use raw checkpoint weights even when EMA weights are available",
    )
    parser.add_argument(
        "--ema-variant",
        default=None,
        help="EDM half-life in kimg (500, 1000, or 2000); defaults to config",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ema_config = config["training"].get("ema") or {}
    checkpoint_ema = checkpoint.get("ema") or {}
    module = PMFColorizerModule(
        model=config["model"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        auxiliary_weight=config["training"]["auxiliary_weight"],
        norm_p=config["training"].get("norm_p", 1.0),
        norm_eps=config["training"].get("norm_eps", 0.01),
        optimizer=config["training"].get("optimizer", "muon"),
        adam_b2=config["training"].get("adam_b2", 0.95),
        lr_schedule=config["training"].get("lr_schedule", "constant"),
        ema_enabled=ema_config.get("enabled", True),
        ema_type=checkpoint_ema.get(
            "ema_type",
            "fixed" if "shadow" in checkpoint_ema else ema_config.get("type", "edm"),
        ),
        ema_half_lives_kimg=checkpoint_ema.get(
            "half_lives_kimg", ema_config.get("half_lives_kimg", [500, 1000, 2000])
        ),
        ema_decay=checkpoint_ema.get("decay", ema_config.get("decay", 0.9999)),
        ema_update_after_step=checkpoint_ema.get(
            "update_after_step", ema_config.get("update_after_step", 0)
        ),
        ema_update_every=checkpoint_ema.get(
            "update_every", ema_config.get("update_every", 1)
        ),
        ema_validation_variant=ema_config.get("validation_variant"),
        # Loss networks are not needed or loaded during standalone inference.
        lpips_enabled=False,
        convnext_enabled=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict)
    if not args.no_ema:
        module.load_ema_state_dict(checkpoint.get("ema"))
    module.eval()
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = module.model.resolution
    if image.shape[:2] != size:
        image = cv2.resize(
            image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC
        )
    L, _ = rgb_to_lab(image)
    L = L.unsqueeze(0)
    ema_variant = args.ema_variant or ema_config.get("validation_variant")
    with module.ema_scope(ema_variant):
        generated = module.model.sample(L, seed=args.seed)
        rgb = lab_to_rgb(L, generated)[0]
    cv2.imwrite(args.output, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
