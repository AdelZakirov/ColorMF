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
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    module = PMFColorizerModule(
        model=config["model"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        auxiliary_weight=config["training"]["auxiliary_weight"],
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict)
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
    generated = module.model.sample(L, seed=args.seed)
    rgb = lab_to_rgb(L, generated)[0]
    cv2.imwrite(args.output, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()

