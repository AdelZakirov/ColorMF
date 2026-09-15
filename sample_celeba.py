"""Sample a deterministic CelebA subset with native and original-size outputs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from sample import load_module, prepare_input
from src.lab import lab_to_rgb, rgb_to_lab


def read_manifest(filename: str, image_root: str) -> list[tuple[str, Path]]:
    root = Path(image_root)
    entries = []
    for line in Path(filename).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        path_text = line.split("\t", 1)[-1]
        name = Path(path_text).name
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((name, path))
    if not entries:
        raise ValueError(f"manifest is empty: {filename}")
    return entries


def l_to_grayscale(L: torch.Tensor) -> np.ndarray:
    values = ((L[0, 0].float().cpu().numpy() + 1.0) * 127.5)
    return np.repeat(values.clip(0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def save_rgb(filename: Path, image: np.ndarray) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def make_grid(
    records: list[dict],
    filename: Path,
    size_key: str,
    color_key: str,
    sample_seeds: list[int],
) -> None:
    first_image = records[0][size_key]
    height, width = first_image.shape[:2]
    header_height = 24
    canvas = Image.new("RGB", (width * 7, header_height + height * len(records)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["L", "original"] + [f"seed {seed}" for seed in sample_seeds]
    for column, label in enumerate(labels):
        draw.text((column * width + 4, 5), label, fill="black")
    for row, record in enumerate(records):
        images = [
            record[size_key],
            record[f"original_{size_key}"],
            *record[color_key],
        ]
        for column, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (column * width, header_height + row * height))
    filename.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pmf_t_64_colorization.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pmf_t_4_64/last.ckpt")
    parser.add_argument(
        "--manifest",
        default="/mnt/IMAGING/HUB/DATASETS/general_datasets/faces/celeba/celeba.txt",
    )
    parser.add_argument(
        "--image-root",
        default="/mnt/IMAGING/HUB/DATASETS/general_datasets/faces/celeba/256",
    )
    parser.add_argument("--output-root", default="celeba")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=20260915)
    parser.add_argument("--sample-seeds", type=int, nargs=5, default=[1, 2, 3, 4, 5])
    parser.add_argument("--ema-variant", default=None)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("count must be positive")
    entries = read_manifest(args.manifest, args.image_root)
    if args.count > len(entries):
        raise ValueError(f"count {args.count} exceeds manifest size {len(entries)}")
    selected = random.Random(args.selection_seed).sample(entries, args.count)

    module, config, checkpoint = load_module(args.config, args.checkpoint)
    if not args.no_ema:
        module.load_ema_state_dict(checkpoint.get("ema"))
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    ema_variant = args.ema_variant or config["training"].get("ema", {}).get(
        "validation_variant"
    )

    output_root = Path(args.output_root)
    native_root = output_root / "64"
    original_root = output_root / "original_size"
    records = []
    with module.ema_scope(ema_variant):
        for index, (name, path) in enumerate(selected, start=1):
            source_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if source_bgr is None:
                raise FileNotFoundError(path)
            original_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb, resized_L = prepare_input(str(path), module.model.resolution)
            original_L, _ = rgb_to_lab(original_rgb)
            resized_L = resized_L.unsqueeze(0).to(device)
            generated = module.sample(
                resized_L,
                seeds=args.sample_seeds,
                image_ids=[name],
            ).cpu()

            native_L = resized_L.cpu()
            native_colors = list(
                lab_to_rgb(native_L.expand(len(args.sample_seeds), -1, -1, -1), generated)
            )
            original_L = original_L.unsqueeze(0)
            original_size_ab = torch.nn.functional.interpolate(
                generated,
                size=original_L.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            original_colors = list(
                lab_to_rgb(original_L.expand(len(args.sample_seeds), -1, -1, -1), original_size_ab)
            )
            for seed, native_color, original_color in zip(
                args.sample_seeds, native_colors, original_colors
            ):
                save_rgb(native_root / f"seed_{seed}" / name, native_color)
                save_rgb(original_root / f"seed_{seed}" / name, original_color)

            records.append(
                {
                    "name": name,
                    "L_64": l_to_grayscale(native_L),
                    "original_L_64": resized_rgb,
                    "colors_64": native_colors,
                    "L_original_size": l_to_grayscale(original_L),
                    "original_L_original_size": original_rgb,
                    "colors_original_size": original_colors,
                }
            )
            print(f"[{index:02d}/{len(selected):02d}] {name}")

    (output_root / "selected_images.txt").write_text(
        "\n".join(name for name, _ in selected) + "\n", encoding="utf-8"
    )
    make_grid(
        records,
        output_root / "grid_64.png",
        "L_64",
        "colors_64",
        args.sample_seeds,
    )
    make_grid(
        records,
        original_root / "grid.png",
        "L_original_size",
        "colors_original_size",
        args.sample_seeds,
    )
    print(f"saved {len(records)} images with seeds {args.sample_seeds} under {output_root}")


if __name__ == "__main__":
    main()
