#!/usr/bin/env python
"""Minimal Qwen-Image-2.1 image-editing test.

Example:

    .venv_diff/bin/python -m experiments.colorization_losses.qwen_image_edit \
        --input /path/to/input.jpg \
        --output /tmp/qwen_result.png

The RGB input is converted to a grayscale RGB image before it is given to
Qwen. The script saves Qwen's raw colorized image and does not apply Lab
conversion or source-L replacement.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# The complete Qwen-Image-2.1 download is already stored here on this machine.
# Set this before importing/loading the pipeline so Hugging Face uses it instead
# of starting a second download in ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", "/tmp/qwen_hf_cache")

import torch
from PIL import Image, ImageOps
from diffusers import QwenImage21Pipeline


MODEL = "Qwen/Qwen-Image-2.1"
CACHE_DIR = "/tmp/qwen_hf_cache/hub"

# Shared with the benchmark generator so prompt changes cannot silently diverge.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.colorization_losses.dataset import PLAUSIBLE_PROMPT as PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="input image; it is converted to grayscale before Qwen sees it",
    )
    parser.add_argument("--output", required=True, type=Path, help="output PNG/JPG path")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--size", type=int, default=256, help="square editing size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL} on {device} with {dtype}")

    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL,
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
    )
    if device == "cuda":
        # Needed for the 24 GiB GPU setup: the full model does not fit when
        # all modules are kept on the GPU at once.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    # pipe.to(device)
    with Image.open(args.input) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (args.size, args.size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    # Qwen receives only L information, represented as a 3-channel grayscale
    # image because the image-conditioned pipeline expects an RGB-like input.
    image = ImageOps.grayscale(image).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=PROMPT,
        image=image,
        width=args.size,
        height=args.size,
        output_resolution=args.size,
        num_inference_steps=args.steps,
        # true_cfg_scale=4.0,
        # negative_prompt=(
        #     "redrawing, new objects, removed objects, changed geometry, changed composition, "
        #     "global contrast boost, global saturation boost, exposure change, lighting change, "
        #     "tone mapping, cinematic color grading"
        # ),
        generator=generator,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
