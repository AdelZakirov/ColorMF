"""Colorize full images at their original size: python -m eval.original_size."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .checkpoint import load_model
from .data import _index, load_rgb
from src.lab import lab_to_rgb, normalize_ab


def prepare_luminance(image: np.ndarray, resolution: tuple[int, int]):
    """Extract original L, resize its byte encoding, then normalize for the model."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("expected HWC uint8 RGB")
    encoded = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0]
    height, width = resolution
    resized = cv2.resize(encoded, (width, height), interpolation=cv2.INTER_CUBIC)
    original_L = torch.from_numpy(encoded.copy()).float()[None, None] / 127.5 - 1
    model_L = torch.from_numpy(resized.copy()).float()[None] / 127.5 - 1
    return original_L, model_L


def original_size_rgb(original_L: torch.Tensor, generated_ab: torch.Tensor, *,
                      alpha: float = 1.0) -> np.ndarray:
    """Scale predicted ab, upsample it, and combine with untouched original L."""
    if generated_ab.ndim != 4 or generated_ab.shape[1] != 2:
        raise ValueError("generated_ab must be [B,2,H,W]")
    if original_L.ndim != 4 or original_L.shape[1] != 1:
        raise ValueError("original_L must be [B,1,H,W]")
    if original_L.shape[0] != generated_ab.shape[0]:
        raise ValueError("L and ab batch sizes must match")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    scaled_ab = generated_ab.float()
    if alpha != 1.0:
        # Physical a*/b*=0 maps to +0.5/127.5 in the legacy normalization.
        # Scale around that neutral point, preserving the alpha=1 path exactly.
        neutral = normalize_ab(scaled_ab.new_zeros(()))
        scaled_ab = (scaled_ab - neutral) * alpha + neutral
    resized_ab = F.interpolate(scaled_ab,
                               size=original_L.shape[-2:],
                               mode="bicubic", align_corners=False)
    return lab_to_rgb(original_L, resized_ab)


def colorize(checkpoint: Path, source: Path, output: Path, *, limit: int | None = None,
             offset: int = 0, alpha: float = 1.0,
             seed: int = 1, device: str = "cpu", batch_size: int = 4,
             use_ema: bool = True, ema_variant: str | None = None, workers: int = 1,
             progress=print) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size and limit must be positive")
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output trees must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    selected = sorted(_index(source).items())[offset:][:limit]
    if not selected:
        raise ValueError("source must contain images")
    model, metadata = load_model(Path(checkpoint), use_ema=use_ema, ema_variant=ema_variant)
    model.to(device)
    metadata.update(source=str(source), output=str(output), device=str(device), precision="float32",
                    alpha=float(alpha), alpha_space="physical CIELAB a*/b* around zero", seed=seed, batch_size=batch_size, workers=workers, selection_offset=offset,
                    selection_limit=limit, model_resolution=list(model.resolution),
                    selected_image_ids=[key for key, _ in selected],
                    pipeline="original RGB -> original L -> resized L -> normalized L -> predicted ab -> physical chroma scaling -> original-size ab + original L -> RGB",
                    normalization="OpenCV 8-bit LAB channel / 127.5 - 1",
                    luminance_resize="OpenCV INTER_CUBIC on uint8 L, then normalize; full image, no crop",
                    chroma_resize="PyTorch bicubic on continuous normalized ab, align_corners=False",
                    rgb_conversion="OpenCV 8-bit LAB -> RGB; round/clip only at final display boundary")
    # This pass performs full-image luminance resizing rather than the checkpoint's crop.
    metadata.pop("resize_strategy", None)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation_pending.json").write_text(json.dumps(metadata, indent=2) + "\n")
    exports = []
    def prepare(item):
        _, path = item
        rgb = np.rint(load_rgb(path) * 255).astype(np.uint8)
        return prepare_luminance(rgb, model.resolution)

    def export(item):
        key, path, original_L, ab = item
        with torch.inference_mode():
            rgb = original_size_rgb(original_L, ab, alpha=alpha)[0]
        target = output / f"{key}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".png.tmp")
        Image.fromarray(rgb).save(temporary, format="PNG")
        temporary.replace(target)
        return {"image_id": key, "source": str(path), "prediction": str(target),
                "width": rgb.shape[1], "height": rgb.shape[0]}

    with (ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool, \
            torch.inference_mode(), (output / "generation.jsonl").open("w") as journal:
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            prepared = list(pool.map(prepare, batch) if pool else map(prepare, batch))
            originals, inputs = zip(*prepared)
            luminance = torch.stack(inputs).to(device)
            generated = model.sample(luminance, seeds=[seed] * len(batch),
                                     image_ids=[key for key, _ in batch],
                                     noise_scale=metadata["noise_scale"]).cpu()
            if not torch.isfinite(generated).all():
                raise ValueError("model generated nonfinite ab")
            items = [(key, path, original_L, generated[index:index + 1])
                     for index, ((key, path), original_L) in enumerate(zip(batch, originals))]
            rows = list(pool.map(export, items) if pool else map(export, items))
            exports.extend(rows)
            for row in rows:
                journal.write(json.dumps(row) + "\n")
            journal.flush()
            if progress:
                progress(f"Colorized {min(start + batch_size, len(selected))}/{len(selected)} full images")
    metadata["exported_predictions"] = exports
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "generation_pending.json").unlink()
    return metadata


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many sorted image IDs before applying --limit")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scale physical Lab a* and b* around zero before RGB conversion")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="CPU decode/export threads")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-variant")
    args = parser.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    try:
        metadata = colorize(args.checkpoint, args.input, args.output, limit=args.limit,
                            offset=args.offset, alpha=args.alpha,
                            seed=args.seed, device=device, batch_size=args.batch_size,
                            use_ema=not args.no_ema, ema_variant=args.ema_variant, workers=args.workers,
                            progress=lambda message: print(message, flush=True))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        raise SystemExit(f"Original-size colorization failed: {error}") from error
    print(f"Saved {len(metadata['exported_predictions'])} original-size images in {metadata['output']}")


if __name__ == "__main__":
    main()
