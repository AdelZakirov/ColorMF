#!/usr/bin/env python
"""Generate controlled colorization-error examples.

Examples::

    .venv_diff/bin/python -m experiments.colorization_losses.generate \
        --input path/to/images --output-dir loss_eval/synthetic \
        --types wrong_hue low_saturation color_bleeding subtle_color_error

    .venv_diff/bin/python -m experiments.colorization_losses.generate \
        --input path/to/images --output-dir loss_eval/all --types all \
        --qwen-device cuda

The Qwen variants require the local Qwen-Image-2.1 weights and a device with
enough memory.  Synthetic variants do not load PyTorch or Qwen.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

# The complete Qwen-Image-2.1 download is already cached here on this host.
# This is set before diffusers imports the Hugging Face hub configuration.
os.environ.setdefault("HF_HOME", "/tmp/qwen_hf_cache")

# Allow module execution from the repository root
# without requiring callers to set PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.colorization_losses.dataset import (
    QWEN_VARIANTS,
    SYNTHETIC_VARIANTS,
    compose_with_source_l,
    generate_synthetic,
    prompts_for_qwen,
    requested_variants,
    rgb_to_lab,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _image_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"not a supported image: {path}")
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise ValueError("no input images found")
    return unique


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _load_qwen(
    model_id: str,
    device: str,
    cpu_offload: bool = False,
    sequential_offload: bool = False,
):
    import torch
    from diffusers import QwenImage21Pipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Loading {model_id} on {device} ({dtype})")
    pipe = QwenImage21Pipeline.from_pretrained(
        model_id,
        cache_dir="/tmp/qwen_hf_cache/hub",
        torch_dtype=dtype,
    )
    if sequential_offload and device.startswith("cuda"):
        # Sequential offload moves individual submodules instead of keeping
        # a complete component resident on the GPU. It is slower, but it is
        # the safest mode when another process is using part of the VRAM.
        pipe.enable_sequential_cpu_offload()
    elif cpu_offload and device.startswith("cuda"):
        # The full 2.1 checkpoint does not fit on a 24-GiB card together with
        # its vision encoder and VAE. Accelerate moves each submodule to the
        # GPU only for its forward pass.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe, device, torch


def _qwen_edit(
    pipe,
    torch,
    device: str,
    source: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    resolution: int,
    true_cfg_scale: float,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "image": source,
        "width": resolution,
        "height": resolution,
        "num_inference_steps": steps,
        "generator": generator,
        "output_resolution": resolution,
    }
    kwargs["true_cfg_scale"] = true_cfg_scale
    if true_cfg_scale > 1.0:
        kwargs["true_cfg_scale"] = true_cfg_scale
        kwargs["negative_prompt"] = (
            "redrawing, new objects, removed objects, changed geometry, changed composition, "
            "global contrast boost, global saturation boost, exposure change, lighting change, "
            "tone mapping, cinematic color grading"
        )
    result = pipe(**kwargs)
    image = result.images[0]
    del result
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return image


def _qwen_edit_batch(
    pipe,
    torch,
    device: str,
    source: Image.Image,
    prompts: list[str],
    seed: int,
    steps: int,
    resolution: int,
    true_cfg_scale: float,
) -> list[Image.Image]:
    """Generate several edits for one condition image in one pipeline call.

    Qwen accepts one condition image shared by a prompt batch. Independent
    generators keep each prompt on the same seed convention as the old
    one-at-a-time path while avoiding a second text/image encoding pass.
    """

    generators = [torch.Generator(device=device).manual_seed(seed) for _ in prompts]
    kwargs = {
        "prompt": prompts,
        "image": source,
        "width": resolution,
        "height": resolution,
        "num_inference_steps": steps,
        "generator": generators,
        "output_resolution": resolution,
    }
    if true_cfg_scale > 1.0:
        kwargs["true_cfg_scale"] = true_cfg_scale
        kwargs["negative_prompt"] = [
            "redrawing, new objects, removed objects, changed geometry, changed composition, "
            "global contrast boost, global saturation boost, exposure change, lighting change, "
            "tone mapping, cinematic color grading"
        ] * len(prompts)
    result = pipe(**kwargs)
    images = [image.convert("RGB") for image in result.images]
    del result
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return images


def _safe_stem(path: Path, used: set[str]) -> str:
    stem = path.stem or "image"
    candidate = stem
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="RGB").save(path)


def _manifest_path(path: Path, output_dir: Path) -> str:
    return str(path.resolve().relative_to(output_dir.resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="RGB image(s) or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["all"],
        help="variants to generate; default: all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--qwen-device", default="auto", help="auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--qwen-steps", type=int, default=40)
    parser.add_argument("--qwen-resolution", type=int, default=256)
    parser.add_argument(
        "--qwen-true-cfg-scale",
        type=float,
        default=1.0,
        help="true CFG scale for stronger edit adherence; 1 disables CFG",
    )
    parser.add_argument(
        "--qwen-cpu-offload",
        action="store_true",
        help="offload Qwen submodules to CPU between forward passes (lower VRAM, slower)",
    )
    parser.add_argument(
        "--qwen-sequential-offload",
        action="store_true",
        help="offload Qwen layers sequentially (lowest VRAM, slowest)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing output directory without regenerating recorded files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variants = requested_variants(args.types)
    inputs = _image_paths(args.input)
    if args.qwen_steps < 1 or args.qwen_resolution < 1 or args.qwen_true_cfg_scale < 1:
        raise ValueError("steps/resolution must be positive and CFG must be >= 1")
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    protocol = {
        "version": 2, "seed": args.seed, "model": args.qwen_model,
        "steps": args.qwen_steps, "resolution": args.qwen_resolution,
        "true_cfg_scale": args.qwen_true_cfg_scale, "prompts": prompts_for_qwen(),
        "condition": "PIL_grayscale_RGB", "projection": "source_L_generated_ab_gamut_compression",
        "source_chroma_magnitude_reused": False,
        "batch_per_source": True,
    }
    protocol_path = output_dir / "generation_protocol.json"
    if args.resume:
        if not protocol_path.exists() or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume protocol differs or is missing; use a new output directory")
    elif any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; use matching --resume or a new directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    existing_records: set[tuple[str, str]] = set()
    existing_sample_dirs: dict[str, Path] = {}
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as previous_manifest:
            for line in previous_manifest:
                if not line.strip():
                    continue
                record = json.loads(line)
                input_key = str(Path(record["input"]).resolve())
                required = [output_dir / record["output"]]
                if record["type"] in QWEN_VARIANTS:
                    required += [output_dir / record["raw_output"], output_dir / record["condition_output"]]
                if all(path.exists() for path in required):
                    existing_records.add((input_key, record["type"]))
                if record["type"] == "x_gt":
                    existing_sample_dirs[input_key] = (output_dir / record["output"]).resolve().parent

    qwen_variants = tuple(v for v in variants if v in QWEN_VARIANTS)
    if args.resume:
        qwen_needed = any(
            (str(input_path.resolve()), variant) not in existing_records
            for input_path in inputs
            for variant in qwen_variants
        )
    else:
        qwen_needed = bool(qwen_variants)
    qwen = None
    if qwen_needed:
        qwen = _load_qwen(
            args.qwen_model,
            args.qwen_device,
            cpu_offload=args.qwen_cpu_offload,
            sequential_offload=args.qwen_sequential_offload,
        )
        prompts = prompts_for_qwen()
    else:
        prompts = {}

    used_stems: set[str] = {path.name for path in output_dir.iterdir() if path.is_dir()}
    manifest_mode = "a" if args.resume else "w"
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
        for input_path in inputs:
            input_key = str(input_path.resolve())
            existing_sample_dir = existing_sample_dirs.get(input_key)
            image = _load_rgb(input_path)
            if qwen_variants:
                # Keep x_gt, the condition image, and Qwen outputs in exactly
                # the same 256x256 coordinate system for direct loss checks.
                image = ImageOps.fit(
                    image,
                    (args.qwen_resolution, args.qwen_resolution),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
            source_rgb = np.asarray(image, dtype=np.uint8)
            if existing_sample_dir is not None:
                sample_dir = existing_sample_dir
            else:
                sample_dir = output_dir / _safe_stem(input_path, used_stems)
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_path = sample_dir / "x_gt.png"
            if not source_path.exists():
                image.save(source_path)

            base_record = {
                "input": str(input_path.resolve()),
                "source": _manifest_path(source_path, output_dir),
                "width": int(image.width),
                "height": int(image.height),
            }
            if (input_key, "x_gt") not in existing_records:
                manifest.write(json.dumps({**base_record, "type": "x_gt", "output": _manifest_path(source_path, output_dir)}, ensure_ascii=False) + "\n")
                manifest.flush()
                existing_records.add((input_key, "x_gt"))

            synthetic = generate_synthetic(source_rgb) if any(v in SYNTHETIC_VARIANTS for v in variants) else {}
            qwen_outputs: dict[str, Image.Image] = {}
            if qwen_variants and qwen is not None:
                missing_qwen = [
                    variant for variant in qwen_variants
                    if (input_key, variant) not in existing_records
                    or not (sample_dir / f"{variant}.png").exists()
                ]
                if missing_qwen:
                    pipe, device, torch = qwen
                    qwen_source = ImageOps.grayscale(image).convert("RGB")
                    edited_images = _qwen_edit_batch(
                        pipe,
                        torch,
                        device,
                        qwen_source,
                        [prompts[variant] for variant in missing_qwen],
                        seed=args.seed,
                        steps=args.qwen_steps,
                        resolution=args.qwen_resolution,
                        true_cfg_scale=args.qwen_true_cfg_scale,
                    )
                    qwen_outputs.update(zip(missing_qwen, edited_images))
            for variant in variants:
                output_path = sample_dir / f"{variant}.png"
                if (input_key, variant) in existing_records and output_path.exists():
                    continue
                if variant in SYNTHETIC_VARIANTS:
                    output = synthetic[variant]
                    metadata = {"generation": "deterministic_lab", "source_l_reused": True}
                else:
                    assert qwen is not None
                    pipe, device, torch = qwen
                    # At the requested 256x256 operating point, use the same
                    # square center-crop convention as ColorMF's model input.
                    # This avoids Qwen's low-resolution aspect-ratio padding
                    # mismatch while keeping the actual generation size fixed.
                    # Match the tested standalone script: Qwen sees only the
                    # source luminance, not the ground-truth RGB colors.
                    qwen_source = ImageOps.grayscale(image).convert("RGB")
                    edited = qwen_outputs[variant]
                    raw_dir = sample_dir / "raw"
                    raw_dir.mkdir(exist_ok=True)
                    raw_path = raw_dir / f"{variant}.png"
                    condition_path = raw_dir / "condition.png"
                    edited.save(raw_path)  # Save before any resize or Lab projection.
                    qwen_source.save(condition_path)
                    raw_size = list(edited.size)
                    edited = edited.convert("RGB").resize(qwen_source.size, Image.Resampling.LANCZOS)
                    generated_rgb = np.asarray(edited, dtype=np.uint8)
                    output = compose_with_source_l(source_rgb, generated_rgb)
                    source_lab, raw_lab, final_lab = map(rgb_to_lab, (source_rgb, generated_rgb, output))
                    diagnostics = {
                        "raw_mean_abs_delta_L": float(np.abs(raw_lab[..., 0]-source_lab[..., 0]).mean()),
                        "projected_mean_abs_delta_L": float(np.abs(final_lab[..., 0]-source_lab[..., 0]).mean()),
                        "projected_mean_delta_ab": float(np.linalg.norm(final_lab[..., 1:]-source_lab[..., 1:], axis=-1).mean()),
                        "projection_mean_delta_ab": float(np.linalg.norm(final_lab[..., 1:]-raw_lab[..., 1:], axis=-1).mean()),
                    }
                    metadata = {
                        "generation": "qwen-image-2.1-image-edit",
                        "generation_protocol_version": 2,
                        "raw_output": _manifest_path(raw_path, output_dir),
                        "condition_output": _manifest_path(condition_path, output_dir),
                        "raw_output_size": raw_size,
                        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        "diagnostics": diagnostics,
                        "semantic_label_status": "unverified_candidate",
                        "negative_prompt_enabled": args.qwen_true_cfg_scale > 1.0,
                        "qwen_batch_size": len(qwen_outputs),
                        "qwen_model": args.qwen_model,
                        "qwen_seed": args.seed,
                        "qwen_steps": args.qwen_steps,
                        "qwen_resolution": args.qwen_resolution,
                        "qwen_true_cfg_scale": args.qwen_true_cfg_scale,
                        "qwen_input_size": list(qwen_source.size),
                        "output_size": list(qwen_source.size),
                        "source_l_reused": True,
                        "qwen_condition": "PIL_grayscale_RGB",
                        "source_chroma_magnitude_reused": False,
                        "prompt": prompts[variant],
                    }

                _save_rgb(output, output_path)
                manifest.write(json.dumps({**base_record, **metadata, "type": variant,
                                           "output": _manifest_path(output_path, output_dir)},
                                          ensure_ascii=False) + "\n")
                manifest.flush()
                existing_records.add((input_key, variant))
            print(f"Generated {len(variants)} variants for {input_path} -> {sample_dir}")

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
