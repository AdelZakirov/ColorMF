"""Generate and evaluate ColorMF predictions: python -m eval.checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
import yaml

from .config import METRICS, EvalConfig
from .data import ImageRecord, SampleRecord, _index, load_rgb
from .pipeline import evaluate_records
from .reporting import save_report, summary


def load_model(checkpoint_path: Path, *, model_config: Path | None = None,
               use_ema: bool = True, ema_variant: str | None = None):
    """Load inference weights only; never instantiate training/loss networks.

    Checkpoints are trusted local PyTorch files (Lightning saves Python objects).
    mmap avoids eagerly copying their optimizer tensors into RAM.
    """
    from src.model import PixelMeanFlowB

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False,
                            mmap=True)
    parameters = checkpoint.get("hyper_parameters", {})
    config = {}
    if model_config is not None:
        with model_config.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("model config must be a YAML mapping")
    architecture = config.get("model", parameters.get("model"))
    if not architecture:
        raise ValueError("checkpoint has no model hyperparameters; provide --model-config")
    noise_scale = float(config.get("training", {}).get(
        "noise_scale", parameters.get("noise_scale", 1.0)))
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("checkpoint noise_scale must be finite and positive")
    ema = checkpoint.get("ema") or {}
    selected_variant = None
    if not use_ema and ema_variant is not None:
        raise ValueError("--ema-variant cannot be used with --no-ema")
    if use_ema and ema:
        shadows = ema.get("shadows")
        if shadows is None and ema.get("shadow") is not None:
            shadows = {"fixed": ema["shadow"]}
        if not shadows or not ema.get("started", ema.get("num_updates", 0) > 0):
            raise ValueError("checkpoint EMA is not ready; use --no-ema explicitly")
        selected_variant = str(ema_variant or config.get("training", {}).get(
            "ema", {}).get("validation_variant") or parameters.get("ema_validation_variant")
                               or next(iter(shadows)))
        if selected_variant not in shadows:
            raise ValueError(f"unknown EMA variant {selected_variant}; choose from {list(shadows)}")
        state = shadows[selected_variant]
    else:
        if ema_variant is not None:
            raise ValueError("checkpoint has no EMA weights")
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.removeprefix("model."): value for key, value in state.items()
                 if key.startswith("model.")} if any(
                     key.startswith("model.") for key in state) else state
    # Normal construction also initializes nonpersistent buffers (RoPE), which
    # cannot be recovered from state_dict. Optimizer tensors stay memory mapped.
    model = PixelMeanFlowB(**architecture)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    strategy = config.get("data", {}).get("resize_strategy", checkpoint.get(
        "datamodule_hyper_parameters", {}).get("resize_strategy", "center_crop"))
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()), "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"), "model": architecture,
        "weights": "ema" if selected_variant is not None else "raw",
        "ema_variant": selected_variant, "noise_scale": noise_scale,
        "resize_strategy": strategy,
    }
    return model, metadata


def prepare_rgb(path: Path, resolution: tuple[int, int], strategy: str) -> np.ndarray:
    """Use the evaluator's decoder and the training/inference spatial transform."""
    if strategy == "stretch":
        rgb = load_rgb(path, resolution, resize_backend="opencv")
        return np.rint(rgb * 255).astype(np.uint8)
    if strategy != "center_crop":
        raise ValueError("resize_strategy must be center_crop or stretch")
    from src.data import adm_center_crop

    if resolution[0] != resolution[1]:
        raise ValueError("ADM center crop requires a square resolution")
    return adm_center_crop(np.rint(load_rgb(path) * 255).astype(np.uint8), resolution[0])


def generate(checkpoint: Path, ground_truth: Path, output: Path, *,
             model_config: Path | None = None, sample_seeds: tuple[int, ...] = (1,),
             device: str = "cpu", batch_size: int = 4, precision: str = "float32",
             use_ema: bool = True, ema_variant: str | None = None,
             resize_strategy: str | None = None, limit: int | None = None,
             warmup: int = 1, progress=print) -> tuple[list[ImageRecord], dict]:
    from src.lab import lab_to_rgb, rgb_to_lab

    ground_truth, output = Path(ground_truth).resolve(), Path(output).resolve()
    checkpoint = Path(checkpoint).resolve()
    if batch_size <= 0 or warmup < 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size/limit must be positive and warmup nonnegative")
    if not sample_seeds or len(set(sample_seeds)) != len(sample_seeds):
        raise ValueError("sample_seeds must be nonempty and unique")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    if output.is_relative_to(ground_truth) or ground_truth.is_relative_to(output):
        raise ValueError("output and source image trees must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("checkpoint output must be empty; choose a new directory to avoid stale samples")
    sources = _index(ground_truth)
    if not sources:
        raise ValueError("ground_truth must contain images")
    selected = sorted(sources.items())[:limit]
    model, metadata = load_model(checkpoint, model_config=model_config,
                                 use_ema=use_ema, ema_variant=ema_variant)
    strategy = resize_strategy or metadata["resize_strategy"]
    if strategy not in ("center_crop", "stretch"):
        raise ValueError("resize_strategy must be center_crop or stretch")
    device = torch.device(device)
    model.to(device)
    metadata.update(device=str(device), precision=precision, resize_strategy=strategy,
                    resolution=list(model.resolution), sample_seeds=list(sample_seeds),
                    source_ground_truth=str(ground_truth), source_image_count=len(sources),
                    selected_image_ids=[key for key, _ in selected],
                    source_paths={key: str(path) for key, path in selected})
    if progress:
        progress(f"Loaded {metadata['weights']} weights (EMA={metadata['ema_variant']}), "
                 f"step={metadata['global_step']}, noise_scale={metadata['noise_scale']}, "
                 f"{model.resolution}, {strategy}, {device}")
    records = []
    for key, source in selected:
        target_path = output / "ground_truth" / f"{key}.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prepare_rgb(source, model.resolution, strategy)).save(target_path)
        samples = tuple(SampleRecord(f"seed_{seed}", output / "predictions" /
                                    f"seed_{seed}" / f"{key}.png") for seed in sample_seeds)
        records.append(ImageRecord(key, target_path, samples))
    jobs = [(record, seed, sample) for record in records
            for seed, sample in zip(sample_seeds, record.samples)]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed, nfe = 0.0, 0
    calls = 0
    with torch.inference_mode():
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            luminance = torch.stack([rgb_to_lab(np.rint(load_rgb(
                record.ground_truth) * 255).astype(np.uint8))[0] for record, _, _ in batch]).to(device)
            seeds = [seed for _, seed, _ in batch]
            ids = [record.image_id for record, _, _ in batch]
            context = (torch.autocast(device.type, dtype=torch.bfloat16)
                       if precision == "bfloat16" else nullcontext())
            with context:
                if start == 0:
                    for _ in range(warmup):
                        model.sample(luminance, seeds=seeds, image_ids=ids,
                                     noise_scale=metadata["noise_scale"])
                synchronize()
                started = perf_counter()
                generated = model.sample(luminance, seeds=seeds, image_ids=ids,
                                         noise_scale=metadata["noise_scale"])
                synchronize()
                elapsed += perf_counter() - started
            nfe += model.last_sample_nfe * len(batch)
            calls += 1
            if not torch.isfinite(generated).all():
                raise ValueError(f"checkpoint generated nonfinite values for {ids}")
            for (_, _, sample), rgb in zip(batch, lab_to_rgb(luminance.cpu(), generated.cpu())):
                sample.path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(sample.path)
            if progress:
                progress(f"Generated {min(start + batch_size, len(jobs))}/{len(jobs)} predictions")
    metadata["timing"] = {
        "status": "measured", "nfe": nfe / len(jobs),
        "latency_seconds": elapsed / len(jobs),
        "throughput_images_per_second": len(jobs) / elapsed,
        "sampling_seconds": elapsed, "generated_images": len(jobs),
        "model_calls": calls, "batch_size": batch_size, "warmup_batches": warmup,
        "scope": "model.sample only; synchronized CUDA; excludes loading, transfer, decoding, PNG writing, metrics; latency is amortized per prediction",
    }
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return records, metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, help="Optional training YAML override; otherwise use checkpoint hyperparameters")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New/empty run directory")
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32, help="Metric batch size")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--resize-strategy", choices=("center_crop", "stretch"), help="Defaults to the checkpoint's data preprocessing")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-variant")
    parser.add_argument("--limit", type=int, help="First N relative image IDs in sorted order")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--fid-sampling", choices=("first", "all"), default="first")
    parser.add_argument("--colorfulness-variant", choices=("absolute", "signed"), default="absolute")
    args = parser.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    try:
        config = EvalConfig(args.output / "predictions", args.output / "ground_truth",
                            output=args.output, metrics=tuple(args.metrics), layout="sample-dirs",
                            sample_ids=tuple(f"seed_{seed}" for seed in args.sample_seeds),
                            device=device, batch_size=args.batch_size, lpips_net=args.lpips_net,
                            fid_sampling=args.fid_sampling, colorfulness_variant=args.colorfulness_variant)
        # Fail before expensive generation if FID cannot run on this selection.
        if "fid" in args.metrics and min(len(_index(args.ground_truth)), args.limit or math.inf) < 2:
            raise ValueError("FID needs at least two ground-truth images")
        records, metadata = generate(
            args.checkpoint, args.ground_truth, args.output, model_config=args.model_config,
            sample_seeds=tuple(args.sample_seeds), device=device, batch_size=args.inference_batch_size,
            precision=args.precision, use_ema=not args.no_ema, ema_variant=args.ema_variant,
            resize_strategy=args.resize_strategy, limit=args.limit, warmup=args.warmup)
        print("Evaluating saved predictions...", flush=True)
        report = evaluate_records(records, config, progress=lambda done, total: print(
            f"Evaluated {done}/{total} images", flush=True))
        report["inference"] = {key: value for key, value in metadata.items() if key != "timing"}
        report["timing"]["inference"] = metadata["timing"]
        save_report(report, config.output)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        raise SystemExit(f"Checkpoint evaluation failed: {error}") from error
    print(summary(report))
    print(f"Results: {config.output / 'report.json'} (+ CSV)")


if __name__ == "__main__":
    main()
