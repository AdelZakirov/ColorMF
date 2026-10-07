"""Fixed ImageNet val5k protocol: prepare once, evaluate many checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch

from ._parallel import ordered_map
from .checkpoint import generate
from .config import METRICS, EvalConfig
from .data import ImageRecord, _images, _index, load_rgb
from .fid_cache import DECODER_PROTOCOL, cache_metadata, load_statistics, save_statistics
from .metrics.color import colorfulness
from .metrics.fid import FIDMetric
from .pipeline import evaluate_records
from .reporting import save_report, summary


COUNT = 5000
SIZE = 256
DEFAULT_DATASET = Path("data/imagenet_val5k_256")
PROTOCOL = {
    "name": "imagenet_val5k_center_square_linear256_v1",
    "selection": "ILSVRC2012_val_00000001 through ILSVRC2012_val_00005000, numeric filename order; independent of class folders",
    "decoder": DECODER_PROTOCOL,
    "crop": "side=min(H,W); top=(H-side)//2; left=(W-side)//2; crop before resize",
    "resize": "OpenCV cv2.INTER_LINEAR on uint8 RGB, 256x256; gamma-encoded sRGB",
    "augmentations": "none; no random crop, no flip",
    "storage": "lossless uint8 RGB PNG; GT is the prepared RGB, without a LAB round trip",
    "model_input": "cv2.COLOR_RGB2LAB on prepared uint8 RGB; L_byte / 127.5 - 1",
    "metrics": "prepared RGB GT and saved RGB predictions, no external resize",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """Commit metadata only after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def center_square_linear(rgb: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Maximum center square, then a single linear resize; odd margins floor."""
    height, width = rgb.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    crop = rgb[top:top + side, left:left + side]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), [top, left, side]


def _select(source: Path) -> list[tuple[str, Path]]:
    expected = [f"ILSVRC2012_val_{number:08d}" for number in range(1, COUNT + 1)]
    wanted = set(expected)
    found = {}
    for path in _images(source):
        if path.stem in wanted:
            if path.stem in found:
                raise ValueError(f"duplicate ImageNet validation filename: {path.stem}")
            found[path.stem] = path
    missing = [key for key in expected if key not in found]
    if missing:
        raise ValueError(f"source must contain all first {COUNT} canonical validation filenames; "
                         f"missing {len(missing)} (first: {missing[0]})")
    return [(key, found[key]) for key in expected]


def validate_dataset(dataset: Path, *, workers: int = 4) -> dict:
    """Check protocol, exact membership and content, including before cache reuse."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    expected = [f"ILSVRC2012_val_{number:08d}" for number in range(1, COUNT + 1)]
    if (manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL or
            manifest.get("count") != COUNT or
            [row["image_id"] for row in manifest["images"]] != expected):
        raise ValueError("prepared dataset manifest does not match the fixed val5k protocol")
    images = _index(dataset / "rgb")
    if set(images) != set(expected) or any(path.suffix != ".png" for path in images.values()):
        raise ValueError("prepared GT membership differs from manifest (missing/extra images)")

    def check(row):
        if _sha256(images[row["image_id"]]) != row["rgb_sha256"]:
            raise ValueError(f"prepared GT content changed: {row['image_id']}")
    list(ordered_map(check, manifest["images"], workers))
    return manifest


def prepare_dataset(source: Path, dataset: Path = DEFAULT_DATASET, *,
                    workers: int = 4, progress=print) -> dict:
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    if dataset.is_relative_to(source) or source.is_relative_to(dataset):
        raise ValueError("source and prepared dataset must not overlap")
    if dataset.exists():
        manifest = validate_dataset(dataset, workers=workers)
        if progress:
            progress(f"Verified existing {COUNT}-image dataset: {dataset}")
        return manifest
    selected = _select(source)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    # Interrupted preparation never leaves an apparently complete dataset.
    with tempfile.TemporaryDirectory(dir=dataset.parent, prefix=dataset.name + ".") as staging:
        root = Path(staging)
        (root / "rgb").mkdir()

        def prepare(item):
            key, path = item
            before = path.stat()
            source_hash = _sha256(path)
            original = np.rint(load_rgb(path) * 255).astype(np.uint8)
            rgb, crop = center_square_linear(original)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"source changed during preparation: {path}")
            output = root / "rgb" / f"{key}.png"
            Image.fromarray(rgb).save(output)
            return {"image_id": key, "source": str(path), "source_sha256": source_hash,
                    "source_size_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
                    "decoded_size_hw": list(original.shape[:2]), "crop_top_left_side": crop,
                    "rgb_sha256": _sha256(output),
                    "colorfulness_absolute": colorfulness(rgb.astype(np.float32) / 255, "absolute"),
                    "colorfulness_signed": colorfulness(rgb.astype(np.float32) / 255, "signed")}

        rows = []
        for number, row in enumerate(ordered_map(prepare, selected, workers), 1):
            rows.append(row)
            if progress and (number % 100 == 0 or number == COUNT):
                progress(f"Prepared {number}/{COUNT} RGB images", flush=True)
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                    "protocol": PROTOCOL, "count": COUNT, "source_root": str(source),
                    "versions": {"opencv": cv2.__version__, "pillow": Image.__version__,
                                 "numpy": np.__version__}, "images": rows}
        _write_json(root / "manifest.json", manifest)
        _write_json(root / "gt_summary.json", {
            "count": COUNT, "manifest_sha256": _sha256(root / "manifest.json"),
            "colorfulness": {variant: float(np.mean([row[f"colorfulness_{variant}"] for row in rows]))
                             for variant in ("absolute", "signed")},
            "per_image_values": "manifest.json",
            "fid": "real Inception mean/scatter cached separately; FID distance requires predictions",
        })
        root.rename(dataset)
    return manifest


def fid_cache_path(dataset: Path, device: str, batch_size: int) -> Path:
    # Existing evaluator keys caches by device type and batch size for reproducibility.
    return Path(dataset).resolve() / "stats" / f"fid_real_{torch.device(device).type}_bs{batch_size}.npz"


def prepare_fid(dataset: Path, *, device: str = "cpu", batch_size: int = 32,
                workers: int = 4, progress=print) -> Path:
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    dataset = Path(dataset).resolve()
    validate_dataset(dataset, workers=workers)
    records = [ImageRecord(key, path, ()) for key, path in sorted(_index(dataset / "rgb").items())]
    path = fid_cache_path(dataset, device, batch_size)
    config = EvalConfig(dataset / "rgb", dataset / "rgb", output=dataset / "stats",
                        metrics=("fid",), device=device, batch_size=batch_size, workers=workers)
    metric = FIDMetric(device, batch_size)
    metadata = cache_metadata(records, config, metric)
    if path.exists():
        load_statistics(path, metadata)
        if progress:
            progress(f"Verified GT FID cache: {path}")
        return path
    if progress:
        progress(f"Computing GT FID moments ({device}, batch={batch_size})...", flush=True)

    def images():
        decoded = ordered_map(lambda record: load_rgb(record.ground_truth), records, workers)
        for number, rgb in enumerate(decoded, 1):
            yield rgb
            if progress and (number % (batch_size * 10) == 0 or number == COUNT):
                progress(f"GT Inception features {number}/{COUNT}", flush=True)

    statistics = metric.statistics(images())
    if metadata != cache_metadata(records, config, metric):
        raise ValueError("GT changed during FID extraction; statistics were not saved")
    save_statistics(path, statistics, metadata)
    if progress:
        progress(f"Saved GT FID moments: {path}", flush=True)
    return path


def run_checkpoint(checkpoint: Path, dataset: Path, output: Path, *, device: str = "cpu",
                   batch_size: int = 32, inference_batch_size: int = 4,
                   workers: int = 4, sample_seeds: tuple[int, ...] = (1,),
                   metrics: tuple[str, ...] = METRICS, **generation_options) -> dict:
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest = validate_dataset(dataset, workers=workers)
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("run output and prepared dataset must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("run output must be a new/empty directory")
    config = EvalConfig(output / "predictions", dataset / "rgb", output=output,
                        layout="sample-dirs", sample_ids=tuple(f"seed_{seed}" for seed in sample_seeds),
                        device=device, batch_size=batch_size, workers=workers, metrics=metrics,
                        fid_real_stats=fid_cache_path(dataset, device, batch_size) if "fid" in metrics else None)
    # Validate/create the shared GT cache before spending time on model generation.
    if "fid" in metrics:
        prepare_fid(dataset, device=device, batch_size=batch_size, workers=workers)
    records, metadata = generate(checkpoint, dataset / "rgb", output, resize_strategy="prepared",
                                 sample_seeds=sample_seeds, device=device,
                                 batch_size=inference_batch_size, **generation_options)
    metadata["validation_protocol"] = manifest["protocol"]
    metadata["gt_manifest"] = str(dataset / "manifest.json")
    metadata["gt_manifest_sha256"] = _sha256(dataset / "manifest.json")
    _write_json(output / "generation.json", metadata)
    print("Evaluating val5k predictions...", flush=True)

    def progress(phase, done, total):
        if done % 100 == 0 or done == total:
            print(f"{phase}: {done}/{total}", flush=True)

    report = evaluate_records(records, config, phase_progress=progress)
    report["protocol"]["validation_dataset"] = manifest["protocol"]
    report["inference"] = {key: value for key, value in metadata.items() if key != "timing"}
    report["timing"]["inference"] = metadata["timing"]
    save_report(report, output)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare fixed RGB GT, colorfulness and FID moments")
    prepare.add_argument("--source", type=Path, required=True, help="Original ImageNet validation tree")
    prepare.add_argument("--skip-fid", action="store_true", help="Prepare RGB now; compute FID moments on first run")
    run = commands.add_parser("run", help="Generate and score a checkpoint on the fixed prepared GT")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New/empty run directory")
    run.add_argument("--model-config", type=Path)
    run.add_argument("--sample-seeds", nargs="+", type=int, default=[1])
    run.add_argument("--inference-batch-size", type=int, default=4)
    run.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    run.add_argument("--no-ema", action="store_true")
    run.add_argument("--ema-variant")
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    for subparser in (prepare, run):
        subparser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        subparser.add_argument("--device", default="auto")
        subparser.add_argument("--batch-size", type=int, default=32, help="FID/LPIPS metric batch size")
        subparser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    try:
        if args.command == "prepare":
            prepare_dataset(args.source, args.dataset, workers=args.workers)
            if not args.skip_fid:
                prepare_fid(args.dataset, device=device, batch_size=args.batch_size, workers=args.workers)
            print(f"Prepared dataset: {args.dataset.resolve()}")
        else:
            report = run_checkpoint(
                args.checkpoint, args.dataset, args.output, device=device, batch_size=args.batch_size,
                workers=args.workers, sample_seeds=tuple(args.sample_seeds), metrics=tuple(args.metrics),
                model_config=args.model_config, inference_batch_size=args.inference_batch_size,
                precision=args.precision, use_ema=not args.no_ema, ema_variant=args.ema_variant,
                warmup=args.warmup)
            print(summary(report))
            print(f"Results: {args.output.resolve() / 'report.json'} (+ CSV)")
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as error:
        raise SystemExit(f"ImageNet val5k failed: {error}") from error


if __name__ == "__main__":
    main()
