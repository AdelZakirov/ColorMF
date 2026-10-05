"""Paired first-5k alpha ablation on the pinned epoch-239 Val50k run.

Run from the repository root with the project's inference/evaluation environment:
    .venv/bin/python -m eval.run_alpha_sweep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

import cv2
import torch

from eval.config import EvalConfig
from eval.original_size import colorize
from eval.data import ImageRecord, SampleRecord, _index
from eval.pipeline import evaluate_records
from eval.reporting import save_report, summary

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/mnt/IMAGING/HUB/DATASETS/general_datasets/imagenet/imagenet1k/source/val_set_full")
PRED_ROOT = SOURCE.parent / "val_set_full_cmf"
ALPHA1_PRED = PRED_ROOT / "original_size_50000"
ALPHA13_PRED = PRED_ROOT / "original_size_5000_alpha_1.3"
OUTPUT = PRED_ROOT / "metrics_alpha_val5k"
CHECKPOINT = ROOT / "eval/results/imagenet_val_original_size_50000/checkpoint_epoch239_step1050000.ckpt"
EXPECTED_EPOCH = 239
EXPECTED_STEP = 1_050_000
EXPECTED_SHA256 = "3fc8069cb06f1501438a936e2fd7ac2c69d5fde1cbb6d9adbf8206941c13b9cb"
LOCAL_OUTPUT = ROOT / "eval/results/imagenet_val5k_alpha"
COUNT = 5_000
SAMPLE_IDS = tuple(f"ILSVRC2012_val_{i:08d}" for i in range(1, COUNT + 1))
METRICS = ("fid", "colorfulness", "delta_colorfulness", "psnr")


def check_base_predictions() -> dict:
    manifest = json.loads((ALPHA1_PRED / "generation.json").read_text())
    if (manifest.get("epoch"), manifest.get("global_step")) != (EXPECTED_EPOCH, EXPECTED_STEP):
        raise ValueError("Val50k baseline predictions are not from the pinned epoch-239 checkpoint")
    if manifest.get("seed") != 1 or manifest.get("noise_scale") != 0.25:
        raise ValueError("Val50k baseline sampling settings differ from the recorded run")
    if tuple(manifest.get("selected_image_ids", ()))[:COUNT] != SAMPLE_IDS:
        raise ValueError("The baseline output does not begin with the expected first Val5k IDs")
    if manifest.get("weights") != "ema" or manifest.get("ema_variant") != "500":
        raise ValueError("Baseline must use EMA 500")
    if manifest.get("alpha", 1.0) != 1.0 or manifest.get("checkpoint_sha256") != EXPECTED_SHA256:
        raise ValueError("Baseline alpha/checkpoint SHA256 does not match")
    return manifest


def score(predictions: Path, alpha: float) -> dict:
    config = EvalConfig(
        predictions=predictions,
        ground_truth=SOURCE,
        output=OUTPUT / f"alpha_{alpha:.1f}",
        metrics=METRICS,
        layout="single",
        allow_subset=True,
        device="cuda",
        batch_size=8,
        workers=8,
        colorfulness_variant="absolute",
        fid_sampling="first",
    )
    started = time.monotonic()
    def progress(phase, done, total):
        if done % 1000 == 0 or done == total:
            print(f"alpha={alpha:.1f} {phase}: {done}/{total}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    ground_truth = _index(SOURCE)
    predicted = _index(predictions)
    missing = set(SAMPLE_IDS) - (ground_truth.keys() & predicted.keys())
    if missing:
        raise ValueError(f"Missing paired first-Val5k images: {sorted(missing)[:10]}")
    records = [ImageRecord(key, ground_truth[key], (SampleRecord("single", predicted[key]),))
               for key in SAMPLE_IDS]
    coverage = {"ground_truth_images": len(ground_truth), "evaluated_images": COUNT,
                "predictions": COUNT, "k_min": 1, "k_max": 1,
                "unevaluated_ground_truth_ids": sorted(set(ground_truth)-set(SAMPLE_IDS))}
    report = evaluate_records(records, config, coverage=coverage, phase_progress=progress)
    report["generation"] = {"checkpoint_sha256": EXPECTED_SHA256, "epoch": EXPECTED_EPOCH,
                            "global_step": EXPECTED_STEP, "alpha": alpha,
                            "alpha_space": "physical CIELAB a*/b* around zero",
                            "ema_variant": "500", "seed": 1, "noise_scale": 0.25}

    if report["coverage"]["evaluated_images"] != COUNT or report["fid"]["real_count"] != COUNT or report["fid"]["generated_count"] != COUNT:
        raise ValueError("Expected exactly 5,000 paired GT/prediction images")
    save_report(report, config.output)
    save_report(report, LOCAL_OUTPUT / f"alpha_{alpha:.1f}")
    print(f"alpha={alpha:.1f}\n{summary(report)}", flush=True)
    return report


def main() -> None:
    global SOURCE, ALPHA1_PRED, ALPHA13_PRED, OUTPUT, CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--baseline", type=Path, default=ALPHA1_PRED)
    parser.add_argument("--predictions", type=Path, default=ALPHA13_PRED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evaluate-only", action="store_true", help="Validate and score a completed matching alpha=1.3 generation")
    args = parser.parse_args()
    SOURCE, ALPHA1_PRED, ALPHA13_PRED, OUTPUT, CHECKPOINT = (
        args.source, args.baseline, args.predictions, args.output, args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    baseline_manifest = check_base_predictions()
    checkpoint_sha256 = hashlib.file_digest(CHECKPOINT.open("rb"), "sha256").hexdigest()
    if checkpoint_sha256 != EXPECTED_SHA256:
        raise ValueError("Pinned checkpoint SHA256 changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"CUDA: {torch.cuda.get_device_name()}; checkpoint SHA256 verified", flush=True)
    # Check alpha=1 remains identical to the stored baseline before the full run.
    with tempfile.TemporaryDirectory(prefix="colormf_alpha_check_") as temporary:
        colorize(CHECKPOINT, SOURCE, Path(temporary)/"alpha1", limit=8,
                 alpha=1.0, seed=1, device="cuda", batch_size=4,
                 use_ema=True, ema_variant="500", workers=4, progress=None)
        for image_id in SAMPLE_IDS[:8]:
            actual = np.asarray(Image.open(Path(temporary)/"alpha1"/f"{image_id}.png"))
            baseline = np.asarray(Image.open(ALPHA1_PRED/f"{image_id}.png"))
            np.testing.assert_array_equal(actual, baseline)
    print("CUDA regression passed: alpha=1 pixels match the Val50k baseline on 8 images.", flush=True)
    started = time.monotonic()
    def progress(message):
        done = int(message.split()[1].split("/")[0])
        if done % 500 == 0 or done == COUNT:
            print(f"alpha=1.3 {message}; elapsed={time.monotonic()-started:.1f}s", flush=True)


    if args.evaluate_only:
        generation = json.loads((ALPHA13_PRED/"generation.json").read_text())
        expected = {"epoch": EXPECTED_EPOCH, "global_step": EXPECTED_STEP,
                    "alpha": 1.3, "alpha_space": "physical CIELAB a*/b* around zero",
                    "checkpoint_sha256": EXPECTED_SHA256, "seed": 1,
                    "weights": "ema", "ema_variant": "500", "noise_scale": .25}
        for key,value in expected.items():
            if generation.get(key) != value:
                raise ValueError(f"Cannot reuse generation: {key} differs")
        if tuple(generation["selected_image_ids"]) != SAMPLE_IDS or set(_index(ALPHA13_PRED)) != set(SAMPLE_IDS):
            raise ValueError("Cannot reuse generation: image IDs differ")
    elif not ALPHA13_PRED.exists():
        generation = colorize(
            CHECKPOINT, SOURCE, ALPHA13_PRED,
            limit=COUNT, offset=0, alpha=1.3, seed=1,
            device="cuda", batch_size=4, use_ema=True, ema_variant="500", workers=4, progress=progress,
        )
        if (generation["epoch"], generation["global_step"]) != (EXPECTED_EPOCH, EXPECTED_STEP):
            raise ValueError("Generated alpha=1.3 outputs came from the wrong checkpoint")
        generation["checkpoint_sha256"] = checkpoint_sha256
        (ALPHA13_PRED / "generation.json").write_text(json.dumps(generation, indent=2) + "\n")
    else:
        raise FileExistsError(f"Refusing to reuse a possibly stale output directory: {ALPHA13_PRED}")

    def verify(image_id):
        with Image.open(SOURCE/f"{image_id}.JPEG") as original, Image.open(ALPHA13_PRED/f"{image_id}.png") as generated:
            assert ImageOps.exif_transpose(original).size == generated.size and generated.mode == "RGB", image_id
            generated.verify()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, SAMPLE_IDS))
    print("Verified all 5,000 generated PNGs and original dimensions.", flush=True)
    report_1 = score(ALPHA1_PRED, 1.0)
    report_13 = score(ALPHA13_PRED, 1.3)
    results = {"alpha_1.0": report_1, "alpha_1.3": report_13}
    (OUTPUT / "comparison.json").write_text(json.dumps({
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha256,
        "epoch": EXPECTED_EPOCH, "global_step": EXPECTED_STEP,
        "sample_ids": list(SAMPLE_IDS), "baseline_predictions": str(ALPHA1_PRED),
        "baseline_manifest_checkpoint": baseline_manifest.get("checkpoint"),
        "alpha_1.3_predictions": str(ALPHA13_PRED),
        "metrics": list(METRICS), "results": results,
    }, indent=2) + "\n", encoding="utf-8")

    value = {}
    for alpha, report in ((1.0, report_1), (1.3, report_13)):
        agg = report["aggregate"]["mean_over_samples"]
        value[alpha] = {
            "fid": report["fid"]["value"],
            "colorfulness": agg["colorfulness"],
            "delta_cf_per_image": agg["delta_colorfulness"],
            "delta_cf_means": agg["delta_mean_colorfulness"],
            "psnr": agg["psnr"],
        }
    lines = [
        "# ColorMF Val5k chroma scaling",
        "",
        f"Same first {COUNT} ImageNet Val IDs, epoch {EXPECTED_EPOCH}/step {EXPECTED_STEP}, EMA 500, seed 1, noise 0.25.",
        "Alpha scales physical CIELAB a* and b* around zero before Lab-to-RGB conversion; original L is unchanged.",
        "Baseline alpha=1.0 reuses the first 5,000 predictions from the complete epoch-239 Val50k run.",
        "",
        "| Alpha | FID ↓ | CF | ΔCF per image | ΔCF of means | PSNR ↑ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for alpha in (1.0, 1.3):
        m = value[alpha]
        lines.append(f"| {alpha:.1f} | {m['fid']:.6f} | {m['colorfulness']:.6f} | "
                     f"{m['delta_cf_per_image']:.6f} | {m['delta_cf_means']:.6f} | {m['psnr']:.6f} |")
    lines += [
        "",
        "FID uses the same evaluator for both variants. ΔCF per image is mean absolute per-image CF difference; ΔCF of means is the absolute difference between dataset mean CF values (the paper-comparison convention).",
        "",
        f"Checkpoint SHA256: `{checkpoint_sha256}`.",
        "",
        f"GT mean colorfulness: {report_1['aggregate']['mean_over_samples']['colorfulness_gt']:.6f}.",
        "The earlier Val5k paper-comparison run used epoch 234; its alpha=1 scores are not the baseline for this epoch-239 paired ablation.",
    ]
    text = "\n".join(lines) + "\n"
    (OUTPUT / "comparison.md").write_text(text, encoding="utf-8")
    local_report = LOCAL_OUTPUT / "comparison.md"
    local_report.write_text(text, encoding="utf-8")
    (LOCAL_OUTPUT / "comparison.json").write_text((OUTPUT / "comparison.json").read_text(), encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
