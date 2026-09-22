#!/usr/bin/env python3
"""Evaluate colorization losses, image gradients, and their redundancy.

The runner uses the generated Flickr counterexamples.  Each corrupt image is
also evaluated at several controlled severities by interpolating only its
Lab chroma away from the ground truth while reusing the ground-truth L*.
"""

from __future__ import annotations

import argparse
import csv
import html
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
VENVSITE = ROOT / ".venv" / "lib" / "python3.13" / "site-packages"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if VENVSITE.exists() and str(VENVSITE) not in sys.path:
    # .venv_diff provides transformers 5.17 (DINOv3), while .venv provides
    # the repository's LPIPS and Kornia packages.
    sys.path.append(str(VENVSITE))

from kornia.color import lab_to_rgb
from experiments.colorization_losses.losses import LOSS_NAMES, LossConfig, LossSuite, physical_lab


ERROR_TYPES = (
    "wrong_hue",
    "low_saturation",
    "color_bleeding",
    "subtle_color_error",
    "semantically_wrong_color",
    "plausible_alternative_colorization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--severity-levels", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--convnext-model", default="facebook/convnextv2-base-22k-224")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rgb(path: Path) -> Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def sample_dirs(dataset_dir: Path, max_images: int | None) -> list[Path]:
    result = [path for path in sorted(dataset_dir.iterdir()) if path.is_dir() and (path / "x_gt.png").exists()]
    if max_images is not None:
        result = result[:max_images]
    if not result:
        raise FileNotFoundError(f"No sample directories with x_gt.png found under {dataset_dir}")
    return result


def chroma_endpoint(gt: Tensor, corrupt: Tensor) -> Tensor:
    """Project each chroma ray into RGB gamut at exactly the source L*."""
    lab = physical_lab(gt)
    delta = physical_lab(corrupt)[:, 1:] - lab[:, 1:]
    low = torch.zeros_like(lab[:, :1]); high = torch.ones_like(low)
    for _ in range(24):
        mid = (low + high) / 2
        rgb = lab_to_rgb(torch.cat((lab[:, :1], lab[:, 1:] + mid * delta), 1), clip=False)
        valid = ((rgb >= -1e-6) & (rgb <= 1 + 1e-6)).all(1, keepdim=True)
        low = torch.where(valid, mid, low)
        high = torch.where(valid, high, mid)
    return lab[:, 1:] + low * delta


def severity_image(gt: Tensor, corrupt: Tensor, alpha: float) -> tuple[Tensor, float]:
    lab = physical_lab(gt)
    ab = lab[:, 1:] + alpha * (chroma_endpoint(gt, corrupt) - lab[:, 1:])
    return lab_to_rgb(torch.cat((lab[:, :1], ab), 1), clip=False), float(
        torch.linalg.vector_norm(ab - lab[:, 1:], dim=1).mean())


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_arr -= x_arr.mean()
    y_arr -= y_arr.mean()
    denominator = float(np.linalg.norm(x_arr) * np.linalg.norm(y_arr))
    return float(np.dot(x_arr, y_arr) / denominator) if denominator else float("nan")


def average_ranks(values: Sequence[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_arr, kind="mergesort")
    ranks = np.empty(len(values_arr), dtype=np.float64)
    sorted_values = values_arr[order]
    index = 0
    while index < len(values_arr):
        end = index + 1
        while end < len(values_arr) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_metrics(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["error_type"]].append(row)

    all_type_means: dict[str, dict[str, float]] = defaultdict(dict)
    for error_type, group in grouped.items():
        for loss in LOSS_NAMES:
            all_type_means[loss][error_type] = mean(float(row[loss]) for row in group)
            all_type_means[f"gradient_rms::{loss}"][error_type] = mean(
                float(row[f"gradient_rms_{loss}"]) for row in group)

    loss_denominators = {
        metric: sum(values.values())
        for metric, values in all_type_means.items()
    }
    output = []
    for error_type, group in sorted(grouped.items()):
        for loss in LOSS_NAMES:
            loss_mean = all_type_means[loss][error_type]
            gradient_norm_values = [float(row[f"gradient_norm_{loss}"]) for row in group]
            gradient_rms_values = [float(row[f"gradient_rms_{loss}"]) for row in group]
            gradient_key = f"gradient_rms::{loss}"
            output.append({
                "error_type": error_type,
                "loss": loss,
                "n": len(group),
                "mean_loss": loss_mean,
                "median_loss": median(float(row[loss]) for row in group),
                "mean_gradient_norm": mean(gradient_norm_values),
                "median_gradient_norm": median(gradient_norm_values),
                "mean_gradient_rms": mean(gradient_rms_values),
                "median_gradient_rms": median(gradient_rms_values),
                "loss_share_across_types": loss_mean / loss_denominators[loss] if loss_denominators[loss] else 0.0,
                "gradient_rms_share_across_types": (
                    all_type_means[gradient_key][error_type] / loss_denominators[gradient_key]
                    if loss_denominators[gradient_key] else 0.0
                ),
            })
    return output


def severity_correlations(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["error_type"]].append(row)
    metrics = list(LOSS_NAMES)
    metrics.extend(f"gradient_norm_{loss}" for loss in LOSS_NAMES)
    metrics.extend(f"gradient_rms_{loss}" for loss in LOSS_NAMES)
    targets = ("severity_ab", "severity_level")
    output = []
    for error_type, group in sorted(grouped.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            for target in targets:
                target_values = [float(row[target]) for row in group]
                output.append({
                    "error_type": error_type,
                    "metric": metric,
                    "target": target,
                    "n": len(group),
                    "pearson": pearson(values, target_values),
                    "spearman": spearman(values, target_values),
                })
    return output


def cosine_rows(cosines: dict[str, dict[tuple[str, str], list[float]]]) -> list[dict]:
    output = []
    for scope, pairs in sorted(cosines.items()):
        for (loss_a, loss_b), values in sorted(pairs.items()):
            valid = [v for v in values if math.isfinite(v)]
            output.append({
                "scope": scope,
                "loss_a": loss_a,
                "loss_b": loss_b,
                "n": len(values),
                "n_valid": len(valid),
                "mean_cosine": mean(valid),
                "median_cosine": median(valid),
                "std_cosine": float(np.std(valid)) if valid else float("nan"),
            })
    return output


def recommendation_text(aggregate: list[dict], cosine: list[dict], metadata: dict) -> str:
    return """# Revised colorization loss diagnostic

The previous ConvNeXt-first / DINO-second recommendation is withdrawn.
Huber + gradient is a reasonable baseline, not a validated training optimum.
Feature losses use spatial features; derivatives are with respect to physical ab
at fixed L. Full-image resize is deterministic and is not an exact pMF replication.
Gamut projection shortens chroma rays before evaluation; no RGB hard clipping.

Within-loss shares describe the distribution across error types, not strength
across losses. Raw magnitudes and gradient norms depend on arbitrary weighting.
Low gradient cosine means distinct directions, not useful complementarity.
Zero-gradient cosine and constant correlations are undefined, not zero.

Semantic and plausible variants are paired within each source at equal mean
chroma distance. This controls magnitude but not spatial support, texture,
local color distribution or correctness of generated semantic labels. Human
review and object masks are still needed before claiming semantic understanding.
Paired bootstrap resamples source images, not dependent severity rows.
Interpolation toward GT can itself change whether a color is semantically wrong.
Training ablations on held-out images are required to recommend a loss mixture.

See semantic_paired.csv, semantic_summary.csv, within_image_monotonicity.csv,
and per_sample_metrics.csv. All gradient quantities use physical Lab units.

Configuration:
""" + json.dumps(metadata, indent=2)


def write_html_report(path, aggregate, correlations, cosine, metadata):
    path.write_text('<!doctype html><meta charset="utf-8"><title>Revised loss diagnostic</title>'
                    '<pre style="white-space:pre-wrap">' + html.escape(
                        recommendation_text(aggregate, cosine, metadata)) + '</pre>')


def paired_statistics(rows, output_dir):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row['sample_id'], row['severity_level'])][row['error_type']] = row
    paired = []
    for (sample, level), group in groups.items():
        a, b = group['semantically_wrong_color'], group['plausible_alternative_colorization']
        for loss in LOSS_NAMES:
            paired.append(dict(sample_id=sample, severity_level=level, loss=loss,
                severity_difference=a['severity_ab']-b['severity_ab'],
                difference=a[loss]-b[loss], semantic_greater=float(a[loss]>b[loss])))
    write_csv(output_dir/'semantic_paired.csv', paired, list(paired[0]))
    summaries=[]
    for loss in LOSS_NAMES:
        per_source=defaultdict(list)
        for row in paired:
            if row['loss']==loss: per_source[row['sample_id']].append(row['semantic_greater'])
        values=np.array([mean(v) for v in per_source.values()])
        rng=np.random.default_rng(0)
        boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        summaries.append(dict(loss=loss,n_sources=len(values),semantic_win_fraction=values.mean(),
                              ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    write_csv(output_dir/'semantic_summary.csv',summaries,list(summaries[0]))
    curves=defaultdict(list)
    for row in rows: curves[(row['sample_id'],row['error_type'])].append(row)
    monotonic=[]
    for (sample,error),group in curves.items():
        group=sorted(group,key=lambda r:r['severity_level'])
        for loss in LOSS_NAMES:
            monotonic.append(dict(sample_id=sample,error_type=error,loss=loss,
                spearman=spearman([r['severity_ab'] for r in group],[r[loss] for r in group])))
    write_csv(output_dir/'within_image_monotonicity.csv',monotonic,list(monotonic[0]))


def process_batch(
    batch: list[dict],
    suite: LossSuite,
    device: torch.device,
    cosine_values: dict[str, dict[tuple[str, str], list[float]]],
) -> list[dict]:
    predicted = torch.cat([item["predicted_ab"] for item in batch], dim=0).to(device)
    target = torch.cat([item["target"] for item in batch], dim=0).to(device)
    L = physical_lab(target)[:, :1].detach()
    target_ab = physical_lab(target)[:, 1:].detach()
    predicted.requires_grad_(True)
    values: dict[str, Tensor] = {}
    gradients: dict[str, Tensor] = {}
    for loss_name in LOSS_NAMES:
        value = suite.loss_ab(loss_name, predicted, target_ab, L)
        values[loss_name] = value.detach()
        gradients[loss_name] = torch.autograd.grad(value.sum(), predicted)[0].detach()
        del value

    result = []
    for index, item in enumerate(batch):
        row = {
            "sample_id": item["sample_id"],
            "error_type": item["error_type"],
            "severity_level": item["severity_level"],
            "severity_ab": item["severity_ab"],
        }
        for loss_name in LOSS_NAMES:
            gradient = gradients[loss_name][index:index + 1]
            row[loss_name] = float(values[loss_name][index].cpu())
            row[f"gradient_norm_{loss_name}"] = float(gradient.flatten(1).norm(dim=1).cpu())
            row[f"gradient_rms_{loss_name}"] = float(gradient.pow(2).mean().sqrt().cpu())
        result.append(row)

    for scope, scope_items in (("all", result), (batch[0]["error_type"], result)):
        # A batch is allowed to contain mixed types, so type-scoped cosines are
        # accumulated one item at a time below; the `all` scope uses all rows.
        if scope != "all":
            continue
        for index, _ in enumerate(scope_items):
            for left_index, left in enumerate(LOSS_NAMES):
                left_gradient = gradients[left][index].flatten()
                for right in LOSS_NAMES[left_index + 1:]:
                    right_gradient = gradients[right][index].flatten()
                    denom = left_gradient.double().norm() * right_gradient.double().norm()
                    cosine = float(torch.dot(left_gradient.double(), right_gradient.double()) / denom) if denom > 0 else float("nan")
                    cosine_values[scope][(left, right)].append(float(cosine))
                    cosine_values[result[index]["error_type"]][(left, right)].append(float(cosine))
    del values, predicted, target, gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    levels = [float(value) for value in args.severity_levels.split(",") if value.strip()]
    if not levels or any(level <= 0.0 or level > 1.0 for level in levels):
        raise ValueError("--severity-levels must contain values in (0, 1]")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metrics_path = output_dir / "per_sample_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    directories = sample_dirs(dataset_dir, args.max_images)
    print(f"Loading loss models on {device} for {len(directories)} samples", flush=True)
    config = LossConfig(dino_model=args.dino_model, convnext_model=args.convnext_model)
    suite = LossSuite(
        device=device,
        config=config,
        dino_model=args.dino_model,
        convnext_model=args.convnext_model,
        local_files_only=args.local_files_only,
    )

    cosine_values: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    all_rows: list[dict] = []
    batch: list[dict] = []
    total = len(directories) * len(ERROR_TYPES) * len(levels)
    completed = 0
    for sample_index, sample_dir in enumerate(directories, start=1):
        gt = load_rgb(sample_dir / "x_gt.png").unsqueeze(0)
        gt_lab = physical_lab(gt)
        endpoints = {e: chroma_endpoint(gt, load_rgb(sample_dir / f"{e}.png").unsqueeze(0))
                     for e in ERROR_TYPES}
        semantic_types = ("semantically_wrong_color", "plausible_alternative_colorization")
        distances = {e: float(torch.linalg.vector_norm(endpoints[e]-gt_lab[:,1:],dim=1).mean())
                     for e in ERROR_TYPES}
        matched = min(distances[e] for e in semantic_types)
        for error_type in ERROR_TYPES:
            corrupt_path = sample_dir / f"{error_type}.png"
            if not corrupt_path.exists():
                raise FileNotFoundError(corrupt_path)
            corrupt = load_rgb(corrupt_path).unsqueeze(0)
            for severity_level in levels:
                scale = matched / max(distances[error_type], 1e-12) if error_type in semantic_types else 1.0
                predicted_ab = gt_lab[:,1:] + severity_level * scale * (endpoints[error_type]-gt_lab[:,1:])
                severity_ab = float(torch.linalg.vector_norm(predicted_ab-gt_lab[:,1:],dim=1).mean())
                batch.append({
                    "sample_id": sample_dir.name,
                    "error_type": error_type,
                    "severity_level": severity_level,
                    "severity_ab": severity_ab,
                    "predicted_ab": predicted_ab,
                    "target": gt,
                })
                if len(batch) >= args.batch_size:
                    all_rows.extend(process_batch(batch, suite, device, cosine_values))
                    completed += len(batch)
                    print(f"processed {completed}/{total}", flush=True)
                    batch = []
        del gt
    if batch:
        all_rows.extend(process_batch(batch, suite, device, cosine_values))
        completed += len(batch)
        print(f"processed {completed}/{total}", flush=True)

    metric_fields = ["sample_id", "error_type", "severity_level", "severity_ab"]
    metric_fields.extend(LOSS_NAMES)
    metric_fields.extend(f"gradient_norm_{loss}" for loss in LOSS_NAMES)
    metric_fields.extend(f"gradient_rms_{loss}" for loss in LOSS_NAMES)
    write_csv(metrics_path, all_rows, metric_fields)

    paired_statistics(all_rows, output_dir)
    aggregate = aggregate_metrics(all_rows)
    aggregate_fields = [
        "error_type", "loss", "n", "mean_loss", "median_loss",
        "mean_gradient_norm", "median_gradient_norm", "mean_gradient_rms",
        "median_gradient_rms", "loss_share_across_types", "gradient_rms_share_across_types",
    ]
    write_csv(output_dir / "sensitivity_by_type.csv", aggregate, aggregate_fields)

    correlations = severity_correlations(all_rows)
    write_csv(
        output_dir / "severity_correlation.csv",
        correlations,
        ["error_type", "metric", "target", "n", "pearson", "spearman"],
    )

    cosine = cosine_rows(cosine_values)
    write_csv(
        output_dir / "gradient_cosine_similarity.csv",
        cosine,
        ["scope", "loss_a", "loss_b", "n", "n_valid", "mean_cosine", "median_cosine", "std_cosine"],
    )

    metadata = suite.metadata()
    metadata.update({
        "protocol_version": 2,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "src/loss_comparison.py")},
        "package_versions": {p: importlib.metadata.version(p)
                             for p in ("torch", "transformers", "kornia", "lpips")},
        "gradient_space": "physical_ab_fixed_L",
        "semantic_matching": "within source, equal mean ab distance after gamut projection",
        "generation_protocol": json.loads((dataset_dir / "generation_protocol.json").read_text())
            if (dataset_dir / "generation_protocol.json").exists() else None,
        "generation_manifest_sha256": hashlib.sha256((dataset_dir / "manifest.jsonl").read_bytes()).hexdigest()
            if (dataset_dir / "manifest.jsonl").exists() else None,
        "semantic_labels": "unverified generator candidates; exploratory, not human semantic ground truth",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "sample_count": len(directories),
        "pair_count": len(all_rows),
        "severity_levels": levels,
        "device": str(device),
        "batch_size": args.batch_size,
        "error_types": list(ERROR_TYPES),
    })
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(recommendation_text(aggregate, cosine, metadata))
    write_html_report(output_dir / "report.html", aggregate, correlations, cosine, metadata)
    print(f"Wrote evaluation outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
