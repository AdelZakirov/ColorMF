"""Evaluate records independently of their producer or checkpoint runner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from time import perf_counter

import numpy as np

from .config import EvalConfig
from ._parallel import ordered_map
from .data import ImageRecord, discover, load_rgb, validate_pair
from .metrics.color import colorfulness, delta_e00_lab, to_lab
from .fid_cache import DECODER_PROTOCOL, cache_metadata, load_statistics, save_statistics
from .metrics.fid import FIDMetric, frechet_distance
from .metrics.lpips import LPIPSMetric
from .metrics.pixel import psnr, ssim


def _means(rows: Iterable[dict]) -> dict:
    values = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            if value is not None:
                values[name].append(value)
    return {name: float(np.mean(values[name])) for name in sorted(values)}


def _versions() -> dict:
    result = {}
    for package in ("numpy", "Pillow", "opencv-python-headless", "scikit-image", "scipy", "torch",
                    "torchvision", "lpips", "pytorch-fid", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _load_record(record: ImageRecord, config: EvalConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    target = load_rgb(record.ground_truth, config.resize, resize_backend=config.resize_backend)
    predictions = [load_rgb(sample.path, config.resize, resize_backend=config.resize_backend)
                   for sample in record.samples]
    for prediction in predictions:
        try:
            validate_pair(prediction, target)
        except ValueError as error:
            raise ValueError(f"image {record.image_id!r}: {error}") from error
    return target, predictions


def _evaluate_image(record: ImageRecord, config: EvalConfig) -> tuple[dict, list[dict]]:
    target, predictions = _load_record(record, config)

    target_colorfulness = None
    if "colorfulness" in config.metrics or "delta_colorfulness" in config.metrics:
        target_colorfulness = colorfulness(target, config.colorfulness_variant)
    sample_metrics = []
    target_lab = to_lab(target) if "delta_e00" in config.metrics else None
    for prediction in predictions:
        values = {}
        if "psnr" in config.metrics:
            values["psnr"] = psnr(prediction, target)
        if "ssim" in config.metrics:
            values["ssim"] = ssim(prediction, target)
        if target_colorfulness is not None:
            cf = colorfulness(prediction, config.colorfulness_variant)
            if "colorfulness" in config.metrics:
                values.update(colorfulness=cf, colorfulness_gt=target_colorfulness)
            if "delta_colorfulness" in config.metrics:
                values.update(delta_colorfulness=abs(cf - target_colorfulness),
                              signed_delta_colorfulness=cf - target_colorfulness)
        if "delta_e00" in config.metrics:
            values["delta_e00"] = delta_e00_lab(to_lab(prediction), target_lab)
        sample_metrics.append(values)

    k = len(predictions)
    pair_count = k * (k - 1) // 2
    best = {}
    if "delta_e00" in config.metrics:
        sample, values = min(zip(record.samples, sample_metrics),
                             key=lambda item: item[1]["delta_e00"])
        best["delta_e00"] = {"value": values["delta_e00"], "sample_id": sample.sample_id}

    sample_rows = [{"image_id": record.image_id, "sample_id": sample.sample_id,
                    "prediction": str(sample.path), "ground_truth": str(record.ground_truth),
                    "height": target.shape[0], "width": target.shape[1], "metrics": values}
                   for sample, values in zip(record.samples, sample_metrics)]
    image_row = {"image_id": record.image_id, "k": k,
                 "first_sample_id": record.samples[0].sample_id,
                 "mean_over_samples": _means(sample_metrics),
                 "first_sample": dict(sample_metrics[0]), "best_of_k": best,
                 "lpips_diversity": None,
                 "diversity_pairs": pair_count if "lpips" in config.metrics else None}
    return image_row, sample_rows


def _evaluate_lpips(records: list[ImageRecord], config: EvalConfig, metric: LPIPSMetric,
                    image_rows: list[dict], sample_rows: list[dict],
                    progress: Callable[[int, int], None] | None = None) -> None:
    # Stream ordered pairs from bounded decoded-record prefetch, retaining only
    # the queued records and a neural batch rather than the entire dataset.
    def pairs():
        loaded = ordered_map(lambda record: _load_record(record, config), records, config.workers)
        for target, predictions in loaded:
            yield from ((prediction, target) for prediction in predictions)
            yield from combinations(predictions, 2)

    scores = metric.iter_distances(pairs())
    samples = iter(sample_rows)
    for index, (record, row) in enumerate(zip(records, image_rows), start=1):
        values = [next(scores) for _ in record.samples]
        for value in values:
            next(samples)["metrics"]["lpips"] = value
        row["mean_over_samples"]["lpips"] = float(np.mean(values))
        row["first_sample"]["lpips"] = values[0]
        selected = min(range(len(values)), key=values.__getitem__)
        row["best_of_k"]["lpips"] = {"value": values[selected],
                                     "sample_id": record.samples[selected].sample_id}
        if row["diversity_pairs"]:
            pair_count = row["diversity_pairs"]
            row["lpips_diversity"] = sum(next(scores) for _ in range(pair_count)) / pair_count
        if progress:
            progress(index, len(records))


def _image_results(records: list[ImageRecord], config: EvalConfig,
                   before_image: Callable[[ImageRecord], None] | None = None):
    def score(record):
        if before_image:
            before_image(record)
        return _evaluate_image(record, config)

    yield from ordered_map(score, records, config.workers)


def evaluate_records(records: Iterable[ImageRecord], config: EvalConfig, *,
                     lpips_metric: LPIPSMetric | None = None,
                     fid_metric: FIDMetric | None = None,
                     coverage: dict | None = None,
                     progress: Callable[[int, int], None] | None = None,
                     phase_progress: Callable[[str, int, int], None] | None = None,
                     before_image: Callable[[ImageRecord], None] | None = None) -> dict:
    """The inference stage only needs to produce these disk records.

    NFE/inference latency/throughput remain unmeasured unless a generation runner
    explicitly supplies those measurements.
    Injected neural backends are marked in the report and are not paper scores.
    A concurrent disk producer can supply before_image to wait for each record's
    atomically saved predictions before CPU scoring. Neural stages follow CPU
    scoring, when all predictions are available. The callback runs on workers.
    Evaluation wall time then includes producer readiness waits.
    """
    started = perf_counter()
    records = list(records)
    if not records or any(not record.samples for record in records):
        raise ValueError("evaluation needs nonempty records and samples")
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("image IDs must be unique")
    if any(len({s.sample_id for s in record.samples}) != len(record.samples)
           for record in records):
        raise ValueError("sample IDs must be unique within each image")
    counts = [len(record.samples) for record in records]
    if len(set(counts)) > 1 and not config.allow_variable_k:
        raise ValueError("variable K requires allow_variable_k")
    if "fid" in config.metrics and len(records) < 2:
        raise ValueError("FID needs at least two ground-truth images")
    if "lpips" in config.metrics and lpips_metric is None:
        lpips_metric = LPIPSMetric(config.lpips_net, config.device, config.batch_size)
    if "fid" in config.metrics and fid_metric is None:
        fid_metric = FIDMetric(config.device, config.batch_size)
    real, real_cache, real_metadata = None, None, None
    if "fid" in config.metrics and config.fid_real_stats is not None:
        if fid_metric.injected:
            raise ValueError("GT FID disk caches require the production extractor, not an injected network")
        real_metadata = cache_metadata(records, config, fid_metric)
        if config.fid_real_stats.exists():
            real, cache_status = load_statistics(config.fid_real_stats, real_metadata)
            real_cache = {"path": str(config.fid_real_stats), "status": cache_status}
            if phase_progress:
                phase_progress("fid_real_cached", real.count, real.count)
    sample_rows, image_rows = [], []
    for index, (image_row, samples) in enumerate(_image_results(records, config, before_image), start=1):
        image_rows.append(image_row)
        sample_rows.extend(samples)
        if progress and "lpips" not in config.metrics:
            progress(index, len(records))
        if phase_progress:
            phase_progress("image_metrics", index, len(records))
    if "lpips" in config.metrics:
        def lpips_progress(done, total):
            if progress:
                progress(done, total)
            if phase_progress:
                phase_progress("lpips", done, total)
        _evaluate_lpips(records, config, lpips_metric, image_rows, sample_rows, lpips_progress)
    aggregate = {scope: _means(row[scope] for row in image_rows)
                 for scope in ("mean_over_samples", "first_sample")}
    if "delta_colorfulness" in config.metrics:
        for metrics in aggregate.values():
            # Some papers instead report |mean(CF_pred) - mean(CF_GT)|.
            metrics["delta_mean_colorfulness"] = abs(metrics["signed_delta_colorfulness"])
    aggregate["stochastic"] = _means(
        {**{f"best_of_k_{name}": score["value"] for name, score in row["best_of_k"].items()},
         "lpips_diversity": row["lpips_diversity"]}
        for row in image_rows)
    if "lpips" in config.metrics:
        aggregate["stochastic"]["diversity_images"] = sum(row["lpips_diversity"] is not None
                                                          for row in image_rows)
    fid_result = None
    messages = []
    if "fid" in config.metrics:
        real_paths = [record.ground_truth for record in records]
        pred_paths = [sample.path for record in records for sample in
                      (record.samples[:1] if config.fid_sampling == "first" else record.samples)]
        def fid_images(paths, phase):
            images = ordered_map(lambda path: load_rgb(
                path, config.resize, resize_backend=config.resize_backend), paths, config.workers)
            for index, image in enumerate(images, start=1):
                yield image
                if phase_progress:
                    phase_progress(phase, index, len(paths))
        if real is None:
            real = fid_metric.statistics(fid_images(real_paths, "fid_real"))
            if config.fid_real_stats is not None:
                if real_metadata != cache_metadata(records, config, fid_metric):
                    raise ValueError("GT files changed while computing FID statistics; cache was not saved")
                save_statistics(config.fid_real_stats, real, real_metadata)
                real_cache = {"path": str(config.fid_real_stats), "status": "computed"}
        generated = fid_metric.statistics(fid_images(pred_paths, "fid_generated"))
        fid_result = {"value": frechet_distance(real, generated),
                      "real_count": real.count, "generated_count": generated.count,
                      "sampling": config.fid_sampling}
        if real_cache is not None:
            fid_result["real_statistics_cache"] = real_cache
        if min(real.count, generated.count) <= fid_metric.dims:
            messages.append("FID sample count is <= feature dimension; covariance is rank-deficient. "
                            "This small-data score is not a reliable paper comparison.")
        if config.fid_sampling == "all":
            messages.append("FID uses all generated samples against one GT per input; "
                            "counts/conditional weights differ from single-sample FID.")
        if not fid_metric.injected:
            messages.append("FID uses the pytorch-fid protocol, which differs from DDColor's "
                            "public evaluation backend. Match the target paper's protocol or "
                            "re-evaluate baseline images with this evaluator before comparing scores.")
    if min(counts) != max(counts):
        messages.append("K varies by image: best-of-K scores are not directly comparable to fixed-K runs.")
    injected = (("lpips" in config.metrics and lpips_metric.injected) or
                ("fid" in config.metrics and fid_metric.injected))
    if injected:
        messages.append("Injected neural backend(s): technical validation only, not pretrained metric scores.")
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(), "versions": _versions(),
        "coverage": coverage or {"evaluated_images": len(records), "predictions": sum(counts),
                                 "k_min": min(counts), "k_max": max(counts)},
        "protocol": {
            "images": DECODER_PROTOCOL,
            "resize": "none" if config.resize is None else
                      ("OpenCV INTER_CUBIC" if config.resize_backend == "opencv" else "Pillow bicubic")
                      + ", both sets, no crop; unchanged if already at target size",
            "pixel_metrics": "RGB, data_range=1; per-image then macro average",
            "ssim": "Gaussian 11x11, sigma=1.5, population covariance, K1=.01 K2=.03",
            "colorfulness": f"RGB [0,255], {config.colorfulness_variant} opponent channels, ddof=0",
            "delta_colorfulness": "mean per-image absolute CF difference; signed difference also reported; "
                                  "delta_mean_colorfulness is abs difference of dataset means",
            "delta_e00": "sRGB -> physical CIELAB D65/2deg; kL=kC=kH=1; mean over pixels",
            "aggregation": "equal weight per input image, equal sample weight within each image",
            "best_of_k": "minimum whole-image score, independently for LPIPS and delta_e00",
            "diversity": "mean LPIPS over all unordered pairs per image, macro mean over K>=2 inputs",
            "sample_order": "sample_ids config order, otherwise lexicographic",
            "lpips": lpips_metric.protocol() if "lpips" in config.metrics else None,
            "fid": fid_metric.protocol() if "fid" in config.metrics else None,
            "injected_neural_backends": bool(injected)},
        "aggregate": aggregate, "fid": fid_result, "per_image": image_rows,
        "per_sample": sample_rows, "warnings": messages,
        "timing": {"evaluation_seconds": perf_counter() - started,
                   "evaluation_scope": "metric pipeline wall time" +
                       (", including producer readiness waits" if before_image else ""),
                   "inference": {"status": "not_measured", "nfe": None,
                                 "latency_seconds": None, "throughput_images_per_second": None}},
    }


def evaluate(config: EvalConfig, **kwargs) -> dict:
    records, coverage = discover(config)
    return evaluate_records(records, config, coverage=coverage, **kwargs)
