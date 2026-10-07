"""Compute real train RGB statistics and stream baseline generated EMA init."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

if __name__ == "__main__":
    # Leave CPU cores for parallel image decoding; respect an explicit override.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    print("[FD stats] Starting; importing PyTorch and project dependencies...", flush=True)

# Support the documented `python scripts/compute_fd_stats.py` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from eval.metrics.fid import FeatureStatistics, create_fid_network, extract_fid_features
from posttrain_fd import source_model
from src.fd_module import (RGB_SURROGATE, UPSTREAM_SHA, fd_rgb, make_data,
                           train_identity, validate_config)
from src.lab import lab_to_rgb, rgb_to_lab
from third_party.fd_loss.queue import FeatureQueue


def log(message):
    print(f"[FD stats] {message}", flush=True)


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Progress:
    """Time-throttled progress; no extra dependency and no CUDA synchronization."""
    def __init__(self, label, interval, unit="images", scale=1):
        self.label, self.interval, self.unit, self.scale = label, interval, unit, scale
        self.started = self.last_log = perf_counter()
        self.first = True

    def update(self, completed, total):
        now = perf_counter()
        if not (self.first or completed == total or now - self.last_log >= self.interval):
            return
        elapsed = max(now - self.started, 1e-9)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else 0
        counts = (f"{self.label}: {completed / self.scale:,.0f}/{total / self.scale:,.0f} "
                  f"{self.unit} ({100 * completed / total:.1f}%)")
        if self.first and completed < total:
            log(f"{counts} | measuring speed...")
        else:
            log(f"{counts} | {rate / self.scale:.1f} {self.unit}/s | "
                f"elapsed {duration(elapsed)} | ETA {duration(eta)}")
        self.first = False
        self.last_log = now


def compare_rgb(L, ab):
    """Measure float surrogate versus the exact OpenCV/uint8 evaluator path."""
    groups = {"source_lab": ab, "out_of_gamut_lab": ab * 3 + 0.4}
    report = {}
    for name, chroma in groups.items():
        surrogate = fd_rgb(L, chroma).cpu().numpy().transpose(0, 2, 3, 1)
        official = lab_to_rgb(L.cpu(), chroma.cpu()).astype(np.float32) / 255
        error = np.abs(surrogate - official)
        report[name] = {"mae_rgb_0_1": float(error.mean()),
                        "rmse_rgb_0_1": float(np.sqrt(np.mean(error**2))),
                        "p99_absolute_rgb_0_1": float(np.quantile(error, 0.99)),
                        "max_absolute_rgb_0_1": float(error.max()),
                        "pixels_compared": int(error.size // 3)}
    return report


class RealRGBDataset(Dataset):
    """Use the training decoder/geometry, but skip LAB work for real Inception."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _, image = self.data.load_rgb(index)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255


def init_stats_worker(_worker_id):
    import cv2
    cv2.setNumThreads(1)


def stats_loader(data, indices, batch_size, workers, device):
    options = {"num_workers": workers, "pin_memory": device.type == "cuda",
               "worker_init_fn": init_stats_worker}
    if workers:
        options["prefetch_factor"] = 2
    return DataLoader(Subset(data, indices), batch_size=batch_size, **options)


def collect_real_statistics(network, loader, device, progress, total):
    """Overlap bounded CPU FP64 updates with GPU extraction; preserve merge order."""
    statistics = FeatureStatistics()
    pending, count, processed = [], 0, 0
    conversion, update = None, None
    # One writer owns the moments. At most one update and one pending block live.
    with ThreadPoolExecutor(max_workers=1) as worker:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            for rgb in loader:
                features = extract_fid_features(network, rgb.to(device, non_blocking=True))
                if conversion is None:
                    # Only the range diagnostic needs LAB, once rather than per image.
                    lab = [rgb_to_lab(image) for image in
                           (rgb.numpy().transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)]
                    L, ab = (torch.stack(values) for values in zip(*lab))
                    conversion = compare_rgb(L, ab)
                pending.append(features.cpu().numpy())
                count += len(features)
                if count >= 1024:
                    if update is not None:
                        update.result()
                    update = worker.submit(statistics.update, np.concatenate(pending))
                    pending, count = [], 0
                processed += len(features)
                progress.update(processed, total)
            if update is not None:
                update.result()
            if pending:
                statistics.update(np.concatenate(pending))
    return statistics, conversion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fd_inception_posttrain.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--real-limit", type=int, help="Smoke only; normal real stats use the full train split")
    parser.add_argument("--generated-samples", type=int, help="Smoke only; defaults to 50k from config")
    parser.add_argument("--batch-size", type=int, help="Baseline generator batch (defaults to fd.stats_batch_size)")
    parser.add_argument("--real-batch-size", type=int,
                        help="Inception-only batch (default: fd.real_stats_batch_size or 32)")
    parser.add_argument("--num-workers", type=int,
                        help="Parallel image readers (default: fd.stats_num_workers or data.num_workers)")
    parser.add_argument("--log-every-seconds", type=float, default=5.0,
                        help="Progress interval (default: 5 seconds); stage messages always print")
    args = parser.parse_args(argv)
    if not math.isfinite(args.log_every_seconds) or args.log_every_seconds <= 0:
        parser.error("--log-every-seconds must be finite and positive")
    log(f"Reading config: {args.config}")
    config = yaml.safe_load(Path(args.config).read_text())
    validate_config(config)
    hash_report = None
    def hash_progress(completed, total):
        nonlocal hash_report
        if hash_report is None:
            hash_report = Progress("Checkpoint SHA256", args.log_every_seconds, "MiB", 2**20)
        hash_report.update(completed, total)
    model, provenance = source_model(config, progress=log, hash_progress=hash_progress)
    log(f"Moving generator to {args.device}")
    model.to(args.device)
    log(f"Loading frozen pytorch-fid Inception pool3/2048 on {args.device}")
    network = create_fid_network(args.device)
    log(f"Reading train split: {config['data'].get('train_manifest') or config['data'].get('train_root')}")
    data = make_data(config)
    log(f"Train split loaded: {len(data):,} images; computing index fingerprint")
    index_report = Progress("Train index", args.log_every_seconds, "entries")
    identity = train_identity(data, progress=index_report.update)
    batch_size = args.batch_size if args.batch_size is not None else config["fd"]["stats_batch_size"]
    real_batch_size = (args.real_batch_size if args.real_batch_size is not None else
                       config["fd"].get("real_stats_batch_size", 32))
    workers = (args.num_workers if args.num_workers is not None else
               config["fd"].get("stats_num_workers", config["data"].get("num_workers", 0)))
    real_count = args.real_limit if args.real_limit is not None else len(data)
    generated_count = (args.generated_samples if args.generated_samples is not None else
                       config["fd"]["generated_init_samples"])
    if min(batch_size, real_batch_size) < 1 or workers < 0 or not 2 <= real_count <= len(data) or generated_count < 2:
        raise ValueError("positive batch sizes, nonnegative workers, >=2 real/generated images, real-limit <= train size required")
    device = torch.device(args.device)
    output = Path(config["fd"]["stats_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log(f"Plan: {real_count:,} real images + {generated_count:,} generated samples; "
        f"real batch={real_batch_size}, generated batch={batch_size}, workers={workers}, "
        f"geometry={identity['resize_strategy']} {identity['resolution']}; output={output.resolve()}")
    log("Building seeded train permutation and data loaders")
    seed = config["seed"]
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed)).tolist()
    real_loader = stats_loader(RealRGBDataset(data), order[:real_count], real_batch_size, workers, device)
    started = perf_counter()
    log("Starting parallel RGB decoding -> Inception -> background FP64 moments "
        "(waiting for the first decoded batch)")
    real_progress = Progress("Real stats", args.log_every_seconds)
    statistics, conversion = collect_real_statistics(network, real_loader, device, real_progress, real_count)
    log("Finalizing real FP64 mean/covariance")
    mu, sigma = statistics.finalize()
    real_metadata = {"identity": identity, "sample_count": statistics.count,
                     "sample_selection": "seeded train permutation", "seed": seed,
                     "covariance_dtype": "float64", "covariance_ddof": 1,
                     "rgb": "original decoded train RGB after geometry /255",
                     "smoke_subset": statistics.count < len(data) or bool(config.get("dataset_provenance")),
                     "dataset_provenance": config.get("dataset_provenance"), "upstream_sha": UPSTREAM_SHA}
    log(f"Saving real statistics: {output / 'real.npz'}")
    np.savez(output / "real.npz", mu=mu, sigma=sigma,
             metadata=json.dumps(real_metadata, sort_keys=True))
    log(f"Real statistics saved: {statistics.count:,} images")
    log("Preparing generated feature EMA streaming initialization")
    queue = FeatureQueue(size=50000, feat_dim=2048, ema_beta=config["fd"]["ema_beta"]).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(seed)
    indices = (order * ((generated_count + len(order) - 1) // len(order)))[:generated_count]
    data.return_rgb = False
    generated_loader = stats_loader(data, indices, batch_size, workers, device)
    generated = 0
    log("Starting baseline generation on train-L (waiting for the first decoded batch)")
    generated_progress = Progress("Generated EMA", args.log_every_seconds)
    with torch.no_grad():
        for batch in generated_loader:
            L = batch["L"].to(args.device, non_blocking=True)
            noise = torch.randn((len(L), 2, *model.resolution), device=args.device,
                                generator=generator) * provenance["noise_scale"]
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                                enabled=config["training"]["precision"] == "bf16-mixed"):
                ab = model.sample_from_noise(L, noise)
            with torch.autocast(torch.device(args.device).type, enabled=False):
                features = extract_fid_features(network, fd_rgb(L, ab))
                if not torch.isfinite(features).all():
                    raise ValueError("nonfinite baseline features during initialization")
                queue.accumulate_batch(features)
            generated += len(L)
            if generated == len(L):
                conversion["baseline_generated"] = compare_rgb(L, ab)
            generated_progress.update(generated, generated_count)
        log("Finalizing generated FP64 mean and second moment")
        queue._finalize_streaming_init()
    metadata = {"identity": identity, "provenance": provenance, "sample_count": generated_count,
                "rgb_surrogate": RGB_SURROGATE, "covariance_ddof": 0,
                "moments": "E[x], E[xxT]; population covariance = m2 - mu muT",
                "seed": seed, "generator_precision": config["training"]["precision"],
                "ema_beta": config["fd"]["ema_beta"], "upstream_sha": UPSTREAM_SHA,
                "smoke_initialization": generated_count != 50000}
    log(f"Saving generated EMA: {output / 'generated.pt'}")
    torch.save({"queue": {k: v.cpu() for k, v in queue.state_dict().items()},
                "metadata": metadata}, output / "generated.pt")
    report = {"rgb_conversion": conversion, "seconds": perf_counter() - started,
              "real_count": real_count, "generated_count": generated_count}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"Done: {real_count:,} real / {generated_count:,} generated; "
        f"statistics loop elapsed {duration(report['seconds'])}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
