"""Validated GT moments, with provenance for safe reuse across checkpoints."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import tempfile

import numpy as np

from .metrics.fid import FeatureStatistics


DECODER_PROTOCOL = (
    "EXIF-oriented 8-bit sRGB; compatible embedded ICC converted; "
    "untagged RGB, unreadable/unusable ICC and RGB with incompatible CMYK ICC assumed sRGB; "
    "untagged CMYK converted with Pillow's default mapping; HWC [0,1]"
)
PACKAGES = ("Pillow", "torch", "torchvision", "pytorch-fid")


def _digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def cache_metadata(records, config, metric):
    """Stat signatures catch edited files without decoding or hashing all pixels."""
    sources = [(record.image_id, str(record.ground_truth.resolve())) for record in records]
    signatures = []
    latest_mtime = 0
    for image_id, filename in sources:
        info = Path(filename).stat()
        signatures.append((image_id, filename, info.st_size, info.st_mtime_ns))
        latest_mtime = max(latest_mtime, info.st_mtime_ns)
    packages = PACKAGES + (("opencv-python-headless",) if config.resize is not None and
                           config.resize_backend == "opencv" else ())
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    # 'extractor' duplicates existing protocol fields and was added after Val50k.
    protocol = {key: value for key, value in metric.protocol().items() if key != "extractor"}
    return {"schema_version": 1, "count": len(records),
            "sources_sha256": _digest(sources), "files_sha256": _digest(signatures),
            "latest_source_mtime_ns": latest_mtime,
            "decoder": DECODER_PROTOCOL,
            "resize": list(config.resize) if config.resize is not None else None,
            "resize_backend": config.resize_backend if config.resize is not None else None,
            "batch_size": getattr(metric, "batch_size", config.batch_size),
            "device_type": str(getattr(metric, "device", config.device)).split(":")[0],
            "fid": protocol, "versions": versions}


def _validate_statistics(statistics, metadata):
    dims = metadata["fid"]["dims"]
    if statistics.count != metadata["count"] or statistics.count < 2:
        raise ValueError("GT FID cache image count does not match evaluated GT")
    if (statistics.mean.shape != (dims,) or statistics.scatter.shape != (dims, dims) or
            statistics.mean.dtype != np.float64 or statistics.scatter.dtype != np.float64 or
            not np.isfinite(statistics.mean).all() or not np.isfinite(statistics.scatter).all() or
            not np.allclose(statistics.scatter, statistics.scatter.T, rtol=0, atol=1e-10) or
            np.any(np.diag(statistics.scatter) < 0)):
        raise ValueError("GT FID cache must contain finite FP64 mean and symmetric scatter of the correct dimension")


def _validate_legacy(path, metadata):
    """The archived Val50k npz stores provenance in its companion report.json."""
    report_path = path.parent / "report.json"
    if path.name != "fid_real_statistics.npz" or not report_path.is_file():
        raise ValueError("GT FID cache has no metadata; legacy caches require fid_real_statistics.npz "
                         "and a matching companion report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config, protocol = report["config"], report["protocol"]
    sources = list(dict.fromkeys((row["image_id"], str(Path(row["ground_truth"]).resolve()))
                                 for row in report["per_sample"]))
    fid = {key: value for key, value in protocol["fid"].items() if key != "extractor"}
    if (len(sources) != metadata["count"] or _digest(sources) != metadata["sources_sha256"] or
            report["fid"]["real_count"] != metadata["count"] or
            config["resize"] != metadata["resize"] or
            config["batch_size"] != metadata["batch_size"] or
            config["device"].split(":")[0] != metadata["device_type"] or
            (metadata["resize"] is not None and config.get("resize_backend", "pillow") !=
             metadata["resize_backend"]) or protocol["images"] != metadata["decoder"] or
            fid != metadata["fid"] or
            any(report["versions"].get(package) != value
                for package, value in metadata["versions"].items())):
        raise ValueError("Legacy GT FID cache does not match the GT subset, preprocessing or extractor")
    if metadata["latest_source_mtime_ns"] > path.stat().st_mtime_ns:
        raise ValueError("GT files are newer than the legacy FID cache; recompute into a new cache file")


def load_statistics(path, metadata):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as saved:
            count = saved["count"]
            if count.shape != () or count.dtype.kind not in "iu":
                raise ValueError("GT FID cache count must be an integer scalar")
            statistics = FeatureStatistics(int(count), saved["mean"], saved["scatter"])
            if "metadata" in saved:
                actual = json.loads(str(saved["metadata"].item()))
                if actual != metadata:
                    raise ValueError("GT FID cache does not match the GT files/subset, preprocessing or extractor; "
                                     "use a separate cache file for this protocol")
                status = "loaded"
            else:
                _validate_legacy(path, metadata)
                status = "legacy_loaded"
        _validate_statistics(statistics, metadata)
        return statistics, status
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed GT FID cache or companion report: {path}") from error


def save_statistics(path, statistics, metadata):
    """Atomic replacement: interrupted writes never leave a partial cache."""
    _validate_statistics(statistics, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, count=statistics.count, mean=statistics.mean, scatter=statistics.scatter,
                     metadata=json.dumps(metadata, sort_keys=True))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
