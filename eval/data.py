"""Deterministic disk records and a common sRGB preprocessing boundary."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .config import EvalConfig

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    ground_truth: Path
    samples: tuple[SampleRecord, ...]


def _images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and
                  p.suffix.lower() in EXTENSIONS)


def _index(root: Path) -> dict[str, Path]:
    result = {}
    for path in _images(root):
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in result:
            raise ValueError(f"duplicate image ID {key!r}: {result[key]} and {path}")
        result[key] = path
    return result


def discover(config: EvalConfig) -> tuple[list[ImageRecord], dict]:
    """Pair by relative path without extension, never by traversal order."""
    truth = _index(config.ground_truth)
    grouped: dict[str, dict[str, Path]] = {}
    if config.layout == "single":
        grouped = {key: {"single": path} for key, path in _index(config.predictions).items()}
    else:
        for path in _images(config.predictions):
            relative = path.relative_to(config.predictions)
            if len(relative.parts) < 2:
                directory = "a sample" if config.layout == "sample-dirs" else "an image"
                raise ValueError(f"{config.layout} requires {directory} directory: {path}")
            if config.layout == "sample-dirs":
                sample_id = relative.parts[0]
                key = Path(*relative.parts[1:]).with_suffix("").as_posix()
            else:
                key, sample_id = relative.parent.as_posix(), relative.stem
            samples = grouped.setdefault(key, {})
            if sample_id in samples:
                raise ValueError(f"duplicate image/sample ID: {key}/{sample_id}")
            samples[sample_id] = path
    if not truth or not grouped:
        raise ValueError("ground truth and predictions must both contain images")
    extra = sorted(set(grouped) - set(truth))
    missing = sorted(set(truth) - set(grouped))
    if extra:
        raise ValueError(f"predictions without ground truth: {extra[:10]}")
    if missing and not config.allow_subset:
        raise ValueError(f"missing predictions for {len(missing)} images: {missing[:10]}; "
                         "use allow_subset explicitly to evaluate a subset")
    records = []
    for key, samples in sorted(grouped.items()):
        ids = config.sample_ids or tuple(sorted(samples))
        absent = set(ids) - set(samples)
        if absent:
            raise ValueError(f"image {key!r} is missing requested samples: {sorted(absent)}")
        records.append(ImageRecord(key, truth[key], tuple(
            SampleRecord(sample_id, samples[sample_id]) for sample_id in ids)))
    counts = [len(record.samples) for record in records]
    if len(set(counts)) > 1 and not config.allow_variable_k:
        raise ValueError("variable K across images; select sample_ids or explicitly "
                         "set allow_variable_k")
    if config.layout == "sample-dirs" and not config.allow_variable_k:
        sample_sets = {frozenset(s.sample_id for s in record.samples) for record in records}
        if len(sample_sets) > 1:
            raise ValueError("sample directories must cover the same image IDs")
    return records, {
        "ground_truth_images": len(truth), "evaluated_images": len(records),
        "predictions": sum(counts), "k_min": min(counts), "k_max": max(counts),
        "unevaluated_ground_truth_ids": missing,
    }


def validate_rgb(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3
            or min(image.shape[:2]) == 0 or image.dtype.kind != "f"):
        raise ValueError("expected a nonempty HWC floating-point RGB array in [0,1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("RGB values must be finite and in [0,1]; no automatic clipping")


def load_rgb(path: Path, resize: tuple[int, int] | None = None, *,
             resize_backend: str = "pillow") -> np.ndarray:
    """Decode 8-bit images, apply EXIF orientation/ICC, return HWC float32 [0,1]."""
    if resize_backend not in ("pillow", "opencv"):
        raise ValueError("resize_backend must be pillow or opencv")
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"animated/multi-frame images are unsupported: {path}")
        if source.mode not in ("RGB", "RGBA", "L", "LA", "P", "CMYK"):
            raise ValueError(f"unsupported image mode {source.mode!r} in {path}; "
                             "export an 8-bit sRGB image explicitly")
        # Pillow can silently decode 16-bit RGB PNG/TIFF into 8-bit RGB mode.
        # Check source precision before the pixel decoder discards that detail.
        if source.format == "PNG":
            with Path(path).open("rb") as handle:
                header = handle.read(25)
            if len(header) == 25 and header[24] > 8:
                raise ValueError(f"high-bit-depth PNG requires explicit 8-bit export: {path}")
        if source.format == "TIFF":
            bits = source.tag_v2.get(258, (8,))  # TIFF BitsPerSample
            if any(value > 8 for value in (bits if isinstance(bits, tuple) else (bits,))):
                raise ValueError(f"high-bit-depth TIFF requires explicit 8-bit export: {path}")
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or "transparency" in image.info:
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"transparent images require explicit compositing: {path}")
        profile = image.info.get("icc_profile")
        if profile:
            try:
                try:
                    input_profile = ImageCms.ImageCmsProfile(BytesIO(profile))
                except (OSError, ValueError):
                    warnings.warn(f"Ignoring unreadable ICC profile; decoded RGB assumed sRGB: {path}",
                                  RuntimeWarning, stacklevel=2)
                    input_profile = None
                profile_space = input_profile.profile.xcolor_space.strip() if input_profile else None
                # An RGB file sometimes retains a printing profile after export.
                # A CMYK profile cannot describe its three decoded RGB channels;
                # preserve those pixels under the same assumption as untagged RGB.
                if input_profile is None:
                    image = image.convert("RGB")
                elif image.mode in ("RGB", "RGBA", "P") and profile_space == "CMYK":
                    warnings.warn(f"Ignoring incompatible CMYK ICC profile on {image.mode} image; "
                                  f"decoded RGB assumed sRGB: {path}", RuntimeWarning, stacklevel=2)
                    image = image.convert("RGB")
                else:
                    # Some grayscale JPEGs carry RGB ICC profiles. Expand their L
                    # samples to RGB before applying that profile; native GRAY
                    # profiles must keep L input for the color-management transform.
                    if (image.mode in ("P", "RGBA") or
                            (image.mode in ("L", "LA") and profile_space == "RGB")):
                        image = image.convert("RGB")
                    elif image.mode == "LA":
                        image = image.convert("L")
                    try:
                        transform = ImageCms.buildTransform(
                            input_profile, ImageCms.createProfile("sRGB"), image.mode, "RGB")
                    except ImageCms.PyCMSError:
                        warnings.warn(f"Ignoring unusable ICC profile; decoded RGB assumed sRGB: {path}",
                                      RuntimeWarning, stacklevel=2)
                        image = image.convert("RGB")
                    else:
                        image = ImageCms.applyTransform(image, transform)
            except (OSError, ValueError, ImageCms.PyCMSError) as error:
                raise ValueError(f"cannot convert ICC profile to sRGB: {path}") from error
        else:
            if image.mode == "CMYK":
                warnings.warn(f"Untagged CMYK converted with Pillow's default CMYK-to-RGB mapping: {path}",
                              RuntimeWarning, stacklevel=2)
            image = image.convert("RGB")
        if resize is not None and image.size != (resize[1], resize[0]):
            if resize_backend == "opencv":
                import cv2
                pixels = cv2.resize(np.asarray(image), (resize[1], resize[0]),
                                    interpolation=cv2.INTER_CUBIC)
                return pixels.astype(np.float32) / 255.0
            image = image.resize((resize[1], resize[0]), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def validate_pair(first: np.ndarray, second: np.ndarray) -> None:
    validate_rgb(first)
    validate_rgb(second)
    if first.shape != second.shape:
        raise ValueError(f"image shapes differ: {first.shape} vs {second.shape}; "
                         "set resize explicitly to compare at a common resolution")
