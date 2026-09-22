"""Controlled color perturbations for colorization-loss evaluation.

The functions in this module operate in CIELAB and deliberately modify only
the chroma channels.  They use a small, dependency-free (apart from NumPy)
implementation of sRGB <-> CIELAB so the evaluation generator also works in
the lightweight ``.venv_diff`` environment.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


_SRGB_TO_XYZ = np.asarray(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ).astype(np.float32)
_D65 = np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
_DELTA = 6.0 / 29.0


def _lab_f(value: np.ndarray) -> np.ndarray:
    return np.where(
        value > _DELTA**3,
        np.cbrt(np.maximum(value, 0.0)),
        value / (3.0 * _DELTA**2) + 4.0 / 29.0,
    )


def _lab_f_inv(value: np.ndarray) -> np.ndarray:
    return np.where(
        value > _DELTA,
        value**3,
        3.0 * _DELTA**2 * (value - 4.0 / 29.0),
    )


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an RGB uint8 array shaped [height, width, 3]")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to CIELAB as float32 ``[L*, a*, b*]``."""

    _validate_rgb(rgb)
    srgb = rgb.astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ _SRGB_TO_XYZ.T
    f = _lab_f(xyz / _D65)
    lab = np.empty_like(f, dtype=np.float32)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _lab_to_srgb_float(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.float32, copy=False)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = np.stack([_lab_f_inv(fx), _lab_f_inv(fy), _lab_f_inv(fz)], axis=-1)
    xyz *= _D65
    linear = xyz @ _XYZ_TO_SRGB.T
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return srgb


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB float data to RGB uint8, clipping only at sRGB output."""

    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError("expected a Lab array shaped [height, width, 3]")
    srgb = _lab_to_srgb_float(lab)
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _compress_chroma_to_srgb_gamut(lab: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Reduce only ``a*``/``b*`` where needed to keep source ``L*`` in gamut."""

    srgb = _lab_to_srgb_float(lab)
    invalid = np.any((srgb < 0.0) | (srgb > 1.0), axis=-1)
    if not np.any(invalid):
        return lab

    low = np.zeros(lab.shape[:2], dtype=np.float32)
    high = np.ones(lab.shape[:2], dtype=np.float32)
    for _ in range(iterations):
        middle = (low + high) * 0.5
        candidate = lab.copy()
        candidate[..., 1:] *= middle[..., None]
        candidate_rgb = _lab_to_srgb_float(candidate)
        valid = np.all((candidate_rgb >= 0.0) & (candidate_rgb <= 1.0), axis=-1)
        low = np.where(invalid & valid, middle, low)
        high = np.where(invalid & valid, high, np.where(invalid, middle, high))

    result = lab.copy()
    result[..., 1:] *= np.where(invalid, low, 1.0)[..., None]
    return result


def _hue_rotate(ab: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees).astype(np.float32)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    a, b = ab[..., 0], ab[..., 1]
    return np.stack([cosine * a - sine * b, sine * a + cosine * b], axis=-1)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    positions = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(positions**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _blur_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(array, axis, -1)
    radius = len(kernel) // 2
    padded = np.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(radius, radius)], mode="reflect")
    flat = padded.reshape(-1, padded.shape[-1])
    blurred = np.stack(
        [np.convolve(row, kernel, mode="valid") for row in flat], axis=0
    ).reshape(moved.shape)
    return np.moveaxis(blurred, -1, axis)


def _gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(float(sigma))
    return _blur_axis(_blur_axis(array, kernel, axis=1), kernel, axis=0)


def _reflect_indices(length: int, offset: int) -> np.ndarray:
    if length < 2:
        return np.zeros(length, dtype=np.int64)
    period = 2 * length - 2
    folded = (np.arange(length, dtype=np.int64) + offset) % period
    return np.where(folded < length, folded, period - folded)


def _shift_reflect(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    y = _reflect_indices(array.shape[0], dy)
    x = _reflect_indices(array.shape[1], dx)
    return array[np.ix_(y, x)]


def recolor_lab(lab: np.ndarray, variant: str) -> np.ndarray:
    """Return one deterministic chroma-only perturbation.

    ``lab[..., 0]`` is copied byte-for-byte from the input.  The color
    bleeding variant intentionally blurs and offsets only chroma, which makes
    color cross nearby object boundaries without smearing luminance edges.
    """

    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError("expected a Lab array shaped [height, width, 3]")
    if variant not in {"wrong_hue", "low_saturation", "color_bleeding", "subtle_color_error"}:
        raise ValueError(f"unknown synthetic variant: {variant}")

    source_ab = lab[..., 1:]
    if variant == "wrong_hue":
        ab = _hue_rotate(source_ab, degrees=55.0)
    elif variant == "low_saturation":
        ab = source_ab * 0.32
    elif variant == "subtle_color_error":
        ab = _hue_rotate(source_ab * 0.92, degrees=8.0)
    else:
        height, width = lab.shape[:2]
        sigma = max(1.5, min(height, width) * 0.025)
        blurred = _gaussian_blur(source_ab, sigma=sigma)
        shifted = _shift_reflect(source_ab, dx=max(1, width // 28), dy=max(1, height // 36))
        # The local blur creates boundary bleed; the small offset adds a
        # directional donor color so the error is not merely desaturation.
        ab = 0.52 * source_ab + 0.31 * blurred + 0.17 * shifted

    result = lab.copy()
    result[..., 1:] = ab
    return result


SYNTHETIC_VARIANTS: tuple[str, ...] = (
    "wrong_hue",
    "low_saturation",
    "color_bleeding",
    "subtle_color_error",
)

QWEN_VARIANTS: tuple[str, ...] = (
    "semantically_wrong_color",
    "plausible_alternative_colorization",
)


def generate_synthetic(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate all four synthetic variants as RGB uint8 images."""

    lab = rgb_to_lab(rgb)
    return {variant: lab_to_rgb(recolor_lab(lab, variant)) for variant in SYNTHETIC_VARIANTS}


def compose_with_source_l(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    preserve_source_chroma: bool = False,
) -> np.ndarray:
    """Keep source luminance and take generated chroma.

    When ``preserve_source_chroma`` is enabled, Qwen supplies the new hue and
    spatial chroma pattern while the per-pixel source ``C*`` magnitude is
    retained. This prevents a plausible alternative from degenerating into a
    global saturation/contrast boost.
    """

    if source_rgb.shape != generated_rgb.shape:
        raise ValueError("source and generated RGB images must have identical shapes")
    source_lab = rgb_to_lab(source_rgb)
    generated_lab = rgb_to_lab(generated_rgb)
    composed = np.empty_like(source_lab)
    composed[..., 0] = source_lab[..., 0]
    generated_ab = generated_lab[..., 1:]
    if preserve_source_chroma:
        source_chroma = np.linalg.norm(source_lab[..., 1:], axis=-1)
        generated_chroma = np.linalg.norm(generated_ab, axis=-1)
        scale = source_chroma / np.maximum(generated_chroma, 1e-6)
        generated_ab = generated_ab * scale[..., None]
    composed[..., 1:] = generated_ab
    # Qwen can propose chroma outside the sRGB gamut. Compressing chroma
    # before encoding preserves the source luminance better than per-channel
    # RGB clipping while retaining the intended hue as far as possible.
    return lab_to_rgb(_compress_chroma_to_srgb_gamut(composed))


PLAUSIBLE_PROMPT = (
    "Colorize this grayscale image. Preserve the exact scene structure, "
    "composition, object count, object positions, shapes, contours, boundaries, "
    "textures, fine details, shadows, highlights, and brightness structure. "
    "Do not redraw, add, remove, move, distort, or hallucinate any content. "
    "Keep the original grayscale luminance and change only the chromatic colors. "
    "Create a realistic, coherent, natural colorization with plausible colors "
    "for every object and material."
)


def prompts_for_qwen() -> Dict[str, str]:
    """Natural colorization and an implausible-color control, same geometry prompt.

    Alternative means an independent colorization from grayscale, not a forced
    distance from GT or a prescribed palette. Human plausibility remains unverified.
    """
    return {
        "plausible_alternative_colorization": PLAUSIBLE_PROMPT,
        "semantically_wrong_color": PLAUSIBLE_PROMPT.replace(
            "Create a realistic, coherent, natural colorization with plausible colors "
            "for every object and material.",
            "Create a coherent colorization with clearly semantically implausible "
            "colors for recognizable objects and materials: for example blue or "
            "violet skin, purple vegetation, or green skies. Retain natural shading "
            "and local texture; change object colors rather than applying a global color filter."
        ),
    }


def requested_variants(names: Iterable[str]) -> tuple[str, ...]:
    """Expand ``all`` and validate CLI variant names."""

    names = tuple(names)
    if not names or "all" in names:
        return SYNTHETIC_VARIANTS + QWEN_VARIANTS
    valid = set(SYNTHETIC_VARIANTS + QWEN_VARIANTS)
    unknown = sorted(set(names) - valid)
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return tuple(dict.fromkeys(names))
