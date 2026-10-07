"""Colorfulness and physical CIELAB D65/2-degree CIEDE2000 differences."""

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from ..data import validate_pair, validate_rgb


def colorfulness(image: np.ndarray, variant: str = "absolute") -> float:
    """Hasler/Suesstrunk opponent score on RGB [0,255].

    'absolute' matches DDColor's public evaluation code; 'signed' uses the
    original signed opponent channels. Their variances are not interchangeable.
    """
    validate_rgb(image)
    red, green, blue = np.moveaxis(image.astype(np.float64) * 255.0, -1, 0)
    rg, yb = red - green, 0.5 * (red + green) - blue
    if variant == "absolute":
        rg, yb = np.abs(rg), np.abs(yb)
    elif variant != "signed":
        raise ValueError("colorfulness variant must be absolute or signed")
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def delta_colorfulness(prediction: np.ndarray, target: np.ndarray,
                       variant: str = "absolute") -> float:
    """Per-image absolute CF difference; zero means matching colorfulness."""
    validate_pair(prediction, target)
    return abs(colorfulness(prediction, variant) - colorfulness(target, variant))


def to_lab(image: np.ndarray) -> np.ndarray:
    validate_rgb(image)
    return rgb2lab(image.astype(np.float64), illuminant="D65", observer="2",
                   channel_axis=-1)


def delta_e00(prediction: np.ndarray, target: np.ndarray) -> float:
    validate_pair(prediction, target)
    return delta_e00_lab(to_lab(prediction), to_lab(target))


def delta_e00_lab(prediction_lab: np.ndarray, target_lab: np.ndarray) -> float:
    """CIELAB arrays already converted through to_lab; enables reuse of GT LAB."""
    return float(deltaE_ciede2000(prediction_lab, target_lab,
                                kL=1, kC=1, kH=1, channel_axis=-1).mean())
