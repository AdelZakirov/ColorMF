import numpy as np

from experiments.colorization_losses.dataset import (
    QWEN_VARIANTS,
    SYNTHETIC_VARIANTS,
    compose_with_source_l,
    generate_synthetic,
    lab_to_rgb,
    recolor_lab,
    rgb_to_lab,
)


def test_lab_roundtrip_is_close_for_in_gamut_gradient():
    y, x = np.mgrid[:32, :32]
    rgb = np.stack([x * 7, y * 7, (x + y) * 3], axis=-1).clip(0, 255).astype(np.uint8)
    restored = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(restored.astype(np.int16) - rgb.astype(np.int16)).max() <= 2


def test_synthetic_variants_change_chroma_and_keep_lab_l_exact():
    rgb = np.full((24, 32, 3), [210, 100, 50], dtype=np.uint8)
    rgb[8:16, 8:24] = [40, 150, 220]
    lab = rgb_to_lab(rgb)
    for variant in SYNTHETIC_VARIANTS:
        changed = recolor_lab(lab, variant)
        assert np.array_equal(changed[..., 0], lab[..., 0])
        assert not np.array_equal(changed[..., 1:], lab[..., 1:])


def test_generated_images_keep_geometry_and_are_deterministic():
    rgb = np.zeros((19, 27, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(27, dtype=np.uint8)[None, :]
    rgb[..., 1] = 80
    rgb[..., 2] = 170
    first = generate_synthetic(rgb)
    second = generate_synthetic(rgb)
    assert tuple(first) == SYNTHETIC_VARIANTS
    assert all(np.array_equal(first[key], second[key]) for key in SYNTHETIC_VARIANTS)
    assert all(value.shape == rgb.shape and value.dtype == np.uint8 for value in first.values())


def test_qwen_projection_uses_source_luminance():
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    source[...] = [180, 100, 50]
    generated = np.zeros_like(source)
    generated[...] = [80, 100, 180]
    projected = compose_with_source_l(source, generated)
    source_l = rgb_to_lab(source)[..., 0]
    projected_l = rgb_to_lab(projected)[..., 0]
    assert np.abs(source_l - projected_l).max() < 1.5
    assert QWEN_VARIANTS == ("semantically_wrong_color", "plausible_alternative_colorization")


def test_plausible_projection_can_keep_source_chroma_strength():
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    source[...] = [180, 100, 50]
    generated = np.zeros_like(source)
    generated[...] = [80, 100, 180]
    projected = compose_with_source_l(source, generated, preserve_source_chroma=True)
    source_chroma = np.linalg.norm(rgb_to_lab(source)[..., 1:], axis=-1)
    projected_chroma = np.linalg.norm(rgb_to_lab(projected)[..., 1:], axis=-1)
    assert np.abs(source_chroma - projected_chroma).mean() < 1.0
