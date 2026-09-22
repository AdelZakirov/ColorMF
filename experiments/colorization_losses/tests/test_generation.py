"""Regression tests for the natural-colorization benchmark protocol."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.colorization_losses import generate as generator
from experiments.colorization_losses.dataset import (
    PLAUSIBLE_PROMPT,
    compose_with_source_l,
    prompts_for_qwen,
    rgb_to_lab,
)


class FakeTorch:
    class Generator:
        def __init__(self, device):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self


def test_qwen_defaults_match_standalone_without_negative_guidance():
    args = generator.build_parser().parse_args(['--input', 'a.png', '--output-dir', 'out'])
    assert (args.seed, args.qwen_true_cfg_scale, args.qwen_steps, args.qwen_resolution) == (42, 1., 40, 256)
    calls = []
    image = Image.new('RGB', (16, 16))

    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[image])

    generator._qwen_edit(pipe, FakeTorch, 'cpu', image, PLAUSIBLE_PROMPT, 42, 40, 256, 1.)
    assert calls[0]['true_cfg_scale'] == 1.
    assert 'negative_prompt' not in calls[0]
    assert calls[0]['generator'].seed == 42
    assert prompts_for_qwen()['plausible_alternative_colorization'] == PLAUSIBLE_PROMPT
    assert 'teal' not in PLAUSIBLE_PROMPT and 'amber' not in PLAUSIBLE_PROMPT


def test_projection_preserves_generated_chroma_not_gt_magnitude():
    source = np.full((8, 8, 3), [180, 100, 50], dtype=np.uint8)
    generated = np.full_like(source, [120, 120, 120])
    projected = compose_with_source_l(source, generated)
    assert np.linalg.norm(rgb_to_lab(projected)[..., 1:], axis=-1).mean() < 1
    assert np.linalg.norm(rgb_to_lab(source)[..., 1:], axis=-1).mean() > 30


def test_raw_projection_resume_and_protocol_guard(tmp_path, monkeypatch):
    source = tmp_path / 'input.png'
    Image.new('RGB', (20, 30), (180, 100, 50)).save(source)
    out = tmp_path / 'out'
    calls = []

    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new('RGB', (16, 16), (100, 120, 180))])

    monkeypatch.setattr(generator, '_load_qwen', lambda *a, **kw: (pipe, 'cpu', FakeTorch))
    argv = ['generate', '--input', str(source), '--output-dir', str(out),
            '--types', 'plausible_alternative_colorization', '--qwen-resolution', '32']
    monkeypatch.setattr(sys, 'argv', argv)
    generator.main()
    records = [json.loads(line) for line in (out/'manifest.jsonl').read_text().splitlines()]
    record = records[-1]
    assert record['source_chroma_magnitude_reused'] is False
    assert record['semantic_label_status'] == 'unverified_candidate'
    assert Image.open(out / record['raw_output']).size == (16, 16)
    assert Image.open(out / record['output']).size == (32, 32)
    assert record['diagnostics']['projected_mean_abs_delta_L'] < 1
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    generator.main()
    assert len(calls) == 1
    # Missing raw artifacts must be regenerated, not silently skipped.
    from pathlib import Path
    (out / record['raw_output']).unlink()
    generator.main()
    assert len(calls) == 2
    monkeypatch.setattr(sys, 'argv', argv + ['--resume', '--seed', '999'])
    with pytest.raises(ValueError, match='protocol'):
        generator.main()
