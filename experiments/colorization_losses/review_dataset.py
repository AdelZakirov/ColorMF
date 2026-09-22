#!/usr/bin/env python3
"""Audit generated files and produce a side-by-side raw/projection review gallery."""
import argparse
import csv
import hashlib
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.colorization_losses.dataset import rgb_to_lab


def review(root, expected, partial=False):
    def stored_path(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    protocol = json.loads((root / 'generation_protocol.json').read_text())
    records = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()]
    by_sample = {}
    for row in records:
        sample = Path(row['source']).parent.name
        by_sample.setdefault(sample, {})[row['type']] = row
    if not partial and len(by_sample) != expected:
        raise ValueError(f'Expected {expected} sources, got {len(by_sample)}')
    rows = []
    panels = []
    expected_types = {'x_gt','wrong_hue','low_saturation','color_bleeding','subtle_color_error',
                      'semantically_wrong_color','plausible_alternative_colorization'}
    for sample, variants in sorted(by_sample.items()):
        if set(variants) != expected_types:
            if partial:
                continue
            raise ValueError(f'{sample}: incomplete variants')
        for variant, row in variants.items():
            with Image.open(stored_path(row['output'])) as im:
                im.verify()
            if variant == 'x_gt' or 'qwen_model' not in row:
                continue
            raw = stored_path(row['raw_output'])
            if hashlib.sha256(raw.read_bytes()).hexdigest() != row['raw_sha256']:
                raise ValueError(f'{sample}: raw checksum mismatch')
            if row['qwen_seed'] != protocol['seed'] or row['source_chroma_magnitude_reused']:
                raise ValueError(f'{sample}: protocol mismatch')
            generated = np.asarray(Image.open(raw).convert('RGB'))
            chroma = np.linalg.norm(rgb_to_lab(generated)[..., 1:], axis=-1)
            # Screening flag only: grayscale may be valid for some scenes.
            near_gray = float(chroma.mean()) < 2.0
            rows.append({'sample_id': sample, 'variant': variant, **row['diagnostics'],
                         'raw_mean_chroma': float(chroma.mean()),
                         'review_flag': 'near_grayscale' if near_gray else '',
                         'review_status': row['semantic_label_status']})
        alt = variants['plausible_alternative_colorization']
        wrong = variants['semantically_wrong_color']
        columns = [('GT', variants['x_gt']['output']), ('Qwen condition', alt['condition_output']),
                   ('Plausible: raw', alt['raw_output']), ('Plausible: fixed L', alt['output']),
                   ('Wrong: raw', wrong['raw_output']), ('Wrong: fixed L', wrong['output'])]
        images = ''.join('<figure><img loading="lazy" src="'+html.escape(os.path.relpath(stored_path(path), root), quote=True)
                         +'"><figcaption>'+label+'</figcaption></figure>' for label,path in columns)
        panels.append('<section><h2>'+html.escape(sample)+'</h2><div class="row">'+images+'</div></section>')
    with (root/'generation_diagnostics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    summary = {'partial': partial, 'source_count': len(rows)//2, 'variant_count': len(rows)//2*7,
               'qwen_count': len(rows), 'seed': protocol['seed'],
               'max_projected_mean_abs_delta_L': max(r['projected_mean_abs_delta_L'] for r in rows),
               'mean_raw_abs_delta_L': float(np.mean([r['raw_mean_abs_delta_L'] for r in rows])),
               'near_grayscale_candidates': [{'sample_id': r['sample_id'], 'variant': r['variant']}
                    for r in rows if r['review_flag']],
               'human_semantic_validation': 'pending; file and numeric checks do not establish plausibility'}
    (root/'dataset_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen loss dataset review</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#eee}.row{display:flex;flex-wrap:wrap}'
        'figure{margin:4px}img{width:200px;height:200px;object-fit:contain}section{margin-bottom:30px}'
        'figcaption{text-align:center}</style><h1>Qwen generation v2 — seed '+str(protocol['seed'])+'</h1>'
        '<p>'+('PARTIAL PREVIEW — generation in progress. ' if partial else '')+'Unverified candidates. Inspect material colors, boundaries and raw→fixed-L changes. '
        'A prompt label is not a human plausibility label.</p>'+''.join(panels))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_dir',type=Path)
    parser.add_argument('--expected',type=int,default=154)
    parser.add_argument('--partial', action='store_true', help='Preview completed samples during generation')
    args=parser.parse_args();review(args.dataset_dir.resolve(),args.expected,args.partial)
