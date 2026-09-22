# Colorization loss experiment

This directory is the self-contained final experiment: generation and evaluation
code, 154-source dataset, numerical results, reports, and regression tests. The
main analysis is [`results/review.md`](results/review.md), with a standalone
[`results/review.html`](results/review.html) version. Dataset inspection is in
[`data/gallery.html`](data/gallery.html).

The old plausible set is a teal/amber prompt followed by GT chroma magnitude
replacement. It must not be treated as a validated set of natural alternative
colorizations. Correcting loss implementations or severity matching does not
repair this dataset confound.

The generator shares the natural-colorization prompt with `qwen_image_edit.py`.
Defaults: seed **42**, Qwen-Image-2.1, 40 steps,
256×256, CFG 1, no negative prompt. Input geometry and PIL grayscale conversion
match the standalone script. PIL grayscale is luminance-like intensity, not
literal CIELAB L*. Ground-truth RGB colors are not given to Qwen.
The two prompts are evaluated together in one pipeline batch for each source;
this cuts runtime by roughly three times. It changes floating-point batching
slightly, but the raw output remains visually equivalent to the standalone
single-prompt call (the measured mean absolute pixel difference in the check was
3.9/255).

Both semantic-wrong and plausible candidates use the same seed, sampling and
postprocessing protocol; only the requested color plausibility changes. The
plausible prompt does not require a particular palette or distance from GT.
A plausible answer may agree closely with GT. Do not reject it simply to create
a larger difference or select candidates using the losses being evaluated.

Each sample stores:

- `x_gt.png` and the four deterministic synthetic variants.
- `raw/condition.png`: actual Qwen grayscale input.
- `raw/plausible_alternative_colorization.png` and
  `raw/semantically_wrong_color.png`: untouched model outputs, before resizing
  or Lab conversion.
- Top-level Qwen variant PNGs: original source L plus Qwen ab, with necessary
  gamut compression. **No GT chroma magnitude replacement.** PNG quantization
  causes small residual L errors; the evaluator constructs exactly fixed-L
  chroma leaves for derivatives.
- Manifest entries: exact prompt, seed, guidance settings, raw checksum,
  raw/output sizes, projection diagnostics, and `unverified_candidate` status.
- `generation_protocol.json`: resume is rejected when the protocol differs.

The committed final dataset is in `data/` and the evaluated outputs are in
`results/`. The evaluator defaults point to these paths. Generation and
loss-method versions are separate; both final protocols are version 2.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv_diff/bin/python -m experiments.colorization_losses.generate \
  --input /mnt/IMAGING/HUB/DATASETS/general_datasets/flickr_small/256 \
  --output-dir experiments/colorization_losses/data --types all --seed 42 \
  --qwen-device cuda --qwen-cpu-offload --resume

.venv/bin/python -m experiments.colorization_losses.review_dataset \
  experiments/colorization_losses/data --expected 154

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv_diff/bin/python -m experiments.colorization_losses.evaluate \
  --device cuda --batch-size 4 --local-files-only

.venv/bin/python -m experiments.colorization_losses.summarize \
  experiments/colorization_losses/results

.venv/bin/python -m pytest -q experiments/colorization_losses/tests
```

For a new output directory omit `--resume`. To change the seed or any recorded
protocol setting, use a new output directory; do not mix generations.

## Quality interpretation

Raw vs fixed-L images are shown side by side in `gallery.html`. Check recognizable
material colors, object boundaries, fine details and any chroma changes caused
by gamut projection. The manifest's numerical diagnostics identify changes in
L and ab; they cannot determine semantic plausibility. Prompt labels are not
human labels. Both original endpoints and severity-matched examples need human
review before making claims about semantic discrimination. Fixed-L projection
cannot repair displaced chroma boundaries if Qwen changes scene geometry.

Seed 42 is a reproducible single draw per input, not a guarantee of the best
colorization. Exact bitwise reproducibility also depends on hardware and library
versions. A training recommendation still requires held-out training ablations.
