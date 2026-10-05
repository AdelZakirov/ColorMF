# Saved-image evaluation

Independent evaluation of saved colorizations. The saved-image pipeline has no
imports of the training model, checkpoint loading, or generation. An optional
`eval.checkpoint` runner generates predictions before invoking that pipeline.
Run commands from the repository root.

```bash
.venv/bin/python -m pip install -r eval/requirements.txt
```

## Checkpoint generation and evaluation

Install the repository's `requirements.txt` as well for checkpoint inference.
The runner reads the architecture, noise scale, EMA choice, and data resize
strategy from the checkpoint. For the supplied checkpoint:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m eval.checkpoint \
  --checkpoint /mnt/WORKSPACE/aza_workspace/ColorMF/checkpoints/pmf_b_16_256/last.ckpt \
  --ground-truth /mnt/IMAGING/HUB/DATASETS/general_datasets/imagenet/custom_val \
  --output eval/results/pmf_b_16_256_cuda \
  --sample-seeds 1 2 3 4 --device cuda \
  --inference-batch-size 4 --batch-size 8
```

Choose a **new or empty** output directory for each generation run. Defaults are
one sample (`seed_1`), automatic CUDA/CPU selection, float32, one warmup batch,
and all seven metrics. `--inference-batch-size` bounds model sampling batches;
`--batch-size` bounds metric batches. `--precision bfloat16` enables autocast.
`--limit N` selects the first N relative image IDs in sorted order. Seeds are
stable per relative image ID and independent of inference batch size, and their
CLI order defines the first sample used for FID. FID requires at least two inputs.
GPU kernels and mixed precision can introduce small rounding differences across
batch layouts, even with identical seeds.

The runner loads EMA weights when available (`--ema-variant 500` selects a
variant); `--no-ema` selects raw weights. An unready EMA fails explicitly.
`--model-config configs/pmf_b_256_colorization.yaml` overrides checkpoint model
and noise settings or supplies missing hyperparameters; strict weight loading
rejects architecture mismatches. Only load trusted PyTorch checkpoints.

`--resize-strategy center_crop` uses the training dataset's ADM center crop;
`--resize-strategy stretch` uses OpenCV cubic resize like `sample.prepare_input`.
The checkpoint's recorded strategy is the default. Both inference luminance and
evaluation GT come from the **same transformed RGB image**. GT is exported as
lossless PNG, so evaluation needs no additional resize. EXIF/ICC decoding follows
the evaluator's existing sRGB rules.

The run directory contains `predictions/seed_N/<relative_id>.png`,
`ground_truth/<relative_id>.png`, `generation.json`, and the four report files
described below. `generation.json` records source paths, selection, checkpoint
step, EMA, noise scale, precision, preprocessing, and sampling timing before
metric evaluation. If a metric download fails, the saved outputs can be evaluated
with `python -m eval` without regenerating predictions. For example:

```bash
.venv/bin/python -m eval \
  --predictions eval/results/pmf_b_16_256_cuda/predictions \
  --ground-truth eval/results/pmf_b_16_256_cuda/ground_truth \
  --layout sample-dirs --sample-ids seed_1 seed_2 seed_3 seed_4 \
  --device cuda --output eval/results/pmf_b_16_256_rescored
```

Checkpoint reports fill `timing.inference` with NFE per prediction, total measured
sampling time, amortized latency per prediction, and predictions per second.
Warmup, checkpoint loading, input decoding/device transfer, PNG encoding, and
metric time are excluded; CUDA timing synchronizes before/after `model.sample`.
This is sampling throughput for the recorded batch size, not full application
latency. Saved-image-only runs continue to report inference as unmeasured.

## Quick start

### Full-image colorization at original resolution

```bash
.venv/bin/python -m eval.original_size \
  --checkpoint /path/to/last.ckpt --input /path/to/original_images \
  --output /path/to/original_size_predictions --limit 100 \
  --seed 1 --device cuda --batch-size 4 --ema-variant 500
```

This extracts OpenCV LAB L from the original RGB image, resizes **L alone** to
the checkpoint's resolution with OpenCV cubic interpolation, then normalizes
the resized L as `L_byte / 127.5 - 1`. Predicted normalized ab is bicubically
resized back to each original image's dimensions (`align_corners=False`) and
combined with its untouched original normalized L. Only the final LAB-to-RGB
conversion rounds/clips to 8-bit. The output keeps the complete image at its
original spatial resolution, with no crop. PNG filenames preserve relative image
IDs. `generation.json` records the pipeline and each source/output size.
`--workers 4` overlaps CPU decoding and PNG export within each model batch;
it preserves sampling order and batch size. Outputs are saved atomically.
`generation.jsonl` records completed batches during generation, and
`generation_pending.json` records the settings until the final manifest is ready.
To extend an earlier first-1,000 run, use `--offset 1000 --limit 4000` with
a new output directory. The offset skips sorted image IDs before the limit;
sampling seeds remain tied to each image ID.
`--alpha 1.3` scales both continuous predicted normalized LAB chroma channels
(`a` and `b`) by 1.3 before resizing them and converting LAB to RGB; the
default `--alpha 1.0` preserves the original behavior.

To score these outputs, use `python -m eval` against the original GT directory,
without `--resize`, and use `--allow-subset` if only the first 100 were generated.
RGB conversion can change reconstructed luminance for out-of-gamut colors.

For a long full-ImageNet run, first copy `last.ckpt` to a stable checkpoint file:
training can update that pathname. The Val50k run on 2026-10-05 uses a pinned
epoch-239, step-1,050,000 checkpoint (EMA 500, seed 1, float32, batch 4), with
all 50,000 predictions regenerated because the previous Val5k used epoch 234.
Its original-size PNGs are in
`/mnt/IMAGING/HUB/DATASETS/general_datasets/imagenet/imagenet1k/source/val_set_full_cmf/original_size_50000/`.
Reports and the source decoding audit are in `eval/results/imagenet_val_original_size_50000/`.
The full dataset requires no `--allow-subset` and no external `--resize`.
The paired first-Val5k chroma-scale comparison reuses those epoch-239 alpha=1
predictions and generates alpha=1.3 with the same checkpoint, seed, and images.
Alpha scales physical CIELAB a*/b* around zero, accounting for the offset in
the model normalization. Run `.venv/bin/python -m eval.run_alpha_sweep`;
outputs and the metrics table are written under the shared Val output folder.
A local copy of both metric reports and the comparison is retained in
`eval/results/imagenet_val5k_alpha/`. The alpha=1 baseline is rechecked against
a fresh eight-image CUDA generation before the full run.

### Evaluate saved RGB predictions

One prediction per image:

```bash
.venv/bin/python -m eval \
  --predictions /path/to/predictions \
  --ground-truth /path/to/ground_truth \
  --output eval/results/run_1
```

All seven metrics run by default. Production FID and LPIPS require their own
pretrained **metric networks**, and their libraries may download weights on first
use into the standard Torch cache. They never load a ColorMF checkpoint. For a
run without any neural network loading/downloads, select only analytic metrics:

```bash
.venv/bin/python -m eval \
  --predictions /path/to/predictions \
  --ground-truth /path/to/ground_truth \
  --metrics psnr ssim colorfulness delta_colorfulness delta_e00 \
  --output eval/results/pixel_only
```

The example YAML covers `sample_celeba.py`'s seed directories:

```bash
.venv/bin/python -m eval --config eval/config.example.yaml \
  --predictions /path/to/celeba/64 \
  --ground-truth /path/to/celeba/images
```

Edit/copy the YAML for your dataset. YAML paths resolve relative to the YAML file;
CLI path overrides resolve relative to the current working directory. All CLI
options override corresponding YAML values. Boolean flags support both forms,
e.g. `--allow-subset` / `--no-allow-subset`. `--device cuda:0` changes the metric
network device; CPU is the default. `--batch-size` bounds neural metric batches.
`--workers 8` runs the per-image CPU metrics concurrently with bounded memory,
preserving input order and macro aggregation; the default is one worker.
LPIPS batches span consecutive inputs of the same size, including K=1, and
include both GT comparisons and diversity pairs without changing their order.

## Input layouts and pairing

Ground truth is always a recursive image tree. IDs are relative paths without
extensions, preserving subdirectories. `gt/class_a/one.jpg` matches image ID
`class_a/one`; file extensions can differ. Duplicate IDs (e.g. both `one.jpg`
and `one.png`) fail. Matching never depends on image order or basename alone.

| Layout | Ground truth | Predictions |
| --- | --- | --- |
| `single` (default) | `gt/class_a/one.jpg` | `pred/class_a/one.png` |
| `sample-dirs` | `gt/class_a/one.jpg` | `pred/seed_1/class_a/one.png`, `pred/seed_2/class_a/one.png` |
| `per-image` | `gt/class_a/one.jpg` | `pred/class_a/one/sample_0.png`, `pred/class_a/one/sample_1.png` |

Use `--layout sample-dirs` for `sample_celeba.py`'s `64/` or `original_size/`
prediction roots; do not include contact-sheet grids in the evaluated root.
`single` assigns sample ID `single`; the other layouts use the seed directory
or sample filename stem, respectively. Supported extensions: PNG, JPEG, BMP,
TIFF, WebP (case insensitive).

By default predictions must cover **every** GT image and have uniform K. For
a deliberately selected subset (e.g. 20 saved CelebA outputs vs a full GT root),
use `--allow-subset`; the report records all unevaluated GT IDs. Predictions
without GT always fail. Sample directories must cover the same image IDs;
missing seeds cannot silently shrink K.

`--sample-ids seed_1 seed_2 seed_3` explicitly selects and orders K samples per
image. Requested samples must exist for every input. Otherwise all samples are
selected in lexicographic order (so `seed_10` sorts before `seed_2`). For fixed-K
paper comparisons always specify the same sample list. `--allow-variable-k`
permits unequal K with a report warning; it also relaxes equal seed coverage.

## Preprocessing and metric definitions

Inputs must be saved 8-bit RGB or grayscale images. Grayscale is replicated
into RGB. EXIF orientation is applied; valid embedded ICC profiles are converted
to sRGB, and untagged RGB files are assumed sRGB. Unreadable/unusable ICC profiles and CMYK
profiles attached to RGB pixels produce a warning; their decoded RGB pixels are
preserved and assumed sRGB. CMYK pixels with a usable profile are converted
to sRGB; untagged CMYK JPEGs use Pillow's default CMYK-to-RGB mapping and emit
a warning. Transparent images require explicit compositing before evaluation.
High-bit-depth, floating-point and
multi-frame/animated files are rejected rather than silently quantized.

The common representation is **HWC float32 sRGB [0,1]**. This is gamma-encoded
sRGB, not linear RGB, BGR, or ColorMF's normalized LAB. The reusable metric API
rejects nonfinite/out-of-range values and unequal shapes; it never auto-clips.
Paired images must have equal spatial sizes. To deliberately change resolution,
set `--resize HEIGHT WIDTH` (or YAML `resize: [64, 64]`). Both GT and predictions
are bicubically resized before any metric; images already at that size are
unchanged. There is no implicit crop. `--resize-backend pillow` is the default;
`--resize-backend opencv` uses `cv2.INTER_CUBIC` on uint8 RGB, matching the resize
in ColorMF's `sample.prepare_input`. These implementations produce different
pixels when downsampling and must not be mixed within a validation protocol.

For native ColorMF outputs evaluated against original GT images, use
`resize: [64, 64]` and `resize_backend: opencv`, as in the example YAML. This
prepares GT at the model's input resolution while leaving saved 64x64 outputs
unchanged. Alternatively, export GT after the exact generation preprocessing
and evaluate it without resize. For `original_size/` predictions, keep GT and
predictions at their original resolution unless a target paper specifies another
protocol. The common ICC/EXIF decoding rules above apply to both resize backends.

| Metric | Definition / convention | Interpretation |
| --- | --- | --- |
| `fid` | FID-specific InceptionV3 pool3, 2048 features, `pytorch-fid` weights; internal bilinear 299×299 resize; float64 unbiased covariance | Lower |
| `lpips` | Official `lpips`, version 0.1; default AlexNet, RGB mapped to [-1,1]; `--lpips-net alex/vgg/squeeze` | Lower |
| `psnr` | `skimage.metrics.peak_signal_noise_ratio`, all RGB channels, data range 1 | Higher; identical images give +∞ |
| `ssim` | `skimage.metrics.structural_similarity`, RGB channel average, Gaussian 11×11 window, sigma 1.5, population covariance, K1=.01, K2=.03 | Higher |
| `colorfulness` | Opponent-channel score `hypot(std(rg),std(yb)) + 0.3*hypot(mean(rg),mean(yb))`, RGB on [0,255], population std | Vividness; higher is not necessarily more accurate |
| `delta_colorfulness` | Absolute per-image `CF(pred) - CF(GT)`; signed difference also emitted | Lower absolute difference |
| `delta_e00` | sRGB → physical CIELAB D65, 2° observer; scikit-image CIEDE2000 with kL=kC=kH=1, then mean over all pixels | Lower |

SSIM requires at least 11×11. This protocol requires at least 31×31 for AlexNet
LPIPS and 32×32 for VGG/Squeeze LPIPS, with no hidden upscaling. ΔE00 includes
lightness error from the saved RGB; it does not overwrite prediction L with GT L.

Colorfulness has two explicitly named conventions: default `absolute` matches
[DDColor's implementation](https://github.com/piddnad/DDColor/blob/master/basicsr/metrics/colorfulness.py),
using `rg=abs(R-G)` and `yb=abs((R+G)/2-B)`. `--colorfulness-variant signed` uses
the original signed opponent channels. These produce different scores; the
selected variant is written into the protocol. `colorfulness_gt` is also saved.
`delta_colorfulness` averages absolute per-image differences, which differs from
the absolute difference of dataset means (`delta_mean_colorfulness`, also saved).
`signed_delta_colorfulness` reports the signed difference (macro average); check
the definition used by your paper.

## Stochastic evaluation and aggregation

Each image has K selected samples. Per-sample rows contain paired scores. The
per-image table contains their mean, the first sample's scores, and:

- **LPIPS diversity**: average of all K(K−1)/2 unordered sample pairs. K=1 gives
  `null`, not a fabricated zero. The dataset average uses only images with K≥2,
  and records `diversity_images`. Identical multiple samples give zero diversity.
- **Best-of-K LPIPS**: minimum whole-image LPIPS to GT, with selected sample ID.
- **Best-of-K ΔE00**: minimum whole-image mean ΔE00 to GT, independently selected
  with its sample ID. It never picks a different sample for each pixel.

Dataset `aggregate.mean_over_samples` first averages selected samples within
each input and then averages inputs equally. This prevents inputs with more
samples from dominating. `aggregate.first_sample` scores just the explicitly
first sample per input, suitable for a single-output baseline.
`aggregate.stochastic` averages the per-image best-of-K and diversity scores.
PSNR is averaged in dB (mean of image PSNRs), not derived from pooled MSE.

FID is a **dataset distribution** score and has no meaningful per-image FID.
Default `--fid-sampling first` uses one first sample per input against the same
N GT images, with the same selection as `aggregate.first_sample`. Ordering is
deterministic and independent of GT quality; it never uses an oracle best sample.
Use `--sample-ids seed_3` to evaluate a specific seed. `--fid-sampling all`
uses N×K predictions against N GT images (each GT once), records both counts, and
warns about the changed protocol; variable K changes input weighting in this mode.

FID features stream through bounded batches, including the final partial batch;
mean/covariance are merged in float64 without retaining every feature. At least
two real and generated images are required. N≤2048 yields rank-deficient
covariance and a warning; such tiny synthetic-dataset scores only validate code.
The distance uses SciPy's `sqrtm` and the reference FID formula/stabilization;
calling SciPy directly also avoids `pytorch-fid 0.3.0`'s obsolete `disp=False`
call on SciPy 1.18+. Tiny negative numerical residues are clipped to zero;
significant negative/nonfinite/complex results fail.

For comparisons with published tables, match the dataset/split, GT subset,
resolution, crop, image format/compression, FID backend, LPIPS backbone,
RGB-vs-luma PSNR/SSIM, Colorfulness variant, K, and selection policy. A score with
the same name is not automatically comparable across differing protocols.
The pipeline intentionally does not claim to reproduce a paper's complete
data/inference protocol from its reported metric names alone.

In particular, the public [DDColor FID backend](https://github.com/piddnad/DDColor/blob/master/basicsr/metrics/custom_fid.py)
uses a different Inception implementation and normalization from this evaluator's
`pytorch-fid` backend. Matching DDColor's Colorfulness definition does **not** also
match its FID protocol. Production FID runs emit this warning in the console and
report. For a controlled comparison, run the saved outputs of each baseline
through this evaluator with the same GT, preprocessing and sampling policy;
direct comparisons to a published table require that table's exact protocol.

## Results and Python integration

The output directory receives:

- `report.json`: config, package versions, metric protocols, coverage, aggregate
  metrics, dataset FID, per-image/per-sample rows, warnings, and timing.
- `summary.csv`: aggregate scores with their scope.
- `per_image.csv`: sample means, first-sample scores, best-of-K values and chosen
  sample IDs, diversity and pair counts.
- `per_sample.csv`: image/sample IDs, disk paths, resolution, paired scores.

JSON is strict: mathematical infinity (identity PSNR) is encoded as the string
`"+Infinity"`; unavailable quantities are `null`; NaN never gets written. CSV
uses the same infinity string and empty cells for unavailable values. Re-running
into the same output directory replaces these four files. Output cannot live
inside an input image tree. The console prints coverage and a concise summary.

```python
from pathlib import Path
from eval.config import EvalConfig
from eval.pipeline import evaluate, evaluate_records
from eval.reporting import save_report

config = EvalConfig(Path("pred"), Path("gt"), output=Path("eval/results/run"))
report = evaluate(config)
save_report(report, config.output)
```

`evaluate_records(iterable_of_ImageRecord, config)` separates evaluation from
record discovery. The checkpoint runner generates/saves predictions and
passes `ImageRecord` / `SampleRecord` instances directly without changing metrics
or aggregation. Such records must already be selected/ordered; directory-related
config options apply in `discover()`. `progress(done, total)` is an optional API
callback after each paired-image evaluation.

`timing.evaluation_seconds` measures evaluation (including metric initialization
and FID extraction, excluding discovery/report writing). `timing.inference`
for saved-image-only evaluation contains `status: not_measured` and null NFE,
latency, throughput.
Checkpoint measurements populate that separate block with warmup, device
synchronization, and a recorded workload;
disk decoding/metric runtime must not masquerade as model latency.

## Offline tests and reproducible technical smoke report

```bash
.venv/bin/python -m pytest -q eval/tests
.venv/bin/python -m eval.tests.smoke --output eval/results/smoke
```

Tests cover identities, known PSNR/LAB/ΔE00 references, Colorfulness conventions,
normalization, shape/range failures, streaming covariance/parity with the FID
reference, directory pairing/coverage, multi-sample aggregation, best-of-K
selection, native ColorMF resize parity, LPIPS batching across inputs and sizes,
CLI/YAML overrides, JSON/CSV serialization, checkpoint raw/EMA loading, preserved
RoPE buffers, sampling parity, crop parity with training, seed/batch independence,
and checkpoint CLI output reuse. The smoke command
creates four 64×64 mock GT images and two samples per image, then runs every
metric path with deterministic substitute LPIPS/Inception networks. Both the
report and console mark these injected backends as **technical validation only**.
No pretrained metric weights or ColorMF weights are loaded or downloaded by
the tests/smoke run. Neural metric values there are not real LPIPS/FID scores.

## References

- [Paper comparison](paper_comparison.md): selected recent colorization papers,
  published metric tables, and their correspondence to this evaluator.

- [LPIPS paper and official implementation](https://github.com/richzhang/PerceptualSimilarity)
  (Zhang et al., CVPR 2018).
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid), including FID-specific
  Inception weights and preprocessing; original [FID paper](https://arxiv.org/abs/1706.08500)
  (Heusel et al., NeurIPS 2017).
- [scikit-image metrics documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
  for PSNR and Wang et al.'s SSIM settings;
  [color documentation](https://scikit-image.org/docs/stable/api/skimage.color.html)
  for physical LAB and CIEDE2000.
- [CIEDE2000 implementation notes and supplementary reference pairs](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/)
  (Sharma, Wu, Dalal, 2005).
- [DDColor colorfulness evaluation source](https://github.com/piddnad/DDColor/blob/master/basicsr/metrics/colorfulness.py)
  and [DDColor paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kang_DDColor_Towards_Photo-Realistic_Image_Colorization_via_Dual_Decoders_ICCV_2023_paper.pdf)
  (Kang et al., ICCV 2023); Hasler/Suesstrunk's Colorfulness score.
- [GCP-Colorization evaluation protocol](https://github.com/ToTheBeginning/GCP-Colorization)
  for examples of dataset/resolution/crop-sensitive FID/CF comparisons.
