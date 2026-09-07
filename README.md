# ColorMF

ColorMF is a minimal conditional Pixel MeanFlow (pMF) colorizer. The only
stochastic state is normalized chroma `ab`; normalized luminance `L` is
concatenated as a fixed condition:

```text
z_t = (1 - t) * ab + t * epsilon
model input = cat(z_t, L)
```

The model is a from-scratch DiT-style transformer and standard inference uses
one evaluation at `(r=0, t=1)`. It has no U-Net, diffusion schedule, VAE,
latent state, CFG, pretrained image encoder, or bounded output activation.

After loading a checkpoint into `PMFColorizerModule`, sampling is exposed as:

```python
ab = model.sample(L, seed=42, image_ids=["image-001"])
ab_samples = model.sample(
    L[:1], seeds=[1, 2, 3, 4], image_ids=["image-001"]
)
lab = model.sample_lab(L[:1], seed=42, image_ids=["image-001"])
```

## Reference audit

The primary source of truth is the official pMF PyTorch implementation:

* paper: https://arxiv.org/abs/2601.22158
* repository: https://github.com/Lyy-iiis/pMF
* PyTorch branch: https://github.com/Lyy-iiis/pMF/tree/torch
* improved MeanFlow reference: https://arxiv.org/abs/2512.02012

The implementation audit was performed against the official PyTorch branch at
the commit recorded below. The exact SHA is kept here so future changes do not
silently drift from the reference:

```text
official pMF repository SHA: 75f6073042c21f7104686261a0c4784db4ede9d1
official pMF torch-branch SHA: 990e81a84249dbd68a128accef67eb95621d10b1
official Improved MeanFlow SHA: bf60cd7cb653f6628e59d48034b333c5eba445e2
```

The official `torch` branch is inference-only; the training audit therefore
uses the official JAX `main` commit for `(r,t)` sampling and the objective,
with the torch commit checked for the equivalent clean-x conversion and
one-step update. ColorMF deliberately omits CFG and all ImageNet-specific
conditioning. Its transformer embeds only `h=t-r`, while `t` still enters
the clean-x velocity conversion.

Audited reference locations are `pmf.py:150-181` for pair sampling,
`pmf.py:385-397` for interpolation and the stabilized target,
`pmf.py:411-428` for the JVP/stop-gradient compound field,
`pmf.py:426-462` for adaptive losses, and
`models/pmfDiT.py:336-378` for clean-x/velocity heads and endpoint
conversion. The torch branch's corresponding one-step solver is
`pmf.py:78-145`.

The equation-to-code map is:

| pMF concept | ColorMF implementation | Reference role |
| --- | --- | --- |
| `z_t = (1-t)x + t epsilon` | `src/pmf.py:interpolate` | linear probability path |
| ordered logit-normal `(r,t)` sampling | `src/pmf.py:sample_rt` | pMF `sample_tr` |
| clean-x output | `PMFTiny.clean_head` | data endpoint estimate |
| average velocity | `src/pmf.py:average_velocity` | `(z-x_hat)/clip(t,0.05,1)` |
| auxiliary instantaneous velocity | `PMFTiny.velocity_head` plus `average_velocity` | sampled-interval v loss and h=0 tangent |
| JVP primals/tangents | `src/pmf.py:jvp_average_velocity` | `(z,r,t)` along `(v_dir,0,1)` with fixed L |
| stop-gradient | `src/pmf.py:meanflow_terms` | `jvp.detach()` in corrected velocity |
| main and auxiliary losses | `src/pmf.py:meanflow_terms` | adaptive summed velocity losses |
| one-step sampler | `PMFTiny.sample` | one main model evaluation |

The independent test oracle in `tests/test_pmf_math.py` repeats the equations
without using the transformer implementation and compares forward values,
JVPs, losses, and parameter gradients.

The production target uses the reference stabilized form
`(z_t-x)/clip(t, 0.05, 1)`, which equals `epsilon-x` away from the low-time
endpoint. Main and auxiliary residuals are summed per example and adaptively
normalized with `S / stop_gradient((S + 0.01)^1)`. The configurable
`auxiliary_weight` defaults to one, matching the official pMF sum; no
perceptual losses are enabled in V1.

The pilot explicitly uses the published 256px B/16 logit-normal recipe
`p_mean=0.8`, `p_std=0.8`, with the flow-matching diagonal proportion and
uniform replacement probability configurable under `training.time_sampling`.

## LAB convention

The legacy `/mnt/WORKSPACE/aza_workspace/palette` loader was audited. It reads
RGB with OpenCV, uses a random 256 crop followed by resize, brightness/
contrast or CLAHE, horizontal flip, and optional border/texture/noise
augmentation, converts with `cv2.COLOR_RGB2LAB`, and applies
`Normalize(max_pixel_value=127.5)`. ColorMF preserves the conversion and
uses crop/resize plus horizontal flip only. It intentionally conditions on
OpenCV LAB `L`, not the legacy loader's `ToGray(RGB)` output, and its training
crop is always selected when the source image is large enough. These are
intentional differences rather than a claim of legacy-equivalent preprocessing:
photometric and synthetic noise augmentation would change the conditional
color distribution rather than merely regularize geometry.

* `L_norm = L_opencv / 127.5 - 1`;
* `ab_norm = ab_opencv / 127.5 - 1`;
* physical `L* = (L_norm + 1) * 50`;
* physical `a*/b* = (ab_norm + 1) * 127.5 - 128`.

`src/lab.py` is the single conversion boundary. `compose_lab` copies the
original `L` tensor unchanged. `lab_to_rgb` clips only the OpenCV byte
encoding for visualization; out-of-sRGB-gamut colors can therefore be
clipped, and no luminance-preservation claim is made for that display step.

For very large datasets, use a line manifest. `IndexedManifest` stores only
byte offsets and opens one image lazily per sample instead of loading image
metadata or pixels into RAM.

## pMF-Tiny size

The default 256x256 configuration uses patch size 16, hidden size 384, depth
12, 8 heads, and MLP ratio 4. The exact report from `PMFTiny().parameter_report()`
is:

```text
total training parameters: 32,905,600
inference-required parameters: 32,708,480
auxiliary-v-head parameters: 197,120
```

The auxiliary velocity head is omitted from the one-step sampling forward
pass. The clean head and transformer trunk remain required for inference.

## Training and sampling

Install PyTorch, PyTorch Lightning, OpenCV, Pillow, NumPy, and PyYAML in the
target environment. Then configure manifests or roots in
`configs/pilot.yaml`:

```bash
python train.py --config configs/pilot.yaml
python sample.py --config configs/pilot.yaml \
  --checkpoint checkpoints/last.ckpt \
  --input input.jpg --output colorized.png --seed 42
```

Set `training.devices` to an integer greater than one to use Lightning-native
`strategy="ddp"`. Training metrics use `sync_dist=True`; validation metrics
are deduplicated by image ID after gathering padded distributed shards.
Lightning owns distributed sampling and checkpointing, and only global rank
zero writes qualitative grids after collecting requested examples from every
rank. Objective RNG streams are rank-separated and their generator state is
stored in checkpoints. Data-order and augmentation state are not promised to
resume bit-for-bit.

Conservative AdamW, warmup, clipping, and BF16 settings in the pilot config
are experiment settings, not pMF requirements. Mathematical tests are FP32;
the objective explicitly casts adaptive loss reductions to FP32 under BF16
autocast.

## Validation status in this checkout

Implemented and runnable:

* FP32 analytical JVP test;
* independent pMF forward/JVP/loss/gradient comparison;
* fixed 2-D patch positional representation;
* LAB normalization and RGB conversion tests;
* one-step sampling, NFE count, and batch/individual seed reproducibility;
* single-batch transformer pMF smoke test;
* Lightning-native DDP configuration and checkpoint callback;
* Python 3.13 environment with the mounted dataset adapter;
* one real-data BF16 Lightning training batch at the configured 256x256 shape
  on an RTX 4090;
* checkpoint save/resume through Lightning, including objective RNG state and
  protected training-configuration checks;
* one-step GPU sampling with stable image-ID/seed noise and exact luminance
  preservation;
* fixed validation qualitative-grid generation.

`NOT TESTED`: this checkout exposes one GPU, so a multi-GPU DDP smoke test,
distributed metric run, and real multi-GPU pilot remain unverified. The mounted
dataset and single-GPU Lightning path are validated; the DDP and pilot commands
are provided for an environment with at least two visible GPUs.
