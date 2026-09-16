# ColorMF

ColorMF is a faithful PyTorch port of official Pixel Mean Flow B, adapted from
RGB generation to conditional `p(ab | L)` colorization. The stochastic state is
normalized CIELAB chroma only:

```text
z_t = (1 - t) ab + t epsilon
network input = concat(z_t, L)
network outputs = u_ab, v_ab
```

`L` is a fixed condition: it is never noised, interpolated, predicted, or used
as a JVP primal/tangent. There are no class labels, CFG, semantic encoders,
cross-attention, or ControlNet-style paths.

The model also supports dedicated luminance conditioning. Set
`model.conditioning.mode` to `separate` to embed `z_ab` and `L` with independent
bottleneck patch embedders, fuse their aligned spatial tokens residually, and
optionally reinject the projected luminance tokens before each transformer
block. `concat` is the default and remains compatible with existing checkpoints:

```yaml
model:
  conditioning:
    mode: concat  # concat | separate
    reinject: true
```

## Pinned references

- JAX training/objective: `Lyy-iiis/pMF@75f6073042c21f7104686261a0c4784db4ede9d1`
- PyTorch architecture: `Lyy-iiis/pMF@990e81a84249dbd68a128accef67eb95621d10b1`
- ColorMF starting point: `513626c5a0c61b8e89214ab3f5daed35f49fec18`
- Optax Muon/NAdam reference immediately preceding the pMF commit:
  `google-deepmind/optax@e3a96e9487d1a9c670b67395f02603439e52905b`

The model port preserves `BottleneckPatchEmbedder`, scaled-variance
`TorchLinear`, RMSNorm, QK RMSNorm, spatial-only 2-D RoPE, SwiGLU, learned
position embeddings, four time prefix tokens, zero vector residual gates,
zero final projections, and the shared/deep-dual-head topology.

For every faithful config the topology is:

```text
8 shared blocks
  +-- 8 u blocks -> two-channel u output
  +-- 8 v blocks -> two-channel v output (training only)
```

Patch-dependent parameter reports are:

| geometry | total | shared | u branch | v branch | inference shared+u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/4 | 171,092,928 | 57,744,768 | 56,674,080 | 56,674,080 | 114,418,848 |
| 128/8 | 171,259,008 | 57,763,200 | 56,747,904 | 56,747,904 | 114,511,104 |
| 256/16 | 171,923,328 | 57,836,928 | 57,043,200 | 57,043,200 | 114,880,128 |

## Faithful experiment configs

| config | name | geometry | status |
| --- | --- | --- | --- |
| `configs/pmf_b_64_colorization.yaml` | pMF-B/4-64 | 64 / 4 = 16x16 tokens | resolution adaptation |
| `configs/pmf_b_128_colorization.yaml` | pMF-B/8-128 | 128 / 8 = 16x16 tokens | resolution adaptation |
| `configs/pmf_b_256_colorization.yaml` | pMF-B/16-256 | 256 / 16 = 16x16 tokens | official B/16 geometry, conditional task |

All use the published B/16 training settings: `P_mean=0.8`, `P_std=0.8`,
`data_proportion=0.5`, adaptive `p=1`, epsilon `0.01`, Muon at `1e-3`
with Adam beta2 `0.95`, 320 epochs, constant LR, LPIPS-VGG `0.4`,
ConvNeXt-V2-Base-22k `0.1`, perceptual cutoff `t<0.8`, and EDM EMA half-lives
`[500, 1000, 2000]` kimg. Batch size 8 with accumulation 128 targets global
batch 1024 on one process; change both values together for the available GPU
count so their product with world size remains 1024.

`configs/pilot.yaml` is a small infrastructure smoke config using the same
architecture implementation; it is not a faithful pMF-B capacity preset.

## Objective and inference

`src/pmf.py` follows official `pmf.py`:

```text
v_target = (z_t - ab) / clip(t, 0.05, 1)
u = (z_t - xhat_u) / clip(t, 0.05, 1)
v = (z_t - xhat_v) / clip(t, 0.05, 1)
v_dir = v(z_t, h=0)
J = JVP[u(z,t,r); (stop_grad(v_dir), 1, 0)]
V = u + (t-r) stop_grad(J)
loss = adaptive(||V-v_target||^2) + adaptive(||v-v_target||^2)
```

The perceptual reconstruction is `ab_hat = z_t - t*u`. Predicted and target
chroma are combined with the same original `L`, denormalized to physical LAB,
and converted by Kornia 0.8.3 `lab_to_rgb(..., clip=False)`. Inspection of
that version confirms this disables its final RGB `[0,1]` clamp (Kornia still
applies the standard internal non-negative `fz` guard during LAB-to-XYZ).
LPIPS receives
`2*RGB-1`; matching official pMF, ConvNeXt receives that same `[-1,1]` tensor
directly without an additional ImageNet mean/std transform. No clamp is applied to
generated chroma or to Kornia RGB before the frozen loss networks.
The paired random resized crop receives the same checkpointed objective
generator as `(r,t)` and noise, so resumed training also reproduces perceptual
crop sampling.

Sampling is exactly one NFE. For `z_1 ~ N(0,I)`, `t=1`, `r=0`, the network
runs its shared and u branches once and returns `ab = z_1-u`. The v branch is
not executed. CPU per-image noise keyed by `(image_id, seed)` keeps the random
stream independent of batching and DDP rank.

## EMA and optimizer

`EMAManager` preserves checkpoint/save/restore/scoped-evaluation behavior and
adds all three official EDM shadows. Decay uses actual global images processed:

```text
half_life = min(configured_kimg * 1000, images_seen * 0.05)
beta = 0.5 ** (images_in_this_update / half_life)
```

Validation and `sample.py --ema-variant {500,1000,2000}` can select any shadow;
`--no-ema` selects raw weights. Fixed-decay EMA remains available only via
`type: fixed`.

`src/optimizer.py` ports `optax.contrib.muon`: 2-D matrices use bias-corrected
Nesterov momentum, five-step Frobenius-preconditioned Newton-Schulz
orthogonalization, and Optax width scaling. Non-2-D tensors use
Nesterov-AdamW, matching
Optax's default parameter partition. The unavoidable framework difference is
PyTorch's transposed linear-kernel storage; width scaling explicitly maps its
shape back to `(fan_in, fan_out)`. AdamW remains an explicit legacy option.

## Data and commands

The FFHQ adapter preserves the existing OpenCV LAB encoding and now uses the
official pMF-style aspect-preserving center crop. Horizontal flip is disabled
in the faithful configs, matching the pinned official B/16 default. No color
augmentation is used.

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/python train.py --config configs/pmf_b_128_colorization.yaml
./venv/bin/python sample.py --config configs/pmf_b_128_colorization.yaml \
  --checkpoint checkpoints/pmf_b_8_128/last.ckpt \
  --input input.jpg --output colorized.png --seed 42 --ema-variant 1000
./venv/bin/pip install -r requirements-dev.txt  # Optax parity test only
./venv/bin/python -m pytest -q
```

PyTorch Lightning retains BF16, DDP, deterministic objective RNG checkpointing,
MLflow, qualitative grids, and checkpoint resume.

Validation runs at the end of each epoch by default. To validate every fixed
number of optimizer steps, set `training.validation_check_interval_steps`; the
launcher converts it through `accumulate_grad_batches`, so for example
`validation_check_interval_steps: 100` with accumulation 64 validates every
6,400 train batches, or every 100 optimizer steps.

Training-step metrics are sent to MLflow every optimizer step by default via
`training.log_every_n_steps: 1`; increase this value only when reducing metric
history volume is more important than a dense loss curve.

Training can resume both the Lightning checkpoint state and an existing MLflow
run. Pass the checkpoint and run ID explicitly; the MLflow tracking URI comes
from the config:

```bash
./.venv/bin/python train.py \
  --config configs/pmf_s_64_colorization.yaml \
  --resume checkpoints/pmf_s_4_64/last.ckpt \
  --mlflow-run-id 47b32d0328924e1baa571351a54a6136
```

The run ID can also be set as `training.logger.run_id` in YAML. The CLI value
takes precedence. When attaching an existing run, the launcher does not log
the flattened config parameters again because MLflow parameters are immutable;
runtime metadata tags and new metrics continue in the same run.

The mathematically equivalent diagonal optimization is controlled by
`training.split_diagonal_jvp`; it avoids JVP work for samples where `r == t`
and defaults to `false`. On the current RTX 4090 S-model benchmark (batch 20,
BF16), it measured `0.2108 s/step` versus `0.1894 s/step` for the legacy path,
so it remains an A/B option rather than the default.

An experimental compile path is available with
`training.compile_mode: max-autotune-no-cudagraphs`. It compiles the model
forward and auxiliary direction in place, preserving checkpoint parameter
names. Leave it `null` for the eager baseline until a representative run has
confirmed wall-clock and memory behavior. On the same benchmark, compiled
legacy measured `0.191 s/step` after a `45.2 s` compile, while compiled split
measured `0.241 s/step` after a `157.6 s` compile.

## Audit map

| Official pMF component | ColorMF implementation | adaptation and reason |
| --- | --- | --- |
| `models/embedder.py:BottleneckPatchEmbedder` | `src/model.py:BottleneckPatchEmbedder` | input is `[z_ab,L]`, still 3 channels |
| `TimestepEmbedder` + four time tokens | `src/model.py:TimestepEmbedder`, `time_tokens` | only `h=t-r`; ImageNet/CFG tokens removed |
| RMSNorm, QK norm, RoPE attention | `RMSNorm`, `RoPEAttention` | RoPE still touches spatial tokens only |
| SwiGLU + zero vector gates | `SwiGLUMlp`, `TransformerBlock` | exact architecture behavior |
| 8 shared / 8 u / 8 v | `shared_blocks`, `u_blocks`, `v_blocks` | heads output 2-channel chroma endpoints |
| clean-x velocity conversion | `PixelMeanFlowB._velocity` | conversion is performed only for `ab` |
| JAX pMF JVP and adaptive losses | `src/pmf.py:meanflow_terms` | `L` is closed over and fixed |
| auxiliary LPIPS / ConvNeXt | `src/perceptual.py` | losses operate on LAB-reconstructed RGB |
| Optax Muon | `src/optimizer.py:Muon` | PyTorch tensor layout mapped explicitly |
| EDM multi-EMA | `src/ema.py:EMAManager` | step*1024 generalized to actual images |

## Explicit deviations from official pMF

1. RGB ImageNet generation becomes two-channel `ab` generation conditioned by
   concatenating fixed one-channel `L`; class/CFG conditioning is removed.
2. FFHQ replaces ImageNet and uses the established ColorMF LAB normalization.
3. 64/4 and 128/8 are documented resolution adaptations, not published presets.
4. Perceptual RGB is obtained through differentiable Kornia LAB conversion;
   after mapping to `[-1,1]`, LPIPS and ConvNeXt preprocessing follows the
   official auxiliary-loss behavior.
5. The optimizer and distributed runtime are PyTorch/Lightning ports. Muon
   follows Optax math, but kernels and distributed execution use PyTorch layouts.

No other task-level architecture or objective redesign is intentional.
