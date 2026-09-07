Implement a minimal but mathematically correct conditional Pixel MeanFlow model for probabilistic image colorization in PyTorch + PyTorch Lightning.

# 1. Goal

Build the smallest clean implementation that lets us answer:

> Can a small Pixel MeanFlow transformer trained from scratch learn the conditional distribution `p(ab | L)` and produce strong stochastic colorization in one neural-network forward pass?

Input:

```text
L ∈ R[B,1,H,W]
```

from CIELAB.

Generate:

```text
ab ∈ R[B,2,H,W]
```

Desired distribution:

```text
p(ab | L)
```

Priorities:

1. correct pMF mathematics;
2. simple model;
3. successful multi-GPU DDP training;
4. genuine stochastic conditional generation;
5. 1-NFE inference.

Different random seeds for the same `L` must be able to produce different plausible `ab`.

The generative model must be trained from scratch.

Use:

* PyTorch;
* PyTorch Lightning;
* transformer backbone;
* Pixel MeanFlow.

Do not use U-Net, DDPM, VAE, latent diffusion, ControlNet, text encoders, or pretrained image generators.

Do not try to reproduce every engineering feature from the pMF paper in V1.

---

# 2. Existing dataset

Dataset code already exists here:

```text
/mnt/WORKSPACE/aza_workspace/palette/
```

This is a legacy DDPM project. Ignore its DDPM model but inspect and reuse the dataset implementation where sensible.

Determine:

* RGB loading;
* LAB conversion;
* normalization;
* crop/resize;
* augmentation;
* train/validation split;
* DataLoader behavior.

Do not unnecessarily rebuild the data pipeline.

The dataset contains millions of images, so avoid designs requiring all dataset metadata in RAM.

---

# 3. References and audit

Use the official pMF implementation as the primary source of truth.

Paper:

```text
https://arxiv.org/abs/2601.22158
```

Official repository:

```text
https://github.com/Lyy-iiis/pMF
```

Official PyTorch branch:

```text
https://github.com/Lyy-iiis/pMF/tree/torch
```

Improved MeanFlow reference:

```text
https://arxiv.org/abs/2512.02012
```

SiT may be consulted only for clean transformer implementation ideas:

```text
https://github.com/willisma/SiT
```

Before implementing the MeanFlow objective:

1. inspect the official code;
2. record exact commit SHAs of repositories used;
3. identify the exact reference implementation of:

   * interpolation;
   * `(r,t)` sampling;
   * clean-x / average-velocity parameterization;
   * auxiliary instantaneous velocity;
   * JVP;
   * stop-gradient;
   * loss;
   * one-step sampling;
4. add a short README mapping:

```text
paper equation / concept -> our PyTorch function -> reference implementation
```

Do not replace the official formulation with an approximate generic Flow Matching implementation.

---

# 4. Conditional formulation

The stochastic state contains only chroma.

Let:

```text
x = real ab
epsilon ~ N(0, I)
```

Use the pMF interpolation:

```text
z_t = (1 - t) * x + t * epsilon
```

`L` is fixed conditioning.

Model input:

```python
model_input = torch.cat([z_t, L], dim=1)
```

giving spatial channels:

```text
[z_t_a, z_t_b, L]
```

Core invariant:

> `L` is conditioning only. The stochastic state, noise, interpolation, prediction target and JVP spatial tangent operate only on `ab`.

Never noise, predict, reconstruct, or include `L` in the generative state.

When differentiating the MeanFlow objective, `L` is fixed.

---

# 5. Core pMF mathematics

This is the critical part.

Mirror the current official pMF formulation.

The network uses the pMF clean-x / average-velocity path plus the auxiliary instantaneous-velocity prediction used by current pMF training.

Conceptually:

```text
f_theta(z, r, t; L) -> predicted clean ab
```

with corresponding average velocity:

```text
u_theta(z, r, t; L)
```

using the exact boundary/endpoint treatment from the reference implementation.

Do not introduce unsafe raw division at `t=0`.

For the linear interpolation:

```text
v_target = epsilon - x
```

or the exactly equivalent stabilized expression used by the reference.

The auxiliary predicted instantaneous velocity provides the spatial direction of the JVP.

The core structure must match the reference logic:

```text
v_dir = predicted instantaneous velocity

J = JVP of u_theta
    with respect to (z, t, r)
    along (v_dir, 1, 0)

L is fixed.

V_theta =
    u_theta
    + (t - r) * stop_gradient(J)
```

Main MeanFlow loss:

```text
V_theta vs instantaneous velocity target
```

Also retain the small auxiliary loss:

```text
predicted v vs velocity target
```

Do not implement CFG or class conditioning in V1. `L` is always present.

Remove ImageNet-specific CFG machinery rather than adapting it.

---

# 6. JVP implementation

Prefer native PyTorch forward-mode AD:

```python
torch.func.jvp
```

Keep the pMF mathematical core isolated, for example:

```text
src/meanflow/objective.py
```

Do not bury JVP logic inside the Lightning training step.

The code should make it obvious:

* what the primals are;
* what the tangents are;
* that `L` is fixed;
* where `detach` / stop-gradient occurs;
* what the velocity target is.

Correct mathematics matters more than using an optimized attention backend.

If an optimized attention implementation is incompatible with forward-mode AD, fall back to standard attention. Do not alter the MeanFlow equations to accommodate the backend.

---

# 7. Mandatory mathematical tests

A finite loss is not sufficient validation.

Implement:

### Analytical JVP test

Create a deliberately simple toy function with a hand-computable JVP.

Compare:

```text
analytical JVP
vs
torch.func.jvp
```

using a tight FP32 tolerance.

### Independent pMF reference test

Implement a tiny reference version of the pMF equations independent of the transformer and compare it against the production implementation for:

* forward values;
* JVP values;
* loss values;
* gradients.

Run these tests in FP32 first.

Only after FP32 correctness passes should BF16 training be tested.

---

# 8. Model: pMF-Tiny

Use a small DiT/SiT/pMF-style transformer.

Initial target:

```text
resolution: 256x256
patch size: 16
depth: ~12
heads: ~8
total training parameters: roughly 30M-50M
```

First implement the chosen transformer block correctly, then calculate the exact parameter count.

Choose hidden width based on the resulting count rather than hard-coding a guessed width.

Report separately:

```text
total training parameters
inference-required parameters
auxiliary-v-head parameters
```

The exact size is less important than staying in the intended Tiny regime.

Conditioning must remain simple:

```python
torch.cat([z_t, L], dim=1)
```

then patchify.

Do not add cross-attention, separate `L` encoders, ControlNet, DINO, CLIP, or semantic encoders.

---

# 9. LAB representation and conversion

Centralize LAB handling.

At minimum implement/test:

```text
normalize_L
denormalize_L
normalize_ab
denormalize_ab
LAB -> RGB
```

Reuse the existing dataset conversion if it is already correct.

Document the actual dataset normalization.

Do not add unexplained:

```text
tanh
sigmoid
clamp
```

to the model output.

LAB normalization is a representation choice, not a reason to artificially bound the generative network.

Generated LAB is:

```text
[L_original, generated_ab]
```

Guarantee:

```text
output_lab[:, L] == input_lab[:, L]
```

up to tensor numerical precision.

For visualization, explicitly convert:

```text
normalized ab -> physical ab -> LAB -> RGB
```

Document gamut handling and clipping.

Do not claim that LAB -> RGB conversion preserves luminance exactly: out-of-sRGB-gamut colors may be clipped.

This is a visualization concern for V1, not a training objective.

---

# 10. PyTorch Lightning and DDP

Implement a minimal:

```text
PMFColorizerModule(pl.LightningModule)
```

Use a DataModule only if it helps reuse the existing data layer.

DDP is required for V1 and must use Lightning-native distributed training. Do not implement custom process groups or gradient synchronization.

Support:

* one-GPU debugging;
* multi-GPU DDP;
* checkpoint save/resume;
* BF16 mixed precision;
* distributed logging.

Use the appropriate Lightning strategy for the installed version, e.g.:

```text
strategy="ddp"
```

Ensure:

* distributed sampling works correctly;
* validation outputs are not duplicated across ranks;
* only the appropriate rank writes image grids and checkpoints;
* metrics are synchronized correctly;
* fixed-seed sampling is independent of batch partitioning;
* no rank performs unnecessary duplicate dataset-wide work.

---

# 11. Optimizer and precision

Use AdamW for V1.

Configurable:

```text
learning rate
weight decay
warmup
gradient clipping
```

Choose conservative initial values based on similar transformer training and document them as experiment settings, not pMF requirements.

Do not implement Muon yet.

Mathematical/unit tests:

```text
FP32
```

Normal training:

```text
bf16-mixed
```

If needed for numerical stability, allow JVP-related scalar math, divisions, and loss reductions to run in FP32 under BF16 autocast.

Document explicit casts.

Follow the reference endpoint policy rather than requiring undefined raw `t=0` divisions to work.

---

# 12. Sampling

Implement true one-step generation.

Example:

```python
generated_ab = model.sample(
    L,
    seed=42,
)
```

and:

```python
samples = model.sample(
    L,
    seeds=[1, 2, 3, 4],
)
```

Standard inference must require:

```text
1 NFE
```

meaning one main generative-model evaluation.

Count/log main model evaluations so this can be tested.

Reproducibility invariant:

```text
same image_id + seed -> same result
```

independently of:

* batch size;
* GPU count;
* DDP rank.

---

# 13. Minimal validation

V1 validation should remain lightweight.

Log:

* pMF loss;
* auxiliary velocity loss;
* total loss;
* learning rate;
* gradient norm;
* samples/sec;
* GPU count;
* global batch size.

For a small fixed list of validation image IDs, save:

```text
L | GT | seed1 | seed2 | seed3 | seed4
```

Derive fixed seeds from:

```text
image ID + sample index
```

so results remain identical across batching and GPU configurations.

Write qualitative grids once, from the appropriate rank.

Do not build semantic-selection infrastructure. Fixed manually chosen validation IDs are sufficient.

---

# 14. V1 non-goals

Do not implement unless required for core correctness:

```text
Muon
EMA
CFG
class conditioning
LPIPS training
ConvNeXt/VGG perceptual training
adversarial loss
colorfulness loss
FID
KID
CIEDE2000
best-of-K metrics
semantic image selection
full ImageNet benchmark
Hugging Face export
Gradio
large-scale training
scaling studies
```

Also do not include overfit experiments or synthetic multimodality experiments in this implementation brief.

Those will be separate follow-up experiments after the implementation works.

Do not build abstractions for hypothetical future features.

---

# 15. Implementation order

Follow this order.

## Stage 1: audit

Inspect:

* existing dataset;
* official pMF implementation;
* repository commit SHAs;
* equation -> code mapping;
* chosen transformer block;
* expected parameter count;
* installed Lightning version and DDP API.

Write a short audit note before substantial implementation.

## Stage 2: mathematical core

Implement:

* LAB utilities;
* interpolation;
* `(r,t)` sampling;
* pMF objective;
* auxiliary velocity prediction;
* JVP;
* stop-gradient;
* one-step sampling;
* analytical/reference tests.

Run the FP32 mathematical tests.

Do not proceed if they fail.

## Stage 3: model + Lightning + DDP

Implement:

* pMF-Tiny;
* exact parameter reporting;
* LightningModule;
* AdamW;
* checkpoint save/resume;
* qualitative image logging;
* Lightning-native DDP;
* synchronized metrics;
* rank-safe validation.

Run:

* one-batch single-GPU smoke test;
* one-batch multi-GPU DDP smoke test;
* checkpoint save/resume test;
* one-step sampling test;
* cross-batch/DDP seed reproducibility test.

## Stage 4: real pilot

Run a small real-data pilot using the intended multi-GPU configuration.

Generate fixed qualitative grids.

Report:

* training command;
* hardware;
* GPU count;
* global/per-GPU batch size;
* exact parameter count;
* throughput;
* losses;
* qualitative outputs;
* anything not actually tested.

Never claim verification for something that was only implemented.

Use:

```text
NOT TESTED
```

when appropriate.

---

# 16. Repository structure

Respect the existing project layout where sensible.

Otherwise keep additions small, approximately:

```text
src/
    model.py
    pmf.py
    lab.py
    lightning_module.py

tests/
    test_lab.py
    test_pmf_math.py
    test_sampling.py
    test_distributed_smoke.py

configs/
    pilot.yaml

train.py
sample.py
README.md
```

Do not create a large framework.

---

# 17. V1 completion criteria

V1 is complete only when:

1. reference repositories and commit SHAs are documented;
2. pMF equations are mapped to our implementation;
3. analytical JVP test passes in FP32;
4. independent pMF reference-comparison tests pass;
5. LAB tests pass;
6. exact parameter counts are reported;
7. one training batch succeeds on one GPU;
8. one training batch succeeds under multi-GPU DDP;
9. checkpoint save/resume works;
10. distributed metrics behave correctly;
11. one-step sampling uses exactly 1 main model evaluation;
12. seed reproducibility holds across batch sizes and DDP ranks;
13. a real-data multi-GPU pilot has been run;
14. fixed qualitative validation outputs have been generated.

Overfit and synthetic multimodality experiments are explicitly not required for V1 completion.

---

# Core principle

Optimize for answering one scientific question correctly, not for feature completeness:

```text
noise_ab + fixed L
        ↓
small conditional Pixel MeanFlow transformer
        ↓
stochastic ab
        ↓
1 NFE
```

The implementation must also work in the actual intended training setup:

```text
multiple GPUs
    ↓
Lightning-native DDP
    ↓
correct synchronized training and validation
```

Prefer a small implementation whose pMF mathematics and distributed behavior are trustworthy over a feature-rich implementation whose core objective is uncertain.

