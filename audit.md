# Independent audit

**Do not begin substantial training. The production JVP implements the wrong time derivative, and the tests reproduce that mistake.**

Audited commit: `94b5ae252b2d5aaef396e560b14c9c4b6295a31f`.

I read all project source, configuration, tests, README, and the mounted legacy dataset implementation. **No project files were modified.** No existing project training logs or checkpoints were present.

Executed:
- Existing suite: **10 passed, 1 skipped**.
- Independent FP32 analytical/reference checks.
- Two-process CPU/Gloo Lightning DDP.
- One real-data, 256x256 BF16 training step on RTX 4090.
- Checkpoint restoration and continuation comparison.
- Instrumented sampling, attention/JVP, and CUDA batching checks.

Runtime: PyTorch `2.14.0+cu130`, Lightning `2.6.5`, OpenCV `5.0.0`, NumPy `2.5.3`.

## Reference provenance

| Official source | Exact commit |
|---|---|
| pMF main - training | `75f6073042c21f7104686261a0c4784db4ede9d1` |
| pMF torch - inference | `990e81a84249dbd68a128accef67eb95621d10b1` |
| iMF main | `bf60cd7cb653f6628e59d48034b333c5eba445e2` |
| JAX 0.4.34 - clipping derivative rules | `affba367c5533df8900e32cbc3d31ca92dd1c1ea` |

Papers inspected: [pMF v3](https://arxiv.org/html/2601.22158v3), [iMF v2](https://arxiv.org/html/2512.02012v2). SiT was not used.

The official pMF torch branch is **inference-only**. Training comparisons therefore use pinned official JAX code, particularly [the objective](https://github.com/Lyy-iiis/pMF/blob/75f6073042c21f7104686261a0c4784db4ede9d1/pmf.py#L389-L436), [head conversion](https://github.com/Lyy-iiis/pMF/blob/75f6073042c21f7104686261a0c4784db4ede9d1/models/pmfDiT.py#L336-L378), and [training RNG](https://github.com/Lyy-iiis/pMF/blob/75f6073042c21f7104686261a0c4784db4ede9d1/train.py#L37-L42).

## A. Mathematical correctness

### A1 - BLOCKER: JVP differentiates destination time instead of current time

**Location:** `src/pmf.py:131-147`, used at `186-188`; incorrect documentation also at `README.md:69`.

**Current behavior**
```python
primals  = (z, r, t)
tangents = (v_dir.detach(), 1, 0)
```

This computes:

$$
D_z u[v_{\rm dir}] + \partial_r u.
$$

**Official behavior:** The reference orders arguments as `(z,t,r)` and uses `(v_dir,1,0)`. With this project's argument order, the correct tangent is:

```python
(v_dir.detach(), zeros_like(r), ones_like(t))
```

It must compute:

$$
D_z u[v_{\rm dir}] + \partial_t u,
$$

holding `r` and `L` fixed.

For this h-only network, $h=t-r$, the error both **reverses the interval-embedding derivative and omits the time-denominator derivative**.

**Executed evidence:** An independently derived FP32 oracle produced:

| Quantity | Production | Correct reference |
|---|---:|---:|
| Total loss | 1.9991715 | 1.9992610 |
| Clean-scale parameter gradient | 0.0144832 | 0.6657651 |
| Interval parameter gradient | 0.0189591 | 0.3290360 |

Maximum JVP error: **2.263279**. Nearly identical displayed losses conceal radically different gradients.

**Smallest fix:** Swap the two time tangents; correct the documentation and replace the erroneous test oracle.

**Existing tests catch it?** **No.** They encode the same wrong ordering.

### A2 - MEDIUM: Time-sampling recipe silently differs from the published pMF preset

**Location:** `src/pmf.py:42-44,174-177`; `configs/pilot.yaml`.

**Current:** Training always uses logit-normal `(-0.4,1.0)`, inherited through function defaults. The Lightning/configuration path cannot select another distribution.

**Reference:** The official [256px B/16 preset](https://github.com/Lyy-iiis/pMF/blob/75f6073042c21f7104686261a0c4784db4ede9d1/configs/pMF_B_16_config.yml#L4-L10) uses `(0.8,0.8)`. The former values do exist as upstream class defaults, so this is **not an intrinsically invalid distribution**, but it is an undisclosed recipe difference.

**Smallest fix:** Explicitly configure the selected distribution and document whether reproducing the pMF preset or intentionally adapting it.

**Existing tests catch it?** No sampling-distribution test exists.

### Mathematical components that do match

- Interpolation: $z_t=(1-t)x+t\epsilon$.
- Ordering: $r\le t$.
- Half-batch diagonal assignment happens **before sorting**, matching pMF.
- Main head predicts image-like chroma; $u=(z-\hat x)/\operatorname{clip}(t,0.05,1)$.
- Auxiliary head also predicts an image-like quantity before velocity conversion.
- JVP direction comes from the auxiliary head at `r=t`.
- Auxiliary loss uses its prediction at the sampled interval.
- Stabilized target is $(z-x)/\operatorname{clip}(t,0.05,1)$, **not** exactly $\epsilon-x$ below `0.05`.
- JVP correction is detached.
- Main and auxiliary squared residuals are independently summed per example, adaptively weighted, and batch-averaged. Default auxiliary coefficient is one.

These correct components do not compensate for A1.

## B. Conditional colorization formulation

**The ab-only adaptation itself is coherent.**

At `src/pmf.py:166-188` and `src/model.py:190-215`:
- Noise, interpolation, targets, predictions, and spatial JVP directions have two chroma channels.
- `L` is concatenated unchanged on every model evaluation.
- `L` is closed over, not a JVP primal.
- Both heads receive the same conditioning.
- There is no ground-truth `ab` conditioning bypass.
- No luminance reconstruction loss or generated luminance exists.

### B1 - HIGH: Transformer has no spatial positional representation

**Location:** `src/model.py:156-168,205-210`.

**Current:** Patch tokens enter global attention without positional embeddings, relative positions, or RoPE.

**Risk:** The model cannot distinguish arbitrary rearrangements of whole patches except through patch content. It has intra-patch pixel coordinates, but no representation of inter-patch geometry or adjacency.

**Executed evidence:** After making residual gates nonzero, arbitrary joint patch permutation of `(z,L)` permuted the output identically, within **5.96e-8**.

**Reference:** Official pMF uses positional embeddings and 2-D RoPE. Simplifying its architecture is permitted here; silently eliminating all inter-patch position information is a consequential restriction.

**Smallest fix:** Add a fixed 2-D positional embedding before the transformer blocks. No capacity increase or architectural redesign is needed.

**Existing tests catch it?** No.

### Architecture and parameter accounting

Actual trainable tensor counts:

| Category | Parameters |
|---|---:|
| Total training | **32,905,600** |
| Inference-required | **32,708,480** |
| Training-only auxiliary head | **197,120** |

This is within the requested Tiny regime.

Patchification/unpatchification and two-channel output shapes are consistent. The implementation uses LayerNorm, GELU, adaLN-style gates, and two linear output heads - not the official RMSNorm/SwiGLU/separate-branch architecture. Output weights are initialized normally rather than upstream's zero initialization. These are architectural adaptations, not independently demonstrated mathematical failures.

## C. PyTorch/JVP/numerics

### Exact production differentiation contract

- **Primals:** `(z,r,t)`.
- **Actual tangents:** `(detached auxiliary velocity,1,0)` - wrong, as A1 explains.
- **Fixed:** `L`, model parameters as forward-mode coordinates, and the separately computed direction.
- **Stops:** Direction at `src/pmf.py:146`; complete JVP correction at `188`; adaptive denominators at `194`.
- **Reverse gradients:** Retained through the primal main prediction and auxiliary prediction.
- **Target:** Not explicitly detached, but has no gradient path with ordinary dataset tensors.

### C1 - LOW: Exact clipping-boundary derivatives differ from reference

**Location:** `src/pmf.py:115-118`.

**Current:** Installed PyTorch's `clamp` derivative is zero at exactly `t=0.05` and `t=1`.

**Reference:** Pinned JAX's min/max clipping convention gives derivative `0.5` at those ties.

**Risk:** After correcting A1, exact-boundary JVPs still differ. This is low severity because ordinary FP32 logit-normal draws almost never hit these boundaries; one-step inference does not differentiate them.

**Smallest fix:** Use the corresponding min/max composition and test boundary derivatives.

**Existing tests catch it?** No; the endpoint test only checks a value at `t=0.01`.

### Executed numerical findings

- Values at `t=0`, `0.01`, `0.05`, and `1` were finite: **no raw division-by-zero defect found**.
- Attention is explicit matmul/softmax, not SDPA/Flash.
- Attention forward-mode versus reverse-mode JVP error: **5.96e-8**.
- Whole small transformer, correct-direction FP32 forward/reverse JVP error: **4.17e-7**.
- Under BF16 autocast, model projections run BF16; state/time tensors, converted velocities, JVP result, correction, and loss are FP32.
- A nonzero-gate small-model BF16/FP32 JVP comparison had approximately **0.53% relative L2 difference**. FP32 output dtype does not mean the entire derivative was computed in FP32.
- Real-data BF16 training produced finite gradients.

I found no attention-backend incompatibility in the installed runtime. Long-run numerical stability remains unverified.

## D. Lightning/DDP

For Lightning-specific findings below, official pMF supplies a JAX training system - not an equivalent Lightning implementation or guarantee.

### D1 - MEDIUM: Fixed validation images on nonzero ranks are discarded

**Location:** `src/lightning_module.py:179-195`.

**Current:** Only rank zero collects examples, then all ranks gather their dictionaries. Nonzero-rank dictionaries are empty.

**Risk/evidence:** In the two-process test, rank one processed IDs `1,3,0` but collected none. Only IDs `0,2,4` produced grids.

**Reference/requirement:** This violates the project's rank-independent fixed-validation requirement.

**Smallest fix:** Collect requested examples on every rank; retain rank-zero-only writing.

**Existing tests catch it?** No; the DDP test has no validation loader.

### D2 - MEDIUM: Distributed validation counts padded duplicates

**Location:** `train.py:68`; `src/data.py:168-175`; `src/lightning_module.py:163-177`.

**Current:** Lightning's `DistributedSampler` pads validation to equal shard lengths. Synchronized metrics include those duplicates.

**Risk/evidence:** Five examples became six observations across two ranks: ID `0` appeared twice. The configured real validation set also contains only **five images**, so the bias is material.

**Reference/requirement:** Metric synchronization does not guarantee unique-example aggregation.

**Smallest fix:** Deduplicate per-image validation contributions before reduction, or evaluate this tiny validation set once without distributed padding.

**Existing tests catch it?** No.

### D3 - MEDIUM: Training RNG streams are shared across ranks

**Location:** `train.py:27`; `src/lightning_module.py:83-88`; `src/pmf.py:170-177`.

**Current:** The objective uses default RNGs after identical process seeding; no rank-specific generator is passed.

**Executed evidence:** Both CPU DDP ranks had identical RNG states before each training step. Consequently equal-shaped batches reuse the same noise/time random draws across ranks.

**Risk:** Individual draws remain marginally valid, but distributed Monte Carlo samples are correlated.

**Reference:** Official pMF folds both training step and device-axis index into its RNG.

**Smallest fix:** Give the objective an explicit rank-separated RNG stream, with checkpointed state or deterministic step/microbatch keys. Keep validation randomness separate.

**Existing tests catch it?** No. GPU-specific stream behavior was not executed across multiple GPUs.

### D4 - MEDIUM: Resume restores training state, but not equivalent stochastic continuation

**Location:** `train.py:27,32-40,70`; `src/lightning_module.py:228-246`; `src/data.py:98-107,165`.

**Verified restored:** Both heads, all model weights, Adam moments, optimizer step, LR, scheduler position, and Lightning global step.

**Not restored:** Explicit RNG/data-stream state. Restarting also reconstructs the scheduler lambda from current configuration.

**Executed evidence:** Same-configuration resumed and uninterrupted runs both reached step two, but their final parameters differed by up to **0.0003480874**.

**Reference:** Official objective randomness is keyed to step/device; Lightning state restoration alone does not reproduce that behavior.

**Smallest fix:** Clearly define statistical versus exact resume. Exact continuation additionally needs objective RNG and data-order/augmentation-state handling; prevent silent scheduler-configuration changes on resume.

**Existing tests catch it?** There is no checkpoint/resume test.

### D5 - MEDIUM: Seed guarantee is stronger than actual CUDA reproducibility

**Location:** `src/model.py:250-295`; `tests/test_sampling.py:21-34`.

**Current:** CPU-generated noise is correctly keyed by `image_id + seed`. Network results still depend numerically on batching.

**Executed CUDA differences, batched versus individual:**
- FP32 maximum difference: **2.38e-7**.
- BF16 maximum difference: **0.001953125**.

**Risk:** The strict "same result regardless of batch size" claim is false if it means bitwise equality.

**Reference:** pMF does not establish cross-device bitwise reproducibility.

**Smallest fix:** Specify and test a numerical tolerance. If exact equality is mandatory, enforce a fixed per-image execution shape/precision rather than promising arbitrary batched equivalence.

**Existing tests catch it?** No; they only check a small CPU model with tolerance.

### D6 - MEDIUM: Logged losses conceal learning progress

**Location:** `src/lightning_module.py:91-114,163-177`.

**Current:** Logs report the adaptively normalized optimization losses, each approximately one.

**Evidence:** Full-resolution training reported total loss `1.9999995`; A1's severely wrong gradients also barely changed the displayed loss.

**Reference:** Official pMF separately logs **unweighted velocity MSEs**, while retaining adaptive weighting for optimization.

**Smallest fix:** Keep the training objective; additionally log raw main and auxiliary velocity MSEs.

**Existing tests catch it?** No logging assertions exist.

### Distributed behavior that did work

Two CPU/Gloo ranks completed two steps with:
- Lightning-native sharding and synchronized metrics;
- disjoint training shards in the divisible-length test;
- finite gradients;
- identical fixed-image samples across ranks.

Configured nominal global batch is:

$$
8 \times \text{world size} \times \text{accumulation}.
$$

The pilot currently selects one device and accumulation one. Final partial batches can be smaller. Multi-GPU/NCCL and accumulated-gradient behavior remain unverified.

## E. Dataset/LAB

**Production normalization is correct for OpenCV uint8 LAB:**

$$
L_n=L_8/127.5-1,\qquad ab_n=ab_8/127.5-1
$$

$$
L^*=(L_n+1)50,\qquad ab^*=(ab_n+1)127.5-128.
$$

Thus neutral physical chroma maps to approximately `0.00392157`, not exactly zero.

The executable path is:
- BGR decode -> RGB;
- geometric preprocessing;
- OpenCV uint8 LAB;
- FP32 normalized tensors;
- unrestricted generated `ab`;
- unchanged normalized `L` concatenated with generated chroma.

`compose_lab` preserves `L` exactly. Display conversion separately rounds/clamps LAB byte encoding and converts to RGB. That includes **LAB encoding-range clipping**, not merely sRGB gamut clipping. It does not affect training tensors.

The configured dataset contains **25,388 training images and five validation images**, not the full legacy multi-million-image mixture.

### E1 - LOW: Legacy-dataset compatibility claims are inaccurate

**Location:** `README.md:85-104`; legacy `/mnt/WORKSPACE/aza_workspace/palette/src/data/transforms.py:284-305,338-352`.

**Current claim:** The new loader preserves the legacy LAB/geometric convention.

**Actual legacy behavior:** Chroma comes from LAB, but conditioning comes from `ToGray(RGB)`, **not LAB L**. Legacy cropping is probabilistic; the new loader always crops eligible training images.

**Risk:** Misleading comparisons with legacy training. The new LAB-L conditioning is nevertheless the correct choice for this specification.

**Reference:** Official pMF does not prescribe LAB preprocessing.

**Smallest fix:** Document these intentional changes; do not restore the legacy grayscale condition.

**Existing tests catch it?** No legacy-equivalence test exists.

## F. Tests

### F1 - HIGH: Mathematical tests provide false confidence

**Location:** `tests/test_pmf_math.py:14-25,33-65,70-86`.

- The analytical test checks PyTorch arithmetic directly, not production JVP wiring.
- It explicitly tests differentiation along `r`, reproducing the wrong interpretation.
- The "independent" reference copies the production tangent-order bug.
- Its toy network ignores both time arguments, weakening interval-conditioning coverage.

**Reference:** Official pMF requires differentiation in current time with destination fixed.

**Smallest fix:** Use an independent time-dependent analytical field and correct reference coordinates; compare JVPs and parameter gradients. Include a regression that fails when time tangents are swapped.

**Would current tests catch A1?** No - they all passed during this audit.

Other important coverage limitations:

| Test | What it actually establishes | What it misses |
|---|---|---|
| LAB round trips | Some inverse consistency | Absolute convention errors can cancel |
| NFE test | A counter equals one | Counter is assigned `1`, not measured |
| Seed test | Small CPU reproducibility | CUDA/BF16, mixed IDs, distributed partitioning |
| Device-count test | `device_count >= 0` | Tautological; no distributed evidence |
| DDP smoke | Optional training-only execution | Validation, unique aggregation, RNG independence, resume |
| Checkpoint/resume | **Absent** | All continuation guarantees |

Independent hooks measured **one patch-embedding/trunk call, one clean-head call, zero auxiliary-head calls** during sampling. Thus **actual standard sampling is genuinely 1 NFE**, despite the weak existing test.

Training currently performs three model calls: primal prediction, boundary auxiliary prediction, and JVP evaluation. These are not hidden inference calls.

## G. Minor engineering issues

### G1 - MEDIUM: Sampling CLI fails for already-correct-size inputs

**Location:** `sample.py:41-47`.

**Current:** `L` gains a batch dimension only inside the resize branch.

**Executed evidence:** An input matching the configured resolution failed with:
```text
ValueError: L must be [B,1,H,W] and match z
```
A differently sized input succeeded.

The CLI also assumes resolution is iterable, although the model accepts an integer.

**Reference/requirement:** Sampling requires `[B,1,H,W]`; upstream has no equivalent LAB CLI.

**Smallest fix:** Always add the batch dimension and use the constructed model's normalized resolution. Align resize preprocessing with the dataset while touching this path.

**Existing tests catch it?** No CLI test exists.

### G2 - LOW: Throughput is not end-to-end training throughput

**Location:** `src/lightning_module.py:82-89,115-121`.

**Current:** Timing covers objective construction, omits backward/optimizer/data loading, lacks explicit CUDA completion, and averages rank-local rates.

**Risk:** Reported samples/sec cannot justify compute budgeting.

**Reference:** Official pMF measures completed training intervals.

**Smallest fix:** Measure completed step intervals and report actual global examples processed.

**Existing tests catch it?** No.

---

## Minimal ordered patch plan before substantial compute

1. **Correct A1 and replace the mathematical oracle.** Require analytical, forward-value, and parameter-gradient agreement.
2. **Add spatial positional information.**
3. **Fix distributed validation collection/duplicate accounting and separate training RNG streams by rank.**
4. **Make the chosen time distribution explicit and log raw velocity errors.**
5. **Fix CLI input handling; establish tested sampling and resume contracts without overstating exact reproducibility.**
6. Run the corrected FP32/BF16 suite and a real **two-GPU/NCCL** train-validate-checkpoint-resume smoke test before a larger pilot.

The clipping-tie correction and documentation updates are small accompanying fixes, not reasons for redesign.

## VERIFIED

- Correct ab-only state and fixed-L conditioning structure.
- Actual parameter counts and appropriate Tiny scale.
- Actual 1-NFE sampling; auxiliary head omitted.
- Correct production LAB encoding and exact assembled-L preservation.
- Independent proof of the production JVP/gradient error.
- Explicit attention forward AD in FP32 and BF16 execution.
- One real-data 256px BF16 training step with finite gradients.
- CPU/Gloo DDP training and the reported validation/RNG defects.
- Model, Adam, scheduler, and step restoration - not exact continuation.
- Sampling CLI failure and CUDA batching differences.

## NOT VERIFIED

- Corrected production pMF mathematics: no patch was applied.
- Multi-GPU/NCCL, H100/H200, or multi-node behavior.
- Exact data/RNG continuation or cross-GPU sampling equality.
- Gradient accumulation in distributed execution.
- Long-run numerical stability, convergence, or plausible multimodal colorization.
- Historical training claims without retained artifacts.

NOT SAFE FOR REAL TRAINING
