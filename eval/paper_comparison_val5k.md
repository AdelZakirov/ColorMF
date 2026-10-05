# ColorMF versus published ImageNet validation 5k results

Checked 5 October 2026. These are published reference values alongside our
completed original-resolution run, not a benchmark with identical preprocessing
and metric backends. All rows use 5,000 ImageNet validation images, but the
selection and evaluation resolution can differ. Baseline models were not run here.
`—` means no value in the cited table. PSNR is in dB; LPIPS measures distance to GT.
CF measures colorfulness; a higher CF alone does not establish better quality.

| Model / publication | 5k selection | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CF | ΔCF ↓ | ΔE00 ↓ | Source of values |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **ColorMF, EMA 500, seed 1** | First 5k, original resolution | **9.7225** | **22.3355** | **0.9045** | **0.1812** | **25.9583** | **12.6924**\* | **11.6270** | [Our report](results/imagenet_val_original_size_5000/report.json) |
| MultiColor, ACM MM 2024 | First 5k | 2.17 | 24.69 | — | — | 38.24 | 0.03 | — | [Table 2, §5.1](https://arxiv.org/html/2408.04172) |
| SeAda, IJCAI 2025 | val5k | 3.36 | 24.53 | — | — | 36.98 | 1.23 | — | [Table 1](https://www.ijcai.org/proceedings/2025/0106.pdf) |
| DDColor-large, ICCV 2023 | val5k | 3.92 | 23.85 | — | — | 38.26 | 0.05 | — | [Table 1, val5k columns](https://arxiv.org/html/2212.11613) |
| L-CAD, NeurIPS 2023, scarce description | First 5k, center crop 256×256 | 4.36 | 24.47 | 0.92 | 0.16 | 34.04 | 3.68 | — | [Table 3, §4.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) |
| ColorFormer, ECCV 2022 | 5k comparison in L-CAD | 4.64 | 23.14 | 0.89 | 0.18 | 37.95 | 0.23 | — | [L-CAD Table 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) |
| CT², ECCV 2022 | 5k comparison in L-CAD | 5.51 | 23.50 | 0.92 | 0.19 | 38.48 | 2.17 | — | [L-CAD Table 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf) |
| UniColor, TOG 2022, unconditional | Random 5 per class | 9.46 | — | — | 0.1945 | 39.01 | — | — | [Tables 1–2, §5](https://arxiv.org/html/2209.11223) |

\* ColorMF's table ΔCF is the absolute difference of dataset means,
`abs(mean(CF_pred) - mean(CF_GT))`, matching the convention used in the
DDColor/MultiColor comparisons. Our mean absolute **per-image** ΔCF is **18.1038**;
GT mean CF is **38.6507**. Published ΔCF values are preserved as reported;
L-CAD's exact aggregation convention is not established by the cited description.

ColorMF predicts ab from resized 256×256 original LAB L, then restores ab to each
source's original size and reconstructs RGB with the original L. Metrics use
original-size RGB with no external resize/crop. In contrast, L-CAD explicitly
uses 256×256 center crops, and MultiColor specifies 256×256 images. UniColor's
balanced random subset differs from our first-5k subset. These differences prevent
claims of superiority from the numeric ordering alone.

Our FID uses TensorFlow-compatible `pytorch-fid` InceptionV3 pool3/2048; DDColor's
public FID implementation uses a different normalization/backend. Our LPIPS is
official v0.1 AlexNet. See the [full protocol comparison](paper_comparison.md#соответствие-текущему-evaluator)
and [run summary](results/imagenet_val_original_size_5000/README.md).

ColorFormer and CT² rows deliberately retain all metrics from the same L-CAD
table. For example, CT² ΔCF is 2.17 there, versus 0.27 in DDColor/MultiColor's
table; combining those values would mix published experiments. L-CAD receives a
generic scarce description rather than a detailed description of GT colors.

The strongest practical concern in our run is low colorfulness: mean CF 25.96
versus GT 38.65. This supports the observed muted output. A direct baseline
comparison requires running the baseline on the same image IDs through the same
evaluation protocol.
