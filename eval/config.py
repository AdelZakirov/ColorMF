"""Validated evaluation protocol, shared by the CLI and Python API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


METRICS = ("fid", "lpips", "psnr", "ssim", "colorfulness",
           "delta_colorfulness", "delta_e00")


@dataclass(frozen=True)
class EvalConfig:
    predictions: Path
    ground_truth: Path
    output: Path = Path("eval/results")
    metrics: tuple[str, ...] = METRICS
    layout: str = "single"
    resize: tuple[int, int] | None = None  # height, width; both image sets
    sample_ids: tuple[str, ...] | None = None
    allow_subset: bool = False
    allow_variable_k: bool = False
    device: str = "cpu"
    batch_size: int = 32
    workers: int = 4
    lpips_net: str = "alex"
    colorfulness_variant: str = "absolute"
    fid_sampling: str = "first"
    resize_backend: str = "pillow"
    fid_real_stats: Path | None = None

    def __post_init__(self) -> None:
        for name in ("predictions", "ground_truth", "output"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if self.fid_real_stats is not None:
            object.__setattr__(self, "fid_real_stats", Path(self.fid_real_stats).expanduser().resolve())
            if self.fid_real_stats.suffix != ".npz":
                raise ValueError("fid_real_stats must be an .npz file")
        if isinstance(self.metrics, str):
            raise ValueError("metrics must be a list of metric names")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not self.metrics or set(self.metrics) - set(METRICS):
            raise ValueError(f"metrics must be selected from {METRICS}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicates")
        if self.layout not in ("single", "sample-dirs", "per-image"):
            raise ValueError("layout must be single, sample-dirs, or per-image")
        if self.resize is not None:
            if (len(self.resize) != 2 or
                    any(type(v) is not int or v <= 0 for v in self.resize)):
                raise ValueError("resize must be [height, width] with positive integers")
            object.__setattr__(self, "resize", tuple(self.resize))
        if self.resize_backend not in ("pillow", "opencv"):
            raise ValueError("resize_backend must be pillow or opencv")
        if self.sample_ids is not None:
            if (isinstance(self.sample_ids, str) or not self.sample_ids or
                    any(not isinstance(v, str) or not v for v in self.sample_ids) or
                    len(set(self.sample_ids)) != len(self.sample_ids)):
                raise ValueError("sample_ids must be a nonempty list of unique strings")
            object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(self.workers) is not int or self.workers <= 0:
            raise ValueError("workers must be a positive integer")
        for name in ("allow_subset", "allow_variable_k"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.lpips_net not in ("alex", "vgg", "squeeze"):
            raise ValueError("lpips_net must be alex, vgg, or squeeze")
        if self.colorfulness_variant not in ("absolute", "signed"):
            raise ValueError("colorfulness_variant must be absolute or signed")
        if self.fid_sampling not in ("first", "all"):
            raise ValueError("fid_sampling must be first or all")
        # Writing results into either input tree could contaminate the next run.
        if any(self.output.is_relative_to(root)
               for root in (self.predictions, self.ground_truth)):
            raise ValueError("output must be outside the input image directories")

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}
