"""Run with python -m eval from the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import METRICS, EvalConfig
from .pipeline import evaluate
from .reporting import save_report, summary


def parse_config(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config; relative paths are relative to this file")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", nargs="+", choices=METRICS)
    parser.add_argument("--layout", choices=("single", "sample-dirs", "per-image"))
    parser.add_argument("--resize", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--resize-backend", choices=("pillow", "opencv"),
                        help="Bicubic resize implementation; use opencv for ColorMF native outputs")
    parser.add_argument("--sample-ids", nargs="+", help="Select/order samples, e.g. seed_1 seed_2 seed_3")
    parser.add_argument("--allow-subset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-variable-k", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", help="Device for metric networks only; default cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--colorfulness-variant", choices=("absolute", "signed"))
    parser.add_argument("--fid-sampling", choices=("first", "all"))
    args = vars(parser.parse_args(argv))
    filename = args.pop("config")
    values = {}
    if filename is not None:
        with filename.open(encoding="utf-8") as handle:
            try:
                values = yaml.safe_load(handle)
            except yaml.YAMLError as error:
                raise ValueError(f"invalid YAML config: {error}") from error
        if not isinstance(values, dict):
            raise ValueError("evaluation config must be a YAML mapping")
        unknown = set(values) - set(EvalConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for key in ("predictions", "ground_truth", "output"):
            if key in values:
                values[key] = filename.resolve().parent / Path(values[key]).expanduser()
    # Explicit CLI options override YAML; omitted flags preserve YAML values.
    values.update({key: value for key, value in args.items() if value is not None})
    if "predictions" not in values or "ground_truth" not in values:
        raise ValueError("provide predictions and ground_truth via CLI or YAML")
    try:
        return EvalConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid config value type: {error}") from error


def main(argv: list[str] | None = None) -> None:
    try:
        config = parse_config(argv)
        report = evaluate(config)
        save_report(report, config.output)
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(f"Evaluation failed: {error}") from error
    print(summary(report))
    print(f"Results: {config.output / 'report.json'} (+ CSV)")


if __name__ == "__main__":
    main()
