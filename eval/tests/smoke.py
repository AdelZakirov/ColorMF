"""Create mock disk images and a full report without loading pretrained weights.

Run: python -m eval.tests.smoke --output eval/results/smoke
Neural stub scores are for wiring validation only, never paper comparisons.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from eval.config import EvalConfig
from eval.metrics.fid import FIDMetric
from eval.metrics.lpips import LPIPSMetric
from eval.pipeline import evaluate
from eval.reporting import save_report, summary
from eval.tests.mocks import MockDistance, MockFeatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/results/smoke"))
    root = parser.parse_args().output.resolve()
    for index in range(4):
        image = np.random.default_rng(index).integers(20, 200, (64, 64, 3), dtype=np.uint8)
        for folder, values in (("gt", image), ("pred/seed_1", 255 - image), ("pred/seed_2", image)):
            path = root / folder / f"image_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(values).save(path)
    config = EvalConfig(root / "pred", root / "gt", output=root / "report", layout="sample-dirs")
    report = evaluate(config, lpips_metric=LPIPSMetric(network=MockDistance()),
                      fid_metric=FIDMetric(network=MockFeatures(), dims=3))
    save_report(report, config.output)
    print(summary(report))
    print(f"Results: {config.output / 'report.json'}")


if __name__ == "__main__":
    main()
