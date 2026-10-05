"""Strict JSON, tabular CSV and a concise human-readable summary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN is not a valid evaluation result")
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _flatten(row: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def _csv(path: Path, rows: list[dict]) -> None:
    flat = [_flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def save_report(report: dict, output: Path) -> None:
    output = Path(output)
    report = _json_safe(report)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(payload, encoding="utf-8")
    _csv(output / "per_image.csv", report["per_image"])
    _csv(output / "per_sample.csv", report["per_sample"])
    summary_rows = [{"scope": scope, "metric": name, "value": value}
                    for scope, metrics in report["aggregate"].items()
                    for name, value in metrics.items()]
    if report["fid"] is not None:
        summary_rows.append({"scope": "dataset", "metric": "fid", "value": report["fid"]["value"]})
    _csv(output / "summary.csv", summary_rows)


def summary(report: dict) -> str:
    coverage = report["coverage"]
    lines = [f"Evaluated {coverage['evaluated_images']} images / {coverage['predictions']} predictions "
             f"(K={coverage['k_min']}..{coverage['k_max']})"]
    if report["protocol"]["injected_neural_backends"]:
        lines.append("TECHNICAL TEST: injected neural backends; scores are not paper-comparable")
    for scope, metrics in report["aggregate"].items():
        if metrics:
            lines.append(f"{scope}: " + ", ".join(f"{name}={value:.6g}"
                                                 for name, value in metrics.items()))
    if report["fid"] is not None:
        fid = report["fid"]
        lines.append(f"fid={fid['value']:.6g} (real={fid['real_count']}, "
                     f"generated={fid['generated_count']}, sampling={fid['sampling']})")
    lines.extend(f"Warning: {message}" for message in report["warnings"])
    return "\n".join(lines)
