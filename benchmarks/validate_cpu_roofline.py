#!/usr/bin/env python3
"""Validate CPU roofline evidence, including its arithmetic invariants."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1.0"
RELATIVE_TOLERANCE = 1e-6


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-9)


def _require(report: dict[str, Any], path: str) -> Any:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing required evidence field: {path}")
        value = value[part]
    return value


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("benchmark_class") != "CPU_ROOFLINE":
        errors.append("benchmark_class must be CPU_ROOFLINE")
    if report.get("evidence_classification") != "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION":
        errors.append("local roofline evidence must remain hosted-validation pending")

    try:
        bandwidth = _require(report, "independent_bandwidth")
        selected = _require(bandwidth, "selected_result")
        selected_workers = _require(bandwidth, "selected_worker_count")
        if selected is None:
            errors.append("selected_result is unavailable; roofline must be DERIVED_OR_UNAVAILABLE")
        else:
            if selected.get("workers") != selected_workers:
                errors.append("selected bandwidth row does not match selected_worker_count")
            for workload_name in ("read_only", "copy", "triad"):
                workload = selected.get(workload_name)
                if not isinstance(workload, dict):
                    errors.append(f"selected result missing {workload_name}")
                    continue
                traffic = workload.get("bytes_read", 0) + workload.get("bytes_written", 0)
                if workload.get("traffic_bytes") != traffic:
                    errors.append(f"{workload_name} traffic_bytes does not equal read + write bytes")
                wall = workload.get("wall_seconds")
                rate = workload.get("aggregate_bytes_per_sec")
                if not isinstance(wall, (int, float)) or wall <= 0:
                    errors.append(f"{workload_name} wall_seconds must be positive")
                elif not _close(rate, traffic / wall):
                    errors.append(f"{workload_name} rate does not equal traffic_bytes / wall_seconds")
                wall_samples = workload.get("wall_seconds_samples")
                if isinstance(wall_samples, list) and wall_samples:
                    if not _close(wall, statistics.median(float(item) for item in wall_samples)):
                        errors.append(f"{workload_name} wall_seconds is not the median raw wall sample")
                if not _close(workload.get("aggregate_gb_per_sec"), rate / 1e9):
                    errors.append(f"{workload_name} GB/s conversion is inconsistent")
                if not _close(workload.get("aggregate_gib_per_sec"), rate / (1024 ** 3)):
                    errors.append(f"{workload_name} GiB/s conversion is inconsistent")

        equation = _require(report, "roofline_estimate.equation_inputs")
        aggregate = equation.get("aggregate_bandwidth_bytes_per_sec")
        executed = equation.get("executed_bytes_per_token")
        predicted = _require(report, "roofline_estimate.predicted_tok_per_sec")
        if aggregate is None or executed is None:
            if predicted is not None:
                errors.append("predicted roofline must be null when equation inputs are unavailable")
        elif executed <= 0 or aggregate <= 0:
            errors.append("equation inputs must be positive when present")
        elif not _close(predicted, aggregate / executed):
            errors.append("derived_tok_per_sec != aggregate_bandwidth_bytes_per_sec / executed_bytes_per_token")
        if equation.get("selected_worker_count") != selected_workers:
            errors.append("equation selected worker count does not match bandwidth selection")
        if selected is not None:
            selected_rate = selected.get("read_only", {}).get("aggregate_bytes_per_sec")
            if not _close(aggregate, selected_rate):
                errors.append("equation bandwidth does not match selected read-only row")
        categories = _require(report, "counters.logical_byte_categories")
        total = categories.get("total_logical_bytes_per_token", {}).get("value")
        if isinstance(total, (int, float)) and executed is not None and not _close(total, executed):
            errors.append("executed_bytes_per_token does not match total logical bytes per token")
    except (TypeError, ValueError, KeyError) as error:
        errors.append(str(error))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text(encoding="utf-8"))
    errors = validate_report(report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"valid: {args.evidence}")


if __name__ == "__main__":
    main()
