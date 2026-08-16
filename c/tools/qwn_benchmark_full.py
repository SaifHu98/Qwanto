#!/usr/bin/env python3
"""Report that optimization-suite projections are not release evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizations", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": "3.0.0",
        "benchmark_id": f"full-suite-{time.time_ns()}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_classification": "PROJECTED",
        "error_reason": "Optimization-suite targets require independently measured native runs and are not release evidence.",
        "benchmark_parameters": {
            "optimizations": args.optimizations,
            "tasks": args.tasks,
            "iterations": args.iterations,
        },
        "measured_evidence": None,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
