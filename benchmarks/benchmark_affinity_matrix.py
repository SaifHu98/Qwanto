#!/usr/bin/env python3
"""Compare OS scheduling with explicit OpenMP affinity policies.

This is an opt-in benchmark.  It never runs during application startup and
keeps each policy in its own release-quality persistent evidence record.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .benchmark_release_quality import run_release_quality
except ImportError:
    from benchmark_release_quality import run_release_quality


def _run_policy(*, policy: str, model: str, executable: str, threads: int,
                max_tokens: int, repeats: int, timeout: float) -> dict:
    saved = {key: os.environ.get(key) for key in ("OMP_PROC_BIND", "OMP_PLACES")}
    try:
        if policy == "os-default":
            os.environ.pop("OMP_PROC_BIND", None)
            os.environ.pop("OMP_PLACES", None)
            overrides = None
        else:
            overrides = {"OMP_PROC_BIND": policy, "OMP_PLACES": "cores"}
        report = run_release_quality(
            model=model, executable=executable, threads=threads,
            max_tokens=max_tokens, warmup_tokens=1, repeats=repeats,
            timeout=timeout, variant=f"affinity-{policy}",
            env_overrides=overrides, pending_hosted_validation=True,
        )
        report["affinity_policy"] = policy
        return report
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = ("os-default", "close", "spread")
    reports = {}
    for policy in policies:
        report = _run_policy(
            policy=policy, model=args.model, executable=args.executable,
            threads=args.threads, max_tokens=args.max_tokens,
            repeats=args.repeats, timeout=args.timeout,
        )
        path = output_dir / f"affinity-{policy}-{args.threads}t-{args.max_tokens}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports[policy] = {
            "classification": report.get("evidence_classification"),
            "median_tok_per_sec": report.get("summary", {}).get("decode_tok_per_sec_median"),
            "p5_tok_per_sec": report.get("summary", {}).get("decode_tok_per_sec_p5"),
            "p95_latency_ms": report.get("summary", {}).get("decode_latency_ms_p95"),
            "cv": report.get("summary", {}).get("decode_tok_per_sec_cv"),
            "path": str(path),
        }
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
