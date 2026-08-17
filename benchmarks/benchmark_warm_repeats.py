#!/usr/bin/env python3
"""Repeat persistent warm-decode runs and report median/p95 evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from .benchmark_runtime_phases import run_phase_benchmark
except ImportError:
    from benchmark_runtime_phases import run_phase_benchmark


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def run_repeated_warm_decode(
    *, model: str, executable: str, backend: str, threads: int | None,
    prompt: str, context_size: int, max_tokens: int, seed: int,
    warmup_tokens: int, repeats: int, timeout: float,
) -> dict:
    if repeats < 5:
        raise ValueError("at least five persistent warm-decode runs are required")
    reports = []
    for _ in range(repeats):
        reports.append(run_phase_benchmark(
            model, "warm-decode", prompt, max_tokens, executable,
            backend, threads, context_size, seed, warmup_tokens, timeout,
        ))
    decode = [r["measurements"].get("decode_tok_per_sec") for r in reports
              if r["evidence_classification"] == "MEASURED"
              and isinstance(r["measurements"].get("decode_tok_per_sec"), (int, float))]
    prefill = [r["measurements"].get("prefill_tok_per_sec") for r in reports
               if r["evidence_classification"] == "MEASURED"
               and isinstance(r["measurements"].get("prefill_tok_per_sec"), (int, float))]
    return {
        "schema_version": "1.0.0",
        "mode": "persistent-warm-decode-repeats",
        "repeats_requested": repeats,
        "reports": reports,
        "summary": {
            "measured_runs": len(decode),
            "decode_tok_per_sec_median": statistics.median(decode) if decode else None,
            "decode_tok_per_sec_p95": percentile95(decode),
            "prefill_tok_per_sec_median": statistics.median(prefill) if prefill else None,
            "prefill_tok_per_sec_p95": percentile95(prefill),
            "all_runs_pid_reuse_proven": bool(reports) and all(
                r["execution_evidence"]["same_pid_sequential_requests"] for r in reports
            ),
            "all_runs_measured": len(decode) == repeats,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_repeated_warm_decode(
        model=args.model, executable=args.executable, backend=args.backend,
        threads=args.threads, prompt=args.prompt, context_size=args.context_size,
        max_tokens=args.max_tokens, seed=args.seed, warmup_tokens=args.warmup_tokens,
        repeats=args.repeats, timeout=args.timeout,
    )
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()
