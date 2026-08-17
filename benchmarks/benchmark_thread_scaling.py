#!/usr/bin/env python3
"""Run the same persistent warm-decode workload at explicit thread counts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from benchmark_runtime_phases import run_phase_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", default=None)
    parser.add_argument(
        "--threads",
        default=None,
        help="Comma-separated requested thread counts (default: 1,2,4,8,16 and available cores)",
    )
    parser.add_argument("--prompt", default="Measure the local QWN decode path.")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="benchmark_thread_scaling.json")
    args = parser.parse_args()
    if args.threads:
        thread_counts = [int(value.strip()) for value in args.threads.split(",") if value.strip()]
    else:
        thread_counts = [1, 2, 4, 8, 16]
        available_cores = os.cpu_count() or 1
        if available_cores not in thread_counts:
            thread_counts.append(available_cores)
    rows = []
    for threads in thread_counts:
        rows.append(run_phase_benchmark(
            args.model, "warm-decode", args.prompt, args.max_tokens,
            args.executable, args.backend, threads, args.context_size, args.seed,
            args.warmup_tokens, args.timeout,
        ))
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "model": str(Path(args.model).expanduser().resolve()),
            "prompt": args.prompt,
            "context_size": args.context_size,
            "seed": args.seed,
            "backend": args.backend,
            "warmup_tokens": args.warmup_tokens,
            "max_tokens": args.max_tokens,
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "rows": [{
            "requested_threads": row["benchmark_parameters"]["cpu_threads_requested"],
            "classification": row["evidence_classification"],
            "active_threads": row["runtime_metadata"]["active_cpu_threads"],
            "decode_tok_per_sec": row["measurements"].get("decode_tok_per_sec"),
        } for row in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
