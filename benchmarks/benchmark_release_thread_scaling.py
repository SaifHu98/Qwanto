#!/usr/bin/env python3
"""Run release-quality persistent CPU scaling at attributable worker counts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .benchmark_release_quality import run_release_quality
except ImportError:
    from benchmark_release_quality import run_release_quality


CLASSIFICATION = "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION"


def _physical_processors() -> int | None:
    if os.name == "nt":
        command = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    check=False, timeout=10)
            value = int(result.stdout.strip())
            return value if value > 0 else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    core_ids: set[str] = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/core_id"):
        try:
            core_ids.add(path.read_text(encoding="ascii").strip())
        except OSError:
            pass
    return len(core_ids) or None


def _worker_counts(requested: list[str] | None) -> list[int]:
    logical = os.cpu_count() or 1
    physical = _physical_processors()
    if requested:
        values = [physical if item == "physical" else logical if item == "logical" else int(item)
                  for item in requested]
    else:
        values = [1, 2, 4, 6, 8, 12, 16, physical or logical, logical]
    return sorted({max(1, min(logical, int(value))) for value in values})


def run_scaling(args: argparse.Namespace) -> dict:
    model = Path(args.model).expanduser().resolve()
    executable = Path(args.executable).expanduser().resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for workers in _worker_counts(args.workers):
        report = run_release_quality(
            model=str(model), executable=str(executable), backend="cpu", threads=workers,
            prompt=args.prompt, context_size=args.context_size, max_tokens=args.max_tokens,
            seed=args.seed, warmup_tokens=1, repeats=args.repeats, timeout=args.timeout,
            variant=f"release-thread-scaling-{workers}",
            env_overrides={"OMP_PROC_BIND": args.proc_bind} if args.proc_bind else None,
            pending_hosted_validation=True,
        )
        report_path = output.with_name(f"{output.stem}-{workers}workers.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary = report.get("summary", {})
        rows.append({
            "workers_requested": workers,
            "workers_active": report.get("runtime_metadata", {}).get("active_cpu_threads", "Unavailable"),
            "selected_kernel": report.get("runtime_metadata", {}).get("actual_executed_kernel", "Unavailable"),
            "classification": report.get("evidence_classification"),
            "median_tok_per_sec": summary.get("decode_tok_per_sec_median"),
            "p5_tok_per_sec": summary.get("decode_tok_per_sec_p5"),
            "p95_latency_ms": summary.get("decode_latency_ms_p95"),
            "cv": summary.get("decode_tok_per_sec_cv"),
            "pid_reuse_proven": summary.get("pid_reuse_proven"),
            "executable_sha256": report.get("executable", {}).get("sha256"),
            "model_sha256": report.get("model", {}).get("sha256"),
            "git_commit": report.get("runtime_metadata", {}).get("git_commit"),
            "git_worktree_dirty": report.get("runtime_metadata", {}).get("git_worktree_dirty"),
            "evidence_path": str(report_path),
            "invalid_reasons": report.get("invalid_reasons", []),
        })
    result = {
        "schema_version": "1.0.0",
        "benchmark_class": "RELEASE_THREAD_SCALING",
        "benchmark_id": f"qwn-release-thread-scaling-{time.time_ns()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_classification": CLASSIFICATION,
        "workload": {
            "max_tokens": args.max_tokens, "repeats": args.repeats, "warmup_tokens": 1,
            "prompt": args.prompt, "context_size": args.context_size, "seed": args.seed,
            "affinity": args.proc_bind or "OS default",
        },
        "host": {"os": platform.platform(), "logical_processors": os.cpu_count() or 1,
                 "physical_processors": _physical_processors()},
        "rows": rows,
        "selection": {
            "production_worker_selection": "not changed by this experiment",
            "reason": "worker choice must be based on release-quality evidence and hosted validation",
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", nargs="*", help="integers plus physical/logical")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--proc-bind", choices=("close", "spread"))
    args = parser.parse_args()
    result = run_scaling(args)
    print(json.dumps({"classification": result["evidence_classification"],
                      "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
