#!/usr/bin/env python3
"""Run an explicit, local CPU thread autotuning trial.

Autotuning is opt-in.  The tool never runs during qwnrun startup and stores
enough identity data to invalidate a result when the host, executable, model,
or context class changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .benchmark_runtime_phases import run_phase_benchmark, sha256_file
except ImportError:
    from benchmark_runtime_phases import run_phase_benchmark, sha256_file


def _physical_cpu_count() -> int:
    try:
        import psutil  # type: ignore
        value = psutil.cpu_count(logical=False)
        if value:
            return int(value)
    except (ImportError, OSError):
        pass
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores", "/value"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            values = [int(match) for match in re.findall(r"NumberOfCores=(\d+)", result.stdout)]
            if values:
                return sum(values)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return os.cpu_count() or 1


def candidate_threads() -> list[int]:
    logical = os.cpu_count() or 1
    physical = max(1, min(logical, _physical_cpu_count()))
    return sorted({count for count in (1, 2, 4, 8, 16, physical, logical) if count <= logical})


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, min(len(values) - 1, int(0.95 * len(values) + 0.999999) - 1))]


def _cache_key(identity: dict) -> str:
    payload = json.dumps({
        "cpu": identity["cpu"],
        "logical_cores": identity["logical_cores"],
        "physical_cores": identity["physical_cores"],
        "executable_sha256": identity["executable_sha256"],
        "model_sha256": identity["model_sha256"],
        "context_size_class": identity["context_size_class"],
        "backend": identity["backend"],
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_autotune(
    *, model: str, executable: str, backend: str = "cpu", context_size: int = 4096,
    prompt: str = "Measure the local QWN decode path.", max_tokens: int = 64,
    warmup_tokens: int = 1, trials: int = 3, timeout: float = 240.0,
    cache_path: str | None = None,
) -> dict:
    if trials < 2:
        raise ValueError("at least two repeated trials are required")
    model_path = Path(model).expanduser().resolve()
    executable_path = Path(executable).expanduser().resolve()
    evidence_id = f"qwn-autotune-{time.time_ns()}"
    candidates = candidate_threads()
    result: dict = {
        "schema_version": "1.1.0",
        "evidence_id": evidence_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": {
            "candidates": candidates,
            "trials_per_candidate": trials,
            "warmup_tokens": warmup_tokens,
            "selection": "highest median decode throughput; p95 latency and variance are tie-breakers",
        },
        "identity": {
            "cpu": platform.processor() or "Unavailable",
            "logical_cores": os.cpu_count() or 1,
            "physical_cores": _physical_cpu_count(),
            "executable_sha256": sha256_file(executable_path),
            "model_sha256": sha256_file(model_path),
            "model_dtype": "Unavailable",
            "context_size_class": str(context_size),
            "backend": backend,
        },
        "trials": [],
        "autotuned_threads": None,
        "actual_active_threads": None,
        "selection_reason": "Unavailable",
        "classification": "UNAVAILABLE",
    }
    result["cache_key"] = _cache_key(result["identity"])
    result["cache_path"] = str(Path(cache_path).expanduser().resolve()) if cache_path else None
    result["cache_hit"] = False
    if cache_path:
        cache_file = Path(cache_path).expanduser().resolve()
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("cache_key") == result["cache_key"]:
                cached["cache_hit"] = True
                cached["cache_path"] = str(cache_file)
                return cached
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    for requested in candidates:
        rows = []
        for trial in range(trials):
            report = run_phase_benchmark(
                str(model_path), "warm-decode", prompt, max_tokens,
                str(executable_path), backend, requested, context_size, 0,
                warmup_tokens, timeout,
            )
            measurements = report.get("measurements", {})
            rows.append({
                "trial": trial + 1,
                "classification": report.get("evidence_classification"),
                "requested_threads": requested,
                "active_threads": report.get("runtime_metadata", {}).get("active_cpu_threads"),
                "decode_tok_per_sec": measurements.get("decode_tok_per_sec"),
                "decode_wall_ms": measurements.get("decode_wall_ms"),
                "pid_reuse_proven": report.get("execution_evidence", {}).get("same_pid_sequential_requests", False),
                "kernel": report.get("runtime_metadata", {}).get("selected_cpu_isa_kernel"),
                "model_dtype": report.get("runtime_metadata", {}).get("model_dtype"),
                "report": report,
            })
            if result["identity"]["model_dtype"] == "Unavailable":
                result["identity"]["model_dtype"] = rows[-1]["model_dtype"] or "Unavailable"
        measured = [row for row in rows if row["classification"] == "MEASURED" and isinstance(row["decode_tok_per_sec"], (int, float))]
        throughputs = [float(row["decode_tok_per_sec"]) for row in measured]
        latencies = [float(row["decode_wall_ms"]) for row in measured if isinstance(row["decode_wall_ms"], (int, float))]
        result["trials"].append({
            "requested_threads": requested,
            "runs": rows,
            "median_decode_tok_per_sec": _median(throughputs),
            "p95_decode_wall_ms": _p95(latencies),
            "variance": statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0,
            "all_measured": len(measured) == trials,
        })
    valid = [trial for trial in result["trials"] if trial["all_measured"] and trial["median_decode_tok_per_sec"] is not None]
    if valid:
        winner = max(valid, key=lambda trial: (
            float(trial["median_decode_tok_per_sec"]),
            -(float(trial["p95_decode_wall_ms"]) if trial["p95_decode_wall_ms"] is not None else float("inf")),
            -float(trial["variance"]),
        ))
        result["autotuned_threads"] = winner["requested_threads"]
        active = [run["active_threads"] for run in winner["runs"] if isinstance(run["active_threads"], int)]
        result["actual_active_threads"] = statistics.median(active) if active else None
        result["selection_reason"] = (
            f"selected {winner['requested_threads']} workers by median decode throughput "
            f"({winner['median_decode_tok_per_sec']:.6f} tok/s); p95 latency and variance were tie-breakers"
        )
        result["classification"] = "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION"
    if cache_path and result["classification"] != "UNAVAILABLE":
        cache_file = Path(cache_path).expanduser().resolve()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--prompt", default="Measure the local QWN decode path.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_autotune(
        model=args.model, executable=args.executable, backend=args.backend,
        context_size=args.context_size, prompt=args.prompt, max_tokens=args.max_tokens,
        warmup_tokens=args.warmup_tokens, trials=args.trials, timeout=args.timeout,
        cache_path=args.cache_path,
    )
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("evidence_id", "autotuned_threads", "actual_active_threads", "selection_reason", "classification")}, indent=2))
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()
