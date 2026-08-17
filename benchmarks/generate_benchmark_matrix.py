#!/usr/bin/env python3
"""Normalize real benchmark evidence into the auditable Qwanto matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"


def _value(value, unavailable: str = "Unavailable"):
    return unavailable if value is None else value


def _row(evidence: dict) -> dict:
    runtime = evidence.get("runtime_metadata") or {}
    model = evidence.get("model_metadata") or {}
    params = evidence.get("benchmark_parameters") or {}
    measured = evidence.get("measured_evidence") or {}
    host = evidence.get("host_environment") or {}
    gpu = (host.get("gpus_detected") or [None])[0] or {}
    return {
        "benchmark_id": evidence.get("benchmark_id", "Unavailable"),
        "timestamp_utc": evidence.get("timestamp_utc", "Unavailable"),
        "qwn_version": _value(runtime.get("qwn_version")),
        "git_commit": _value(runtime.get("git_commit")),
        "git_worktree_dirty": runtime.get("git_worktree_dirty", "Unavailable"),
        "executable_path": _value(runtime.get("executable_path")),
        "executable_sha256": _value(runtime.get("executable_sha256")),
        "model_path": _value(model.get("path")),
        "model_sha256": _value(model.get("sha256")),
        "model_architecture": _value(model.get("architecture")),
        "qwn_dtype": _value(model.get("qwn_dtype")),
        "backend_requested": _value(params.get("backend_requested")),
        "backend_actual": _value(measured.get("backend_actual")),
        "selected_cpu_isa_kernel": _value(runtime.get("selected_cpu_isa_kernel")),
        "selected_kernel": _value(measured.get("selected_kernel")),
        "active_cpu_thread_count": _value(runtime.get("active_thread_count")),
        "gpu_device": _value(measured.get("gpu_device") or (gpu.get("name") if measured.get("backend_actual") == "cuda" else None)),
        "gpu_driver": _value(gpu.get("driver_version") if measured.get("backend_actual") == "cuda" else None),
        "cuda_dll_sha256": _value(measured.get("cuda_dll_sha256") or runtime.get("cuda_dll_sha256")),
        "cuda_kernel_type": _value(measured.get("selected_kernel") if measured.get("backend_actual") == "cuda" else None),
        "gpu_matmul_count": _value(measured.get("gpu_matmul_count")),
        "cpu_fallback_count": _value(measured.get("cpu_fallback_count")),
        "gpu_vram_resident_bytes": _value(measured.get("vram_measured_bytes")),
        "gpu_upload_bytes": _value(measured.get("cuda_upload_bytes")),
        "gpu_utilization": _value(measured.get("gpu_utilization")),
        "warmup": {
            "tokens": _value(params.get("warmup_tokens")),
            "returncode": _value(params.get("warmup_returncode")),
            "timed_out": _value(params.get("warmup_timed_out")),
        },
        "prompt": _value(params.get("prompt")),
        "context_size": _value(params.get("context_size")),
        "seed": _value(params.get("seed")),
        "requested_tokens": _value(params.get("max_tokens_requested")),
        "token_count": _value(measured.get("generated_tokens")),
        "prefill_throughput_tok_s": _value(measured.get("prefill_tok_per_sec")),
        "decode_throughput_tok_s": _value(measured.get("decode_tok_per_sec") or measured.get("tok_per_sec")),
        "ttft_ms": _value(measured.get("ttft_ms")),
        "wall_seconds": _value(measured.get("wall_seconds")),
        "evidence_classification": evidence.get("evidence_classification", "UNAVAILABLE"),
        "reason": evidence.get("error_reason") or "",
        "reproduce_command": (params.get("command_argv") or []),
    }


def load_evidence(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(_row(item) for item in payload)
        else:
            rows.append(_row(payload))
    return rows


def validate_rows(rows: list[dict]) -> None:
    for row in rows:
        if row["evidence_classification"] != "MEASURED":
            continue
        required = ("qwn_version", "git_commit", "executable_sha256", "model_sha256", "model_architecture", "qwn_dtype", "backend_actual", "selected_kernel", "prompt", "context_size", "seed", "token_count", "wall_seconds")
        missing = [key for key in required if row.get(key) in (None, "", "Unavailable", "file_not_found", "file_unreadable")]
        if missing:
            raise ValueError(f"MEASURED row {row['benchmark_id']} is missing evidence: {', '.join(missing)}")
        if row["backend_requested"] == "cuda" and (row["backend_actual"] != "cuda" or row["gpu_matmul_count"] in ("Unavailable", 0) or row["cpu_fallback_count"] not in (0, "0")):
            raise ValueError(f"CUDA row {row['benchmark_id']} has no GPU-only execution proof")


def build_matrix(paths: list[Path]) -> dict:
    rows = load_evidence(paths)
    validate_rows(rows)
    source_paths = []
    for path in paths:
        try:
            source_paths.append(path.relative_to(ROOT).as_posix())
        except ValueError:
            source_paths.append(str(path))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_evidence": source_paths,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "benchmark_matrix.json")
    args = parser.parse_args()
    evidence_paths = args.evidence or [ROOT / "benchmark_evidence.json"]
    matrix = build_matrix([path.resolve() for path in evidence_paths])
    args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"[OUTPUT] Matrix written to {args.output.resolve()} ({len(matrix['rows'])} rows)")


if __name__ == "__main__":
    main()
