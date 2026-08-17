#!/usr/bin/env python3
"""Collect local CPU roofline evidence without confusing estimates with counters.

The runtime portion uses the release-quality persistent qwnrun harness.  The
bandwidth portion is an independent STREAM-like NumPy workload over a buffer
larger than the model's hot tensor working set.  Every reported quantity has a
source label so a descriptor-derived traffic estimate cannot be mistaken for a
hardware memory-read measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on minimal CI images
    np = None

try:
    from .benchmark_release_quality import run_release_quality
except ImportError:
    from benchmark_release_quality import run_release_quality


SCHEMA_VERSION = "1.0.0"
EVIDENCE_CLASSIFICATION = "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION"
QWN_HEADER_SIZE = 4096
QWN_INLINE_MAX = 29
QWN_DESC_SIZE = 136
QWN_OVERFLOW_HEADER_SIZE = 32
QWN_HEADER_PREFIX = struct.Struct("<16s6IQ8Q")
QWN_DESC = struct.Struct("<64s3I4QQQQI")
QWN_OVERFLOW = struct.Struct("<IIQQQ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _qwn_metadata(path: Path) -> dict[str, Any]:
    """Read only the stable packed QWN index; qwnrun remains the validator."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(QWN_HEADER_SIZE)
        handle.seek(file_size - 8)
        tail_offset = struct.unpack("<Q", handle.read(8))[0]
    if len(header) < QWN_HEADER_PREFIX.size:
        raise ValueError("QWN header is truncated")
    fields = QWN_HEADER_PREFIX.unpack_from(header, 0)
    magic, version, flags, arch_code, n_tensors, inline_count, reserved, n_params, *arch_dims = fields
    if magic != b"QWANTO_NATIVE_V1" or version != 1:
        raise ValueError("unsupported QWN header")
    inline_count = min(inline_count, QWN_INLINE_MAX)
    descriptors: list[tuple[Any, ...]] = []
    for index in range(inline_count):
        start = QWN_HEADER_PREFIX.size + index * QWN_DESC_SIZE
        descriptors.append(QWN_DESC.unpack_from(header, start))
    overflow_count = max(0, n_tensors - inline_count)
    if overflow_count:
        with path.open("rb") as handle:
            handle.seek(tail_offset)
            overflow_header = handle.read(QWN_OVERFLOW_HEADER_SIZE)
            count, desc_size, desc_offset, index_offset, _ = QWN_OVERFLOW.unpack(overflow_header)
            if count != overflow_count or desc_size != QWN_DESC_SIZE:
                raise ValueError("QWN overflow index is inconsistent")
            handle.seek(desc_offset)
            for _ in range(count):
                descriptors.append(QWN_DESC.unpack(handle.read(QWN_DESC_SIZE)))
    dtype_names = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q8_0", 4: "BF16", 5: "BYTES",
        6: "VSQ", 7: "VSQ_ULTRA", 8: "HYPER_VSQ", 9: "HYPER_VSQ2",
    }
    tensor_bytes = sum(int(item[10]) for item in descriptors)
    dtype_counts: dict[str, int] = {}
    for item in descriptors:
        dtype = dtype_names.get(int(item[2]), f"UNKNOWN_{item[2]}")
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return {
        "file_size_bytes": file_size,
        "mapped_tensor_bytes": tensor_bytes,
        "tensor_count": len(descriptors),
        "dtype_counts": dtype_counts,
        "header_version": version,
        "architecture_code": arch_code,
        "parameter_count": n_params,
        "architecture_dimensions": arch_dims,
        "tail_index_offset": tail_offset,
        "source": "qwn_packed_header_and_tensor_descriptors",
    }


def _run_stream_workload(size_bytes: int, workers: int, repetitions: int) -> dict[str, Any]:
    if np is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "NumPy is not installed; independent STREAM-like workload not run",
            "source": "unavailable",
        }
    item_count = max(1, size_bytes // np.dtype(np.float32).itemsize)
    source = np.ones(item_count, dtype=np.float32)
    destination = np.empty_like(source)
    chunks = np.array_split(np.arange(item_count), max(1, workers))

    def read_chunk(indexes: Any) -> float:
        return float(np.sum(source[indexes], dtype=np.float64))

    def copy_chunk(indexes: Any) -> None:
        np.copyto(destination[indexes], source[indexes])

    read_rates: list[float] = []
    copy_rates: list[float] = []
    read_checksums: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            read_checksums.append(sum(pool.map(read_chunk, chunks)))
        elapsed = time.perf_counter() - started
        read_rates.append(size_bytes / elapsed / 1e9)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(copy_chunk, chunks))
        elapsed = time.perf_counter() - started
        copy_rates.append(size_bytes * 2 / elapsed / 1e9)
    expected = float(item_count)
    if any(abs(value - expected) > max(1.0, expected * 1e-6) for value in read_checksums):
        raise RuntimeError("STREAM-like read checksum changed")
    return {
        "status": "MEASURED",
        "workers": workers,
        "buffer_bytes": size_bytes,
        "repetitions": repetitions,
        "read_only": {
            "median_gb_per_sec": statistics.median(read_rates),
            "p5_gb_per_sec": _percentile(read_rates, 0.05),
            "samples_gb_per_sec": read_rates,
            "source": "independent_numpy_stream_like_read",
        },
        "copy": {
            "median_gb_per_sec": statistics.median(copy_rates),
            "p5_gb_per_sec": _percentile(copy_rates, 0.05),
            "samples_gb_per_sec": copy_rates,
            "source": "independent_numpy_stream_like_copy",
        },
    }


def _bandwidth_matrix(size_bytes: int, workers: list[int], repetitions: int) -> list[dict[str, Any]]:
    return [_run_stream_workload(size_bytes, count, repetitions) for count in workers]


def _unavailable_counter(reason: str) -> dict[str, Any]:
    return {"value": None, "source": "unavailable", "reason": reason}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    executable = Path(args.executable).expanduser().resolve()
    model_meta = _qwn_metadata(model)
    logical_workers = os.cpu_count() or 1
    candidates = args.workers or [1, 2, 4, 6, 8, 12, 16, logical_workers]
    workers = sorted({max(1, min(logical_workers, int(value))) for value in candidates})
    release = run_release_quality(
        model=str(model), executable=str(executable), threads=args.threads,
        max_tokens=args.max_tokens, warmup_tokens=1, repeats=args.repeats,
        timeout=args.timeout, variant="roofline_release_quality",
        pending_hosted_validation=True,
    )
    runtime_meta = release.get("runtime_metadata", {})
    summary = release.get("summary", {})
    logical_bytes = runtime_meta.get("hypervsq2_logical_weight_bytes")
    logical_flops = runtime_meta.get("hypervsq2_logical_flops")
    if isinstance(logical_bytes, (int, float)) and summary.get("measured_runs"):
        logical_bytes_per_token = logical_bytes / (summary["measured_runs"] * args.max_tokens)
        bytes_source = "qwnrun_descriptor_traffic_counter"
    else:
        logical_bytes_per_token = None
        bytes_source = "unavailable"
    if isinstance(logical_flops, (int, float)) and summary.get("measured_runs"):
        flops_per_token = logical_flops / (summary["measured_runs"] * args.max_tokens)
        flops_source = "qwnrun_matmul_shape_counter"
    else:
        flops_per_token = None
        flops_source = "unavailable"
    intensity = (flops_per_token / logical_bytes_per_token
                 if flops_per_token is not None and logical_bytes_per_token else None)
    bandwidth = _bandwidth_matrix(args.buffer_mib * 1024 * 1024, workers, args.stream_repetitions)
    best_bandwidth = max(
        (row["read_only"]["median_gb_per_sec"] for row in bandwidth if row.get("status") == "MEASURED"),
        default=None,
    )
    roofline = (
        best_bandwidth * intensity if best_bandwidth is not None and intensity is not None else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_class": "CPU_ROOFLINE",
        "benchmark_id": f"qwn-cpu-roofline-{time.time_ns()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "host": {"os": platform.platform(), "cpu": platform.processor() or "Unavailable",
                 "logical_processors": logical_workers},
        "source_identity": {
            "git_commit": runtime_meta.get("git_commit", "Unavailable"),
            "git_worktree_dirty": runtime_meta.get("git_worktree_dirty", "Unavailable"),
            "executable": str(executable), "executable_sha256": _sha256(executable),
            "model": str(model), "model_sha256": _sha256(model),
        },
        "model": model_meta,
        "runtime": release,
        "independent_bandwidth": {
            "workload": "read-only sum and copy",
            "buffer_bytes": args.buffer_mib * 1024 * 1024,
            "worker_counts": workers,
            "repetitions": args.stream_repetitions,
            "results": bandwidth,
            "best_read_only_median_gb_per_sec": best_bandwidth,
            "source": "independent_stream_like_benchmark",
        },
        "counters": {
            "mapped_tensor_bytes": {"value": model_meta["mapped_tensor_bytes"], "source": model_meta["source"]},
            "logical_weight_bytes_per_token": {"value": logical_bytes_per_token, "source": bytes_source},
            "memory_reads": _unavailable_counter("qwnrun does not expose a trustworthy process read counter on this host"),
            "memory_controller_bandwidth": _unavailable_counter("no supported hardware profiler configured"),
            "l1_l2_l3_misses": _unavailable_counter("no supported hardware profiler configured"),
            "page_faults": _unavailable_counter("process counter sampling is not wired into the persistent harness"),
            "cpu_cycles": _unavailable_counter("no supported hardware profiler configured"),
            "instructions": _unavailable_counter("no supported hardware profiler configured"),
            "vector_instructions": _unavailable_counter("no supported hardware profiler configured"),
            "openmp_synchronization_ms": _unavailable_counter("OpenMP barrier timing is not separately instrumented"),
            "kernel_compute_ms": {"value": runtime_meta.get("hypervsq2_kernel_ms"), "source": "qwnrun_hypervsq2_wall_timer"},
            "non_kernel_decoder_ms": _unavailable_counter("decoder phase timer is not yet split from protocol and sampling"),
            "prefill_ms": {"value": summary.get("prefill_ms_median"), "source": "qwnrun_generation_metrics_prefill_boundary"},
            "decode_ms": {"value": summary.get("decode_latency_ms_median"), "source": "qwnrun_generation_metrics_decode_boundary"},
            "swiglu_ms": {"value": runtime_meta.get("swiglu_ms"), "source": "qwnrun_swiglu_wall_timer"},
        },
        "arithmetic_intensity": {"flops_per_token": {"value": flops_per_token, "source": flops_source},
                                  "bytes_per_token": {"value": logical_bytes_per_token, "source": bytes_source},
                                  "flops_per_byte": {"value": intensity, "source": "derived_estimate"}},
        "roofline_estimate": {
            "predicted_tok_per_sec": roofline,
            "source": "derived_estimate" if roofline is not None else "unavailable",
            "assumptions": ["descriptor-derived traffic is not hardware read traffic",
                            "independent read-only bandwidth is an upper-bound proxy",
                            "decoder overhead and cache reuse are excluded from the estimate"],
        },
        "actual_throughput": {
            "median_tok_per_sec": summary.get("decode_tok_per_sec_median"),
            "p5_tok_per_sec": summary.get("decode_tok_per_sec_p5"),
            "source": "qwnrun_release_quality_persistent_decode",
        },
        "unavailable_metrics": {
            "cuda": "UNAVAILABLE: CUDA implementation intentionally out of scope",
            "gpu_matmul_count": runtime_meta.get("gpu_matmul_count", 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--buffer-mib", type=int, default=256)
    parser.add_argument("--stream-repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="*")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"classification": report["evidence_classification"],
                      "actual": report["actual_throughput"],
                      "roofline": report["roofline_estimate"],
                      "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
