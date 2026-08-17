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
import subprocess
import statistics
import struct
import sys
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


SCHEMA_VERSION = "1.1.0"
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


def _llc_bytes() -> tuple[int | None, str]:
    """Return a best-effort LLC size without treating an unknown size as proof."""
    if sys.platform == "win32":
        command = (
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
            "-ExpandProperty L3CacheSize)"
        )
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, check=True, timeout=5,
            )
            value = int(result.stdout.strip())
            return value * 1024, "Win32_Processor.L3CacheSize_kib"
        except (OSError, ValueError, subprocess.SubprocessError):
            return None, "unavailable"
    cache_sizes = sorted(Path("/sys/devices/system/cpu").glob("cpu0/cache/index*/size"))
    for candidate in reversed(cache_sizes):
        try:
            text = candidate.read_text(encoding="utf-8").strip().upper()
            multiplier = 1024 if text.endswith("KIB") else 1024 * 1024 if text.endswith("MIB") else 1
            number = text.rstrip("KIBM")
            return int(number) * multiplier, str(candidate)
        except (OSError, ValueError):
            continue
    return None, "unavailable"


def _rate_record(
    *, workload: str, workers: int, buffer_bytes: int, bytes_read: int,
    bytes_written: int, wall_samples: list[float], samples: list[float],
) -> dict[str, Any]:
    wall_seconds = statistics.median(wall_samples) if wall_samples else 0.0
    traffic_bytes = bytes_read + bytes_written
    aggregate_bytes_per_sec = traffic_bytes / wall_seconds if wall_seconds > 0 else None
    return {
        "workload": workload,
        "workers": workers,
        "buffer_bytes": buffer_bytes,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "traffic_bytes": traffic_bytes,
        "wall_seconds": wall_seconds,
        "wall_seconds_samples": wall_samples,
        "aggregate_bytes_per_sec": aggregate_bytes_per_sec,
        "aggregate_gb_per_sec": aggregate_bytes_per_sec / 1e9 if aggregate_bytes_per_sec else None,
        "aggregate_gib_per_sec": aggregate_bytes_per_sec / (1024 ** 3) if aggregate_bytes_per_sec else None,
        "median_gb_per_sec": statistics.median(samples) if samples else None,
        "p5_gb_per_sec": _percentile(samples, 0.05),
        "samples_gb_per_sec": samples,
        "source": "independent_numpy_stream_like_workload",
        "rate_definition": "traffic_bytes / wall_seconds",
    }


def _run_stream_workload(
    size_bytes: int, workers: int, repetitions: int, warmup_repetitions: int = 2,
) -> dict[str, Any]:
    if np is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "NumPy is not installed; independent STREAM-like workload not run",
            "source": "unavailable",
        }
    item_count = max(1, size_bytes // np.dtype(np.float32).itemsize)
    # Oversize allocation plus a 64-byte aligned view keeps the measured work
    # independent of a small allocator alignment accident.
    raw_source = np.ones(item_count + 16, dtype=np.float32)
    raw_destination = np.empty_like(raw_source)
    source_offset = (-raw_source.ctypes.data // raw_source.itemsize) % 16
    destination_offset = (-raw_destination.ctypes.data // raw_destination.itemsize) % 16
    source = raw_source[source_offset:source_offset + item_count]
    destination = raw_destination[destination_offset:destination_offset + item_count]
    chunk_size = (item_count + max(1, workers) - 1) // max(1, workers)
    chunks = [(start, min(item_count, start + chunk_size))
              for start in range(0, item_count, chunk_size)]

    def read_chunk(bounds: tuple[int, int]) -> float:
        start, end = bounds
        return float(np.sum(source[start:end], dtype=np.float64))

    def copy_chunk(bounds: tuple[int, int]) -> None:
        start, end = bounds
        np.copyto(destination[start:end], source[start:end])

    def triad_chunk(bounds: tuple[int, int]) -> None:
        start, end = bounds
        destination[start:end] = source[start:end] * np.float32(1.25) + np.float32(0.5)

    read_rates: list[float] = []
    copy_rates: list[float] = []
    triad_rates: list[float] = []
    read_walls: list[float] = []
    copy_walls: list[float] = []
    triad_walls: list[float] = []
    checksums: list[float] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in range(max(0, warmup_repetitions)):
            list(pool.map(read_chunk, chunks))
            list(pool.map(copy_chunk, chunks))
            list(pool.map(triad_chunk, chunks))
        for _ in range(max(1, repetitions)):
            started = time.perf_counter()
            checksums.append(sum(pool.map(read_chunk, chunks)))
            read_elapsed = time.perf_counter() - started
            read_walls.append(read_elapsed)
            read_rates.append(size_bytes / read_elapsed / 1e9)

            started = time.perf_counter()
            list(pool.map(copy_chunk, chunks))
            copy_elapsed = time.perf_counter() - started
            copy_walls.append(copy_elapsed)
            copy_rates.append(size_bytes * 2 / copy_elapsed / 1e9)

            started = time.perf_counter()
            list(pool.map(triad_chunk, chunks))
            triad_elapsed = time.perf_counter() - started
            triad_walls.append(triad_elapsed)
            triad_rates.append(size_bytes * 3 / triad_elapsed / 1e9)
    expected = float(item_count)
    if any(abs(value - expected) > max(1.0, expected * 1e-6) for value in checksums):
        raise RuntimeError("STREAM-like read checksum changed")
    if not np.allclose(destination[: min(1024, item_count)], np.float32(1.75), rtol=0, atol=1e-5):
        raise RuntimeError("STREAM-like triad checksum changed")
    llc_size, llc_source = _llc_bytes()
    return {
        "status": "MEASURED",
        "workers": workers,
        "buffer_bytes": size_bytes,
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "alignment_bytes": 64,
        "source_alignment_modulo": int(source.ctypes.data % 64),
        "destination_alignment_modulo": int(destination.ctypes.data % 64),
        "llc_bytes": llc_size,
        "llc_source": llc_source,
        "buffer_exceeds_llc": size_bytes > llc_size if llc_size else None,
        "read_only": _rate_record(
            workload="read_only", workers=workers, buffer_bytes=size_bytes,
            bytes_read=size_bytes, bytes_written=0,
            wall_samples=read_walls,
            samples=read_rates,
        ),
        "copy": _rate_record(
            workload="copy", workers=workers, buffer_bytes=size_bytes,
            bytes_read=size_bytes, bytes_written=size_bytes,
            wall_samples=copy_walls,
            samples=copy_rates,
        ),
        "triad": _rate_record(
            workload="triad", workers=workers, buffer_bytes=size_bytes,
            bytes_read=size_bytes * 2, bytes_written=size_bytes,
            wall_samples=triad_walls,
            samples=triad_rates,
        ),
    }


def _bandwidth_matrix(size_bytes: int, workers: list[int], repetitions: int) -> list[dict[str, Any]]:
    return [_run_stream_workload(size_bytes, count, repetitions) for count in workers]


def _unavailable_counter(reason: str) -> dict[str, Any]:
    return {"value": None, "source": "unavailable", "reason": reason}


def _counter_value(runtime_meta: dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = runtime_meta.get(key, fallback)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback


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
    forward_tokens = int(summary.get("forward_tokens_including_warmup") or 0)
    logical_bytes = runtime_meta.get("hypervsq2_logical_weight_bytes")
    logical_flops = runtime_meta.get("hypervsq2_logical_flops")
    if isinstance(logical_bytes, (int, float)) and forward_tokens:
        logical_bytes_per_token = logical_bytes / forward_tokens
        bytes_source = "qwnrun_descriptor_traffic_counter"
    else:
        logical_bytes_per_token = None
        bytes_source = "unavailable"
    if isinstance(logical_flops, (int, float)) and forward_tokens:
        flops_per_token = logical_flops / forward_tokens
        flops_source = "qwnrun_matmul_shape_counter"
    else:
        flops_per_token = None
        flops_source = "unavailable"
    bandwidth = _bandwidth_matrix(args.buffer_mib * 1024 * 1024, workers, args.stream_repetitions)
    actual_workers = runtime_meta.get("active_cpu_threads")
    selected_workers = int(actual_workers) if isinstance(actual_workers, (int, float)) else args.threads
    selected_result = next(
        (row for row in bandwidth if row.get("workers") == selected_workers and row.get("status") == "MEASURED"),
        None,
    )
    if selected_result is None and selected_workers != args.threads:
        selected_workers = args.threads
        selected_result = next(
            (row for row in bandwidth if row.get("workers") == selected_workers and row.get("status") == "MEASURED"),
            None,
        )
    selected_read = selected_result.get("read_only") if selected_result else None
    aggregate_bandwidth = selected_read.get("aggregate_bytes_per_sec") if selected_read else None
    predicted = aggregate_bandwidth / logical_bytes_per_token if aggregate_bandwidth and logical_bytes_per_token else None
    logical_categories = {
        name: {"value": _counter_value(runtime_meta, name), "source": "qwnrun_logical_execution_counter"}
        for name in (
            "logical_tensor_visits", "logical_repeated_tensor_accesses", "logical_tensors_skipped",
            "logical_embedding_bytes", "logical_attention_bytes", "logical_ffn_bytes",
            "logical_lm_head_bytes", "logical_other_weight_bytes", "logical_kv_bytes", "logical_activation_bytes",
            "logical_temporary_bytes", "logical_weight_bytes_per_token",
            "logical_kv_bytes_per_token", "logical_activation_bytes_per_token",
            "total_logical_bytes_per_token",
        )
    }
    for item in logical_categories.values():
        if item["value"] is None:
            item.update({"source": "unavailable", "reason": "runtime counter not exposed by this executable"})
    if forward_tokens:
        for name in ("logical_embedding_bytes", "logical_attention_bytes", "logical_ffn_bytes",
                     "logical_lm_head_bytes", "logical_other_weight_bytes", "logical_kv_bytes",
                     "logical_activation_bytes", "logical_temporary_bytes"):
            if logical_categories[name]["value"] is not None:
                logical_categories[name]["cumulative_value"] = logical_categories[name]["value"]
                logical_categories[name]["value"] /= forward_tokens
                logical_categories[name]["unit"] = "bytes_per_forward_token"
        weight_parts = [logical_categories[name]["value"] for name in (
            "logical_embedding_bytes", "logical_attention_bytes", "logical_ffn_bytes",
            "logical_lm_head_bytes", "logical_other_weight_bytes")
                        if logical_categories[name]["value"] is not None]
        total_parts = weight_parts + [logical_categories[name]["value"] for name in (
            "logical_kv_bytes", "logical_activation_bytes", "logical_temporary_bytes")
                                      if logical_categories[name]["value"] is not None]
        if len(weight_parts) == 5:
            logical_categories["logical_weight_bytes_per_token"] = {
                "value": sum(weight_parts), "source": "derived_from_qwnrun_logical_execution_counters",
                "unit": "bytes_per_forward_token",
            }
        if len(total_parts) == 8:
            logical_categories["total_logical_bytes_per_token"] = {
                "value": sum(total_parts), "source": "derived_from_qwn_logical_execution_counters",
                "unit": "bytes_per_forward_token",
            }
        for source_name, derived_name in (
            ("logical_kv_bytes", "logical_kv_bytes_per_token"),
            ("logical_activation_bytes", "logical_activation_bytes_per_token"),
        ):
            if logical_categories[source_name].get("value") is not None:
                logical_categories[derived_name] = {
                    "value": logical_categories[source_name]["value"],
                    "source": "derived_from_qwn_logical_execution_counters",
                    "unit": "bytes_per_forward_token",
                }
    total_logical_bytes_per_token = logical_categories["total_logical_bytes_per_token"].get("value")
    if not isinstance(total_logical_bytes_per_token, (int, float)):
        total_logical_bytes_per_token = logical_bytes_per_token
        executed_bytes_source = bytes_source
    else:
        executed_bytes_source = "derived_from_qwn_logical_execution_counters"
    intensity = (flops_per_token / total_logical_bytes_per_token
                 if flops_per_token is not None and total_logical_bytes_per_token else None)
    predicted = (aggregate_bandwidth / total_logical_bytes_per_token
                 if aggregate_bandwidth and total_logical_bytes_per_token else None)
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
            "selected_worker_count": selected_workers,
            "selected_result": selected_result,
            "source": "independent_stream_like_benchmark",
        },
        "counters": {
            "mapped_tensor_bytes": {"value": model_meta["mapped_tensor_bytes"], "source": model_meta["source"]},
            "logical_weight_bytes_per_token": {"value": logical_bytes_per_token, "source": bytes_source},
            "logical_byte_categories": logical_categories,
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
                                  "bytes_per_token": {"value": total_logical_bytes_per_token, "source": executed_bytes_source},
                                  "flops_per_byte": {"value": intensity, "source": "derived_estimate"}},
        "roofline_estimate": {
            "predicted_tok_per_sec": predicted,
            "source": "derived_estimate" if predicted is not None else "DERIVED_OR_UNAVAILABLE",
            "equation_inputs": {
                "aggregate_bandwidth_bytes_per_sec": aggregate_bandwidth,
                "executed_bytes_per_token": total_logical_bytes_per_token,
                "selected_worker_count": selected_workers,
                "bandwidth_unit": "bytes_per_second",
                "bytes_per_token_unit": "bytes_per_token",
                "equation": "predicted_tok_per_sec = aggregate_bandwidth_bytes_per_sec / executed_bytes_per_token",
            },
            "assumptions": ["descriptor-derived traffic is not hardware read traffic",
                            "selected worker-count read-only bandwidth is an upper-bound proxy",
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
