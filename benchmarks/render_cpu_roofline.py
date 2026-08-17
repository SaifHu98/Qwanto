#!/usr/bin/env python3
"""Render the CPU roofline markdown report from machine-readable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _value(item: dict, digits: int = 3) -> str:
    value = item.get("value")
    if value is None:
        return "Unavailable"
    return f"{float(value):.{digits}f}"


def render(evidence: dict) -> str:
    runtime = evidence["runtime"]
    summary = runtime["summary"]
    counters = evidence["counters"]
    intensity = evidence["arithmetic_intensity"]
    roofline = evidence["roofline_estimate"]
    actual = evidence["actual_throughput"]
    bandwidth = evidence["independent_bandwidth"]
    identity = evidence["source_identity"]
    model = evidence["model"]
    predicted = roofline.get("predicted_tok_per_sec")
    measured = actual.get("median_tok_per_sec")
    percentage = (measured / predicted * 100.0) if predicted and measured else None
    rows = []
    for result in bandwidth["results"]:
        rows.append(
            f"| {result['workers']} | {_value(result['read_only'], 3)} | "
            f"{_value(result['read_only'], 3)} | {_value(result['copy'], 3)} |"
        )
    bandwidth_table = "\n".join(rows)
    return f"""# CPU Roofline Analysis — 2026-08-17

This report is generated from `{identity['executable']}` and the measured evidence file. It is local evidence and remains **{evidence['evidence_classification']}** until the exact final runtime commit passes hosted validation. It is not a release-quality claim.

## Reproduction identity

| Field | Value |
|---|---|
| Git commit | `{identity['git_commit']}` |
| Worktree dirty during measurement | `{identity['git_worktree_dirty']}` |
| Executable SHA-256 | `{identity['executable_sha256']}` |
| Model SHA-256 | `{identity['model_sha256']}` |
| Host | {evidence['host']['os']} |
| CPU | {evidence['host']['cpu']} |
| Active workers | {runtime['runtime_metadata'].get('active_cpu_threads', 'Unavailable')} |
| Selected kernel | {runtime['runtime_metadata'].get('actual_executed_kernel', 'Unavailable')} |
| OpenMP compiled/runtime loaded | {runtime['runtime_metadata'].get('openmp_compiled')} / {runtime['runtime_metadata'].get('openmp_runtime_loaded')} |

The model file is **{model['file_size_bytes'] / (1024 ** 3):.3f} GiB** with **{model['mapped_tensor_bytes'] / (1024 ** 3):.3f} GiB** of mapped tensor payload across {model['tensor_count']} tensors. The dominant dtype is `{runtime['runtime_metadata'].get('model_dtype', 'Unavailable')}`.

## Independent bandwidth measurement

The benchmark used a {bandwidth['buffer_bytes'] / (1024 ** 2):.0f} MiB NumPy buffer, {bandwidth['repetitions']} repetitions, and read-only plus copy workloads. Values below are independent stream-like measurements, not qwnrun hardware-counter readings.

| Workers | Read median GB/s | Read p5 GB/s | Copy median GB/s |
|---:|---:|---:|---:|
{bandwidth_table}

At the selected 8 workers, the read-only median was **{next((r['read_only']['median_gb_per_sec'] for r in bandwidth['results'] if r['workers'] == 8), None):.3f} GB/s** and the copy median was **{next((r['copy']['median_gb_per_sec'] for r in bandwidth['results'] if r['workers'] == 8), None):.3f} GB/s**. These are independent bandwidth proxies, not measured memory-controller bandwidth.

## Arithmetic intensity and roofline estimate

| Quantity | Value | Source / limitation |
|---|---:|---|
| Logical weight bytes per token | {_value(intensity['bytes_per_token'])} | {intensity['bytes_per_token']['source']} from HyperVSQ-2 descriptors |
| Logical FLOPs per token | {_value(intensity['flops_per_token'])} | {intensity['flops_per_token']['source']} from descriptor-derived operations |
| Arithmetic intensity | {_value(intensity['flops_per_byte'], 6)} FLOP/byte | derived estimate |
| Predicted throughput | {predicted:.3f} tok/s | {roofline['source']} using selected independent bandwidth |
| Actual persistent decode median | {measured:.3f} tok/s | {actual['source']} |
| Actual / predicted estimate | {percentage:.2f}% | derived comparison, not a hardware efficiency measurement |

The logical bytes/token value is not a process read counter. It assumes the descriptor-derived logical traffic and therefore cannot prove that every byte was fetched from DRAM. The predicted throughput excludes decoder overhead, cache reuse, synchronization, sampling, and page/cache effects. It is a roofline estimate only.

## Time and counter evidence

| Counter | Result | Source |
|---|---:|---|
| HyperVSQ-2 kernel time | {_value(counters['kernel_compute_ms'])} ms aggregate | {counters['kernel_compute_ms']['source']} |
| SwiGLU time | {_value(counters['swiglu_ms'])} ms aggregate | {counters['swiglu_ms']['source']} |
| Prefill median | {_value(counters['prefill_ms'])} ms | {counters['prefill_ms']['source']} |
| Decode median | {_value(counters['decode_ms'])} ms | {counters['decode_ms']['source']} |
| Process memory reads | {counters['memory_reads']['value']} | {counters['memory_reads']['source']} |
| Memory-controller bandwidth | {counters['memory_controller_bandwidth']['value']} | {counters['memory_controller_bandwidth']['source']} |
| L1/L2/L3 misses | {counters['l1_l2_l3_misses']['value']} | {counters['l1_l2_l3_misses']['source']} |
| CPU cycles/instructions/vector instructions | unavailable | no supported hardware profiler configured |
| OpenMP synchronization time | unavailable | barrier timing is not separately instrumented |

## Interpretation

The measured local decode result is **{measured:.6f} tok/s median** with p5 **{actual.get('p5_tok_per_sec'):.6f} tok/s**. The independent bandwidth run does not justify assuming 40–60 GB/s, and the derived roofline must not be presented as measured hardware bandwidth. The current evidence supports a CPU VNNI path and a memory-sensitive workload, but it does not identify every decoder bottleneck; profiler-backed cache, cycles, instructions, and memory-controller counters remain unavailable on this host.

## Source evidence

- Machine-readable evidence: `benchmarks/evidence/windows/2026-08-17/phase3-local/roofline-8t-64.json`
- Harness: `benchmarks/benchmark_cpu_roofline.py`
- Runtime benchmark: `benchmarks/benchmark_release_quality.py`
- Model format: `docs/qwn-format.md`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    Path(args.output).write_text(render(evidence), encoding="utf-8")
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
