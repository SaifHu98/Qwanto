#!/usr/bin/env python3
"""Render the attributable local CPU Phase A ablation matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CLASSIFICATION = "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    name: str,
    evidence: Path,
    *,
    delayed: str,
    row_block: str = "1",
    unpack: str = "shift/mask",
    swiglu: str = "exact scalar",
    affinity: str = "OS default",
    correctness: str,
    decision: str,
) -> dict:
    report = _load(evidence)
    summary = report.get("summary", {})
    metadata = report.get("runtime_metadata", {})
    return {
        "variant": name,
        "threads": metadata.get("active_cpu_threads", "Unavailable"),
        "row_block": row_block,
        "delayed_reduction": delayed,
        "unpack": unpack,
        "swiglu": swiglu,
        "affinity": affinity,
        "kv_mode": report.get("runtime_config_snapshot", {}).get("kv_cache_mode", "fp16"),
        "median_tok_per_sec": summary.get("decode_tok_per_sec_median"),
        "p5_tok_per_sec": summary.get("decode_tok_per_sec_p5"),
        "p95_latency_ms": summary.get("decode_latency_ms_p95"),
        "correctness": correctness,
        "decision": decision,
        "classification": report.get("evidence_classification", CLASSIFICATION),
        "evidence_path": evidence.as_posix(),
        "executable_sha256": report.get("executable", {}).get("sha256"),
        "model_sha256": report.get("model", {}).get("sha256"),
        "git_commit": metadata.get("git_commit"),
        "git_worktree_dirty": metadata.get("git_worktree_dirty"),
        "executed_kernel": metadata.get("actual_executed_kernel"),
        "delayed_reduction_invocation_count": metadata.get("hypervsq2_delayed_reduction_invocation_count", 0),
        "row_block_invocation_count": metadata.get("hypervsq2_row_block_invocation_count", 0),
    }


def build_report(evidence_dir: Path) -> dict:
    correctness_report = _load(evidence_dir / "delayed-reduction-correctness.json")
    exact = correctness_report.get("agreement", {}).get("exact_stream_text_match") is True
    baseline = evidence_dir / "final-baseline-8t-64.json"
    delayed = evidence_dir / "final-delayed-8t-64.json"
    rows = [
        _row("Clean VNNI baseline", baseline, delayed="Disabled",
             correctness="scalar/VNNI differential passed", decision="CONTROL"),
        _row("Delayed reduction", delayed, delayed="Enabled",
             correctness="140/140 differential + exact stream agreement" if exact else "failed",
             decision="VALIDATED_AND_ENABLED"),
        _row("2-row HyperVSQ-2 blocking", evidence_dir / "final-row2-8t-64.json",
             delayed="Enabled", row_block="2", correctness="140/140 differential",
             decision="REJECTED_PERFORMANCE"),
        _row("4-row HyperVSQ-2 blocking", evidence_dir / "final-row4-8t-64.json",
             delayed="Enabled", row_block="4", correctness="140/140 differential",
             decision="REJECTED_PERFORMANCE"),
        _row("Current 2-bit unpack", delayed, delayed="Enabled", unpack="shift/mask",
             correctness="unpack equality + 140/140 differential",
             decision="VALIDATED_CURRENT_IMPLEMENTATION"),
        _row("SIMD SwiGLU candidate", delayed, delayed="Enabled", swiglu="exact scalar",
             correctness="exact scalar reference; SIMD not adopted",
             decision="VALIDATED_NOT_BENEFICIAL"),
        _row("OS-default affinity", evidence_dir / "affinity-final" / "affinity-os-default-8t-64.json",
             delayed="Enabled", affinity="OS default", correctness="persistent PID and runtime counters",
             decision="VALIDATED_NOT_BENEFICIAL"),
        _row("Combined production CPU path", delayed, delayed="Enabled", affinity="OS default",
             correctness="140/140 differential + exact stream agreement",
             decision="VALIDATED_AND_SELECTED"),
    ]
    return {
        "schema_version": "1.1.0",
        "benchmark_class": "CPU_PHASE_A_ABLATION_MATRIX",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_classification": CLASSIFICATION,
        "rows": rows,
        "accepted_variants": ["delayed_reduction", "current_2bit_unpack", "os_default_affinity", "combined_production_cpu_path"],
        "rejected_variants": ["2-row HyperVSQ-2 blocking", "4-row HyperVSQ-2 blocking", "SIMD SwiGLU candidate"],
        "separate_unavailable_features": {
            "kv_quantization": "Only typed fp16/auto is currently accepted; no validated long-context quantized mode.",
            "speculative_decoding": "Runtime-wired draft/target verification and rollback evidence are not available.",
            "cuda": "Intentionally out of scope; GPU matmul count remains zero.",
        },
        "notes": [
            "All rows use the same final executable/model identity and fixed 64-token configuration; 128-token production evidence is stored separately.",
            "The OS-default affinity row is the control winner in both 64-token and 128-token release-quality matrices; close/spread remain opt-in and are not enabled.",
            "SwiGLU remains exact scalar because its measured contribution is immaterial relative to HyperVSQ-2 kernel time; no approximation was promoted.",
            "All local measurements remain pending the complete hosted workflow.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# CPU Phase A Local Ablation Matrix",
        "",
        f"Evidence classification: **{report['evidence_classification']}**. This report is not release-verified.",
        "",
        "| Variant | Threads | Row block | Delayed reduction | Unpack | SwiGLU | Affinity | KV mode | Median tok/s | p5 tok/s | p95 latency | Correctness | Decision |",
        "|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['variant']} | {row['threads']} | {row['row_block']} | {row['delayed_reduction']} | {row['unpack']} | {row['swiglu']} | {row['affinity']} | {row['kv_mode']} | {row['median_tok_per_sec']:.6f} | {row['p5_tok_per_sec']:.6f} | {row['p95_latency_ms']:.3f} | {row['correctness']} | {row['decision']} |"
        )
    lines += [
        "",
        "## Attribution",
        "",
        "Each row records the evidence path, executable/model SHA-256, source commit, dirty-worktree state, actual kernel, and execution counters in the machine-readable JSON.",
        "The production path enables delayed reduction by default and retains `QWN_HYPERVSQ2_DISABLE_DELAYED_REDUCTION=1` only as a developer ablation override.",
        "",
        "## Separate feature boundaries",
        "",
    ]
    for feature, status in report["separate_unavailable_features"].items():
        lines.append(f"- **{feature}:** {status}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = build_report(Path(args.evidence_dir))
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"accepted_variants": report["accepted_variants"], "classification": report["evidence_classification"]}, indent=2))


if __name__ == "__main__":
    main()
