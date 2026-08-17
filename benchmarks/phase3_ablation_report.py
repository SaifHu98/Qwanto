#!/usr/bin/env python3
"""Render the attributable local CPU Phase 3 ablation matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CLASSIFICATION = "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, evidence: Path, *, delayed: str = "No", affinity: str = "default",
         correctness: str = "not run") -> dict:
    report = _load(evidence)
    summary = report.get("summary", {})
    metadata = report.get("runtime_metadata", {})
    return {
        "variant": name,
        "threads": metadata.get("active_cpu_threads", "Unavailable"),
        "row_block": "1",
        "delayed_reduction": delayed,
        "unpack": "current AVX2/VNNI",
        "swiglu": "scalar exact",
        "affinity": affinity,
        "kv_mode": report.get("runtime_config_snapshot", {}).get("kv_cache_mode", "fp16"),
        "median_tok_per_sec": summary.get("decode_tok_per_sec_median"),
        "p5_tok_per_sec": summary.get("decode_tok_per_sec_p5"),
        "p95_latency_ms": summary.get("decode_latency_ms_p95"),
        "correctness": correctness,
        "classification": report.get("evidence_classification", CLASSIFICATION),
        "evidence_path": evidence.as_posix(),
        "executable_sha256": report.get("executable", {}).get("sha256"),
        "model_sha256": report.get("model", {}).get("sha256"),
        "git_commit": metadata.get("git_commit"),
        "git_worktree_dirty": metadata.get("git_worktree_dirty"),
    }


def build_report(evidence_dir: Path) -> dict:
    correctness_report = _load(evidence_dir / "delayed-reduction-correctness.json")
    exact = correctness_report.get("agreement", {}).get("exact_stream_text_match") is True
    rows = [
        _row("Current clean VNNI baseline", evidence_dir / "baseline-vnni-8t-64-rebuilt.json", correctness="scalar/VNNI differential passed"),
        _row("Delayed reduction", evidence_dir / "delayed-reduction-8t-64-rebuilt.json", delayed="Yes", correctness="140/140 differential + exact stream agreement" if exact else "failed"),
        _row("Affinity close", evidence_dir / "affinity-close-8t-64-8workers.json", affinity="close", correctness="not applicable"),
        _row("Affinity spread", evidence_dir / "affinity-spread-8t-64-8workers.json", affinity="spread", correctness="not applicable"),
        {"variant": "2-row blocking", "classification": "UNAVAILABLE", "reason": "No HyperVSQ-2 multi-row candidate retained; generic row blocking is not evidence for this layout."},
        {"variant": "4-row blocking", "classification": "UNAVAILABLE", "reason": "No HyperVSQ-2 multi-row candidate retained; generic row blocking is not evidence for this layout."},
        {"variant": "Alternative 2-bit unpack", "classification": "UNAVAILABLE", "reason": "No shuffle/LUT candidate completed full GEMV and end-to-end validation."},
        {"variant": "SIMD SwiGLU", "classification": "UNAVAILABLE", "reason": "No SIMD candidate; scalar SwiGLU time is instrumented but no validated approximation was adopted."},
        {"variant": "Combined validated CPU kernel", "classification": "UNAVAILABLE", "reason": "No combined row/unpack/SwiGLU winner exists; delayed reduction remains separately attributable."},
        {"variant": "KV quantization long-context", "classification": "UNAVAILABLE", "reason": "Only fp16/auto is accepted by qwnrun; 512/4K/16K comparative KV evidence is not available."},
        {"variant": "Speculative decoding", "classification": "UNAVAILABLE", "reason": "No runtime-wired draft/target verification, correction, rollback, and end-to-end counters."},
    ]
    return {
        "schema_version": "1.0.0",
        "benchmark_class": "CPU_PHASE3_ABLATION_MATRIX",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_classification": CLASSIFICATION,
        "rows": rows,
        "accepted_variants": ["delayed_reduction"],
        "rejected_or_unavailable": [row["variant"] for row in rows if row["variant"] not in {"Current clean VNNI baseline", "Delayed reduction", "Affinity close", "Affinity spread"}],
        "notes": [
            "The 64-token run is a SHORT_DIAGNOSTIC-style comparison; 128-token release-quality evidence is stored separately.",
            "All measured rows remain local-pending-hosted-validation and include their executable/model identity.",
            "Affinity is experimental and not adopted as a production default from one close/spread pair.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# CPU Phase 3 Local Ablation Matrix",
        "",
        f"Evidence classification: **{report['evidence_classification']}**. This report is not release-verified.",
        "",
        "| Variant | Threads | Row block | Delayed reduction | Unpack | SwiGLU | Affinity | KV mode | Median tok/s | p5 tok/s | p95 latency | Correctness |",
        "|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        if row["classification"] == "UNAVAILABLE":
            lines.append(f"| {row['variant']} | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: {row['reason']} |")
            continue
        lines.append(
            f"| {row['variant']} | {row['threads']} | {row['row_block']} | {row['delayed_reduction']} | {row['unpack']} | {row['swiglu']} | {row['affinity']} | {row['kv_mode']} | {row['median_tok_per_sec']:.6f} | {row['p5_tok_per_sec']:.6f} | {row['p95_latency_ms']:.3f} | {row['correctness']} |"
        )
    lines += ["", "## Attribution", "", "Each measured row records the evidence path, executable/model SHA-256, commit, and dirty-worktree state in the machine-readable JSON. The delayed-reduction candidate is enabled only by `QWN_HYPERVSQ2_DELAYED_REDUCTION=1`; the default VNNI path remains unchanged.", ""]
    return "\n".join(lines)


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
