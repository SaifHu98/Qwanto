#!/usr/bin/env python3
"""Run the bounded Phase 2 CPU ablation set without inventing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .benchmark_release_quality import run_release_quality
except ImportError:
    from benchmark_release_quality import run_release_quality


UNIMPLEMENTED = {
    "thread_autotune": "Requires an explicit local autotune run; not silently folded into the final variant.",
    "lm_head_correction": "Instrumentation is present; no predictive correction algorithm is implemented.",
    "row_blocking": "No row-blocking variant is retained without end-to-end evidence.",
    "two_bit_unpack": "No alternative unpack variant is retained without full GEMV and decode evidence.",
    "simd_swiglu": "No SIMD SwiGLU variant is retained without correctness and end-to-end evidence.",
}


def _summary(report: dict, evidence_path: Path) -> dict:
    summary = report.get("summary", {})
    return {
        "variant": report.get("configuration", {}).get("variant"),
        "classification": report.get("evidence_classification"),
        "invalid_reasons": report.get("invalid_reasons", []),
        "evidence_path": str(evidence_path),
        "executable_sha256": report.get("executable", {}).get("sha256"),
        "model_sha256": report.get("model", {}).get("sha256"),
        "selected_kernel": report.get("runtime_metadata", {}).get("actual_executed_kernel", "Unavailable"),
        "active_threads": report.get("runtime_metadata", {}).get("active_cpu_threads", "Unavailable"),
        "decode_tok_per_sec_median": summary.get("decode_tok_per_sec_median"),
        "decode_tok_per_sec_p95": summary.get("decode_tok_per_sec_p95"),
        "decode_latency_ms_median": summary.get("decode_latency_ms_median"),
        "decode_latency_ms_p95": summary.get("decode_latency_ms_p95"),
        "ttft_ms_median": summary.get("ttft_ms_median"),
        "decode_tok_per_sec_cv": summary.get("decode_tok_per_sec_cv"),
        "activation_sum_mode": report.get("runtime_metadata", {}).get("activation_sum_mode", "Unavailable"),
    }


def run_ablation(*, model: str, executable: str, output: str, threads: int | None,
                 repeats: int, max_tokens: int, timeout: float) -> dict:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    measured_reports: dict[str, dict] = {}
    variants = (
        ("baseline", {"QWN_DISABLE_ACTIVATION_SUMS": "1"}),
        ("activation_sum_precompute", {"QWN_DISABLE_ACTIVATION_SUMS": "0"}),
    )
    for name, environment in variants:
        report_path = output_path.with_name(f"{output_path.stem}-{name}.json")
        report = run_release_quality(
            model=model, executable=executable, threads=threads, repeats=repeats,
            max_tokens=max_tokens, warmup_tokens=8, timeout=timeout,
            variant=name, env_overrides=environment,
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        rows.append(_summary(report, report_path))
        measured_reports[name] = report

    baseline = next((row for row in rows if row["variant"] == "baseline"), None)
    optimized = next((row for row in rows if row["variant"] == "activation_sum_precompute"), None)
    accepted = []
    rejected = []
    if (baseline and optimized and baseline["classification"] == "MEASURED" and
            optimized["classification"] == "MEASURED" and
            (optimized["decode_tok_per_sec_median"] or 0) > (baseline["decode_tok_per_sec_median"] or 0)):
        accepted.append("activation_sum_precompute")
    else:
        rejected.append("activation_sum_precompute")

    for name, reason in UNIMPLEMENTED.items():
        rows.append({"variant": name, "classification": "UNAVAILABLE", "reason": reason})
        rejected.append(name)

    final = optimized if "activation_sum_precompute" in accepted else baseline
    result = {
        "schema_version": "1.0.0",
        "benchmark_class": "ABLATION_SUMMARY",
        "identity": {
            "executable_sha256": final.get("executable_sha256") if final else "Unavailable",
            "model_sha256": final.get("model_sha256") if final else "Unavailable",
            "requested_threads": threads if threads is not None else "auto",
            "generated_tokens": max_tokens,
        },
        "variants": rows,
        "accepted_variants": accepted,
        "rejected_variants": rejected,
        "final_variant": final["variant"] if final else "Unavailable",
        "final_selection_reason": (
            "Activation-sum precompute improved measured median decode throughput over the same-config baseline."
            if "activation_sum_precompute" in accepted else
            "No optimization variant passed the measured same-config acceptance rule."
        ),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    result = run_ablation(model=args.model, executable=args.executable, output=args.output,
                          threads=args.threads, repeats=args.repeats,
                          max_tokens=args.max_tokens, timeout=args.timeout)
    print(json.dumps({key: result[key] for key in ("accepted_variants", "rejected_variants", "final_variant")}, indent=2))


if __name__ == "__main__":
    main()
