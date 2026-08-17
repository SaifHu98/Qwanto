#!/usr/bin/env python3
"""Run a release-quality persistent CPU decode benchmark.

This benchmark deliberately keeps one already-ready qwnrun process alive,
excludes one warmup request, and requires seven measured requests under the
same PID.  It never turns a short diagnostic result into release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

try:
    from .benchmark_runtime_phases import (
        PersistentQwnrun,
        base_report,
        build_info,
        finish,
        model_manifest_metadata,
        parse_key_values,
        revision,
        sha256_file,
    )
    from .runtime_config_snapshot import make_runtime_config_snapshot, update_runtime_config_snapshot
except ImportError:
    from benchmark_runtime_phases import (
        PersistentQwnrun,
        base_report,
        build_info,
        finish,
        model_manifest_metadata,
        parse_key_values,
        revision,
        sha256_file,
    )
    from runtime_config_snapshot import make_runtime_config_snapshot, update_runtime_config_snapshot


SCHEMA_VERSION = "1.0.0"
DEFAULT_VARIANCE_LIMIT = 0.20
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _repository_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _kernel_invocation_count(backend: str, runtime_fields: dict) -> object:
    """Select the executed-kernel counter for the requested backend."""
    if backend == "cuda":
        return runtime_fields.get(
            "gpu_kernel_launch_count", runtime_fields.get("gpu_matmul_count", 0)
        )
    return runtime_fields.get("hypervsq2_matmul_count", 0)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _cv(values: list[float]) -> float | None:
    if len(values) < 2 or not values or statistics.mean(values) == 0:
        return 0.0 if values else None
    return statistics.stdev(values) / statistics.mean(values)


def _build_runtime_metadata(report: dict, text: str, fields: dict) -> None:
    report["runtime_metadata"]["build_info"] = text
    for source, target in (
        ("compiler", "compiler"),
        ("compiler_version", "compiler_version"),
        ("optimization_flags", "optimization_flags"),
        ("binary_sha256", "binary_sha256"),
        ("openmp_compiled", "openmp_compiled"),
        ("openmp_runtime_loaded", "openmp_runtime_loaded"),
        ("requested_threads", "requested_cpu_threads"),
        ("actual_executed_kernel", "actual_executed_kernel"),
        ("selected_isa_kernel", "actual_executed_kernel"),
        ("preferred_kernel_candidate", "preferred_kernel_candidate"),
        ("binary_avx2_kernel", "binary_avx2_kernel"),
        ("binary_vnni_kernel", "binary_vnni_kernel"),
        ("hypervsq2_delayed_reduction_invocation_count", "hypervsq2_delayed_reduction_invocation_count"),
        ("row_block_invocation_count", "hypervsq2_row_block_invocation_count"),
        ("logical_tensor_visits", "logical_tensor_visits"),
        ("logical_repeated_tensor_accesses", "logical_repeated_tensor_accesses"),
        ("logical_tensors_skipped", "logical_tensors_skipped"),
        ("logical_embedding_bytes", "logical_embedding_bytes"),
        ("logical_attention_bytes", "logical_attention_bytes"),
        ("logical_ffn_bytes", "logical_ffn_bytes"),
        ("logical_lm_head_bytes", "logical_lm_head_bytes"),
        ("logical_other_weight_bytes", "logical_other_weight_bytes"),
        ("logical_kv_bytes", "logical_kv_bytes"),
        ("logical_activation_bytes", "logical_activation_bytes"),
        ("logical_temporary_bytes", "logical_temporary_bytes"),
    ):
        if source in fields:
            report["runtime_metadata"][target] = fields[source]
    features = [name for name in ("avx2", "f16c", "fma", "vnni", "avx512f")
                if fields.get(f"cpu_{name}") is True]
    report["runtime_metadata"]["cpu_feature_detection"] = ",".join(features) or "Unavailable"


def _invalid_reasons(
    reports: list[dict], *, required_tokens: int, variance_limit: float,
    invocation_count: int | None, worktree_dirty: bool,
) -> list[str]:
    reasons: list[str] = []
    if len(reports) < 7:
        reasons.append("fewer than seven measured requests")
    if any(r.get("evidence_classification") != "MEASURED" for r in reports):
        reasons.append("one or more requests were not measured")
    if any((r.get("measurements", {}).get("generated_tokens") or 0) < required_tokens for r in reports):
        reasons.append(f"a request generated fewer than {required_tokens} tokens")
    pids = [r.get("execution_evidence", {}).get("process_pid") for r in reports]
    if not pids or any(pid != pids[0] for pid in pids):
        reasons.append("persistent process PID was not reused")
    snapshots = [r.get("runtime_config_snapshot", {}) for r in reports]
    if snapshots and any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        reasons.append("runtime configuration changed between requests")
    if invocation_count is not None and invocation_count <= 0:
        reasons.append("kernel invocation count is zero")
    if any(r.get("runtime_metadata", {}).get("executable_sha256") in (None, "file_not_found")
           or r.get("model_metadata", {}).get("sha256") in (None, "file_not_found") for r in reports):
        reasons.append("executable or model hash is missing")
    if worktree_dirty:
        reasons.append("worktree was dirty while evidence was generated")
    throughputs = [
        float(r["measurements"]["decode_tok_per_sec"])
        for r in reports if _number(r.get("measurements", {}).get("decode_tok_per_sec")) is not None
    ]
    if _cv(throughputs) is not None and _cv(throughputs) > variance_limit:
        reasons.append(f"throughput coefficient of variation exceeds {variance_limit:.2f}")
    return reasons


def run_release_quality(
    *, model: str, executable: str, backend: str = "cpu", threads: int | None = None,
    prompt: str = "Explain zero-copy NVMe memory tiering in Qwanto.", context_size: int = 4096,
    max_tokens: int = 64, seed: int = 0, warmup_tokens: int = 8, repeats: int = 7,
    timeout: float = 300.0, variance_limit: float = DEFAULT_VARIANCE_LIMIT,
    variant: str = "final", env_overrides: dict[str, str] | None = None,
    pending_hosted_validation: bool = False,
) -> dict:
    model_path = Path(model).expanduser().resolve()
    executable_path = Path(executable).expanduser().resolve()
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_class": "RELEASE_QUALITY",
        "benchmark_id": f"qwn-release-cpu-{__import__('time').time_ns()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"repository_relative_path": _repository_relative(model_path),
                  "sha256": sha256_file(model_path)},
        "executable": {"repository_relative_path": _repository_relative(executable_path),
                        "sha256": sha256_file(executable_path)},
        "configuration": {
            "backend_requested": backend, "threads_requested": threads if threads is not None else "auto",
            "context_size": context_size, "max_tokens": max_tokens, "seed": seed,
            "prompt": prompt, "warmup_tokens": warmup_tokens, "repeats_requested": repeats,
            "variant": variant,
            "sampler": {"temperature": 0.0, "top_p": 1.0, "greedy": True},
        },
        "evidence_classification": "UNAVAILABLE",
        "invalid_reasons": [],
        "runtime_metadata": {},
        "warmup": {},
        "requests": [],
        "summary": {},
    }
    if repeats < 7:
        report["invalid_reasons"] = ["release-quality benchmark requires at least seven measured requests"]
        report["evidence_classification"] = "INVALID"
        return report
    if not model_path.is_file() or not executable_path.is_file():
        report["invalid_reasons"] = ["model or executable is unavailable"]
        return report

    text, fields, returncode = build_info(executable_path, backend, threads)
    report["runtime_metadata"] = {
        "build_info_returncode": returncode,
        "build_info": text,
        "git_worktree_dirty": None,
        "model_dtype": "Unavailable",
        "executable_sha256": report["executable"]["sha256"],
        "model_hash": report["model"]["sha256"],
    }
    _build_runtime_metadata(report, text, fields)
    report["runtime_metadata"].update(revision())
    report["model_metadata"] = model_manifest_metadata(model_path)
    report["model_metadata"]["sha256"] = report["model"]["sha256"]
    report["runtime_config_snapshot"] = make_runtime_config_snapshot(
        backend=backend, context_size=context_size, max_tokens=max_tokens, seed=seed,
        prompt=prompt, threads=threads, warmup_tokens=warmup_tokens,
    )

    runtime = None
    try:
        runtime = PersistentQwnrun(executable_path, model_path, backend, context_size,
                                   max_tokens, threads, seed, timeout, env_overrides)
        report["warmup"] = runtime.request("release-warmup", prompt, warmup_tokens)
        for index in range(repeats):
            item = runtime.request(f"release-{index + 1}", prompt, max_tokens)
            item["process_pid"] = runtime.pid
            report["requests"].append(item)
        report["runtime_metadata"]["process_pid"] = runtime.pid
    except Exception as error:  # benchmark remains a truthful unavailable record
        report["invalid_reasons"] = [f"persistent qwnrun measurement failed: {error}"]
        return report
    finally:
        if runtime is not None:
            stdout, stderr, returncode = runtime.close()
            report["runtime_metadata"]["runtime_returncode"] = returncode
            report["runtime_metadata"]["stderr_sha256"] = hashlib.sha256(stderr.encode()).hexdigest()
            runtime_fields = parse_key_values(stderr)
            report["runtime_metadata"].update({
                "model_dtype": runtime_fields.get("model_dtype", "Unavailable"),
                "actual_executed_kernel": runtime_fields.get("hot_path_isa_kernel", runtime_fields.get("kernel", "Unavailable")),
                "kernel_invocation_count": _kernel_invocation_count(backend, runtime_fields),
                "gpu_matmul_count": runtime_fields.get("gpu_matmul_count", 0),
                "gpu_kernel_launch_count": runtime_fields.get("gpu_kernel_launch_count", 0),
                "gpu_projection_count": runtime_fields.get("gpu_projection_count", 0),
                "cpu_fallback_count": runtime_fields.get("cpu_fallback_count", 0),
                "active_cpu_threads": runtime_fields.get("active_threads", "Unavailable"),
                "activation_sum_mode": runtime_fields.get("activation_sum_mode", "Unavailable"),
                "activation_sum_precompute_calls": runtime_fields.get("activation_sum_precompute_calls", 0),
                "activation_sum_reuse_count": runtime_fields.get("activation_sum_reuse_count", 0),
                "activation_sum_recompute_count": runtime_fields.get("activation_sum_recompute_count", 0),
                "final_lm_head_calls": runtime_fields.get("final_lm_head_calls", 0),
                "final_lm_head_ms": runtime_fields.get("final_lm_head_ms", 0),
                "intermediate_lm_head_calls": runtime_fields.get("intermediate_lm_head_calls", 0),
                "intermediate_lm_head_ms": runtime_fields.get("intermediate_lm_head_ms", 0),
                "early_exit_decisions": runtime_fields.get("early_exit_decisions", 0),
                "layers_skipped": runtime_fields.get("layers_skipped", 0),
                "tokens_saved": runtime_fields.get("tokens_saved", 0),
                "hypervsq2_logical_weight_bytes": runtime_fields.get("hypervsq2_logical_weight_bytes", 0),
                "hypervsq2_logical_flops": runtime_fields.get("hypervsq2_logical_flops", 0),
                "hypervsq2_kernel_ms": runtime_fields.get("hypervsq2_kernel_ms", 0),
                "swiglu_calls": runtime_fields.get("swiglu_calls", 0),
                "swiglu_elements": runtime_fields.get("swiglu_elements", 0),
                "swiglu_ms": runtime_fields.get("swiglu_ms", 0),
                "hypervsq2_reductions_per_row": runtime_fields.get("hypervsq2_reductions_per_row", 0),
                "hypervsq2_reduction_mode": runtime_fields.get("hypervsq2_reduction_mode", "Unavailable"),
                "hypervsq2_delayed_reduction_invocation_count": runtime_fields.get("delayed_reduction_invocation_count", 0),
                "hypervsq2_row_block_invocation_count": runtime_fields.get("row_block_invocation_count", 0),
                "logical_tensor_visits": runtime_fields.get("logical_tensor_visits", 0),
                "logical_repeated_tensor_accesses": runtime_fields.get("logical_repeated_tensor_accesses", 0),
                "logical_tensors_skipped": runtime_fields.get("logical_tensors_skipped", 0),
                "logical_embedding_bytes": runtime_fields.get("logical_embedding_bytes", 0),
                "logical_attention_bytes": runtime_fields.get("logical_attention_bytes", 0),
                "logical_ffn_bytes": runtime_fields.get("logical_ffn_bytes", 0),
                "logical_lm_head_bytes": runtime_fields.get("logical_lm_head_bytes", 0),
                "logical_other_weight_bytes": runtime_fields.get("logical_other_weight_bytes", 0),
                "logical_kv_bytes": runtime_fields.get("logical_kv_bytes", 0),
                "logical_activation_bytes": runtime_fields.get("logical_activation_bytes", 0),
                "logical_temporary_bytes": runtime_fields.get("logical_temporary_bytes", 0),
            })
            update_runtime_config_snapshot(report["runtime_config_snapshot"], runtime_fields)

    for item in report["requests"]:
        update_runtime_config_snapshot(report["runtime_config_snapshot"], item)
    for request in report["requests"]:
        request["runtime_config_snapshot"] = dict(report["runtime_config_snapshot"])
    throughput = [float(r["decode_tok_per_sec"]) for r in report["requests"]]
    latency = [float(r["decode_wall_ms"]) for r in report["requests"]]
    ttft = [float(r["first_token_ms"]) for r in report["requests"]]
    tokens = [int(r["generated_tokens"]) for r in report["requests"]]
    forward_tokens = sum(
        int(item.get("prompt_tokens", 0)) + int(item.get("generated_tokens", 0))
        for item in [report["warmup"], *report["requests"]]
    )
    prefill_ms = [float(r["prefill_ms"]) for r in report["requests"]
                  if _number(r.get("prefill_ms")) is not None and float(r["prefill_ms"]) > 0]
    prefill = [float(r["prefill_tok_per_sec"]) for r in report["requests"]
               if _number(r.get("prefill_tok_per_sec")) is not None]
    report["summary"] = {
        "measured_runs": len(report["requests"]),
        "generated_tokens_per_run": tokens,
        "forward_tokens_including_warmup": forward_tokens,
        "decode_tok_per_sec_median": _median(throughput),
        "decode_tok_per_sec_p5": percentile(throughput, 0.05),
        "decode_tok_per_sec_min": min(throughput),
        "decode_tok_per_sec_max": max(throughput),
        "decode_latency_ms_median": _median(latency),
        "decode_latency_ms_p95": percentile(latency, 0.95),
        "ttft_ms_median": _median(ttft),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "prefill_ms_median": _median(prefill_ms),
        "prefill_tok_per_sec_median": _median(prefill),
        "decode_tok_per_sec_cv": _cv(throughput),
        "pid_reuse_proven": len({r["process_pid"] for r in report["requests"]}) == 1,
        "thermal_power": "Unavailable: no direct sensor measurement",
    }
    dirty_reason = "worktree was dirty while evidence was generated"
    invalid_reasons = _invalid_reasons(
        [{"evidence_classification": "MEASURED", "measurements": r,
          "execution_evidence": {"process_pid": r["process_pid"]},
          "runtime_config_snapshot": r["runtime_config_snapshot"],
          "runtime_metadata": report["runtime_metadata"],
          "model_metadata": report["model_metadata"]} for r in report["requests"]],
        required_tokens=max_tokens, variance_limit=variance_limit,
        invocation_count=_number(report["runtime_metadata"].get("kernel_invocation_count")),
        worktree_dirty=bool(report["runtime_metadata"].get("git_worktree_dirty")),
    )
    report["validation_notes"] = []
    if pending_hosted_validation and dirty_reason in invalid_reasons:
        invalid_reasons.remove(dirty_reason)
        report["validation_notes"].append(dirty_reason)
    report["invalid_reasons"] = invalid_reasons
    report["evidence_classification"] = (
        "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION" if pending_hosted_validation
        else "MEASURED"
    ) if not report["invalid_reasons"] else "INVALID"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--variance-limit", type=float, default=DEFAULT_VARIANCE_LIMIT)
    parser.add_argument("--variant", default="final")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--env", action="append", default=[], metavar="NAME=VALUE",
        help="set an explicit benchmark-only environment variable (repeatable)",
    )
    parser.add_argument(
        "--pending-hosted-validation", action="store_true",
        help="classify valid local evidence pending the required hosted CI gate",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env_overrides: dict[str, str] = {}
    for entry in args.env:
        if "=" not in entry:
            parser.error(f"--env requires NAME=VALUE: {entry}")
        name, value = entry.split("=", 1)
        if not name or "=" in name:
            parser.error(f"invalid --env name: {name!r}")
        env_overrides[name] = value
    report = run_release_quality(
        model=args.model, executable=args.executable, backend=args.backend,
        threads=args.threads, prompt=args.prompt, context_size=args.context_size,
        max_tokens=args.max_tokens, seed=args.seed, warmup_tokens=args.warmup_tokens,
        repeats=args.repeats, timeout=args.timeout, variance_limit=args.variance_limit,
        pending_hosted_validation=args.pending_hosted_validation,
        variant=args.variant,
        env_overrides=env_overrides or None,
    )
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"classification": report["evidence_classification"],
                      "summary": report["summary"],
                      "invalid_reasons": report["invalid_reasons"]}, indent=2))
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()
