#!/usr/bin/env python3
"""Compare deterministic streamed output from two persistent qwnrun binaries."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .benchmark_runtime_phases import PersistentQwnrun, revision, sha256_file
except ImportError:
    from benchmark_runtime_phases import PersistentQwnrun, revision, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PENDING_CLASSIFICATION = "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION"


def _run(
    executable: Path,
    model: Path,
    prompt: str,
    context_size: int,
    max_tokens: int,
    threads: int,
    seed: int,
    env_overrides: dict[str, str] | None,
) -> dict:
    runtime = None
    try:
        runtime = PersistentQwnrun(
            executable, model, "cpu", context_size, max_tokens, threads, seed, 300.0,
            env_overrides,
        )
        runtime.request("agreement-warmup", prompt, 8)
        result = runtime.request(
            "agreement-measured", prompt, max_tokens, capture_data=True,
        )
        return {
            "executable": str(executable),
            "executable_sha256": sha256_file(executable),
            "pid": runtime.pid,
            "ready_ms": runtime.ready_ms,
            "result": result,
        }
    finally:
        if runtime is not None:
            _, stderr, returncode = runtime.close()
            result = locals().get("result")
            if isinstance(result, dict):
                result["runtime_returncode"] = returncode
                result["stderr_tail"] = stderr[-4000:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument(
        "--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.",
    )
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    baseline = Path(args.baseline).expanduser().resolve()
    candidate = Path(args.candidate).expanduser().resolve()
    common = {
        "model_sha256": sha256_file(model),
        "model_path": str(model),
        "prompt": args.prompt,
        "context_size": args.context_size,
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "seed": args.seed,
    }
    error = None
    baseline_run = None
    candidate_run = None
    try:
        baseline_run = _run(
            baseline, model, args.prompt, args.context_size, args.max_tokens,
            args.threads, args.seed, None,
        )
        candidate_run = _run(
            candidate, model, args.prompt, args.context_size, args.max_tokens,
            args.threads, args.seed, {"QWN_HYPERVSQ2_DELAYED_REDUCTION": "1"},
        )
    except Exception as exc:  # evidence remains explicit and machine-readable
        error = f"{type(exc).__name__}: {exc}"

    baseline_text = ((baseline_run or {}).get("result") or {}).get("stream_text")
    candidate_text = ((candidate_run or {}).get("result") or {}).get("stream_text")
    report = {
        "schema_version": "1.0.0",
        "benchmark_id": f"qwn-output-agreement-{__import__('time').time_ns()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_class": "CPU_PHASE3_CORRECTNESS",
        "evidence_classification": PENDING_CLASSIFICATION,
        "runtime_config_snapshot": {
            **common,
            "backend_requested": "cpu",
            "backend_actual_required": "cpu",
            "sampler": {"temperature": 0.0, "top_p": 1.0, "thinking": "none"},
        },
        "repository": revision(),
        "baseline": baseline_run,
        "candidate": candidate_run,
        "agreement": {
            "stream_text_available": baseline_text is not None and candidate_text is not None,
            "exact_stream_text_match": (
                baseline_text == candidate_text
                if baseline_text is not None and candidate_text is not None else None
            ),
            "baseline_stream_sha256": __import__("hashlib").sha256(
                (baseline_text or "").encode("utf-8")
            ).hexdigest() if baseline_text is not None else None,
            "candidate_stream_sha256": __import__("hashlib").sha256(
                (candidate_text or "").encode("utf-8")
            ).hexdigest() if candidate_text is not None else None,
        },
        "error": error,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": report["evidence_classification"],
        "exact_stream_text_match": report["agreement"]["exact_stream_text_match"],
        "baseline_pid": (baseline_run or {}).get("pid"),
        "candidate_pid": (candidate_run or {}).get("pid"),
        "error": error,
        "output": str(output.resolve()),
    }, indent=2))
    return 0 if report["agreement"]["exact_stream_text_match"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
