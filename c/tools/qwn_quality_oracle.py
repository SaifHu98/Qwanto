"""Run a deterministic native-logit quality oracle against a checked-in fixture.

The fixture is deliberately external to the product runtime: it must contain
reference top-k IDs (and optionally values) produced by an independently
validated implementation.  This tool never invents a reference and exits
non-zero when a required comparison cannot be performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from queue import Queue
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_k(values: Iterable[float], k: int) -> List[int]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1], reverse=True)
    return [index for index, _ in indexed[:k]]


def _readline(stream, timeout: float) -> str:
    # qwnrun's protocol is line-buffered.  A worker timeout is still required
    # so a broken model cannot make the oracle hang forever.
    result: Queue[str] = Queue(maxsize=1)
    threading.Thread(target=lambda: result.put(stream.readline()), daemon=True).start()
    try:
        line = result.get(timeout=timeout)
    except Exception as error:
        raise TimeoutError("qwnrun protocol timeout") from error
    if not line:
        raise RuntimeError("qwnrun closed the protocol stream")
    return line.rstrip("\r\n")


def _start(qwnrun: Path, model: Path, backend: str, context: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["SERVE"] = "1"
    return subprocess.Popen(
        [str(qwnrun), str(model), "--serve", "--backend", backend,
         "--ctx-size", str(context), "--max-tokens", "1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, env=env,
    )


def _forward(process: subprocess.Popen, token: int, timeout: float) -> List[float]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("qwnrun pipes are unavailable")
    process.stdin.write(f"FORWARD {token}\n")
    process.stdin.flush()
    header = _readline(process.stdout, timeout)
    if not header.startswith("LOGITS "):
        raise RuntimeError(f"unexpected qwnrun response: {header}")
    vocab = int(header.split()[1])
    values = [float(_readline(process.stdout, timeout)) for _ in range(vocab)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("native logits contain NaN or infinity")
    return values


def _reset(process: subprocess.Popen, timeout: float) -> None:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("qwnrun pipes are unavailable")
    process.stdin.write("RESET\n")
    process.stdin.flush()
    response = _readline(process.stdout, timeout)
    if response != "RESET":
        raise RuntimeError(f"unexpected reset response: {response}")


def run_oracle(qwnrun: Path, model: Path, fixture: Dict[str, Any],
               backend: str, context: int, timeout: float) -> Dict[str, Any]:
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture must contain a non-empty cases list")
    expected_model_hash = fixture.get("model_sha256")
    model_hash = _sha256(model)
    if expected_model_hash and expected_model_hash != model_hash:
        raise ValueError("fixture model_sha256 does not match the supplied model")

    process = _start(qwnrun, model, backend, context)
    results: List[Dict[str, Any]] = []
    try:
        # The SERVE readiness record is binary-delimited but remains one line.
        ready = _readline(process.stdout, timeout) if process.stdout else ""
        if "READY" not in ready:
            raise RuntimeError(f"qwnrun did not become ready: {ready}")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("tokens"), list):
                raise ValueError("each oracle case needs a tokens list")
            _reset(process, timeout)
            actual: Optional[List[float]] = None
            for raw_token in case["tokens"]:
                token = int(raw_token)
                if token < 0:
                    raise ValueError("oracle token IDs must be non-negative")
                actual = _forward(process, token, timeout)
            if actual is None:
                raise ValueError("oracle case tokens cannot be empty")
            k = int(case.get("top_k", fixture.get("top_k", 10)))
            expected_ids = case.get("expected_top_ids")
            expected_next = case.get("expected_next_token")
            actual_ids = _top_k(actual, k)
            checks: Dict[str, Any] = {}
            passed = True
            if expected_ids is not None:
                if not isinstance(expected_ids, list):
                    raise ValueError("expected_top_ids must be a list")
                overlap = len(set(actual_ids) & set(int(value) for value in expected_ids[:k]))
                checks["top_k_overlap"] = overlap / max(1, min(k, len(expected_ids)))
                passed = passed and actual_ids[:len(expected_ids[:k])] == [int(value) for value in expected_ids[:k]]
            if expected_next is not None:
                actual_next = actual_ids[0]
                checks["expected_next_token"] = int(expected_next)
                checks["actual_next_token"] = actual_next
                passed = passed and actual_next == int(expected_next)
            expected_values = case.get("expected_top_values")
            if expected_values is not None:
                tolerance = float(case.get("value_tolerance", fixture.get("value_tolerance", 1e-3)))
                if len(expected_values) > len(actual_ids):
                    raise ValueError("expected_top_values exceeds native vocabulary")
                errors = [abs(actual[index] - float(value))
                          for index, value in zip(actual_ids, expected_values)]
                checks["max_top_value_abs_error"] = max(errors, default=0.0)
                passed = passed and all(error <= tolerance for error in errors)
            results.append({"name": case.get("name", f"case-{len(results)}"),
                            "tokens": [int(value) for value in case["tokens"]],
                            "actual_top_ids": actual_ids,
                            "checks": checks, "passed": passed})
    finally:
        process.kill()
        process.wait(timeout=5)
    return {
        "schema_version": 1,
        "document_type": "qwn_model_quality_oracle_result",
        "status": "MEASURED_PASS" if all(row["passed"] for row in results) else "MEASURED_FAIL",
        "model": {"path": str(model.resolve()), "sha256": model_hash},
        "qwnrun": {"path": str(qwnrun.resolve()), "sha256": _sha256(qwnrun)},
        "backend": backend,
        "context_size": context,
        "fixture_id": fixture.get("fixture_id"),
        "reference_provenance": fixture.get("reference_provenance"),
        "cases": results,
        "no_unmeasured_claim": True,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwnrun", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        result = run_oracle(args.qwnrun, args.model, fixture, args.backend,
                            args.context, args.timeout)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "MEASURED_PASS" else 1
    except (OSError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as error:
        print(f"qwn-quality-oracle: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
