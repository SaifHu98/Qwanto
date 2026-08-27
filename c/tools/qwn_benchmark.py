"""Reproducible native QWN benchmark runner.

This is a measurement harness, not a performance claim generator.  It writes
one row per prompt only when qwnrun exits successfully and reports the exact
model/executable hashes and runtime configuration used for that row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from qwn_benchmark_v2 import BenchmarkConfig, BenchmarkRunner, render_markdown
except ImportError:  # pragma: no cover - package import path
    from tools.qwn_benchmark_v2 import BenchmarkConfig, BenchmarkRunner, render_markdown


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(stderr: str) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    detail = next((line for line in stderr.splitlines()
                   if line.startswith("qwnrun result detail:")), "")
    for key, raw in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)", detail):
        try:
            rows[key] = int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            rows[key] = raw
    result = next((line for line in stderr.splitlines()
                   if line.startswith("qwnrun result: status=")), "")
    for key, raw in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)", result):
        try:
            rows[key] = int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            rows[key] = raw
    return rows


def run_benchmark(qwnrun: Path, model: Path, prompts: List[Dict[str, Any]],
                  backend: str, context: int, max_tokens: int,
                  repeats: int) -> Dict[str, Any]:
    if not prompts:
        raise ValueError("benchmark prompts must be non-empty")
    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str):
            raise ValueError("each prompt needs a text string")
        for repeat in range(repeats):
            command = [str(qwnrun), str(model), prompt["text"],
                       "--backend", backend, "--ctx-size", str(context),
                       "--max-tokens", str(max_tokens), "--seed", "0"]
            started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True,
                                       check=False)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stats = _stats(completed.stderr)
            rows.append({
                "name": prompt.get("name", f"prompt-{len(rows)}"),
                "repeat": repeat,
                "status": "MEASURED" if completed.returncode == 0 and stats.get("status") == "ok" else "FAILED",
                "returncode": completed.returncode,
                "elapsed_ms_process": elapsed_ms,
                "stats": stats,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-4000:],
            })
    measured = [row for row in rows if row["status"] == "MEASURED"]
    return {
        "schema_version": 1,
        "document_type": "qwn_official_benchmark_result",
        "status": "MEASURED" if len(measured) == len(rows) else "INCOMPLETE",
        "model": {"path": str(model.resolve()), "sha256": _sha256(model)},
        "qwnrun": {"path": str(qwnrun.resolve()), "sha256": _sha256(qwnrun)},
        "configuration": {"backend": backend, "context_size": context,
                           "max_tokens": max_tokens, "seed": 0,
                           "repeats": repeats},
        "rows": rows,
        "no_projected_performance_claim": True,
    }


def _measurement_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwnrun", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.repeats < 1 or args.max_tokens < 1 or args.context < 1:
            raise ValueError("context, max-tokens, and repeats must be positive")
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
        result = run_benchmark(args.qwnrun, args.model, prompts, args.backend,
                               args.context, args.max_tokens, args.repeats)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "MEASURED" else 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwn-benchmark: {error}", file=sys.stderr)
        return 2


def run_real_benchmark(model_path: str,
                       prompt: str = "Explain quantum computing in detail.",
                       n_gen: int = 64) -> bool:
    """Backward-compatible real benchmark entry point.

    The historical function is retained for callers and delegates to the
    existing v2 harness. It returns true only for an observed successful run.
    """
    warnings.warn("run_real_benchmark is deprecated; use qwn_benchmark_v2 directly",
                  DeprecationWarning, stacklevel=2)
    report = BenchmarkRunner(BenchmarkConfig(model_path=Path(model_path).resolve(),
                                             prompt=prompt, n_gen=n_gen)).run()
    print(render_markdown(report))
    return report.aggregate.get("status") == "ok"


def main(argv: Optional[List[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and not values[0].startswith("-"):
        warnings.warn("positional benchmark CLI is deprecated; use --prompts",
                      DeprecationWarning, stacklevel=2)
        parser = argparse.ArgumentParser(description="[legacy] Qwanto benchmark")
        parser.add_argument("model")
        parser.add_argument("--tokens", type=int, default=64)
        parser.add_argument("--prompt", default="Explain quantum computing in detail.")
        args, _unknown = parser.parse_known_args(values)
        return 0 if run_real_benchmark(args.model, args.prompt, args.tokens) else 1
    return _measurement_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
