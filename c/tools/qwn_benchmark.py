#!/usr/bin/env python3
"""
Qwanto Real-World Profiling & Inference Benchmark Harness
=========================================================

.. deprecated::
    This module is superseded by ``qwn_benchmark_v2.py`` which
    implements the real end-to-end measurement protocol described in
    section 10 of ``Full Improve Plan.md`` (no mock values, structured
    JSON output, environment capture, warmup/measurement round
    separation).  The legacy CLI is preserved here as a thin
    backward-compatible shim that delegates to the new harness.

The shim still accepts the old arguments (``model``, ``--tokens``,
``--prompt``) so existing scripts do not break.  When run, it forwards
the request to ``qwn_benchmark_v2.main`` and prints a Markdown summary
that matches the old output style.
"""

import argparse
import sys
import warnings
from pathlib import Path

tools_dir = Path(__file__).resolve().parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

try:
    from qwn_benchmark_v2 import BenchmarkConfig, BenchmarkRunner, render_markdown
except Exception as exc:  # pragma: no cover - import error path
    sys.stderr.write(f"qwn_benchmark: cannot load v2 harness ({exc!r})\n")
    raise


def _legacy_main(argv=None) -> int:  # pragma: no cover - thin wrapper
    warnings.warn(
        "qwn_benchmark is deprecated; use qwn_benchmark_v2 instead",
        DeprecationWarning, stacklevel=2)
    parser = argparse.ArgumentParser(
        description="[legacy] Qwanto benchmark - delegates to v2")
    parser.add_argument("model", help="Path to .qwn model file")
    parser.add_argument("--tokens", type=int, default=64,
                        help="Tokens to generate per round")
    parser.add_argument("--prompt", type=str,
                        default="Explain quantum computing in detail.",
                        help="Input prompt")
    args, _unknown = parser.parse_known_args(argv)
    cfg = BenchmarkConfig(
        model_path=Path(args.model).resolve(),
        prompt=args.prompt,
        n_gen=args.tokens,
    )
    report = BenchmarkRunner(cfg).run()
    print(render_markdown(report))
    return 0


def run_real_benchmark(model_path: str, prompt: str = "Explain quantum computing in detail.", n_gen: int = 64):
    """Backward-compat shim.  Delegates to ``qwn_benchmark_v2``.

    The legacy function used to fabricate ``tok_per_sec`` and PPL
    numbers from a closed-form formula (``t_gen_total = 0.001 * n_gen/32``)
    — see section 1 of ``Full Improve Plan.md``.  It now returns ``True``
    only after the real harness has produced a structured report.
    """
    warnings.warn(
        "qwn_benchmark.run_real_benchmark is deprecated; use "
        "qwn_benchmark_v2.BenchmarkRunner directly.",
        DeprecationWarning, stacklevel=2)
    cfg = BenchmarkConfig(
        model_path=Path(model_path).resolve(),
        prompt=prompt,
        n_gen=n_gen,
    )
    report = BenchmarkRunner(cfg).run()
    print(render_markdown(report))
    return report.aggregate.get("status") == "ok"


if __name__ == "__main__":
    raise SystemExit(_legacy_main())
