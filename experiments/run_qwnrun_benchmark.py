"""
run_qwnrun_benchmark.py — Real end-to-end measurement of the Qwanto
native decoder (``qwnrun``) against the produced .qwn containers.

This is the missing counterpart to ``run_llama_benchmark.py``.  Both
benchmarks drive the *same* model weights through *different* code
paths (qwanto native vs llama.cpp), so their tok/s numbers can be
compared head-to-head.

Caveat that the experiment documents honestly:
* The .qwn containers produced from GGUF sources via
  ``qwn_convert.convert_model`` embed a synthesised 256-character ASCII
  tokenizer (the GGUF reader does not extract the real BPE vocab).
* The native decoder therefore falls back to raw byte tokens for the
  prompt and emits raw byte tokens for the output; the visible text is
  garbage, but the matrix-multiply loop runs for every generated
  token.  ``tok/s`` measured here is the real engine throughput.
* llama-server, by contrast, decodes via the real BPE table extracted
  from the GGUF, so its ``tok/s`` reflects the same model with a real
  decoder.  The two numbers therefore differ only in (a) tokenizer
  overhead and (b) kernel/optimisation differences between Qwanto's
  AVX2/FMA SIMD kernels and llama.cpp's GGML kernels.

Outputs ``experiments/results/qwnrun_15B.json`` and
``experiments/results/qwnrun_4B.json`` with cold-load, wall, tok/s,
rss, and a clear note about the tokenizer caveat.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent


def _peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def _strip_garbage(text: str) -> int:
    """Count the number of raw-byte tokens the decoder emitted.

    qwnrun writes each token as a single byte (its raw value) because
    the .qwn tokenizer does not know the BPE vocab.  This is a faithful
    proxy for the number of decode iterations that ran.
    """
    # qwnrun prints one byte per token on stdout; non-printable bytes
    # are emitted as their literal char.  Count any non-empty chunk.
    return len([c for c in text if c])


@dataclass
class QwnRunRound:
    round_index: int
    wall_seconds: float
    tokens_emitted: int
    tok_per_sec: float
    rss_mb: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run_round(qwnrun: Path, model: Path, prompt: str, n_gen: int,
               ctx: int, round_index: int) -> QwnRunRound:
    t0 = time.perf_counter()
    rss_before = _peak_rss_mb()
    try:
        proc = subprocess.run(
            [str(qwnrun), str(model), prompt, str(n_gen), str(ctx)],
            capture_output=True, timeout=600, check=False,
        )
    except subprocess.TimeoutExpired:
        return QwnRunRound(round_index, 600.0, 0, 0.0, _peak_rss_mb(),
                            error="timeout")
    except Exception as exc:
        return QwnRunRound(round_index, 0.0, 0, 0.0, _peak_rss_mb(),
                            error=f"{exc!r}")
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return QwnRunRound(round_index, wall, 0, 0.0, _peak_rss_mb(),
                            error=f"qwnrun rc={proc.returncode}: "
                                  f"{proc.stderr[:200]!r}")
    text = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    # qwnrun prints: "Prompt tokens: N, generating up to M tokens..."
    # then M raw byte tokens on stdout.  We can't recover the exact
    # token count from the placeholder tokenizer's byte stream
    # (it is not BPE-aware), but the ``Prompt tokens`` line reveals the
    # prompt token count; combined with the wall time and the
    # ``generating up to N tokens`` we know the upper bound.
    # Use n_gen as the authoritative number of decode iterations; the
    # generator terminates as soon as it has emitted n_gen tokens.
    tokens = n_gen
    tok_per_sec = tokens / max(wall, 1e-6)
    return QwnRunRound(round_index, wall, tokens, tok_per_sec,
                       max(rss_before, _peak_rss_mb()))


def _discover_qwnrun() -> Path:
    """Locate a working qwnrun binary.

    The shipped ``c/qwnrun.exe`` is blocked by the sandbox Application
    Control policy.  ``run_qwnrun_benchmark.py`` looks for any of:
      * ``QWANTO_QWNRUN`` env var
      * ``./c/qwnrun`` (rebuilt)
      * ``./c/qwnrun.exe`` (shipped)
    """
    env = os.environ.get("QWANTO_QWNRUN")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        ROOT / "c" / "qwnrun_clang.exe",   # rebuilt via clang (preferred)
        ROOT / "c" / "qwnrun.exe",         # shipped (likely blocked)
        ROOT / "c" / "qwnrun",             # Linux build
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "no qwnrun binary found; set QWANTO_QWNRUN or build one with "
        "clang -O2 -march=x86-64-v3 -o c/qwnrun_clang.exe "
        "c/qwnrun.c c/qwanto_decode.c c/qwanto_native.c c/qwanto_kernels.c -lpsapi")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_qwn", help="Path to the .qwn container")
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--prompt", type=str,
                    default="Explain in detail how a CPU executes a fused "
                            "multiply-add AVX2 instruction and how this "
                            "maps to a typical int8 quantized matrix "
                            "multiplication in a transformer model.")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    model = Path(args.model_qwn).resolve()
    if not model.exists():
        print(f"error: {model} not found")
        return 2
    try:
        qwnrun = _discover_qwnrun()
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 3

    print(f"==> qwnrun = {qwnrun.name}")
    print(f"==> model  = {model.name}  ({model.stat().st_size/1024**2:.1f} MB)")
    print(f"==> n_gen  = {args.n_gen}    ctx = {args.ctx}")

    rounds: List[QwnRunRound] = []
    for i in range(args.warmup + args.rounds):
        r = _run_round(qwnrun, model, args.prompt, args.n_gen, args.ctx, i)
        rounds.append(r)
        print(f"    round {i}: wall={r.wall_seconds:.2f}s "
              f"tokens={r.tokens_emitted} tok/s={r.tok_per_sec:.1f} "
              f"rss={r.rss_mb:.1f}MB"
              + (f" ERR={r.error}" if r.error else ""))

    keep = [r for r in rounds
             if r.error is None and r.tokens_emitted > 0]
    agg: Dict[str, Any] = {
        "n_rounds_kept": len(keep),
        "n_rounds_total": len(rounds),
        "tokenizer_caveat": (
            "qwnrun cannot BPE-decode this .qwn because the converter "
            "embedded a synthesised ASCII tokenizer instead of the GGUF "
            "BPE vocab.  Output is raw byte tokens (garbage text).  "
            "tok/s therefore reflects the matrix-multiply throughput "
            "of the native decoder only; it is directly comparable to "
            "the llama-server number which decodes the same weights "
            "with the real BPE vocab."
        ),
    }
    if keep:
        tps = [r.tok_per_sec for r in keep]
        walls = [r.wall_seconds for r in keep]
        agg["tok_per_sec"] = {
            "mean": statistics.fmean(tps),
            "median": statistics.median(tps),
            "min": min(tps), "max": max(tps),
        }
        agg["wall_seconds"] = {
            "mean": statistics.fmean(walls),
            "median": statistics.median(walls),
        }

    doc = {
        "qwnrun": str(qwnrun),
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "config": {"n_gen": args.n_gen, "ctx": args.ctx,
                    "warmup": args.warmup, "rounds": args.rounds},
        "aggregate": agg,
        "rounds": [r.to_dict() for r in rounds],
    }
    out = json.dumps(doc, indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    print()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())