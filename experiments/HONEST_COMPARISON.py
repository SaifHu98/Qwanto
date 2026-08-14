"""
HONEST_COMPARISON.py — End-to-end comparison of qwnrun vs llama-server
with the actual, unfiltered numbers and a clear statement of every
known limitation encountered in this workspace.

Run from the project root:

    python experiments/HONEST_COMPARISON.py
"""

from __future__ import annotations

import io
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Force UTF-8 stdout so the report renders arrows / em-dashes / box
# characters on cp1252 consoles (Windows default).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def run_qwnrun(qwnrun: Path, model: Path, n_gen: int = 4, ctx: int = 2048):
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([str(qwnrun), str(model), "Hi",
                                str(n_gen), str(ctx)],
                               capture_output=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        return {"wall_seconds": 600.0, "tokens": 0,
                 "tok_per_sec": 0.0, "error": "timeout"}
    except Exception as exc:
        return {"wall_seconds": 0.0, "tokens": 0,
                 "tok_per_sec": 0.0, "error": f"{exc!r}"}
    wall = time.perf_counter() - t0
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    return {
        "wall_seconds": wall,
        "tokens": n_gen if proc.returncode == 0 else 0,
        "tok_per_sec": (n_gen / wall) if wall > 0 and proc.returncode == 0 else 0.0,
        "returncode": proc.returncode,
        "stdout": out[:200],
        "stderr": err[:200],
        "error": (None if proc.returncode == 0
                  else f"rc={proc.returncode}: {err.strip()[:120]}"),
    }


def main():
    qwnrun = ROOT / "c" / "qwnrun_clang.exe"
    if not qwnrun.exists():
        print(f"!! qwnrun_clang.exe not found at {qwnrun}")
        return 1

    print("=" * 75)
    print("   HONEST Qwanto Engine Benchmark — qwnrun vs llama-server")
    print("=" * 75)
    print()
    print("BUILD CONTEXT")
    print("-" * 75)
    print(f"  qwnrun binary:       {qwnrun}")
    print(f"  Build toolchain:     clang 21.1.6 (MSVC target, no OpenMP, no CUDA)")
    print(f"  -march:              x86-64-v3 (AVX2 + FMA + F16C enabled)")
    print(f"  OpenMP threads:      0  (single-threaded — no libomp available)")
    print(f"  CUDA offload:        0  (no nvcc in this workspace)")
    print(f"  Sockets: 1, Cores: 32, RAM: 31 GB, GPUs: AMD 483MB + RTX 16GB (idle)")
    print()

    cases = [
        ("1.5B Q4_K_M",
         "experiments/results/1.5B_q4_0.qwn",
         "K-quant passthrough -- qwnrun has no K-quant decoder, "
         "loads bytes as F16 (garbage weights)"),
        ("4B Q4_0",
         "experiments/results/4B_q4_0.qwn",
         "Qwen3.5 hybrid (full_attention_interval=4) -- q_proj output=8192 "
         "(16 heads x 512 effective head_dim) but config stored head_dim=256 "
         "-> matmul shape mismatch at layer 3"),
        ("4B HyperVSQ-2",
         "experiments/results/4B_hyper_vsq2.qwn",
         "Same Qwen3.5 hybrid bug as above"),
    ]

    print("=" * 75)
    print("   qwnrun MEASUREMENTS (clang build, single-threaded, CPU only)")
    print("=" * 75)
    for label, path, note in cases:
        model = ROOT / path
        if not model.exists():
            print(f"\n{label}: SKIP (file not found)")
            continue
        print(f"\n>>> {label}")
        print(f"    file:  {path}  ({model.stat().st_size/1024**2:.1f} MB)")
        print(f"    note:  {note}")
        # 4 rounds for stable mean
        results = []
        for i in range(4):
            r = run_qwnrun(qwnrun, model, n_gen=4 if "1.5B" in label else 2)
            results.append(r)
            print(f"    round {i}: wall={r['wall_seconds']:.2f}s "
                  f"tokens={r['tokens']} tok/s={r['tok_per_sec']:.2f} "
                  f"{r.get('error') or ''}")
        ok = [r for r in results if r["error"] is None and r["tokens"] > 0]
        if ok:
            tps = [r["tok_per_sec"] for r in ok]
            print(f"    MEAN tok/s (kept rounds): {statistics.fmean(tps):.2f}")
        else:
            print("    MEAN tok/s: n/a (all rounds failed)")

    # Load llama-server results
    print()
    print("=" * 75)
    print("   llama-server MEASUREMENTS (real, from experiments/results/)")
    print("=" * 75)
    for key, label in [("llama_15B.json", "1.5B Q4_K_M (GGUF)"),
                       ("llama_4B.json",  "4B BF16 (GGUF)")]:
        path = ROOT / "experiments" / "results" / key
        if not path.exists():
            print(f"\n{label}: SKIP ({key} not found)")
            continue
        d = json.loads(path.read_text())
        agg = d.get("aggregate", {})
        print(f"\n>>> {label}")
        print(f"    cold load: {agg.get('cold_load_seconds'):.2f}s")
        print(f"    TTFT mean:  {agg.get('ttft_ms',{}).get('mean'):.0f} ms")
        if "decode_tok_s" in agg:
            tps = agg["decode_tok_s"]
            print(f"    decode tok/s: mean {tps['mean']:.1f} "
                  f"median {tps['median']:.1f} "
                  f"min {tps['min']:.1f} max {tps['max']:.1f}")
        print(f"    rounds kept: {agg.get('n_rounds_kept')}")

    # Honest verdict
    print()
    print("=" * 75)
    print("   HONEST VERDICT")
    print("=" * 75)
    print("""
The Qwanto native decoder (qwnrun) does not run end-to-end on either
attached model in this workspace. Three independent failures:

  1. Build: only clang is available. clang on this Windows has no
     libomp, so the native decoder runs SINGLE-THREADED (CPU 9% in
     Task Manager). The original gcc-built qwnrun.exe had OpenMP and
     would have used all 32 cores.

  2. K-quant: the .qwn for the 1.5B Q4_K_M source stores K-quant
     blocks as opaque bytes with dtype_id=DT_F16. qwnrun reads them as
     F16 weights (garbage). qwnrun needs a real K-quant block decoder
     to run K-quant sources, which does not exist in this release.

  3. Qwen3.5 hybrid: the 4B is a Qwen3.5 hybrid
     (full_attention_interval=4). Its q_proj output is 8192 (16 heads
     x 512 effective head_dim) but the GGUF metadata reports
     key_length=256. The native decoder uses a single global head_dim
     from the config, so the Q matmul shape check fails at layer 3.
     This is a real engine limitation; it does not handle variable
     head_dim per layer.

The llama-server path (which is part of Qwanto's runtime matrix and
uses the same GGUF models) DOES run end-to-end and produced the
real numbers in the README:
    1.5B Q4_K_M: 201 tok/s decode, 107 ms TTFT, 2.07 s cold load
    4B BF16:      48 tok/s decode, 358 ms TTFT, 7.78 s cold load

To make qwnrun actually usable on these models you need (in order):
    1. A real GCC or MSVC toolchain with OpenMP support -- gives you
       multi-threaded SIMD kernels (CPU 100%, expected 50-150 tok/s
       on the 4B for single-batch CPU decode).
    2. CUDA toolkit + the ``make CUDA_DLL=1 cuda-dll`` target -- gives
       you RTX offload for the Q4_0 matmul path.
    3. A K-quant decoder in qwanto_kernels.c -- for the 1.5B Q4_K_M
       model the engine currently has to be re-quantized to Q4_0
       through the converter (no K-quant native path exists yet).
    4. Per-layer head_dim / MoE routing in qwanto_decode.c -- for the
       Qwen3.5 hybrid the engine currently only handles dense
       transformers with a single global head_dim.

None of those are out-of-scope for the project; they are simply
multi-week C-engine refactors that fall under Phases 3-6 of
Full Improve Plan.md and were deliberately deferred from this
session.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())