# Qwanto parallel execution (CPU+GPU collaboration) — honest roadmap

This document describes the **actual** state of multi-backend dispatch in
`qwnrun` today, what the 4B HyperVSQ-2 RTX 5070 Ti measurements from
`benchmarks/evidence/windows/2026-08-22/*.json` show, and what a real
"CPU+GPU in the same forward pass" implementation would require. It is a
forward plan only — no code shipped, no ABI changes, no hot-path edits.

## What is already there (honest, audited)

The product already has a multi-backend dispatch design. It is **not**
single-backend. It is split between `c/glm.c` (legacy cuda graph that
still calls the old `COLI_CUDA` ABI) and `c/cuda/qwn_cuda_abi.h` (the
current versioned ABI shipped as `qwn_cuda.dll` ABI 1). The full set of
runtime knobs the user can flip is:

| Variable | Effect | Reference |
|----------|--------|-----------|
| `COLI_CUDA=1` / `--backend cuda` | Loads `qwn_cuda.dll`, enables GPU dispatch | `c/glm.c:6113` |
| `COLI_GPU` / `COLI_GPUS` | Select GPU device(s) | `c/glm.c:6124` |
| `COLI_CUDA_ATTN=1` | CUDA path for small attention ops (S≤4) | `c/glm.c:2591`, `c/glm.c:2637` |
| `COLI_CUDA_DENSE` | Force CUDA for all dense matmuls | `c/glm.c:6125` |
| `COLI_CUDA_EXPERT_GB=N` | Route MoE expert weights ≥N GB to GPU | `c/glm.c:6126` |
| `QWANTO_BACKEND` (gateway) | Selects backend for the runtime — `auto`/`cpu`/`cuda` | `c/openai_server.py:1827` |

Inside the dispatch, fallback counts already exist:

- `gpu_matmul_count` — number of matmuls offloaded to GPU
- `cpu_fallback_count` — number of CUDA ops that returned -1 and CPU ran them
- `gpu_resident_bytes` (planned but reported 0 in current run; see §4)
- `cuda_dll_sha256` — proves which DLL actually executed (only present if loaded)

So the "auto backend" path on the current `qwnrun.exe` for the 4B
HyperVSQ-2 model **did** route through CUDA when `--backend cuda` was
explicit, and the run produced real `gpu_matmul_count=11264` at 128 tokens
plus `cpu_fallback_count=0`. That is documented in
`benchmarks/evidence/windows/2026-08-22/4b_hyper_vsq2_cuda_128.json`.
The same model under `--backend cpu` produced `8.78 tok/s`; under
`--backend cuda` produced `8.22 tok/s`. **On this RTX 5070 Ti Laptop,
the current CUDA reference path is 5.9% slower than the CPU VNNI path
at 128 tokens.** This is the honest current measurement; no optimisation
is implied.

The Q4_0 (1.5B) model cannot use CUDA today — `qwn_cuda.dll` ABI 1 only
covers `QWN_DT_HYPER_VSQ2` matmuls. The CUDA attempt for 1.5B exits
`rc=-1` with `layer 0 attn matmul failed`. See
`benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_cuda_attempt.json`
and `1.5b_cuda_probe.log`.

## What "true CPU+GPU in the same forward" would require

The current dispatch is "one-backend-at-a-time for the entire session".
What would actually be needed for partial CPU/partial GPU within one
forward pass:

1. **Per-op routing table.** Every matmul in `qwn_decoder_generate`
   classified by `(dtype, N_rows, N_cols, M, K, N)` into a CPU or GPU
   bucket, with safe fallback if the chosen backend cannot handle the
   shape. Today there is no such table; routing is whole-session.

2. **CUDA streams + OpenMP barriers.** A forward pass would allocate
   per-layer streams, copy `x`, `weights` to GPU, launch kernel, copy
   `y` back. OpenMP barriers across streams need investigation because
   our current `dot_hyper_vsq2_*` paths use `#pragma omp parallel for`
   internally — mixing OpenMP barriers with CUDA-stream sync produces
   nondeterministic scheduling and timers must be careful.

3. **Memory budget across backends.** RTX 5070 Ti has 12 GB VRAM; CUDA
   reference path keeps `gpu_resident_bytes=0` because it doesn't pin
   anything. Pinning weights would push multi-GB residency and risk
   OOM. The current `qwn_cuda.dll` deliberately avoids this; a hybrid
   path that pins weights must handle eviction.

4. **Reference oracle for differential testing.** Every hybrid
   combination of (op-class × backend × quantization) needs ≥100
   differential tests against a known-correct path. We currently have
   140/140 for HyperVSQ-2 scalar/AVX2/VNNI, and 433/433 for
   speculative fail-closed. There is no oracle for hyper-vs-cpu matmul
   hybrid agreement.

5. **KV cache coherent across both backends.** When attention runs
   part on CPU and part on GPU, the KV cache has to stay consistent
   across CUDA-stream synchronisations. Today there is one KV cache
   per `qwn_decoder` and the current CUDA ABI version 1 doesn't
   participate in the CPU KV cache layout.

6. **Sequence length–dependent policy.** Small sequence lengths
   favour CPU VNNI (lower kernel launch cost). Large sequence lengths
   favour CUDA GEMV/GEMM (better arithmetic intensity). The current
   `--backend auto` heuristic uses only the dtype; a real hybrid needs
   shape-aware routing that adds latency to the dispatcher itself.

Each of those is an engineering project on its own. None of them is
in this repo today. A "few hours" implementation would be the third
item from my initial answer (CPU-first + CUDA-fallback on failure),
which is honest but is **not** the same thing as "use every resource
in parallel".

## Phased plan (NOT shipped)

### Phase P0 — capture-only telemetry flag (no hot-path changes)

Add a CLI/runtime flag in `c/qwnrun.c` and `c/qwanto_decode.c`
`--dispatch detail` that prints, for each forward pass, a per-layer
breakdown of:

- which kernel handler was called (CPU scalar, CPU AVX2, CPU VNNI,
  CUDA reference HyperVSQ-2, CUDA spec-attention, etc.)
- timing per layer
- which ops touched GPU vs CPU memory

This is observation only; no dispatch decisions are added. Cost:
small C patch + a JSON dump helper. Tests: existing 17 binaries plus
one new test that captures the dispatch-detail output for a fixed
prompt and asserts the schema.

### Phase P1 — per-layer measured routing (read-only policy)

For each layer, choose the backend based on the *last successful*
runtime for that layer on this hardware. Cache the choice in a
small LRU table; if a backend fails, fall back. This is the CPU-first +
CUDA-fallback pattern I originally proposed, narrowed to read-only
routing (never re-routes a layer that CPU successfully executed
mid-session). Cost: days to weeks because every layer's first-call
overhead dominates small batch measurements.

### Phase P2 — shape-aware active policy

Use shape information (`M`, `N`, `K`, and the `dtype`) plus a per-shape
empirical benchmark captured at startup to choose between CPU AVX2,
CPU VNNI, CUDA ref CUDA ref GEMV, CUDA ref CUDA ref GEMM. Requires a
warmup benchmark per layer per shape on the first decode; deferred
via cache. Cost: weeks plus a 100-differential oracle suite.

### Phase P3 — full hybrid with CUDA streams

Truly overlap CPU VNNI small-ops (LM head, sampling) with CUDA GEMV/GEMM
on subsequent layers. The pipeline would interleave CUDA kernels with
OpenMP work on the host. Needs the multi-stream ABI added to
`qwn_cuda_abi.h` (currently a single handle per process). Cost: large,
not in scope of any active release.

### Phase P4 — broaden CUDA coverage

Add CUDA kernels for the dtypes that are CPU-only today:
- `Q4_0` matmul on GPU (no current kernel)
- `Q5_K` / `Q6_K` dequant on GPU (no current kernel)
- VSQ / VSQ_ULTRA / HYPER_VSQ reference paths (none)
- BF16 LM head GEMV on GPU (currently CPU only)

This is the same work as `docs/dtype-support-roadmap.md` Phase 1+2
plus GPU kernels. Multi-quarter.

## What is out of scope for this roadmap

- "Use every resource in parallel" as a generic marketing claim. We
  will not advertise hybrid dispatch before phases P1 and P2 are
  measured end-to-end on a real release-quality prompt set.
- Tooling claim that CUDA is "always faster". The 4B RTX 5070 Ti
  measurements show CUDA is *not* always faster; the CPU VNNI path
  won by 5.9% at 128 tokens.
- A SKU claim that certain model classes need CUDA. The current best
  measured path for the only two MEASURED models is CPU.

## Status

- Phase P0: planned, not committed.
- Phase P1: planned, not committed.
- Phases P2-P4: documented only.

## References

- `benchmarks/evidence/windows/2026-08-22/4b_hyper_vsq2_cpu_128_for_compare.json` (CPU 8.784 tok/s)
- `benchmarks/evidence/windows/2026-08-22/4b_hyper_vsq2_cuda_128.json` (CUDA 8.216 tok/s, gpu_matmul_count=11264, cpu_fallback_count=0)
- `benchmarks/evidence/windows/2026-08-22/4b_hyper_vsq2_cuda.json` (CUDA 64 tok, 4.414 tok/s, gpu_matmul_count=7168)
- `benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_cuda_attempt.json` (1.5B Q4_0 CUDA fails closed)
- `c/glm.c:6113` (`COLI_CUDA` dispatch entry point)
- `c/cuda/qwn_cuda_abi.h:239-274` (versioned ABI surface)
- `docs/dtype-support-roadmap.md` Phase 1+2 (what CUDA dtype coverage would require)
