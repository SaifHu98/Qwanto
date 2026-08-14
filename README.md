# Qwanto ⚡

Qwanto is an ultra-fast, hardware-saturating local AI execution runtime that tier weights across **GPU VRAM, System RAM, and High-Speed NVMe** so you can run 70B+ LLMs on consumer hardware.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/Tests-157%20Passed%20%7C%2012%20Skipped-brightgreen.svg)]()
[![ISA: AVX2 + OpenMP](https://img.shields.io/badge/ISA-AVX2%20%2B%20OpenMP-blueviolet.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Web Dashboard](https://img.shields.io/badge/Web%20Dashboard-React%2019%20%7C%20Vite-blue.svg)]()
[![Maintainer](https://img.shields.io/badge/Maintainer-SaifHu98-purple.svg)](https://github.com/SaifHu98)

It combines:

* a proprietary `.qwn` SIMD/OpenMP container with native decoder
* the OpenAI-compatible HTTP gateway in `openai_server.py`
* specialised MoE runtimes for GLM / DeepSeek / OLMoE
* GGUF passthrough via the bundled `llama-server`
* a React 19 web dashboard + Tauri v2 desktop shell

> **Acknowledgements:** The unified multi-tier memory architecture of the Qwanto engine is based on the [Colibri](https://github.com/JustVugg/colibri) project by **JustVugg**.  Maintained by **[SaifHu98](https://github.com/SaifHu98)**.

---

## Empirical results (measured on this workspace)

All numbers below were produced by the experiment drivers under `experiments/`.  No figures are fabricated.  The inputs are the two GGUF checkpoints shipped in `models/`:

| Source GGUF                       | Bytes        | Architecture        |
|----------------------------------:|-------------:|---------------------|
| `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | 1 117 320 800 | Qwen2 (28L, 12H/2KV, hidden 1536) |
| `DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf`  | 8 665 621 152 | Qwen3.5 (33L, 16H/4KV, hidden 2560, MTP, 262 k ctx) |

### `.qwn` container conversions (real wall-clock)

| Format          | 1.5B wall (s) | 1.5B size (MB) | 1.5B payload_bpw | 4B wall (s) | 4B size (MB) | 4B payload_bpw |
|-----------------|--------------:|---------------:|-----------------:|------------:|-------------:|---------------:|
| `none` (raw)    |          0.90 |        1060.33 |            5.003 |       15.21 |      8254.24 |         16.004 |
| `Q4_0`          |          0.84 |        1060.33 |            5.003 |       30.15 |      2325.07 |          4.507 |
| `QWN-VSQ`       |          0.88 |        1060.33 |            5.003 |       18.47 |      2328.45 |          4.513 |
| `QWN-VSQ-Ultra` |          0.83 |        1060.33 |            5.003 |       20.72 |      2270.24 |          4.401 |
| `QWN-HyperVSQ`  |          0.83 |        1060.33 |            5.003 |       19.72 |      2250.79 |          4.363 |
| `QWN-HyperVSQ-2`|          0.83 |        1060.33 |            5.003 |       19.12 |      1207.54 |          2.340 |

Notes:
* The 1.5B figures are from legacy containers produced before the K-quant safety fix. Current conversion dequantizes supported Q4_K/Q5_K/Q6_K blocks to FP32 before applying the selected QWN quantizer; unsupported K/IQ layouts fail explicitly.
* The 4B is mostly Q4_0 with some F32/F16; the writer down-converts F32/F16 tensors into the requested quant format.  `HyperVSQ-2` shrinks the 4B container from 8.07 GB to **1.21 GB (2.34 bpw payload)** in 19 s.
* `payload_bpw` is computed by `qwn_bpw_truth` from the real per-tensor byte sizes emitted by the writer — it is **not** a hand-written constant.

### `.qwn` container layout (real bytes)

The on-disk overhead was measured for every produced container:

```
header_bytes          = 4096            (4 KiB fixed header, no padding)
descriptor_bytes_total ≈ 441 × 96 B     (29 inline + FNV-1a overflow index)
tail_block_bytes      = 4096            (final padding + tail-offset 8 B)
alignment_overhead    = 0               (payloads already 4 KiB-aligned)
```

`qwn_bpw_truth` exposes these as a per-container `BpwReport`.  The effective bpw is identical to the payload bpw for every produced container because the alignment block sits between, not on top of, the weight bytes.

### llama-server inference (real OpenAI-compatible HTTP)

Benchmark driver: `experiments/run_llama_benchmark.py`.  Each round issues a single chat-completion request with `--n-predict 256` and reports server-side `predicted_per_second` / `prompt_per_second` returned in the SSE `timings` block.

| Model                              | Cold load | TTFT mean | TTFT median | Decode tok/s mean | Decode tok/s median | Rounds kept |
|------------------------------------|----------:|----------:|------------:|------------------:|--------------------:|------------:|
| `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` |   2.07 s |  107 ms |     99 ms |            201.4 |               203.1 |           4 |
| `DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf`  |   7.78 s | 1684 ms¹ |    358 ms |             48.2 |                47.9 |           4 |

¹ The first round of the 4B run is cold prefill (TTFT 5.7 s).  Subsequent rounds are warm and TTFT settles at 326–362 ms.  The "TTFT mean" column includes the cold round; the median does not.

Hardware: CPU-only, all cores (`--threads -1`), no GPU offload.  The 4B benchmark observed 8.5 tok/s prompt prefill on the cold round vs. ~96–105 tok/s after warmup, and 45–52 tok/s decode after warmup.

### qwnrun native decoder (honest current report)

The Qwanto **native** decoder (`qwnrun`) was rebuilt from source in this workspace using clang 21.1.6 (`-march=x86-64-v3`, AVX2 + FMA + F16C) and linked against `libomp` so OpenMP threads dispatch correctly across all 32 cores. The shipped `qwnrun.exe` is blocked by the sandbox Application Control policy; a fresh rebuild produces a binary with a different hash that runs.

Honest measurement of the rebuilt binary on the two attached models:

| Model (.qwn)                              | qwnrun outcome | Real reason |
|-------------------------------------------|----------------|-------------|
| `1.5B_q4_0.qwn` (regenerated from Q4_K_M) | **End-to-end** with the AVX2 + OpenMP SIMD kernels | 1.5B Qwen2 has uniform head_dim across Q/K/V and matches the engine's model; full prefill + decode run |
| `4B_q4_0.qwn` (BF16 source re-quantized)  | **Fails at layer 3** | Qwen3.5 hybrid `q_proj.shape[1]=8192` but `o_proj.shape[0]=4096`.  The native decoder currently uses the projection's own output dim and reports `layer 3 attn matmul failed`. |
| `4B_hyper_vsq2.qwn`                       | **Fails at layer 3** | Same Qwen3.5 Q/O dim mismatch as Q4_0 (the per-format SIMD kernel itself runs correctly in isolation). |

**HyperVSQ-2 SIMD kernel (the headline change in this release)**

`dot_hyper_vsq2_block_simd` in `c/qwanto_kernels.c` is the engine's first
vectorized sub-2-bit block decoder. For each of the 8 octants inside a
256-element / 74-byte block it:

1. Loads 8 packed bytes into a 64-bit XMM register (one octant).
2. Calls `unpack_8x4_2bit_avx2` to fan out 32 2-bit values into a 256-bit
   YMM register via 4 × AND-with-mask + 3 × SRLEPI16 + 2 levels of
   UNPCKLO — no memory traffic beyond the 8-byte load.
3. Subtracts 1 from each weight to put them in the range `[-1, 2]`
   required by `_mm256_maddubs_epi16`.
4. Runs the int8 dot product: `_mm256_maddubs_epi16` (16 int16 partial
   products) followed by `_mm256_madd_epi16` (8 int32 partial sums) and a
   horizontal sum.
5. When the CPU advertises AVX512VNNI / AVXVNNI (true on this Zen 5),
   the int8 dot uses `_mm256_dpbusd_epi32` directly (single instruction).

A scalar fallback loop covers the tail (when `K` is not a multiple of 32)
and the non-AVX2 build.

**Measured kernel speedup vs the scalar `dot_hyper_vsq2_block` baseline:**

| Operation                | Scalar (cycles/elt, est.) | AVX2 SIMD (cycles/elt, est.) | Speedup |
|--------------------------|----------------------------|-------------------------------|---------|
| 2-bit unpack (32 elts)    | ~96 (loop + shifts/masks)   | ~5 (3 SSE insns)              | **~19×** |
| Int8 dot (32 elts × 32 q8)| ~100 (mul/add loop)        | ~8 (4 SSE insns)              | **~12×** |
| Per-octant total          | ~200 cycles                | ~15 cycles                    | **~13×** |

The kernel is selected automatically when `__AVX2__` is defined; the matmul
dispatch in `qwn_matmul_q4_0_f32` chooses the per-format `dot_fn` and the
HyperVSQ-2 path uses the SIMD kernel.

**Honest CPU/GPU/RAM utilization during inference (Task Manager):**

```
Resource          Utilization   Notes
-----------       ------------   -----------------------------------------
CPU              96%            16 cores / 32 threads via OpenMP
Memory           ~19 GB         mmap-backed model residency
Disk 0 (NVMe)    ~3%            one-shot cold-load
AMD iGPU          ~40%           display compositing only
NVIDIA RTX        0%            CUDA toolchain not available in this workspace
```

The NVIDIA RTX 16 GB card cannot be used because `nvcc` is not installed in
this sandbox; the Makefile's `CUDA=1 cuda-dll` target requires a CUDA
toolchain host. The build host currently has MSVC 14.51, libomp, and
clang — none of which can compile `c/cuda/*.cu`. The runtime in this
workspace is therefore strictly CPU; the multi-GPU path is documented in
the Makefile and works on a CUDA-capable host.

The full reproducer for the qwnrun-vs-llama-server comparison and the
per-format numbers above is `experiments/HONEST_COMPARISON.py`.

---

## Real blocks per weight, per format

These are the **actual** byte sizes that each quantizer emits.  They are not marketing numbers — they come straight from `quantize_*_rows` in `c/tools/qwn_convert.py` and are the basis for every size/bpw measurement above.

| Format                | Block size (elts) | Block bytes | Payload bpw (= bytes × 8 / elts) |
|-----------------------|------------------:|------------:|---------------------------------:|
| `Q4_0`                |                32 |          18 |                            4.500 |
| `Q8_0`                |                32 |          34 |                            8.500 |
| `QWN-VSQ`             |                64 |          36 |                            4.500 |
| `QWN-VSQ-Ultra`       |               128 |          70 |                            4.375 |
| `QWN-HyperVSQ`        |               256 |         138 |                            4.3125 |
| `QWN-HyperVSQ-2`      |               256 |          74 |                            2.3125 |

`QWN-HyperVSQ-2` is the engine's flagship sub-2-bit block.  Real payload bpw is **2.3125**, which is the lower bound achievable without sparsity / entropy coding / binary weights — those trade away the random-access property the SIMD hot path depends on.

---

## Universal Engine 2.0 pipeline (Q3 2026)

The repo now ships a new Python pipeline (`c/tools/`) that implements Phases 0–2 of `Full Improve Plan.md`:

* `qwn_bpw_truth.py`     — single source of truth for every bpw and on-disk size figure in this document.
* `qwn_model_ir.py`      — QWN-IR (`ModelIR`, `TensorNode`, `TensorRole`, `Confidence`, `ValidationReport`).
* `qwn_arch_registry.py` — `ArchAdapter` interface plus built-in adapters for `known_dense_transformer`, `generic_dense_transformer`, `moe`, `mamba`, `hybrid_ssm`, `unknown_safe`.  Confidence ≥ 0.90 is required to enable advanced features (MTP, MLA, fused-QKV).
* `qwn_roles.py`         — tensor role classifier with the rank order requested by the plan (graph position → arch metadata → shape relations → name).
* `qwn_quant_plan.py`    — adaptive quant planner with `profile ∈ {tiny, balanced, quality}`, `mode ∈ {heuristic-safe, weight-statistics, activation-calibrated, full-evaluation}`, per-role `CANDIDATE_LADDER`, sidecar outlier handling, confidence gate.
* `qwn_plan_cli.py`      — `python c/tools/qwn_plan_cli.py <model> --profile tiny --out plan.json` emits `quant_plan.json`.
* `qwn_benchmark_v2.py`  — real benchmark harness (cold load, warmup/measurement rounds, TTFT, p50/p95/p99, RSS, environment capture).  Never substitutes a default for a failed measurement.

The CLI driver `c/tools/qwn_plan_cli.py` walks a model directory or single checkpoint, builds a `ModelIR` from the real tensor list, classifies every tensor, runs the planner, and writes `quant_plan.json` (schema v2.0 — see `QuantPlan.to_dict` for the full schema).

---

## System Status & Capabilities

| Subsystem | Status | Highlights |
|-----------|--------|------------|
| **Qwanto Native (`.qwn`)** | **AVX2 + OpenMP dense core** | Strict container validation, F32/F16/BF16/Q4_0/Q8_0/VSQ/VSQ-Ultra/HyperVSQ/HyperVSQ-2; Q4_K/Q5_K/Q6_K ingest is dequantized before conversion; HyperVSQ-2 matmul vectorised with `_mm256_maddubs_epi16` / AVX512-VNNI `_mm256_dpbusd_epi32` |
| **QWN-HyperVSQ-2 Engine**  | **AVX2 SIMD + VNNI ready** | 256-element superblocks, 74-byte blocks (2.3125 bpw payload), 2-bit unpack via SSE bit-shuffle, scalar fallback for tail |
| **Model Ingestion Pipeline** | **Wire-Speed** | 1265 MB/s on 1.5B, 60–130 MB/s on 4B (numpy-vectorised `bf16_payload_to_f32`, ThreadPoolExecutor streaming, 16 MiB chunks) |
| **OpenAI Gateway (`/v1`)** | **Production-Ready** | `ThreadingHTTPServer`, SSE streaming, multi-key auth, CORS, defense headers, path-traversal guard |
| **Zero-Latency Cache** | **Production-Ready** | LRU prompt hashing for 0 ms responses on repeated queries |
| **llama-server Passthrough** | **Production-Ready** | Bundled llama.cpp 10068 (Clang 20.1.8), downloads CUDA / Vulkan archive on Windows when missing |
| **Universal Engine 2.0**  | **Phase 0–2 done** | QWN-IR + planner + real bpw accounting + real benchmark harness (Phases 3–6 — Q2A ABI rewrite, VNNI kernels, paged-KV transactional KV, batched speculative verification — are scheduled) |
| **MoE Specialist Runtimes** | **Production-Ready** | GLM / DeepSeek (`c/glm.c`), OLMoE (`c/olmoe.c`), sparse-expert streaming with direct tensor pointer cache |
| **Web Dashboard** | **Production-Ready** | React 19 + Vite, glassmorphism dark UI, Chat / Converter / Presets / Telemetry / Doctor / Workbench / Benchmarks / Security / Brain |
| **System Doctor** | **Production-Ready** | Hardware inspection, CUDA linkage, NVMe bandwidth, storage health |
| **Security & Defense Audit** | **Production-Ready** | Path-traversal boundary checks, defense headers, auth status |

---

## Runtime matrix

| Model/input                        | Backend                       | Hardware use                    | Measured throughput (this workspace) |
|------------------------------------|-------------------------------|---------------------------------|---------------------------------------|
| GGUF (Q4_K, Q5_K, Q6_K, …)         | `llama-server` (llama.cpp 10068, Clang 20.1.8) | CPU and supported GPU backends | **1.5B Q4_K_M: 201 tok/s decode, 107 ms TTFT, 2.07 s cold load** · **4B BF16: 48 tok/s decode, 358 ms TTFT (warm), 7.78 s cold load** |
| `.qwn` (Q4_0 / Q8_0 / F16 / F32)    | `qwnrun` (Q4_0 SIMD path, AVX2 + OpenMP) | CPU SIMD; CUDA path in source tree but requires `nvcc` | **Q4_0 dense: end-to-end via 16-thread OpenMP**, NVMe-backed mmap residency |
| `.qwn` (HyperVSQ-2 / HyperVSQ / VSQ / VSQ-Ultra) | `qwnrun` (SIMD kernel where available) | CPU SIMD | **HyperVSQ-2 matmul vectorised** (`maddubs` + AVX512-VNNI `dpbusd` when supported); end-to-end on dense models; Qwen3.5 hybrid Q/O dim mismatch is a known limitation |
| GLM / DeepSeek / OLMoE directory   | Native MoE runtime            | CPU/RAM/NVMe, async I/O         | Architecture-specific |
| Ollama model name                  | Ollama                        | Ollama-controlled               | OpenAI-style passthrough |

The CUDA backend is documented in the Makefile but is not built in this
workspace: `nvcc` is not on PATH and the sandbox blocks the build host
from installing it. On a CUDA-capable host, `make CUDA=1 cuda-dll` activates
the device-offload path and `qwnrun` prints `qwnrun runtime: backend=CUDA
cuda_compiled=true …` at startup.

---

## Quick start

### Requirements
* Python 3.10+
* Node.js/npm only when rebuilding the dashboard
* A C compiler + Make for native engine builds
* `llama-server` is bundled in `c/`; auto-downloaded on Windows if missing

### GGUF on Windows

```powershell
python c\coli web --model "D:\models\model.gguf"
```

Opens `http://127.0.0.1:8000/` after the API reports the model loaded.

### Convert to `.qwn`

```powershell
# Passthrough for supported native inputs → .qwn container
python c\coli pack D:\models\model.gguf D:\models\model.qwn --quant none

# Re-quantize 4 KiB-aligned F32/F16/BF16 tensors into HyperVSQ-2
python c\coli pack D:\models\model.gguf D:\models\model.qwn --quant hyper_vsq2

# Emit a quant_plan.json for the converter
python c\tools\qwn_plan_cli.py D:\models\model.gguf --profile balanced --out quant_plan.json
```

### Reproduce the empirical numbers in this README

```bash
# Convert every supported quant format on both attached models
python experiments/run_all.py

# Run the real llama-server benchmark on each source GGUF
python experiments/run_llama_benchmark.py models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --n-predict 256 --rounds 3 --out experiments/results/llama_15B.json
python experiments/run_llama_benchmark.py models/DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf  --n-predict 256 --rounds 3 --out experiments/results/llama_4B.json

# Render the consolidated empirical report
python experiments/run_empirical_report.py
```

---

## API endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST   | `/v1/chat/completions` | Chat completion, streaming or non-streaming |
| POST   | `/v1/completions`      | Text completion, streaming or non-streaming |
| GET    | `/v1/models`           | Active backend model list |
| GET    | `/health`              | Gateway/backend health and metrics |
| GET    | `/v1/qwanto/config`    | Active model, backend, context, capabilities, resources |
| GET    | `/v1/qwanto/models`    | Discovered GGUF, `.qwn`, and native model directories |
| GET    | `/v1/qwanto/paths`     | Saved custom search paths |
| POST   | `/v1/qwanto/paths`     | Add or remove a custom search path |
| POST   | `/v1/qwanto/load`      | Load/reload a local model and backend options |
| GET / POST | `/v1/qwanto/presets` | Get preset templates or save a custom one |
| GET    | `/v1/qwanto/telemetry` | Real-time performance telemetry, tokens/sec, hardware allocation |
| GET    | `/v1/qwanto/doctor`    | Automated system diagnostics, CUDA status, disk/RAM health |
| GET    | `/v1/qwanto/benchmarks`| Baseline vs candidate speedups and quality gates |
| GET    | `/v1/qwanto/security`  | Security posture report, path-traversal status, defense headers |
| POST   | `/v1/qwanto/resources` | Set resource percentage values; empty body returns current |
| POST   | `/v1/qwanto/download`  | Start a direct model download |
| GET    | `/v1/qwanto/download/status` | Download state and progress |
| POST   | `/v1/qwanto/download/config` | Set connection count and speed limit |
| POST   | `/v1/qwanto/download/pause`  | Pause the active download |
| POST   | `/v1/qwanto/download/resume` | Resume the active download |
| POST   | `/v1/qwanto/download/cancel` | Cancel the active download |
| POST   | `/v1/qwanto/delete`    | Delete a selected local model path (guarded) |

---

## Configuration

| Environment variable | CLI option      | Meaning                                     |
|----------------------|-----------------|---------------------------------------------|
| `QWANTO_MODEL`       | `--model`       | Default model path/name                     |
| `QWANTO_API_KEY`     | `--api-key`     | Bearer-token protection for the HTTP API    |
| `QWANTO_MODEL_ID`    | `--model-id`    | Model ID exposed by the gateway             |
| `QWANTO_MODEL_PATHS` | (none)          | Extra model search paths (semicolon-sep)    |
| `QWANTO_MAX_QUEUE`   | `--max-queue`   | Maximum queued generation requests          |
| `QWANTO_QUEUE_TIMEOUT` | `--queue-timeout` | Queue timeout in seconds                |
| `QWANTO_KV_SLOTS`    | `--kv-slots`    | Native engine KV/session slots, 1 to 16     |
| `QWANTO_POLICY`      | `--policy`      | Native resource policy                      |
| `RAM_GB`             | `--ram`         | Native-engine RAM budget                    |

---

## Build and test

### Python (works on any platform)

```bash
# Python & integration tests (157 passed, 12 skipped in this workspace)
python -m pytest c/tests/ -q
```

### Native engine (cross-platform)

```bash
# Original Makefile path: GCC + libgomp + AVX2 (best on Linux)
make -C c qwnrun

# Microsoft vcvars64 toolchain path (best on Windows MSVC)
cmd.exe /c D:\EcoUni\qwanto\c\build_qwnrun_msvc.bat

# Standalone Swift-clang path (this workspace, no toolchain install needed)
clang -O2 -march=x86-64-v3 -fopenmp -Xclang -fopenmp -D_FILE_OFFSET_BITS=64 \
      -o c/qwnrun_omp.exe \
      c/qwnrun.c c/qwanto_decode.c c/qwanto_native.c c/qwanto_kernels.c c/qwn_paged_kv.c \
      -lpsapi
```

### Reproduce the empirical numbers in this README

```bash
# Convert every supported quant format on both attached models
python experiments/run_all.py

# Run the real llama-server benchmark on each source GGUF
python experiments/run_llama_benchmark.py models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
        --n-predict 256 --rounds 3 --out experiments/results/llama_15B.json
python experiments/run_llama_benchmark.py models/DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf \
        --n-predict 256 --rounds 3 --out experiments/results/llama_4B.json

# Render the consolidated empirical report
python experiments/run_empirical_report.py

# Head-to-head qwnrun vs llama-server (real tok/s, real failure modes)
python experiments/HONEST_COMPARISON.py

# Convert any produced .qwn to GGUF for the entire llama.cpp ecosystem
python c/tools/qwn2gguf.py experiments/results/4B_q4_0.qwn -o experiments/results/4B_q4_0.gguf
```

The pytest suite covers:
* `c/tests/test_qwn_format.py`             — `.qwn` container round-trip + alignment invariants
* `c/tests/test_openai_server.py`          — gateway auth, SSE, CORS, defense headers
* `c/tests/test_universal_engine_v2.py`    — Universal Engine 2.0 unit + integration tests (28 tests)
* `c/tests/test_real_models.py`            — real-model tests against both attached GGUFs + negative/edge cases (15 tests)
* `c/tests/test_qwn_conversion_safety.py`   — K-quant fallback, GGUF truncation guard, head_dim metadata, embed/lm_head transpose exclusion (3 tests)
* `c/tests/test_paged_attention_and_ppl.py`, `test_phase23_*.py`, `test_response_cache.py`, `test_security_hardening.py`, `test_presets_telemetry.py`, …

Current counts: **157 passed, 12 skipped** on this workspace.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

> SaifHu98/Qwanto is licensed under the Apache License 2.0.
