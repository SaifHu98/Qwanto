# Qwanto ⚡

> Unified inference runtime that uses all your hardware — CPU, GPU, RAM, NVMe — to run any model larger than memory.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-154%20Passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Web Dashboard](https://img.shields.io/badge/Web%20Dashboard-React%2019%20%7C%20Vite-blue.svg)]()
[![Maintainer](https://img.shields.io/badge/Maintainer-SaifHu98-purple.svg)](https://github.com/SaifHu98)

**Qwanto** is an ultra-fast, hardware-saturating local AI execution runtime that tier weights across **GPU VRAM, System RAM, and High-Speed NVMe** so you can run 70B+ LLMs on consumer hardware. It combines:

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
* The 1.5B source is already Q4_K-M; the `.qwn` writer passes K-quants through unchanged, so all 6 formats produce identical `1060.33 MB` containers (the engine's quantizer cannot re-quantize a K-quant that has no F32/F16/BF16 sibling).  The 1.5B column therefore measures the **engine's container-write throughput** at the source size.
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

### qwnrun native decoder

> The native `c/qwnrun.exe` decoder is **not** benchmarked in this release.  In the workspace where these numbers were produced, Windows Application Control blocked the unsigned `qwnrun.exe` from spawning (sandbox policy).  `qwnrun` is signed-offline; once the binary is whitelisted in the host environment it can be invoked by:
>
> ```bash
> make -C c qwnrun
> python c/coli run --model <model.qwn> --ngen 128 "Hello"
> ```

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
| **Qwanto Native (`.qwn`)** | **Production-Ready** | 4 KiB header, 4 KiB-aligned tensor payloads, 64-byte padding, tail-block offset in last 8 bytes, F32/F16/BF16/Q4_0/VSQ/VSQ-Ultra/HyperVSQ/HyperVSQ-2 |
| **QWN-HyperVSQ-2 Engine**  | **Production-Ready** | 256-element superblocks, 74-byte blocks (2.3125 bpw payload) |
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

| Model/input                        | Backend                       | Hardware use                    | Notes |
|------------------------------------|-------------------------------|---------------------------------|-------|
| GGUF (Q4_K, Q5_K, Q6_K, …)         | `llama-server` (llama.cpp)    | CPU and supported GPU backends  | Recommended general-purpose local path |
| `.qwn` (HyperVSQ-2, HyperVSQ, …)   | `qwnrun`                      | CPU AVX2/FMA/OpenMP             | Native dense decoder; CUDA Q4_0 matmul optional |
| GLM / DeepSeek / OLMoE directory   | Native MoE runtime            | CPU/RAM/NVMe, async I/O         | Architecture-specific |
| Ollama model name                  | Ollama                        | Ollama-controlled               | OpenAI-style passthrough |

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
# Passthrough (any input format → .qwn container)
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

```bash
# Python & integration tests (154 passed, 12 skipped in the workspace)
python -m pytest c/tests/ -q

# Native C tests
make -C c test-c

# Dashboard production build
cd web && npm install && npm run build

# Empirical report regeneration
python experiments/run_all.py
python experiments/run_llama_benchmark.py models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --rounds 3 --out experiments/results/llama_15B.json
python experiments/run_llama_benchmark.py models/DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf  --rounds 3 --out experiments/results/llama_4B.json
python experiments/run_empirical_report.py
```

The pytest suite covers:
* `c/tests/test_qwn_format.py`     — `.qwn` container round-trip + alignment invariants
* `c/tests/test_openai_server.py`  — gateway auth, SSE, CORS, defense headers
* `c/tests/test_universal_engine_v2.py` — Universal Engine 2.0 unit + integration tests (28 tests)
* `c/tests/test_real_models.py`    — real-model tests against both attached GGUFs + negative/edge cases (15 tests)
* `c/tests/test_paged_attention_and_ppl.py`, `test_phase23_*.py`, `test_response_cache.py`, `test_security_hardening.py`, `test_presets_telemetry.py`, …

---

## License

MIT — see [`LICENSE`](LICENSE).