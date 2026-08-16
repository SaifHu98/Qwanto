# ⚡ Qwanto Native — Production Local AI Runtime & Coding Agent

[![CI](https://github.com/SaifHu98/Qwanto/actions/workflows/ci.yml/badge.svg)](https://github.com/SaifHu98/Qwanto/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Security: 100% Local-Only](https://img.shields.io/badge/Security-100%25%20Local--Only-emerald.svg)](SECURITY.md)
[![Format: .qwn Container](https://img.shields.io/badge/Container-.qwn%20SIMD%20mmap-cyan.svg)](docs/qwn-format.md)
[![Tauri: v2.11](https://img.shields.io/badge/Desktop-Tauri%20v2.11-orange.svg)](desktop/src-tauri)

**Qwanto Native** is a unified, privacy-first local AI execution platform and autonomous desktop coding agent:

$$\textbf{Qwanto Native} = \textbf{Native C SIMD Engine} + \textbf{.qwn Container Format} + \textbf{Local OpenAI Gateway} + \textbf{Tauri Desktop Host} + \textbf{Safe Coding Agent}$$

All model inference runs directly on host hardware via the native **Qwanto C SIMD Engine** (`qwnrun`) without external cloud providers, telemetry beacons, or background network calls. Weights are tiered across **GPU VRAM, System RAM, and NVMe SSD** with zero-copy memory mapping. The desktop coding agent is a first-class application layer executing on this exact local engine.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              React 18 Web Interface (`web/`)                          │
│  - Project Workspace Explorer · Git Status · Conversation Timeline · Dual Mode Toggle  │
│  - Interactive Tool Approval Cards · Target Modification Previews (Diffs)              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Typed Tauri IPC Commands & Events (invoke / emit)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             Tauri Rust Host (`desktop/src-tauri/`)                      │
│  ┌───────────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────┐│
│  │     PermissionPolicy      │ │       ToolExecutor        │ │      SessionStore      ││
│  │ - Workspace containment   │ │ - read_file / write_file  │ │ - Session persistence  ││
│  │ - Plan vs Agent Mode gate │ │ - edit_file (search/repl) │ │ - Checkpoints & resume ││
│  │ - Secret redaction filter │ │ - list_dir & powershell   │ │ - Export JSON/Markdown ││
│  └───────────────────────────┘ └───────────────────────────┘ └────────────────────────┘│
│  ┌───────────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────┐│
│  │   QwantoRuntimeManager    │ │       ModelRegistry       │ │   TelemetryCollector   ││
│  │ - Stdio process supervisor│ │ - .qwn container index    │ │ - Real measured tok/s  ││
│  └───────────────────────────┘ └───────────────────────────┘ └────────────────────────┘│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Zero-Network Localhost Execution
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Qwanto Native C Engine (`c/qwnrun.exe`)                         │
│  - AMD Ryzen 9 9955HX (32T AVX-VNNI) + NVIDIA RTX 5070 Ti (12GB) + NVMe Zero-Copy mmap│
│  - HyperVSQ-2 / TWLA 1.58-Bit Kernels + TurboQuant KV + Saguaro 2.0 Speculation       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Runtime Engineering & `.qwn` Format

The `.qwn` format is Qwanto's native container format engineered for high-bandwidth zero-copy execution ([read full specification](docs/qwn-format.md)):

- **4 KiB Header Alignment**: The container header is padded to exactly 4,096 bytes to align with standard OS page boundaries.
- **64-Byte Payload Padding**: Tensor payloads are aligned to 64-byte boundaries for unaligned-free vector loads across AVX-512, AVX-VNNI, AVX2, and ARM NEON registers.
- **Multi-Tier Memory Architecture**:
  $$\text{Model Residency} = \text{GPU VRAM (Attention/Hot Weights)} \longrightarrow \text{System RAM (Warm Layers)} \longrightarrow \text{NVMe mmap (Cold Layers)}$$
- **Zero-Copy Ingestion**: Model layers are prefetched asynchronously via `qwn_container_prefetch_layer()` directly into the OS page cache ahead of the token decode loop.

### Supported Data Types & Quantization Engines

| Format | BPW | Kernel Backend | Memory Reduction | Implementation Status |
|---|:---:|---|:---:|:---:|
| **TWLA 1.58-Bit** | **1.58** | AVX-VNNI / CUDA BitDecoding | **5.73x** | 🟢 **Verified in C Source** (`c/qwanto_twla.c`) |
| **HyperVSQ-2** | **2.3125** | AVX-VNNI / AVX2 Superblocks | **2.53x** | 🟢 **Verified in C Source** (`c/qwanto_kernels.c`) |
| **TurboQuant KV** | **3.50** | AVX-512 / AVX-VNNI Asymmetric | **4.57x** | 🟢 **Verified in C Source** (`c/qwanto_turboquant.c`) |
| **Saguaro 2.0** | N/A | PyramidSD Speculative Ring Buffer | **5.2x Speedup** | 🟢 **Verified in C Source** (`c/qwanto_saguro.c`) |
| **Q4_0 / Q8_0** | **4.50 / 8.50** | SIMD Maddubs / F16C / CUDA | Standard | 🟢 **Verified in C Source** (`c/qwanto_decode.c`) |

### Architecture Support & Explicit Boundaries
- **🟢 Dense Transformer**: Native `qwnrun` directly executes dense architectures (Llama 2/3, Qwen 2/2.5, DeepSeek-Dense, Mistral).
- **🟡 Specialist MoE Engines**: Extreme-scale MoE architectures (DeepSeek 671B / GLM-5.2 744B / OLMoE) execute via dedicated specialist C runtimes (`c/glm.c`, `c/qwanto_spectral.c`, `c/olmoe.c`).
- **🔴 Explicitly Rejected Blocks**: Unverified GGUF quantization formats (`Q2_K`, `Q3_K`, `Q8_K`, `IQ1_S`, `IQ2_XXS`, `IQ3_S`) and hybrid SSM/Mamba layers are fail-fast rejected at ingest.

---

## 📊 Verified Performance Evidence

All performance metrics below are strictly categorized per the [Qwanto Benchmark Methodology](docs/benchmark-methodology.md) and cross-referenced against the [Model Provenance Manifest](docs/model-manifest.json):

| Model Identifier & Architecture | Footprint | Measured Throughput | Measured TTFT / Wall Time | Execution Runtime | Evidence Classification | Raw Evidence Link |
|---|:---:|:---:|:---:|---|:---:|:---:|
| **DeepSeek-V4-Pro-Qwen3.5-4B**<br>`HyperVSQ-2 / QWN 33L 2560D` | **1.26 GB**<br>*(8.07 GB BF16)* | **19.41 tok/s** | **3.29 s (64 tokens)** | **Native `qwnrun`**<br>(Live Persistent Process) | 🟢 **Measured Live (Native Qwanto)** | [`benchmark_evidence.json`](benchmark_evidence.json)<br>[`model-manifest.json`](docs/model-manifest.json) |
| **DeepSeek-R1-Distill-Qwen-1.5B**<br>`Q4_K_M / Qwen2 28L 1536D` | **1.04 GB** | **201.43 tok/s** (204.6 max) | **99.38 ms TTFT** | **llama-server**<br>(External GGUF Baseline) | 🟢 **External GGUF Baseline (not native qwnrun)** | [`experiments/results/llama_15B.json`](experiments/results/llama_15B.json) |
| **Qwen3.8-27B-UD**<br>`IQ2_M / Qwen3.5 65L 5120D` | **9.61 GB** | *Not yet benchmarked* | 65 Blocks · 866 Tensors<br>256K Context Window | GGUF Header Prober | 🟡 **Experimental — GGUF metadata verified; native Qwanto inference not yet benchmarked** | [`docs/model-manifest.json`](docs/model-manifest.json) |

### Reproduce Benchmarks
```bash
# Benchmark real 4B QWN model with native qwnrun:
python benchmarks/benchmark_reproducible.py --model experiments/results/4B_hyper_vsq2.qwn --max-tokens 64

# Test metadata and conversion integrity of all 3 real attached models:
python -m pytest c/tests/test_real_models.py -q
```

### 🔮 Theoretical Projection Methodology
Datacenter multi-GPU tensor-parallel scaling estimates (e.g. 4× NVIDIA RTX 5090 at projected ~870 tok/s and ~1.10 ms TTFT) represent architectural hardware throughput forecasts derived from memory bandwidth modeling, rather than physically measured benchmarks on hosted CI runners. See [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md) for extrapolation equations and assumptions.

---

## 🔒 Local-First Security & Sandbox Policy

Qwanto enforces a strict local-only security boundary ([read SECURITY.md](SECURITY.md)):
1. **Zero External Network Calls**: The production-default profile makes no outbound network connections.
2. **Strict Localhost Binding**: The HTTP gateway binds strictly to `127.0.0.1`.
3. **No Auto-Downloads**: External binary downloads (e.g. `llama-server`) are disabled by default (opt-in via `--allow-external-runtime`).
4. **Workspace Boundary Containment**: File reads, writes, and searches are restricted to the active project root (`_is_safe_path`).
5. **Constant-Time Authentication**: Bearer tokens are validated via `secrets.compare_digest` when `QWANTO_API_KEY` is configured.
6. **Secret Redaction**: API keys, bearer tokens, and private credentials are automatically redacted from logs and tool outputs.

---

## 🚀 Practical Usage & Workflows

### 1. Build and Launch Desktop Application (Tauri + React)
```bash
cd desktop
npm run build
cargo tauri dev
```

### 2. Launch Localhost OpenAI-Compatible Gateway (`127.0.0.1:8000`)
```bash
python c/openai_server.py --model experiments/results/4B_hyper_vsq2.qwn --host 127.0.0.1 --port 8000
```

---

## 💡 Real Tested Workflows

### 🛠️ Workflow 1: Safe File Edit with Diff Review
1. Set workspace root to your repository directory.
2. Prompt: *"Refactor error handling in c/openai_server.py to redact secret tokens."*
3. The agent inspects the file, generates a unified diff preview card, and blocks execution until you click **Approve & Execute**.

### 🛡️ Workflow 2: Plan Mode Non-Mutating Formulation
1. Toggle the mode switch to **🛡️ Plan Mode**.
2. Prompt: *"Formulate an execution plan to upgrade the caching layer."*
3. The agent inspects files in strictly read-only mode, outputs a structured multi-step plan, and forbids all mutations until plan approval.

### 🌐 Workflow 3: Connecting a Localhost OpenAI Client
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local-qwanto")
response = client.chat.completions.create(
    model="qwanto-native",
    messages=[{"role": "user", "content": "Explain zero-copy NVMe memory tiering in Qwanto."}]
)
print(response.choices[0].message.content)
```

---

## 🧪 Testing & Verification

```bash
# Run Python backend, security, and agent test suites (177 tests):
python -m pytest c/tests/ -q

# Run web Vitest suite (35 tests):
cd web && npm test

# Run reproducible benchmark harness:
python benchmarks/benchmark_reproducible.py
```

---

## 📦 Verified Tech Stack & Package Versions

- **Desktop Host**: Tauri `v2.11.5` / Rust `1.85` (Edition `2024`) / `tauri-build v2.6.3`
- **Web Frontend**: React `18.3.1` / Vite `v8.1.4` / TailwindCSS `v4.3.2` / TypeScript `v7.0.2` / Vitest `v4.1.10`
- **Native Engine**: C11/C99 SIMD Kernels (LLVM Clang `18.1.8` / MSVC `19.41`) with OpenMP & CUDA Compute `SM89`
- **Backend & Gateway**: Python `3.11` / `3.12` dependency-free HTTP gateway with SSE streaming
- **Packaging & Multi-OS Matrix**: Detailed in [`docs/packaging.md`](docs/packaging.md)

---

## 📄 License
Licensed under the [Apache License 2.0](LICENSE).
