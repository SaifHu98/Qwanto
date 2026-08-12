# Qwanto ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Tests-108%20Passed-brightgreen.svg)]()
[![Frontend](https://img.shields.io/badge/Web%20Dashboard-React%2019%20%7C%20Vite-blue.svg)]()
[![Author](https://img.shields.io/badge/Maintainer-SaifHu98-purple.svg)](https://github.com/SaifHu98)

Qwanto is a high-performance local inference gateway, specialized MoE C engine, and modern web workspace. It provides an OpenAI-compatible HTTP API, live telemetry metrics, prompt tuning studio, interactive API workbench, model manager, a native GLM-5.2/OLMoE execution path, GGUF integration via `llama-server`, and an experimental compact model format named Qwanto Native (`.qwn`).

### ⚡ Unified Inference Engine
The Qwanto core engine acts as a **unified inference runtime that uses all your hardware — CPU, GPU, RAM, NVMe — to run any model larger than memory** at maximum performance. 

> **Acknowledgements:** The core unified multi-tier memory execution architecture of the Qwanto engine is proudly based on the exceptional [Colibri](https://github.com/JustVugg/colibri) project created by **JustVugg**.

Developed and maintained by **[SaifHu98](https://github.com/SaifHu98)**.

<p align="center">
  <img width="2816" height="1536" alt="Qwanto dashboard" src="https://github.com/user-attachments/assets/6c98e09d-12bb-4261-95da-f154b74f5235" />
</p>

## Status

| Area | Status | Scope |
|------|--------|-------|
| OpenAI-compatible gateway | Working | Chat/text completions, SSE streaming, model listing, API-key protection, CORS |
| Prompt Studio & Tuning | **New** | Custom system prompts, temperature/top-p presets, 1-click studio templates |
| Live Telemetry & Metrics | **New** | Tokens/sec tracking, generation throughput, hardware allocation, request telemetry |
| API Workbench | **New** | Multi-language code generator (cURL, Python, TypeScript, Rust) |
| System Doctor Diagnostics | **New** | Automated installation, CUDA linkage, storage permission, and hardware verification |
| Security & Defense Audit | **New** | Path traversal boundary enforcement, HTTP defense headers (`nosniff`, `DENY`), constant-time auth |
| Verified Benchmarks & Gates | **New** | Empirical baseline vs candidate speedup reporting and automated regression gate checks |
| GGUF runtime | Working through llama.cpp | Uses an installed or Windows auto-downloaded `llama-server` |
| Native GLM runtime | Working for its target architecture | GLM-5.2 MoE engine with CPU/RAM/NVMe execution and optional GPU backends |
| Native OLMoE runtime | Architecture-specific | Separate OLMoE C runtime and conversion path |
| Ollama adapter | Working | CLI/API proxy to a local Ollama server |
| Qwanto Native `.qwn` | Experimental | Dense Llama/Qwen-style decoder graph with important compatibility limits |
| Web dashboard | Working | Chat, prompt studio, live telemetry, API workbench, doctor diagnostics, security audit, benchmarks, logs, resource controls |

The current verification snapshot is:

- `108 passed, 3 skipped` with `python -m pytest c/tests/ -q`
- Scalar and AVX2 `.qwn` C tests pass
- Native decoder logits match an independent Python reference within expected FP16 KV-cache tolerance
- Persistent native engine protocol test passes
- Frontend TypeScript/Vite production build passes (`dist/` build verified)
- System Doctor diagnostics and Security Audit checks passed

## Runtime Matrix

| Model/input | Backend | Hardware use | Notes |
|-------------|---------|--------------|-------|
| GGUF | llama.cpp | CPU and supported llama.cpp GPU backends | Recommended general-purpose local path |
| GLM-5.2 converted directory | Qwanto native GLM engine | CPU, RAM, NVMe; optional CUDA/Metal paths | Specialized MoE runtime, not a generic Hugging Face decoder |
| OLMoE converted directory | `olmoe` runtime | CPU/RAM/disk path | Architecture-specific |
| Ollama model name | Ollama | Controlled by Ollama | Qwanto forwards OpenAI-style requests |
| `.qwn` | `qwnrun` | CPU scalar/AVX2; optional CUDA Q4_0 matmul | Experimental dense decoder; see limitations |

"Supported by GGUF" means supported by the installed `llama-server` version.
It does not mean that every model architecture or quantization is guaranteed to
load.

## Main Features

### OpenAI-Compatible HTTP Gateway

- `POST /v1/chat/completions`
- `POST /v1/completions`
- Streaming and non-streaming responses
- Usage and timing fields where the active backend provides them
- Request queueing in the gateway
- Mid-generation cancellation in the specialized GLM engine; the experimental
  `.qwn` engine currently processes one submitted generation synchronously
- API-key authentication with `QWANTO_API_KEY`
- Configurable CORS origins
- Tool declaration/prompt rendering and tool-call parsing on the native gateway

Compatibility is intentionally not described as complete OpenAI API parity.
JSON-schema `response_format` and arbitrary structured-output enforcement are
not implemented; non-text `response_format` requests are rejected.

### GGUF and llama.cpp

For GGUF files, Qwanto starts a local `llama-server` and proxies its OpenAI API.
Before launch, the resource orchestrator reads selected GGUF metadata and builds
a launch plan.

The planner currently:

- Reads architecture, layer, embedding, head, KV-head, and context metadata
- Estimates model and KV-cache memory
- Reads NVIDIA free VRAM through `nvidia-smi` when available
- Selects `-ngl`, CPU threads, batch size, and micro-batch size
- Produces a proportional `--tensor-split` for multiple detected NVIDIA GPUs
- Uses `--mmap`, allowing the operating system to page file-backed weights

This is a heuristic launch planner, not a proof of optimal placement. For
AMD/Intel/Vulkan systems where VRAM is not discovered by `nvidia-smi`, Qwanto
delegates layer placement to llama.cpp with `-ngl 999`.

Supported acceleration controls are applied only when the installed
`llama-server --help` reports the corresponding flag:

| UI option | llama.cpp option | Default |
|-----------|------------------|---------|
| Flash Attention | `-fa` / `--flash-attn` | Enabled |
| KV cache type | `-ctk`, `-ctv` | `q4_0` |
| Speculative decoding | `-md <draft.gguf>` | Disabled until a draft path is supplied |
| Multi-GPU split | `-ts` | Automatic when multiple NVIDIA GPUs are detected |

Quantized KV cache reduces memory use but can affect output quality. Speculative
decoding speed depends on draft-model compatibility and acceptance rate; a
fixed 2x or 3x gain is not guaranteed.

#### Windows llama-server Download

If `llama-server` is missing on Windows, Qwanto attempts to download a release
archive from the latest llama.cpp GitHub release and extracts the executable and
its DLL dependencies into `c/`.

- NVIDIA detected: prefer a CUDA archive, then Vulkan
- Other or unknown GPU: prefer Vulkan
- Detection failure falls back to the general selection logic

The automatic release downloader currently searches Windows release assets.
On Linux and macOS, install/build `llama-server` separately and put it on
`PATH`.

### Native GLM and OLMoE Paths

The repository contains architecture-specific C runtimes and conversion tools.
They are not generic replacements for llama.cpp.

The GLM path includes:

- Expert streaming from disk
- RAM expert cache and tier planning
- Asynchronous I/O paths
- Quantized CPU kernels
- Optional CUDA and Metal code paths
- Persistent engine protocol and KV slots
- GLM-specific tokenizer, attention, MoE, and speculative/MTP logic

The OLMoE path uses its own model assumptions and converter. Qwen-MoE,
DeepSeek-MoE, and arbitrary MoE architectures are not automatically supported
by these native executables.

## Qwanto Native (`.qwn`)

`.qwn` is an experimental single-file container and independent decoder path.
It is separate from GGUF and does not use llama.cpp for CPU decoding.

### Container Layout

- Fixed 4 KiB header
- Up to 29 inline tensor descriptors
- Overflow descriptors and a sorted FNV-1a hash index for additional tensors
- Absolute tail-block offset in the final 8 bytes of the file
- Every tensor payload begins on a 4 KiB boundary
- Tensor storage is padded to a 64-byte boundary
- Supported container dtypes: F32, F16, BF16, Q4_0, and raw bytes
- Embedded `config.json` and `tokenizer.json` payloads when found beside the
  source safetensors

### Converter

The converter accepts one safetensors file or a directory of safetensors
shards. Matrix conversion is streamed one row at a time instead of loading the
entire checkpoint into RAM.

Q4_0 uses 32-value blocks containing an FP16 scale and 16 packed bytes. Matrix
rows with a K-tail are zero-padded inside their final quantization block.

```bash
# From the repository root
python c/coli pack ./model-directory ./model.qwn --quant q4_0

# Keep source F32/F16/BF16 tensors without matrix Q4_0 conversion
python c/coli pack ./model-directory ./model.qwn --quant none

# Inspect metadata, tensor shapes, offsets, and dtypes
python c/coli inspect ./model.qwn
```

### Native Decoder

The current decoder implements a dense Llama/Qwen-style graph:

- Byte-level BPE tokenizer using the tokenizer implementation in `c/tok.h`
- RMSNorm
- Split-half RoPE
- Causal attention with GQA/MQA
- Optional per-head Q/K RMSNorm used by Qwen3-style models
- FP16 key/value cache
- SwiGLU MLP and residual connections
- Tied or separate LM head
- Greedy decoding
- Temperature sampling with nucleus filtering over the retained top 256
  candidates
- Persistent stdin/stdout engine protocol used by the HTTP gateway

CPU Q4_0 inference uses per-token Q8 activation quantization, an AVX2 block-dot
path, a scalar K-tail fallback, and a persistent 64-byte-aligned scratch arena.
There is no `malloc` in the Q4_0 token matmul path.

The native runtime also includes:

- A tensor residency plan that prioritizes hot tensors for a GPU budget, then
  RAM, leaving the remainder file-backed through mmap
- OS page prefetch/drop helpers
- An optional CUDA Q4_0 resident-weight matmul entry point
- CPU fallback if the optional CUDA entry point is unavailable

Build and run:

```bash
make -C c qwnrun
python c/coli run --model ./model.qwn --ngen 128 "Hello"
python c/coli chat --model ./model.qwn
python c/coli web --model ./model.qwn
```

On this Windows workspace, `c/qwnrun.exe` was built with Clang as a CPU/AVX2
binary. Native CUDA requires rebuilding with the optional CUDA backend.

### `.qwn` Limitations

- The native graph is limited to dense Llama/Qwen-style tensor names and
  operations.
- MoE, MLA, multimodal encoders, cross-attention, and arbitrary custom model
  code are not mapped by the `.qwn` decoder.
- Tokenizer compatibility is limited to the byte-level BPE behavior implemented
  in `c/tok.h`; not every Hugging Face tokenizer pipeline is equivalent.
- Chat-template handling is not generalized for every model family. Raw prompt
  execution is the clearest validation path.
- RoPE scaling variants beyond the implemented default behavior are not
  guaranteed.
- Real-model quality and performance benchmarks have not yet been published.
- The CUDA `.qwn` kernel was not executed in the current environment because
  the CUDA compiler was unavailable.
- `.qwn` files are loadable by path, but the dashboard's automatic model scan
  currently discovers GGUF files and native model directories, not `.qwn`
  files.

For broad model compatibility, GGUF through llama.cpp remains the recommended
runtime.

## Web Dashboard

| Area | Implemented behavior |
|------|----------------------|
| Chat | SSE rendering, token speed, TTFT, stop generation, persistent conversations |
| Message actions | Copy assistant response; retry action restores the prior user prompt for editing/resending |
| Models | Discover GGUF/native directories, load, delete, download, and manage search paths |
| Downloads | Configurable parallel connections, speed limit, progress, pause, resume, cancel |
| Brain | Native MoE visualization when tier/expert data is available |
| Logs | Captured browser errors/warnings with copy-to-clipboard |
| Voice input | Browser Web Speech API when supported by the browser |
| Acceleration | Context size, Flash Attention, KV type, speculative draft path |

### Web Search and Attachments

The dashboard currently contains web-search and attachment UI controls, but
they are not complete retrieval or multimodal features:

- Web search only adds a search-query marker to the text prompt. Qwanto does
  not fetch search-engine results.
- Attached files are read by the browser UI, but only the attachment name and
  MIME type are added to the prompt. File contents are not sent to the model.
- Images are not passed as vision-model inputs.
- PDF/document parsing is not implemented.

These controls should be treated as UI scaffolding, not as working web
retrieval, RAG, file analysis, or vision support.

### Resource Controls

The dashboard exposes CPU, RAM, VRAM, and disk percentages. At present:

- CPU percentage affects the thread count used on a subsequent llama.cpp model
  load/reload.
- RAM, VRAM, and disk percentages are stored and returned by the API but are not
  hard runtime limits for all backends.
- The native GLM `--ram`, `--vram`, and `--auto-tier` CLI planning path is
  separate from these dashboard percentages.

Do not treat the dashboard sliders as operating-system resource enforcement.

## Quick Start

### Requirements

- Python 3.10 or newer
- Node.js/npm only when rebuilding the dashboard
- A C compiler and Make for native-engine builds
- `llama-server` on `PATH` for GGUF on Linux/macOS
- Sufficient storage for the selected model

### GGUF on Windows

```powershell
python c\coli web --model "D:\models\model.gguf"
```

Qwanto opens `http://127.0.0.1:8000/` after the API becomes healthy. The first
GGUF launch may download a Windows llama.cpp bundle.

### GGUF on Linux/macOS

Install or build `llama-server`, ensure it is on `PATH`, then run:

```bash
python3 c/coli web --model /path/to/model.gguf
```

### Native GLM Setup

```bash
cd c
./setup.sh
./coli doctor --model /path/to/converted-model
./coli web --model /path/to/converted-model --ram 20 --auto-tier
```

`setup.sh` is focused on the GLM engine. Build the experimental `.qwn` runtime
separately with `make qwnrun`.

### Rebuild the Dashboard

```bash
cd web
npm install
npm run build
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/chat/completions` | Chat completion, streaming or non-streaming |
| POST | `/v1/completions` | Text completion, streaming or non-streaming |
| GET | `/v1/models` | Active backend model list |
| GET | `/health` | Gateway/backend health and available metrics |
| GET | `/v1/qwanto/config` | Active model, backend, context, capabilities, resources |
| GET | `/v1/qwanto/models` | Discovered GGUF files and native model directories |
| GET | `/v1/qwanto/paths` | Saved custom search paths |
| POST | `/v1/qwanto/paths` | Add or remove a custom search path |
| POST | `/v1/qwanto/load` | Load/reload a local model and backend options |
| POST | `/v1/qwanto/resources` | Set resource percentage values; empty body returns current values |
| POST | `/v1/qwanto/download` | Start a direct model download |
| GET | `/v1/qwanto/download/status` | Download state and progress |
| POST | `/v1/qwanto/download/config` | Set connection count and speed limit |
| POST | `/v1/qwanto/download/pause` | Pause the active download |
| POST | `/v1/qwanto/download/resume` | Resume the active download |
| POST | `/v1/qwanto/download/cancel` | Cancel the active download |
| POST | `/v1/qwanto/delete` | Delete a selected local model path |

Example model load body:

```json
{
  "model_path": "D:\\models\\model.gguf",
  "backend": "auto",
  "ctx_size": 16384,
  "flash_attention": true,
  "kv_cache_quant": "q4_0",
  "speculative_decoding": false,
  "draft_model_path": ""
}
```

## Configuration

| Environment variable | CLI option | Meaning |
|----------------------|------------|---------|
| `QWANTO_MODEL` | `--model` | Default model path/name |
| `QWANTO_API_KEY` | `--api-key` | Bearer-token protection for the HTTP API |
| `QWANTO_MODEL_ID` | `--model-id` | Model ID exposed by the gateway |
| `QWANTO_MODEL_PATHS` | None | Extra model search paths separated by semicolons |
| `QWANTO_MAX_QUEUE` | `--max-queue` | Maximum queued generation requests |
| `QWANTO_QUEUE_TIMEOUT` | `--queue-timeout` | Queue timeout in seconds |
| `QWANTO_KV_SLOTS` | `--kv-slots` | Native engine KV/session slots, 1 to 16 |
| `QWANTO_POLICY` | `--policy` | Native resource policy |
| `RAM_GB` | `--ram` | Native-engine RAM budget |

HTTP bind settings are CLI options:

```text
--host 127.0.0.1
--port 8000
```

When binding beyond localhost, set `QWANTO_API_KEY`. The server emits a warning
if it is exposed without an API key.

Session files written in the project root include:

- `.qwanto_settings.json`: active local model, backend, context size, and
  llama.cpp acceleration choices
- `.qwanto_model_paths.json`: custom discovery paths
- Browser localStorage: conversations, selected view, and UI preferences

KV/session slots are implemented by the specialized GLM engine. The current
`.qwn` persistent process resets its decoder for each submitted request and does
not provide independent persistent conversation slots.

## Build and Test

### Python and Integration Tests

```bash
python -m pytest c/tests/ -q
```

Some tests compile small native executables with Clang. Hardware-dependent tests
are skipped when their requirements are unavailable.

### Native C Tests

```bash
make -C c test-c
```

### Dashboard

```bash
cd web
npm test
npm run build
```

### Optional CUDA

Linux direct-link build:

```bash
make -C c CUDA=1 qwnrun
```

Windows uses the optional `coli_cuda.dll` loader path and requires a compatible
CUDA Toolkit plus MSVC environment. See `c/build_cuda.bat` and the comments in
`c/Makefile`. If the DLL lacks the newer `.qwn` CUDA symbol, existing CUDA paths
remain available and `.qwn` matmul falls back to CPU.

## Project Structure

```text
qwanto/
|-- c/
|   |-- coli                    CLI entry point
|   |-- openai_server.py        HTTP gateway, streaming, downloads, model API
|   |-- backends.py             llama.cpp, Ollama, and compatible API adapters
|   |-- orchestrator.py         GGUF metadata reader and launch heuristic
|   |-- resource_plan.py        Native tier/resource planning
|   |-- glm.c                   Specialized GLM native engine
|   |-- olmoe.c                 Specialized OLMoE runtime
|   |-- qwanto_native.c/.h      .qwn mmap container and tensor index
|   |-- qwanto_kernels.c/.h     Q4xQ8 CPU kernels and scratch arena
|   |-- qwanto_decode.c/.h      Experimental dense native decoder
|   |-- qwnrun.c                Standalone/persistent .qwn executable
|   |-- backend_cuda.cu/.h      Optional CUDA backend
|   `-- tools/qwn_convert.py    Streaming safetensors-to-.qwn converter
|-- web/
|   |-- src/App.tsx             Dashboard and chat UI
|   |-- src/Brain.tsx           Native MoE visualization
|   `-- src/lib/api.ts          Browser API client
|-- run_server.bat              Example Windows native-GLM launcher
|-- run_server.sh               Example Linux/macOS native-GLM launcher
`-- LICENSE                     Apache License 2.0
```

The launcher scripts contain example model paths and should be edited or
overridden before use. They are not generic zero-configuration launchers.

## Troubleshooting

### llama-server is missing on Linux/macOS

Install/build llama.cpp and ensure `llama-server` is on `PATH`. The built-in
release downloader currently targets Windows archives.

### llama-server DLL is missing on Windows

Remove the partially extracted `llama-server.exe` and related llama.cpp DLLs
from `c/`, then restart Qwanto so the complete archive is extracted again.

### Windows blocks llama-server

`WinError 4551` indicates AppLocker/WDAC policy. Run an approved
`llama-server` manually on Qwanto's expected local port (`8080` by default), or
ask the system administrator to allow the executable.

### `.qwn` does not load

- Confirm `config.json` and `tokenizer.json` were present beside the source
  safetensors before packing.
- Run `python c/coli inspect model.qwn`.
- Confirm the model uses the supported dense tensor names.
- Rebuild `qwnrun` after source changes.

### Dashboard search or attachments do not provide external context

This is expected in the current version. Those controls do not yet implement
web retrieval, document parsing, RAG, or vision input.

### Resource slider does not cap RAM/VRAM

Only the CPU percentage currently changes llama.cpp launch threads. Use native
CLI resource flags and backend-specific settings for actual planning.

## License

Apache License 2.0. See [LICENSE](LICENSE).
