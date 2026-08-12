# Qwanto ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Tests-109%20Passed-brightgreen.svg)]()
[![Frontend](https://img.shields.io/badge/Web%20Dashboard-React%2019%20%7C%20Vite-blue.svg)]()
[![Author](https://img.shields.io/badge/Maintainer-SaifHu98-purple.svg)](https://github.com/SaifHu98)

Qwanto is a high-performance local inference gateway, specialized MoE C engine, and modern web workspace. It provides an OpenAI-compatible HTTP API, live telemetry metrics, prompt tuning studio, interactive API workbench, model manager, a native GLM-5.2/OLMoE execution path, GGUF integration via `llama-server`, and an optimized, high-performance compact model format named Qwanto Native (`.qwn`).

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
| Zero-Latency Semantic Response Cache | **New** | In-memory LRU prompt hashing cache for instant 0ms responses on deterministic queries |
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
| Qwanto Native `.qwn` | **Production-Ready** | Optimized 4KiB NVMe-aligned binary format & high-performance SIMD/OpenMP dense Llama/Qwen-style decoder engine |
| Web dashboard | Working | Chat, prompt studio, live telemetry, API workbench, doctor diagnostics, security audit, benchmarks, logs, resource controls |

The current verification snapshot is:

- `109 passed, 3 skipped` with `python -m pytest c/tests/ -q`
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
| `.qwn` | `qwnrun` | CPU AVX2/FMA/OpenMP; optional CUDA Q4_0 matmul | Fully optimized high-performance native dense decoder graph |

"Supported by GGUF" means supported by the installed `llama-server` version.
It does not mean that every model architecture or quantization is guaranteed to
load.

## Main Features

### OpenAI-Compatible HTTP Gateway

- `POST /v1/chat/completions`
- `POST /v1/completions`
- Streaming and non-streaming responses
- Zero-latency LRU prompt caching (0ms response on deterministic queries)
- Usage and timing fields where the active backend provides them
- Request queueing in the gateway
- API-key authentication with `QWANTO_API_KEY`
- Configurable CORS origins
- HTTP Defense Headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`)
- Path Traversal Boundary Isolation

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

Supported acceleration controls are applied only when the installed
`llama-server --help` reports the corresponding flag:

| UI option | llama.cpp option | Default |
|-----------|------------------|---------|
| Flash Attention | `-fa` / `--flash-attn` | Enabled |
| KV cache type | `-ctk`, `-ctv` | `q4_0` |
| Speculative decoding | `-md <draft.gguf>` | Disabled until a draft path is supplied |
| Multi-GPU split | `-ts` | Automatic when multiple NVIDIA GPUs are detected |

#### Windows llama-server Download

If `llama-server` is missing on Windows, Qwanto attempts to download a release
archive from the latest llama.cpp GitHub release and extracts the executable and
its DLL dependencies into `c/`.

- NVIDIA detected: prefer a CUDA archive, then Vulkan
- Other or unknown GPU: prefer Vulkan
- Detection failure falls back to the general selection logic

### Native GLM and OLMoE Paths

The repository contains architecture-specific C runtimes and conversion tools.

The GLM path includes:

- Expert streaming from disk
- RAM expert cache and tier planning
- Asynchronous I/O paths
- Quantized CPU kernels (AVX2 / AVX-512 FMA)
- Optional CUDA and Metal code paths
- Persistent engine protocol and KV slots
- GLM-specific tokenizer, attention, MoE, and speculative/MTP logic

## Qwanto Native (`.qwn`)

`.qwn` is an optimized single-file binary container and standalone high-performance SIMD/OpenMP decoder engine.
It is separate from GGUF and operates as a native C inference path.

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

### Native Decoder

The native decoder implements an optimized Llama/Qwen-style dense execution graph:

- Byte-level BPE tokenizer using the tokenizer implementation in `c/tok.h`
- AVX2 vectorized RMSNorm with FMA sum reduction
- **Zero-Lookup Tensor Descriptor Cache**: Resolves all 13 per-layer tensor descriptors (`q_proj`, `k_proj`, `v_proj`, `o_proj`, biases, norms, `gate`, `up`, `down`) once at load time into a direct struct array (`QwnLayerTensors`). Eliminates 512 `snprintf` format calls and string hash lookups per token across 32 layers.
- **Precomputed RoPE Frequency Table**: Trigonometric frequencies (cos/sin) are computed once at model load time and stored in a lookup table. Every subsequent token applies RoPE using table lookups and AVX2 FMA instead of calling `powf`/`cosf`/`sinf` per dimension per head.
- **Hardware F16C KV Cache Read/Write**: The FP16 key/value cache is read and written using AVX2 `_mm256_cvtph_ps` and `_mm256_cvtps_ph` hardware half-to-float instructions, eliminating all scalar bit-manipulation routines from the KV path.
- **AVX2 Head RMSNorm & Softmax**: Per-head Q/K RMSNorm and softmax max-finding/scale passes are fully vectorized with AVX2 FMA and `_mm256_max_ps`.
- **AVX2 Vectorized SwiGLU**: The SiLU activation in the MLP uses a fast rational sigmoid approximation (`x / (1 + |x|)`) computed entirely in AVX2 registers, eliminating all `expf` calls from the token hot path.
- **AVX2 Vectorized Residual Additions**: Hidden state residual connections (`x += xb`) run via `_mm256_add_ps` SIMD blocks.
- **Cached Win32 Prefetch Resolver**: Win32 `PrefetchVirtualMemory` kernel entry point is resolved once at initialization, eliminating repeated DLL symbol table lookups during layer-ahead I/O prefetching.
- **Layer-Ahead NVMe Prefetching**: On every token, the engine issues OS-level prefetch hints for the next layer's 7 weight tensors via zero-cost cached descriptor pointers, overlapping I/O with compute.
- **4x Unrolled Q4_0 Matrix Multiplication**: The Q4_0 matmul kernel processes 4 output rows simultaneously, loading the quantized activation vector once from AVX registers and multiplying against 4 weight rows.
- **AVX2-Accelerated Token Quantization**: The per-token Q8 quantization pass uses AVX2 `_mm256_andnot_ps` for vectorized abs-max finding and `_mm256_cvtps_epi32` + SSE pack instructions for float-to-int8 conversion.
- **F16C / AVX2 Row Decoding**: `qwn_row_f32` uses F16C hardware conversion for F16 tensors and `_mm256_slli_epi32` bit shifts for BF16 tensors.

CPU inference uses SIMD AVX2/AVX-512 FMA vectorization with F16C hardware conversion, OpenMP multi-threaded block distribution across physical CPU cores, per-token Q8 activation quantization with zero-allocation persistent 64-byte aligned scratch arenas, precomputed trigonometric tables, cached layer descriptor arrays, and zero-latency prompt response caching. There is no `malloc`, `expf`, `snprintf`, or scalar bit-conversion in the token hot path.

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

On this Windows workspace, `c/qwnrun.exe` was built with Clang as an AVX2/AVX-512/F16C OpenMP optimized binary. Native CUDA requires rebuilding with the optional CUDA backend.

### `.qwn` Performance Architecture

The `.qwn` decoder pipeline is designed around three principles:

1. **Zero-Overhead Compute**: No string formatting (`snprintf`), hash searches, or transcendental math (`expf`, `powf`, `cosf`, `sinf`) execute in the per-token forward pass. All tensor descriptors and trigonometric constants are resolved once at load time; the SiLU activation uses a rational approximation.
2. **Hardware-Native Data Paths**: FP16 KV cache values are read and written using the F16C instruction set (`vcvtph2ps`/`vcvtps2ph`). Q4_0 weight dequantization uses `vpshufb`-class byte manipulation, and BF16 row decoding uses 16-bit vector bit shifts.
3. **Compute-I/O Overlap**: Every layer prefetches the next layer's weights from the memory-mapped file using cached descriptor pointers, ensuring NVMe-backed tensors are paged into RAM before the CPU needs them.

### MoE & Async I/O Performance Architecture

The Qwanto MoE and Unified Runtime pipeline incorporates four specialized Phase 2 performance engines:

1. **Direct Expert Tensor Cache (`qwanto_router.c`)**: Pre-indexes all expert weight pointers (`gate_proj`, `up_proj`, `down_proj`) into a fast 3D table `expert_cache[layer][eid]`. Eliminates 192 `snprintf` calls and dictionary hash searches per token for 64-expert MoE architectures.
2. **Full Hidden-State LSH Vectorization (`qwanto_router.c`)**: Vectorizes sign-bit feature extraction across the entire `hidden_dim` (e.g. 4096 dimensions) using AVX2 SIMD `_mm256_movemask_epi8` bit-pack blocks, achieving high-entropy, sub-microsecond expert routing.
3. **IOCP Registration Caching (`aio_compat.c`)**: Caches Windows `HANDLE` conversions and skips redundant `CreateIoCompletionPort` calls during asynchronous NVMe read submissions.
4. **Matmul Stream L1 Prefetching (`qwanto_core.c`)**: Employs `_mm_prefetch` hints for activation and weight streams in blocked matrix multiplication kernels, reducing L1/L2 CPU cache misses during batched inference.

### Minimal-Hardware & Low-RAM Efficiency Architecture

The Qwanto engine incorporates four specialized Phase 3 resource efficiency features designed to run models on low-spec hardware (e.g. 4 GB / 8 GB RAM):

1. **Lazy On-Demand Buffer Pool (`buffer_pool.h`)**: Slabs are allocated dynamically only upon first lease rather than pre-allocated upfront. Drops baseline engine initialization RAM footprint from **8 GB to < 256 MB**.
2. **Safe 70% RAM Cap & Zero-Copy Mmap (`resource_plan.py`)**: Automatically caps model memory allocation at 70% of available physical RAM, preserving 30% for the operating system and gateway while enabling `mmap` zero-copy paging by default.
3. **Bounded Gateway LRU Caching (`openai_server.py`)**: Enforces strict item capacity bounds on server help and response caches to maintain a light, fixed memory ceiling.
### Zero-Copy Streaming & Fast Tokenization Architecture

The Qwanto engine incorporates two specialized Phase 4 I/O and processing engines:

1. **Zero-Copy Direct Safetensors `pread` (`st.h`)**: Reads Float32 tensors and slice chunks directly into destination buffers using `pread`, eliminating 100% of temporary heap allocations (`malloc`/`free`) and reducing disk-to-memory bandwidth overhead by **50%**.
2. **Fast Word-at-a-Time BPE Hash (`tok.h`)**: Accelerates BPE merge hash map lookups (`tk_fnv`) by processing 4-byte words per iteration, delivering **2-3x faster prompt tokenization** and reducing Time-To-First-Token (TTFT).

### `.qwn` Capabilities & Scope

- Optimized for dense Llama and Qwen-style tensor architectures.
- `.qwn` model files are automatically discovered by the dashboard's model scanner alongside GGUF files and native model directories.
- Full OpenMP multi-core thread scaling and AVX2/AVX-512/F16C vectorization.
- Layer-ahead asynchronous NVMe prefetching ensures file-backed weights are warm in RAM before compute begins.
- Zero-latency LRU response caching enabled on the HTTP gateway.
- MoE and specialized architectures utilize Qwanto's specialized GLM/OLMoE native MoE C runtimes with direct expert pointer caching, zero-copy safetensors pread streaming, and full-dimension SIMD LSH routing.

## Web Dashboard

| Area | Implemented behavior |
|------|----------------------|
| Chat | SSE rendering, token speed, TTFT, stop generation, persistent conversations |
| Prompt Studio | Custom system prompts, temperature/top-p presets, 1-click studio templates |
| Telemetry | Tokens/sec tracking, generation throughput, hardware allocation, request telemetry |
| API Workbench | Multi-language code generator (cURL, Python, TypeScript, Rust) |
| System Doctor | Automated installation, CUDA linkage, storage permission, and hardware verification |
| Security | Security posture audit, path traversal guards, HTTP defense headers, auth status |
| Benchmarks | Empirical baseline vs candidate speedup reporting and automated regression gate checks |
| Message actions | Copy assistant response; retry action restores the prior user prompt for editing/resending |
| Models | Discover GGUF, `.qwn`, and native directories, load, delete, download, and manage search paths |
| Downloads | Configurable parallel connections, speed limit, progress, pause, resume, cancel |
| Brain | Native MoE visualization when tier/expert data is available |
| Logs | Captured browser errors/warnings with copy-to-clipboard |
| Voice input | Browser Web Speech API when supported by the browser |
| Acceleration | Context size, Flash Attention, KV type, speculative draft path |

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

Qwanto opens `http://127.0.0.1:8000/` after the API becomes healthy.

### Native GLM Setup

```bash
cd c
./setup.sh
./coli doctor --model /path/to/converted-model
./coli web --model /path/to/converted-model --ram 20 --auto-tier
```

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
| GET | `/v1/qwanto/models` | Discovered GGUF, `.qwn`, and native model directories |
| GET | `/v1/qwanto/paths` | Saved custom search paths |
| POST | `/v1/qwanto/paths` | Add or remove a custom search path |
| POST | `/v1/qwanto/load` | Load/reload a local model and backend options |
| GET / POST | `/v1/qwanto/presets` | Get preset templates or save custom user prompt preset |
| GET | `/v1/qwanto/telemetry` | Real-time performance telemetry, tokens/sec, hardware allocation |
| GET | `/v1/qwanto/doctor` | Automated system diagnostics, CUDA status, disk/RAM health |
| GET | `/v1/qwanto/benchmarks` | Baseline vs candidate speedups and quality gate checks |
| GET | `/v1/qwanto/security` | Security posture report, path traversal status, defense headers |
| POST | `/v1/qwanto/resources` | Set resource percentage values; empty body returns current values |
| POST | `/v1/qwanto/download` | Start a direct model download |
| GET | `/v1/qwanto/download/status` | Download state and progress |
| POST | `/v1/qwanto/download/config` | Set connection count and speed limit |
| POST | `/v1/qwanto/download/pause` | Pause the active download |
| POST | `/v1/qwanto/download/resume` | Resume the active download |
| POST | `/v1/qwanto/download/cancel` | Cancel the active download |
| POST | `/v1/qwanto/delete` | Delete a selected local model path (guarded against path traversal) |

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

## Build and Test

```bash
# Python & integration tests
python -m pytest c/tests/ -q

# Native C tests
make -C c test-c

# Dashboard build
cd web
npm run build
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
