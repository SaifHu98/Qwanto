# Architecture Map: FeatherCore Current State

This document captures the current engineering layout and execution pathways of **FeatherCore** (formerly Colibrì) as of the baseline audit.

---

## 1. Execution Pathways

### A. Model Loading
- **Entrypoint:** `coli` CLI -> `openai_server.py` or direct `./glm` binary execution.
- **Metadata Parsing:** Reads model structure directly from `config.json` to configure the `Cfg` structure.
- **Tensors Allocation:** Tensors are defined by the `QT` structure, supporting:
  - Format `0` (FP32)
  - Format `1` (INT8)
  - Format `2` (INT4 per-row)
  - Format `3` (INT2 packed)
  - Format `4` (INT4 grouped)
- **Shard Mapping:** Tensors mapped via `compat.h` using standard POSIX mapping or `compat_mmap` on Windows.

### B. Expert Routing & Cache Lookup
- **Routing Algorithm:** DeepSeek-V3 style sigmoid routing utilizing `routed_scaling_factor` and a shared expert pathway.
- **Cache Sizing:** Auto-computed at startup (`cap_for_ram`) based on physical memory budgets to keep resident memory inside bounds.
- **LRU Cache Mechanism:** Evaluates active experts using `ESlot` structures containing quantized weights and scales. When cache misses occur, pages are loaded from storage into cache slots.
- **Pre-pinning:** Historically used experts from `.coli_usage` are loaded directly to RAM to act as a hot cache.

### C. Storage I/O (Disk Reads)
- **POSIX compat layer:** Defined in `compat.h`.
- **Windows Path:** POSIX `pread` mapped to synchronous `ReadFile` using `OVERLAPPED` offsets.
- **Standby Page Cache:** `compat_fadvise` acts as a hint to preload pages using a dedicated `PILOT` readahead background thread.

### D. Prefill & Decode
- **Prefill Path:** Initiated via `forward_all()` (for teacher forcing/batch input evaluation).
- **Decode Path:** Initiated via `generate()`, which runs single-token output loops.
- **Speculative decoding (MTP):** Supported via `DRAFT=n` (GLM's own multi-token prediction head at layer 78). Prefill evaluates draft tokens in a batched forward.
- **Grammar Drafting:** Supported via `GRAMMAR` environment flags, enforcing syntax-level drafts during prefill.

### E. KV-Cache & Mux Server
- **KV-Cache Persistence:** Compression and serialization of MLA KV arrays to a local `.coli_kv` file at the end of each session, permitting hot reloading.
- **Gateway HTTP Server:** `openai_server.py` hosts a `ThreadingHTTPServer` that pipes JSON SSE (Server-Sent Events) chunks.
- **Tool-Call Streaming:** A state-machine parser in `openai_server.py` intercepts `BOX_START` (`<tool_call>`) / `BOX_END` (`</tool_call>`) structures and streams them directly into OpenAI delta chunks in real time.

---

## 2. Shared Libraries & GPU Offloading
- **CUDA Dynamic Loading:** Native Windows/Linux binary uses dynamic loader (`backend_loader.c`) to link and invoke matrices on `coli_cuda.dll` dynamically.
- **AVX SIMD Kernels:** Matrix-vector dot products are accelerated using custom AVX2 instruction sets (`maddubs` for INT8).
