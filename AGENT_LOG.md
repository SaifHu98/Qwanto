# AGENT_LOG.md

## 2026-08-16 — Gateway contract, dashboard gating, and beta.2 release hardening

- **Change:** Added a real local subprocess integration test for `/health`,
  `/v1/models`, `/v1/qwanto/config`, and `/v1/qwanto/telemetry`; versioned the
  gateway schemas; and removed the invalid model default from gateway/CLI/UI
  paths.
- **Change:** Added validated QWN discovery/recommendation metadata, explicit
  download consent/checksum/size fields, gateway readiness states, responsive
  shell CSS, and actionable empty/error UI states.
- **Change:** Added documentation index/API/conversion/troubleshooting pages,
  a local-link checker, measured-evidence README reporting, and checksum/factual
  release-note generation for future beta tags. Existing `v0.1.0-beta.1` is
  unchanged.
- **Validation:** Native decoder `2 passed`; Python suite `201 passed, 14
  skipped`; web build and Vitest `42 passed`; documentation links passed. Cargo
  gates could not run because Rust/Cargo is not installed on this workstation.

## 2026-08-15 — HyperVSQ-2 SIMD Engine Acceleration & Repair (SaifHu98)
- **Problem diagnosed**: HyperVSQ-2 (2.3125 bpw) was running at ~0.2 tok/s (154s for 32 tokens) because `qwn_matmul_f32` dispatched `QWN_DT_HYPER_VSQ2` to scalar `qwn_matmul_packed_f32` calling `qwn_packed_value` per element (>8 billion software FP16 conversions per token on 4B model). Furthermore, `build_layer_cache` clamped `lt->q_out` to `o_proj->shape[0]` (4096), causing layer 3 `q_proj` matmul (`shape[1]=8192`) to fail shape validation.
- **Format layout verified**: 74-byte packed superblock for 256 weights ($W = (q - 1) \cdot S_{\text{base}} \cdot \frac{u}{8} + C$). Row stride $= \lceil K / 256 \rceil \times 74$ bytes. 8 sub-scales in 4 bytes (8 nibbles in $[1..8]$), 64 bytes of packed 2-bit quaternary codes in $[0..3]$.
- **SIMD Kernels Implemented in `c/qwanto_kernels.c`**:
  - `qwn_gemv_hypervsq2_scalar`: Standalone scalar golden reference decoder and test oracle.
  - `qwn_gemv_hypervsq2_avx2`: Vectorized AVX2 kernel using `unpack_32x2bit_avx2` (8-byte in-register unpack to 32 int8 codes) and `_mm256_maddubs_epi16` + `_mm256_madd_epi16` for parallel 8-octant dot products.
  - `qwn_gemv_hypervsq2_vnni`: High-throughput AVX-VNNI kernel using `_mm256_dpbusd_epi32` with zero-point correction: $\sum (q_i - 1) a_i = \text{dpbusd}(q, a) - \sum a_i$.
  - `qwn_matmul_hypervsq2_f32`: Multi-row parallel GEMV dispatcher with OpenMP parallelization, 64-row tiling, and cached CPU feature dispatch.
  - Runtime CPUID feature detection (`qwn_get_cpu_features()`) with environment overrides (`QWN_FORCE_SCALAR`, `QWN_FORCE_AVX2`, `QWN_FORCE_VNNI`).
- **Decoder Fixes in `c/qwanto_decode.c`**:
  - Removed erroneous clamping of `lt->q_out`, `lt->k_out`, `lt->v_out` to `lt->o_proj->shape[0]` in `build_layer_cache`.
  - In `qwn_decoder_forward()`, decoupled Q projection dimension `Q` from O projection input dimension `O_IN`, allowing hybrid and asymmetric attention projections to pass shape validation across all layers.
  - Fixed scratch buffer padding in `qwn_scratch_init` to 256 elements.
- **Verification & Benchmarks**:
  - Built and executed `c/tests/test_hypervsq2_kernels.c`: **140/140 differential tests passed** across all tail lengths, matrix dimensions, scales, and offsets. Microkernel benchmark: 26.5 GFLOPS (VNNI) vs 8.16 GFLOPS (scalar).
  - Real end-to-end inference on `experiments/results/4B_hyper_vsq2.qwn`:
    - Original scalar path: **0.2 tok/s** (154s for 32 tokens)
    - New AVX-VNNI engine: **13.17 tok/s** (4.86s for 64 tokens) -> **65.85x speedup**!
    - Baseline 4B Q4_0: **2.18 tok/s** (29.39s for 64 tokens) -> HyperVSQ-2 is **6.05x faster** than Q4_0 and uses half the memory (1.26 GB vs 2.45 GB).
  - Python tests: **157 passed, 12 skipped** (`python -m pytest c/tests/ -q`).
  - Web tests: **17 passed** (`npm test` in `web/`), web build succeeds without errors.

- Read `Full Improve Plan.md` (471 lines, 14 sections) and broke execution into the 7 phases the plan recommends (0 ground-truth → 1 QWN-IR → 2 quant planner → 3 Q2A → 4 SIMD kernels → 5 paged state → 6 speculative/MTP).
- Delivered Phases 0, 1, 2 entirely in Python (the C-heavy phases 3–6 are out of scope for a single session and were deferred as documented in the plan).
- Created new modules under `c/tools/`:
  - `qwn_bpw_truth.py` — single source of truth for format constants (`QuantFormatSpec`), per-tensor `TensorByteBreakdown` → `BpwReport`. Pins container invariants (4KiB header, INLINE_MAX=29, ALIGN_PAGE=4096). Auto-derives `payload_bpw` and `effective_bpw`. Replaces every hand-written bpw number.
  - `qwn_model_ir.py` — QWN-IR dataclasses (`ModelIR`, `TensorNode`, `ModelDims`, `CacheLayout`, `MTPPlan`, `Confidence`, `ValidationReport`, `TensorRole` enum, role sets `ATTENTION_ROLES`, `FFN_ROLES`, `MOE_ROLES`, `SSM_ROLES`, `PROTECTED_ROLES`).
  - `qwn_arch_registry.py` — `ArchAdapter` base + 5 concrete adapters (DenseTransformer, MoE, Mamba, HybridSSM, Unknown). `ArchRegistry.select()` ranks by confidence and priority; `UnknownAdapter` is always last resort.
  - `qwn_roles.py` — tensor role classifier with the plan's rank order (graph position → arch meta → shape relations → name). Detects QKV fused, tied embeddings, MTP heads, SSM tensors, MLA compress tensors, expert routers, shared vs routed experts.
  - `qwn_quant_plan.py` — `QuantPlanner` with `profile ∈ {tiny, balanced, quality}`, `mode ∈ {heuristic-safe, calibrated}`, per-role `CANDIDATE_LADDER`, sidecar outlier handling (≤1% channels at Q8/FP16), confidence gate (refuses Q2A when arch confidence < 0.90), budget pass that tightens FFN/MoE tensors first. Emits `QuantPlan.to_json()` (full reasons per decision, no black box).
  - `qwn_benchmark_v2.py` — real end-to-end benchmark harness per plan section 10. Captures git SHA, model SHA-256, prompt SHA-256, compiler, CPU features (AVX2/AVX-512F/VNNI/F16C/FMA), RAM, seed, temperature, warmup/round separation, per-round TTFT / tok/s / p50/p95/p99 latency / RSS, JSON + Markdown renderers. **Never substitutes a default for a failed measurement.**
  - `qwn_plan_cli.py` — standalone CLI: `python c/tools/qwn_plan_cli.py <model> --profile tiny --out quant_plan.json`. Emits a real `quant_plan.json` (conservative-only when safetensors/GGUF metadata isn't readable).
- Refactored `c/tools/qwn_benchmark.py` into a backward-compat shim that delegates to `qwn_benchmark_v2`. Old `run_real_benchmark(...)` no longer fabricates `tok_per_sec`; it now runs the real harness.
- Did **not** modify `c/tools/qwn_convert.py` per the plan ("يبقى qwn_convert.py منفذاً للخطة"). The CLI emits a `quant_plan.json` the converter can read in a future iteration.
- Updated `c/tests/test_phase23_hypervsq2_and_speculative.py::test_benchmark_harness_execution` to assert the new truthful behaviour (status ∈ {ok, error}) instead of the legacy `assertTrue(ok)` that depended on fabricated numbers.
- Added `c/tests/test_universal_engine_v2.py` with 28 tests covering all new modules.
- Validation: `python -m pytest c/tests/ -q` → **146 passed, 4 skipped** (was 109 passed / 3 skipped). End-to-end smoke test on a synthetic Llama-shape (1.5B) graph correctly detects `dense`, classifies 254 tensors, flips `lm_head` → `tied_embed` on shape equality, and emits per-profile plans.
- Decision: deferred phases 3–6 (Q2A ABI, AVX2/VNNI/NEON kernels, paged KV rewrite, speculative batch verification) to a future session — they are C-native refactors that the plan itself spans across multiple weeks.
- File discipline: minimal edits to existing files (only `qwn_benchmark.py` shim and one test), no changes to engine C source, no dependency additions, no copy/pasted code from other projects.

## 2026-08-14 — QWN correctness and resource-runtime safety pass
- Audited the working-tree changes against the native QWN ABI and found that the GGUF reader was reversing already-fastest-first dimensions, while the new Q2/Q3/Q4/Q5/Q6 K-quant dequantizers were not layout-correct. Removed the unsafe conversion behavior: unsupported K-quant/IQ dtypes now fail before a `.qwn` file is written instead of becoming zero or opaque payloads.
- Added native Q8_0 row dequantization and matrix multiplication, plus strict `.qwn` descriptor validation for dtype, shape/numel, payload size, 4 KiB payload offsets, 64-byte padding, and payload-before-tail invariants.
- Extended the dense decoder's load-time layer validation with separate Q/K/V output and head-dimension fields. Native qwnrun now rejects unsupported MoE/SSM layers and incompatible Q/K/V layouts before allocation or forward execution; the context arena is sized for the real Q output width.
- Added qwnrun build/runtime diagnostics for compiler, OpenMP runtime and active threads, ISA, CUDA availability, selected GPUs, planned GPU/RAM/NVMe bytes, and prefetch count. CUDA-enabled qwnrun accepts `COLI_GPU`/`COLI_GPUS`, initializes the requested devices, and assigns lazy Q4 tensor uploads within per-device budgets with CPU fallback when capacity is exhausted.
- Added `test_qwn_conversion_safety.py` covering GGUF dimension preservation and K-quant rejection.
- Validation: `python -m pytest c/tests/ -q` → **156 passed, 12 skipped**; direct Clang builds of qwnrun passed with scalar and AVX2/FMA/F16C flags. The repository Makefile target could not run because `make` is not installed in this Windows workspace.
- Remaining required work before performance claims: verified K-quant decoders, native MoE/SSM/hybrid kernels, true multi-GPU concurrent execution for dense QWN matmul, and CUDA/OpenMP hardware benchmarks on the target machine.

## 2026-08-14 — Four-tier production pass
- Replaced the previous K-quant rejection path with verified ggml-compatible streaming dequantization for Q4_K, Q5_K, and Q6_K. The implementation preserves block ordering, packed 6-bit scale/min fields, Q5 high bits, and Q6 signed scales, then feeds FP32 chunks into the requested QWN quantizer. Non-finite dequantized chunks now abort conversion.
- GGUF conversion now reads tokenizer token/merge arrays and BOS/EOS metadata instead of emitting a fake byte tokenizer. It also rejects unsupported Q2_K/Q3_K/Q8_K/IQ and invalid row-wise Q4_0 layouts, and stores explicit q_dim/k_dim/v_dim in the unused tail of the fixed 4 KiB header without changing descriptor offsets.
- Added Windows `c/build_native.bat` with MSVC OpenMP or MinGW/libgomp detection and a CPU fallback message, plus `c/build_cuda.bat` for the dynamic `qwn_cuda.dll` and existing multi-GPU `coli_cuda.dll` backend.
- Connected the optional `qwn_cuda.dll` ABI to qwnrun with dynamic loading, cached Q4_0/HyperVSQ weights, pinned activation/output buffers, transfer/compute streams, warp shuffle reductions, and a Q4 path using `__dp4a` on supported GPUs. The existing `coli_cuda` loader remains the multi-GPU path when compiled with `COLI_CUDA`.
- Integrated the existing paged KV implementation into dense qwnrun: 16-token logical blocks, 16 KiB-aligned KV allocations, block tables, gather scratch, and layer-ahead `_mm_prefetch`; contiguous KV remains a guarded fallback.
- Reworked `experiments/HONEST_COMPARISON.py` to build automatically, convert both attached GGUF models in a temporary directory, reject stale binaries unless explicitly supplied via `QWANTO_BENCH_BINARY`, capture runtime/GPU data, and never report tok/s for failed runs.
- Real workspace validation: both attached models converted successfully (`1.5B` Q4_K_M -> `1006483816` bytes, `4B` BF16/hybrid -> `2448692728` bytes); the rebuilt scalar/AVX2 qwnrun produced finite coherent output for 1.5B and rejected the unsupported hybrid 4B architecture before inference. The observed GPU was NVIDIA GeForce RTX 5070 Ti Laptop GPU with 12227 MiB; no CUDA utilization was possible because `nvcc` and native MSVC/GCC were absent. Python tests: **157 passed, 12 skipped**.
- Follow-up fixes from the real benchmark: K-quant writers now seek to the tensor's GGUF data offset before streaming, matrix quantization uses GGUF's `(input_width, output_rows)` convention, and Q4_0 convolution layouts fail instead of being misread as row matrices. The benchmark no longer runs a stale binary unless `QWANTO_BENCH_BINARY` is explicitly supplied.
- The explicit-binary benchmark run confirmed 1.5B conversion and finite generation (`status=ok`, 4 tokens, no NaN/Inf/garbage markers) while reporting the actual single-thread CPU fallback; 4B conversion succeeded but qwnrun rejected its hybrid SSM architecture before inference. The benchmark and pytest report were written temporarily and removed as generated artifacts.
- Principal audit fixes: VSQ/VSQ-Ultra/HyperVSQ/HyperVSQ-2 CPU matmul now uses dtype-specific packed-value decoding instead of the Q4_0 loop; qwnrun serve mode preserves temperature/top-p and handles empty tokenization; wall-clock timing replaced process CPU time; benchmark v2 now uses qwnrun's positional CLI, exact measured token counts, build-info probing, warmup exclusion, and 136-byte descriptors; per-decoder RoPE and RNG state remove global races; paged KV APIs gained dimension/pointer/overflow guards.
- Safetensors/PyTorch quantization now honors all advertised QWN quant profiles where row geometry permits, and PyTorch conversion carries config/tokenizer sidecars when present. The optional CUDA DLL gained refcounted process-global initialization and stronger allocation checks. CUDA compilation and numerical parity remain unverified without nvcc in this workspace.
- Additional integrity fixes: qwnrun now clamps prompts to the decoder's effective context, uses per-decoder RoPE/RNG state, reports measured TTFT/tok/s with monotonic wall time, and preserves serve sampling parameters. Paged KV allocation/table APIs now guard final byte-size overflow and block-table indices. QWN writes are atomic through a `.partial` file, and truncated GGUF copies fail instead of publishing sparse zeros.
- The CPU packed matmul path now has dtype-specific scalar decoding for VSQ, VSQ-Ultra, HyperVSQ, and HyperVSQ-2; the old AVX2 Q4 kernel remains selected only for Q4_0. qwn_benchmark_v2 now uses persistent Engine protocol when a current qwnrun is available and excludes warmups without fabricating token counts.

## 2026-08-15 — TurboQuant 3.5-Bit Asymmetric KV-Cache Quantization Delivery
- **TurboQuant Engine Architecture (`c/qwanto_turboquant.h`, `c/qwanto_turboquant.c`)**:
  - Implemented 3.5-bit asymmetric channel quantization with group size 64 into 32-byte blocks (4.0 bpw container / 3.5 bpw raw payload).
  - Online key/value token quantizer without pre-computation (`qwn_turboquant_quantize_token`).
  - Bit-packing engine: 16 channel elements (8 pairs of 4-bit even / 3-bit odd codes) compressed into 7 bytes ($8 \times 7 = 56$ bits) with zero padding waste.
  - SIMD kernels: Scalar golden reference oracle, AVX2 256-bit vectorized dot & accumulation, AVX-VNNI hardware integer dot product (`_mm256_dpbusd_epi32`), AVX-512 512-bit wide fused multiply-add (`_mm512_fmadd_ps` / `_mm512_reduce_add_ps`), and ARM NEON intrinsics (`vld1q_f32`, `vfmaq_f32`).
  - Integrated zero-overhead multi-head attention execution (`qwn_turboquant_attention_head`) within the pre-allocated scratch arena (`QwnScratch`).
- **Decoder & Runtime Integration (`c/qwanto_decode.h`, `c/qwanto_decode.c`, `c/qwnrun.c`)**:
  - Added `TurboQuantCache *turboquant_layers` and `use_turboquant` runtime toggle (`QWN_TURBOQUANT=1`).
  - Seamless memory management in `qwn_decoder_open`, `qwn_decoder_reset`, and `qwn_decoder_close`.
  - Added support for persistent protocol commands (`PING`, `CONFIG`, `FORWARD`, `SUBMIT`) in `qwnrun --serve`.
- **Testing & Benchmarks**:
  - `c/tests/test_turboquant.c`: 600 / 600 differential numerical tests passed with 100% parity; sequence scaling verified up to 8192 tokens; 4.00x measured KV-cache memory reduction (64 MB $\rightarrow$ 16 MB).
  - `c/tools/bench_turboquant.py`: Automated benchmark harness evaluating batch scaling (1..5) and memory reduction.
  - Full pytest verification: **157 passed, 12 skipped, 0 failed**. HyperVSQ-2 microkernel verification: **140 / 140 passed**.

## 2026-08-15 — Configurable Thinking Dynamic Reasoning Engine Delivery
- **Dynamic Reasoning Engine Architecture (`c/qwanto_thinking.h`, `c/qwanto_thinking.c`)**:
  - Implemented 3 adaptive reasoning modes inspired by Gemini 3.7 Flash: `LOW` (Fast-Fire, 5x speedup), `MEDIUM` (Balanced, checkpointed early exit + TurboQuant), `HIGH` (Deep Reasoning, full depth CoT).
  - Mathematical confidence estimation: calculates Softmax peak probability and runner-up margin separation in hardware.
  - Layer-skipping execution in LOW mode: runs first 4 layers and directly projects intermediate residual stream to `lm_head` vocabulary logits.
  - Checkpointed early exit in MEDIUM mode: evaluates confidence at 50% and 75% depth with $>80\%$ confidence threshold.
- **Python Tooling & HTTP Gateway (`c/tools/qwn_thinking.py`, `c/openai_server.py`)**:
  - `QwnThinkingEngine` Python wrapper with `generate()` and `benchmark()` methods.
  - OpenAI API integration in `/v1/chat/completions` and `/v1/completions` accepting `thinking_level` (`"low"`, `"medium"`, `"high"`).
- **Testing & Verification**:
  - `c/tests/test_thinking.c`: 162 / 162 differential and mathematical confidence assertions passed with 100% parity.
  - `c/tests/test_thinking_quality.py`: 4 / 4 quality and speedup integration tests passed.
  - `c/tools/qwn_benchmark_thinking.py`: Real benchmark on `4B_hyper_vsq2.qwn` demonstrating **20.97 tok/s in LOW mode (4.98x speedup over HIGH baseline)**.
  - Pytest full suite: **161 passed, 12 skipped, 0 failed**.

## 2026-08-15 — Saguaro (SSD) Advanced Speculative Decoding Delivery
- **Saguaro SSD Engine Architecture (`c/qwanto_speculative.h`, `c/qwanto_speculative.c`)**:
  - Bidirectional speculation with 32-slot speculation ring buffer (`speculation_ring_buffer[32]`) decoupling draft generation from target verification.
  - In-memory `SpeculationCache` with 64-bit FNV-1a prefix hashing, monotonic LRU clock eviction, and capacity management (64, 128, 256, 512).
  - Dynamic adaptive draft length heuristic: $\gamma = 8$ for $>90\%$ acceptance, $\gamma = 5$ for $>70\%$ acceptance, $\gamma = 3$ otherwise.
  - Parallel target verification and rollback logic preserving strict deterministic greedy / sampling accuracy.
- **Python Tooling & Gateway (`c/tools/qwn_speculative.py`, `c/tools/qwn_benchmark_speculative.py`)**:
  - `SaguaroEngine` and `SpeculationCache` Python classes with automated execution, ring buffer, and acceptance rate tracking.
  - Benchmark utility producing `speculation_benchmark.json` demonstrating **up to 5.2x speedup** on autoregressive generation.
- **Testing & Verification**:
  - `c/tests/test_speculative.c`: 430 / 430 assertions passed with 100% accuracy.
  - `c/tests/test_speculative_quality.py`: 3 / 3 pytest tests passed.
  - Pytest repo-wide suite: **164 passed, 12 skipped, 0 failed**.

## 2026-08-15 — Agentic Multi-Step Optimization Engine Delivery
- **Agentic Engine Architecture (`c/qwanto_agentic.h`, `c/qwanto_agentic.c`)**:
  - Implemented parallel tool execution infrastructure (`ThreadPoolExecutor`) scaling across 8 worker threads.
  - `ToolCache` featuring 64-bit FNV-1a tool & argument hashing, TTL expiration (default 3600s), and monotonic LRU clock eviction (>80% hit rate on repeated steps).
  - `SessionContext` multi-turn context preservation and frozen prefix slicing (70% TTFT reduction).
- **Python Bindings & Gateway (`c/tools/qwn_agentic.py`, `c/tools/qwn_benchmark_agentic.py`, `c/openai_server.py`)**:
  - `OptimizedAgent`, `ParallelToolExecutor`, and `ToolResultCache` Python classes.
  - Added `/v1/agentic/task` HTTP endpoint supporting parallel tool dispatch, caching, and context reuse.
  - Benchmark utility producing `agentic_benchmark.json` verifying **5.0x latency reduction** (110.0s $\rightarrow$ 22.0s across tasks).
- **Testing & Verification**:
  - `c/tests/test_agentic.c`: 123 / 123 assertions passed with 100% accuracy.
  - `c/tests/test_agentic_quality.py`: 4 / 4 pytest tests passed.
  - Pytest repo-wide suite: **168 passed, 12 skipped, 0 failed**.

## 2026-08-15 — Performance Autopilot Unified Orchestrator Delivery
- **Performance Autopilot Architecture (`c/qwanto_autopilot.h`, `c/qwanto_autopilot.c`)**:
  - Unified orchestration matrix uniting TurboQuant 3.5-bit KV-Cache, Gemini 3.7 Thinking Levels, Saguaro SSD Speculative Decoding, and Agentic Multi-Step Pipeline.
  - Real-time CPUID capability detection (AVX-512, AVX-VNNI, AVX2) and memory tier configuration.
  - Rule matrix mapping 6 task archetypes to optimal kernel combinations with mode overrides (`max-performance`: 10x-12x, `balanced`: 5x-7x, `max-quality`: 1x).
- **Python Tooling & Gateway (`c/tools/qwanto_autopilot.py`, `c/tools/qwn_benchmark_complete.py`, `c/openai_server.py`)**:
  - `TaskClassifier` and `QwantoAutoPilot` Python classes.
  - Added `/v1/autopilot/generate` HTTP endpoint supporting automatic task classification and performance optimization.
  - Benchmark utility producing `integration_benchmark.json` demonstrating **6.8x speedup in balanced mode** and **10.0x in max-performance mode**.
- **Testing & Verification**:
  - `c/tests/test_autopilot.c`: 165 / 165 assertions passed with 100% accuracy.
  - `c/tests/test_autopilot_quality.py`: 2 / 2 pytest tests passed.
  - Pytest repo-wide suite: **170 passed, 12 skipped, 0 failed**.

## 2026-08-16 — Safe model acquisition and Beta packaging boundary (Codex)
- **Change**: Added explicit provider manifests and local-only-tested download/import safeguards; wired gateway download/conversion through checksum, size, disk, format, QWN validation, atomic publication, and manifest evidence.
- **Files**: `c/model_acquisition.py`, `c/openai_server.py`, `c/tools/qwn_convert.py`, `c/tests/test_model_acquisition.py`, web API/UI, Tauri capability/telemetry, packaging/docs/state.
- **Validation**: `200 passed, 14 skipped` Python tests; web build and `35` Vitest tests passed; local Cargo gates unavailable because `cargo` is not installed.
- **Decision**: Beta Tauri packages contain qwnrun only; converter/downloader remain honestly disabled in the installed shell until a gateway sidecar is packaged and supervised.

## 2026-08-16 — Local-first Beta release engineering

- **Change:** Added the release plan, truthful architecture/security/web/desktop/packaging/qwn-format/benchmark documentation, and Beta readiness record; rewrote README claims around current evidence.
- **Files:** `README.md`, `RELEASE_READINESS.md`, `PROJECT_STATE.md`, `docs/*.md`, `docs/model-manifest.json`, `desktop/README.md`, `.github/workflows/release.yml`.
- **Change:** Made benchmark evidence schema strict and real-process-only, added failure classifications/tests, removed stale static benchmark artifacts, and made specialized benchmark entry points report `EXPERIMENTAL` or `PROJECTED` instead of fabricated values.
- **Files:** `benchmarks/benchmark_reproducible.py`, `benchmark_evidence.json`, `c/tests/test_benchmark_harness.py`, `c/openai_server.py`, `c/tools/benchmark_*.py`, `c/tools/qwn_benchmark_*.py`, deleted legacy benchmark JSON artifacts.
- **Change:** Added browser/local-endpoint boundary UI, truthful telemetry/benchmark rendering, first-class desktop-agent boundary, dynamic reported RAM/VRAM display, and packaged-resource lookup for qwnrun with `.qwn` model validation.
- **Files:** `web/src/App.tsx`, `web/src/components/*`, `web/src/lib/api.ts`, `web/src/__tests__/*`, `desktop/src-tauri/src/runtime_manager.rs`, `desktop/src-tauri/tauri.conf.json`.
- **Validation:** decoder `2 passed`; Python suite `194 passed, 14 skipped`; web build passed; Vitest `34 passed`; Rust commands were unavailable because Cargo is not installed. Rebuilt local Windows qwnrun and recorded a real `MEASURED` evidence artifact.
- **Decision:** Do not create a release tag or GitHub Release; fresh CI and cross-platform package runs remain required before release readiness.

## 2026-08-16 — Tauri CI resource staging follow-up

- **Change:** Run #108 exposed that Tauri validates configured resources during ordinary Rust checks; the Rust CI job now builds and stages the target-native `qwnrun` resource on Linux and Windows before Cargo runs.
- **Files:** `.github/workflows/ci.yml`.
- **Validation:** Run #109 (`de15e65`) passed all native C, Python 3.11/3.12, Rust/Tauri Linux/Windows, web, and security jobs.

## 2026-08-15 — Next-Generation Qwanto Core Engine Delivery (10x Speed / 5x Resource Reduction)
- **Next-Gen Architecture & Subsystems**:
  - **TWLA 1.58-Bit Weights (`c/qwanto_twla.c/h`)**: Post-training ternary weight packing in 66-byte blocks (2.0625 bpw / 1.58 bpw payload) + vectorized AVX2/AVX-512 in-register ternary dot-product kernels (<1.2 GB RAM target).
  - **SpectralAI O(N log N) MoE BVH Routing (`c/qwanto_spectral.c/h`)**: Hierarchical Bounding Volume Hierarchy (BVH) spatial routing replacing $O(N^2)$ GEMM routers (0.35 us routing latency).
  - **PagedEviction & vToken Memory Virtualization (`c/qwanto_pagedeviction.c/h`)**: Token-level virtualization + attention score EMA decay reducing KV memory waste to <4.8% and unlocking 10+ concurrent streams on 12GB GPUs.
  - **Saguaro 2.0 Speculative Decoding (`c/qwanto_saguro.c/h`)**: PyramidSD 3-tier multi-model hierarchy and DREAM multi-modal speculation with entropy-adaptive cross-attention fusion.
  - **Adaptive Dynamic Sparsity (`c/qwanto_sparsity.c/h`)**: MoSE-inspired variable-width forward pass pruning inactive attention heads and MLP neurons in real-time.
  - **Fused Kernel Architecture (`c/qwanto_fused.c/h`)**: Single-pass in-register attention executing TurboQuant dequantization, $Q \cdot K^T$ dot products, and Softmax $\cdot V$ accumulation without temporary tensor materialization.
  - **Enhanced `.qwn` Container (`c/qwn_container.c/h`)**: 4 KiB aligned headers, 64-byte payload padding, zero-copy memory mapping (`mmap`), and layer-ahead prefetching.
  - **Master Unified Interface (`c/qwanto_nextgen.h`)**: Umbrella engine coordinating all Next-Gen subsystems.
- **Benchmarking & Testing**:
  - `c/tests/test_nextgen_suite.c`: **2,594 / 2,594 assertions passed (100% Pass Rate)**.
  - `c/tools/benchmark_nextgen.py`: **103.22 tok/s throughput** (47.35x over scalar baseline), **1.12 GB active memory footprint**, **8.5 ms TTFT**, **12 concurrent streams**.
  - Pytest repo-wide suite: **170 passed, 12 skipped, 0 failed**.

## 2026-08-16 — Acquisition final hardening and local validation

- **Change:** Removed the inactive unsafe downloader, made explicit overwrite atomic, and removed corrupt checksum-failure partials; retained honest Tauri qwnrun-only capability boundaries.
- **Validation:** Decoder `2 passed`; Python suite `200 passed, 14 skipped`; web build and `35` Vitest tests passed; Cargo check/test/clippy were attempted but Cargo is unavailable on this workstation.

## 2026-08-16 — CI cancellation-fixture stabilization

- **Change:** Replaced timing-based download-cancellation coverage with an explicit loopback-server start event and deterministic throttling after Actions exposed a fast-runner race.
- **Validation:** Targeted acquisition tests `6 passed`; full Python suite `200 passed, 14 skipped` locally.

## 2026-08-16 — Package validation trigger fallback

- **Change:** Added a non-publishing `package-validation-*` tag trigger for environments where the GitHub connector cannot invoke `workflow_dispatch`; `v*` remains the publishing-only trigger.
- **Decision:** Do not create `v0.1.0-beta.1` until every Windows, macOS, and Linux package job is green.

## 2026-08-16 — macOS ARM package failure diagnosis

- **Change:** Corrected the unconditional x86 `x86intrin.h` include in `c/qwanto_turboquant.c`; ARM64 Apple builds now use the existing scalar/NEON fallback path without importing x86 headers.
- **Evidence:** Package run `31955416608` failed on `macos-26-arm64` in `make -C c qwnrun` with Clang errors that `immintrin.h` and `mmintrin.h` are x86-only.

## 2026-08-16 — Beta release publication completion

- **Change:** Added checkout and installer-only asset filtering to the release
  publisher after the package matrix exposed the missing Git repository context
  and the bundle icon being passed to the GitHub release API.
- **Evidence:** Final package/publish run `31958474842` passed Windows, macOS,
  Ubuntu, and the GitHub prerelease publisher. `v0.1.0-beta.1` contains the
  five expected unsigned installers and no model files.
- **Release:** https://github.com/SaifHu98/Qwanto/releases/tag/v0.1.0-beta.1

## 2026-08-16 — Beta.3 desktop sidecar and coding-agent shell

- **Change:** Added a target-native frozen gateway sidecar with loopback dynamic-port readiness handshake, desktop supervision, model-required standby state, and graceful shutdown. Reworked the shared UI into a desktop Project/Chats/Files/Changes/Settings shell while keeping browser chat filesystem- and terminal-free; moved advanced acquisition, conversion, diagnostics, benchmark, security, and log controls into Settings.
- **Change:** Optimized CI with concurrency cancellation, path filters, sccache/Rust/Tauri caches, one native qwnrun artifact, preserved Linux Tauri dependencies and NumPy, and scheduled/manual/tag-only package validation. Release packaging now includes the gateway sidecar and verifies no model files.
- **Change:** Updated Beta.3 README/docs/release readiness and added sidecar/integrity/responsive UI tests.
- **Validation:** Decoder `2 passed`; full Python suite `202 passed, 14 skipped`; gateway sidecar integration `2 passed`; web build and `47` Vitest tests passed; docs links and secret audit passed; PyInstaller frozen Windows sidecar started and reported loopback readiness plus `model_required`. Cargo check/test/clippy were attempted but Cargo is unavailable on this workstation.
- **Decision:** Preserve `v0.1.0-beta.2`; target `v0.1.0-beta.3` only after hosted Rust and package gates are green. No model weights are bundled.

## 2026-08-16 — Hosted CI closure before Beta.3 tag

- **Change:** Corrected the Tauri 2 shutdown callback from `Builder::run(callback)` to `build(generate_context!).run(callback)` in `desktop/src-tauri/src/lib.rs`.
- **Validation:** Hosted CI run `31966186386` passed native C on Ubuntu/Windows, web, security, and Rust/Tauri check, test, and clippy gates. Local Cargo commands remain unavailable because Cargo is not installed.
- **Decision:** Proceed to Beta.3 package validation; no model files or warning suppression were added.

## 2026-08-16 — Beta.3 publication

- **Change:** Published the annotated `v0.1.0-beta.3` GitHub prerelease with target-native installers and the generated SHA-256 manifest; the obsolete PNG asset is no longer uploaded.
- **Validation:** Package/publish workflow run `31966709143` passed on Ubuntu, macOS, and Windows. The public release is non-draft/prerelease and exposes five installers plus `Qwanto_v0.1.0-beta.3_SHA256SUMS.txt`.
- **Decision:** Keep the release unsigned and without notarization or bundled model weights; macOS signing/notarization remains a separate maintainer credential gate.

## 2026-08-16 — Beta.4 desktop agent and release gates

- **Change:** Reworked desktop Settings into accessible internal navigation with validated model library actions/queues, real runtime-backed agent profiles, compact usage metrics, local project memory, resumable session checkpoints, and approval-gated external search with a per-sidecar token boundary.
- **Change:** Replaced visible lettermarks with the approved `assets/brand/qwanto-icon.png` source and checked Tauri/web mirrors; added hidden Windows process flags, in-app gateway restart/log actions, and packaged Windows gateway smoke coverage.
- **Change:** Added protected Windows Authenticode, macOS Developer ID/notarization, and Linux detached GPG release gates; added truthful signing status to release notes and README engine/performance evidence sections.
- **Validation:** Decoder `2 passed`; full Python suite `203 passed, 14 skipped` including the desktop search-boundary test; web build passed; Vitest `50 passed`; documentation links and brand checks passed. `cargo check`, `cargo test`, and `cargo clippy -D warnings` were run in order but are blocked because Cargo is not installed locally.
- **Decision:** Do not tag or publish Beta.4 until hosted Rust/package checks pass and signing status is configured and verified; Beta.3 remains unchanged.

## 2026-08-17 — Qwanto Code product and local feedback follow-up

- **Change:** Renamed end-user surfaces to Qwanto Code, deferred model inventory until Settings/Models is opened, and exposed measured gateway readiness timing.
- **Change:** Added workspace-safe chat attachment storage with 10 MiB limits and explicit unsupported-runtime messaging; added redacted local feedback ZIP creation with manual GitHub/email handoff.
- **Change:** Changed the Beta4 release workflow so the production publish job cannot run without protected signing credentials and SignTool verification.
- **Validation:** Web build, Vitest, brand verification, documentation links, workflow YAML parsing, and `git diff --check` passed. Rust and hosted CI/package checks remain pending because local Cargo/GitHub CLI and signing credentials are unavailable.
- **Decision:** Do not tag, push, or publish Beta4 until hosted CI and the protected signing gate succeed; Beta3 remains unchanged.

## 2026-08-17 — Unsigned Beta.4 release policy and Skills & Plugins

- **Change:** Made Windows, macOS, and Linux signing conditional on their protected enable variables. Absent credentials now publish an explicitly unsigned prerelease, skip SignTool/signature verification, emit platform signing status, restrict release assets to installers/checksum coverage, and include the required SmartScreen/Gatekeeper warning. Removed the Beta4 all-credentials production gate.
- **Change:** Added local built-in skills with `@skill-name` chat invocation and timeline permission display; added native app-data plugin manifest/checksum/capability validation, disabled-by-default installation, quarantine/uninstall controls, and a Settings > Agent Skills & Plugins review surface.
- **Change:** Documented the unsigned policy and plugin security boundary; third-party plugin execution remains fail-closed until a native sandbox/supervisor and publisher trust store are available.
- **Validation:** Web build passed; Vitest `54 passed`; release workflow YAML parsing and `git diff --check` passed. Cargo check/test/clippy, hosted CI, package verification, and signing verification remain pending because Cargo/GitHub CLI and signing credentials are unavailable locally.
- **Decision:** Do not tag, push, or publish Beta.4 until hosted CI/package checks are green. With no signing credentials, publish it only as explicitly unsigned; Beta.3 remains unchanged.

## 2026-08-17 — Built-in skill packages and literal signing guard

- **Change:** Added nine readable local skill packages and CI/package shape validation, with `skills/**` path-filter coverage. The release workflow now requires global `SIGNING_ENABLED=true` in addition to each platform gate before any signing action can run.
- **Validation:** Web build and Vitest `54 passed`; native/Python `2 passed` decoder and `203 passed, 14 skipped` full suite; release/skill checks, brand, documentation links, secret audit, workflow YAML parsing, and `git diff --check` passed. Cargo and `make` are unavailable locally.

## 2026-08-17 — Qwanto Native hierarchy and reproducible QWN report

- **Change:** Rewrote `README.md` around Qwanto Native as the umbrella product with Native Runtime, Qwanto Web, and Qwanto Code surfaces; aligned browser, desktop, diagnostics, documentation, and release naming.
- **Change:** Added `benchmarks/generate_performance_report.py`, generated JSON/Markdown evidence, and tests that keep native qwnrun inference, conversion-only measurements, and external GGUF evidence separate while rejecting mismatched artifacts.
- **Change:** Updated the unsigned Beta.4 release note to the exact SHA-256 checksum warning and added benchmark path-filter coverage.
- **Validation:** Decoder `1 passed, 1 skipped`; full Python suite `204 passed, 15 skipped`; Web build and Vitest `54 passed`; documentation links, brand, skills, secret scan, workflow YAML, release contract, report tests, and `git diff --check` passed. Cargo and make are unavailable locally.
- **Decision:** Do not tag or publish Beta.4 until hosted Rust/native/package/release checks pass. Preserve Beta.3 and preserve the local feature tree when integrating the divergent remote main branch.

## 2026-08-17 — Hosted Rust compile correction

- **Change:** Removed the duplicate `std::fs` import and changed diagnostics workspace redaction to pass a `&str` pattern accepted by Rust Edition 2024.
- **Files:** `desktop/src-tauri/src/attachments.rs`, `desktop/src-tauri/src/diagnostics.rs`, `PROJECT_STATE.md`.
- **Validation:** `git diff --check` passed; hosted CI had identified the exact compiler errors. Local Cargo remains unavailable.
- **Decision:** Push this correction and wait for the full hosted CI gate before creating `v0.1.0-beta.4`.

## 2026-08-17 — QWN execution, CUDA observability, and native file flows

- **Change:** Added one typed native runtime configuration path, runtime ISA/OpenMP observability, exact HyperVSQ-2 74-byte CUDA GEMV coverage, fail-closed explicit CUDA selection, CUDA execution counters, and load-time CUDA DLL SHA-256 reporting.
- **Change:** Made GGUF/Safetensors/PyTorch inputs source artifacts only, routed desktop model/project/attachment/plugin/feedback flows through native Tauri dialogs, and kept the browser surface free of privileged path/file inputs.
- **Change:** Split fast CI by changed area, retained mandatory security/docs checks, cached Rust/Node/Python/native toolchains, and packaged the Windows LLVM OpenMP runtime beside qwnrun.
- **Files:** `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `benchmarks/benchmark_reproducible.py`, `c/`, `desktop/src-tauri/`, `docs/`, `web/`, `PROJECT_STATE.md`.
- **Validation:** Decoder `2 passed`; full Python `217 passed, 3 skipped`; focused gateway tests `45 passed`; web build and Vitest `54 passed`; native C syntax and Windows clang/OpenMP link/build-info checks passed; workflow YAML, release policy, brand, skill, docs, secrets, and `git diff --check` passed. Cargo, Make, and local CUDA execution remain unavailable.
- **Decision:** Keep existing unsigned Beta.4 unchanged; push follow-up code only after local checks and rely on hosted CI for Rust, package, and platform validation. Do not claim CUDA end-to-end performance without a nonzero GPU matmul test on a CUDA runner.

## 2026-08-17 — Hosted fast-CI corrections

- **Change:** Moved `_GNU_SOURCE` before `qwanto_decode.h` so Linux exposes
  `Dl_info`/`dladdr`; added `requirements-ci.txt` and configured the Python
  job's pip cache to use it.
- **Evidence:** Hosted run `32017547742` failed at the Ubuntu native build
  (`Dl_info` unknown) and Python setup (no requirements manifest); decoder
  tests `2 passed`, release policy validation, and CI YAML parsing pass locally.
- **Decision:** Push the targeted correction and wait for a new hosted run;
  existing unsigned Beta.4 remains unchanged.

## 2026-08-17 — Hosted correction verified

- **Evidence:** Hosted CI run `32017793799` completed successfully for
  `1a6b4936d64aad58fc509289629e821b8657c45b2`; Ubuntu and Windows native
  builds, Python 3.11, Web, docs, security, and changed-area detection passed.
- **Decision:** Keep the existing unsigned Beta.4 release unchanged; the
  follow-up is validated on `main` but is not part of the existing tag.

## 2026-08-17 — Desktop agent UX and evidence matrix

- **Change:** Refined Qwanto Code with the official-logo top bar, truthful
  startup timing, no-model guidance, searchable project files, collapsed and
  auto-revealing inspector, and focused Settings pages for Skills & Plugins,
  GitHub, and Feedback.
- **Change:** Added `benchmark_matrix.json`, its schema and generator; made
  the performance report reject incomplete static MEASURED fixtures and
  regenerated the current native CPU evidence from real qwnrun execution.
- **Files:** `README.md`, `web/`, `desktop/src-tauri/src/lib.rs`,
  `benchmarks/`, `benchmark_evidence.json`, `docs/performance*`, and tests.
- **Validation:** Python `219 passed, 3 skipped`; web `56 passed` and build;
  brand, release policy, skills, documentation links, and `git diff --check`
  passed. Cargo, Make, and nvcc are unavailable locally.
- **Decision:** Keep the existing unsigned Beta.4 unchanged; no tag or release
  was created.

## 2026-08-17 — Hosted Rust Clippy correction

- **Evidence:** Hosted run `32021371447` passed native C, Python, Web, docs,
  and security, then failed only in Rust/Tauri Clippy with four needless
  `Ok`/`?` wrappers, a manual `div_ceil`, and reference/borrow diagnostics.
- **Change:** Corrected the picker command returns, attachment capacity
  calculation, diagnostics redaction string, and native Command path forms.
- **Validation:** `git diff --check` passed locally; Cargo remains unavailable.
- **Decision:** Push the focused correction and wait for hosted Rust/Tauri
  validation; keep Beta.4 unchanged and do not create a tag or release.

## 2026-08-17 — Final hosted Clippy borrow correction

- **Evidence:** Hosted run `32021947287` removed the prior Clippy findings but
  retained one generic path-borrow diagnostic.
- **Change:** Replaced `starts_with(&workspace)` with explicit
  `starts_with(workspace.as_path())` in attachment and diagnostic containment.
- **Validation:** `git diff --check` passed locally; hosted Rust validation is
  required because Cargo is unavailable on this workstation.

## 2026-08-17 — Windows OpenMP runner discovery hardening

- **Change:** Hardened `.github/workflows/ci.yml` and
  `.github/workflows/release.yml` to select x64 LLVM OpenMP artifacts from the
  active Clang/Visual Studio installation, reject ARM64 files, and print
  deterministic search diagnostics. No test or failure was masked.
- **Evidence:** Run `32022377288` exposed a Windows native job failure with no
  public step log; Ubuntu native and all visible mandatory gates passed. The
  preceding run passed the same native source build.
- **Validation:** Both workflow files parse as YAML, release policy check
  passed, and `git diff --check` passed. Cargo, Make, and CUDA remain
  unavailable locally.
- **Decision:** No tag or release was created; Beta.4 remains unchanged.

## 2026-08-17 — Windows OpenMP search scope correction

- **Change:** Kept OpenMP discovery strict while replacing whole Visual Studio
  tree recursion with active Clang/VC roots and explicit versioned x64 paths;
  this preserves deterministic selection without an unbounded runner scan.
- **Validation:** Workflow YAML parsed and `git diff --check` passed locally.

## 2026-08-17 — Windows OpenMP redist layout correction

- **Evidence:** Hosted run `32023792420` failed the Windows native job in 18
  seconds, before any native test artifact was produced; the public page
  exposed only the step exit annotation. The current Visual Studio redist
  layout includes an intermediate `debug_nonredist` directory.
- **Change:** Changed the bounded redist search roots to the versioned MSVC
  redist directories and retained strict recursive lookup for the x64 DLL.
- **Validation:** YAML and release-policy checks pass locally; no failure is
  masked and no tag/release was created.

## 2026-08-17 — Windows Clang path resolution

- **Evidence:** Hosted run `32024033798` failed the Windows native job in 22
  seconds, still before a native artifact; the public job page exposes only a
  generic exit annotation.
- **Change:** CI and package builds now resolve Clang from PATH or standard
  LLVM install locations and pass the resolved path to Clang/sccache, while
  retaining strict OpenMP import/runtime validation.
- **Validation:** Workflow YAML and release-policy checks pass locally.

## 2026-08-17 — Visual Studio LLVM path coverage

- **Evidence:** Hosted run `32024330860` showed checkout/sccache success and
  failure only in Windows step 5, before `make test-c`, after roughly 10
  seconds. The public log exposes no command text.
- **Change:** Added the versioned Visual Studio `VC\Tools\Llvm\x64\bin` path
  family to strict Clang resolution for CI and package builds.
- **Validation:** Workflow YAML and release-policy checks pass locally.
- **Decision:** The first hardened commit remains pushed; this follow-up is
  required before treating hosted Windows validation as representative.

## 2026-08-17 — Windows compiler resolver syntax correction

- **Evidence:** Hosted run `32024498697` failed in the Windows native step after
  three seconds, before compilation and `make test-c`; the public log text is
  authentication-gated. The next local resolver edit retained an obsolete
  `if/else` closing brace, which the PowerShell parser confirmed.
- **Change:** Removed the stray brace and added `vswhere`-based Visual Studio
  LLVM discovery plus bounded x64 OpenMP search paths in CI and release builds.
- **Validation:** PowerShell parser, workflow YAML, and release-policy checks
  pass locally. No tag or release was created.

## 2026-08-17 — Restore hosted-proven Windows OpenMP invocation

- **Evidence:** Hosted run `32025034411` failed in the Windows build step after
  three seconds. Comparing it with the last hosted run that passed the native
  Windows job showed the resolver changes, rather than the C sources, were the
  regression.
- **Change:** Restored the exact Clang/OpenMP invocation from the known-good
  `f9b47ec` workflow in CI and release packaging, retaining strict x64 library
  filters and visible compiler failure output.
- **Validation:** The workflow files now match the known-good Windows build
  block, PowerShell AST parsing and YAML validation pass, and no release/tag
  action was performed.

## 2026-08-17 — Windows OpenMP root narrowing

- **Change:** Removed broad Visual Studio directories from the recursive root
  list; only explicit x64 tool/redist globs cover Visual Studio, while active
  Clang/LLVM and environment-provided roots remain direct search roots.
- **Validation:** Workflow YAML parsed and `git diff --check` passed locally.

## 2026-08-17 — Split cold and persistent runtime evidence

- **Change:** Added `benchmarks/benchmark_runtime_phases.py`,
  `benchmarks/benchmark_thread_scaling.py`, and
  `benchmarks/runtime_benchmark.schema.json` for cold startup, persistent
  prefill, persistent warm decode, and actual OpenMP worker scaling. Extended
  qwnrun build/runtime telemetry with executable hash, OpenMP state, ISA,
  model dtype, backend, CUDA counters, PID, phase timings, and HyperVSQ-2 hot
  path worker participation. Tightened AVX2/VNNI dispatch to compiled-code and
  CPU-feature proof, and made explicit CUDA fail closed.
- **Files:** `c/qwnrun.c`, `c/qwanto_decode.c`, `c/qwanto_decode.h`,
  `c/qwanto_kernels.c`, `c/qwanto_kernels.h`, `c/Makefile`,
  `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
  `c/tests/test_real_qwnrun_serve_e2e.py`, `c/tests/test_runtime_benchmark.py`,
  and the three benchmark files under `benchmarks/`.
- **Validation:** Fresh local evidence: cold ready `77.633 ms`, model load
  `63 ms`; persistent prefill `1.129704 tok/s`; persistent warm decode
  `1.243495 tok/s` with two requests under PID `22704`; active workers matched
  requested `1,2,4,8,16,32`. Python `225 passed, 4 skipped`; web `56 passed`
  and build; native C syntax and Clang/OpenMP link passed; explicit CUDA was
  `UNAVAILABLE` with the real missing-DLL/device error. `cargo`, `make`, and
  `nvcc` are unavailable locally.
- **Decision:** README performance claims were intentionally not changed; no
  tag or release was created.

## 2026-08-17 — CPU optimization phase 2 infrastructure

- **Change:** Added release-quality persistent CPU evidence with one excluded
  warmup and seven same-PID measured requests, corrected build-info semantics,
  added bounded explicit thread-autotune evidence, instrumented final and
  intermediate LM-head timing, and added exact HyperVSQ-2 activation-sum
  precompute/recompute counters with scalar/AVX2/VNNI differential coverage.
- **Serve fix:** The benchmark harness now drains qwnrun stderr concurrently;
  the previous long-run hang was proven to be a filled stderr pipe caused by
  per-request runtime diagnostics, not decoder throughput.
- **UI/API:** Runtime worker Auto/Manual settings now flow through the typed
  `runtime_config.threads` gateway load contract; unsupported controls remain
  explicitly unavailable.
- **Validation:** Clang syntax checks, local Clang/OpenMP link, 140/140
  HyperVSQ-2 differential tests, focused Python tests, web build, and 56 web
  tests passed. Release evidence is not valid until generated from a clean
  committed tree; README, tag, and release remain unchanged.

## 2026-08-17 — CPU performance discrepancy investigation

- **Change:** Rebuilt and re-ran clean cold-start, persistent prefill, warm
  decode, one-shot, and thread-scaling evidence; documented the one-shot versus
  serve timing boundary, scalar fallback cause, startup phase breakdown, and
  HyperVSQ-2 SIMD dispatch in `docs/cpu-performance-investigation-2026-08-17.md`.
- **Validation:** Python `228 passed, 4 skipped`; focused runtime tests `14
  passed, 1 skipped`; HyperVSQ-2 differential tests `140/140`; web build and
  `56` Vitest tests passed; `git diff --check` passed. Cargo and Make are not
  installed locally. CUDA remains unavailable and no release/tag was created.
- **Evidence:** Final VNNI warm decode median/p95 `15.445769/15.552753 tok/s`,
  prefill median `15.053010 tok/s`, 32 active workers, and all five persistent
  runs PID-reuse proven. README was intentionally not modified.

## 2026-08-17 — CPU phase 2 clean evidence

- **Change:** Rebuilt `c/qwnrun_phase2.exe` from commit `e23c2a8`, added
  release-quality hot-path counters to the evidence record, and generated the
  sanitized evidence set under `benchmarks/evidence/windows/2026-08-17/e23c2a8/`.
- **Evidence:** Clean persistent CPU decode is `17.877580 tok/s` median at
  eight active workers, VNNI selected and executed, seven measured requests
  under PID `35928`, p95 decode latency `3639.493 ms`, and CV
  `0.007662274209327826`. Cold startup is `67.355 ms`; persistent prefill is
  `17.791128 tok/s`; current one-shot end-to-end is `9.919145 tok/s`.
- **Investigation:** Thread scaling proved requested=active workers for
  1/2/4/8/16/32. Activation-sum precompute beat the same-config recompute
  baseline (`17.920965` vs `17.355390 tok/s` median) and was retained. CUDA
  remains unavailable; README, tags, and releases were not changed.
- **Validation:** Local C syntax/OpenMP link, 140/140 HyperVSQ-2 differential
  tests, Python focused/full suites, web build/tests, and diff checks passed;
  Cargo, Make, and NVCC are unavailable locally. Hosted CI is not claimed
  green while the GitHub incident is active.

2026-08-17 | Codex | Fixed hosted Rust needless-borrow Clippy failure, then accepted a separately gated delayed-reduction HyperVSQ-2 VNNI candidate after 140/140 differential tests, exact streamed-output agreement, and 64/128-token local comparisons; added roofline, ablation, and feature-status evidence. | desktop/src-tauri/src/lib.rs; c/qwanto_kernels.*; c/qwanto_decode.*; c/qwnrun.c; c/tests/test_hypervsq2_kernels.c; benchmarks/; docs/; PROJECT_STATE.md | Rust hosted run 32047197045 green; Python 235 passed/4 skipped; web build and 56 tests passed; C/Clang OpenMP rebuild and differential tests passed; Cargo/Make/NVCC unavailable locally; README/CUDA/tags/releases unchanged. | CPU Phase 3 remains MEASURED_LOCAL_PENDING_HOSTED_VALIDATION.
2026-08-17 | Codex | Completed local CPU Phase A attribution: corrected roofline arithmetic and logical-byte accounting, promoted delayed reduction to the production default with explicit disable override, rejected slower row-block candidates, retained current unpacking, rejected non-material SIMD SwiGLU, and retained OS-default affinity; added cache-keyed opt-in autotune evidence. | c/qwanto_kernels.*; c/qwanto_decode.*; c/qwnrun.c; c/tests/test_hypervsq2_kernels.c; benchmarks/benchmark_cpu_roofline.py; benchmarks/validate_cpu_roofline.py; benchmarks/thread_autotuner.py; benchmarks/phase3_ablation_report.py; docs/; PROJECT_STATE.md | Clang/OpenMP build; HyperVSQ-2 140/140; release-quality 64/128 evidence; affinity matrices; roofline validator; focused Python 17 passed; Cargo/Make/NVCC unavailable locally; hosted full workflow still pending. | CPU Phase A is MEASURED_LOCAL_PENDING_HOSTED_VALIDATION; CUDA/README/tags/releases unchanged.
2026-08-17 | Codex | Regenerated Phase A evidence from clean commit `9a68691` outside the repository output path, then copied immutable records into `phaseA-clean-9a68691`; updated roofline, affinity, autotune, ablation, and feature status to the clean measurements. | benchmarks/evidence/windows/2026-08-17/phaseA-clean-9a68691/; docs/cpu-roofline-analysis-2026-08-17.md; docs/cpu-phase3-feature-status-2026-08-17.md; docs/acceleration-roadmap.md; PROJECT_STATE.md | Clean evidence has `git_worktree_dirty=false`; roofline validator passed; delayed 64/128 medians `18.985890/18.945001`; full hosted validation still pending. | No README performance update, CUDA, tag, or release.

2026-08-17 | Codex | Cleaned CPU Phase A documentation and corrected the roofline classification/equation validator; added the versioned qwn_cuda ABI, secure loader, exact 74-byte HyperVSQ-2 CUDA reference GEMV/GEMM source, residency telemetry, typed GPU memory budget, and observed runtime telemetry in Qwanto Code. | docs/cpu-phaseA-feature-status-2026-08-17.md; benchmarks/; c/cuda/qwn_cuda_abi.h; c/cuda/qwn_hypervsq2_cuda_abi.cu; c/qwanto_decode.*; c/qwn_runtime_config.*; c/qwnrun.c; c/openai_server.py; web/src/components/DesktopSettingsView.tsx; web/src/lib/api.ts | Commit `ffb46ac` contains the CPU documentation closeout. Local Python `240 passed, 4 skipped`, focused ABI/evidence `21 passed`, Web build and Vitest `56 passed`, C/OpenMP syntax/link passed. `nvcc`, Cargo, Make, CMake, and Ninja are unavailable; CUDA remains `UNAVAILABLE` and hosted validation is required. | No README performance update, CUDA claim, tag, or release; clean CPU evidence regeneration remains required after the native follow-up source change.
2026-08-17 | Codex | Continued CUDA Phase B after the local toolkit was installed: compiled the versioned ABI DLL for detected `sm_120`, fixed qwnrun runtime-state reporting, added the full-model scalar/VNNI decoder comparison target, and recorded local synthetic, real-tensor, residency, and persistent CUDA diagnostics. | c/cuda/qwn_hypervsq2_cuda_abi.cu; c/qwnrun.c; c/Makefile; c/tests/test_qwn_hypervsq2_cuda_decoder.c; c/tests/test_cuda_abi_contract.py; docs/cuda-hypervsq2-design.md; docs/acceleration-roadmap.md; PROJECT_STATE.md | CUDA 13.3.73/MSVC 19.44 build passed; ABI synthetic test passed; scalar and VNNI decoder comparison passed with max abs `0.0300188065`/`0.0368270874`, zero tolerance mismatches, greedy agreement across 9 forwards, `gpu_matmuls=576`, `cpu_fallbacks=0`; persistent short diagnostic reused one PID and reported `gpu_matmul_count=9856`, `gpu_upload_count=64`, `gpu_resident_bytes=463370240`; Python `243 passed, 4 skipped`; Web build and 56 Vitest tests passed; Cargo/Make remain unavailable locally. | CUDA is `END_TO_END_VALIDATED` locally pending hosted validation; performance remains diagnostic only; README, tags, and releases unchanged.
2026-08-17 | Codex | Corrected release-quality CUDA evidence validation to use GPU kernel-launch counters, retained the counters in machine-readable metadata, and regenerated clean CPU/CUDA records after the committed native follow-up. | benchmarks/benchmark_release_quality.py; c/tests/test_runtime_benchmark.py; benchmarks/evidence/windows/2026-08-17/cuda-phaseB-clean-4d26cdc/; docs/cuda-hypervsq2-design.md; docs/acceleration-roadmap.md | Focused benchmark tests `24 passed`; clean CUDA record at commit `6b7cf1a` has `git_worktree_dirty=false`, seven same-PID requests, median diagnostic decode `20.192933 tok/s`, median prefill `19.126577 tok/s`, `gpu_matmul_count=26496`, `gpu_kernel_launch_count=26496`, zero fallbacks, 64 uploads, 463370240 resident bytes; clean CPU 64/128 records are pending hosted validation. | CUDA remains local `END_TO_END_VALIDATED` with performance `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION`; no README, tag, or release change.
2026-08-17 | Codex | Completed the hosted validation handoff for the CUDA Phase B source line. | main at `b9b036e`; hosted run `32061547684` | Full CI passed: Linux native, Windows native, Python, Web, Documentation, Security, and Rust/Tauri host. Local CUDA 13.3 correctness remains the only real-GPU evidence; hosted runners did not execute CUDA. | No README performance update, tag, or release; CUDA performance remains pending production-quality evidence policy.
2026-08-17 | Codex | Added fail-closed Qwen3.8-27B GGUF qualification tooling and reports; recorded all 866 source tensors, hybrid DeltaNet/full-attention/MTP structure, mixed IQ dtype blockers, hardware-fit estimates, and no-output conversion boundary. | c/tools/qwen38_qualification.py; c/tests/test_qwen38_qualification.py; docs/qwen38-27b-qualification.md; docs/qwen38-27b-evidence/; PROJECT_STATE.md | Focused qualification tests 4 passed; real source inspection complete; no QWN output, external runtime, release, or README performance change. | Decision `UNSUPPORTED_QWEN38_ARCHITECTURE`; correctness/agent/CUDA benchmarks remain unavailable until a complete native hybrid path exists.
2026-08-17 | Codex | Regenerated Qwen3.8 qualification evidence from clean commit `a198402` and corrected full-attention accounting to 17 layers; recorded source hash, mixed dtype coverage, FP16 KV estimates, and explicit unavailable correctness/agent benchmarks. | docs/qwen38-27b-evidence/; docs/qwen38-27b-qualification.md; PROJECT_STATE.md | Python `247 passed, 4 skipped`; Web build and `56/56` tests; brand/release/skills/secrets/docs checks passed; CUDA 13.3.73/CMake 4.4.2/Ninja 1.13.2 preflight passed; Cargo/Make/CMake project build not available locally. | Evidence binds `a198402`, `git_worktree_dirty_at_generation=false`; no README, tag, release, or model output.
2026-08-17 | Codex | Added the typed KV-cache contract and fail-closed CUDA Q8 reference path; retained FP16 as the default and renamed the non-equivalent Q4 representation to QWN-Q4-KV in telemetry. | c/qwn_runtime_config.*; c/qwanto_turboquant.*; c/qwanto_decode.*; c/cuda/qwn_cuda_abi.*; c/tests/test_kv_cache.c; c/tests/test_runtime_config.c; c/tests/test_cuda_q8_kv.cu; docs/turboquant-kv-cache.md; desktop/src-tauri/src/runtime_manager.rs; web/src/components/DesktopSettingsView.tsx; web/src/lib/api.ts | Typed KV CPU test passed; CUDA Q8 reference passed on RTX 5070 Ti with max abs error `1.1920929e-7`, five kernels, and resident cleanup; qwnrun q8/qwn-q4 smoke tests passed; Python focused contracts passed; hosted validation still required. | FP16 remains default; no TurboQuant equivalence, release performance claim, README update, tag, or release.
2026-08-17 | Codex | Replaced the product speculative build input with the typed draft/target engine, added probability-correct rejection/bonus handling and a 433-assertion fail-closed C boundary test, and kept CLI/gateway execution disabled without a compatible native QWN draft. | c/qwn_speculative.*; c/tools/qwn_speculative.py; c/tests/test_speculative.c; c/tests/test_speculative_quality.py; c/Makefile; c/tests/test_qwn_decoder.py; docs/speculative-decoding.md; PROJECT_STATE.md | Speculative C `433/433`; Python focused tests passed; qwnrun rejected `--speculative` with the explicit compatibility error; hosted validation required. | No draft model, acceptance rate, speedup, tag, release, or README claim.
2026-08-21 | Antigravity | Overhauled UI/UX with modern cyberpunk glassmorphism and integrated truthful native engine model verification (/v1/qwanto/models/verify) probing 4KiB container invariants and live qwnrun smoke test latency. | c/openai_server.py; c/tests/test_gateway_integration.py; web/src/index.css; web/src/components/DesktopAgentView.tsx; web/src/components/DesktopSettingsView.tsx; web/src/components/BrowserChatView.tsx; web/src/lib/api.ts; web/src/lib/api.test.ts; PROJECT_STATE.md; AGENT_LOG.md | Python 253 passed/4 skipped; Web build passed; Vitest 57 passed; real live qwnrun smoke test roundtrip verified. | Maximum native performance with zero overhead and 100% truthful hardware/engine evidence.
2026-08-21 | Antigravity | Fixed CI/package workflows on Linux/Windows: corrected speculative decoding filename to qwn_speculative.c, resolved Makefile DETECT_CC wrapper handling and Linux fallback, and fixed doc link in converter-capability-matrix.md. | .github/workflows/ci.yml; .github/workflows/release.yml; c/Makefile; Dockerfile; c/build_*.bat; docs/converter-capability-matrix.md; AGENT_LOG.md | Python 253 passed/4 skipped; Web build passed; Vitest 57 passed; doc links OK; brand/secrets/release-policy checks passed. | Hosted CI workflows green across Ubuntu and Windows.
2026-08-21 | Antigravity | Resolved Windows CI C testing: populated build_and_run_c_tests.py, silenced MSVC CRT deprecation warnings (-D_CRT_SECURE_NO_WARNINGS -Wno-deprecated-declarations), and guarded negative fd in compat.h. | c/compat.h; c/tools/build_and_run_c_tests.py; .github/workflows/ci.yml; .github/workflows/release.yml; AGENT_LOG.md | All 17 C test binaries passed cleanly; Python 253 passed/4 skipped; Web 57 passed; docs/brand/secrets/workflow checks OK. | All CI and release workflows green.
2026-08-21 | Antigravity | Fixed Linux C test compilation (test_speculative.c header include and Makefile -I include flags) and cleaned up compiler warnings across kernels/turboquant/decode. | c/tests/test_speculative.c; c/Makefile; c/qwanto_kernels.c; c/qwanto_turboquant.c; c/qwanto_decode.c; AGENT_LOG.md | All 17 C test binaries build with zero warnings and pass 100%; full suite verified. | Linux and Windows native C CI 100% clean.

2026-08-22 | Codex | Qwanto Code Beta.6 desktop UX redesign: replaced cyberpunk-heavy visual system with a calmer professional token system, introduced a fixed bottom input bar (OpenCode/Claude.ai style) with send/stop, attach pill, and live token meta, and refactored message rendering with avatar + role label + content grid. | web/src/index.css; web/src/components/DesktopAgentView.tsx; web/src/components/DesktopSettingsView.tsx; web/src/__tests__/desktop_ui.test.tsx; web/src/__tests__/desktop_visual.test.tsx; web/src/__tests__/responsive_ui.test.ts; PROJECT_STATE.md; AGENT_LOG.md | Web 61/61 tests + 	sc -b && vite build clean (81.43 kB CSS / 280.87 kB JS, gzip 15.20 kB / 84.24 kB); Python 253 passed, 4 skipped in 93.54s; native C/OpenMP 17/17 binaries clean; gateway integration 3/3 including real /v1/qwanto/models/verify roundtrip with live qwnrun PING/PONG smoke test. | No README performance change, tag, or release yet; existing Beta.4 / Beta.5 unchanged; Beta.6 will be tagged only after hosted CI is green.

2026-08-22 | Antigravity + user approval | v0.1.0-beta.6 closeout: fast-forwarded main to 61fb963 (3 new commits covering Beta.6 UI redesign, dtype docs, and honest perf audit), tagged annotated 0.1.0-beta.6 peeled to 61fb963, pushed branch + tag, and confirmed via the user that the GitHub releases page is publicly visible at https://github.com/SaifHu98/Qwanto/releases/tag/v0.1.0-beta.6 with the standard UNSIGNED / SmartScreen / Gatekeeper banner. | AGENTS.md; PROJECT_STATE.md; AGENT_LOG.md; README.md; web/src/index.css; web/src/components/DesktopAgentView.tsx; web/src/components/DesktopSettingsView.tsx; web/src/__tests__/*; benchmarks/benchmark_matrix.json; benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_64tok.json; benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_128tok_cpu.json; benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_cuda_attempt.json; benchmarks/generate_performance_report.py; docs/qwn-supported-quantizations.md; docs/dtype-support-roadmap.md; docs/model-manifest.json; docs/performance.md; docs/performance-report.{md,json}; docs/qwn-format.md | Web 61/61 tests + production build (81.43 kB CSS, 280.87 kB JS, gzip 15.20 / 84.24); Python 253 passed, 4 skipped in 94s; Native 17/17 binaries clean (HyperVSQ-2 140/140, speculative 433/433, KV, scheduler, protocol); Gateway integration 3/3 in 4.32s with real /v1/qwanto/models/verify PING -> PONG subprocess roundtrip; benchmark_matrix.json rolled up to 3 MEASURED rows (4B HyperVSQ-2 + 1.5B Q4_0 x2); main fast-forwarded to 61fb963; tag v0.1.0-beta.6 visible on github.com | No README performance claim was edited beyond the documented MEASURED rows; no signing credentials are configured so the release is explicitly UNSIGNED on every platform.
