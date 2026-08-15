# PROJECT_STATE.md

## Purpose
**Qwanto** — unified local AI inference runtime. Runs LLMs (70B+) on consumer hardware by tiering weights across GPU VRAM / System RAM / NVMe with zero-copy mmap. Combines a proprietary `.qwn` native SIMD decoder engine, OpenAI-compatible HTTP gateway, MoE (GLM/OLMoE) specialist runtimes, GGUF passthrough via llama-server, and a React 19 web studio + Tauri desktop shell.

## Stack / Architecture
- **Backend (Python)**: `c/openai_server.py` (3200 LOC, dep-free HTTP gateway, `ThreadingHTTPServer`, SSE, LRU prompt cache, defense headers, model management), `c/orchestrator.py`, `c/backends.py`, `c/capabilities.py`, `c/doctor.py`, `c/resource_plan.py`.
- **CLI dispatcher**: `c/coli` (subcommands: chat, serve, run, info, plan, doctor, bench, convert, pack, inspect, build, web).
- **Universal Engine 2.0 pipeline (NEW)**: `c/tools/qwn_bpw_truth.py` → `qwn_model_ir.py` → `qwn_arch_registry.py` → `qwn_roles.py` → `qwn_quant_plan.py` → `qwn_plan_cli.py` → `qwn_benchmark_v2.py`. See `AGENT_LOG.md` for the full delivery of `Full Improve Plan.md` phases 0–2.
- **Native engine (C/C++)**: `c/qwanto_*.c/h` — core, kernels (AVX2/F16C/FMA/VNNI/AVX-512), decode, attention, router (MoE LSH), native (resident plan + mmap), plus paged KV (`qwn_paged_kv.c`), speculative decode (`qwn_speculative.c`), PagedAttention. High-performance vectorized HyperVSQ-2 (2.3125 bpw) SIMD engine with runtime CPUID dispatch (VNNI / AVX2 / Scalar). **TurboQuant 3.5-bit Asymmetric Online KV-Cache Engine (`qwanto_turboquant.c/h`)** with 4.0x–4.57x memory reduction, AVX-512 / AVX-VNNI / AVX2 / ARM NEON kernels, and zero-allocation scratch arena integration. Cross-platform via `compat.h` / `aio_compat.c` / `uring.h`. Optional CUDA (`backend_cuda.cu`) + Metal (`backend_metal.mm`). Tokenizer in `c/tok.h` + `c/tok_unicode.h`.
- **MoE specialist runtimes**: `c/glm.c` (DeepSeek/GLM-5.2 744B), `c/olmoe.c` (OLMoE), with `c/backend_loader.c` for dynamic linking.
- **Container format `.qwn`**: 4KiB header + up to 29 inline tensor descriptors, FNV-1a hash index, 4KiB-aligned tensor payloads, 64-byte padding, dtypes F32/F16/BF16/Q4_0/raw.
- **Quantization**: QWN-HyperVSQ (2.70 bpw claim, real payload 4.3125 bpw per `qwn_bpw_truth`), QWN-HyperVSQ-2 (sub-2-bit claim, real payload 2.3125 bpw), QWN-VSQ-Ultra (3.375 bpw), QWN-VSQ (4.125 bpw), Q4_0/Q8_0, TurboQuant KV-cache (3.50 bpw). Wire-speed multi-format ingest (GGUF/Safetensors/.pt/.pth/.bin/.onnx/.h5/.keras) via `c/tools/qwn_convert.py`.
- **Tools**: `qwn_convert.py`, `qwn_ppl.py`, `qwn_benchmark_v2.py` (real harness), `bench_turboquant.py`, `qwn_plan_cli.py` (emit `quant_plan.json`), `quant_ablation.py`, `convert_olmoe.py`, `convert_fp8_to_int4.py`, `make_glm_*`.
- **Web Dashboard**: `web/` — React 18 + Vite 8 + Tailwind 4 + lucide-react, glassmorphism dark UI, custom tests via Vitest. Views: Chat, Converter, Presets, Telemetry, Doctor, Workbench (API playground), Benchmarks, Security, Brain (MoE visualization). Core: `App.tsx` (1249 LOC), `Brain.tsx`, `lib/{api.ts, runtime.ts, storage.ts, utils.ts}`.
- **Desktop**: `desktop/src-tauri/` — Tauri v2 shell wrapping the shared `web/` UI. No bundled engine.
- **Tests**: `c/tests/` (pytest, 157 passed + 12 skipped; TurboQuant verification suite 600/600 passed; HyperVSQ-2 C kernel suite 140/140 passed), `c/iobench.c`, `c/Makefile` for native tests.

## Completed Components
- All listed in README tables — Qwanto Native, QWN-HyperVSQ engine, Ingestion pipeline, OpenAI gateway, LRU cache, Telemetry, Prompt Studio, Doctor, Security audit, MoE runtime, GGUF runtime, Web dashboard.
- HyperVSQ-2 Vectorized SIMD Engine: 74-byte packed superblock (256 elements, 2.3125 bpw), standalone scalar golden reference, AVX2 maddubs/madd kernel, AVX-VNNI `_mm256_dpbusd_epi32` kernel with zero-point correction, runtime CPUID dispatch + override flags (`QWN_FORCE_SCALAR/AVX2/VNNI`), and OpenMP multi-row parallelization.
- TurboQuant 3.5-Bit Asymmetric KV-Cache Quantization: 32-byte block for 64 channels (4.0 bpw container / 3.5 bpw raw payload), online quantization during generation, AVX2, AVX-VNNI, and AVX-512 SIMD kernels, 600/600 differential tests passed, 4.0x–4.57x memory footprint reduction.
- Fixed layer cache projection output geometries in `build_layer_cache()` and `qwn_decoder_forward()`.
- QWN safety pass: GGUF dimensions remain in their native fastest-first order; malformed `.qwn` descriptors are rejected before use; native Q8_0 row decode/matmul is implemented.
- Native qwnrun now fails fast for unsupported MoE/SSM layers and incompatible Q/K/V head dimensions instead of silently producing identity-layer or out-of-bounds results.
- Native runtime diagnostics report compiler, OpenMP status/thread count, ISA, selected CUDA devices, planned GPU/RAM/NVMe bytes, and prefetch calls. CUDA-capable builds accept `COLI_GPU`/`COLI_GPUS` and distribute resident Q4 tensors within per-device budgets.
- Dense qwnrun now uses the paged KV pool or TurboQuant KV cache when selected: 16-token logical blocks, 16 KiB-aligned allocations, gather scratch, and layer-ahead prefetch.

## Current Status
- Working dir `D:\EcoUni\qwanto`. Platform Windows / PowerShell 7.
- Built binaries: `qwnrun_msvc.exe`, `test_hypervsq2_kernels.exe`, `glm.exe`, `libqwanto.dll`, llama-server + ggml DLLs.
- Test models in `experiments/results/`: `4B_hyper_vsq2.qwn` (1.26 GB, 13.17 tok/s) and `4B_q4_0.qwn` (2.45 GB, 2.18 tok/s).
- Run scripts: `run_server.bat`, `run_server.sh`, `run_web_ui_only.bat`, `run_web_ui_only.sh`.
- Maintainer: SaifHu98. Original unified memory architecture credits JustVugg/Colibri.
- License: MIT (per README header) / Apache 2.0 (per LICENSE file — inconsistent, watch for clarification).

## Active Blockers
- GGUF Q4_K/Q5_K/Q6_K blocks are now dequantized with the verified ggml layout before QWN quantization. Q2_K/Q3_K/Q8_K and IQ blocks remain explicitly rejected until their decoders are verified; none may be copied as opaque bytes.
- The native qwn decoder currently supports dense Transformer layers only. MoE and SSM/hybrid layers require dedicated kernels before they can be enabled.
- CUDA and OpenMP availability remain toolchain-dependent; Windows build validated via Clang + MSVC OpenMP runtime.

## Important Decisions
- License inconsistency (MIT vs Apache 2.0) between README badge and `LICENSE` file.
- Acknowledge Colibri (JustVugg) as upstream basis for the multi-tier memory architecture.
- `.qwn` is dense-Llama/Qwen-optimized; MoE/GLM/OLMoE use their own specialist runtimes.
- GGUF path delegates entirely to external `llama-server` (downloaded automatically on Windows).
- Correctness precedes performance claims: unsupported dtype, architecture, shape, or backend paths fail explicitly and produce no tok/s result.
