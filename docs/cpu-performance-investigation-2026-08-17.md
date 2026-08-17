# CPU Performance Investigation — 2026-08-17

## Scope

This investigation resolves the difference between the earlier one-shot result
near `8.154199 tok/s` and the persistent warm-decode result near
`1.243495 tok/s`. It covers the committed source at `70083b17332bb32d4f409d91d8188f58def20eec`.

No CUDA implementation, README performance update, tag, or release was made.
The evidence below is local Windows evidence and is not a cross-host comparison.

## Clean rebuild and identity

The worktree was clean before the final evidence generation. The pre-existing
untracked `benchmark_evidence_msvc.json` was kept outside the worktree while
the evidence was generated, so the records contain `git_worktree_dirty=false`.
It remains a user artifact and is not part of this change.

Model:

- Path: `experiments/results/4B_hyper_vsq2.qwn`
- Architecture: `qwen35`
- QWN dtype: `HyperVSQ-2` (QWN 2.31, 74-byte block layout)
- SHA-256: `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`

Executable:

- Path: `c/qwnrun_investigation.exe`
- SHA-256: `a89a65d508f080c24b3496d23ad673e3edbb1918d8292aa1c18d3524c6a6a03c`
- Git commit: `70083b17332bb32d4f409d91d8188f58def20eec`
- Compiler: Clang `21.1.6`
- Optimization flags: `O3,mavx2,mf16c,mfma,openmp`
- OpenMP compiled: `true`
- OpenMP runtime loaded: `true`
- OpenMP runtime version: `202011`
- CPU features detected: AVX2, F16C, FMA, AVX-VNNI, AVX-512F
- Binary kernels present: AVX2 `true`, VNNI `true`
- Runtime-selected kernel for this model: `vnni`
- CUDA: not used; GPU matmul count `0`

The Windows rebuild used the repository's Clang/OpenMP source set and flags,
with the local OpenMP import library and `psapi` library supplied by the active
LLVM/Windows toolchain. The equivalent CI build command is:

```powershell
clang -O3 -mavx2 -mf16c -mfma -fopenmp '-DQWN_BUILD_OPT_FLAGS="O3,mavx2,mf16c,mfma,openmp"' `
  qwnrun.c qwn_runtime_config.c qwanto_decode.c qwanto_native.c qwanto_kernels.c `
  qwanto_turboquant.c qwanto_gpu.c qwanto_autopilot.c qwanto_thinking.c `
  qwanto_speculative.c qwanto_agentic.c qwanto_bitdecoding.c qwanto_jetspec.c `
  qwanto_talon.c qwanto_sliminfer.c qwanto_pquant.c qwanto_littlebit.c `
  qwn_paged_kv.c -o qwnrun_investigation.exe -lpsapi <path-to-libomp.lib>
```

Raw build-info command and output:

```text
qwnrun_investigation.exe --build-info --json --backend cpu --threads 32 --thinking none
{"compiler":"clang","compiler_version":"21.1.6","optimization_flags":"O3,mavx2,mf16c,mfma,openmp","openmp_compiled":true,"openmp_runtime_loaded":true,"openmp_version":"202011","requested_threads":32,"active_threads":32,"cpu_features":{"avx2":true,"f16c":true,"fma":true,"vnni":true,"avx512f":true},"binary_avx2_kernel":true,"binary_vnni_kernel":true,"selected_isa_kernel":"vnni","binary_sha256":"a89a65d508f080c24b3496d23ad673e3edbb1918d8292aa1c18d3524c6a6a03c","backend_requested":"cpu","backend_actual":"Unavailable","gpu_matmul_count":0,"cpu_fallback_count":0,"model_dtype":"Unavailable","thinking_mode":"none","kv_cache_mode":"fp16","quantization":"auto","kernel_requested":"auto","pid":25376}
```

`backend_actual=Unavailable` in build-info is intentional: no model matmul has
run in that process. In an actual request, the result record changes to
`backend_actual=cpu`, reports `kernel=vnni`, and reports the hot-path worker
count.

## Reproduction commands

All commands used the same model, prompt, context, seed, sampler, backend, and
explicit `--thinking none` unless the workload-specific token count differed.
Each benchmark record includes a `runtime_config_snapshot` containing these
values, the prompt hash, selected kernel, model dtype, and decode function.

```powershell
python -m benchmarks.benchmark_runtime_phases --mode cold-start --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun_investigation.exe --backend cpu --threads 32 --context-size 4096 --max-tokens 8 --seed 0 --timeout 180 --output D:\EcoUni\qwanto-investigation-clean\cold_start_70083b1.json

python -m benchmarks.benchmark_runtime_phases --mode prefill --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun_investigation.exe --backend cpu --threads 32 --context-size 4096 --max-tokens 8 --seed 0 --warmup-tokens 8 --timeout 240 --output D:\EcoUni\qwanto-investigation-clean\prefill_70083b1.json

python -m benchmarks.benchmark_warm_repeats --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun_investigation.exe --backend cpu --threads 32 --context-size 4096 --max-tokens 8 --seed 0 --warmup-tokens 8 --repeats 5 --timeout 240 --output D:\EcoUni\qwanto-investigation-clean\warm_repeats_70083b1.json

python -m benchmarks.benchmark_reproducible --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun_investigation.exe --backend cpu --threads 32 --context-size 4096 --max-tokens 64 --seed 0 --warmup-tokens 8 --timeout 240 --output D:\EcoUni\qwanto-investigation-clean\one_shot_64_70083b1.json

python -m benchmarks.benchmark_thread_scaling --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun_investigation.exe --backend cpu --threads 1,2,4,8,16,32 --context-size 4096 --max-tokens 8 --seed 0 --warmup-tokens 8 --timeout 300 --output D:\EcoUni\qwanto-investigation-clean\thread_scaling_70083b1.json

python -m pytest c/tests/test_runtime_benchmark.py c/tests/test_phase23_hypervsq2_and_speculative.py c/tests/test_real_qwnrun_serve_e2e.py -q
c/tests/test_hypervsq2_investigation.exe
```

Evidence files:

- `D:\EcoUni\qwanto-investigation-clean\cold_start_70083b1.json`
- `D:\EcoUni\qwanto-investigation-clean\prefill_70083b1.json`
- `D:\EcoUni\qwanto-investigation-clean\warm_repeats_70083b1.json`
- `D:\EcoUni\qwanto-investigation-clean\one_shot_64_70083b1.json`
- `D:\EcoUni\qwanto-investigation-clean\thread_scaling_70083b1.json`
- Scalar comparison: `D:\EcoUni\qwanto-investigation-clean\warm_decode_repeats_scalar_before.json`

## Raw runtime output

The final executable was also run directly with two generated tokens:

```text
qwnrun result: status=ok tokens=2 wall_seconds=3.313000 ttft_ms=3169.805 tok_per_sec=0.603682 thinking_level=none
qwnrun result detail: backend=cpu kernel=vnni gpu_matmul_count=0 cpu_fallback_count=0 active_threads=32 dispatch_reason=cpu_vnni=yes;binary_vnni=yes;dtype=hypervsq2-74;selected=vnni decode_function=qwn_decoder_generate thinking_mode=none prompt_tokens=48 prefill_ms=3169.745 decode_wall_ms=137.678 sampling_ms=0.121 prefill_tok_per_sec=15.143175 decode_tok_per_sec=14.526607 generation_wall_ms=3313.000 process_create_ms=Unavailable file_open_ms=0.044 mmap_ms=0.018 metadata_parse_ms=0.063 tokenizer_init_ms=0.059 kv_cache_alloc_ms=46.688 advisory_preload_ms=0.094 first_tensor_touch_ms=52.388 first_real_forward_ms=179.527 total_end_to_end_ms=3375.000 config_backend=cpu context_size=4096 max_tokens=2 seed=0 kv_cache_mode=fp16 quantization=auto kernel_requested=auto temperature=0 top_p=1
```

The persistent harness's final `DONE` lines contain the same structured fields,
including request ID, PID, prefill time, decode-only time, and dispatch reason.
The warm-repeats record proves two sequential requests under one PID for every
one of its five measured runs.

## Discrepancy root cause

The two historical numbers were not measurements of the same quantity:

| Evidence | Workload and runtime | Result |
| --- | --- | ---: |
| Historical one-shot | A process-per-run 64-token end-to-end command from an older binary; process/model setup and prompt prefill were included and amortized over 64 tokens | `8.154199 tok/s` |
| Clean scalar persistent baseline | Persistent `--serve`, two sequential requests under each PID, scalar OpenMP build, one active hot-path worker, 8-token requests | median `1.078828 tok/s`, p95 `1.241919 tok/s` |
| Current one-shot | Current binary, VNNI, 32 workers, explicit `thinking_mode=none`, 64-token end-to-end workload | `8.404391 tok/s` end-to-end; internal prefill `14.689058 tok/s`, decode `15.370191 tok/s` |
| Current persistent warm decode | Current binary, same prompt/config family, already-ready process, warmup first, two sequential requests per run, 8 measured tokens | median `15.445769 tok/s`, p95 `15.552753 tok/s` |

The proven causes are therefore:

1. **Different timing boundaries.** The old one-shot rate divided all process,
   model, prompt, and token-generation work by 64 generated tokens. Warm decode
   intentionally starts after readiness, warmup, and prompt prefill, and divides
   only the measured generated-token interval. These are not interchangeable
   rates.
2. **Different executable and kernel.** The scalar comparison used executable
   SHA-256 `b846ae3be0d48eae4dfe481b17e87496d876c9b5696ccbaeb447b8e514694c6d`
   with one active worker. The final executable contains and selects the exact
   HyperVSQ-2 VNNI path and records 32 active workers. The old `8.154199` record
   was also generated by an older, dirty source state and cannot be treated as
   clean evidence for the current runtime.
3. **Decode-path defaults were previously ambiguous.** Before the typed
   `thinking_mode` path was made explicit, one-shot and serve could enter
   different generation functions. The reproducible harness now passes
   `--thinking none`; both one-shot and serve records state
   `decode_function=qwn_decoder_generate`, `thinking_mode=none`, greedy
   `temperature=0`, `top_p=1`, `context_size=4096`, and `seed=0`.
4. **Protocol overhead is not the explanation for the old gap.** Current
   persistent records separate `prefill_ms`, `decode_wall_ms`, `sampling_ms`,
   and `protocol_request_wall_ms`. The protocol request wall time is the
   prefill-plus-decode request interval, while the measured decode value is the
   post-prefill generation interval. The large historical difference tracks the
   scalar kernel/worker configuration and measurement boundary, not a hidden
   stdout or framing fallback.

The runtime configuration snapshots and the regression test
`test_runtime_snapshot_exposes_decode_path_and_is_comparable` prevent an
equivalent one-shot and persistent configuration from silently diverging again.

## Cold-start breakdown

The regenerated cold record reports:

| Field | Measured value |
| --- | ---: |
| `process_create_ms` | `3.134 ms` |
| `file_open_ms` | `0.058 ms` |
| `mmap_ms` | `0.025 ms` |
| `metadata_parse_ms` | `0.072 ms` |
| `tokenizer_init_ms` | `0.071 ms` |
| `kv_cache_alloc_ms` | `87.610 ms` |
| `advisory_preload_ms` | `0.125 ms` |
| `first_tensor_touch_ms` | `93.695 ms` |
| `first_real_forward_ms` | `Unavailable` in cold-start-only mode |
| `prompt_prefill_ms` | `Unavailable` in cold-start-only mode |
| `decode_ms` | `Unavailable` in cold-start-only mode |
| `total_end_to_end_ms` | `Unavailable` in cold-start-only mode |
| `cold_start_ms` | `121.834 ms` |
| `model_load_ms` / `runtime_ready_ms` | `109.000 ms` |

The cold mode stops after the READY record, so it does not execute inference.
The persistent prefill record measured `first_real_forward_ms=188.622 ms` and
the current one-shot record measured `first_real_forward_ms=180.799 ms`.
Consequently, the `0.025 ms` mmap operation is only mapping the file; it is not
model-load completion. The separate first-tensor-touch and first-forward values
show where actual model access begins.

## HyperVSQ-2 dispatch analysis

The real QWN HyperVSQ-2 path computes the 74-byte block layout with 256 values
per block. The dispatcher in `c/qwanto_kernels.c` checks all of the following
before selecting a SIMD function:

| Decision | Result on this host/build |
| --- | --- |
| CPU supports AVX2/FMA/F16C | Yes |
| CPU supports AVX-VNNI | Yes |
| Binary contains AVX2 kernel | Yes |
| Binary contains VNNI kernel | Yes |
| Model dtype is HyperVSQ-2 74-byte path | Yes |
| Dispatcher selection | `vnni` |
| Hot-path execution | Proven by `kernel=vnni`, matmul counters, and differential tests |

The previous scalar binary had OpenMP but did not contain the compiled AVX2
kernel, so the safe dispatcher selected scalar. OpenMP availability alone does
not imply an ISA kernel is present. If CPU support, compiled code, or dtype
compatibility is absent, the safe fallback remains scalar and the dispatch
reason exposes the failed condition rather than forcing an instruction set.

The direct HyperVSQ-2 verification executable reported:

```text
Detected CPU Features:
  AVX2:     YES
  F16C:     YES
  FMA:      YES
  AVX-VNNI: YES
  AVX-512F: YES
Differential Tests: 140 passed / 140 total
[SUCCESS] All differential numerical tests passed!
```

Its microbenchmarks are kernel-level diagnostics, not product throughput
claims. The current run measured approximately `25.00–26.40 GFLOPS` for VNNI
versus `8.17–8.26 GFLOPS` scalar across the tested shapes. End-to-end evidence
uses the qwnrun records above, not these synthetic-kernel numbers.

## Warm performance and worker scaling

The five-run persistent workload used prompt
`Explain zero-copy NVMe memory tiering in Qwanto.`, 48 prompt tokens,
`context_size=4096`, `seed=0`, `max_tokens=8`, 8 warmup tokens, greedy sampling,
CPU backend, and VNNI at the selected 32-worker setting.

| Measurement | Before: scalar | After: VNNI |
| --- | ---: | ---: |
| Warm decode median | `1.078828 tok/s` | `15.445769 tok/s` |
| Warm decode p95 | `1.241919 tok/s` | `15.552753 tok/s` |
| Prefill median | `1.084978 tok/s` | `15.053010 tok/s` |
| Prefill p95 | `1.247275 tok/s` | `15.533832 tok/s` |
| Hot-path workers | `1` | `32` |
| PID reuse proof | 2 requests/PID | 2 requests/PID |

The thread-scaling workload intentionally uses its own fixed prompt
(`Measure the local QWN decode path.`) and must not be compared numerically with
the table above. It measured active workers and decode throughput as follows:

| Requested workers | Active workers | Decode tok/s |
| ---: | ---: | ---: |
| 1 | 1 | `6.302023` |
| 2 | 2 | `10.235522` |
| 4 | 4 | `14.544804` |
| 8 | 8 | `16.518765` |
| 16 | 16 | `18.136674` |
| 32 | 32 | `15.138301` |

This demonstrates actual worker participation, not a claim of monotonic
scaling. The 32-worker result is lower than the 16-worker result for this
short workload, so no fixed scaling target is asserted.

## CUDA status

CUDA remains `UNAVAILABLE` in this phase. The detected RTX 5070 Ti is hardware
detection only. There is no exact 74-byte HyperVSQ-2 CUDA implementation in
this change, no proven CUDA model matmul, and no valid CUDA benchmark row.
`gpu_matmul_count=0`, and no result is classified as CUDA `MEASURED`.

## Validation status and blockers

Passed locally:

- Focused benchmark/runtime/serve tests: `14 passed, 1 skipped`.
- HyperVSQ-2 differential suite: `140/140` numerical tests passed.
- Five-run persistent evidence: all runs measured; PID reuse proven for all.
- Thread scaling: requested worker counts `1,2,4,8,16,32` matched active hot-path workers.
- Evidence records: clean commit identity and `git_worktree_dirty=false`.

The full Python suite, web suite, and native checks had passed earlier in this
investigation; they remain required after the documentation/state changes.
Cargo, Make, and NVCC are not installed on this Windows workstation, so Rust,
Makefile, and CUDA-toolchain gates require hosted/platform CI. README
performance claims remain intentionally unchanged, and no release or tag was
created.
