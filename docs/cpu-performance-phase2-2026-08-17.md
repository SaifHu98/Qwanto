# CPU Performance Phase 2 — 2026-08-17

## Scope and conclusion

This report records the clean-commit CPU investigation after commit
`e23c2a87f0498bd02281e2db9b6f01c6ddda56ad`. It does not implement CUDA, change
README performance claims, create a tag, or create a release.

The apparent `8.15 tok/s` versus `1.24 tok/s` discrepancy had two causes:

1. The old one-shot number was end-to-end: process creation, model access,
   initialization, prompt prefill, and decode were divided by the generated
   token count. It was not warm decode throughput.
2. The slow persistent comparison used an older scalar executable and a
   short-request measurement. That executable had OpenMP but did not contain
   a usable compiled HyperVSQ-2 SIMD kernel, so safe dispatch selected scalar
   work and the hot path ran with one effective worker. Persistent protocol
   framing was not the throughput fallback.

The clean current executable uses the same decoder function in one-shot and
serve mode. It selects VNNI only after CPU feature detection, compiled-kernel
availability, and HyperVSQ-2 74-byte dtype compatibility all succeed. The
release-quality persistent result is `17.877580 tok/s` median at eight active
workers; this is a new local measurement, not a README claim.

## Clean identity

The pre-existing untracked user file `benchmark_evidence_msvc.json` was moved
reversibly outside the repository while the clean evidence was generated. It
was not staged or deleted. The evidence records report
`git_worktree_dirty=false`.

| Item | Value |
| --- | --- |
| Git commit | `e23c2a87f0498bd02281e2db9b6f01c6ddda56ad` |
| Executable | `c/qwnrun_phase2.exe` |
| Executable SHA-256 | `3cca5eb31638ccaf8dad90992d46bd3828b6e2b9d09304bbf560a87e02e9f24b` |
| Compiler | Clang `21.1.6` |
| Optimization flags | `O3,mavx2,mf16c,mfma,openmp,phase2` |
| OpenMP compiled/runtime | `true / true` |
| Model | `experiments/results/4B_hyper_vsq2.qwn` |
| Model SHA-256 | `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36` |
| Model dtype | `HYPER_VSQ2` / HyperVSQ-2, QWN 2.31, 74-byte block |

Build command used from `c/`:

```powershell
clang -O3 -mavx2 -mf16c -mfma -fopenmp '-DQWN_BUILD_OPT_FLAGS="O3,mavx2,mf16c,mfma,openmp,phase2"' qwnrun.c qwn_runtime_config.c qwanto_decode.c qwanto_native.c qwanto_kernels.c qwanto_turboquant.c qwanto_gpu.c qwanto_autopilot.c qwanto_thinking.c qwanto_speculative.c qwanto_agentic.c qwanto_bitdecoding.c qwanto_jetspec.c qwanto_talon.c qwanto_sliminfer.c qwanto_pquant.c qwanto_littlebit.c qwn_paged_kv.c -o qwnrun_phase2.exe -lpsapi <local-libomp.lib>
```

Raw build-info output (model-free mode intentionally reports actual kernel as
`Unavailable`):

```text
{"compiler":"clang","compiler_version":"21.1.6","optimization_flags":"O3,mavx2,mf16c,mfma,openmp,phase2","openmp_compiled":true,"openmp_runtime_loaded":true,"requested_threads":8,"active_threads":"Unavailable","cpu_features":{"avx2":true,"f16c":true,"fma":true,"vnni":true,"avx512f":true},"compiled_kernels":{"avx2":true,"vnni":true},"preferred_kernel_candidate":"vnni","actual_executed_kernel":"Unavailable","backend_requested":"cpu","backend_actual":"Unavailable","gpu_matmul_count":0,"cpu_fallback_count":0}
```

## Configuration parity and discrepancy analysis

The current one-shot, prefill, and persistent records use the same model,
48-character prompt, context `4096`, seed `0`, CPU backend, `fp16` KV cache,
greedy sampling (`temperature=0`, `top_p=1`), `qwn_decoder_generate`,
HyperVSQ-2 dtype, and VNNI dispatch. Each record contains a
`runtime_config_snapshot`; the persistent release record also writes the same
snapshot onto every measured request and proves PID reuse.

The timing boundaries are intentionally different:

| Measurement | What it includes | Result | Classification |
| --- | --- | ---: | --- |
| Current one-shot, 64 tokens | Process/model setup + first forward + 48-token prefill + decode | `9.919145 tok/s` end-to-end | MEASURED |
| Warm prefill, persistent | Loaded persistent process, 48 prompt tokens | `17.791128 tok/s` | MEASURED |
| Warm decode, persistent | Warmup first, then token generation only | `17.877580 tok/s` median | MEASURED |
| Earlier scalar persistent comparison | Older scalar path and short diagnostic boundary | approximately `1.24 tok/s` reported in the incident | Not comparable to current release evidence |
| Historical one-shot result | Older process-per-run end-to-end workload | approximately `8.15 tok/s` | Historical evidence only |

The current one-shot record reports `first_real_forward_ms=139.606`,
`prefill_ms=2681.365`, `decode_wall_ms=3632.146`, and
`total_end_to_end_ms=6375`. The warm release-quality record excludes those
startup/prefill costs from its decode rate and requires one warmup plus seven
measured requests under PID `35928`.

The harness fix that made long persistent runs reliable was concurrent stderr
draining. `qwnrun` emits structured runtime detail after each request; the old
harness allowed that pipe to fill and hang. That was a harness deadlock, not a
decoder performance path. Streaming protocol framing remains byte-correct and
the token events are not buffered until completion.

## Cold startup and first real model access

Cold-start mode measures readiness only; it does not run a forward pass, so
forward, prefill, and decode fields are explicitly null there. The clean
record is:

| Field | ms |
| --- | ---: |
| `process_create_ms` | `3.417` |
| `file_open_ms` | `0.202` |
| `mmap_ms` | `0.077` |
| `metadata_parse_ms` | `0.113` |
| `tokenizer_init_ms` | `0.053` |
| `kv_cache_alloc_ms` | `39.659` |
| `advisory_preload_ms` | `0.063` |
| `first_tensor_touch_ms` | `45.103` |
| `first_real_forward_ms` | unavailable in cold-start-only mode |
| `prompt_prefill_ms` | unavailable in cold-start-only mode |
| `decode_ms` | unavailable in cold-start-only mode |
| `cold_start_ms` | `67.355` |
| `runtime_ready_ms` | `62.000` |

The persistent prefill run then measured `first_real_forward_ms=148.344` and
`prompt_prefill_ms=2697.974`. Therefore `mmap_ms` is only the mapping call; it
is not model-load completion and is not reported as such.

## HyperVSQ-2 dispatch decision

The dispatch decision is made at runtime for the exact 74-byte / 256-element
HyperVSQ-2 representation:

| Check | Clean result | Decision |
| --- | --- | --- |
| CPU supports AVX2/F16C/FMA | yes | eligible for AVX2 path |
| CPU supports AVX-VNNI | yes | eligible for VNNI path |
| Binary contains AVX2 kernel | yes | available |
| Binary contains VNNI kernel | yes | available |
| Model dtype is HyperVSQ-2 74-byte | yes | compatible |
| Dispatcher selected | `vnni` | actual executed kernel |
| Fallback reason | none | no scalar fallback in clean evidence |

The earlier scalar executable failed the compiled-kernel availability check;
OpenMP alone is not evidence that an ISA kernel exists. The runtime now keeps
scalar as the reference fallback, reports preferred candidate separately from
actual execution, and does not force an ISA through an environment variable.

The native differential suite completed `140/140` HyperVSQ-2 numerical tests.
It compares scalar reference results with the SIMD paths over synthetic and
real tensor samples within the test tolerances.

## Release-quality warm decode

Command:

```powershell
python -m benchmarks.benchmark_release_quality --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun_phase2.exe --backend cpu --threads 8 --context-size 4096 --max-tokens 64 --seed 0 --warmup-tokens 8 --repeats 7 --timeout 300 --output D:\EcoUni\qwanto-investigation\phase2-release-clean-e23c2a8-8.json
```

Raw summary output:

```text
classification: MEASURED
measured_runs: 7
generated_tokens_per_run: [64, 64, 64, 64, 64, 64, 64]
decode_tok_per_sec_median: 17.877580
decode_tok_per_sec_min: 17.584867
decode_tok_per_sec_max: 17.931275
decode_latency_ms_p95: 3639.493
ttft_ms_median: 2702.370
decode_tok_per_sec_cv: 0.007662274209327826
pid_reuse_proven: true
invalid_reasons: []
```

Runtime counters for the same report were: actual kernel `vnni`, active
workers `8`, HyperVSQ-2 matmul invocations `53760`, GPU matmuls `0`, CPU
fallbacks `0`, activation sums `precomputed`, precompute calls `53760`, reuse
count `464056320`, final LM-head calls `112`, final LM-head time `810.007 ms`,
intermediate LM-head calls `0`, early exits `0`, and layers skipped `0`.
Thermal and power sensors were unavailable and are not inferred.

## Thread scaling

All rows below use the same model, prompt, context, seed, warmup, and eight
generated measured tokens. The runtime-reported active worker count matched
the request for every row:

| Requested workers | Active workers | Decode tok/s | Classification |
| ---: | ---: | ---: | --- |
| 1 | 1 | `6.366276` | MEASURED |
| 2 | 2 | `10.322348` | MEASURED |
| 4 | 4 | `14.296633` | MEASURED |
| 8 | 8 | `18.191522` | MEASURED |
| 16 | 16 | `16.350098` | MEASURED |
| 32 | 32 | `15.669376` | MEASURED |

This is evidence of configuration reachability and a measured workload curve,
not a universal optimal-thread claim. No startup autotune was enabled.

## Activation-sum ablation

The same seven-request, same-PID, 64-token protocol was run at eight workers
with only activation-sum mode changed:

| Variant | Mode | Median decode tok/s | Median decode ms | CV | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | recomputed | `17.355390` | `3687.615` | `0.010976` | comparison baseline |
| Precompute | precomputed | `17.920965` | `3571.236` | `0.009302` | retained |

Precompute is retained because it won the same-configuration measured median
comparison. This is an ablation result, not a fixed performance guarantee.
Thread autotune, LM-head correction, delayed reductions, row blocking,
alternative two-bit unpacking, and SIMD SwiGLU remain explicitly
`UNAVAILABLE` in `ablation_summary.json`; no unsupported variant is presented
as active.

## Evidence files

The sanitized, checked-in evidence set is
[`benchmarks/evidence/windows/2026-08-17/e23c2a8`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/):

- [`release_quality_cpu.json`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/release_quality_cpu.json)
- [`cold_start.json`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/cold_start.json)
- [`prefill.json`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/prefill.json)
- [`one_shot.json`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/one_shot.json)
- [`thread_scaling.json`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/thread_scaling.json)
- [`ablation_summary.json`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/ablation_summary.json)
- [`hashes.sha256`](../benchmarks/evidence/windows/2026-08-17/e23c2a8/hashes.sha256)

The original raw Windows JSON files remain outside the repository under
`D:\EcoUni\qwanto-investigation` and were not staged as source or model data.

## Validation and blockers

Passed locally:

- `python -m pytest c/tests/ -q`: `232 passed, 4 skipped` before the final
  evidence-only changes; focused benchmark tests: `21 passed`.
- `python -m compileall -q benchmarks c/tests`.
- Clang syntax checks for the changed native sources.
- Clang/OpenMP linked Windows executable and `qwnrun --build-info --json`.
- HyperVSQ-2 differential test: `140/140 passed`.
- Web production build and Vitest: `56 passed`.
- `git diff --check`.

Not available on this workstation:

- `cargo check`, `cargo test`, and `cargo clippy`: Cargo is not installed.
- `make -C c test-c`: Make is not installed.
- CUDA correctness/performance: NVCC/CUDA runtime execution is unavailable;
  CUDA remains `UNAVAILABLE` and the detected RTX 5070 Ti is not treated as
  inference evidence.
- Hosted CI verification: GitHub was reporting an incident during this
  investigation, so no hosted run is claimed as green.

No README performance claims, release, or tag were changed by this phase.
