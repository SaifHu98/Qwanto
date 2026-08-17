# CPU Phase A Feature Status — 2026-08-17

This is local engineering evidence. Every performance record below is
`MEASURED_LOCAL_PENDING_HOSTED_VALIDATION`; it is not release-verified. CUDA,
README performance edits, tags, and releases were intentionally excluded.

| Feature | Final local status | Evidence and boundary |
|---|---|---|
| HyperVSQ-2 scalar reference | `VALIDATED` | `c/qwanto_kernels.c`; scalar differential reference is covered by the 140-case suite. |
| HyperVSQ-2 AVX2/FMA/F16C | `VALIDATED` | Runtime-safe dispatch and differential/kernel tests pass; actual model path selects VNNI on this host. |
| HyperVSQ-2 VNNI | `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION` | Final binary `4d3f3a4b...f3fb5e621`, model `43c128cd...2d29e36`, active 8 workers, actual kernel `vnni-delayed-reduction`. |
| Delayed VNNI reduction | `VALIDATED_AND_ENABLED` locally | Production default; `QWN_HYPERVSQ2_DISABLE_DELAYED_REDUCTION=1` is an explicit developer ablation override. 64-token median `18.619635` vs `17.319207` baseline; 128-token median `18.540185` vs `17.286976`. Counter is nonzero and streamed output agrees exactly. |
| HyperVSQ-2 multi-row blocking | `VALIDATED_NOT_BENEFICIAL` | Exact 74-byte candidates pass differential tests, but row-2 `15.193425` and row-4 `14.485682` tok/s are slower than the same-build delayed path; row-block counter is zero in production. |
| Alternative 2-bit unpacking | `VALIDATED_CURRENT_IMPLEMENTATION` | Shift/mask and LUT agree over 10,000 random samples; one-million-call diagnostic: shift/mask `0.9675 ms`, LUT `2.3925 ms`. LUT is not selected. Full model remains on current shift/mask path. |
| SIMD SwiGLU | `VALIDATED_NOT_BENEFICIAL` | Scalar exact path remains. Final diagnostic `swiglu_ms / hypervsq2_kernel_ms` is approximately `0.14%`; no approximation was promoted without material end-to-end benefit. |
| CPU affinity | `VALIDATED_NOT_BENEFICIAL` | OS-default wins repeated release-quality 64-token (`18.780863`) and 128-token (`18.827638`) controls over close/spread. No affinity policy is enabled by default. |
| Thread autotuning | `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION` | Opt-in tuner selected 8 workers from measured 64-token candidates. Cache key binds CPU topology, executable/model hashes, context class, and backend; it never runs at startup. |
| KV cache | `FP16_VALIDATED; quantized modes unavailable` | qwnrun accepts typed `fp16`/`auto`; no validated q8/q4 long-context evidence exists. |
| CUDA | `UNAVAILABLE` | No CUDA implementation was started. GPU matmul count is zero; GPU detection is not treated as inference. |
| Speculative decoding | `PROTOTYPE / disabled` | No validated draft QWN, probability correction, KV rollback, or end-to-end counters. |
| JetSpec | `REFERENCE_ONLY` | Placeholder scaffold remains disabled; synthetic telemetry is not emitted. |
| NVMe out-of-core | `NOT_REQUIRED` for resident 4B | Current model remains mmap/page-cache resident; no direct I/O is enabled or claimed. |

## Release-quality local measurements

| Workload | Delayed production median | p5 tok/s | p95 decode latency | Prefill median | PID reuse |
|---|---:|---:|---:|---:|---|
| 64 generated tokens | `18.619635` tok/s | `18.345278` | `3488.636 ms` | `18.403622` tok/s | proven |
| 128 generated tokens | `18.540185` tok/s | `18.347673` | `6976.362 ms` | `18.709167` tok/s | proven |

The matching same-binary delayed-disabled controls were `17.319207` tok/s and
`17.286976` tok/s respectively. These values are local pending-hosted evidence,
not README claims.

## Roofline correction

The final report uses the selected 8-worker aggregate read proxy of
`36,647,854,509.95 bytes/s` and executed logical total bytes of
`481,200,178.14 bytes/token`:

```text
36,647,854,509.95 / 481,200,178.14 = 76.159270 tok/s
```

This is `DERIVED_OR_UNAVAILABLE` in the hardware sense: memory-controller
traffic, process reads, cache misses, cycles, instructions, vector counters,
and OpenMP barrier time were unavailable. The stream benchmark uses an aligned
1 GiB buffer beyond the detected LLC and reports read-only, copy, and triad
separately. Logical bytes/token are execution counters, not DRAM traffic.

## Evidence paths

- Roofline JSON: `benchmarks/evidence/windows/2026-08-17/phase3-local/roofline-final-8t-64.json`
- Roofline report: `docs/cpu-roofline-analysis-2026-08-17.md`
- Final delayed 64/128: `benchmarks/evidence/windows/2026-08-17/phase3-local/final-delayed-8t-64.json`, `final-delayed-8t-128.json`
- Same-build baseline 64/128: `final-baseline-8t-64.json`, `final-baseline-8t-128.json`
- Row candidates: `final-row2-8t-64.json`, `final-row4-8t-64.json`
- Affinity matrices: `affinity-final/`, `affinity-final-128/`
- Autotune: `autotune-final.json` and `autotune-final-cache.json`
- Ablation: `benchmarks/evidence/windows/2026-08-17/phase3-local/phaseA-ablation-final.json`, `docs/cpu-phaseA-ablation-2026-08-17.md`
- Differential test: `c/tests/test_hypervsq2_kernels_phaseA_final.exe` — 140/140
