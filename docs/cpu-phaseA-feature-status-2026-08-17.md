# CPU Phase A Feature Status — 2026-08-17

## Scope and evidence classification

This document is the canonical closeout for the local CPU Phase A work. It
covers the exact HyperVSQ-2 QWN 2.31, 74-byte / 256-element path and the
production-default VNNI delayed-reduction dispatch. Performance records are
`MEASURED_LOCAL_PENDING_HOSTED_VALIDATION`: they were generated from a clean
source commit, but the final exact commit still requires the complete hosted
CPU workflow. They are not release-verified and are not README claims.

## Final CPU Phase A status

| Feature | Final status | Evidence boundary |
|---|---|---|
| HyperVSQ-2 scalar reference | `VALIDATED` | 140-case scalar differential suite. |
| HyperVSQ-2 AVX2/FMA/F16C | `VALIDATED` | Runtime-safe compiled/runtime feature dispatch and differential coverage. |
| VNNI delayed reduction | `VALIDATED_AND_ENABLED_LOCALLY_PENDING_HOSTED` | Default for compatible HyperVSQ-2 VNNI execution; exact streamed output agreement and nonzero invocation counter. |
| Multi-row blocking | `VALIDATED_NOT_BENEFICIAL` | Exact 74-byte candidates were slower end-to-end than the selected single-row path. |
| Alternative 2-bit unpack | `VALIDATED_CURRENT_IMPLEMENTATION` | Current shift/mask implementation beat the tested LUT candidate end-to-end and in the unpack diagnostic. |
| SIMD SwiGLU approximation | `VALIDATED_NOT_BENEFICIAL` | Scalar exact math remains selected; activation time was not material to full decode. |
| CPU affinity override | `VALIDATED_NOT_BENEFICIAL` | OS-default scheduling won the clean release-quality comparisons. |
| Thread autotuner | `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION` | Opt-in only; clean evidence selected 8 workers and caches the decision by host/executable/model/context identity. |
| FP16 KV cache | `VALIDATED` | Current supported KV mode. |
| Quantized KV cache | `UNAVAILABLE` | No typed, correctness-validated q8/q4/q3 runtime path. |
| CUDA | `UNAVAILABLE` | No compiled or executed versioned CUDA DLL in the CPU evidence; GPU matmul count is zero. |
| Speculative decoding | `PROTOTYPE_DISABLED` | No compatible draft model and complete acceptance/rollback evidence. |
| JetSpec | `REFERENCE_ONLY` | Source references do not constitute an active runtime algorithm. |
| NVMe out-of-core | `NOT_REQUIRED_FOR_RESIDENT_4B` | The resident 4B workload remains mmap/page-cache based; direct I/O is not enabled. |

## Release-quality local performance

All rows use the clean Phase A executable/model identity recorded below, one
warmup request plus seven measured persistent requests under one PID, fixed
prompt/configuration, VNNI delayed reduction, 8 workers, OS-default affinity,
and FP16 KV.

| Workload | Median decode tok/s | P5 tok/s | P95 decode latency | Median prefill tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| 64 generated tokens | 18.985890 | 18.814534 | 3401.626 ms | 18.919199 | Exact streamed agreement |
| 128 generated tokens | 18.945001 | 18.726433 | 6835.258 ms | 19.176576 | Exact streamed agreement |

Same-build delayed-disabled controls were 17.764969 tok/s for 64 tokens and
17.737912 tok/s for 128 tokens. The differential suite passed 140/140 cases.
There were zero GPU matmuls and these records contain no CUDA performance
claim.

## Roofline boundary

The corrected proxy equation is:

```text
34,421,311,165.34 bytes/s
/ 481,038,007.52 logical bytes/token
= 71.556323 derived tok/s
```

Its classification is exactly
`DERIVED_PROXY_NOT_HARDWARE_MEASURED`. The numerator is an independent aligned
stream read proxy selected at 8 workers; the denominator comes from Qwanto
descriptor/execution counters. It is not DRAM-controller traffic. Cache reuse,
re-reads, synchronization, quantization work, attention, sampling, and other
decoder work prevent treating 71.556323 tok/s as an expected or achievable
runtime result. No trustworthy hardware memory-controller counter was available
on the host. The schema validator recomputes the equation from the JSON inputs.

## Rejected experiments and reasons

- Multi-row 2/4/8 candidates were correctness-safe but slower on the complete
  model, so no row-block variant is enabled.
- The LUT unpack candidate was correct but slower than the current shift/mask
  implementation; the current implementation is the validated winner.
- SIMD SwiGLU approximation was not promoted because scalar `expf` was a
  negligible fraction of measured decode and no end-to-end win was shown.
- Close/spread affinity overrides did not beat OS-default scheduling and are
  not enabled by default.
- Quantized KV, CUDA, speculative decoding, JetSpec, and NVMe tiering have no
  release-quality implementation/evidence in this phase.

## Remaining accelerator roadmap

The next accelerator phase is the versioned HyperVSQ-2 CUDA ABI and exact
74-byte kernel. It must progress through `UNAVAILABLE`, `COMPILED`,
`KERNEL_CORRECT`, `END_TO_END_VALIDATED`, and `MEASURED`; a detected GPU or
loaded DLL is not sufficient. TurboQuant, speculative decoding, JetSpec, and
out-of-core NVMe remain explicitly out of scope for this closeout.

## Evidence paths and identity

- Clean evidence source commit: `9a686913d33e9621c95955b53e8b9e980cb01456`.
- Current documentation/recording commit: `6fd30075e51bc398be68b07bd8b477b0810278e2`.
- Previous pushed CPU main: `182bc56`; the clean source commit exists in the
  current history. At the Phase A evidence boundary (`9a68691` to `6fd3007`)
  no native runtime diff existed. This CUDA follow-up intentionally changes
  native loader/runtime sources after that boundary, so the Phase A evidence
  remains bound to its recorded executable/source identity until a new clean
  CPU regeneration is completed.
- Worktree and `origin/main` were clean/equal at the start of this follow-up,
  apart from preserved user-owned local artifacts outside the commit scope.
- Executable SHA-256: `4d3f3a4ab9eca86023b49056439298e7333f0d53f74a70af935c3e9f3fb5e621`.
- Model SHA-256: `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`.
- Evidence directory: `benchmarks/evidence/windows/2026-08-17/phaseA-clean-9a68691/`.
- Roofline JSON: `benchmarks/evidence/windows/2026-08-17/phaseA-clean-9a68691/roofline-final-8t-64.json`.
- Ablation JSON/report: `benchmarks/evidence/windows/2026-08-17/phaseA-clean-9a68691/phaseA-ablation-final.json` and `docs/cpu-phaseA-ablation-2026-08-17.md`.
- Differential test binary: `c/tests/test_hypervsq2_kernels_phaseA_final.exe` (140/140).

## Hosted validation status

The push-triggered run `32056218272`
([Actions run](https://github.com/SaifHu98/Qwanto/actions/runs/32056218272))
passed the scheduled Security, Docs, Python, Native C Linux, and Native C
Windows jobs, but path filters skipped Web and Rust/Tauri. An unauthenticated
full `workflow_dispatch` attempt returned HTTP 401, so it is not evidence of
success. Full hosted CPU validation remains pending and must be run manually
from the Actions UI on the final exact commit before this phase is marked
complete or any README performance table is changed.
