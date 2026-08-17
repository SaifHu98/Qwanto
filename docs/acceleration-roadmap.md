# Qwanto Native Acceleration Roadmap

Status is tracked independently for each runtime capability. A feature is not
called active from configuration alone: runtime counters, correctness tests,
and reproducible evidence must prove execution. Performance evidence generated
locally remains `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION` until the exact final
commit passes the complete hosted workflow.

## Baseline

- CPU Phase A evidence source commit: `9a686913d33e9621c95955b53e8b9e980cb01456`.
  The CUDA follow-up changes native sources after that evidence boundary; a
  clean CPU evidence regeneration is required before treating Phase A as
  current. Hosted validation remains pending for the final follow-up commit.
- Native model: `experiments/results/4B_hyper_vsq2.qwn`, HyperVSQ-2 QWN 2.31,
  74-byte blocks containing 256 values.
- Model SHA-256:
  `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`.
- Current CPU path: VNNI, eight active workers, FP16 KV cache, zero GPU
  matmuls and zero CPU fallbacks from the measured local path.
- Final local Phase A binary: `c/qwnrun_phaseA_final.exe`, SHA-256
  `4d3f3a4b9eca86023b49056439298e7333f0d53f74a70af935c3e9f3fb5e621`.
- Final local production path: delayed VNNI reduction, OS-default affinity,
  eight active workers, FP16 KV, zero GPU matmuls. 64-token median is
  `18.985890 tok/s`; 128-token median is `18.945001 tok/s`.
- Local filtered CI for the pushed CPU work: [32049440479](https://github.com/SaifHu98/Qwanto/actions/runs/32049440479)
  and [32049660230](https://github.com/SaifHu98/Qwanto/actions/runs/32049660230).
  Native/Python/security/docs checks passed where selected; Rust and Web were
  skipped by path filters. Full dispatch was attempted but the unauthenticated
  API returned HTTP 401. This is not complete hosted validation.

## Feature status

| Feature | Current status | Target status | Implementation commit | Correctness evidence | Benchmark evidence | Enabled by default | Unresolved risks |
|---|---|---|---|---|---|---|---|
| HyperVSQ-2 scalar/AVX2/VNNI CPU path | VNNI measured locally | Hosted-validated CPU path | `e23c2a8` and later | `c/tests/test_hypervsq2_kernels.c`, 140/140 | `docs/cpu-performance-phase2-2026-08-17.md` | Yes, auto-dispatch | Host-specific scaling and profiler counters remain limited |
| Delayed VNNI reduction | `VALIDATED_AND_ENABLED` locally; hosted pending | Hosted-validated safe VNNI dispatch | `cb3ca35` plus `9a68691` | 140/140 differential, exact stream agreement, nonzero invocation counter | `phaseA-clean-9a68691/final-delayed-8t-64.json`, `final-delayed-8t-128.json` | Yes | Hosted full workflow and final docs/evidence commit remain required |
| HyperVSQ-2 multi-row blocking | `VALIDATED_NOT_BENEFICIAL` | Keep development candidates for future shape-specific work | `9a68691` | 140/140 differential; row-2/row-4 full-model regressions | `phaseA-clean-9a68691/final-row2-8t-64.json`, `final-row4-8t-64.json` | No | Register pressure and shape-specific variants were slower end-to-end |
| Alternative 2-bit unpacking | `VALIDATED_CURRENT_IMPLEMENTATION` | Keep current shift/mask path | `9a68691` | 10,000 random unpack equalities plus 140/140 differential | `phaseA-clean-9a68691/phaseA-ablation-final.json`; native test output | Yes, current implementation | LUT is slower in the diagnostic and not used end-to-end |
| SIMD SwiGLU | `VALIDATED_NOT_BENEFICIAL` | Retain exact scalar behavior | `9a68691` | Exact scalar path; measured contribution approximately 0.14% of HyperVSQ kernel time | `phaseA-clean-9a68691/final-delayed-8t-64.json` | No fast approximation | No material end-to-end target justifies approximation risk |
| CPU affinity/autotuning | `VALIDATED_NOT_BENEFICIAL` for affinity; opt-in tuner measured | OS-default scheduling; cache opt-in measurements | `9a68691` | Repeated 64/128 policy matrices; cache-keyed autotune selected 8 | `phaseA-clean-9a68691/affinity-final/`, `affinity-final-128/`, `autotune-final.json` | OS default only | Do not run long autotune at startup |
| HyperVSQ-2 CUDA | `END_TO_END_VALIDATED` locally; hosted validation pending | `COMPILED` → `KERNEL_CORRECT` → `END_TO_END_VALIDATED` → `MEASURED` | ABI v1 synthetic test; scalar/VNNI real-model decoder comparison; zero required-layer CPU fallbacks | `cuda-phaseB-clean-4d26cdc/cuda-release-short-diagnostic.json`; `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION` | No | Release-quality CUDA repetitions, hosted gates, and a clean evidence commit remain required |
| Typed quantized KV cache | FP16/auto only; TurboQuant env scaffold | Typed validated fp16/q8/q4 modes as implemented | — | No quantized KV runtime contract | None | FP16 only | Long-context error, memory, and quality validation |
| Speculative decoding | Prototype scaffold; CLI rejects it | `IMPLEMENTED_REQUIRES_DRAFT_MODEL` or measured | — | No complete distribution/KV rollback evidence | None | No | Draft QWN compatibility and probability correction |
| JetSpec | Reference-only scaffold with placeholders | Real speculative algorithm or disabled reference-only | — | None | None | No | Synthetic token generation and placeholder telemetry must remain absent |
| NVMe out-of-core | `NOT_REQUIRED` for resident 4B model | `EXPERIMENTAL_OUT_OF_CORE` only for non-fitting models | — | Planner/prefetch tests pending | None | No | Page cache, I/O overlap, alignment, and memory safety |
| Qwanto Code runtime integration | UI exposes truthful unavailable states | Accepted typed features with actual counters | Existing desktop/web work | Desktop/runtime tests | Runtime telemetry evidence | Only supported defaults | UI must never infer Active from requested config |

## Execution policy

Phase A completes the CPU candidates and either promotes or rejects them.
Phase B has started with the correctness-first versioned ABI and exact
HyperVSQ-2 reference path. It must remain independently validated; no CUDA
performance claim is promoted until the clean evidence commit passes hosted
checks. Phases C–F remain independently committed.
README performance tables are not
updated until accepted evidence is regenerated from a clean final commit and
the complete hosted workflow is green. No release or tag is created by this
roadmap work.
