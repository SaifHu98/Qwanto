# CPU Phase 3 Feature Status — 2026-08-17

This is a local engineering audit. Performance entries are classified **MEASURED_LOCAL_PENDING_HOSTED_VALIDATION** and are not release-verified. No CUDA implementation was started in this phase.

| Feature | Status | Evidence and boundary |
|---|---|---|
| HyperVSQ-2 scalar reference | VALIDATED | `c/qwanto_kernels.c`; differential suite passed 140/140 cases. |
| HyperVSQ-2 AVX2/FMA/F16C | VALIDATED | `c/qwanto_kernels.c`; differential suite and kernel benchmark execute the AVX2 candidate. |
| HyperVSQ-2 VNNI | MEASURED_LOCAL_PENDING_HOSTED_VALIDATION | The current 74-byte / 256-element path selected `vnni` on the local AMD host; release-quality evidence records the executable/model hashes and active workers. |
| Delayed VNNI reduction | MEASURED_LOCAL_PENDING_HOSTED_VALIDATION | Development-gated by `QWN_HYPERVSQ2_DELAYED_REDUCTION=1`; per-octant scale/offset semantics are retained, exact streamed output agrees with baseline, and 140/140 differential tests pass. |
| HyperVSQ-2 CUDA | UNAVAILABLE | CUDA was intentionally not implemented. Local evidence has zero GPU matmuls; hardware detection is not treated as CUDA execution. |
| HyperVSQ-2 multi-row blocking | UNAVAILABLE | No candidate was retained for the real 74-byte layout; generic row blocking is not evidence for this kernel. |
| Alternative 2-bit unpacking | UNAVAILABLE | The current AVX2 unpack path is tested, but no shuffle/LUT alternative passed full GEMV and end-to-end acceptance. |
| SIMD SwiGLU | UNAVAILABLE | Scalar `expf` remains the implementation. SwiGLU timing is instrumented, but no bounded approximation has passed the required numerical and end-to-end gates. |
| CPU affinity close/spread | EXPERIMENTAL | Local close/spread measurements exist in `phase3-local`; no production default was changed because there is no repeated OS-default control series yet. |
| KV cache FP16 | MEASURED_LOCAL_PENDING_HOSTED_VALIDATION | `qwnrun` accepts `fp16`/`auto` and current 4B local evidence uses FP16 KV. |
| TurboQuant KV cache | PROTOTYPE | `c/qwanto_turboquant.c` contains kernels and `c/qwanto_decode.c` has an environment-gated path, but typed runtime configuration accepts only FP16/auto and no 512/4K/16K comparative validation exists. It is not an integrated product feature. |
| Speculative decoding | PROTOTYPE | `c/qwanto_speculative.c` contains draft/verify scaffolding, but qwnrun rejects `--speculative`; draft-model compatibility, probability correction, robust rollback/commit, runtime wiring, and end-to-end counters are not proven. |
| JetSpec | REFERENCE_ONLY | `c/qwanto_jetspec.c` contains a tree scaffold with synthetic token generation and placeholder initialized metrics; it is not called by qwnrun and has no measured end-to-end evidence. |
| NVMe direct-I/O tiering | REFERENCE_ONLY | Current resident 4B execution remains mmap/page-cache based. Direct I/O is not enabled or claimed for this model. |

## Speculative-decoding readiness audit

The source contains a target/draft pipeline shape, cache helpers, acceptance accounting, and position rollback assignments. That is not sufficient for a product-ready feature. The current implementation does not provide a verified draft tokenizer/model pair, batched target verification with a tested probability correction rule, complete KV-cache commit/rollback semantics, qwnrun runtime wiring, or counters for proposed/accepted/rejected tokens, draft time, verification time, rollback time, baseline throughput, and net throughput. Consequently the feature remains `PROTOTYPE` and is rejected by the runtime configuration parser.

JetSpec is even earlier: its tree expansion derives draft token IDs from the parent token and depth, and its initialization seeds placeholder acceptance/speedup values. Those values are not measurements and are intentionally excluded from all performance evidence.

## KV-cache audit boundary

The runtime validation in `c/qwn_runtime_config.c` accepts only `fp16` and `auto`; other KV modes fail explicitly. The `QWN_TURBOQUANT` environment path is not equivalent to a typed, reproducible runtime configuration. No claim is made for TurboQuant, Q4/Q3, 4K/16K context savings, attention bandwidth, logit error, or long-context quality until those tests exist.

## Evidence paths

- Roofline: `benchmarks/evidence/windows/2026-08-17/phase3-local/roofline-8t-64.json`
- Release-quality VNNI baseline: `benchmarks/evidence/windows/2026-08-17/phase3-local/baseline-vnni-8t-64-rebuilt.json` and `baseline-vnni-8t-128-rebuilt.json`
- Delayed reduction: `benchmarks/evidence/windows/2026-08-17/phase3-local/delayed-reduction-8t-64-rebuilt.json` and `delayed-reduction-8t-128-rebuilt.json`
- Output agreement: `benchmarks/evidence/windows/2026-08-17/phase3-local/delayed-reduction-correctness.json`
- Ablation matrix: `benchmarks/evidence/windows/2026-08-17/phase3-local/phase3-ablation.json`
