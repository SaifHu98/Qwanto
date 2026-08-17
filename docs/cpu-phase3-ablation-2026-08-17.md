# CPU Phase 3 Local Ablation Matrix

Evidence classification: **MEASURED_LOCAL_PENDING_HOSTED_VALIDATION**. This report is not release-verified.

| Variant | Threads | Row block | Delayed reduction | Unpack | SwiGLU | Affinity | KV mode | Median tok/s | p5 tok/s | p95 latency | Correctness |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|
| Current clean VNNI baseline | 8 | 1 | No | current AVX2/VNNI | scalar exact | default | fp16 | 17.981481 | 17.897881 | 3575.842 | scalar/VNNI differential passed |
| Delayed reduction | 8 | 1 | Yes | current AVX2/VNNI | scalar exact | default | fp16 | 18.877406 | 18.704164 | 3421.698 | 140/140 differential + exact stream agreement |
| Affinity close | 8 | 1 | No | current AVX2/VNNI | scalar exact | close | fp16 | 17.186347 | 16.878935 | 3791.708 | not applicable |
| Affinity spread | 8 | 1 | No | current AVX2/VNNI | scalar exact | spread | fp16 | 16.716366 | 16.108747 | 3972.997 | not applicable |
| 2-row blocking | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: No HyperVSQ-2 multi-row candidate retained; generic row blocking is not evidence for this layout. |
| 4-row blocking | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: No HyperVSQ-2 multi-row candidate retained; generic row blocking is not evidence for this layout. |
| Alternative 2-bit unpack | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: No shuffle/LUT candidate completed full GEMV and end-to-end validation. |
| SIMD SwiGLU | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: No SIMD candidate; scalar SwiGLU time is instrumented but no validated approximation was adopted. |
| Combined validated CPU kernel | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: No combined row/unpack/SwiGLU winner exists; delayed reduction remains separately attributable. |
| KV quantization long-context | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: Only fp16/auto is accepted by qwnrun; 512/4K/16K comparative KV evidence is not available. |
| Speculative decoding | — | — | — | — | — | — | — | — | — | — | UNAVAILABLE: No runtime-wired draft/target verification, correction, rollback, and end-to-end counters. |

## Attribution

Each measured row records the evidence path, executable/model SHA-256, commit, and dirty-worktree state in the machine-readable JSON. The delayed-reduction candidate is enabled only by `QWN_HYPERVSQ2_DELAYED_REDUCTION=1`; the default VNNI path remains unchanged.
