# Qwanto Native acceleration roadmap

This roadmap records implementation state, not targets. A feature becomes
active only after its typed runtime path executes and its counters prove that
execution. Local measurements are `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION`
until the exact final commit passes the complete hosted workflow.

## Preserved baseline

- Qualification commits and evidence remain in history: `a1984025880cd2de41c442472f1b2c951b882b5f` and `54942491159ed2bc8730abd581311c8d9b6cc515`.
- Qwen3.8-27B has a local Q4_0 QWN output and CPU main-path integration proof;
  MTP/MoE/CUDA/quality/benchmark gates remain open; native IQ tensor-row
  decoding is now verified independently but does not complete Flash-Next.
- Validated implementation model: `experiments/results/4B_hyper_vsq2.qwn`, SHA-256 `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`.
- CPU Phase A is the VNNI delayed-reduction path with OS-default affinity, eight workers, and FP16 KV. Its evidence remains local pending hosted validation.

## Status matrix

| Capability | Current status | Enabled by default | Correctness evidence | Performance evidence | Remaining gate |
|---|---|---:|---|---|---|
| HyperVSQ-2 CPU scalar/AVX2/VNNI | `VALIDATED` | Yes, safe dispatch | 140/140 differential | CPU Phase A evidence | Full hosted validation on final source |
| VNNI delayed reduction | `VALIDATED_AND_ENABLED_LOCALLY_PENDING_HOSTED` | Yes for compatible VNNI | Exact stream agreement; nonzero invocation counter | `phaseA-clean-9a68691/` | Regenerate if runtime changes; hosted gate |
| Multi-row blocking | `VALIDATED_NOT_BENEFICIAL` | No | Differential tests pass | Slower full-model variants | None for current product path |
| Alternative unpacking | `VALIDATED_CURRENT_IMPLEMENTATION` | Current shift/mask | Random and differential equality | Alternatives slower end-to-end | None for current product path |
| SIMD SwiGLU approximation | `VALIDATED_NOT_BENEFICIAL` | No | Exact scalar retained | Contribution not material | None for current product path |
| CPU affinity override | `VALIDATED_NOT_BENEFICIAL` | No; OS default | Safe policy tests | OS default won repeated runs | None for current product path |
| Typed FP16 KV | `VALIDATED` | Yes | Existing decoder path | Existing CPU evidence | Long-context evidence remains separate |
| Typed Q8 KV | `ATTENTION_CORRECT` locally | No | Scalar cache and CUDA Q8 oracle | Not release measured | 512/4K/16K quality and hosted gates |
| QWN-Q4-KV compatibility cache | `REFERENCE_IMPLEMENTED` | No | Existing scalar compatibility tests | No product claim | Exact TurboQuant equivalence is not claimed |
| CUDA Q8 KV reference | `KERNEL_CORRECT` locally | No | `max_abs_error=1.1920929e-7` on RTX 5070 Ti | Not measured | Full model coverage and hosted compile gate |
| HyperVSQ-2 CUDA decoder | `END_TO_END_VALIDATED` locally | No | Scalar/VNNI comparison, zero fallbacks | Diagnostic only | Reproduce from final commit; no README claim |
| Correct speculative decoding | `IMPLEMENTED_REQUIRES_COMPATIBLE_DRAFT_MODEL` | No | Probability/transaction code is gated | None | Compatible native QWN draft and quality evidence |
| JetSpec tree speculation | `REFERENCE_ONLY` | No | Deterministic fixture structure tests | None | Draft/MTP tree decoder and distribution gates |
| NVMe out-of-core | `NOT_REQUIRED_FOR_RESIDENT_4B` | No | No direct-I/O path enabled | None | Only for models that do not fit RAM |
| Qwen3.8 hybrid runtime | `CPU_MAIN_PATH_INTEGRATION_VERIFIED` | No | Real local Q4_0 one-token run | None | MTP/native IQ/MoE/CUDA/quality oracle and benchmark gates |

## Evidence and safety rules

The current CPU and CUDA evidence is local and pending a full hosted run on
the final exact commit. A detected GPU, loaded DLL, isolated cache kernel, or
requested backend is not full-model CUDA evidence. Explicit CUDA fails closed
if required projection or cache coverage is unavailable; auto mode records the
reason and may remain on CPU.

TurboQuant is not used as a label for the current arbitrary Q4 cache. The
runtime reports `QWN-Q4-KV` until the cited algorithm is independently matched.
Speculation and JetSpec do not initialize acceptance or speedup counters and
cannot be enabled by an environment variable or UI flag.

## Sequential implementation boundary

Phase 1 establishes the typed KV contract and reference CPU/CUDA Q8 cache.
Phase 2 establishes correct draft/target probability and rollback wiring but
remains disabled without a compatible native draft. Phase 3 keeps JetSpec
reference-only until it can use that transaction engine with real proposals.
The converter capability axes and Qwen3.8 architecture work are separate
follow-up milestones. No README performance claim, tag, or release is changed
by this roadmap.
