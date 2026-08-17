# CPU Phase A Local Ablation Matrix

Evidence classification: **MEASURED_LOCAL_PENDING_HOSTED_VALIDATION**. This report is not release-verified.

| Variant | Threads | Row block | Delayed reduction | Unpack | SwiGLU | Affinity | KV mode | Median tok/s | p5 tok/s | p95 latency | Correctness | Decision |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|---|
| Clean VNNI baseline | 8 | 1 | Disabled | shift/mask | exact scalar | OS default | fp16 | 17.764969 | 17.545289 | 3647.703 | scalar/VNNI differential passed | CONTROL |
| Delayed reduction | 8 | 1 | Enabled | shift/mask | exact scalar | OS default | fp16 | 18.985890 | 18.814534 | 3401.626 | 140/140 differential + exact stream agreement | VALIDATED_AND_ENABLED |
| 2-row HyperVSQ-2 blocking | 8 | 2 | Enabled | shift/mask | exact scalar | OS default | fp16 | 15.680256 | 15.594943 | 4103.894 | 140/140 differential | REJECTED_PERFORMANCE |
| 4-row HyperVSQ-2 blocking | 8 | 4 | Enabled | shift/mask | exact scalar | OS default | fp16 | 14.571689 | 14.514902 | 4409.262 | 140/140 differential | REJECTED_PERFORMANCE |
| Current 2-bit unpack | 8 | 1 | Enabled | shift/mask | exact scalar | OS default | fp16 | 18.985890 | 18.814534 | 3401.626 | unpack equality + 140/140 differential | VALIDATED_CURRENT_IMPLEMENTATION |
| SIMD SwiGLU candidate | 8 | 1 | Enabled | shift/mask | exact scalar | OS default | fp16 | 18.985890 | 18.814534 | 3401.626 | exact scalar reference; SIMD not adopted | VALIDATED_NOT_BENEFICIAL |
| OS-default affinity | 8 | 1 | Enabled | shift/mask | exact scalar | OS default | fp16 | 18.890454 | 18.272314 | 3502.567 | persistent PID and runtime counters | VALIDATED_NOT_BENEFICIAL |
| Combined production CPU path | 8 | 1 | Enabled | shift/mask | exact scalar | OS default | fp16 | 18.985890 | 18.814534 | 3401.626 | 140/140 differential + exact stream agreement | VALIDATED_AND_SELECTED |

## Attribution

Each row records the evidence path, executable/model SHA-256, source commit, dirty-worktree state, actual kernel, and execution counters in the machine-readable JSON.
The production path enables delayed reduction by default and retains `QWN_HYPERVSQ2_DISABLE_DELAYED_REDUCTION=1` only as a developer ablation override.

## Separate feature boundaries

- **kv_quantization:** Only typed fp16/auto is currently accepted; no validated long-context quantized mode.
- **speculative_decoding:** Runtime-wired draft/target verification and rollback evidence are not available.
- **cuda:** Intentionally out of scope; GPU matmul count remains zero.
