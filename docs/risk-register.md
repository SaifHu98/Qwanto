# Risk Register: FeatherCore Optimizations

This register tracks architectural and runtime risks introduced by proposed optimizations in **FeatherCore**.

---

## 1. Active Risk Matrix

| Risk ID | Description | Likelihood | Impact | Mitigation / Fallback Path |
|---|---|---|---|---|
| **R01** | **Float Point Rounding Discrepancy**<br>Using fused math kernels or parallel GPU threads flips marginal token outputs. | High | Medium | Provide `DRAFT=0` (disabling MTP) and `IDOT=0 COLI_CUDA=0` switches to fallback to portable C execution. |
| **R02** | **Memory Swapping / OOM Spills**<br>Aggressive RAM caching on 4GB systems exhausts memory, triggering the OS OOM killer. | Medium | High | Rely on `resource_plan.py` strict budget clamping (< 1.5 GB cache) and enforce `COLI_MMAP=1`. |
| **R03** | **Thread Deadlocks under Load**<br>Concurrently streaming SSE client requests causes race conditions inside OMP loops or background I/O threads. | Low | High | Use atomic variables (`_Atomic`) and lock-free thread queues where possible. Run concurrent curl tests. |
| **R04** | **KV Cache Disk Corruption**<br>Abrupt program termination during KV persistence serializes corrupted `.coli_kv` states. | Medium | Low | Write to a temporary file (`.coli_kv.tmp`) first, and perform an atomic rename on success. |
| **R05** | **Windows I/O Latency Spikes**<br>Windows synchronous `ReadFile` calls block the main execution thread when experts miss cache. | High | Medium | Ensure the PILOT readahead thread is active and pre-populating page caches via `compat_fadvise`. |
