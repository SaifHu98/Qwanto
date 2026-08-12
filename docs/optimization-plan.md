# Optimization Plan: FeatherCore

This document outlines the prioritized engineering targets and stages for **FeatherCore** to improve inference throughput, memory efficiency, and responsiveness.

---

## 1. Prioritized Engineering Targets

### Target A: Reduce Time-to-First-Token (TTFT) & Disk Latency
- **Subsystem:** Memory Mapping & PILOT Prefetching.
- **Approach:** Optimize the predictive prefetch heuristics in `moe()` inside `glm.c`. Focus on tuning `PILOT` lookahead offsets based on active context depth.
- **Stage:** 1 (High priority).

### Target B: Minimize Avoidable Memory Overhead (Up to 50% reduction)
- **Subsystem:** Dynamic Expert Cache size adjustment.
- **Approach:** Refine `cap_for_ram()` calculations. If VRAM offloading is active, immediately free the equivalent host RAM cache slots to avoid double-residency of expert weights.
- **Stage:** 2 (Medium priority).

### Target C: Maximize End-to-End Inference Throughput (Stretch Target: 2x)
- **Subsystem:** CUDA & Metal Fused Kernels.
- **Approach:** Validate compilation paths in MSVC/MinGW to guarantee `ARCH=native` vectorization maps optimally to AVX-VNNI instructions for CPU paths, and force-offload MLM attention projections to GPU.
- **Stage:** 3 (Medium priority).

---

## 2. Implementation Methodology & Guidelines

- **Subsystem Isolation:** Work in independent, isolated branches. Avoid global refactors.
- **Correctness First:** Before declaring any speed improvement, run `make test-c` to verify that the output remains token-exact with the reference transformer oracle.
- **Incremental Benchmarking:** Validate each stage against `docs/performance-baseline.md`.
- **Rejection Criteria:** Immediately rollback and discard any optimization if it results in:
  1. Output token mismatches (divergence from the oracle).
  2. Memory leaks or RSS climbing beyond bounds.
  3. Deadlocks in multi-threaded loops (OMP/pthreads).
  4. Corruption of the persisted KV-cache.
