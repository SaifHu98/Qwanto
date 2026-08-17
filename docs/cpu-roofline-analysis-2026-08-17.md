# CPU Roofline Analysis — 2026-08-17

This report is generated from `D:\EcoUni\qwanto\c\qwnrun_phaseA_final.exe` and the measured evidence file. It is local evidence and remains **MEASURED_LOCAL_PENDING_HOSTED_VALIDATION** until the exact final runtime commit passes hosted validation. It is not a release-quality claim.

## Reproduction identity

| Field | Value |
|---|---|
| Git commit | `182bc56e62ee565e505fbea2c493518577fd92ef` |
| Worktree dirty during measurement | `True` |
| Executable SHA-256 | `4d3f3a4ab9eca86023b49056439298e7333f0d53f74a70af935c3e9f3fb5e621` |
| Model SHA-256 | `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36` |
| Host | Windows-11-10.0.26200-SP0 |
| CPU | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD |
| Active workers | 8 |
| Selected kernel | vnni-delayed-reduction |
| OpenMP compiled/runtime loaded | True / True |

The model file is **1.179 GiB** with **1.179 GiB** of mapped tensor payload across 443 tensors. The dominant dtype is `HYPER_VSQ2`.

## Independent bandwidth measurement

The benchmark used a 1024 MiB aligned NumPy buffer, 5 measured repetitions after 2 warmups, and read-only, copy, and triad workloads. Values below are independent stream-like measurements, not qwnrun hardware-counter readings. A bandwidth row is selected at the runtime worker count; rows from other worker counts are not substituted into the equation.

| Workers | Read median GB/s | Read p5 GB/s | Copy median GB/s | Triad median GB/s |
|---:|---:|---:|---:|---:|
| 1 | 10.550 | 9.818 | 46.434 | 9.554 |
| 2 | 20.637 | 15.609 | 47.281 | 13.025 |
| 4 | 33.780 | 32.679 | 47.361 | 15.190 |
| 6 | 35.516 | 34.004 | 46.384 | 16.175 |
| 8 | 36.648 | 36.249 | 47.661 | 16.468 |
| 12 | 37.264 | 36.104 | 53.016 | 16.677 |
| 16 | 40.650 | 38.175 | 52.965 | 15.834 |
| 32 | 44.379 | 42.137 | 54.326 | 15.802 |

The selected worker count was **8**. Its read-only aggregate rate was **36.648 GB/s** (**34.131 GiB/s**). These are independent bandwidth proxies, not measured memory-controller bandwidth.

## Arithmetic intensity and roofline estimate

| Quantity | Value | Source / limitation |
|---|---:|---|
| Total logical bytes per token | 481200178.137 | derived_from_qwn_logical_execution_counters from executed logical counters |
| Logical FLOPs per token | 3206021120.000 | qwnrun_matmul_shape_counter from descriptor-derived operations |
| Arithmetic intensity | 6.662552 FLOP/byte | derived estimate |
| Predicted throughput | 76.159 tok/s | derived_estimate using selected aggregate bytes/s |
| Actual persistent decode median | 18.876335 tok/s | qwnrun_release_quality_persistent_decode |
| Actual / predicted estimate | 24.79% | derived comparison, not a hardware efficiency measurement |

Equation inputs: `36647854509.94771` bytes/s ÷ `481200178.13685477` bytes/token = `76.1592704554755` tok/s. The machine-readable validator recomputes this equation and rejects inconsistent evidence.

The logical bytes/token value is not a process read counter. It assumes the descriptor-derived logical traffic and therefore cannot prove that every byte was fetched from DRAM. The predicted throughput excludes decoder overhead, cache reuse, synchronization, sampling, and page/cache effects. It is a roofline estimate only.

## Time and counter evidence

| Counter | Result | Source |
|---|---:|---|
| HyperVSQ-2 kernel time | 11825.441 ms aggregate | qwnrun_hypervsq2_wall_timer |
| SwiGLU time | 16.304 ms aggregate | qwnrun_swiglu_wall_timer |
| Prefill median | 2538.690 ms | qwnrun_generation_metrics_prefill_boundary |
| Decode median | 3390.489 ms | qwnrun_generation_metrics_decode_boundary |
| Process memory reads | None | unavailable |
| Memory-controller bandwidth | None | unavailable |
| L1/L2/L3 misses | None | unavailable |
| CPU cycles/instructions/vector instructions | unavailable | no supported hardware profiler configured |
| OpenMP synchronization time | unavailable | barrier timing is not separately instrumented |

## Interpretation

The measured local decode result is **18.876335 tok/s median** with p5 **18.802478 tok/s**. The independent bandwidth run does not justify assuming 40–60 GB/s, and the derived roofline must not be presented as measured hardware bandwidth. The current evidence supports a CPU VNNI path and a memory-sensitive workload, but it does not identify every decoder bottleneck; profiler-backed cache, cycles, instructions, and memory-controller counters remain unavailable on this host.

## Source evidence

- Machine-readable evidence: the path supplied to the renderer
- Harness: `benchmarks/benchmark_cpu_roofline.py`
- Runtime benchmark: `benchmarks/benchmark_release_quality.py`
- Model format: `docs/qwn-format.md`
