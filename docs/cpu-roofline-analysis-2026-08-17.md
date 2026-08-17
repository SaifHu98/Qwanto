# CPU Roofline Analysis — 2026-08-17

This report is generated from `D:\EcoUni\qwanto\c\qwnrun_phaseA_final.exe` and the measured evidence file. It is local evidence and remains **MEASURED_LOCAL_PENDING_HOSTED_VALIDATION** until the exact final runtime commit passes hosted validation. It is not a release-quality claim.

## Reproduction identity

| Field | Value |
|---|---|
| Git commit | `9a686913d33e9621c95955b53e8b9e980cb01456` |
| Worktree dirty during measurement | `False` |
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
| 1 | 9.174 | 8.700 | 41.654 | 8.063 |
| 2 | 19.770 | 17.493 | 45.336 | 12.246 |
| 4 | 34.993 | 34.058 | 46.200 | 14.759 |
| 6 | 33.687 | 32.126 | 45.076 | 15.431 |
| 8 | 34.421 | 28.610 | 47.643 | 16.371 |
| 12 | 37.980 | 36.183 | 51.514 | 16.004 |
| 16 | 39.622 | 36.143 | 51.976 | 15.691 |
| 32 | 44.608 | 43.047 | 52.817 | 15.328 |

The selected worker count was **8**. Its read-only aggregate rate was **34.421 GB/s** (**32.057 GiB/s**). These are independent bandwidth proxies, not measured memory-controller bandwidth.

## Arithmetic intensity and roofline estimate

| Quantity | Value | Source / limitation |
|---|---:|---|
| Total logical bytes per token | 481038007.515 | derived_from_qwn_logical_execution_counters from executed logical counters |
| Logical FLOPs per token | 3206021120.000 | qwnrun_matmul_shape_counter from descriptor-derived operations |
| Arithmetic intensity | 6.664798 FLOP/byte | derived estimate |
| Predicted throughput | 71.556 tok/s | `DERIVED_PROXY_NOT_HARDWARE_MEASURED`; derived only from selected aggregate stream proxy |
| Actual persistent decode median | 18.463467 tok/s | qwnrun_release_quality_persistent_decode |
| Actual / predicted estimate | 25.80% | derived comparison, not a hardware efficiency measurement |

Equation inputs: `34421311165.34105` bytes/s ÷ `481038007.515006` bytes/token = `71.55632325844289` tok/s. The machine-readable validator recomputes this equation and rejects inconsistent evidence.

The classification is **`DERIVED_PROXY_NOT_HARDWARE_MEASURED`**. The logical
bytes/token value is not a process read counter. It assumes descriptor-derived
logical traffic and therefore cannot prove that every byte was fetched from
DRAM. The predicted throughput excludes decoder overhead, cache reuse,
synchronization, sampling, and page/cache effects. It is a proxy equation, not
a hardware-measured runtime ceiling or a product performance claim.

## Time and counter evidence

| Counter | Result | Source |
|---|---:|---|
| HyperVSQ-2 kernel time | 12446.774 ms aggregate | qwnrun_hypervsq2_wall_timer |
| SwiGLU time | 16.719 ms aggregate | qwnrun_swiglu_wall_timer |
| Prefill median | 2612.611 ms | qwnrun_generation_metrics_prefill_boundary |
| Decode median | 3466.305 ms | qwnrun_generation_metrics_decode_boundary |
| Process memory reads | None | unavailable |
| Memory-controller bandwidth | None | unavailable |
| L1/L2/L3 misses | None | unavailable |
| CPU cycles/instructions/vector instructions | unavailable | no supported hardware profiler configured |
| OpenMP synchronization time | unavailable | barrier timing is not separately instrumented |

## Interpretation

The measured local decode result is **18.463467 tok/s median** with p5 **18.023277 tok/s**. The independent bandwidth run does not justify assuming 40–60 GB/s, and the derived roofline must not be presented as measured hardware bandwidth. The current evidence supports a CPU VNNI path and a memory-sensitive workload, but it does not identify every decoder bottleneck; profiler-backed cache, cycles, instructions, and memory-controller counters remain unavailable on this host.

## Source evidence

- Machine-readable evidence: the path supplied to the renderer
- Harness: `benchmarks/benchmark_cpu_roofline.py`
- Runtime benchmark: `benchmarks/benchmark_release_quality.py`
- Model format: `docs/qwn-format.md`
