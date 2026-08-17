# CPU Roofline Analysis — 2026-08-17

This report is generated from `D:\EcoUni\qwanto\c\qwnrun_phase3.exe` and the measured evidence file. It is local evidence and remains **MEASURED_LOCAL_PENDING_HOSTED_VALIDATION** until the exact final runtime commit passes hosted validation. It is not a release-quality claim.

## Reproduction identity

| Field | Value |
|---|---|
| Git commit | `cb3ca351b79bbbf2b9244a29ee286dbbc80cc17c` |
| Worktree dirty during measurement | `True` |
| Executable SHA-256 | `81503e04278d007e45fc85c7b69edd0d1cecf250152128370adfa53eb05a5454` |
| Model SHA-256 | `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36` |
| Host | Windows-11-10.0.26200-SP0 |
| CPU | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD |
| Active workers | 8 |
| Selected kernel | vnni |
| OpenMP compiled/runtime loaded | True / True |

The model file is **1.179 GiB** with **1.179 GiB** of mapped tensor payload across 443 tensors. The dominant dtype is `HYPER_VSQ2`.

## Independent bandwidth measurement

The benchmark used a 256 MiB NumPy buffer, 3 repetitions, and read-only plus copy workloads. Values below are independent stream-like measurements, not qwnrun hardware-counter readings.

| Workers | Read median GB/s | Read p5 GB/s | Copy median GB/s |
|---:|---:|---:|---:|
| 1 | Unavailable | Unavailable | Unavailable |
| 2 | Unavailable | Unavailable | Unavailable |
| 4 | Unavailable | Unavailable | Unavailable |
| 6 | Unavailable | Unavailable | Unavailable |
| 8 | Unavailable | Unavailable | Unavailable |
| 12 | Unavailable | Unavailable | Unavailable |
| 16 | Unavailable | Unavailable | Unavailable |
| 32 | Unavailable | Unavailable | Unavailable |

At the selected 8 workers, the read-only median was **6.367 GB/s** and the copy median was **6.499 GB/s**. These are independent bandwidth proxies, not measured memory-controller bandwidth.

## Arithmetic intensity and roofline estimate

| Quantity | Value | Source / limitation |
|---|---:|---|
| Logical weight bytes per token | 861579040.000 | qwnrun_descriptor_traffic_counter from HyperVSQ-2 descriptors |
| Logical FLOPs per token | 5961195520.000 | qwnrun_matmul_shape_counter from descriptor-derived operations |
| Arithmetic intensity | 6.918919 FLOP/byte | derived estimate |
| Predicted throughput | 50.432 tok/s | derived_estimate using selected independent bandwidth |
| Actual persistent decode median | 17.907 tok/s | qwnrun_release_quality_persistent_decode |
| Actual / predicted estimate | 35.51% | derived comparison, not a hardware efficiency measurement |

The logical bytes/token value is not a process read counter. It assumes the descriptor-derived logical traffic and therefore cannot prove that every byte was fetched from DRAM. The predicted throughput excludes decoder overhead, cache reuse, synchronization, sampling, and page/cache effects. It is a roofline estimate only.

## Time and counter evidence

| Counter | Result | Source |
|---|---:|---|
| HyperVSQ-2 kernel time | 14597.088 ms aggregate | qwnrun_hypervsq2_wall_timer |
| SwiGLU time | 16.413 ms aggregate | qwnrun_swiglu_wall_timer |
| Prefill median | 2654.781 ms | qwnrun_generation_metrics_prefill_boundary |
| Decode median | 3574.042 ms | qwnrun_generation_metrics_decode_boundary |
| Process memory reads | None | unavailable |
| Memory-controller bandwidth | None | unavailable |
| L1/L2/L3 misses | None | unavailable |
| CPU cycles/instructions/vector instructions | unavailable | no supported hardware profiler configured |
| OpenMP synchronization time | unavailable | barrier timing is not separately instrumented |

## Interpretation

The measured local decode result is **17.906896 tok/s median** with p5 **17.778785 tok/s**. The independent bandwidth run does not justify assuming 40–60 GB/s, and the derived roofline must not be presented as measured hardware bandwidth. The current evidence supports a CPU VNNI path and a memory-sensitive workload, but it does not identify every decoder bottleneck; profiler-backed cache, cycles, instructions, and memory-controller counters remain unavailable on this host.

## Source evidence

- Machine-readable evidence: `benchmarks/evidence/windows/2026-08-17/phase3-local/roofline-8t-64.json`
- Harness: `benchmarks/benchmark_cpu_roofline.py`
- Runtime benchmark: `benchmarks/benchmark_release_quality.py`
- Model format: `docs/qwn-format.md`
