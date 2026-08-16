# 🔬 Qwanto Benchmark Methodology & Evidence Standard

## 1. Core Principles

Qwanto enforces strict engineering integrity in all published performance benchmarks:
1. **Zero Fabricated Metrics**: No benchmark figure is published without reproducible raw evidence.
2. **Strict Classification**: Every metric must be explicitly tagged as **Measured**, **Experimental**, or **Projected**.
3. **Machine-Readable Artifacts**: All benchmark runs output standardized JSON records including system environment, SHA-256 model hashes, and exact timing telemetry.

---

## 2. Classification Schema

| Classification | Definition | Evidence Requirement |
|---|---|---|
| 🟢 **Measured** | Verified empirically on physical host hardware under documented operating conditions. | Raw JSON artifact with timestamp, system specs, SHA-256 hash, and reproducible command. |
| 🟡 **Experimental** | Kernel-level SIMD or C assertion tests verified, but full end-to-end user pipeline is undergoing optimization. | Isolated C/Python test suite output (`c/tests/`). |
| 🔵 **Projected** | Mathematical or multi-GPU scaling extrapolation based on hardware FLOPS and bandwidth modeling. | Explicitly labeled as theoretical; never presented as empirical data. |

---

## 3. Host Hardware Profile (Current Reference Baseline)

- **CPU**: AMD Ryzen 9 9955HX 16-Core Processor (16 Cores, 32 Threads, 64 MB L3 Cache, AVX-VNNI, AVX-512)
- **Primary GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB GDDR6, Driver 592.02, CUDA Compute SM89, PCIe 4.0 x16, Tensor Cores active)
- **Secondary GPU**: AMD Radeon(TM) 610M Graphics (512 MB shared RAM, Display Only)
- **RAM**: 32 GB DDR5-5600 MHz Dual-Channel
- **NVMe Storage**: Samsung PM9A1a 1.02TB PCIe 4.0 x4 SSD (Sequential Read: ~3,400 MB/s mmap)
- **Operating System**: Microsoft Windows 11 Pro 64-bit (Build 26200 / 24H2)
- **Native Toolchain**: LLVM Clang 18.1.8 / MSVC 19.41

---

## 4. Reproducibility Command

To run the reproducible benchmark harness:

```bash
# Run benchmark on default native .qwn model:
python benchmarks/benchmark_reproducible.py --model experiments/results/4B_hyper_vsq2.qwn --max-tokens 128 --output benchmark_evidence.json
```

---

## 5. Standard Output Schema

```json
{
  "benchmark_id": "qwn-bench-1786880396",
  "timestamp_utc": "2026-08-16T11:39:56.025011+00:00",
  "hardware_environment": {
    "os": "Windows 11 (Build 10.0.26200)",
    "cpu_brand": "AMD Ryzen 9 9955HX",
    "cpu_threads": 32,
    "gpus": [
      {
        "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "vram_gb": 12.0,
        "compute_cap": "SM89 (Ada Lovelace)"
      }
    ],
    "ram_gb": 32.0
  },
  "model_metadata": {
    "path": "experiments/results/4B_hyper_vsq2.qwn",
    "file_size_bytes": 1266202104,
    "sha256": "43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36",
    "quantization": "TWLA 1.58-Bit Ternary / HyperVSQ-2"
  },
  "measured_evidence": {
    "generated_tokens": 128,
    "ttft_ms": 2.15,
    "wall_seconds": 0.283,
    "tok_per_sec": 452.8,
    "process_rss_mb": 540.0,
    "vram_allocated_gb": 1.82
  },
  "evidence_classification": "EMPIRICAL_MEASURED_LIVE_HOST"
}
```
