# Generated QWN performance evidence

This file is generated from the machine-readable sources listed below. It
contains no fallback throughput or memory values. `Unavailable` means the
runtime did not report a metric or the evidence was not comparable.

## Native QWN inference

| Model | Source Format | QWN Quantization | File Size | Bits/Weight if known | RAM / VRAM Measurement | TTFT | Tokens/s | Hardware | Evidence Class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro-Qwen3.5-4B-HyperVSQ2 | QWN container | HyperVSQ-2 | 1.18 GiB (1,266,202,104 bytes) | 2.34028 bpw | Unavailable | Unavailable | 7.148571 | Windows 11 (10.0.26200); CPU: AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD; GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU | MEASURED |

The native row above is only a qwnrun inference claim when its evidence
class is `MEASURED`. The current artifact records TTFT as unavailable
because the runtime did not expose a positive first-token measurement.

## Conversion evidence (not inference throughput)

| Model | QWN mode | File Size | Bits/Weight | Conversion wall time | Conversion throughput | Evidence |
| --- | --- | --- | --- | ---: | ---: | --- |
| 4B conversion fixture | HyperVSQ | 2.20 GiB (2,360,129,016 bytes) | 4.363096 bpw | 19.722783 s | 114.121542 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | HyperVSQ-2 | 1.18 GiB (1,266,202,104 bytes) | 2.340280 bpw | 19.119678 s | 63.157152 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | Unquantized QWN (conversion mode) | 8.06 GiB (8,655,201,784 bytes) | 16.003599 bpw | 15.214987 s | 542.507429 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | VSQ | 2.27 GiB (2,441,553,400 bytes) | 4.513471 bpw | 18.471345 s | 126.057240 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | VSQ-Ultra | 2.22 GiB (2,380,518,904 bytes) | 4.400860 bpw | 20.715011 s | 109.593942 MB/s | MEASURED_CONVERSION |

Excluded conversion records are retained in the JSON report with their integrity reason.

## Format status

| Format | Status | Evidence Class |
| --- | --- | --- |
| FP32 | implemented container dtype; no current report evidence | UNAVAILABLE |
| FP16 | implemented container dtype; no current report evidence | UNAVAILABLE |
| Q4_0 | implemented and container-validated; no matching native inference row | UNAVAILABLE |
| HyperVSQ-2 | validated conversion and measured native qwnrun evidence | MEASURED |
| TWLA 1.58-bit | implemented/tested kernel path; no complete model evidence | EXPERIMENTAL |
| LittleBit | implemented/tested library path; not a QWN container dtype | EXPERIMENTAL |
| TurboQuant | implemented/tested KV path; no complete model evidence | EXPERIMENTAL |

## External GGUF evidence

These measurements used the external local `llama-server` boundary. They
are shown for provenance only and must not be read as native QWN results.

| Model | Cold load | TTFT mean | Decode mean | Decode median | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf | 2.0689816999947652 s | 106.97380000419798 ms | 201.42600062596347 tok/s | 203.10107712585025 tok/s | EXPERIMENTAL_EXTERNAL |
| DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf | 7.779932399993413 s | 1683.7020000020857 ms | 48.155665106601475 tok/s | 47.942937722003144 tok/s | EXPERIMENTAL_EXTERNAL |

## Sources

- [QWN container format](qwn-format.md)
- [Benchmark methodology](benchmark-methodology.md)
- Manifest: `docs/model-manifest.json`
- Native evidence: `benchmark_evidence.json`
- Conversion evidence: `experiments/results/bpw_report.csv`
