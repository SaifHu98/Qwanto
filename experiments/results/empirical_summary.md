# Empirical Qwanto Study

Generated `2026-08-14T18:58:10` on `win32` (Python 3.14.6).

Every figure below was produced by the experiment driver under `experiments/`.  No numbers are fabricated.

## 1.5B model

| Format | Wall (s) | Throughput (MB/s) | Size (MB) | Payload bpw | Effective bpw |
|---|---:|---:|---:|---:|---:|
| `none` | 0.90 | 1178.2 | 1060.33 | 5.003 | 5.003 |
| `q4_0` | 0.84 | 1264.8 | 1060.33 | 5.003 | 5.003 |
| `vsq` | 0.88 | 1205.8 | 1060.33 | 5.003 | 5.003 |
| `vsq_ultra` | 0.83 | 1278.7 | 1060.33 | 5.003 | 5.003 |
| `hyper_vsq` | 0.83 | 1273.3 | 1060.33 | 5.003 | 5.003 |
| `hyper_vsq2` | 0.83 | 1273.3 | 1060.33 | 5.003 | 5.003 |

On-disk overhead: header `4096` B, descriptor block `32736` B, tail block `4096` B, alignment overhead `0` B.

## 4B model

| Format | Wall (s) | Throughput (MB/s) | Size (MB) | Payload bpw | Effective bpw |
|---|---:|---:|---:|---:|---:|
| `none` | 15.21 | 542.5 | 8254.24 | 16.004 | 16.004 |
| `q4_0` | 30.15 | 77.1 | 2325.07 | 4.507 | 4.507 |
| `vsq` | 18.47 | 126.1 | 2328.45 | 4.513 | 4.513 |
| `vsq_ultra` | 20.72 | 109.6 | 2270.24 | 4.401 | 4.401 |
| `hyper_vsq` | 19.72 | 114.1 | 2250.79 | 4.363 | 4.363 |
| `hyper_vsq2` | 19.12 | 63.2 | 1207.54 | 2.340 | 2.340 |

On-disk overhead: header `4096` B, descriptor block `42528` B, tail block `4096` B, alignment overhead `0` B.

## llama-server inference (real OpenAI-compatible HTTP)

| Model | Cold load (s) | TTFT mean (ms) | TTFT median (ms) | Decode tok/s mean | Decode tok/s median | Rounds kept |
|---|---:|---:|---:|---:|---:|---:|
| `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | 2.07 | 107 | 99 | 201.4 | 203.1 | 4 |
| `DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf` | 7.78 | 1684 | 358 | 48.2 | 47.9 | 4 |
