# Generated QWN performance evidence

This file is generated from the machine-readable sources listed below. It
contains no fallback throughput or memory values. `Unavailable` means the
runtime did not report a metric or the evidence was not comparable.

## Verified Performance Evidence

Numbers below are generated from matching executable/model hashes and the local qwnrun harness. A detected GPU is not treated as CUDA execution.

| Model | Native QWN Format | Size | Backend Actually Used | Decode tok/s | TTFT | Evidence Class | Reproduce |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| DeepSeek-V4-Pro-Qwen3.5-4B-HyperVSQ2 | QWN container (HyperVSQ-2) | 1.18 GiB (1,266,202,104 bytes) | cpu | 8.154199 tok/s | Unavailable | MEASURED | `D:\EcoUni\qwanto\c\qwnrun.exe D:\EcoUni\qwanto\experiments\results\4B_hyper_vsq2.qwn Explain zero-copy NVMe memory tiering in Qwanto. 64 4096 --backend cpu --ctx-size 4096 --max-tokens 64 --seed 0` |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_0 | QWN container (Q4_0) | 959.86 MiB (1,006,483,816 bytes) | cpu | 2.366479 tok/s | 6556.075 ms | MEASURED | `D:\EcoUni\qwanto\c\qwnrun.exe D:\EcoUni\qwanto\models\qwn\DeepSeek_R1_Distill_Qwen_1.5B_Q4_0.qwn Explain zero-copy NVMe memory tiering in Qwanto. 64 2048 --backend auto --ctx-size 2048 --max-tokens 64 --seed 0 --thinking none` |

The native row above is only a product performance claim when its evidence class is `MEASURED`; otherwise it is explicitly unavailable or experimental.

## Conversion evidence (not inference throughput)

| Model | QWN mode | File Size | Bits/Weight | Conversion wall time | Conversion throughput | Evidence |
| --- | --- | --- | --- | ---: | ---: | --- |
| 4B conversion fixture | HyperVSQ | 2.20 GiB (2,360,129,016 bytes) | 4.363096 bpw | 19.722783 s | 114.121542 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | HyperVSQ-2 | 1.18 GiB (1,266,202,104 bytes) | 2.340280 bpw | 19.119678 s | 63.157152 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | Unquantized QWN (conversion mode) | 8.06 GiB (8,655,201,784 bytes) | 16.003599 bpw | 15.214987 s | 542.507429 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | VSQ | 2.27 GiB (2,441,553,400 bytes) | 4.513471 bpw | 18.471345 s | 126.057240 MB/s | MEASURED_CONVERSION |
| 4B conversion fixture | VSQ-Ultra | 2.22 GiB (2,380,518,904 bytes) | 4.400860 bpw | 20.715011 s | 109.593942 MB/s | MEASURED_CONVERSION |

Excluded conversion records are retained in the JSON report with their integrity reason.

## Runtime Feature Status

| Capability | Status | Evidence |
| --- | --- | --- |
| FP32 | implemented container dtype; no current report evidence | `UNAVAILABLE` |
| FP16 | implemented container dtype; no current report evidence | `UNAVAILABLE` |
| Q4_0 | validated conversion and measured native qwnrun evidence | `MEASURED` |
| HyperVSQ-2 | validated conversion and measured native qwnrun evidence | `MEASURED` |
| TWLA | reference only; no validated end-to-end QWN evidence | `EXPERIMENTAL` |
| LittleBit-2 | reference only; no validated end-to-end QWN evidence | `EXPERIMENTAL` |
| TurboQuant | reference only; no validated end-to-end QWN evidence | `EXPERIMENTAL` |
| Q8_0 | implemented container dtype; no current native inference row | `UNAVAILABLE` |
| BF16 | implemented container dtype; no current native inference row | `UNAVAILABLE` |
| VSQ | implemented container dtype; local reference matrix only | `EXPERIMENTAL` |
| VSQ_ULTRA | implemented container dtype; local reference matrix only | `EXPERIMENTAL` |
| HYPER_VSQ | implemented container dtype; local reference matrix only (distinct from HyperVSQ-2) | `EXPERIMENTAL` |
| JetSpec | reference only; no validated end-to-end QWN evidence | `EXPERIMENTAL` |
| SlimInfer | reference only; no validated end-to-end QWN evidence | `EXPERIMENTAL` |
| BitDecoding | reference only; no validated end-to-end QWN evidence | `EXPERIMENTAL` |

## How to reproduce

Run the real local executable; do not substitute an external GGUF runtime.
Both MEASURED rows in the table above are produced by the same harness:

```powershell
# Canonical 4B HyperVSQ-2 row
python benchmarks/benchmark_reproducible.py --model experiments/results/4B_hyper_vsq2.qwn --executable c/qwnrun.exe --backend cpu --context-size 4096 --max-tokens 64 --seed 0 --warmup-tokens 8 --output benchmark_evidence.json

# 1.5B Q4_0 rows (require the converted .qwn in models/qwn/, generated from the GGUF source via `python c/tools/qwn_convert.py convert --quant q4_0`)
python benchmarks/benchmark_reproducible.py --model "models/qwn/DeepSeek_R1_Distill_Qwen_1.5B_Q4_0.qwn" --executable c/qwnrun.exe --backend auto --context-size 2048 --max-tokens 64 --seed 0 --warmup-tokens 4 --output "benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_64tok.json"
python benchmarks/benchmark_reproducible.py --model "models/qwn/DeepSeek_R1_Distill_Qwen_1.5B_Q4_0.qwn" --executable c/qwnrun.exe --backend cpu --threads 8 --context-size 2048 --max-tokens 128 --seed 0 --warmup-tokens 8 --output "benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_128tok_cpu.json"

# Roll up the matrix and the rendered report (every evidence file is passed)
python benchmarks/generate_benchmark_matrix.py --evidence benchmark_evidence.json --evidence benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_64tok.json --evidence benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_128tok_cpu.json --output benchmarks/benchmark_matrix.json
python benchmarks/generate_performance_report.py --evidence benchmark_evidence.json --evidence benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_64tok.json --evidence benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_128tok_cpu.json --manifest docs/model-manifest.json
```

## Research and Future Work

TWLA, LittleBit-2, TurboQuant, JetSpec, SlimInfer, and BitDecoding exist as decoder implementation modules under `c/qwanto_*.c`
(compiled into `qwnrun` via the Makefile `QWNRUN_SRCS` list) but are not exercised by any current measured native-inference row.
Their dtype IDs are not part of the `.qwn` container envelope, so they cannot appear in a published `.qwn` performance row
without first adding the corresponding `QWN_DT_*` enum entry plus a verified reader + kernel pair.
Projections and external GGUF results are not native performance claims.
See `docs/dtype-support-roadmap.md` for the planned expansion.

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
- Native evidence: `benchmark_evidence.json, benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_64tok.json, benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_128tok_cpu.json`
- Conversion evidence: `experiments/results/bpw_report.csv`
