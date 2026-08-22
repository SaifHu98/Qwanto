# QWN performance and quantization

QWN performance evidence is intentionally host- and model-specific. The
published report is generated from the local `qwnrun` evidence schema, the
model manifest, and recorded conversion artifacts:

```sh
python benchmarks/generate_performance_report.py \
  --output docs/performance-report.json \
  --markdown-output docs/performance-report.md
python benchmarks/generate_benchmark_matrix.py
```

The generated [machine-readable report](performance-report.json) and
[rendered evidence](performance-report.md) are the source for the tables
below. A native throughput number is included only after a real `qwnrun`
process exits successfully with a positive token count and a model hash that
matches `docs/model-manifest.json`. A missing sensor, TTFT record, or native
run is shown as `Unavailable`; it is never represented as zero or estimated.

## Verified native evidence

The current local records are:

1. **4B HyperVSQ-2** — Windows `qwnrun` execution of the validated QWN fixture
   recorded in `benchmark_evidence.json`. Its `8.154199 tok/s` is evidence
   for that executable, model hash, prompt, context, seed, token limit,
   and host only.
2. **1.5B Q4_0** — Windows `qwnrun` execution of a converted DeepSeek-R1-Distill
   Qwen GGUF source. Two rows are MEASURED: `1.718469 tok/s` (64 tokens) and
   `2.474 tok/s` (128 tokens, explicit `--threads 8`). TTFT was reported as
   `6556 ms` for the 64-token row and `Unavailable` for the 128-token row.

Both rows are evidence only for the recorded executable, model, prompt,
context, seed, thread count, and host. They are not universal QWN
throughput claims. The host has an RTX 5070 Ti, but both CPU runs reported
no CUDA matmul and must not be described as GPU execution. A separate CUDA
attempt on the 1.5B Q4_0 model failed closed
(`benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_cuda_attempt.json` /
`1.5b_cuda_probe.log`) because `qwn_cuda.dll` ABI 1 only supports
`QWN_DT_HYPER_VSQ2` weights.

The complete generated table, including the exact host and evidence IDs, is
in [`performance-report.md`](performance-report.md). QWN container invariants
are defined in [`qwn-format.md`](qwn-format.md). The complete supported/
unsupported dtype matrix is in
[`qwn-supported-quantizations.md`](qwn-supported-quantizations.md).

## Quantization status and trade-offs

| Format | Status | Intended trade-off |
| --- | --- | --- |
| FP32 | Implemented container dtype; no current report evidence | Highest numerical precision and largest storage footprint. |
| FP16 | Implemented container dtype; no current report evidence | Lower storage and bandwidth than FP32, with reduced precision. |
| BF16 | Implemented container dtype; no current report evidence | Same storage as FP16; different rounding bias suitable for native BF16 source models. |
| Q8_0 | Implemented container dtype; no current performance row | Per-row 8-bit storage; useful as a debug/reference weight. |
| Q4_0 | Implemented and container-validated; native CPU measurement on 1.5B row in `benchmark_matrix.json` | Conventional 4-bit storage trade-off; native speed and quality must be measured on the same model and host. |
| VSQ / VSQ_ULTRA / HYPER_VSQ | Local reference matrix only; no measured native row | Earlier-generation vector-subbit trade-offs; not published as model-level performance. |
| HyperVSQ-2 | Validated conversion and measured native `qwnrun` evidence (4B row) | Smaller weights and lower memory pressure; quality and speed remain model- and kernel-dependent. |
| TWLA / LittleBit-2 / TurboQuant / JetSpec / SlimInfer / BitDecoding | Reference/experimental only; no end-to-end QWN model evidence published | Not a published model-level performance claim. |

TWLA/LittleBit-2/TurboQuant/JetSpec/SlimInfer/BitDecoding exist as decoder
modules under `c/qwanto_*.c` and are wired into the `qwnrun` executable
via the Makefile `QWNRUN_SRCS` list, but no measured row exercises any of
them. Their dtype IDs are not part of the `.qwn` envelope today; adding a
new measured row requires the corresponding `QWN_DT_*` enum entry and
reader.

The format names, dtype IDs, 4 KiB header, 64-byte payload padding, and
validation rules are documented in [`qwn-format.md`](qwn-format.md). The
conversion measurements in the generated report are conversion throughput,
not inference tokens/s.

## Why QWN?

- Container validation rejects malformed metadata and out-of-bounds tensor
  descriptors before native execution.
- Aligned tensor data gives the native decoder a predictable layout for SIMD
  kernels.
- Inference stays local through the native runtime and loopback gateway.
- Manifests and hashes make model provenance explicit.
- The report generator preserves reproducible evidence instead of filling
  gaps with host guesses.
- Runtime placement can use the implemented VRAM, RAM, and memory-mapped
  storage paths where the selected build and model support them.
- The conversion workflow turns user-managed source models into validated
  `.qwn` artifacts without bundling weights in the application.

## Archived external evidence

The generated report may retain historical local `llama-server` measurements
as `EXPERIMENTAL_EXTERNAL` for provenance. They are not part of the supported
runtime path: GGUF is a conversion input, and Qwanto executes validated QWN
containers only. The models, runtime, host, and measurement protocol differ,
so no direct speedup or compression comparison is implied.

For the evidence classification rules and reproducibility requirements, see
[`benchmark-methodology.md`](benchmark-methodology.md).
