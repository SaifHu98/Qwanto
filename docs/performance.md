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

The current local record is one Windows `qwnrun` execution of the validated
4B HyperVSQ-2 container. Its 8.154199 tokens/s is evidence for that executable,
model hash, prompt, context, seed, token limit, and host only. It is not a
universal QWN throughput claim. TTFT and process VRAM allocation were not
reported by that run. The host has an RTX 5070 Ti, but this CPU run reported no
CUDA matmul and must not be described as GPU execution.

The complete generated table, including the exact host and evidence IDs, is in
[`performance-report.md`](performance-report.md). QWN container invariants are
defined in [`qwn-format.md`](qwn-format.md).

## Quantization status and trade-offs

| Format | Status | Intended trade-off |
| --- | --- | --- |
| FP32 | Implemented container dtype; no current report evidence | Highest numerical precision and largest storage footprint. |
| FP16 | Implemented container dtype; no current report evidence | Lower storage and bandwidth than FP32, with reduced precision. |
| Q4_0 | Implemented and container-validated; no matching native inference row in the current report | Conventional 4-bit storage trade-off; native speed and quality must be measured on the same model and host. |
| HyperVSQ-2 | Validated conversion and measured native `qwnrun` evidence | Smaller weights and lower memory pressure; quality and speed remain model- and kernel-dependent. |
| TWLA | Reference/experimental only; no complete model evidence | Experimental ternary/low-bit path; not a published model-level performance claim. |
| LittleBit-2 | Reference/experimental only; no complete model evidence | Low-rank binary factors trade representation size against approximation error. |
| TurboQuant | Reference/experimental only; no complete model evidence | Low-bit KV-cache storage trades cache capacity against approximation behavior. |
| JetSpec / SlimInfer / BitDecoding | Reference/experimental only | No tested Qwanto end-to-end evidence is published. |

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
