# Native quality and benchmark gates

Qwanto keeps model quality and speed as separate, reproducible gates. A QWN
container opening successfully is not evidence that its outputs match a
reference model, and a measured local row is not a claim for another model or
hardware.

## Logit oracle

`c/tools/qwn_quality_oracle.py` consumes a fixture containing independently
generated `expected_top_ids` (and optionally `expected_top_values`) for token
sequences. It starts `qwnrun --serve`, resets state between cases, compares
native logits, and records model/executable SHA-256 values.

```powershell
python c/tools/qwn_quality_oracle.py `
  --qwnrun c/qwnrun.exe `
  --model models/example.qwn `
  --fixture quality-fixture.json `
  --out quality-result.json
```

The fixture must declare its reference provenance. A result is `MEASURED_PASS`
only when every declared case passes. Missing fixtures, missing references,
protocol failures, and model-hash mismatches are errors; they are never treated
as successful quality.

## Benchmark

`c/tools/qwn_benchmark.py` runs a JSON prompt set with fixed seed and runtime
configuration. Each row includes native stderr telemetry, process timing, and
SHA-256 values for both model and executable.

```powershell
python c/tools/qwn_benchmark.py `
  --qwnrun c/qwnrun.exe `
  --model models/example.qwn `
  --prompts benchmarks/qwen38-prompts.json `
  --out benchmarks/qwen38-result.json `
  --backend cpu --context 4096 --max-tokens 128
```

There is currently no official Flash-Next native benchmark result in this
repository. The official Flash-Next download is a four-shard external GGUF
bundle and remains `external_source_only` until the complete QSA, sparse-MoE,
ngram, gated-residual, DeltaNet, and MTP runtime is implemented and qualified.
