# Benchmark methodology and evidence standard

Qwanto publishes a number only when a local `qwnrun` process produced it. The
reproducible harness uses `time.perf_counter()`, captures the exact argv,
records executable/model SHA-256 hashes, and stores stdout/stderr hashes.

Run it with a real local fixture:

```sh
python benchmarks/benchmark_reproducible.py \
  --model experiments/results/4B_hyper_vsq2.qwn \
  --executable c/qwnrun \
  --backend cpu \
  --context-size 4096 \
  --max-tokens 64 \
  --seed 0 \
  --warmup-tokens 8 \
  --output benchmark_evidence.json
python benchmarks/generate_benchmark_matrix.py
```

The command must be rerun on each host. The repository does not provide a
universal hardware profile or fallback throughput value.

## Classifications

| Classification | Meaning | Measured throughput allowed? |
| --- | --- | --- |
| `MEASURED` | Real qwnrun exited zero, reported the actual backend/kernel, produced a positive token count, and wall time was measured with matching executable/model hashes | Yes |
| `UNAVAILABLE` | Required executable/model/sensor is missing, or execution timed out | No |
| `INVALID` | Nonzero exit, malformed output, conflicting records, or zero/negative tokens | No |
| `TEST_FIXTURE` | Explicitly marked test data used to exercise parsers or UI states | No production claim |
| `EXPERIMENTAL` | Real but outside the supported comparison boundary, such as an external backend | Not a native claim |
| `PROJECTED` | A model or planning estimate, never an empirical measurement | No |

Failed records still contain a valid schema, command, hashes where available,
and an explanation. `measured_evidence` is `null` for every non-measured run.
Unknown VRAM, NVMe, RSS, and TTFT values are `null` or explicitly listed as
unavailable; they are never backfilled from a host profile.

## Reproducibility requirements

- Keep the exact model and executable hashes with the artifact.
- Record the exact prompt, requested token limit, context size, fixed seed,
  warmup, requested/actual backend, selected kernel, Qwanto version, Git
  commit, active threads, and runtime counters in `benchmarks/benchmark_matrix.json`.
- A CUDA row is invalid unless GPU matmul count is greater than zero and the
  required-layer CPU fallback count is zero. GPU inventory alone is not CUDA
  execution evidence.
- Do not compare runs across different model hashes, token limits, or backends
  without labeling the comparison.
- Do not report a speedup unless both operands are independently measured and
  comparable.
- Large real-model tests skip only when the named fixture is absent. A present
  fixture must execute and fail on a real error.
