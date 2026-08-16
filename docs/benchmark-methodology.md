# Benchmark methodology and evidence standard

Qwanto publishes a number only when a local `qwnrun` process produced it. The
reproducible harness uses `time.perf_counter()`, captures the exact argv,
records executable/model SHA-256 hashes, and stores stdout/stderr hashes.

Run it with a real local fixture:

```sh
python benchmarks/benchmark_reproducible.py \
  --model experiments/results/4B_hyper_vsq2.qwn \
  --executable c/qwnrun \
  --max-tokens 64 \
  --output benchmark_evidence.json
```

The command must be rerun on each host. The repository does not provide a
universal hardware profile or fallback throughput value.

## Classifications

| Classification | Meaning | Measured throughput allowed? |
| --- | --- | --- |
| `MEASURED` | Real qwnrun exited zero and reported a positive token count; wall time was measured | Yes |
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
- Record prompt length, requested token limit, context size, backend, and
  relevant environment variables.
- Do not compare runs across different model hashes, token limits, or backends
  without labeling the comparison.
- Do not report a speedup unless both operands are independently measured and
  comparable.
- Large real-model tests skip only when the named fixture is absent. A present
  fixture must execute and fail on a real error.
