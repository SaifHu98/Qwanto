# Performance evidence policy

The former FeatherCore baseline table was removed because it contained
machine-specific and unverified performance values. It is not release
evidence and must not be used as a product claim.

Current native evidence is recorded only by the reproducible harness and its
schema:

```powershell
python benchmarks/benchmark_reproducible.py --model path\to\model.qwn --executable path\to\qwnrun.exe --output benchmark_evidence.json
```

See [benchmark methodology](benchmark-methodology.md). A run is `MEASURED`
only when the real local process exits successfully and reports a positive
generated-token count. Missing, malformed, projected, experimental, or
fixture-only results retain their classification and do not expose
throughput as a fallback value.
