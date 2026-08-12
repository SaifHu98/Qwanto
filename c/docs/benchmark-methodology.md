# Performance and Correctness Benchmarking Methodology

This suite implements strict gating for performance optimizations. Any commit making performance claims ("2x faster", "50% lower memory") must pass these automated benchmarks and output a strictly formatted JSON report conforming to `benchmarks/schema.json`.

## Methodology

### 1. Determinism and Hardware Parity
All claims must be accompanied by raw JSON benchmarks proving the median improvement over identical hardware, model hash, cache state, backend, and prompt. If reporting "2x faster", the median token generation speed over measured repetitions must be `>= 1.90x`.

If reporting "50% lower memory", the measured peak RSS (Resident Set Size) must be reduced by `>= 45%` under identical execution constraints.

### 2. Gating Criteria
Before any performance optimization is merged, it must pass the following correctness gates automatically evaluated by `run_matrix.py`:
- **Token Parity**: The generated token IDs must exactly match a known good reference for the given prompt and snapshot.
- **KV Persistence**: Saving the generation KV state to disk and reloading it for a continuation must yield identical output to a continuous, uninterrupted generation.
- **Resource Leaks**: `psutil` sidecars monitor thread counts and file descriptors during queue saturation testing to ensure no unbounded growth.
- **Peak RSS**: Monitored concurrently and must not exceed the specified resource budget.
- **Stream Mismatch**: Incremental Server-Sent Events (SSE) chunks must perfectly reconstruct the non-streaming output for the exact same prompt and parameters.
- **Deadlock Check**: Submitting a burst of parallel asynchronous requests exceeding the max queue depth must return 429s or queue cleanly, without hanging the engine.

### 3. Execution
```bash
python benchmarks/run_matrix.py --engine ./glm.exe --model /path/to/snap
```
This produces `benchmark_results.json`.

### 4. Comparison
```bash
python benchmarks/compare.py baseline.json candidate.json
```
This will fail with exit code 1 if there is a >5% regression without a documented justification, or if the marketing format claims (e.g., "2x faster") are violated.
