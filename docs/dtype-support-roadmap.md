# Qwanto dtype & architecture support roadmap

This document describes the **planned** expansion of Qwanto's native `.qwn`
container to cover formats that are currently refused. It is a forward
plan only — nothing here is shipped, and an item cannot move from "planned"
to "supported" without (a) a verified decoder/kernel pair, (b) a real
`MEASURED` evidence row in `benchmark_evidence.json` binding a specific
binary hash, model hash, and prompt, and (c) the new row appearing in
`docs/qwn-supported-quantizations.md`.

The roadmap is a 4-phase delivery in priority order. Each phase lists the
formats it brings online, the prerequisite oracles it requires, the
acceptance gates it must clear, and the architectural risk it carries.

---

## Phase 1 — read-side K-quant decoders for live `.qwn` outputs

**Why first.** K-quants (`Q4_K`, `Q5_K`, `Q6_K`) are the most common
public-GGUF format. Read-side block decoders already exist
(`c/tools/qwn_convert.py:797-906`) and are used to **re-quantize** GGUF
weights into another `--quant`. Wiring that same decoder through the
runtime kernel table would let users stay in `Q4_K` instead of lossy
re-quantization.

### Targets

- `QWN_DT_Q4_K` (new container dtype ID, ~144 bytes / 256 elements)
- `QWN_DT_Q5_K` (~176 bytes / 256 elements)
- `QWN_DT_Q6_K` (~210 bytes / 256 elements)

### Prerequisites (must exist before the kernel lands)

- Container format description in `docs/qwn-format.md` for each new ID.
  This implies a container version bump from `1` to `2` because the
  descriptor-table layout is fixed; loaders for version 1 must remain
  readable.
- A `dot_q4_k_block` AVX2 / VNNI scalar reference in
  `c/qwanto_kernels.c`. The scalar path must pass a 256-element golden
  test against a host-FP32 reference with absolute error ≤ 1e-3 across
  at least 10 random scales and 100 random blocks.
- A 100-test differential matrix in `c/tests/test_kquant_kernels.c` for
  each of Q4_K / Q5_K / Q6_K, mirroring `test_hypervsq2_kernels.c`.
- CUDA ABI 1 must be versioned to ABI 2 (or extended in-place with
  feature flags) before adding K-quant CUDA kernels. Until then, K-quants
  stay CPU-only.

### Architectural risk

- Container version bump: every existing `qwnrun.exe` on a user machine
  would need to stay forward-compatible. Mitigation: keep version-1
  readers and just ignore the new dtype IDs (`QWN_DT_*` ≥ some sentinel).
- Q5_K high-bit packing has a tricky 6-bit subscale + 8-bit per-block min
  layout; the existing `_dequantize_q5_k_block` must be re-checked
  against `ggml`'s canonical implementation at the byte level. If a
  single bit of the high-bit packing is off, downstream matmul diverges
  by one token; the differential test must catch it because `tok_per_sec`
  cannot.

### Acceptance gates

- `python -m pytest c/tests/ -q` → 0 failures.
- Native C differential tests (`make -C c test-c`) → Q4_K, Q5_K, Q6_K
  differential 100/100 each. Existing 17 binary tests must remain green.
- New `MEASURED` row in `benchmark_evidence.json` binding the new binary
  hash + a `Q4_K`, a `Q5_K`, and a `Q6_K` model SHA + the canonical prompt
  "Explain zero-copy NVMe memory tiering in Qwanto." with `--max-tokens 64`.
- Hosted CI green on Ubuntu + Windows before merge to `main`.

### Phase 1 result

The converter now has exact source decoders for `Q2_K`, `Q3_K`, and `Q8_K`,
and `--quant none` preserves those three canonical block payloads as native
QWN dtypes. Scalar runtime kernels and differential checks are present. The
K paths are CPU-only and have no promoted performance row.

---

## Phase 2 — IQ-type decoders

**Current status.** IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL, and IQ4_XS now have
exact source decoders and are streamed through F32 into an existing QWN
runtime dtype. IQ1 remains refused until its canonical decoder is ported and
differentially tested. These formats are not native QWN dtypes.

### Targets

Reusable block layout for all `IQ*_XS` and `IQ*_S` types, with
significantly higher decoder complexity than K-quants (super-blocks +
sub-blocks + importance-weighted quantisation). Each dtype gets:

- source readers for IQ2/IQ3/IQ4; native QWN IQ dtype IDs remain a separate
  runtime design decision.

### Prerequisites

- Full re-implementation in C of `ggml-quants.c`'s IQ quant/dequant
  routines, **with a port-of-reference check** against the upstream
  llama.cpp at byte granularity. This is the largest single chunk of
  work; without a port-of-reference the IQ dtypes drift from the
  canonical encoding and silently produce garbage.
- Reference CUDA kernels for at least `IQ4_XS` and `IQ4_NL`. The
  smaller IQ1/IQ2/IQ3 kernels are likely too small to beat the
  Q4_0/HyperVSQ-2 CPU path on latency, so they ship CPU-only first.
- New container version (`3` or extended feature flag on version 2)
  with the new dtype IDs.

### Architectural risk

- IQ-types use *importance scores* derived from the data distribution.
  Re-quantizing to a different IQ-type loses the importance table, so
  the converter would have to re-run the importance estimator at
  conversion time. This is the same reason IQ-types can't be lossy
  re-quantized like K-quants can.
- IQ1 / IQ2 dtypes have theoretical quality cliffs. Even with a
  correct decoder, perplexity on standard benchmarks is ~+0.5 vs
  Q4_0. We will not claim IQ-types are "lossless" or even "near
  lossless" — they ship with their measured perplexity delta.

### Acceptance gates

Same as Phase 1, plus:

- Perplexity measurement on `wikitext-2` for at least one IQ variant
  against the same model at Q4_0, recorded honestly with delta.
- README performance claim updated only if the IQ row actually beats
  the corresponding K-quant or Q4_0 row on tok/s; we do not advertise
  IQ-types as "fast" if they are in fact slower at the same precision.

---

## Phase 3 — Dense hybrid (Qwen-3.5 MTP) and MoE

**Why third.** This is the wall: Qwen-3.5 hybrid attention/MLP + MTP
prediction heads (the `DeepSeek-V4-Pro-Qwen3.5-4B-MTP` model in
`D:\EcoUni\qwanto\models\`), and MoE architectures in general
(Mixtral, DeepSeek-MoE). The current converter has **no reference
oracle** for either.

### Current boundary

Qwen3.8-27B conversion and a native CPU main-path integration now exist for
the local Q4_0 artifact. The Gated DeltaNet recurrent state, causal
convolution, full-attention layers, and FFN execute in `qwnrun`. MTP remains
metadata-preserved only, and MoE remains fail-closed. No Qwen3.8 benchmark or
quality claim is promoted.

### Targets

- MTP module: a small linear "next-token predictor" head evaluated after
  the main decoder; can be added as `QwnDecoder::mtp_predict`.
- MoE: `MoELayer` dispatch with `topk` gating and `experts[n]`
  weight tensors. The shared-expert vs routed-expert classification
  already exists in `c/tools/qwn_roles.py`.

### Prerequisites (the hard ones)

- **MTP requires a transformer-level reference oracle.** Without an
  oracle, every decoded token diverges by 1+ positions after only ~3
  predictions, and the bug is unobservable in the matrix because the
  token stream still "looks" different. The reference must be a
  small distilled Qwen-3.5 model whose weights are independently
  verifiable.
- **MoE requires independent expert kernels per layer.** Per-expert
  tensor allocation, gating, and load-balancing are required before
  the first MoE model can be converted and run end-to-end. Routing
  randomization needs an oracle.

### Architectural risk

- These cannot be added without breaking the current
  `dense_decoder_only` invariant in `c/qwanto_decode.c`. The decoder
  layer iteration will become polymorphic; this is the largest
  single architectural change in the codebase.
- A botched MoE dispatcher can leak memory per token. Test must
  include a tight memory-budget stress test, not just correctness.

### Acceptance gates

- A small Qwen-3.5 reference model (≤ 1B parameters) is committed
  under `experiments/results/` with a documented SHA-256.
- Differential test against `transformers` reference produces
  byte-identical logits within ≤ 1e-4 absolute error across 100
  forward passes.
- A first MoE model (Mixtral-style) is converted, run end-to-end, and
  its decode tok/s is recorded as `MEASURED`.
- Hosted CI green.

### Out of scope for Phase 3

- Full Qwen-3.5-4B-MTP end-to-end. That is a separate phase because
  the 4B BF16 reference weights are 8 GB on disk and the existing
  workspace does not have the additional storage or CI runner memory
  budget to bring it in cleanly.

---

## Phase 4 — Hybrid SSM / DeltaNet (Qwen-3.8 27B and friends)

**Why last.** The hardest. Qwen-3.8-27B ships 65 layers: 48 Gated
DeltaNet/SSM layers, 17 full-attention layers, one MTP prediction layer, and
mixed IQ/K dtypes within the same file. The current tree now has the CPU
DeltaNet state machinery, native IQ row decoding, and a Q4_0 main-path
integration, but full support still requires MTP execution, quality oracle,
and CUDA hybrid coverage.

### Targets

- DeltaNet / Gated DeltaNet state iteration inside `qwn_decoder`.
- SSM-only routing entry points.
- Co-existence with Phase 2 IQ-types (the Qwen-3.8 source uses mixed IQ/K
  tensors; native IQ row decoding is verified, while full hybrid model
  integration remains a separate gate).

### Prerequisites

- All of Phases 1-3.
- A small hybrid reference model under `experiments/results/` with
  a committed oracle (Hugging Face `transformers` baseline). Until
  that is committed, Phase 4 **does not start**.

### Architectural risk

- This is the only phase that touches the tiered-memory invariant.
  The DeltaNet state has a different residency profile from the KV
  cache (it grows linearly with sequence length, not with cache
  depth), and our NVMe-mmap prefetch story assumes KV-cache layout.
- The current CPU SSM transition is an integration path, not yet a
  transformer-level correctness oracle; it must be differentially validated
  before full Qwen3.8 support is claimed.

### Acceptance gates

- Same as Phase 3, with the additional gate that **the qualification tool and
  correctness oracle return a full-support decision for at least one
  Qwen-3.8-class source GGUF** before full architecture support is claimed.
  The current qualification file records only
  `CPU_MAIN_PATH_INTEGRATION_VERIFIED`.

---

## What is **not** on this roadmap

- A "convert it anyway" path that bypasses the qualification tool.
  The tool exists precisely to refuse conversions without an oracle.
- Bringing Qwen-3.8-27B online by writing a different decoder that
  silently produces outputs that don't match `transformers`. This
  is exactly what "no fabricated values" in AGENTS.md forbids.
- Speculative decoding with a non-native draft model. Speculative
  requires a typed, validated native QWN draft (see
  `qwn_speculative.c` boundary tests); a draft model that is missing
  or "good enough" is not the same as a draft model that is
  *verified*.
- TurboQuant algorithm parity. The current `QWN-Q4-KV` container
  is honest about being a 4-bit asymmetric channel quantizer, not
  the cited TurboQuant algorithm. Adding TurboQuant parity is
  another research project on its own.

When a phase completes, the corresponding entry in
`docs/qwn-supported-quantizations.md` is upgraded from ❌ / 🟡 to ✅
with a measured evidence row. No silent upgrades.
