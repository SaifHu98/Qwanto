# QWN Container & Conversion: Supported and Unsupported Quantization Formats

This README is the official, audited list of every quantization format that the
native `qwnrun` runtime and the `qwn-convert` CLI accept, every format they
reject, and the exact error class each rejection produces. **No model can be
classified as supported unless it appears in the supported tables below with a
verified measurement row in `benchmark_evidence.json`.**

The source of truth is the code itself, not this document. Every line here
maps to one specific function in `c/tools/qwn_convert.py`,
`c/qwanto_decode.c`, `c/qwanto_kernels.c`, or `c/cuda/qwn_cuda_abi.h`. If a
format is added or removed, this document must be regenerated in the same
commit that flips the gate.

---

## 1. Container dtype IDs (the .qwn in-memory vocabulary)

Defined in `c/qwanto_native.h` (lines 53-62). A `.qwn` file's tensors carry one
of these IDs in the descriptor table. The runtime size math is in
`c/qwanto_native.c:114-156`. Anything outside this table is rejected by the
loader before inference.

| ID  | Name           | Container size per element | Block           | Decoder present? | Kernel present? | CUDA kernel? |
|----:|----------------|---------------------------:|-----------------|------------------|-----------------|---------------|
| `0` | `F32`          | 4 bytes                    | —               | ✅ `qwn_dequant_to_f32` | ✅ `dot_f32` (Q4 pack) | — |
| `1` | `F16`          | 2 bytes                    | —               | ✅                 | ✅ `dot_f16` via fp32 | — |
| `2` | `Q4_0`         | 18 bytes / 32 elements     | 32              | ✅                 | ✅ scalar/AVX2 (`dot_q4_q8_block`) | 🟡 declared at `qwanto_decode.c:874`, ABIs gated |
| `3` | `Q8_0`         | 34 bytes / 32 elements     | 32              | ✅                 | ✅ `dot_q8_q8_block`               | — |
| `4` | `BF16`         | 2 bytes                    | —               | ✅                 | ✅ upcasted to F32 | — |
| `5` | `BYTES`        | 1 byte                     | —               | ✅ for tokenizer + raw bytes | — | — |
| `6` | `VSQ`          | 36 bytes / 64 elements    | 64              | ✅                 | ✅ `dot_vsq_block` (scalar/AVX2/VNNI) | — |
| `7` | `VSQ_ULTRA`    | 70 bytes / 128 elements   | 128             | ✅                 | ✅ `dot_vsq_ultra_block`            | — |
| `8` | `HYPER_VSQ`    | 138 bytes / 256 elements  | 256             | ✅                 | ✅ `dot_hyper_vsq_block`            | — |
| `9` | `HYPER_VSQ2`   | 74 bytes / 256 elements   | 256             | ✅                 | ✅ `dot_hyper_vsq2_block` (scalar/AVX2/VNNI) | ✅ `qwn_cuda.dll` ABI 1, `gemv_hypervsq2` (RTX sm_120) |

Interpretation:

- ✅ — implemented, tested, on the hot path.
- 🟡 — declared in the dispatch table but no end-to-end model path exists for
  the 1.5B / 4B / 27B samples we have. Calling `--backend cuda` against a
  Q4_0-only model today fails closed with `layer N attn matmul failed`. See
  `benchmarks/evidence/windows/2026-08-22/1.5b_cuda_probe.log`.
- — — not in the dispatch path. The runtime **rejects** it from inference,
  even if a tensor of that dtype is technically declared.

The 4 KiB-header container invariants from `AGENTS.md` apply to **all** dtypes
above: descriptors and payloads are validated before any tensor is touched. A
container that lists Q4_K for any tensor fails the loader at the descriptor
step with `"unsupported_dtype"` (see §4).

---

## 2. Conversion `--quant` choices (the writer side)

`c/tools/qwn_convert.py` line 1491 is the source of truth for what the CLI
exposes. The `--quant` choices are exhaustive:

| CLI flag          | QWN container dtype | Status        | Use case                          |
|-------------------|---------------------|---------------|-----------------------------------|
| `q4_0`            | `QWN_DT_Q4_0`       | ✅ Supported  | Default. Works for every dense transformer that fits 2-D row geometry. |
| `q4_k`            | (read-only)         | ❌ Not exposed | GGUF source decoder exists in `_dequantize_q4_k_block` (line 797). It is used to **re-quantize** GGUF Q4_K weights into another `--quant`; it is not selectable as an output because no native `.qwn` Q4_K reader is wired in the runtime. |
| `q5_k`            | (read-only)         | ❌ Not exposed | Same status as Q4_K. `_dequantize_q5_k_block` exists at line 820. |
| `q6_k`            | (read-only)         | ❌ Not exposed | Same status. `_dequantize_q6_k_block` exists at line 841. |
| `vsq`             | `QWN_DT_VSQ`        | 🟡 Experimental | Local reference matrix only. Not used in any measured evidence row. |
| `vsq_ultra`       | `QWN_DT_VSQ_ULTRA`  | 🟡 Experimental | Local reference matrix only. |
| `hyper_vsq`       | `QWN_DT_HYPER_VSQ`  | 🟡 Experimental | Local reference matrix only. |
| `hyper_vsq2`      | `QWN_DT_HYPER_VSQ2` | ✅ Supported  | The only path with a release-quality CPU AVX-VNNI matmul **and** a CUDA ABI implementation. |
| `twla`            | (planned)           | ❌ Not implemented | Listed in `qwn_quant_plan.py` only. No native dtype ID, no kernel. README performance claims remain invalid. |
| `littlebit2`      | (planned)           | ❌ Not implemented | Same as `twla`. |
| `pquant`          | (planned)           | ❌ Not implemented | Same. |
| `none`            | `QWN_DT_F32`        | ✅ Supported  | Stores raw FP32 weights. Largest container; not a quant at all. Used by `experiments/results/15B_none.qwn` and `4B_none.qwn` for the dual-mode validation only. |

Calling `qwn-convert convert --quant <anything not in this table>` exits 2
with `argparse: invalid choice`. There is no silent fallback.

---

## 3. GGUF source dtypes (what the reader accepts)

`c/tools/qwn_convert.py` line 1054-1065 lists every tensor dtype the GGUF
reader understands, by `general.file_type` and tensor-type byte. The error
matrix below is what the converter prints before writing any `.qwn`:

| GGUF file_type | Tensor dtype      | Read?  | Action                                            |
|----------------|-------------------|--------|---------------------------------------------------|
| `0`            | `F32`             | ✅      | Pass through to the chosen `--quant`.             |
| `1`            | `F16`             | ✅      | Pass through.                                     |
| `2`            | `Q4_0`            | ✅      | Pass through.                                     |
| `3`            | `Q4_1`            | ❌     | Reject: `unsupported tensor dtype Q4_1`.          |
| `6`            | `Q5_0`            | ❌     | Reject: `unsupported tensor dtype Q5_0`.          |
| `7`            | `Q5_1`            | ❌     | Reject: `unsupported tensor dtype Q5_1`.          |
| `8`            | `Q8_0`            | ✅      | Dequantize to F32 then re-quantize.               |
| `9`            | `Q8_1`            | ❌     | Reject: `unsupported tensor dtype Q8_1`.          |
| `10`           | `Q2_K`            | ❌     | Reject: no verified K-quant decoder outputs a `.qwn` writer at this dtype. Source tensor is preserved unchanged in the source GGUF; conversion fails before any `.qwn` is written. |
| `11`           | `Q3_K`            | ❌     | Reject, same reason. (Read-side exists for Q3_K dequantization? No — only Q4_K / Q5_K / Q6_K readers are wired. Q2_K and Q3_K are explicitly not.) |
| `12`           | `Q4_K`            | ✅ (read-only) | Block-dequantized to F32, then passed to the chosen `--quant`. There is **no `.qwn` writer for Q4_K.** |
| `13`           | `Q5_K`            | ✅ (read-only) | Same status as Q4_K. `_dequantize_q5_k_block` at line 820. |
| `14`           | `Q6_K`            | ✅ (read-only) | Same status. `_dequantize_q6_k_block` at line 841. |
| `15`           | `Q8_K`            | ❌     | Reject: the only K-quant writer we have not built. Reader is absent. |
| `16`-`24`      | All `IQ*` (`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`) | ❌ | Reject: `Current converter has no exact IQ*/IQ2/IQ3/IQ4 block decoder for these source tensor dtypes; no reinterpretation is permitted.` (`qwn_convert.py:1095`). The whole file is refused, no `.qwn` is written, and the source GGUF is left in place unchanged. |

A **mixed-IQ** GGUF (different tensors at different IQ dtypes within the same
file) is also refused because the converter refuses to translate, e.g., an
`IQ2_M` head into a `.qwn` tensor. Qwen3.8-27B's `IQ2_M` model lands here.

---

## 4. Architecture-level rejection (qwen38 / hybrid / MTP)

Independent of dtype, the converter refuses unsupported architectures
**before** any dtype machinery runs. `c/tools/qwen38_qualification.py`
formalises this; the rest of the converter calls it as a pre-flight.

| Architecture marker                       | Status   | Reason recorded                                   |
|-------------------------------------------|---------|---------------------------------------------------|
| Dense Llama-style (≤ ~32 layers, all attention) | ✅     | Pass. See `1.5B Q4_K_M` measurement in §6.        |
| Hybrid Qwen-3.5 with MTP tensors          | ❌       | "Qwen3.5 hybrid + MTP matrix reference not committed." |
| Hybrid DeltaNet / SSM + attention mix (Qwen3.8-27B) | ❌ | "mixed IQ dtypes not supported by the converter" + no SSM/DeltaNet reference oracle. |
| MoE (Mixtral / DeepSeek-MoE-style)        | ❌       | No reference router implementation; experts not validated. |
| Mamba / SSM-only                          | ❌       | No native matmul dispatch. |
| Tied embeddings check                     | ✅/❌    | Detected automatically by `qwn_roles.py`; tied embeddings are classified `tied_embed`. Verified per-tensor in `c/qwn_quant_plan.py`. |

---

## 5. KV-cache dtype contract (orthogonal to the weight dtype)

From `c/qw_runtime_config.h` and `c/qwanto_turboquant.c`. The KV cache and the
weight container dtype are **separate** knobs. Mixed weight + KV-cache
combinations are not all supported.

| `--quantization` (weight) | `--kv-cache`         | Outcome    |
|---------------------------|----------------------|------------|
| `q4_0`                    | `fp16` (default)     | ✅ CPU runtime path, no CUDA. |
| `q4_0`                    | `q8`                 | ✅ CPU. |
| `q4_0`                    | `turboquant-q4`      | ✅ CPU. The TurboQuant container is exposed as `QWN-Q4-KV` in telemetry to **not** claim equivalence with the cited TurboQuant algorithm (it is a 4-bit asymmetric channel quantizer with 4.0 bpw container / 3.5 bpw payload, group size 64, 32-byte blocks). |
| `hyper_vsq2`              | `fp16` / `q8`        | ✅ Full path; CPU + CUDA ABI 1. |
| `hyper_vsq2`              | `turboquant-q4`      | ✅ CPU + CUDA ABI 1 + TurboQuant. |
| Anything else + CUDA      | anything             | ❌ fail-closed. Explicit `--backend cuda` exits non-zero without producing tokens. |

---

## 6. Measured evidence rows (the only rows that count)

The README performance claim is restricted to model + binary + commit triples
that have a real `MEASURED` row in `benchmark_evidence.json`. Currently:

| Model id                                | Container                | Backend   | Decode tok/s | TTFT (ms) | Evidence file |
|-----------------------------------------|--------------------------|-----------|--------------|-----------|---------------|
| DeepSeek-V4-Pro-Qwen3.5-4B-HyperVSQ2   | HyperVSQ-2 (4B)          | CPU       | 8.154199     | Unavailable | `benchmark_evidence.json` (canonical 4B, model SHA `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`) |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_0 (this run) | Q4_0 (1.5B)            | CPU       | 1.718469 (64 tok) | 6556.075 | `benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_64tok.json` |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_0 (this run) | Q4_0 (1.5B)            | CPU       | 2.473610 (128 tok, 8 threads) | Unavailable | `benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_128tok_cpu.json` |

Both 1.5B rows bind to:

- Binary `D:\EcoUni\qwanto\c\qwnrun.exe`, SHA-256
  `fc9086962f9ba0fa77d758a972ccfce6a0fc04b1cce27fdbc75df04109b22881`,
  git commit `08e7486`
- Model file `D:\EcoUni\qwanto\models\qwn\DeepSeek_R1_Distill_Qwen_1.5B_Q4_0.qwn`,
  SHA-256 `65a63d636c0d3cb691bc705d4b92630a33c848516d297ae4b7e6f5477b9b04cd`
- Source GGUF `D:\EcoUni\qwanto\models\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`,
  SHA-256 `1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f`

CUDA attempt for the 1.5B Q4_0 model: `INVALID` — the qwn_cuda.dll ABI 1
`gemv_hypervsq2` path is wired for `QWN_DT_HYPER_VSQ2` weights only. A pure
`Q4_0` model that calls `--backend cuda` exits `rc=-1` with
`CUDA projection failed or returned no GPU matmul. layer 0 attn matmul failed`
(`benchmarks/evidence/windows/2026-08-22/1.5b_cuda_probe.log`). This is the
correct, fail-closed behaviour. It is **not** an indefinite UNKNOWN — it is a
**CONCLUSIVE** negative result for Q4_0 + CUDA on this model.

---

## 7. Explicit "what we won't claim" list

The following are **not** supported and any README / marketing copy that asserts
otherwise is wrong. Each is mapped to the file that asserts the refusal.

- "Qwen3.8-27B QWN model", "27B in `Q4_K`", "27B in `IQ2_M`" — refused by
  `c/tools/qwen38_qualification.py`. Evidence under `docs/qwen38-27b-evidence/`.
- "K-quant Q2_K / Q3_K as a `.qwn` output" — no writer exists. Source tensor
  is silently rejected.
- "K-quant Q8_K as a `.qwn` output" — no reader, no writer.
- "All IQ-types in `.qwn`" — explicit refusal in `qwn_convert.py:1095`.
- "TWLA / LittleBit-2 / PQuant real model run" — no verified end-to-end
  model. These names appear in `qwn_quant_plan.py` and the `build_and_run_c_tests`
  unity build for future wiring, but not in the dispatch table.
- "`--backend cuda` for non-`hyper_vsq2` model" — fail-closed.
- "Hybrid SSM / DeltaNet / MoE / MTP native inference" — no reference oracle
  committed; refusal is authoritative.
- "Speculative decoding with a missing native QWN draft" — refused at
  `c/qwn_speculative.c` (433/433 boundary tests). No acceptance rate,
  no speedup is claimed.
- "TurboQuant equivalence with the cited algorithm" — rejected by
  telemetry naming (`QWN-Q4-KV`).
- "JetSpec / SlimInfer / BitDecoding measured row" — pipeline is built but no
  end-to-end model is run.

If a future commit removes any of these refusals, it must update this
document in the same commit, and `PROJECT_STATE.md` must mark the new support
status with the corresponding measured evidence row.
