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
| `10` | `Q2_K`        | 84 bytes / 256 elements   | 256             | ✅                 | ✅ scalar exact block decoder | — |
| `11` | `Q3_K`        | 110 bytes / 256 elements  | 256             | ✅                 | ✅ scalar exact block decoder | — |
| `12` | `Q8_K`        | 292 bytes / 256 elements  | 256             | ✅                 | ✅ scalar exact block decoder | — |
| `13` | `IQ2_XXS`     | 66 bytes / 256 elements   | 256             | ✅                 | ✅ canonical GGML grid/sign decoder | — |
| `14` | `IQ2_XS`      | 74 bytes / 256 elements   | 256             | ✅                 | ✅ canonical GGML grid/sign decoder | — |
| `15` | `IQ3_XXS`     | 98 bytes / 256 elements   | 256             | ✅                 | ✅ canonical GGML grid/sign decoder | — |
| `16` | `IQ3_S`       | 110 bytes / 256 elements  | 256             | ✅                 | ✅ canonical GGML grid/sign decoder | — |
| `17` | `IQ2_S`       | 82 bytes / 256 elements   | 256             | ✅                 | ✅ canonical GGML grid/sign decoder | — |
| `18` | `IQ4_NL`      | 18 bytes / 32 elements    | 32              | ✅                 | ✅ canonical GGML codebook decoder | — |
| `19` | `IQ4_XS`      | 136 bytes / 256 elements  | 256             | ✅                 | ✅ canonical GGML scale/codebook decoder | — |

Interpretation:

- ✅ — implemented, tested, on the hot path.
- 🟡 — declared in the dispatch table but no end-to-end model path exists for
  the 1.5B / 4B / 27B samples we have. Calling `--backend cuda` against a
  Q4_0-only model today fails closed with `layer N attn matmul failed`. See
  `benchmarks/evidence/windows/2026-08-22/1.5b_cuda_probe.log`.
- — — not in the dispatch path. The runtime **rejects** it from inference,
  even if a tensor of that dtype is technically declared.

The 4 KiB-header container invariants from `AGENTS.md` apply to **all** dtypes
above: descriptors and payloads are validated before any tensor is touched.
Q4_K/Q5_K/Q6_K remain source-only formats and cannot be stored as native QWN
dtypes.

---

## 2. Conversion `--quant` choices (the writer side)

`c/tools/qwn_convert.py` line 1491 is the source of truth for what the CLI
exposes. The `--quant` choices are exhaustive:

| CLI flag          | QWN container dtype | Status        | Use case                          |
|-------------------|---------------------|---------------|-----------------------------------|
| `q4_0`            | `QWN_DT_Q4_0`       | ✅ Supported  | Default. Works for every dense transformer that fits 2-D row geometry. |
| `q8_0`            | `QWN_DT_Q8_0`       | ✅ Supported  | Symmetric 32-element blocks for quality-first local inference. |
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
| `none`            | source-preserving | ✅ Supported  | Preserves native Q2_K/Q3_K/Q8_K/IQ2/IQ3/IQ4 payloads; other source tensors remain subject to the existing conversion policy. |

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
| `10`           | `Q2_K`            | ✅ | `--quant none` preserves the canonical block payload as native QWN `Q2_K`; other targets dequantize to F32 first. |
| `11`           | `Q3_K`            | ✅ | `--quant none` preserves the canonical block payload as native QWN `Q3_K`; other targets dequantize to F32 first. |
| `12`           | `Q4_K`            | ✅ (read-only) | Block-dequantized to F32, then passed to the chosen `--quant`. There is **no `.qwn` writer for Q4_K.** |
| `13`           | `Q5_K`            | ✅ (read-only) | Same status as Q4_K. `_dequantize_q5_k_block` at line 820. |
| `14`           | `Q6_K`            | ✅ (read-only) | Same status. `_dequantize_q6_k_block` at line 841. |
| `15`           | `Q8_K`            | ✅ | `--quant none` preserves the canonical block payload as native QWN `Q8_K`; other targets dequantize `d * qs` to F32. The stored `bsums` are auxiliary dot-product sums and are not needed for reconstruction. |
| `20`           | `IQ4_NL`          | ✅ | `--quant none` preserves the canonical payload as native QWN IQ4_NL; target quantizations still dequantize first. |
| `21`           | `IQ3_S`           | ✅ | `--quant none` preserves the canonical payload as native QWN IQ3_S; target quantizations still dequantize first. |
| `22`           | `IQ2_S`           | ✅ | `--quant none` preserves the canonical payload as native QWN IQ2_S; target quantizations still dequantize first. |
| `23`           | `IQ4_XS`          | ✅ | `--quant none` preserves the canonical payload as native QWN IQ4_XS; target quantizations still dequantize first. |
| `16`           | `IQ2_XXS`         | ✅ | `--quant none` preserves the canonical payload as native QWN IQ2_XXS; target quantizations still dequantize first. |
| `17`           | `IQ2_XS`          | ✅ | `--quant none` preserves the canonical payload as native QWN IQ2_XS; target quantizations still dequantize first. |
| `18`           | `IQ3_XXS`         | ✅ | `--quant none` preserves the canonical payload as native QWN IQ3_XXS; target quantizations still dequantize first. |
| `19`           | `IQ1_S`           | ❌             | Refused: canonical IQ1 decoder is not yet wired. |

A mixed-IQ GGUF is accepted when every contained IQ block has a verified source
decoder. With `--quant none`, supported IQ blocks are streamed byte-for-byte
into native QWN IQ descriptors and decoded by `qwn_row_f32`; with another
target, they are streamed through F32 and re-quantized. IQ1_S remains refused.

---

## 4. Architecture-level status (Qwen3.8 / hybrid / MTP)

Independent of dtype, the converter refuses unsupported architectures
**before** any dtype machinery runs. `c/tools/qwen38_qualification.py`
formalises this; the rest of the converter calls it as a pre-flight.

| Architecture marker                       | Status   | Reason recorded                                   |
|-------------------------------------------|---------|---------------------------------------------------|
| Dense Llama-style (≤ ~32 layers, all attention) | ✅     | Pass. See `1.5B Q4_K_M` measurement in §6.        |
| Hybrid Qwen-3.5 with MTP tensors          | 🟡       | QWN conversion and CPU DeltaNet/main-path integration exist; MTP execution and quality oracle are pending. |
| Hybrid DeltaNet / SSM + attention mix (Qwen3.8-27B) | 🟡 | Local Q4_0 conversion and one-token CPU main-path integration are verified; native IQ, MTP, CUDA, quality, and benchmark gates remain. |
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

- "Qwen3.8-27B full architecture support" — not yet claimed: the local Q4_0
  conversion and CPU main path are integration-verified, but MTP, native IQ,
  CUDA, quality, and benchmark gates remain. Historical evidence is under
  `docs/qwen38-27b-evidence/`.
- "Q4_K/Q5_K/Q6_K as native `.qwn` dtypes" — these source types are decoded
  and re-quantized into an existing QWN dtype; they are not native QWN runtime
  dtypes. Q2_K/Q3_K/Q8_K are now native QWN dtypes with scalar kernels.
- "IQ1 as a source conversion" — IQ1_S remains refused pending its
  canonical decoder.
- "All IQ-types as native `.qwn` dtypes" — IQ2/IQ3/IQ4 are source readers,
  not native QWN runtime dtypes; IQ1 remains refused.
- "TWLA / LittleBit-2 / PQuant real model run" — no verified end-to-end
  model. These names appear in `qwn_quant_plan.py` and the `build_and_run_c_tests`
  unity build for future wiring, but not in the dispatch table.
- "`--backend cuda` for non-`hyper_vsq2` model" — fail-closed.
- "Native MoE/MTP inference" — no QWN MoE dispatcher or MTP prediction path is
  wired and validated yet. DeltaNet CPU main-path execution is separate and
  has been integration-tested on the local Qwen3.8 conversion.
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
