# Qwen3.8-27B qualification — current integration status

The original fail-closed source qualification was generated from commit
`a1984025880cd2de41c442472f1b2c951b882b5f`. The current working tree has since
added the source conversion contract and a native CPU main-path integration
for the same local source artifact:

`models/Qwen3.8-27B-UD-IQ2_M.gguf`

The source remains an offline conversion input. The current local QWN artifact
is `models/Qwen3.8-27B-Q4_0.qwn`; it is an ignored local diagnostic artifact,
not benchmark evidence or a release asset.

## Decision

**`CPU_MAIN_PATH_INTEGRATION_VERIFIED`**

This status is intentionally narrower than full Qwen3.8 support. The local
Q4_0 conversion and one-token native CPU run execute the main transformer path,
including full attention and Gated DeltaNet recurrent layers. Independent
blockers remain:

1. MTP tensors are preserved by conversion but the native decoder does not yet
   execute the MTP prediction head or speculative MTP transaction path.
2. The local artifact is Q4_0; native QWN IQ2/IQ3/IQ4 payload preservation and
   row decoding are now verified independently, but they are not yet integrated
   into this full hybrid model path. Q2_K/Q3_K/Q8_K have native scalar QWN
   payload support.
3. MoE dispatch, hybrid CUDA execution, independent quality/reference-oracle
   validation, and benchmark evidence remain unavailable.

Therefore this is not a full Qwen3.8 support or performance claim.

## Source inspection

| Field | Observed value |
|---|---:|
| GGUF version | 3 |
| GGUF tensor count | 866 |
| File size | 10,319,907,904 bytes |
| SHA-256 | `04A89EF4FA9C8726D09331433346809BBAB692B4851D49D0738BA8D58A1AE740` |
| Architecture | `qwen35` |
| Hidden size | 5120 |
| Intermediate size | 17408 |
| Vocabulary | 248,320 |
| Context metadata | 262,144 (not attempted) |
| Attention heads / KV heads | 24 / 4 |
| Key/value head length | 256 / 256 |
| Gated DeltaNet state size | 128 |
| DeltaNet groups / inner size | 48 / 10240 |
| MTP prediction layers | 1 |
| Chat template | present; includes optional image/video branches |
| Vision tensors | none |
| LM head | separate `output.weight` and `token_embd.weight` |

Layer placement was read from tensor names, not inferred from the filename:

- Gated DeltaNet/SSM layers: `0,1,2,4,5,6,...,60,61,62` — 48 layers.
- Full-attention layers: `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,64` —
  17 layers.
- MTP tensors: `blk.64.nextn.eh_proj.weight`, `enorm.weight`, `hnorm.weight`,
  and `shared_head_norm.weight`.
- The chat template has image/video markers, but no vision tensors are in this
  file. Text-only qualification would explicitly reject those branches; it
  does not make the hybrid text runtime supported.

## Required tensor coverage

The machine-readable manifest contains one entry for every source tensor,
including source name, expected QWN destination name, shape, source dtype and
block geometry, owning layer, runtime operator, planned destination dtype, and
separate CPU/CUDA implementation statuses:

[architecture-coverage.json](qwen38-27b-evidence/architecture-coverage.json)

The historical manifest reports `coverage_complete=true` for the inspected
source tensor table. Current conversion uses the expanded tensor mapping in
`c/tools/qwn_convert.py`; no partial model is represented as a valid QWN model.

## Conversion and native integration

The converter was exercised against the real source with `--quant q4_0` and
produced `models/Qwen3.8-27B-Q4_0.qwn`. Structural inspection confirmed the
4 KiB header/alignment contract, 868 QWN tensors including configuration and
tokenizer metadata, and `ssm_inner=10240`.

~~~text
qwnrun result: status=ok tokens=1 wall_seconds=24.610000
backend=cpu kernel=avx2-fma-f16c-forced kv_cache_mode=fp16
~~~

This is a single local integration diagnostic. It is not a benchmark row; no
performance evidence file was changed and no throughput claim is made.

- source quantization support: IQ2/IQ3/IQ4 source decoding is available;
- target conversion state: `CONVERTED_Q4_0_LOCAL`;
- native target dtype state: Q4_0 main path verified; supported native IQ row kernels are verified independently;
- managed destination model directory: the Qwanto Code OS app-data model
  directory, never the installation directory.

[conversion-feasibility.json](qwen38-27b-evidence/conversion-feasibility.json)

## Hardware fit

Target hardware was recorded as Ryzen 9 9955HX, 32 GiB RAM, RTX 5070 Ti Laptop
GPU with 12,227 MiB reported VRAM, Windows 11, and CUDA 13.3. FP16 KV estimates
were computed for the 17 full-attention layers only:

| Context | FP16 KV estimate |
|---:|---:|
| 4096 | 285,212,672 bytes |
| 8192 | 570,425,344 bytes |
| 16384 | 1,140,850,688 bytes |
| 32768 | 2,281,701,376 bytes |

The Gated DeltaNet recurrent state is allocated by the current CPU decoder,
but complete VRAM residency, RAM placement, long-context streaming, or
production memory fit cannot be proven by the one-token run. Context 262K was
not attempted. CUDA hybrid execution remains unavailable.

[hardware-fit.json](qwen38-27b-evidence/hardware-fit.json)

The existing 4B CUDA record showing 463,370,240 resident bytes is not a 27B
estimate. It counts only tensors accepted and uploaded by the current
HyperVSQ-2 projection ABI; it is not full-file residency and cannot be used to
claim Qwen3.8 coverage.

## Correctness and agent quality

The correctness oracle and agent evaluation were not run. The one-token native
run proves execution of the current main path, not model-quality parity.
Acceptance criteria were defined in advance in the machine-readable report:
tensor finite/error bounds, layer cosine/error bounds, top-k overlap, KL
divergence, greedy agreement over 100 prompts, chat-template parity, tool-call
JSON validity, and deterministic coding-task records.

- [correctness.json](qwen38-27b-evidence/correctness.json)
- [agent-quality.json](qwen38-27b-evidence/agent-quality.json)

Both are `UNAVAILABLE_NOT_RUN` and explicitly record why. No external runtime
was invoked and none is bundled or selectable by production code.

## Benchmark evidence

No Qwen3.8 benchmark was executed. The historical evidence record has no
promoted performance row; the local integration run is deliberately excluded
from benchmark evidence. Contexts 4096 and 8192 remain future measurement
gates; 262K is outside this qualification phase.

[benchmark-evidence.json](qwen38-27b-evidence/benchmark-evidence.json)

## Validation boundary

The local qualification tests cover source tensor/header contracts, native K
dequantization, real-source conversion metadata, and the native CPU decoder
integration. Full MTP, MoE, CUDA hybrid, model-quality, Rust/Tauri,
and hosted CI validation remain separate gates. No README performance values,
benchmark evidence, tag, or release were changed by this integration work.
