# Qwen3.8-27B qualification — fail-closed result

Generated from commit `e75acee232ce5ad139e4183bd909f4167c58707e` for the
local source artifact:

`models/Qwen3.8-27B-UD-IQ2_M.gguf`

The source is an offline conversion input only. It was not activated through
`qwnrun`, no QWN output was created, and no 31.5 GB Q8 file was downloaded.

## Decision

**`UNSUPPORTED_QWEN38_ARCHITECTURE`**

The decision is fail-closed and has two independent blockers:

1. The file is a `qwen35` hybrid model. It contains 65 layers, 48
   Gated DeltaNet/SSM state-bearing layers, 17 full-attention layers, and
   four MTP tensors in layer 64. The current native decoder does not have a
   validated Gated DeltaNet recurrent-state, hybrid scheduling, or MTP
   execution path.
2. The file advertises GGUF file type 14 (`IQ2_M`) but its tensor table is a
   mixed quantization set: Q2_K/Q3_K, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, F32, and
   BF16. The current converter has no exact decoder for the IQ tensor types;
   none may be reinterpreted as another QWN dtype.

This is not a Qwen3.8 support claim. The valid future outcome remains blocked
until every required operator, tensor mapping, tokenizer path, correctness
oracle, and hardware placement plan is implemented and tested.

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
| DeltaNet groups / inner size | 16 / 6144 |
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

The manifest reports `coverage_complete=true`, `source_tensor_count=866`, and
`conversion_status=BLOCKED_BEFORE_OUTPUT`. No partial model is represented as a
valid QWN model.

## Conversion feasibility

The converter was exercised against the real source with `--quant hyper_vsq2`.
It failed before writing an output:

~~~text
ValueError: native QWN conversion is unavailable until the architecture is fully implemented:
Qwen3.5 hybrid Transformer/SSM execution, SSM state/tensor execution
~~~

The target was confirmed absent after the failure. The report also records the
actual unsupported source dtype IDs and a clearly labelled projected size. The
projection is not a converted artifact or a performance claim:

- projected QWN payload: 7,907,308,544 bytes;
- projected QWN container size: 7,908,225,056 bytes;
- source quantization support: `UNSUPPORTED_SOURCE_QUANTIZATION` for the IQ
  dtypes present;
- conversion state: `REFUSED_BEFORE_CONVERSION`;
- managed destination model directory: the Qwanto Code OS app-data model
  directory, never the installation directory.

[conversion-feasibility.json](qwen38-27b-evidence/conversion-feasibility.json)

## Hardware fit

Target hardware was recorded as Ryzen 9 9955HX, 32 GiB RAM, RTX 5070 Ti Laptop
GPU with 12,227 MiB reported VRAM, Windows 11, and CUDA 13.3. FP16 KV estimates
were computed for the 17 full-attention layers only:

| Context | FP16 KV estimate |
|---:|---:|
| 4096 | 1,090,519,040 bytes |
| 8192 | 2,181,038,080 bytes |
| 16384 | 4,362,076,160 bytes |
| 32768 | 8,724,152,320 bytes |

The Gated DeltaNet recurrent-state size is intentionally `UNAVAILABLE`; the
native state layout is not implemented. Therefore complete VRAM residency,
RAM placement, or measured streaming cannot be proven. Context 262K was not
attempted. The report is classified `HARDWARE_FIT_FAILED` for qualification,
not as evidence that a future complete implementation could never fit.

[hardware-fit.json](qwen38-27b-evidence/hardware-fit.json)

The existing 4B CUDA record showing 463,370,240 resident bytes is not a 27B
estimate. It counts only tensors accepted and uploaded by the current
HyperVSQ-2 projection ABI; it is not full-file residency and cannot be used to
claim Qwen3.8 coverage.

## Correctness and agent quality

The correctness oracle and agent evaluation were not run. Running them before
a valid conversion would make skipped layers or a partial QWN appear valid.
Acceptance criteria were defined in advance in the machine-readable report:
tensor finite/error bounds, layer cosine/error bounds, top-k overlap, KL
divergence, greedy agreement over 100 prompts, chat-template parity, tool-call
JSON validity, and deterministic coding-task records.

- [correctness.json](qwen38-27b-evidence/correctness.json)
- [agent-quality.json](qwen38-27b-evidence/agent-quality.json)

Both are `UNAVAILABLE_NOT_RUN` and explicitly record why. No external runtime
was invoked and none is bundled or selectable by production code.

## Benchmark evidence

No Qwen3.8 benchmark was executed. The evidence record has no executable hash,
QWN model hash, backend result, token rate, or CUDA matmul claim. It is
classified `UNAVAILABLE` and records contexts 4096 and 8192 as future gates;
262K is outside this qualification phase.

[benchmark-evidence.json](qwen38-27b-evidence/benchmark-evidence.json)

## Validation boundary

The local qualification tests cover unknown-IQ rejection, hybrid conversion
failure before output, complete real-source header coverage when the fixture is
present, and the no-QWN-output invariant. Full native CUDA model validation,
Rust/Tauri, and hosted CI remain separate gates. No README performance values,
tag, or release were changed by this qualification experiment.

