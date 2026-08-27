# `.qwn` container format

`.qwn` is the native dense-transformer container consumed by `qwnrun`.
`c/qwn_container.h`, `c/qwn_container.c`, and the conversion tools are the
implementation references; this document records the invariants that the
runtime relies on.

## Layout

| Region | Rule |
| --- | --- |
| Header | Exactly 4096 bytes |
| Header fields | little-endian magic, version, tensor count, layer count, and payload byte count |
| Descriptor table | Fixed `QwnTensorEntry` records: 64-byte name, dtype, dimensions, shape, offset, and size |
| Payload | Tensor blocks are padded to 64-byte boundaries |
| Tail | The conversion format records its final block offset in the final 8 bytes |

Accepted container magic values are `QWN2`, `COLI`, and the legacy `QWN1`.
The native decoder validates the header, tensor count, descriptor bounds, and
payload bounds before using a tensor. The maximum descriptor count is the
compile-time `QWN_MAX_TENSORS` limit in `qwn_container.h`.

## Dtypes

The dtype enum is defined by `c/qwanto_native.h` (`QWN_DT_*`) and read at
runtime in `c/qwanto_native.c::dtype_name`. The mapping below is the only
authoritative one. Anything not present here is refused by the loader
before any tensor is touched; a container that lists an unknown dtype
fails with `unsupported_dtype` and no inference runs.

| ID  | Name           | Bytes per block / elements per block | Implemented? | Notes                                  |
|----:|----------------|--------------------------------------:|--------------|----------------------------------------|
| `0` | `F32`          | —                                    | ✅ Yes        | Used for LayerNorm, embeddings, raw exports. |
| `1` | `F16`          | —                                    | ✅ Yes        | Half-precision scalar/vector path.    |
| `2` | `Q4_0`         | 18 / 32                              | ✅ Yes        | Conventional 4-bit, scalar/AVX2 kernels wired in qwnrun. |
| `3` | `Q8_0`         | 34 / 32                              | ✅ Yes        | Storing/inference both supported; no measured native-inference row in the current performance report. |
| `4` | `BF16`         | —                                    | ✅ Yes        | Stored and dequantized to F32 for matmul. |
| `5` | `BYTES`        | —                                    | ✅ Yes        | Tokenizer bytes and similar opaque blobs. |
| `6` | `VSQ`          | 36 / 64                              | 🟡 Experimental | Local reference only. No current performance row. |
| `7` | `VSQ_ULTRA`    | 70 / 128                             | 🟡 Experimental | Local reference only.                  |
| `8` | `HYPER_VSQ`    | 138 / 256                            | 🟡 Experimental | Earlier-gen HyperVSQ. Distinct from HyperVSQ-2. |
| `9` | `HYPER_VSQ2`   | 74 / 256                             | ✅ Yes        | The only dtype with a release-quality CPU AVX-VNNI matmul AND a CUDA ABI 1 implementation. |
| `10` | `Q2_K`        | 84 / 256                             | ✅ Yes        | Native scalar row decoder. |
| `11` | `Q3_K`        | 110 / 256                            | ✅ Yes        | Native scalar row decoder. |
| `12` | `Q8_K`        | 292 / 256                            | ✅ Yes        | Native scalar row decoder. |
| `13` | `IQ2_XXS`     | 66 / 256                             | ✅ Yes        | Native GGML grid/sign row decoder. |
| `14` | `IQ2_XS`      | 74 / 256                             | ✅ Yes        | Native GGML grid/sign row decoder. |
| `15` | `IQ3_XXS`     | 98 / 256                             | ✅ Yes        | Native GGML grid/sign row decoder. |
| `16` | `IQ3_S`       | 110 / 256                            | ✅ Yes        | Native GGML grid/sign row decoder. |
| `17` | `IQ2_S`       | 82 / 256                             | ✅ Yes        | Native GGML grid/sign row decoder. |
| `18` | `IQ4_NL`      | 18 / 32                              | ✅ Yes        | Native GGML non-linear codebook row decoder. |
| `19` | `IQ4_XS`      | 136 / 256                            | ✅ Yes        | Native GGML scale/codebook row decoder. |

`TWLA`, `LittleBit-2`, `TurboQuant`, `JetSpec`, `SlimInfer`, and `BitDecoding`
are *runtime modules* under `c/qwanto_*.c`, not container dtype IDs. They
exist as decoder implementations or agentic frameworks, not as values
written into a `.qwn` file. A claim like `TWLA is a .qwn dtype` is wrong;
those files are reference or experimental work without published end-to-end
model evidence in any measured row. See `docs/qwn-supported-quantizations.md`
for the complete matrix and the explicit "what we will not claim" list.

A dtype name or bits-per-weight claim is valid only when it comes from the
container descriptor or a conversion artifact; never from marketing copy.


## Runtime relationship

`qwnrun` memory-maps the container and uses the project’s VRAM → RAM → NVMe
residency model. The Tauri package includes the executable, not the container.
The model registry marks malformed containers invalid and marks GGUF,
Safetensors, and PyTorch files as conversion sources rather than silently
treating them as native QWN.
