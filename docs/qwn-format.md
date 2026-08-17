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

The dtype enum is defined by `qwn_container.h` and the conversion tool. Common
values are FP32 (`0`), FP16 (`1`), Q4_0 (`2`), HyperVSQ-2 (`3`), TWLA 1.58-bit
(`4`), and TurboQuant (`5`). A dtype name or bits-per-weight claim is valid only
when it comes from the container descriptor or a conversion artifact.

## Runtime relationship

`qwnrun` memory-maps the container and uses the project’s VRAM → RAM → NVMe
residency model. The Tauri package includes the executable, not the container.
The model registry marks malformed containers invalid and marks GGUF,
Safetensors, and PyTorch files as conversion sources rather than silently
treating them as native QWN.
