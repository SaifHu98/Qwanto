# HyperVSQ-2 CUDA design and preflight — 2026-08-17

## Current status

`UNAVAILABLE` locally. The host has an NVIDIA GeForce RTX 5070 Ti Laptop GPU
(device 0, compute capability 12.0, driver 592.02, 12,227 MiB total and
11,944 MiB free at preflight), but no CUDA Toolkit compiler is installed:

```text
nvcc: NOT FOUND
cl.exe: NOT FOUND
cmake: NOT FOUND
ninja: NOT FOUND
CUDA_PATH/CUDA_HOME: unset
```

`nvidia-smi` and driver detection are hardware inventory only. No CUDA DLL was
built, no CUDA kernel was launched, no GPU matmul was observed, and no CUDA
performance result is reported.

## ABI boundary

`c/cuda/qwn_cuda_abi.h` defines ABI version 1. Every public structure starts
with `QwnCudaAbiHeader { struct_size, abi_version, reserved[4] }`. The contract
contains version/capability queries, device enumeration, explicit context
selection, tensor upload/release, HyperVSQ-2 GEMV, prefill GEMM, synchronization,
telemetry, and last-error retrieval.

The loader in `c/qwanto_decode.c`:

- resolves only `qwn_cuda.dll` beside the application/runtime executable;
- uses `LoadLibraryExA` with DLL-directory search flags on Windows;
- does not accept an arbitrary `QWANTO_CUDA_DLL` path;
- hashes the loaded file before use;
- rejects legacy/unversioned exports and ABI/layout mismatches;
- fails closed for `backend=cuda` and retains a structured CPU fallback reason
  for `backend=auto`.

The typed runtime configuration carries backend, device index, context size,
and an optional `--gpu-memory-budget-mb` limit. Model weights are uploaded once
per tensor handle and are required to fit the configured budget before a GPU
matmul can be reported.

## Exact kernel contract

The new `c/cuda/qwn_hypervsq2_cuda_abi.cu` reference path is separate from the
legacy CUDA sources. It targets only the validated QWN 2.31 HyperVSQ-2 block:

```text
fp16 d_base       2 bytes
fp16 m_base       2 bytes
8 x 4-bit scales  4 bytes
reserved/mask     2 bytes
256 x 2-bit q     64 bytes
total            74 bytes
```

The reference GEMV/GEMM consumes the decoder's symmetric int8 activation
representation and its FP32 scale. This preserves the CPU decoder's actual
activation semantics instead of comparing a raw FP32 activation product with a
quantized CPU result. It applies each octant's scale independently, preserves
the offset term, handles partial final blocks, and accumulates in FP32 after
the integer dot products.

`c/tests/test_qwn_hypervsq2_cuda_abi.cu` covers all packed two-bit patterns,
multiple scales, reserved bytes, random signed int8 activations, a non-aligned
513-column tail, batched GEMM, residency counters, and CPU-equivalent reference
values. It returns a device/toolkit skip when no CUDA device is available; a
skip is not CUDA evidence.

The older `c/cuda/qwn_hypervsq_cuda.cu` exports an unversioned process-wide
interface and is not used by the new `qwn-cuda-dll` target. The separate
`backend_cuda.cu` path contains older general/legacy quantized kernels and is
not reused as the HyperVSQ-2 74-byte implementation. Neither source is valid
evidence for the new ABI or for full-model CUDA execution.

## Evidence ladder

The intended states are `UNAVAILABLE`, `COMPILED`, `KERNEL_CORRECT`,
`END_TO_END_VALIDATED`, and `MEASURED`. The current state is `UNAVAILABLE`:

| Gate | Current result |
|---|---|
| Versioned ABI header/host loader | Implemented; host syntax/link checked |
| NVCC DLL build | `NOT RUN LOCALLY — HOSTED VALIDATION REQUIRED` |
| Synthetic CUDA correctness | `NOT RUN LOCALLY — HOSTED VALIDATION REQUIRED` |
| Real tensor CPU-vs-CUDA correctness | `NOT RUN LOCALLY — HOSTED VALIDATION REQUIRED` |
| Model upload/residency | `NOT RUN LOCALLY — HOSTED VALIDATION REQUIRED` |
| Persistent prefill/decode | `NOT RUN LOCALLY — HOSTED VALIDATION REQUIRED` |
| GPU matmul count / VRAM telemetry | `0` observed; no CUDA runtime execution |

Only a real NVCC build followed by the CUDA ABI test and full decoder
validation may promote these states. A loaded DLL alone is not sufficient;
`backend_actual=CUDA` is valid only after telemetry proves a successful model
matmul.

## Build commands when the toolkit is available

From `c/` in a CUDA/MSVC x64 environment:

```text
make qwn-cuda-dll
make cuda-hypervsq2-test
```

The build must use a detected device-supported architecture; `CUDA_ARCH=native`
must not be replaced by a guessed architecture flag. CUDA benchmark evidence
must include the DLL SHA-256, device/driver, resident bytes, upload bytes,
kernel type, GPU matmul count, and zero required-layer CPU fallbacks.
