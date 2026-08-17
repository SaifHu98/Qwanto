# HyperVSQ-2 CUDA design and preflight — 2026-08-17

## Current status

`END_TO_END_VALIDATED` locally, pending hosted validation. The host has an
NVIDIA GeForce RTX 5070 Ti Laptop GPU (device 0, compute capability 12.0,
driver 592.02, 12,227 MiB total and 11,944 MiB free at preflight). CUDA 13.3,
MSVC 19.44, CMake 4.4.2, and Ninja 1.13.2 are installed and the exact
HyperVSQ-2 reference path has been compiled and exercised.

```text
nvcc: 13.3.73
cl.exe: 19.44.35228
cmake: 4.4.2
ninja: 1.13.2
CUDA architecture: sm_120 (detected from the installed device)
```

The source-level CMake/Ninja tools are available, although this repository's
current Windows CUDA target is a direct NVCC/MSVC command rather than a
CMake-generated build. The DLL was built beside the host executable, and both
the synthetic ABI test and the real-model CPU-vs-CUDA decoder comparison passed.
The short persistent CUDA run is diagnostic local evidence only; it is not a
release-quality performance claim and is not copied into README performance
tables.

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

The real-model decoder comparison uses a documented FP32 logit absolute
tolerance of `0.1` (the GPU warp reduction and CPU reduction can accumulate
the same integer terms in a different floating-point order) plus greedy token
ID agreement across the initial token and eight subsequent forwards. The latest
clean run observed maximum absolute differences of `0.0300188065` (scalar)
and `0.0368270874` (VNNI), with zero values above tolerance and no greedy
token divergence. Scalar and VNNI CPU comparisons are separate invocations;
tolerance alone never proves correctness when token IDs diverge.

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
`END_TO_END_VALIDATED`, and `MEASURED`. The current local state is
`END_TO_END_VALIDATED`, with performance evidence still
`MEASURED_LOCAL_PENDING_HOSTED_VALIDATION`:

| Gate | Current result |
|---|---|
| Versioned ABI header/host loader | Implemented; host syntax/link checked |
| NVCC DLL build | `COMPILED` — `qwn_cuda.dll`, ABI v1, `sm_120` |
| Synthetic CUDA correctness | `KERNEL_CORRECT` — packed patterns, scales, tails, GEMM, telemetry |
| Real tensor CPU-vs-CUDA correctness | `END_TO_END_VALIDATED` locally — scalar and VNNI, 9 greedy forwards |
| Model upload/residency | `VALIDATED` locally — 64 resident tensors, 463,370,240 uploaded bytes |
| Persistent prefill/decode | `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION` — seven-request short diagnostic |
| GPU matmul count / VRAM telemetry | `26,496` cumulative launches, `0` CPU fallbacks in the final request |

Only a real NVCC build followed by the CUDA ABI test and full decoder
validation may promote these states. A loaded DLL alone is not sufficient;
`backend_actual=CUDA` is valid only after telemetry proves a successful model
matmul.

## Local evidence records

The clean CUDA diagnostic record is
`benchmarks/evidence/windows/2026-08-17/cuda-phaseB-clean-4d26cdc/cuda-release-short-diagnostic.json`.
It was generated at commit `6b7cf1a4053bbc6f6c7c0e39d445d9148034c600` with
`git_worktree_dirty=false`, executable SHA-256
`22c64df6aa6a80eab5abe177b87f19ae7b920c2772b06a0a852267ba723a335f`, model
SHA-256 `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`,
and CUDA DLL SHA-256
`7bf5e3e595ed47fba17b79ccf74c334c42b53ba7ea5766738d721e372edd4dcb`.
The seven-request short diagnostic reports median decode `20.192933 tok/s`
and median prefill `19.126577 tok/s`; these are local pending-hosted evidence,
not README or release claims. Clean CPU records regenerated after the native
follow-up are next to it as `cpu-release-64.json` and `cpu-release-128.json`.

## Build commands when the toolkit is available

From `c/` in a CUDA/MSVC x64 environment:

```text
make qwn-cuda-dll
make cuda-hypervsq2-test
make cuda-hypervsq2-decoder-test
```

The build must use a detected device-supported architecture; `CUDA_ARCH=native`
must not be replaced by a guessed architecture flag. CUDA benchmark evidence
must include the DLL SHA-256, device/driver, resident bytes, upload bytes,
kernel type, GPU matmul count, and zero required-layer CPU fallbacks.
