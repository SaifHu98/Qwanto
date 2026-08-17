# Typed KV-cache contract

The native runtime now carries the KV-cache choice in `QwnRuntimeConfig` and
reports the representation only after the decoder has appended and read a
cache. The default is FP16. A command-line option or a UI selection is not
runtime evidence by itself.

## Supported states

| Requested mode | Implemented representation | Local status | Default |
|---|---|---|---|
| `fp16` | FP16 paged/native cache | `VALIDATED` on the existing 4B path | Yes |
| `q8` | Symmetric int8, one FP32 scale per 64 values | `ATTENTION_CORRECT` on CPU; CUDA Q8 reference is `KERNEL_CORRECT` on the RTX host | No |
| `turboquant-q4` | QWN-Q4-KV compatibility representation | `REFERENCE_IMPLEMENTED` on CPU; it is not claimed to reproduce the cited TurboQuant algorithm | No |
| `auto` | FP16 until a measured policy selects another validated mode | `VALIDATED` as the safe default policy | No |

The requested spelling `turboquant-q4` is retained for configuration
compatibility. When that representation executes, telemetry reports
`kv_cache_mode_actual=qwn-q4-kv` and `kv_cache_algorithm=QWN-Q4-KV-scalar`.
This prevents an arbitrary Q4 cache from being presented as the research
algorithm TurboQuant.

## Versioned byte contract

`QwnKvCacheContract` in `c/qwanto_turboquant.h` is ABI version 1. It contains
the structure size, ABI version, cache dtype, block size, scale and zero-point
widths, key/value layouts, page size, alignment, and valid token count.

For Q8, each token is stored head-major as signed int8 values. Every
contiguous 64-value block has one FP32 symmetric scale:

```text
scale = max(abs(finite(values))) / 127, or 1.0 for an all-zero block
q     = round-to-nearest-even(clamp(value / scale, -127, 127))
value = q * scale
```

Non-finite inputs are encoded as zero. Partial final blocks retain their
valid-count metadata and do not read beyond the input. Append, overwrite by
reset-and-append, reset, and context rollover are bounded by the configured
token capacity. The scalar CPU attention reader is the reference oracle.

The CUDA Q8 reference uses the same per-64 scale and quantization rule. It
allocates the cache once, uploads each appended token, and reports kernel,
upload, transfer, and resident-byte counters. Explicit CUDA fails closed when
the required Q8 ABI functions are absent; `auto` may retain CPU execution with
the structured reason.

## Validation boundary

`c/tests/test_kv_cache.c` covers the typed contract and CPU Q8 attention.
`c/tests/test_cuda_q8_kv.cu` compares the CUDA reference reader against the
same quantized CPU oracle. The real 4B CUDA decoder still has a separate
HyperVSQ-2 projection coverage gate; a passing isolated KV test does not turn
the complete model into a CUDA-measured workload.

Quantized KV modes are not enabled by default and no memory saving or quality
claim is published until 512, 4096, and 16384-token quality and stability
records exist for the exact executable and model hashes.
