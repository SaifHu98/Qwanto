# Converter capability matrix

The machine-readable source of truth is
[`converter-capability-matrix.json`](converter-capability-matrix.json). The
four axes are intentionally independent: parsing a container does not imply
that its tensor dtype can be dequantized, and conversion does not imply that
the native runtime can execute the resulting operators.

The converter currently supports exact scalar handling for the listed F32,
F16, BF16, Q4_0, Q4_K, Q5_K, and Q6_K paths where the tensor descriptor and
shape meet their implemented rules. Q8_K and IQ2/IQ3/IQ4 are unsupported
source dtypes in the current converter. An unknown block layout fails closed.

Qwen3.8/Qwen3.5 hybrid models remain
`UNSUPPORTED_QWEN38_ARCHITECTURE`. Their required Gated DeltaNet state,
hybrid scheduling, MTP tensors, and mixed IQ dtypes are not silently skipped
or reinterpreted. The qualification evidence under
[`qwen38-27b-qualification.md`](qwen38-27b-qualification.md) remains the authority.

Conversion is a streaming source-block → canonical FP32/FP16 chunk → QWN
quantization pipeline where the source decoder supports it. Publication is
temporary-file plus validation plus atomic rename; a source artifact is never
activated through `qwnrun`.
