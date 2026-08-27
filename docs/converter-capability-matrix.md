# Converter capability matrix

The machine-readable source of truth is
[`converter-capability-matrix.json`](converter-capability-matrix.json). The
four axes are intentionally independent: parsing a container does not imply
that its tensor dtype can be dequantized, and conversion does not imply that
the native runtime can execute the resulting operators.

The converter currently supports exact scalar handling for the listed F32,
F16, BF16, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2, IQ3, IQ4_XS, and
IQ4_NL source paths where the tensor descriptor and shape meet their implemented
rules. IQ1 source dtypes remain unsupported. An unknown block layout fails
closed.

Qwen3.8/Qwen3.5 hybrid conversion is available for the validated local Q4_0
main-path contract. The native CPU decoder executes the converted Qwen3.8
Gated DeltaNet/full-attention layers, while MTP execution,
MoE dispatch, CUDA hybrid execution, and quality/reference-oracle validation
remain separate gates. The qualification evidence under
[`qwen38-27b-qualification.md`](qwen38-27b-qualification.md) remains the
authority; it does not promote this integration run to benchmark evidence.

Native QWN `Q2_K`, `Q3_K`, and `Q8_K` payloads are supported by the scalar
decoder path. Supported IQ2/IQ3/IQ4 payloads can be preserved as native QWN
IQ descriptors and have differential row-kernel coverage against the Python
GGML reference. This is dtype/runtime evidence only; it does not promote a
hybrid Flash-Next model to full architecture support.

Conversion is a streaming source-block → canonical FP32/FP16 chunk → QWN
quantization pipeline where the source decoder supports it. Publication is
temporary-file plus validation plus atomic rename; a source artifact is never
activated through `qwnrun`.
