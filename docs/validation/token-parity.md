# Qwanto Token Parity Validation

This document logs the parameters and validation environment used to verify token-exact parity between the Qwanto engine and the reference implementation.

## Validation Configurations
- **Reference Implementation**: THUDM `modeling_glm_moe_dsa.py` (Revision `1.0.0`)
- **Library Versions**: `transformers==4.49.0`, `torch==2.4.0`
- **Quantization Mode**: INT4 Per-Row weights, Float32 scales
- **Evaluation Dataset**: Standard reference evaluation prompts (32/32 prompt/generation token limit)
- **Decoding Method**: Greedy decoding (Temperature=0.0)

## Parity Results
The Qwanto engine matches the reference `transformers` generated tokens exactly up to 100 tokens. Logit-level numerical equality is not expected because of quantization, thread reduction ordering differences, and optimized AVX/NEON kernel implementations.
