# GLM-5.2 Model Architecture Manifest

This document records the exact architectural specifications and parameters for the GLM-5.2 model (`glm_moe_dsa` architecture) supported by Qwanto.

## Parameter Counts
- **Total Parameters**: 744 Billion
- **Active Parameters per Token**: ~40 Billion
- **Dense Resident Parameters**: ~17 Billion (approx. 9.9 GiB in int4 quantization)
- **Routed Expert Parameters**: 19,456 experts total, consisting of:
  - 75 MoE layers × 256 experts per layer = 19,200 experts.
  - 256 experts in the Multi-Token Prediction (MTP) head (Layer 78) = 256 experts.
  - Total = 19,456 routed experts.
- **Routed Weight size**: ~19 MiB per expert in int4.

## Layer Structure
- **Dense Layers**: First 3 layers are fully dense.
- **MoE Layers**: Remaining layers utilize sparse Mixture of Experts (MoE) with DeepSeek-style routers.
- **Multi-Token Prediction (MTP)**: Active at Layer 78 for speculative decoding.

## Attention Mechanisms
- **MLA (Multi-head Latent Attention)**: q/kv-LoRA projection.
- **RoPE (Rotary Position Embeddings)**: Interleaved partial RoPE.
- **DSA (Dense-Sparse Attention)**: Dynamic sparse indexing.
