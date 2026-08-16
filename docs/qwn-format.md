# 📦 `.qwn` Container Format & Architecture Specification

## 1. Overview & Container Invariants

The `.qwn` format is Qwanto's native container format designed for zero-copy memory mapping (`mmap`), SIMD register alignment, and high-throughput multi-tier memory offloading (GPU VRAM → RAM → NVMe SSD).

### Core Invariants:
- **4 KiB Header Alignment**: The container header is padded to exactly 4,096 bytes to align with standard OS virtual memory page boundaries.
- **64-Byte Payload Padding**: Tensor data blocks are aligned to 64-byte boundaries to enable unaligned-free vector loads on AVX-512, AVX-VNNI, AVX2, and ARM NEON registers.
- **Zero Materialization Overhead**: Weight tensors remain memory-mapped directly from NVMe/RAM; no intermediate allocation or float unrolling is performed during inference.

---

## 2. Binary Layout

```
+-------------------------------------------------------------------------+
| Header (4096 Bytes / 4 KiB)                                             |
|  - Magic: 0x51574E32 ("QWN2") / 0x434F4C49 ("COLI")                    |
|  - Version: uint32                                                      |
|  - n_tensors: uint32                                                    |
|  - n_layers: uint32                                                     |
|  - total_payload_bytes: uint64                                          |
|  - Reserved Padding: 4072 bytes (zeros)                                 |
+-------------------------------------------------------------------------+
| Tensor Descriptor Table (n_tensors * sizeof(QwnTensorEntry))            |
|  - name[64]: ASCII tensor identifier                                    |
|  - dtype: uint32 (Enum QwnDataType)                                     |
|  - n_dims: uint32                                                       |
|  - shape[4]: uint64[4]                                                  |
|  - offset_bytes: uint64 (relative to payload start)                     |
|  - size_bytes: uint64                                                   |
+-------------------------------------------------------------------------+
| 64-Byte Alignment Padding                                               |
+-------------------------------------------------------------------------+
| Tensor Payloads (64-byte aligned SIMD memory blocks)                    |
|  - Tensor 0 (e.g. model.embed_tokens.weight)                            |
|  - Tensor 1 (e.g. model.layers.0.self_attn.q_proj.weight)               |
|  - ...                                                                  |
+-------------------------------------------------------------------------+
```

---

## 3. Supported Quantization Formats & Data Types

| Data Type Identifier | Enum Value | BPW | Kernel Path | Description |
|---|:---:|:---:|---|---|
| `QWN_DTYPE_TWLA_158` | `4` | **1.58** | AVX-VNNI / BitDecoding | Ternary weights $\{-1, 0, +1\}$ packed 2-bit representations for ultra-low footprint (<1.2 GB RAM). |
| `QWN_DTYPE_HYPER_VSQ2` | `3` | **2.3125** | AVX-VNNI / AVX2 | Packed 74-byte superblocks (256 elements) with runtime zero-point compensation. |
| `QWN_DTYPE_TURBOQUANT` | `5` | **3.50** | AVX-512 / AVX-VNNI | Online asymmetric KV-cache quantization yielding 4.0x–4.57x memory reduction. |
| `QWN_DTYPE_Q4_0` | `2` | **4.50** | SIMD Maddubs / CUDA | Standard 4-bit integer quantization with FP16 group scaling. |
| `QWN_DTYPE_FP16` | `1` | **16.0** | F16C / FMA / Half | Unquantized IEEE 754 half-precision float tensor. |
| `QWN_DTYPE_FP32` | `0` | **32.0** | Scalar / FMA | Standard single-precision float tensor. |

---

## 4. Architecture Support & Boundaries

### 🟢 Natively Supported Architectures:
- **Dense Transformer**: Llama 2/3, Qwen 2/2.5, DeepSeek-Dense, Mistral, Gemma architectures.
- **Layer Execution**: Self-Attention (Rotary/RoPE), SwiGLU / GeLU activations, RMSNorm, PagedAttention.

### 🟡 Specialist MoE Runtimes:
- **DeepSeek 671B / GLM-5.2 (744B)**: Handled via dedicated specialist C runtimes (`c/glm.c`, `c/qwanto_spectral.c`) utilizing BVH spatial routing.
- **OLMoE**: Handled via `c/olmoe.c`.

### 🔴 Explicitly Rejected / Unsupported Formats:
- **Unverified GGUF Quantization Blocks**: `Q2_K`, `Q3_K`, `Q8_K`, `IQ1_S`, `IQ2_XXS`, `IQ3_S` are rejected at ingestion time to prevent incorrect scalar fallbacks or corrupted outputs.
- **Hybrid SSM / Mamba Layers**: Fail-fast validation prevents running non-transformer SSM layers in `qwnrun`.
