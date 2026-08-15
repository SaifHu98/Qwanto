#ifndef QWANTO_TURBOQUANT_H
#define QWANTO_TURBOQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TURBOQUANT_GROUP_SIZE 64
/* 64 values = 4 sub-chunks of 16 values.
 * 16 values packed in 7 bytes (8 pairs of 4-bit / 3-bit codes = 56 bits = 7 bytes).
 * 4 sub-chunks = 28 bytes payload.
 * Plus 2 bytes FP16 scale + 2 bytes FP16 zero_point = 32 bytes total per 64-element block.
 * Footprint = 32 bytes / 64 values = 0.50 bytes/val = 4.0 bpw (3.5 bpw raw payload).
 */
#define TURBOQUANT_BLOCK_BYTES 32
#define TURBOQUANT_PAYLOAD_BYTES 28

#pragma pack(push, 1)
typedef struct {
    uint16_t scale_fp16;      /* FP16 adaptive scale (S) */
    uint16_t zero_point_fp16; /* FP16 zero point (Z = min) */
    uint8_t  packed_data[TURBOQUANT_PAYLOAD_BYTES]; /* 64 values packed into 28 bytes */
} TurboQuantBlock;
#pragma pack(pop)

typedef struct {
    uint8_t* packed_k;       /* 3.5-bit packed keys buffer */
    uint8_t* packed_v;       /* 3.5-bit packed values buffer */
    int n_channels;          /* Total channels per token (heads * head_dim) */
    int n_tokens;            /* Current cached token length */
    int max_tokens;          /* Allocated sequence length capacity */
    int n_heads;             /* Number of attention / KV heads */
    int head_dim;            /* Dimension per head */
    size_t token_stride_k;   /* Bytes per token for Key cache */
    size_t token_stride_v;   /* Bytes per token for Value cache */
    size_t total_bytes;      /* Total allocation size in bytes */
} TurboQuantCache;

/* Initialize TurboQuant cache for a given sequence length and geometry */
int qwn_turboquant_init(TurboQuantCache* cache, int max_tokens, int n_heads, int head_dim);

/* Free TurboQuant cache memory */
void qwn_turboquant_free(TurboQuantCache* cache);

/* Online quantization: quantize a single token's float32 Key or Value vector into TurboQuant blocks */
void qwn_turboquant_quantize_token(const float* src, uint8_t* dst_blocks, int n_channels);

/* Scalar reference oracles for differential testing */
float qwn_turboquant_dot_key_scalar(const float* query, const uint8_t* key_blocks, int dim);
void qwn_turboquant_accum_value_scalar(float score, const uint8_t* value_blocks, float* ctx, int dim);

/* Vectorized AVX2 kernels */
float qwn_turboquant_dot_key_avx2(const float* query, const uint8_t* key_blocks, int dim);
void qwn_turboquant_accum_value_avx2(float score, const uint8_t* value_blocks, float* ctx, int dim);

/* High-throughput AVX-VNNI kernels */
float qwn_turboquant_dot_key_vnni(const float* query, const uint8_t* key_blocks, int dim);
void qwn_turboquant_accum_value_vnni(float score, const uint8_t* value_blocks, float* ctx, int dim);

/* AVX-512 kernels */
float qwn_turboquant_dot_key_avx512(const float* query, const uint8_t* key_blocks, int dim);
void qwn_turboquant_accum_value_avx512(float score, const uint8_t* value_blocks, float* ctx, int dim);

/* Main entry point for TurboQuant attention compute for a single head */
void qwn_turboquant_attention_head(
    const float* query,
    const TurboQuantCache* cache,
    int layer,
    int head_idx,
    int kv_head_idx,
    int pos,
    float scale,
    float* scores_scratch,
    float* ctx_out
);

/* Top-level AVX-512 multi-head attention matrix multiplication */
void qwn_turboquant_matmul_avx512(
    const TurboQuantCache* cache,
    const float* query,
    float* output,
    int n_heads,
    int head_dim
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_TURBOQUANT_H */
