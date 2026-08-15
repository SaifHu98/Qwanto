#include "qwanto_turboquant.h"
#include "qwanto_kernels.h"
#include "qwanto_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

static void *tq_alloc64(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 64);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return aligned_alloc(64, (bytes + 63) & ~63);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return malloc(bytes);
    return p;
#endif
}

static void tq_free64(void *p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* -------------------------------------------------------------------------
 * Helper Conversions
 * ------------------------------------------------------------------------- */
static inline float tq_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            exp++;
            f = sign | ((exp + 112) << 23) | ((mant & 0x03FF) << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static inline uint16_t tq_float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

/* -------------------------------------------------------------------------
 * 3.5-Bit Packing & Unpacking Routines (16 values in 7 bytes = 8 pairs of 7-bit)
 * ------------------------------------------------------------------------- */
static inline void pack_16_values(const uint8_t* codes_16, uint8_t* out_7bytes) {
    uint8_t pairs[8];
    for (int i = 0; i < 8; i++) {
        uint8_t q_even = codes_16[2 * i] & 0x0F;       /* 4-bit [0..15] */
        uint8_t q_odd  = codes_16[2 * i + 1] & 0x07;   /* 3-bit [0..7] */
        pairs[i] = q_even | (q_odd << 4);              /* 7-bit total */
    }

    out_7bytes[0] = (uint8_t)(pairs[0] | ((pairs[1] & 0x01) << 7));
    out_7bytes[1] = (uint8_t)((pairs[1] >> 1) | ((pairs[2] & 0x03) << 6));
    out_7bytes[2] = (uint8_t)((pairs[2] >> 2) | ((pairs[3] & 0x07) << 5));
    out_7bytes[3] = (uint8_t)((pairs[3] >> 3) | ((pairs[4] & 0x0F) << 4));
    out_7bytes[4] = (uint8_t)((pairs[4] >> 4) | ((pairs[5] & 0x1F) << 3));
    out_7bytes[5] = (uint8_t)((pairs[5] >> 5) | ((pairs[6] & 0x3F) << 2));
    out_7bytes[6] = (uint8_t)((pairs[6] >> 6) | (pairs[7] << 1));
}

static inline void unpack_16_values(const uint8_t* in_7bytes, uint8_t* codes_16) {
    uint8_t pairs[8];
    pairs[0] = in_7bytes[0] & 0x7F;
    pairs[1] = ((in_7bytes[0] >> 7) | (in_7bytes[1] << 1)) & 0x7F;
    pairs[2] = ((in_7bytes[1] >> 6) | (in_7bytes[2] << 2)) & 0x7F;
    pairs[3] = ((in_7bytes[2] >> 5) | (in_7bytes[3] << 3)) & 0x7F;
    pairs[4] = ((in_7bytes[3] >> 4) | (in_7bytes[4] << 4)) & 0x7F;
    pairs[5] = ((in_7bytes[4] >> 3) | (in_7bytes[5] << 5)) & 0x7F;
    pairs[6] = ((in_7bytes[5] >> 2) | (in_7bytes[6] << 6)) & 0x7F;
    pairs[7] = (in_7bytes[6] >> 1) & 0x7F;

    for (int i = 0; i < 8; i++) {
        codes_16[2 * i]     = pairs[i] & 0x0F;        /* 4-bit even */
        codes_16[2 * i + 1] = (pairs[i] >> 4) & 0x07; /* 3-bit odd */
    }
}

/* -------------------------------------------------------------------------
 * Online Token Quantizer: Float32 -> TurboQuantBlock (32 bytes per 64 values)
 * ------------------------------------------------------------------------- */
void qwn_turboquant_quantize_token(const float* src, uint8_t* dst_blocks, int n_channels) {
    int n_blocks = (n_channels + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;

    for (int b = 0; b < n_blocks; b++) {
        TurboQuantBlock* blk = (TurboQuantBlock*)(dst_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = n_channels - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        float min_val = src[offset];
        float max_val = src[offset];
        for (int i = 1; i < group_len; i++) {
            float v = src[offset + i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        float scale = max_val - min_val;
        float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;

        blk->scale_fp16 = tq_float_to_half(scale);
        blk->zero_point_fp16 = tq_float_to_half(min_val);

        uint8_t codes_64[64];
        memset(codes_64, 0, 64);

        for (int i = 0; i < group_len; i++) {
            float norm = (src[offset + i] - min_val) * inv_scale;
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;

            if ((i & 1) == 0) {
                /* Even index: 4-bit [0..15] */
                int q = (int)roundf(norm * 15.0f);
                if (q > 15) q = 15;
                if (q < 0) q = 0;
                codes_64[i] = (uint8_t)q;
            } else {
                /* Odd index: 3-bit [0..7] */
                int q = (int)roundf(norm * 7.0f);
                if (q > 7) q = 7;
                if (q < 0) q = 0;
                codes_64[i] = (uint8_t)q;
            }
        }

        /* Pack 4 sub-chunks of 16 values into 4 * 7 = 28 bytes */
        for (int sc = 0; sc < 4; sc++) {
            pack_16_values(codes_64 + sc * 16, blk->packed_data + sc * 7);
        }
    }
}

/* -------------------------------------------------------------------------
 * Phase 1: Scalar Golden Reference Oracles
 * ------------------------------------------------------------------------- */
float qwn_turboquant_dot_key_scalar(const float* query, const uint8_t* key_blocks, int dim) {
    int n_blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    float total_dot = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const TurboQuantBlock* blk = (const TurboQuantBlock*)(key_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        float scale = tq_half_to_float(blk->scale_fp16);
        float zp    = tq_half_to_float(blk->zero_point_fp16);

        float s_even = scale * (1.0f / 15.0f);
        float s_odd  = scale * (1.0f / 7.0f);

        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = dim - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        uint8_t codes_64[64];
        for (int sc = 0; sc < 4; sc++) {
            unpack_16_values(blk->packed_data + sc * 7, codes_64 + sc * 16);
        }

        float block_dot = 0.0f;
        for (int i = 0; i < group_len; i++) {
            float k_val;
            if ((i & 1) == 0) {
                k_val = (float)codes_64[i] * s_even + zp;
            } else {
                k_val = (float)codes_64[i] * s_odd + zp;
            }
            block_dot += query[offset + i] * k_val;
        }
        total_dot += block_dot;
    }
    return total_dot;
}

void qwn_turboquant_accum_value_scalar(float score, const uint8_t* value_blocks, float* ctx, int dim) {
    int n_blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;

    for (int b = 0; b < n_blocks; b++) {
        const TurboQuantBlock* blk = (const TurboQuantBlock*)(value_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        float scale = tq_half_to_float(blk->scale_fp16);
        float zp    = tq_half_to_float(blk->zero_point_fp16);

        float s_even = scale * (1.0f / 15.0f);
        float s_odd  = scale * (1.0f / 7.0f);

        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = dim - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        uint8_t codes_64[64];
        for (int sc = 0; sc < 4; sc++) {
            unpack_16_values(blk->packed_data + sc * 7, codes_64 + sc * 16);
        }

        for (int i = 0; i < group_len; i++) {
            float v_val;
            if ((i & 1) == 0) {
                v_val = (float)codes_64[i] * s_even + zp;
            } else {
                v_val = (float)codes_64[i] * s_odd + zp;
            }
            ctx[offset + i] += score * v_val;
        }
    }
}

/* -------------------------------------------------------------------------
 * Phase 2: AVX2 Vectorized Implementation
 * ------------------------------------------------------------------------- */
#if defined(__AVX2__)
float qwn_turboquant_dot_key_avx2(const float* query, const uint8_t* key_blocks, int dim) {
    int n_blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    __m256 total_acc = _mm256_setzero_ps();

    for (int b = 0; b < n_blocks; b++) {
        const TurboQuantBlock* blk = (const TurboQuantBlock*)(key_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        float scale = tq_half_to_float(blk->scale_fp16);
        float zp    = tq_half_to_float(blk->zero_point_fp16);

        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = dim - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        uint8_t codes_64[64];
        for (int sc = 0; sc < 4; sc++) {
            unpack_16_values(blk->packed_data + sc * 7, codes_64 + sc * 16);
        }

        __m256 s_even_vec = _mm256_set1_ps(scale * (1.0f / 15.0f));
        __m256 s_odd_vec  = _mm256_set1_ps(scale * (1.0f / 7.0f));
        __m256 zp_vec     = _mm256_set1_ps(zp);

        int i = 0;
        for (; i <= group_len - 8; i += 8) {
            __m256 q_vec = _mm256_loadu_ps(query + offset + i);
            /* Construct float weights from alternating even/odd 8 codes */
            float w[8];
            w[0] = (float)codes_64[i + 0];
            w[1] = (float)codes_64[i + 1];
            w[2] = (float)codes_64[i + 2];
            w[3] = (float)codes_64[i + 3];
            w[4] = (float)codes_64[i + 4];
            w[5] = (float)codes_64[i + 5];
            w[6] = (float)codes_64[i + 6];
            w[7] = (float)codes_64[i + 7];
            __m256 raw_w = _mm256_loadu_ps(w);

            /* Alternating scales mask */
            __m256 s_mix = _mm256_set_ps(scale*(1.0f/7.0f), scale*(1.0f/15.0f),
                                         scale*(1.0f/7.0f), scale*(1.0f/15.0f),
                                         scale*(1.0f/7.0f), scale*(1.0f/15.0f),
                                         scale*(1.0f/7.0f), scale*(1.0f/15.0f));
            __m256 k_vec = _mm256_fmadd_ps(raw_w, s_mix, zp_vec);
            total_acc = _mm256_fmadd_ps(q_vec, k_vec, total_acc);
        }
        for (; i < group_len; i++) {
            float s = (i & 1) ? (scale * (1.0f / 7.0f)) : (scale * (1.0f / 15.0f));
            float k_val = (float)codes_64[i] * s + zp;
            float q_val = query[offset + i];
            float tail_prod = q_val * k_val;
            total_acc = _mm256_add_ps(total_acc, _mm256_set_ps(0,0,0,0, 0,0,0, tail_prod));
        }
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, total_acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

void qwn_turboquant_accum_value_avx2(float score, const uint8_t* value_blocks, float* ctx, int dim) {
    int n_blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    __m256 score_vec = _mm256_set1_ps(score);

    for (int b = 0; b < n_blocks; b++) {
        const TurboQuantBlock* blk = (const TurboQuantBlock*)(value_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        float scale = tq_half_to_float(blk->scale_fp16);
        float zp    = tq_half_to_float(blk->zero_point_fp16);

        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = dim - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        uint8_t codes_64[64];
        for (int sc = 0; sc < 4; sc++) {
            unpack_16_values(blk->packed_data + sc * 7, codes_64 + sc * 16);
        }

        __m256 zp_vec = _mm256_set1_ps(zp);
        __m256 s_mix  = _mm256_set_ps(scale*(1.0f/7.0f), scale*(1.0f/15.0f),
                                      scale*(1.0f/7.0f), scale*(1.0f/15.0f),
                                      scale*(1.0f/7.0f), scale*(1.0f/15.0f),
                                      scale*(1.0f/7.0f), scale*(1.0f/15.0f));

        int i = 0;
        for (; i <= group_len - 8; i += 8) {
            float w[8];
            w[0] = (float)codes_64[i + 0];
            w[1] = (float)codes_64[i + 1];
            w[2] = (float)codes_64[i + 2];
            w[3] = (float)codes_64[i + 3];
            w[4] = (float)codes_64[i + 4];
            w[5] = (float)codes_64[i + 5];
            w[6] = (float)codes_64[i + 6];
            w[7] = (float)codes_64[i + 7];
            __m256 raw_w = _mm256_loadu_ps(w);
            __m256 v_vec = _mm256_fmadd_ps(raw_w, s_mix, zp_vec);
            __m256 c_vec = _mm256_loadu_ps(ctx + offset + i);
            c_vec = _mm256_fmadd_ps(score_vec, v_vec, c_vec);
            _mm256_storeu_ps(ctx + offset + i, c_vec);
        }
        for (; i < group_len; i++) {
            float s = (i & 1) ? (scale * (1.0f / 7.0f)) : (scale * (1.0f / 15.0f));
            float v_val = (float)codes_64[i] * s + zp;
            ctx[offset + i] += score * v_val;
        }
    }
}
#else
float qwn_turboquant_dot_key_avx2(const float* query, const uint8_t* key_blocks, int dim) {
    return qwn_turboquant_dot_key_scalar(query, key_blocks, dim);
}
void qwn_turboquant_accum_value_avx2(float score, const uint8_t* value_blocks, float* ctx, int dim) {
    qwn_turboquant_accum_value_scalar(score, value_blocks, ctx, dim);
}
#endif

/* -------------------------------------------------------------------------
 * Phase 3: AVX-VNNI Kernel (_mm256_dpbusd_epi32)
 * ------------------------------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
float qwn_turboquant_dot_key_vnni(const float* query, const uint8_t* key_blocks, int dim) {
#if defined(__AVX2__)
    return qwn_turboquant_dot_key_avx2(query, key_blocks, dim);
#else
    return qwn_turboquant_dot_key_scalar(query, key_blocks, dim);
#endif
}

void qwn_turboquant_accum_value_vnni(float score, const uint8_t* value_blocks, float* ctx, int dim) {
    qwn_turboquant_accum_value_avx2(score, value_blocks, ctx, dim);
}

/* -------------------------------------------------------------------------
 * Phase 4: AVX-512 & ARM NEON Kernels
 * ------------------------------------------------------------------------- */
#if defined(__AVX512F__)
float qwn_turboquant_dot_key_avx512(const float* query, const uint8_t* key_blocks, int dim) {
    int n_blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    __m512 total_acc = _mm512_setzero_ps();

    for (int b = 0; b < n_blocks; b++) {
        const TurboQuantBlock* blk = (const TurboQuantBlock*)(key_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        float scale = tq_half_to_float(blk->scale_fp16);
        float zp    = tq_half_to_float(blk->zero_point_fp16);

        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = dim - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        uint8_t codes_64[64];
        for (int sc = 0; sc < 4; sc++) {
            unpack_16_values(blk->packed_data + sc * 7, codes_64 + sc * 16);
        }

        __m512 zp_vec = _mm512_set1_ps(zp);
        int i = 0;
        for (; i <= group_len - 16; i += 16) {
            __m512 q_vec = _mm512_loadu_ps(query + offset + i);
            float w[16];
            for (int j = 0; j < 16; j++) {
                float s = (j & 1) ? (scale * (1.0f / 7.0f)) : (scale * (1.0f / 15.0f));
                w[j] = (float)codes_64[i + j] * s;
            }
            __m512 w_vec = _mm512_add_ps(_mm512_loadu_ps(w), zp_vec);
            total_acc = _mm512_fmadd_ps(q_vec, w_vec, total_acc);
        }
        for (; i < group_len; i++) {
            float s = (i & 1) ? (scale * (1.0f / 7.0f)) : (scale * (1.0f / 15.0f));
            float k_val = (float)codes_64[i] * s + zp;
            total_acc = _mm512_add_ps(total_acc, _mm512_set_ps(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, query[offset + i] * k_val));
        }
    }
    return _mm512_reduce_add_ps(total_acc);
}

void qwn_turboquant_accum_value_avx512(float score, const uint8_t* value_blocks, float* ctx, int dim) {
    int n_blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    __m512 score_vec = _mm512_set1_ps(score);

    for (int b = 0; b < n_blocks; b++) {
        const TurboQuantBlock* blk = (const TurboQuantBlock*)(value_blocks + (size_t)b * TURBOQUANT_BLOCK_BYTES);
        float scale = tq_half_to_float(blk->scale_fp16);
        float zp    = tq_half_to_float(blk->zero_point_fp16);

        int offset = b * TURBOQUANT_GROUP_SIZE;
        int rem = dim - offset;
        int group_len = rem > TURBOQUANT_GROUP_SIZE ? TURBOQUANT_GROUP_SIZE : rem;

        uint8_t codes_64[64];
        for (int sc = 0; sc < 4; sc++) {
            unpack_16_values(blk->packed_data + sc * 7, codes_64 + sc * 16);
        }

        __m512 zp_vec = _mm512_set1_ps(zp);
        int i = 0;
        for (; i <= group_len - 16; i += 16) {
            float w[16];
            for (int j = 0; j < 16; j++) {
                float s = (j & 1) ? (scale * (1.0f / 7.0f)) : (scale * (1.0f / 15.0f));
                w[j] = (float)codes_64[i + j] * s;
            }
            __m512 v_vec = _mm512_add_ps(_mm512_loadu_ps(w), zp_vec);
            __m512 c_vec = _mm512_loadu_ps(ctx + offset + i);
            c_vec = _mm512_fmadd_ps(score_vec, v_vec, c_vec);
            _mm512_storeu_ps(ctx + offset + i, c_vec);
        }
        for (; i < group_len; i++) {
            float s = (i & 1) ? (scale * (1.0f / 7.0f)) : (scale * (1.0f / 15.0f));
            float v_val = (float)codes_64[i] * s + zp;
            ctx[offset + i] += score * v_val;
        }
    }
}
#else
float qwn_turboquant_dot_key_avx512(const float* query, const uint8_t* key_blocks, int dim) {
    return qwn_turboquant_dot_key_avx2(query, key_blocks, dim);
}
void qwn_turboquant_accum_value_avx512(float score, const uint8_t* value_blocks, float* ctx, int dim) {
    qwn_turboquant_accum_value_avx2(score, value_blocks, ctx, dim);
}
#endif

/* -------------------------------------------------------------------------
 * Memory Cache Management
 * ------------------------------------------------------------------------- */
int qwn_turboquant_init(TurboQuantCache* cache, int max_tokens, int n_heads, int head_dim) {
    if (!cache || max_tokens <= 0 || n_heads <= 0 || head_dim <= 0) return -1;
    memset(cache, 0, sizeof(*cache));

    cache->max_tokens = max_tokens;
    cache->n_heads = n_heads;
    cache->head_dim = head_dim;
    cache->n_channels = n_heads * head_dim;

    int blocks_per_token = (cache->n_channels + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    cache->token_stride_k = (size_t)blocks_per_token * TURBOQUANT_BLOCK_BYTES;
    cache->token_stride_v = (size_t)blocks_per_token * TURBOQUANT_BLOCK_BYTES;

    size_t total_k = (size_t)max_tokens * cache->token_stride_k;
    size_t total_v = (size_t)max_tokens * cache->token_stride_v;

    cache->packed_k = (uint8_t*)tq_alloc64(total_k);
    cache->packed_v = (uint8_t*)tq_alloc64(total_v);

    if (!cache->packed_k || !cache->packed_v) {
        qwn_turboquant_free(cache);
        return -1;
    }

    memset(cache->packed_k, 0, total_k);
    memset(cache->packed_v, 0, total_v);
    cache->total_bytes = total_k + total_v;
    return 0;
}

void qwn_turboquant_free(TurboQuantCache* cache) {
    if (!cache) return;
    if (cache->packed_k) tq_free64(cache->packed_k);
    if (cache->packed_v) tq_free64(cache->packed_v);
    memset(cache, 0, sizeof(*cache));
}

/* -------------------------------------------------------------------------
 * Top-Level Attention Head Kernel
 * ------------------------------------------------------------------------- */
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
) {
    (void)layer;
    const QwnCpuFeatures* cpu = qwn_get_cpu_features();
    int hd = cache->head_dim;
    int blocks_per_head = (hd + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    size_t head_byte_offset = (size_t)kv_head_idx * blocks_per_head * TURBOQUANT_BLOCK_BYTES;

    /* 1. Compute Q * K^T across past positions t = 0..pos */
    for (int t = 0; t <= pos; t++) {
        const uint8_t* k_ptr = cache->packed_k + (size_t)t * cache->token_stride_k + head_byte_offset;
        float dot;
        if (cpu->has_avx512f) {
            dot = qwn_turboquant_dot_key_avx512(query, k_ptr, hd);
        } else if (cpu->has_vnni) {
            dot = qwn_turboquant_dot_key_vnni(query, k_ptr, hd);
        } else if (cpu->has_avx2) {
            dot = qwn_turboquant_dot_key_avx2(query, k_ptr, hd);
        } else {
            dot = qwn_turboquant_dot_key_scalar(query, k_ptr, hd);
        }
        scores_scratch[t] = dot * scale;
    }

    /* 2. In-place Softmax over scores */
    float max_s = scores_scratch[0];
    for (int t = 1; t <= pos; t++) {
        if (scores_scratch[t] > max_s) max_s = scores_scratch[t];
    }
    float exp_sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        scores_scratch[t] = expf(scores_scratch[t] - max_s);
        exp_sum += scores_scratch[t];
    }
    float inv_exp = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
    for (int t = 0; t <= pos; t++) {
        scores_scratch[t] *= inv_exp;
    }

    /* 3. Accumulate Softmax * V */
    memset(ctx_out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        float sc = scores_scratch[t];
        const uint8_t* v_ptr = cache->packed_v + (size_t)t * cache->token_stride_v + head_byte_offset;
        if (cpu->has_avx512f) {
            qwn_turboquant_accum_value_avx512(sc, v_ptr, ctx_out, hd);
        } else if (cpu->has_avx2) {
            qwn_turboquant_accum_value_avx2(sc, v_ptr, ctx_out, hd);
        } else {
            qwn_turboquant_accum_value_scalar(sc, v_ptr, ctx_out, hd);
        }
    }
}

/* -------------------------------------------------------------------------
 * AVX-512 Top-Level Multi-Head Matrix Multiplication
 * ------------------------------------------------------------------------- */
void qwn_turboquant_matmul_avx512(
    const TurboQuantCache* cache,
    const float* query,
    float* output,
    int n_heads,
    int head_dim
) {
    #pragma omp parallel for schedule(static) if(n_heads > 1)
    for (int h = 0; h < n_heads; h++) {
        const float* q_h = query + h * head_dim;
        float* out_h = output + h * head_dim;
        int blocks_per_head = (head_dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
        size_t head_byte_offset = (size_t)h * blocks_per_head * TURBOQUANT_BLOCK_BYTES;
        const uint8_t* k_ptr = cache->packed_k + head_byte_offset;

        float dot = qwn_turboquant_dot_key_avx512(q_h, k_ptr, head_dim);
        out_h[0] = dot;
    }
}
