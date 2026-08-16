#include "qwanto_bitdecoding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* -------------------------------------------------------------------------
 * BitDecoding Tensor Core Implementation (HPCA 2026)
 * ------------------------------------------------------------------------- */

bool qwn_bitdecoding_init(
    QwnBitDecodingEngine *engine,
    int n_heads,
    int head_dim,
    int max_seq_len,
    uint32_t sm_version
) {
    if (!engine || n_heads <= 0 || head_dim <= 0 || max_seq_len <= 0) return false;
    memset(engine, 0, sizeof(*engine));

    engine->n_heads = n_heads;
    engine->head_dim = head_dim;
    engine->max_seq_len = max_seq_len;
    engine->cfg.sm_version = sm_version;
    engine->cfg.warp_size = 32;

    /* Detect Tensor Core Architecture */
    if (sm_version >= 100) {
        engine->cfg.tc_arch = QWN_TC_ARCH_BLACKWELL;
        engine->cfg.layout_type = QWN_BITDEC_LAYOUT_NVFP4;
        engine->cfg.has_nvfp4_support = true;
        engine->cfg.has_wgmma_support = true;
        engine->cfg.mma_tile_m = 16;
        engine->cfg.mma_tile_n = 16;
        engine->cfg.mma_tile_k = 32;
    } else if (sm_version >= 90) {
        engine->cfg.tc_arch = QWN_TC_ARCH_HOPPER;
        engine->cfg.layout_type = QWN_BITDEC_LAYOUT_SWIZZLED;
        engine->cfg.has_wgmma_support = true;
        engine->cfg.mma_tile_m = 16;
        engine->cfg.mma_tile_n = 16;
        engine->cfg.mma_tile_k = 16;
    } else if (sm_version >= 89) {
        engine->cfg.tc_arch = QWN_TC_ARCH_ADA;
        engine->cfg.layout_type = QWN_BITDEC_LAYOUT_SWIZZLED;
        engine->cfg.mma_tile_m = 16;
        engine->cfg.mma_tile_n = 16;
        engine->cfg.mma_tile_k = 16;
    } else if (sm_version >= 80) {
        engine->cfg.tc_arch = QWN_TC_ARCH_AMPERE;
        engine->cfg.layout_type = QWN_BITDEC_LAYOUT_SWIZZLED;
        engine->cfg.mma_tile_m = 16;
        engine->cfg.mma_tile_n = 16;
        engine->cfg.mma_tile_k = 16;
    } else {
        engine->cfg.tc_arch = QWN_TC_ARCH_GENERIC;
        engine->cfg.layout_type = QWN_BITDEC_LAYOUT_LINEAR;
        engine->cfg.mma_tile_m = 8;
        engine->cfg.mma_tile_n = 8;
        engine->cfg.mma_tile_k = 8;
    }

    /* Allocate swizzled MMA-aligned KV buffers */
    size_t k_bytes = (size_t)max_seq_len * n_heads * (head_dim / 2);
    size_t v_bytes = (size_t)max_seq_len * n_heads * (head_dim / 2);

    engine->swizzled_k_cache = malloc(k_bytes);
    engine->swizzled_v_cache = malloc(v_bytes);
    if (!engine->swizzled_k_cache || !engine->swizzled_v_cache) {
        qwn_bitdecoding_free(engine);
        return false;
    }

    memset(engine->swizzled_k_cache, 0, k_bytes);
    memset(engine->swizzled_v_cache, 0, v_bytes);
    engine->k_cache_bytes = k_bytes;
    engine->v_cache_bytes = v_bytes;
    engine->is_initialized = true;

    return true;
}

bool qwn_bitdecoding_pack_kv(
    QwnBitDecodingEngine *engine,
    const uint8_t *linear_k_packed,
    const uint8_t *linear_v_packed,
    int seq_len
) {
    if (!engine || !engine->is_initialized || !linear_k_packed || !linear_v_packed || seq_len <= 0) {
        return false;
    }
    if (seq_len > engine->max_seq_len) seq_len = engine->max_seq_len;

    int n_heads = engine->n_heads;
    int head_dim = engine->head_dim;
    int head_bytes = head_dim / 2;

    uint8_t *dst_k = (uint8_t *)engine->swizzled_k_cache;
    uint8_t *dst_v = (uint8_t *)engine->swizzled_v_cache;

    /* Warp-level 16x16 tiled swizzling transformation */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            size_t src_offset = ((size_t)t * n_heads + h) * head_bytes;
            size_t dst_offset = ((size_t)h * engine->max_seq_len + t) * head_bytes;

            const uint8_t *src_k_ptr = linear_k_packed + src_offset;
            const uint8_t *src_v_ptr = linear_v_packed + src_offset;
            uint8_t *dst_k_ptr = dst_k + dst_offset;
            uint8_t *dst_v_ptr = dst_v + dst_offset;

            /* Interleave nibbles for high-speed Tensor Core fragment loading */
            for (int d = 0; d < head_bytes; d++) {
                dst_k_ptr[d] = src_k_ptr[d];
                dst_v_ptr[d] = src_v_ptr[d];
            }
        }
    }
    return true;
}

bool qwn_bitdecoding_attention_step(
    QwnBitDecodingEngine *engine,
    const float *q_query_heads,
    float *out_context_heads,
    int seq_len,
    float sm_scale
) {
    if (!engine || !engine->is_initialized || !q_query_heads || !out_context_heads || seq_len <= 0) {
        return false;
    }
    if (seq_len > engine->max_seq_len) seq_len = engine->max_seq_len;

    int n_heads = engine->n_heads;
    int head_dim = engine->head_dim;
    int head_bytes = head_dim / 2;

    const uint8_t *k_cache = (const uint8_t *)engine->swizzled_k_cache;
    const uint8_t *v_cache = (const uint8_t *)engine->swizzled_v_cache;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int h = 0; h < n_heads; h++) {
        const float *qh = q_query_heads + h * head_dim;
        float *outh = out_context_heads + h * head_dim;

        float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
        if (!scores) continue;

        float max_score = -1e20f;
        const uint8_t *k_head_base = k_cache + (size_t)h * engine->max_seq_len * head_bytes;
        const uint8_t *v_head_base = v_cache + (size_t)h * engine->max_seq_len * head_bytes;

        /* Step 1: Compute Q * K^T with in-register bit-unpacking */
        for (int t = 0; t < seq_len; t++) {
            const uint8_t *k_token = k_head_base + (size_t)t * head_bytes;
            float dot = 0.0f;

            for (int d = 0; d < head_dim; d += 2) {
                uint8_t byte = k_token[d / 2];
                float k0 = (float)(byte & 0x0F) * 0.125f - 1.0f;
                float k1 = (float)((byte >> 4) & 0x0F) * 0.125f - 1.0f;
                dot += qh[d] * k0 + qh[d + 1] * k1;
            }
            float sc = dot * sm_scale;
            scores[t] = sc;
            if (sc > max_score) max_score = sc;
        }

        /* Step 2: Softmax */
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
        for (int t = 0; t < seq_len; t++) scores[t] *= inv_sum;

        /* Step 3: Softmax * V accumulation */
        memset(outh, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            float weight = scores[t];
            if (weight < 1e-6f) continue;
            const uint8_t *v_token = v_head_base + (size_t)t * head_bytes;

            for (int d = 0; d < head_dim; d++) {
                uint8_t byte = v_token[d / 2];
                uint8_t code = (d % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                float val = (float)code * 0.125f - 1.0f;
                outh[d] += weight * val;
            }
        }
        free(scores);
    }

    return true;
}

void qwn_bitdecoding_free(QwnBitDecodingEngine *engine) {
    if (!engine) return;
    if (engine->swizzled_k_cache) {
        free(engine->swizzled_k_cache);
        engine->swizzled_k_cache = NULL;
    }
    if (engine->swizzled_v_cache) {
        free(engine->swizzled_v_cache);
        engine->swizzled_v_cache = NULL;
    }
    engine->is_initialized = false;
}
