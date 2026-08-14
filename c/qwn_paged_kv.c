#include "qwn_paged_kv.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, mant = h & 1023, bits;
    if (exp == 0) bits = sign << 31;
    else if (exp == 31) bits = (sign << 31) | 0x7f800000u | (mant << 13);
    else bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

static uint16_t float_to_half(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t sign = b >> 31, exp = (b >> 23) & 255, mant = b & 0x7fffff;
    if (exp == 255) return (uint16_t)((sign << 15) | 0x7c00 | !!mant);
    int e = (int)exp - 127 + 15;
    if (e <= 0) return (uint16_t)(sign << 15);
    if (e >= 31) return (uint16_t)((sign << 15) | 0x7c00);
    return (uint16_t)((sign << 15) | ((uint32_t)e << 10) | (mant >> 13));
}

int qwn_kv_pool_init(QwnKVBlockPool *pool, int total_blocks, int layers, int kv_heads, int head_dim) {
    if (!pool || total_blocks < 1 || layers < 1 || kv_heads < 1 || head_dim < 1) return -1;
    memset(pool, 0, sizeof(*pool));
    pool->total_blocks = total_blocks;
    pool->layers = layers;
    pool->kv_heads = kv_heads;
    pool->head_dim = head_dim;

    size_t elems_per_block = (size_t)layers * kv_heads * QWN_PAGE_BLOCK_SIZE * head_dim;
    size_t total_elems = (size_t)total_blocks * elems_per_block;
    pool->block_bytes = elems_per_block * sizeof(uint16_t);

    pool->key_data = (uint16_t *)calloc(total_elems, sizeof(uint16_t));
    pool->val_data = (uint16_t *)calloc(total_elems, sizeof(uint16_t));
    pool->free_stack = (int *)malloc((size_t)total_blocks * sizeof(int));

    if (!pool->key_data || !pool->val_data || !pool->free_stack) {
        qwn_kv_pool_free(pool);
        return -1;
    }

    /* Initialize free stack with all block indices */
    for (int i = 0; i < total_blocks; i++) {
        pool->free_stack[i] = total_blocks - 1 - i;
    }
    pool->free_top = total_blocks;
    return 0;
}

void qwn_kv_pool_free(QwnKVBlockPool *pool) {
    if (!pool) return;
    free(pool->key_data);
    free(pool->val_data);
    free(pool->free_stack);
    memset(pool, 0, sizeof(*pool));
}

int qwn_kv_pool_alloc_block(QwnKVBlockPool *pool) {
    if (!pool || pool->free_top <= 0) return -1;
    return pool->free_stack[--pool->free_top];
}

void qwn_kv_pool_free_block(QwnKVBlockPool *pool, int block_id) {
    if (!pool || block_id < 0 || block_id >= pool->total_blocks || pool->free_top >= pool->total_blocks) return;
    pool->free_stack[pool->free_top++] = block_id;
}

int qwn_block_table_init(QwnBlockTable *bt, int req_id, int initial_capacity) {
    if (!bt) return -1;
    memset(bt, 0, sizeof(*bt));
    bt->req_id = req_id;
    bt->block_capacity = initial_capacity > 4 ? initial_capacity : 4;
    bt->block_ids = (int *)malloc((size_t)bt->block_capacity * sizeof(int));
    if (!bt->block_ids) return -1;
    return 0;
}

int qwn_block_table_append_token(QwnKVBlockPool *pool, QwnBlockTable *bt) {
    if (!pool || !bt) return -1;
    int cur_tokens = bt->num_tokens;
    int required_blocks = (cur_tokens + 1 + QWN_PAGE_BLOCK_SIZE - 1) / QWN_PAGE_BLOCK_SIZE;

    if (required_blocks > bt->block_count) {
        int new_block = qwn_kv_pool_alloc_block(pool);
        if (new_block < 0) return -1; /* Out of physical KV blocks */

        if (bt->block_count >= bt->block_capacity) {
            int new_cap = bt->block_capacity * 2;
            int *new_arr = (int *)realloc(bt->block_ids, (size_t)new_cap * sizeof(int));
            if (!new_arr) {
                qwn_kv_pool_free_block(pool, new_block);
                return -1;
            }
            bt->block_ids = new_arr;
            bt->block_capacity = new_cap;
        }
        bt->block_ids[bt->block_count++] = new_block;
    }
    bt->num_tokens++;
    return 0;
}

void qwn_block_table_free(QwnKVBlockPool *pool, QwnBlockTable *bt) {
    if (!bt) return;
    if (pool && bt->block_ids) {
        for (int i = 0; i < bt->block_count; i++) {
            qwn_kv_pool_free_block(pool, bt->block_ids[i]);
        }
    }
    free(bt->block_ids);
    memset(bt, 0, sizeof(*bt));
}

int qwn_paged_kv_write(QwnKVBlockPool *pool, const QwnBlockTable *bt, int layer,
                       int token_pos, const float *k, const float *v) {
    if (!pool || !bt || token_pos < 0 || token_pos >= bt->num_tokens) return -1;
    int block_idx = token_pos / QWN_PAGE_BLOCK_SIZE;
    int offset_in_block = token_pos % QWN_PAGE_BLOCK_SIZE;

    if (block_idx >= bt->block_count) return -1;
    int phys_block = bt->block_ids[block_idx];

    int HD = pool->head_dim;
    int HK = pool->kv_heads;
    size_t block_stride = (size_t)pool->layers * HK * QWN_PAGE_BLOCK_SIZE * HD;
    size_t layer_offset = (size_t)layer * HK * QWN_PAGE_BLOCK_SIZE * HD;
    size_t base = (size_t)phys_block * block_stride + layer_offset;

    for (int kh = 0; kh < HK; kh++) {
        size_t head_offset = base + (size_t)kh * QWN_PAGE_BLOCK_SIZE * HD + (size_t)offset_in_block * HD;
        const float *k_head = k + kh * HD;
        const float *v_head = v + kh * HD;
        uint16_t *k_dst = pool->key_data + head_offset;
        uint16_t *v_dst = pool->val_data + head_offset;

#if defined(__AVX2__) && defined(__F16C__)
        int j = 0;
        for (; j <= HD - 8; j += 8) {
            __m256 kf = _mm256_loadu_ps(k_head + j);
            __m256 vf = _mm256_loadu_ps(v_head + j);
            __m128i kh16 = _mm256_cvtps_ph(kf, _MM_FROUND_TO_NEAREST_INT);
            __m128i vh16 = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(k_dst + j), kh16);
            _mm_storeu_si128((__m128i *)(v_dst + j), vh16);
        }
        for (; j < HD; j++) {
            k_dst[j] = float_to_half(k_head[j]);
            v_dst[j] = float_to_half(v_head[j]);
        }
#else
        for (int j = 0; j < HD; j++) {
            k_dst[j] = float_to_half(k_head[j]);
            v_dst[j] = float_to_half(v_head[j]);
        }
#endif
    }
    return 0;
}

static void softmax_inplace(float *x, int n) {
    if (n <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += (double)x[i];
    }
    float inv = (float)(1.0 / (sum > 0.0 ? sum : 1.0));
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void qwn_paged_attention_head(const QwnKVBlockPool *pool, const QwnBlockTable *bt,
                              int layer, int head_idx, int kv_head_idx,
                              const float *q_head, float *scores_scratch,
                              float *out_ctx_head) {
    (void)head_idx;
    if (!pool || !bt || bt->num_tokens <= 0) return;
    int num_tokens = bt->num_tokens;
    int HD = pool->head_dim;
    int HK = pool->kv_heads;
    float scale = 1.0f / sqrtf((float)HD);
    size_t block_stride = (size_t)pool->layers * HK * QWN_PAGE_BLOCK_SIZE * HD;
    size_t layer_offset = (size_t)layer * HK * QWN_PAGE_BLOCK_SIZE * HD;

    /* Step 1: Compute Q * K dot products across all paged blocks */
    for (int t = 0; t < num_tokens; t++) {
        int b_idx = t / QWN_PAGE_BLOCK_SIZE;
        int t_offset = t % QWN_PAGE_BLOCK_SIZE;
        int phys_block = bt->block_ids[b_idx];
        size_t base = (size_t)phys_block * block_stride + layer_offset +
                      (size_t)kv_head_idx * QWN_PAGE_BLOCK_SIZE * HD + (size_t)t_offset * HD;
        const uint16_t *kc = pool->key_data + base;
        float dot = 0.0f;

#if defined(__AVX2__) && defined(__F16C__)
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= HD - 8; j += 8) {
            __m128i h8 = _mm_loadu_si128((const __m128i *)(kc + j));
            __m256 kf = _mm256_cvtph_ps(h8);
            __m256 qf = _mm256_loadu_ps(q_head + j);
            acc = _mm256_fmadd_ps(qf, kf, acc);
        }
        float tmp[8]; _mm256_storeu_ps(tmp, acc);
        dot = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; j < HD; j++) dot += q_head[j] * half_to_float(kc[j]);
#else
        for (int j = 0; j < HD; j++) dot += q_head[j] * half_to_float(kc[j]);
#endif
        scores_scratch[t] = dot * scale;
    }

    /* Step 2: Softmax over sequence */
    softmax_inplace(scores_scratch, num_tokens);

    /* Step 3: Compute weighted sum of V across paged blocks */
    memset(out_ctx_head, 0, (size_t)HD * sizeof(float));
    for (int t = 0; t < num_tokens; t++) {
        float sc = scores_scratch[t];
        int b_idx = t / QWN_PAGE_BLOCK_SIZE;
        int t_offset = t % QWN_PAGE_BLOCK_SIZE;
        int phys_block = bt->block_ids[b_idx];
        size_t base = (size_t)phys_block * block_stride + layer_offset +
                      (size_t)kv_head_idx * QWN_PAGE_BLOCK_SIZE * HD + (size_t)t_offset * HD;
        const uint16_t *vc = pool->val_data + base;

#if defined(__AVX2__) && defined(__F16C__)
        __m256 sv = _mm256_set1_ps(sc);
        int j = 0;
        for (; j <= HD - 8; j += 8) {
            __m128i h8 = _mm_loadu_si128((const __m128i *)(vc + j));
            __m256 vf = _mm256_cvtph_ps(h8);
            __m256 ov = _mm256_loadu_ps(out_ctx_head + j);
            ov = _mm256_fmadd_ps(sv, vf, ov);
            _mm256_storeu_ps(out_ctx_head + j, ov);
        }
        for (; j < HD; j++) out_ctx_head[j] += sc * half_to_float(vc[j]);
#else
        for (int j = 0; j < HD; j++) out_ctx_head[j] += sc * half_to_float(vc[j]);
#endif
    }
}
