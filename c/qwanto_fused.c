#include "qwanto_fused.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

int qwn_fused_attention_forward(
    const float *q_head,
    const TurboQuantBlock *k_cache,
    const TurboQuantBlock *v_cache,
    int seq_len,
    int head_dim,
    float scale,
    float *out_head_context
) {
    if (!q_head || !k_cache || !v_cache || !out_head_context || seq_len <= 0 || head_dim <= 0) return -1;

    int blocks_per_head = (head_dim + 63) / 64;
    float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
    if (!scores) return -2;

    float max_score = -1e9f;

    /* Step 1: Direct in-register Q * K^T dot product via TurboQuant SIMD kernels */
    for (int t = 0; t < seq_len; t++) {
        const uint8_t *k_tok = (const uint8_t *)(k_cache + t * blocks_per_head);
        float score = qwn_turboquant_dot_key_avx2(q_head, k_tok, head_dim) * scale;
        scores[t] = score;
        if (score > max_score) max_score = score;
    }

    /* Step 2: Softmax normalization */
    float sum_exp = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }
    float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
    for (int t = 0; t < seq_len; t++) {
        scores[t] *= inv_sum;
    }

    /* Step 3: Direct in-register Softmax * V accumulation via TurboQuant SIMD kernels */
    memset(out_head_context, 0, (size_t)head_dim * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        float weight = scores[t];
        if (weight < 1e-6f) continue; /* Dynamic sparsity skip */

        const uint8_t *v_tok = (const uint8_t *)(v_cache + t * blocks_per_head);
        qwn_turboquant_accum_value_avx2(weight, v_tok, out_head_context, head_dim);
    }

    free(scores);
    return 0;
}
