#include "qwanto_attention.h"
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

int qwanto_linear_attention_init(QwantoLinearAttentionState* state, int n_heads, int head_dim) {
    if (!state || n_heads <= 0 || head_dim <= 0) return -1;
    state->n_heads = n_heads;
    state->head_dim = head_dim;
    state->state = (float*)malloc(n_heads * head_dim * head_dim * sizeof(float));
    if (!state->state) return -1;
    qwanto_linear_attention_reset(state);
    return 0;
}

void qwanto_linear_attention_reset(QwantoLinearAttentionState* state) {
    if (state && state->state) {
        memset(state->state, 0, state->n_heads * state->head_dim * state->head_dim * sizeof(float));
    }
}

void qwanto_linear_attention_destroy(QwantoLinearAttentionState* state) {
    if (state && state->state) {
        free(state->state);
        state->state = NULL;
    }
}

void qwanto_linear_attention_decode(QwantoLinearAttentionState* state, 
                                    const float* Q, const float* K, const float* V, 
                                    float decay, float* O) {
    int n_heads = state->n_heads;
    int d = state->head_dim;
    
    int use_avx512 = 0;
#if defined(__AVX512F__)
    use_avx512 = 1;
#endif

    int use_avx2 = 0;
#if defined(__AVX2__)
    use_avx2 = 1;
#endif

    for (int h = 0; h < n_heads; h++) {
        float* s_ptr = &state->state[h * d * d];
        const float* q_ptr = &Q[h * d];
        const float* k_ptr = &K[h * d];
        const float* v_ptr = &V[h * d];
        float* o_ptr = &O[h * d];

        // 1. State update: S_t = decay * S_{t-1} + K^T * V
        if (use_avx512 && d % 16 == 0) {
#if defined(__AVX512F__)
            __m512 decay_vec = _mm512_set1_ps(decay);
            for (int i = 0; i < d; i++) {
                float k_val = k_ptr[i];
                __m512 k_vec = _mm512_set1_ps(k_val);
                float* row = &s_ptr[i * d];
                for (int j = 0; j < d; j += 16) {
                    __m512 s_vec = _mm512_loadu_ps(&row[j]);
                    __m512 v_vec = _mm512_loadu_ps(&v_ptr[j]);
                    __m512 next_s = _mm512_fmadd_ps(decay_vec, s_vec, _mm512_mul_ps(k_vec, v_vec));
                    _mm512_storeu_ps(&row[j], next_s);
                }
            }
#endif
        } else if (use_avx2 && d % 8 == 0) {
#if defined(__AVX2__)
            __m256 decay_vec = _mm256_set1_ps(decay);
            for (int i = 0; i < d; i++) {
                float k_val = k_ptr[i];
                __m256 k_vec = _mm256_set1_ps(k_val);
                float* row = &s_ptr[i * d];
                for (int j = 0; j < d; j += 8) {
                    __m256 s_vec = _mm256_loadu_ps(&row[j]);
                    __m256 v_vec = _mm256_loadu_ps(&v_ptr[j]);
                    __m256 next_s = _mm256_fmadd_ps(decay_vec, s_vec, _mm256_mul_ps(k_vec, v_vec));
                    _mm256_storeu_ps(&row[j], next_s);
                }
            }
#endif
        } else {
            // Scalar fallback
            for (int i = 0; i < d; i++) {
                float k_val = k_ptr[i];
                float* row = &s_ptr[i * d];
                for (int j = 0; j < d; j++) {
                    row[j] = decay * row[j] + k_val * v_ptr[j];
                }
            }
        }

        // 2. Output projection: O = Q * S_t
        if (use_avx512 && d % 16 == 0) {
#if defined(__AVX512F__)
            memset(o_ptr, 0, d * sizeof(float));
            for (int i = 0; i < d; i++) {
                float q_val = q_ptr[i];
                __m512 q_vec = _mm512_set1_ps(q_val);
                float* row = &s_ptr[i * d];
                for (int j = 0; j < d; j += 16) {
                    __m512 s_vec = _mm512_loadu_ps(&row[j]);
                    __m512 o_vec = _mm512_loadu_ps(&o_ptr[j]);
                    o_vec = _mm512_fmadd_ps(q_vec, s_vec, o_vec);
                    _mm512_storeu_ps(&o_ptr[j], o_vec);
                }
            }
#endif
        } else if (use_avx2 && d % 8 == 0) {
#if defined(__AVX2__)
            memset(o_ptr, 0, d * sizeof(float));
            for (int i = 0; i < d; i++) {
                float q_val = q_ptr[i];
                __m256 q_vec = _mm256_set1_ps(q_val);
                float* row = &s_ptr[i * d];
                for (int j = 0; j < d; j += 8) {
                    __m256 s_vec = _mm256_loadu_ps(&row[j]);
                    __m256 o_vec = _mm256_loadu_ps(&o_ptr[j]);
                    o_vec = _mm256_fmadd_ps(q_vec, s_vec, o_vec);
                    _mm256_storeu_ps(&o_ptr[j], o_vec);
                }
            }
#endif
        } else {
            // Scalar fallback
            memset(o_ptr, 0, d * sizeof(float));
            for (int i = 0; i < d; i++) {
                float q_val = q_ptr[i];
                float* row = &s_ptr[i * d];
                for (int j = 0; j < d; j++) {
                    o_ptr[j] += q_val * row[j];
                }
            }
        }
    }
}

void qwanto_linear_attention_prefill(QwantoLinearAttentionState* state, 
                                     const float* Q, const float* K, const float* V, 
                                     float decay, int seq_len, float* O) {
    int total_dim = state->n_heads * state->head_dim;
    for (int t = 0; t < seq_len; t++) {
        const float* q_t = &Q[t * total_dim];
        const float* k_t = &K[t * total_dim];
        const float* v_t = &V[t * total_dim];
        float* o_t = &O[t * total_dim];
        
        qwanto_linear_attention_decode(state, q_t, k_t, v_t, decay, o_t);
    }
}
