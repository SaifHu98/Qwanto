#ifndef QWANTO_ATTENTION_H
#define QWANTO_ATTENTION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_heads;
    int head_dim;
    float* state; // Pointer to flat state buffer of shape [n_heads, head_dim, head_dim]
} QwantoLinearAttentionState;

/**
 * Initialize Linear Attention state structure.
 * 
 * @param state     Pointer to state structure
 * @param n_heads   Number of heads
 * @param head_dim  Dimension per head (must be multiple of 8 for AVX2, 16 for AVX-512)
 * @return 0 on success, -1 on failure
 */
int qwanto_linear_attention_init(QwantoLinearAttentionState* state, int n_heads, int head_dim);

/**
 * Reset the state matrix to zeros.
 * 
 * @param state     Pointer to initialized state structure
 */
void qwanto_linear_attention_reset(QwantoLinearAttentionState* state);

/**
 * Destroy state and release allocated memory.
 * 
 * @param state     Pointer to initialized state structure
 */
void qwanto_linear_attention_destroy(QwantoLinearAttentionState* state);

/**
 * Execute a single-token decode step of Linear Attention.
 * Updates state in-place: S_t = decay * S_{t-1} + K^T * V
 * Computes output: O = Q * S_t
 * 
 * @param state     Pointer to initialized state structure
 * @param Q         Query tensor of shape [n_heads, head_dim]
 * @param K         Key tensor of shape [n_heads, head_dim]
 * @param V         Value tensor of shape [n_heads, head_dim]
 * @param decay     Exponential decay factor (gamma)
 * @param O         Output tensor of shape [n_heads, head_dim]
 */
void qwanto_linear_attention_decode(QwantoLinearAttentionState* state, 
                                    const float* Q, const float* K, const float* V, 
                                    float decay, float* O);

/**
 * Execute a sequence prefill step of Linear Attention.
 * Sequentially updates the state for seq_len steps and writes sequence outputs.
 * 
 * @param state     Pointer to initialized state structure
 * @param Q         Query tensor of shape [seq_len, n_heads, head_dim]
 * @param K         Key tensor of shape [seq_len, n_heads, head_dim]
 * @param V         Value tensor of shape [seq_len, n_heads, head_dim]
 * @param decay     Exponential decay factor (gamma)
 * @param seq_len   Sequence length
 * @param O         Output tensor of shape [seq_len, n_heads, head_dim]
 */
void qwanto_linear_attention_prefill(QwantoLinearAttentionState* state, 
                                     const float* Q, const float* K, const float* V, 
                                     float decay, int seq_len, float* O);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_ATTENTION_H */
