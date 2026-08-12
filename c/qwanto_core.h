#ifndef QWANTO_CORE_H
#define QWANTO_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform 4-bit packed weights matrix multiplication with 8-bit activations using AVX2.
 * 
 * @param activations   Pointer to int8_t input activations matrix [M, K]
 * @param packed_weights Pointer to uint8_t packed weights [N, K/2] (2 weights/byte, range [-8, 7] after bias)
 * @param outputs       Pointer to int32_t output matrix [M, N]
 * @param m             Output dimension N of weights (rows of weights)
 * @param n             Input dimension K of weights (columns of weights, must be multiple of 32)
 */
void qwanto_matmul_core_avx2(const int8_t* activations, const uint8_t* packed_weights, int32_t* outputs, int m, int n);

/**
 * Perform 4-bit packed weights matrix multiplication with 8-bit activations using AVX-512.
 * 
 * @param activations   Pointer to int8_t input activations matrix [M, K]
 * @param packed_weights Pointer to uint8_t packed weights [N, K/2]
 * @param outputs       Pointer to int32_t output matrix [M, N]
 * @param m             Output dimension N of weights
 * @param n             Input dimension K of weights (must be multiple of 64)
 */
void qwanto_matmul_core_avx512(const int8_t* activations, const uint8_t* packed_weights, int32_t* outputs, int m, int n);

/**
 * Orchestrate cache-blocked matrix multiplication that selects AVX-512 or AVX2 automatically.
 * It partitions the matrix into blocks that fit within L2 cache (1MB target configurations).
 * 
 * @param activations   Pointer to int8_t input activations matrix [M_tokens, K]
 * @param packed_weights Pointer to uint8_t packed weights [N_out, K/2]
 * @param outputs       Pointer to int32_t output matrix [M_tokens, N_out]
 * @param m_tokens      Number of input tokens (rows of activations)
 * @param n_out         Output dimension (rows of weights / cols of outputs)
 * @param k_in          Input dimension (cols of activations / cols of weights)
 */
void qwanto_matmul_blocked(const int8_t* activations, const uint8_t* packed_weights, int32_t* outputs, 
                           int m_tokens, int n_out, int k_in);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_CORE_H */
