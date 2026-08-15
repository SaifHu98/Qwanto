#ifndef QWANTO_TWLA_H
#define QWANTO_TWLA_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * TWLA: 1.58-Bit Ternary Weight & 4-Bit Activation Quantization
 * 
 * Each block quantizes 256 weights into ternary states {-1, 0, +1}:
 * - 64 bytes of 2-bit packed ternary values (4 weights per byte: 00=0, 01=+1, 10=-1)
 * - 2 bytes of FP16 block scaling factor
 * Total block size: 66 bytes for 256 elements (~2.0625 bpw total / 1.58 bpw payload)
 * ------------------------------------------------------------------------- */

#define QWN_TWLA_BLOCK_SIZE 256
#define QWN_TWLA_PAYLOAD_BYTES 64
#define QWN_TWLA_BLOCK_BYTES 66

typedef struct {
    uint8_t packed_weights[QWN_TWLA_PAYLOAD_BYTES]; /* 256 weights packed at 2 bits each */
    uint16_t scale_fp16;                            /* FP16 block scale */
} QwnBlockTWLA;

/* -------------------------------------------------------------------------
 * Core TWLA APIs
 * ------------------------------------------------------------------------- */

/* Quantize an uncompressed float array into TWLA blocks */
void qwn_twla_quantize(const float *src, QwnBlockTWLA *dst, size_t n_elements);

/* Dequantize TWLA blocks back to float array */
void qwn_twla_dequantize(const QwnBlockTWLA *src, float *dst, size_t n_elements);

/* Scalar reference matrix-vector multiplication */
void qwn_twla_vec_dot_scalar(const QwnBlockTWLA *w, const float *x, float *y, size_t n_blocks);

/* Vectorized AVX2 / AVX-VNNI matrix-vector multiplication */
void qwn_twla_vec_dot_avx2(const QwnBlockTWLA *w, const float *x, float *y, size_t n_blocks);

/* Vectorized AVX-512 matrix-vector multiplication */
void qwn_twla_vec_dot_avx512(const QwnBlockTWLA *w, const float *x, float *y, size_t n_blocks);

/* Dynamic CPUID dispatch for TWLA GEMV */
void qwn_twla_gemv(const QwnBlockTWLA *w, const float *x, float *y, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_TWLA_H */
