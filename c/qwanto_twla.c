#include "qwanto_twla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* -------------------------------------------------------------------------
 * FP16 Conversion Utilities
 * ------------------------------------------------------------------------- */
static inline uint16_t qwn_float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float qwn_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;

    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000 | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

/* -------------------------------------------------------------------------
 * Quantization: Float -> TWLA (1.58-bit Ternary)
 * ------------------------------------------------------------------------- */
void qwn_twla_quantize(const float *src, QwnBlockTWLA *dst, size_t n_elements) {
    size_t n_blocks = (n_elements + QWN_TWLA_BLOCK_SIZE - 1) / QWN_TWLA_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const float *block_src = src + b * QWN_TWLA_BLOCK_SIZE;
        size_t elts = (b == n_blocks - 1) ? (n_elements - b * QWN_TWLA_BLOCK_SIZE) : QWN_TWLA_BLOCK_SIZE;

        /* Calculate absolute mean threshold */
        float abs_sum = 0.0f;
        for (size_t i = 0; i < elts; i++) {
            abs_sum += fabsf(block_src[i]);
        }
        float scale = abs_sum / (float)(elts > 0 ? elts : 1);
        if (scale < 1e-8f) scale = 1e-8f;

        float inv_scale = 1.0f / scale;
        dst[b].scale_fp16 = qwn_float_to_half(scale);
        memset(dst[b].packed_weights, 0, QWN_TWLA_PAYLOAD_BYTES);

        for (size_t i = 0; i < elts; i++) {
            float val = block_src[i] * inv_scale;
            int code = 0; /* 00 = 0 */
            if (val > 0.5f) {
                code = 1; /* 01 = +1 */
            } else if (val < -0.5f) {
                code = 2; /* 10 = -1 */
            }

            size_t byte_idx = i / 4;
            size_t shift = (i % 4) * 2;
            dst[b].packed_weights[byte_idx] |= (uint8_t)(code << shift);
        }
    }
}

/* -------------------------------------------------------------------------
 * Dequantization: TWLA -> Float
 * ------------------------------------------------------------------------- */
void qwn_twla_dequantize(const QwnBlockTWLA *src, float *dst, size_t n_elements) {
    size_t n_blocks = (n_elements + QWN_TWLA_BLOCK_SIZE - 1) / QWN_TWLA_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        float scale = qwn_half_to_float(src[b].scale_fp16);
        float *block_dst = dst + b * QWN_TWLA_BLOCK_SIZE;
        size_t elts = (b == n_blocks - 1) ? (n_elements - b * QWN_TWLA_BLOCK_SIZE) : QWN_TWLA_BLOCK_SIZE;

        for (size_t i = 0; i < elts; i++) {
            size_t byte_idx = i / 4;
            size_t shift = (i % 4) * 2;
            int code = (src[b].packed_weights[byte_idx] >> shift) & 0x03;

            float w = 0.0f;
            if (code == 1) w = scale;
            else if (code == 2) w = -scale;

            block_dst[i] = w;
        }
    }
}

/* -------------------------------------------------------------------------
 * Scalar Vector Dot Product
 * ------------------------------------------------------------------------- */
void qwn_twla_vec_dot_scalar(const QwnBlockTWLA *w, const float *x, float *y, size_t n_blocks) {
    float sum = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        float scale = qwn_half_to_float(w[b].scale_fp16);
        const float *block_x = x + b * QWN_TWLA_BLOCK_SIZE;
        float block_acc = 0.0f;

        for (size_t i = 0; i < QWN_TWLA_BLOCK_SIZE; i++) {
            size_t byte_idx = i / 4;
            size_t shift = (i % 4) * 2;
            int code = (w[b].packed_weights[byte_idx] >> shift) & 0x03;

            if (code == 1) {
                block_acc += block_x[i];
            } else if (code == 2) {
                block_acc -= block_x[i];
            }
        }
        sum += block_acc * scale;
    }
    *y = sum;
}

/* -------------------------------------------------------------------------
 * AVX2 Vector Dot Product
 * ------------------------------------------------------------------------- */
void qwn_twla_vec_dot_avx2(const QwnBlockTWLA *w, const float *x, float *y, size_t n_blocks) {
#if defined(__AVX2__)
    float total_sum = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        float scale = qwn_half_to_float(w[b].scale_fp16);
        const float *bx = x + b * QWN_TWLA_BLOCK_SIZE;
        __m256 acc_vec = _mm256_setzero_ps();

        for (size_t i = 0; i < QWN_TWLA_PAYLOAD_BYTES; i += 4) {
            /* Process 16 elements (4 bytes of packed ternary) */
            uint32_t packed;
            memcpy(&packed, &w[b].packed_weights[i], 4);

            float vals[16];
            for (int k = 0; k < 16; k++) {
                int code = (packed >> (k * 2)) & 0x03;
                vals[k] = (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
            }

            __m256 v0 = _mm256_loadu_ps(vals);
            __m256 x0 = _mm256_loadu_ps(bx + i * 4);
            acc_vec = _mm256_fmadd_ps(v0, x0, acc_vec);

            __m256 v1 = _mm256_loadu_ps(vals + 8);
            __m256 x1 = _mm256_loadu_ps(bx + i * 4 + 8);
            acc_vec = _mm256_fmadd_ps(v1, x1, acc_vec);
        }

        /* Horizontal sum of acc_vec */
        __m128 lo = _mm256_castps256_ps128(acc_vec);
        __m128 hi = _mm256_extractf128_ps(acc_vec, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        total_sum += _mm_cvtss_f32(s) * scale;
    }
    *y = total_sum;
#else
    qwn_twla_vec_dot_scalar(w, x, y, n_blocks);
#endif
}

/* -------------------------------------------------------------------------
 * AVX-512 Vector Dot Product
 * ------------------------------------------------------------------------- */
void qwn_twla_vec_dot_avx512(const QwnBlockTWLA *w, const float *x, float *y, size_t n_blocks) {
#if defined(__AVX512F__)
    float total_sum = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        float scale = qwn_half_to_float(w[b].scale_fp16);
        const float *bx = x + b * QWN_TWLA_BLOCK_SIZE;
        __m512 acc_vec = _mm512_setzero_ps();

        for (size_t i = 0; i < QWN_TWLA_PAYLOAD_BYTES; i += 4) {
            uint32_t packed;
            memcpy(&packed, &w[b].packed_weights[i], 4);

            float vals[16];
            for (int k = 0; k < 16; k++) {
                int code = (packed >> (k * 2)) & 0x03;
                vals[k] = (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
            }

            __m512 v0 = _mm512_loadu_ps(vals);
            __m512 x0 = _mm512_loadu_ps(bx + i * 4);
            acc_vec = _mm512_fmadd_ps(v0, x0, acc_vec);
        }

        total_sum += _mm512_reduce_add_ps(acc_vec) * scale;
    }
    *y = total_sum;
#else
    qwn_twla_vec_dot_avx2(w, x, y, n_blocks);
#endif
}

/* -------------------------------------------------------------------------
 * Full GEMV Matrix-Vector Multiplication with OpenMP
 * ------------------------------------------------------------------------- */
void qwn_twla_gemv(const QwnBlockTWLA *w, const float *x, float *y, size_t rows, size_t cols) {
    size_t blocks_per_row = (cols + QWN_TWLA_BLOCK_SIZE - 1) / QWN_TWLA_BLOCK_SIZE;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < (int64_t)rows; r++) {
        const QwnBlockTWLA *row_w = w + r * blocks_per_row;
        qwn_twla_vec_dot_avx2(row_w, x, &y[r], blocks_per_row);
    }
}
