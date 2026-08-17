#ifndef QWANTO_KERNELS_H
#define QWANTO_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include "qwanto_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void   *allocation;
    int8_t *q8;              /* 64-byte aligned [max_tokens, padded_k] */
    float  *token_scales;    /* 64-byte aligned [max_tokens] */
    float  *row_f32;         /* 64-byte aligned [padded_k] */
    int32_t *activation_sums; /* [max_tokens, ceil(padded_k / 32)] */
    int     activation_sum_blocks;
    int     activation_sum_enabled;
    size_t  bytes;
    int     max_tokens;
    int     padded_k;
    /* Observability for the exact HyperVSQ-2 hot path. These counters are
     * updated outside the inner GEMV loop and never affect its arithmetic. */
    uint64_t hypervsq2_matmul_calls;
    uint64_t hypervsq2_worker_participations;
    int      hypervsq2_last_active_threads;
    int      hypervsq2_max_active_threads;
    char     hypervsq2_kernel[32];
    char     hypervsq2_dispatch_reason[128];
    uint64_t activation_sum_precompute_calls;
    uint64_t activation_sum_reuse_count;
    uint64_t activation_sum_recompute_count;
} QwnScratch;

/* Allocate once per session. No malloc/free occurs in the token hot path. */
int  qwn_scratch_init(QwnScratch *s, int max_tokens, int max_k);
void qwn_scratch_destroy(QwnScratch *s);

/* y[M,N] = x[M,K] @ W[N,K]^T.
 * W uses Q4_0 blocks (f16 scale + 32 int4 values).
 * x is quantized per token to Q8 into the persistent scratch arena.
 * Logical K may have a tail; the stored row is padded to ceil(K/32)*32 and
 * the tail is zeroed. Tensor payload start must be >=64-byte aligned. */
int qwn_matmul_q4_0_f32(const QwnModel *m,
                        const QwnTensorDesc *weights,
                        const float *x, int M, int K, int N,
                        QwnScratch *scratch,
                        float *y);

/* Generic matrix dispatch for Q4_0/F32/F16/BF16 weights. */
int qwn_matmul_f32(const QwnModel *m, const QwnTensorDesc *weights,
                   const float *x, int M, int K, int N,
                   QwnScratch *scratch, float *y);

/* HyperVSQ-2 GEMV kernels */
typedef struct {
    int has_avx2;
    int has_f16c;
    int has_fma;
    int has_vnni;       /* AVX-VNNI or AVX512-VNNI */
    int has_avx512f;
    int forced_mode;    /* 0=auto, 1=scalar, 2=avx2, 3=vnni */
} QwnCpuFeatures;

const QwnCpuFeatures *qwn_get_cpu_features(void);
int qwn_cpu_avx2_kernel_compiled(void);
int qwn_cpu_vnni_kernel_compiled(void);
const char *qwn_cpu_kernel_name(void);
int qwn_select_cpu_kernel(const char *kernel, char *error, size_t error_size);

/* Scalar Golden Reference for HyperVSQ-2 GEMV */
void qwn_gemv_hypervsq2_scalar(const uint8_t *raw_blocks, const int8_t *q8,
                              const int32_t *activation_sums,
                              float x_scale, int K, int N,
                              size_t row_bytes, float *out);

/* AVX2 Accelerated HyperVSQ-2 GEMV */
void qwn_gemv_hypervsq2_avx2(const uint8_t *raw_blocks, const int8_t *q8,
                            const int32_t *activation_sums,
                            float x_scale, int K, int N,
                            size_t row_bytes, float *out);

/* AVX-VNNI Accelerated HyperVSQ-2 GEMV */
void qwn_gemv_hypervsq2_vnni(const uint8_t *raw_blocks, const int8_t *q8,
                            const int32_t *activation_sums,
                            float x_scale, int K, int N,
                            size_t row_bytes, float *out);

/* Full matrix multiplication for HyperVSQ-2 */
int qwn_matmul_hypervsq2_f32(const QwnModel *m,
                             const QwnTensorDesc *weights,
                             const float *x, int M, int K, int N,
                             QwnScratch *scratch,
                             float *y);

/* Decode one tensor row to float32 (embedding lookup / tied LM head). */
int qwn_row_f32(const QwnModel *m, const QwnTensorDesc *tensor,
                int row, float *out, int width);

#ifdef __cplusplus
}
#endif

#endif
