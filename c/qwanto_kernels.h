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
    size_t  bytes;
    int     max_tokens;
    int     padded_k;
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

/* Decode one tensor row to float32 (embedding lookup / tied LM head). */
int qwn_row_f32(const QwnModel *m, const QwnTensorDesc *tensor,
                int row, float *out, int width);

#ifdef __cplusplus
}
#endif

#endif
