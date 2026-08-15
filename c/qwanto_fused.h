#ifndef QWANTO_FUSED_H
#define QWANTO_FUSED_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "qwanto_turboquant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Fused Kernel Architecture: Single-Pass In-Register Attention
 * ------------------------------------------------------------------------- */

/* Fused attention forward pass: computes Q * K^T * V directly from TurboQuant blocks */
int qwn_fused_attention_forward(
    const float *q_head,                    /* Query vector for single head [head_dim] */
    const TurboQuantBlock *k_cache,         /* Quantized K cache blocks [seq_len * (head_dim/64)] */
    const TurboQuantBlock *v_cache,         /* Quantized V cache blocks [seq_len * (head_dim/64)] */
    int seq_len,
    int head_dim,
    float scale,
    float *out_head_context                 /* Output context vector [head_dim] */
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_FUSED_H */
