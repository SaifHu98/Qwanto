#ifndef QWANTO_LITTLEBIT_H
#define QWANTO_LITTLEBIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * LittleBit-2: Sub-1-Bit Low-Rank Factorized Binarization (ICML 2026)
 * Compresses weight matrices into sub-1-bit regime (0.45 - 0.75 bpw)
 * by factorizing W into low-rank binarized factors (U, V) and learned scales.
 * ------------------------------------------------------------------------- */

typedef struct {
    int rows;
    int cols;
    int rank;
    
    /* Factor U: [rows, rank] packed binary bits */
    uint64_t *binary_u;
    
    /* Factor V: [cols, rank] packed binary bits */
    uint64_t *binary_v;
    
    /* Learned Rank Scales [rank] */
    float *rank_scales;
    
    size_t total_bytes;
    double bits_per_weight;
    bool is_initialized;
} QwnLittleBitMatrix;

/* Initialize LittleBit-2 Matrix with low-rank target */
bool qwn_littlebit_init(
    QwnLittleBitMatrix *mat,
    int rows,
    int cols,
    int rank
);

/* Encode dense weights into LittleBit-2 factorized representation */
bool qwn_littlebit_encode(
    QwnLittleBitMatrix *mat,
    const float *dense_weights,
    int rows,
    int cols,
    int rank
);

/* High-speed Sub-1-Bit GEMV execution: y = sum_k alpha_k * u_k * (v_k^T * x) */
bool qwn_littlebit_gemv(
    const QwnLittleBitMatrix *mat,
    const float *x_vector,
    float *y_out
);

/* Free LittleBit-2 Matrix Resources */
void qwn_littlebit_free(QwnLittleBitMatrix *mat);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_LITTLEBIT_H */
