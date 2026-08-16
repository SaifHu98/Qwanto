#ifndef QWANTO_PQUANT_H
#define QWANTO_PQUANT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * pQuant: Decoupled 1-Bit + High-Precision Sparse Quantization
 * Decouples dense weights into a dominant 1-bit binary branch (XNOR/POPCNT)
 * and a compact sparse outlier branch for sensitive parameters.
 * ------------------------------------------------------------------------- */

typedef struct {
    int rows;
    int cols;
    int rank;
    
    /* 1-Bit Dominant Binary Branch (Packed bits: rows * (cols / 64) uint64_t) */
    uint64_t *binary_weights;
    float *row_scales;
    
    /* Sparse High-Precision Outlier Branch */
    int outlier_count;
    uint32_t *outlier_indices; /* Encoded (row << 16) | col */
    float *outlier_values;
    
    size_t total_bytes;
    bool is_initialized;
} QwnPQuantMatrix;

/* Initialize pQuant Decoupled Matrix */
bool qwn_pquant_init(
    QwnPQuantMatrix *mat,
    int rows,
    int cols,
    float outlier_ratio
);

/* Encode dense FP32 weights into pQuant representation */
bool qwn_pquant_encode(
    QwnPQuantMatrix *mat,
    const float *dense_weights,
    int rows,
    int cols,
    float outlier_threshold
);

/* Fast In-Register Matrix-Vector Multiplication using XNOR + PopCount */
bool qwn_pquant_gemv(
    const QwnPQuantMatrix *mat,
    const float *x_vector,
    float *y_out
);

/* Free pQuant Matrix Resources */
void qwn_pquant_free(QwnPQuantMatrix *mat);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_PQUANT_H */
