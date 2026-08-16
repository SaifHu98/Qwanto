#include "qwanto_littlebit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

bool qwn_littlebit_init(
    QwnLittleBitMatrix *mat,
    int rows,
    int cols,
    int rank
) {
    if (!mat || rows <= 0 || cols <= 0 || rank <= 0) return false;
    memset(mat, 0, sizeof(*mat));

    mat->rows = rows;
    mat->cols = cols;
    mat->rank = rank;

    size_t u_words = (size_t)rows * ((rank + 63) / 64);
    size_t v_words = (size_t)cols * ((rank + 63) / 64);

    mat->binary_u = (uint64_t *)malloc(u_words * sizeof(uint64_t));
    mat->binary_v = (uint64_t *)malloc(v_words * sizeof(uint64_t));
    mat->rank_scales = (float *)malloc((size_t)rank * sizeof(float));

    if (!mat->binary_u || !mat->binary_v || !mat->rank_scales) {
        qwn_littlebit_free(mat);
        return false;
    }

    memset(mat->binary_u, 0, u_words * sizeof(uint64_t));
    memset(mat->binary_v, 0, v_words * sizeof(uint64_t));
    for (int k = 0; k < rank; k++) mat->rank_scales[k] = 1.0f;

    mat->total_bytes = (u_words + v_words) * sizeof(uint64_t) + (size_t)rank * sizeof(float);
    mat->bits_per_weight = (double)((size_t)rows * rank + (size_t)cols * rank) / (double)((size_t)rows * cols);
    mat->is_initialized = true;

    return true;
}

bool qwn_littlebit_encode(
    QwnLittleBitMatrix *mat,
    const float *dense_weights,
    int rows,
    int cols,
    int rank
) {
    if (!mat || !dense_weights || rows != mat->rows || cols != mat->cols || rank != mat->rank) {
        return false;
    }

    int rank_words = (rank + 63) / 64;

    /* Initialize low-rank binarized factor projections */
    for (int k = 0; k < rank; k++) {
        float scale_accum = 0.0f;
        for (int r = 0; r < rows; r++) {
            float row_val = dense_weights[(size_t)r * cols + (k % cols)];
            if (row_val >= 0.0f) {
                mat->binary_u[r * rank_words + (k / 64)] |= (1ULL << (k % 64));
            }
            scale_accum += fabsf(row_val);
        }
        for (int c = 0; c < cols; c++) {
            float col_val = dense_weights[(size_t)(k % rows) * cols + c];
            if (col_val >= 0.0f) {
                mat->binary_v[c * rank_words + (k / 64)] |= (1ULL << (k % 64));
            }
            scale_accum += fabsf(col_val);
        }
        mat->rank_scales[k] = scale_accum / (float)(rows + cols);
    }

    return true;
}

bool qwn_littlebit_gemv(
    const QwnLittleBitMatrix *mat,
    const float *x_vector,
    float *y_out
) {
    if (!mat || !mat->is_initialized || !x_vector || !y_out) return false;

    int rows = mat->rows;
    int cols = mat->cols;
    int rank = mat->rank;
    int rank_words = (rank + 63) / 64;

    memset(y_out, 0, (size_t)rows * sizeof(float));

    /* Compute rank projection intermediates: dot_v[k] = sum_c sign(v_k[c]) * x[c] */
    float *dot_v = (float *)malloc((size_t)rank * sizeof(float));
    if (!dot_v) return false;
    memset(dot_v, 0, (size_t)rank * sizeof(float));

    for (int k = 0; k < rank; k++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            bool bit = (mat->binary_v[c * rank_words + (k / 64)] >> (k % 64)) & 1ULL;
            sum += (bit ? 1.0f : -1.0f) * x_vector[c];
        }
        dot_v[k] = sum * mat->rank_scales[k];
    }

    /* Accumulate into output: y[r] = sum_k sign(u_k[r]) * dot_v[k] */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int k = 0; k < rank; k++) {
            bool bit = (mat->binary_u[r * rank_words + (k / 64)] >> (k % 64)) & 1ULL;
            sum += (bit ? 1.0f : -1.0f) * dot_v[k];
        }
        y_out[r] = sum;
    }

    free(dot_v);
    return true;
}

void qwn_littlebit_free(QwnLittleBitMatrix *mat) {
    if (!mat) return;
    if (mat->binary_u) free(mat->binary_u);
    if (mat->binary_v) free(mat->binary_v);
    if (mat->rank_scales) free(mat->rank_scales);
    memset(mat, 0, sizeof(*mat));
}
