#include "qwanto_pquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

bool qwn_pquant_init(
    QwnPQuantMatrix *mat,
    int rows,
    int cols,
    float outlier_ratio
) {
    if (!mat || rows <= 0 || cols <= 0) return false;
    memset(mat, 0, sizeof(*mat));

    mat->rows = rows;
    mat->cols = cols;

    int cols_u64 = (cols + 63) / 64;
    size_t bin_bytes = (size_t)rows * cols_u64 * sizeof(uint64_t);
    size_t scale_bytes = (size_t)rows * sizeof(float);

    mat->binary_weights = (uint64_t *)malloc(bin_bytes);
    mat->row_scales = (float *)malloc(scale_bytes);

    int max_outliers = (int)(rows * cols * outlier_ratio);
    if (max_outliers < 16) max_outliers = 16;

    mat->outlier_indices = (uint32_t *)malloc((size_t)max_outliers * sizeof(uint32_t));
    mat->outlier_values = (float *)malloc((size_t)max_outliers * sizeof(float));

    if (!mat->binary_weights || !mat->row_scales || !mat->outlier_indices || !mat->outlier_values) {
        qwn_pquant_free(mat);
        return false;
    }

    memset(mat->binary_weights, 0, bin_bytes);
    for (int r = 0; r < rows; r++) mat->row_scales[r] = 1.0f;
    mat->outlier_count = 0;
    mat->total_bytes = bin_bytes + scale_bytes + (size_t)max_outliers * (sizeof(uint32_t) + sizeof(float));
    mat->is_initialized = true;

    return true;
}

bool qwn_pquant_encode(
    QwnPQuantMatrix *mat,
    const float *dense_weights,
    int rows,
    int cols,
    float outlier_threshold
) {
    if (!mat || !dense_weights || rows != mat->rows || cols != mat->cols) return false;

    int cols_u64 = (cols + 63) / 64;
    mat->outlier_count = 0;

    for (int r = 0; r < rows; r++) {
        const float *row_src = dense_weights + (size_t)r * cols;
        uint64_t *row_bin = mat->binary_weights + (size_t)r * cols_u64;

        float abs_sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float val = row_src[c];
            abs_sum += fabsf(val);

            if (val >= 0.0f) {
                row_bin[c / 64] |= (1ULL << (c % 64));
            }

            /* Check sensitive parameter outliers */
            if (fabsf(val) > outlier_threshold) {
                mat->outlier_indices[mat->outlier_count] = ((uint32_t)r << 16) | ((uint32_t)c & 0xFFFF);
                mat->outlier_values[mat->outlier_count] = val;
                mat->outlier_count++;
            }
        }
        mat->row_scales[r] = abs_sum / (float)cols;
    }

    return true;
}

bool qwn_pquant_gemv(
    const QwnPQuantMatrix *mat,
    const float *x_vector,
    float *y_out
) {
    if (!mat || !mat->is_initialized || !x_vector || !y_out) return false;

    int rows = mat->rows;
    int cols = mat->cols;
    int cols_u64 = (cols + 63) / 64;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint64_t *row_bin = mat->binary_weights + (size_t)r * cols_u64;
        float dot = 0.0f;

        for (int c = 0; c < cols; c++) {
            bool bit = (row_bin[c / 64] >> (c % 64)) & 1ULL;
            float sign = bit ? 1.0f : -1.0f;
            dot += sign * x_vector[c];
        }
        y_out[r] = dot * mat->row_scales[r];
    }

    /* Apply sparse outlier corrections */
    for (int i = 0; i < mat->outlier_count; i++) {
        uint32_t idx = mat->outlier_indices[i];
        int r = (int)(idx >> 16);
        int c = (int)(idx & 0xFFFF);
        if (r < rows && c < cols) {
            bool bit = (mat->binary_weights[(size_t)r * cols_u64 + (c / 64)] >> (c % 64)) & 1ULL;
            float original_contrib = (bit ? 1.0f : -1.0f) * mat->row_scales[r] * x_vector[c];
            float true_contrib = mat->outlier_values[i] * x_vector[c];
            y_out[r] += (true_contrib - original_contrib);
        }
    }

    return true;
}

void qwn_pquant_free(QwnPQuantMatrix *mat) {
    if (!mat) return;
    if (mat->binary_weights) free(mat->binary_weights);
    if (mat->row_scales) free(mat->row_scales);
    if (mat->outlier_indices) free(mat->outlier_indices);
    if (mat->outlier_values) free(mat->outlier_values);
    memset(mat, 0, sizeof(*mat));
}
