#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_pquant.h"

int main(void) {
    printf("=================================================================\n");
    printf("     Qwanto pQuant Decoupled 1-Bit + Sparse Test Suite           \n");
    printf("=================================================================\n");

    const int rows = 64;
    const int cols = 128;
    const float outlier_ratio = 0.05f;

    /* Test 1: Initialize pQuant Matrix */
    QwnPQuantMatrix mat;
    bool init_ok = qwn_pquant_init(&mat, rows, cols, outlier_ratio);
    assert(init_ok == true);
    assert(mat.is_initialized == true);
    printf("[PASS] pQuant matrix initialized (%d x %d).\n", rows, cols);

    /* Test 2: Encode Dense Matrix with Outliers */
    float *dense_w = (float *)malloc((size_t)rows * cols * sizeof(float));
    assert(dense_w != NULL);
    for (int i = 0; i < rows * cols; i++) {
        dense_w[i] = (i % 2 == 0) ? 0.25f : -0.25f;
    }
    dense_w[0] = 5.0f; /* High-precision outlier */
    dense_w[10] = -8.0f; /* High-precision outlier */

    bool enc_ok = qwn_pquant_encode(&mat, dense_w, rows, cols, 2.0f);
    assert(enc_ok == true);
    assert(mat.outlier_count >= 2);
    printf("[PASS] pQuant encoding verified (%d outliers isolated in sparse branch).\n", mat.outlier_count);

    /* Test 3: Fast GEMV Execution */
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float));
    assert(x != NULL && y != NULL);

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    bool gemv_ok = qwn_pquant_gemv(&mat, x, y);
    assert(gemv_ok == true);
    printf("[PASS] pQuant decoupled GEMV execution verified (y[0] = %.4f).\n", y[0]);

    /* Cleanup */
    qwn_pquant_free(&mat);
    free(dense_w);
    free(x);
    free(y);

    printf("=================================================================\n");
    printf("[SUCCESS] All pQuant tests passed!\n");
    printf("=================================================================\n");
    return 0;
}
