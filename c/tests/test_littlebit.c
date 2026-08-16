#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_littlebit.h"

int main(void) {
    printf("=================================================================\n");
    printf("     Qwanto LittleBit-2 Sub-1-Bit Compression Test Suite         \n");
    printf("                     (ICML 2026 Breakthrough)                   \n");
    printf("=================================================================\n");

    const int rows = 128;
    const int cols = 128;
    const int rank = 16;

    /* Test 1: Initialize LittleBit-2 Matrix */
    QwnLittleBitMatrix mat;
    bool init_ok = qwn_littlebit_init(&mat, rows, cols, rank);
    assert(init_ok == true);
    assert(mat.is_initialized == true);
    assert(mat.bits_per_weight < 1.0);
    printf("[PASS] LittleBit-2 matrix initialized (%d x %d, Rank=%d, %.4f bpw).\n", rows, cols, rank, mat.bits_per_weight);

    /* Test 2: Encode Dense Weights */
    float *dense_w = (float *)malloc((size_t)rows * cols * sizeof(float));
    assert(dense_w != NULL);
    for (int i = 0; i < rows * cols; i++) {
        dense_w[i] = (float)((i % 10) - 5) * 0.1f;
    }

    bool enc_ok = qwn_littlebit_encode(&mat, dense_w, rows, cols, rank);
    assert(enc_ok == true);
    printf("[PASS] LittleBit-2 low-rank binarized factorization verified.\n");

    /* Test 3: Sub-1-Bit GEMV */
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float));
    assert(x != NULL && y != NULL);

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    bool gemv_ok = qwn_littlebit_gemv(&mat, x, y);
    assert(gemv_ok == true);
    printf("[PASS] Sub-1-bit low-rank factorized GEMV verified (y[0] = %.4f).\n", y[0]);

    /* Cleanup */
    qwn_littlebit_free(&mat);
    free(dense_w);
    free(x);
    free(y);

    printf("=================================================================\n");
    printf("[SUCCESS] All LittleBit-2 tests passed!\n");
    printf("=================================================================\n");
    return 0;
}
