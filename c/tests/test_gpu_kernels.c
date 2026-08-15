#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_gpu.h"

int main(void) {
    printf("=================================================================\n");
    printf("        Qwanto Unified GPU Compute & Kernel Test Suite           \n");
    printf("=================================================================\n");

    QwnGPUContext ctx;
    bool init_ok = qwn_gpu_init(&ctx, QWN_GPU_BACKEND_AUTO);
    assert(init_ok == true);
    qwn_gpu_print_diagnostics(&ctx);

    /* 1. Test RMSNorm */
    const int hidden_dim = 128;
    float *x = (float *)malloc(hidden_dim * sizeof(float));
    float *w = (float *)malloc(hidden_dim * sizeof(float));
    float *y = (float *)malloc(hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        x[i] = 1.0f;
        w[i] = 2.0f;
    }
    bool rms_ok = qwn_gpu_rmsnorm_forward(&ctx, x, w, y, hidden_dim, 1e-5f);
    assert(rms_ok == true);
    /* Since x is all 1s, RMS = 1.0, normalized is 1.0, multiplied by w(2.0) = 2.0 */
    assert(fabsf(y[0] - 2.0f) < 1e-3f);
    printf("[PASS] qwn_gpu_rmsnorm_forward verified successfully.\n");

    /* 2. Test MatMul */
    const int rows = 64;
    const int cols = 128;
    float *weights = (float *)malloc(rows * cols * sizeof(float));
    float *out_y = (float *)malloc(rows * sizeof(float));
    for (int i = 0; i < rows * cols; i++) weights[i] = 0.01f;
    bool mm_ok = qwn_gpu_matmul_forward(&ctx, weights, x, out_y, rows, cols);
    assert(mm_ok == true);
    assert(fabsf(out_y[0] - 1.28f) < 1e-3f);
    printf("[PASS] qwn_gpu_matmul_forward verified successfully.\n");

    /* 3. Test TurboQuant Attention */
    const int n_heads = 4;
    const int head_dim = 64;
    const int seq_len = 8;
    float *q = (float *)malloc(n_heads * head_dim * sizeof(float));
    uint8_t *k_packed = (uint8_t *)malloc(seq_len * n_heads * (head_dim / 2));
    uint8_t *v_packed = (uint8_t *)malloc(seq_len * n_heads * (head_dim / 2));
    float *out_attn = (float *)malloc(n_heads * head_dim * sizeof(float));

    for (int i = 0; i < n_heads * head_dim; i++) q[i] = 0.5f;
    memset(k_packed, 0x88, seq_len * n_heads * (head_dim / 2)); /* Code 8 => (8*0.125)-1.0 = 0.0 */
    memset(v_packed, 0xAA, seq_len * n_heads * (head_dim / 2)); /* Code 10 => (10*0.125)-1.0 = 0.25 */

    bool attn_ok = qwn_gpu_attention_forward(
        &ctx, q, k_packed, v_packed, out_attn,
        n_heads, head_dim, seq_len, 1.0f / sqrtf((float)head_dim)
    );
    assert(attn_ok == true);
    assert(fabsf(out_attn[0] - 0.25f) < 1e-3f);
    printf("[PASS] qwn_gpu_attention_forward verified successfully.\n");

    /* 4. Test Pinned Memory Allocation */
    void *pinned = qwn_gpu_alloc_pinned(&ctx, 1024 * 1024);
    assert(pinned != NULL);
    qwn_gpu_free_pinned(&ctx, pinned);
    printf("[PASS] Pinned (page-locked) host memory allocation verified.\n");

    /* Cleanup */
    free(x); free(w); free(y);
    free(weights); free(out_y);
    free(q); free(k_packed); free(v_packed); free(out_attn);
    qwn_gpu_shutdown(&ctx);

    printf("=================================================================\n");
    printf("[SUCCESS] All GPU accelerated inference kernels verified!\n");
    printf("=================================================================\n");
    return 0;
}
