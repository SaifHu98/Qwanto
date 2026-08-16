#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_bitdecoding.h"
#include "qwanto_gpu.h"

int main(void) {
    printf("=================================================================\n");
    printf("      Qwanto BitDecoding Tensor Core KV-Cache Test Suite         \n");
    printf("                  (HPCA 2026 Breakthrough)                      \n");
    printf("=================================================================\n");

    const int n_heads = 4;
    const int head_dim = 64;
    const int max_seq_len = 128;
    const int seq_len = 16;

    /* Test 1: Initialize BitDecoding Engine with Blackwell SM100 configuration */
    QwnBitDecodingEngine engine;
    bool init_ok = qwn_bitdecoding_init(&engine, n_heads, head_dim, max_seq_len, 100);
    assert(init_ok == true);
    assert(engine.is_initialized == true);
    assert(engine.cfg.tc_arch == QWN_TC_ARCH_BLACKWELL);
    assert(engine.cfg.has_nvfp4_support == true);
    assert(engine.cfg.mma_tile_m == 16 && engine.cfg.mma_tile_k == 32);
    printf("[PASS] BitDecoding Engine initialized with Blackwell NVFP4 configuration.\n");

    /* Test 2: Pack and swizzle TurboQuant linear cache into Tensor Core layout */
    size_t packed_bytes = (size_t)seq_len * n_heads * (head_dim / 2);
    uint8_t *k_linear = (uint8_t *)malloc(packed_bytes);
    uint8_t *v_linear = (uint8_t *)malloc(packed_bytes);
    assert(k_linear != NULL && v_linear != NULL);

    memset(k_linear, 0x88, packed_bytes); /* Code 8 => 0.0f */
    memset(v_linear, 0xAA, packed_bytes); /* Code 10 => 0.25f */

    bool pack_ok = qwn_bitdecoding_pack_kv(&engine, k_linear, v_linear, seq_len);
    assert(pack_ok == true);
    printf("[PASS] TurboQuant KV-cache successfully swizzled to Tensor Core layout.\n");

    /* Test 3: Execute BitDecoding Attention Step */
    float *q_heads = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    float *out_context = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    for (int i = 0; i < n_heads * head_dim; i++) q_heads[i] = 0.5f;

    bool step_ok = qwn_bitdecoding_attention_step(
        &engine, q_heads, out_context, seq_len, 1.0f / sqrtf((float)head_dim)
    );
    assert(step_ok == true);
    assert(fabsf(out_context[0] - 0.25f) < 1e-3f);
    printf("[PASS] BitDecoding Tensor Core attention forward step passed (result = %.4f).\n", out_context[0]);

    /* Test 4: GPU Context Integrated BitDecoding Dispatch */
    QwnGPUContext gpu_ctx;
    bool gpu_init = qwn_gpu_init(&gpu_ctx, QWN_GPU_BACKEND_AUTO);
    assert(gpu_init == true);

    float *out_gpu_context = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    bool gpu_attn_ok = qwn_gpu_bitdecoding_attention_forward(
        &gpu_ctx, q_heads, k_linear, v_linear, out_gpu_context,
        n_heads, head_dim, seq_len, 1.0f / sqrtf((float)head_dim)
    );
    assert(gpu_attn_ok == true);
    assert(fabsf(out_gpu_context[0] - 0.25f) < 1e-3f);
    printf("[PASS] GPU Context integrated BitDecoding dispatch verified.\n");

    /* Cleanup */
    qwn_bitdecoding_free(&engine);
    qwn_gpu_shutdown(&gpu_ctx);
    free(k_linear);
    free(v_linear);
    free(q_heads);
    free(out_context);
    free(out_gpu_context);

    printf("=================================================================\n");
    printf("[SUCCESS] All BitDecoding Tensor Core KV-cache tests passed!\n");
    printf("=================================================================\n");
    return 0;
}
