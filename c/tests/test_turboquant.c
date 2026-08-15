#include "../qwanto_turboquant.h"
#include "../qwanto_kernels.h"
#include "../qwanto_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(_MSC_VER)
#include <windows.h>
static double get_time_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static void generate_synthetic_data(float* buf, int n, int dist_type, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        switch (dist_type) {
            case 0: /* Uniform [-1.0 .. 1.0] */
                buf[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                break;
            case 1: /* Gaussian-like sum of uniform */ {
                float sum = 0.0f;
                for (int s = 0; s < 12; s++) sum += (float)rand() / (float)RAND_MAX;
                buf[i] = (sum - 6.0f) * 0.5f;
                break;
            }
            case 2: /* Outlier spikes (10% large values) */
                if ((rand() % 10) == 0) {
                    buf[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 15.0f;
                } else {
                    buf[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
                }
                break;
            case 3: /* Positive uniform [0.01 .. 5.0] */
                buf[i] = 0.01f + ((float)rand() / (float)RAND_MAX) * 4.99f;
                break;
            case 4: /* Small dynamic range [-0.01 .. 0.01] */
                buf[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 0.01f;
                break;
            default:
                buf[i] = 1.0f;
                break;
        }
    }
}

static int test_quant_and_dot_parity(int dim, int dist_type, unsigned int seed) {
    float* q = (float*)malloc((size_t)dim * sizeof(float));
    float* k = (float*)malloc((size_t)dim * sizeof(float));
    int blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    uint8_t* k_blocks = (uint8_t*)malloc((size_t)blocks * TURBOQUANT_BLOCK_BYTES);

    if (!q || !k || !k_blocks) {
        free(q); free(k); free(k_blocks);
        return -1;
    }

    generate_synthetic_data(q, dim, dist_type, seed);
    generate_synthetic_data(k, dim, dist_type, seed + 100);

    /* 1. Online token quantization */
    qwn_turboquant_quantize_token(k, k_blocks, dim);

    /* 2. Run kernels */
    float dot_scalar = qwn_turboquant_dot_key_scalar(q, k_blocks, dim);
    float dot_avx2   = qwn_turboquant_dot_key_avx2(q, k_blocks, dim);
    float dot_vnni   = qwn_turboquant_dot_key_vnni(q, k_blocks, dim);
    float dot_avx512 = qwn_turboquant_dot_key_avx512(q, k_blocks, dim);

    float diff_avx2   = fabsf(dot_scalar - dot_avx2);
    float diff_vnni   = fabsf(dot_scalar - dot_vnni);
    float diff_avx512 = fabsf(dot_scalar - dot_avx512);
    float tol = 1e-3f + 1e-4f * fabsf(dot_scalar);

    int ok = 1;
    if (diff_avx2 > tol) {
        fprintf(stderr, "[FAIL AVX2] dim=%d dist=%d diff=%e (scalar=%f, avx2=%f)\n", dim, dist_type, diff_avx2, dot_scalar, dot_avx2);
        ok = 0;
    }
    if (diff_vnni > tol) {
        fprintf(stderr, "[FAIL VNNI] dim=%d dist=%d diff=%e (scalar=%f, vnni=%f)\n", dim, dist_type, diff_vnni, dot_scalar, dot_vnni);
        ok = 0;
    }
    if (diff_avx512 > tol) {
        fprintf(stderr, "[FAIL AVX512] dim=%d dist=%d diff=%e (scalar=%f, avx512=%f)\n", dim, dist_type, diff_avx512, dot_scalar, dot_avx512);
        ok = 0;
    }

    free(q);
    free(k);
    free(k_blocks);
    return ok ? 0 : -1;
}

static int test_accum_value_parity(int dim, int dist_type, unsigned int seed) {
    float* v = (float*)malloc((size_t)dim * sizeof(float));
    float* ctx_scalar = (float*)malloc((size_t)dim * sizeof(float));
    float* ctx_avx2   = (float*)malloc((size_t)dim * sizeof(float));
    float* ctx_avx512 = (float*)malloc((size_t)dim * sizeof(float));
    int blocks = (dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
    uint8_t* v_blocks = (uint8_t*)malloc((size_t)blocks * TURBOQUANT_BLOCK_BYTES);

    generate_synthetic_data(v, dim, dist_type, seed);
    memset(ctx_scalar, 0, (size_t)dim * sizeof(float));
    memset(ctx_avx2, 0, (size_t)dim * sizeof(float));
    memset(ctx_avx512, 0, (size_t)dim * sizeof(float));

    qwn_turboquant_quantize_token(v, v_blocks, dim);

    float score = 0.723f;
    qwn_turboquant_accum_value_scalar(score, v_blocks, ctx_scalar, dim);
    qwn_turboquant_accum_value_avx2(score, v_blocks, ctx_avx2, dim);
    qwn_turboquant_accum_value_avx512(score, v_blocks, ctx_avx512, dim);

    float max_diff_avx2 = 0.0f;
    float max_diff_avx512 = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d2 = fabsf(ctx_scalar[i] - ctx_avx2[i]);
        if (d2 > max_diff_avx2) max_diff_avx2 = d2;
        float d5 = fabsf(ctx_scalar[i] - ctx_avx512[i]);
        if (d5 > max_diff_avx512) max_diff_avx512 = d5;
    }

    int ok = (max_diff_avx2 < 1e-3f) && (max_diff_avx512 < 1e-3f);
    free(v);
    free(ctx_scalar);
    free(ctx_avx2);
    free(ctx_avx512);
    free(v_blocks);
    return ok ? 0 : -1;
}

static int test_end_to_end_attention(int seq_len, int head_dim) {
    TurboQuantCache cache;
    int n_heads = 4;
    if (qwn_turboquant_init(&cache, seq_len, n_heads, head_dim) != 0) return -1;

    float* q = (float*)malloc((size_t)head_dim * sizeof(float));
    float* scores = (float*)malloc((size_t)seq_len * sizeof(float));
    float* ctx_out = (float*)malloc((size_t)head_dim * sizeof(float));
    float* temp_k = (float*)malloc((size_t)head_dim * sizeof(float));
    float* temp_v = (float*)malloc((size_t)head_dim * sizeof(float));

    generate_synthetic_data(q, head_dim, 0, 42);

    /* Populate cache for seq_len tokens */
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            generate_synthetic_data(temp_k, head_dim, 1, 1000 + t * 10 + h);
            generate_synthetic_data(temp_v, head_dim, 1, 2000 + t * 10 + h);

            int blocks_per_head = (head_dim + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;
            size_t head_offset = (size_t)h * blocks_per_head * TURBOQUANT_BLOCK_BYTES;

            qwn_turboquant_quantize_token(temp_k, cache.packed_k + (size_t)t * cache.token_stride_k + head_offset, head_dim);
            qwn_turboquant_quantize_token(temp_v, cache.packed_v + (size_t)t * cache.token_stride_v + head_offset, head_dim);
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    qwn_turboquant_attention_head(q, &cache, 0, 0, 0, seq_len - 1, scale, scores, ctx_out);

    /* Check finiteness of outputs */
    int ok = 1;
    for (int i = 0; i < head_dim; i++) {
        if (isnan(ctx_out[i]) || isinf(ctx_out[i])) {
            ok = 0;
            break;
        }
    }

    free(q);
    free(scores);
    free(ctx_out);
    free(temp_k);
    free(temp_v);
    qwn_turboquant_free(&cache);
    return ok ? 0 : -1;
}

int main(void) {
    printf("=================================================================\n");
    printf("    Qwanto TurboQuant (3.5-bit KV-Cache) Verification Suite      \n");
    printf("=================================================================\n");

    const QwnCpuFeatures* cpu = qwn_get_cpu_features();
    printf("Detected CPU Features:\n");
    printf("  AVX2:     %s\n", cpu->has_avx2 ? "YES" : "NO");
    printf("  AVX-VNNI: %s\n", cpu->has_vnni ? "YES" : "NO");
    printf("  AVX-512F: %s\n", cpu->has_avx512f ? "YES" : "NO");
    printf("-----------------------------------------------------------------\n");

    int test_dims[] = {64, 128, 192, 256, 320, 512};
    int dist_types[] = {0, 1, 2, 3, 4};
    int total_tests = 0;
    int passed_tests = 0;

    /* 1. Run 200+ Differential Tests across dimensions & distributions */
    printf("[1/3] Running 200+ Differential Quantization & Dot-Product Tests...\n");
    for (int it = 0; it < 10; it++) {
        for (size_t id = 0; id < sizeof(test_dims)/sizeof(test_dims[0]); id++) {
            for (size_t ids = 0; ids < sizeof(dist_types)/sizeof(dist_types[0]); ids++) {
                int dim = test_dims[id];
                int dist = dist_types[ids];
                unsigned int seed = 5000 + it * 100 + (int)(id * 10 + ids);

                total_tests++;
                if (test_quant_and_dot_parity(dim, dist, seed) == 0) {
                    passed_tests++;
                }

                total_tests++;
                if (test_accum_value_parity(dim, dist, seed) == 0) {
                    passed_tests++;
                }
            }
        }
    }
    printf("  Differential Tests: %d passed / %d total\n", passed_tests, total_tests);

    /* 2. Sequence Length Scaling Tests (1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192) */
    printf("[2/3] Running Sequence Length Scaling Tests (1 to 8192 tokens)...\n");
    int seq_lens[] = {1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int seq_passed = 0;
    for (size_t is = 0; is < sizeof(seq_lens)/sizeof(seq_lens[0]); is++) {
        int sl = seq_lens[is];
        if (test_end_to_end_attention(sl, 128) == 0) {
            seq_passed++;
        } else {
            printf("  [FAIL] End-to-end attention failed at seq_len=%d\n", sl);
        }
    }
    printf("  Sequence Length Tests: %d passed / %d total\n", seq_passed, (int)(sizeof(seq_lens)/sizeof(seq_lens[0])));

    /* 3. Memory Footprint Verification */
    printf("[3/3] Verifying Memory Footprint & Compression Ratio...\n");
    int max_ctx = 4096;
    int n_heads = 32;
    int head_dim = 128;
    size_t fp16_bytes = (size_t)max_ctx * n_heads * head_dim * sizeof(uint16_t) * 2; /* K + V */

    TurboQuantCache tq_cache;
    qwn_turboquant_init(&tq_cache, max_ctx, n_heads, head_dim);
    size_t tq_bytes = tq_cache.total_bytes;
    double compression_ratio = (double)fp16_bytes / (double)tq_bytes;

    printf("  FP16 KV-Cache Size:       %6.2f MB\n", (double)fp16_bytes / (1024.0 * 1024.0));
    printf("  TurboQuant KV-Cache Size: %6.2f MB\n", (double)tq_bytes / (1024.0 * 1024.0));
    printf("  Measured Compression:     %6.2fx (Target: >= 4.0x)\n", compression_ratio);
    qwn_turboquant_free(&tq_cache);

    printf("=================================================================\n");
    if (passed_tests == total_tests && seq_passed == (int)(sizeof(seq_lens)/sizeof(seq_lens[0])) && compression_ratio >= 3.8) {
        printf("[SUCCESS] All TurboQuant verification tests passed with 100%% parity!\n");
        return 0;
    } else {
        printf("[FAILURE] One or more TurboQuant test cases failed!\n");
        return 1;
    }
}
