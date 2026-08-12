#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../qwanto_core.h"
#include "../qwanto_router.h"
#include "../qwanto_attention.h"

// Helper to get time in seconds
static double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

// 1. Matmul Verification
void test_matmul(void) {
    printf("[Test] Verifying Matrix Multiplication Kernel...\n");
    int m = 128;
    int n = 256;
    int m_tokens = 1;

    int8_t* activations = (int8_t*)malloc(m_tokens * n * sizeof(int8_t));
    uint8_t* packed_weights = (uint8_t*)malloc(m * (n / 2) * sizeof(uint8_t));
    int32_t* outputs_ref = (int32_t*)malloc(m_tokens * m * sizeof(int32_t));
    int32_t* outputs_simd = (int32_t*)malloc(m_tokens * m * sizeof(int32_t));

    // Initialize test data
    for (int i = 0; i < m_tokens * n; i++) {
        activations[i] = (int8_t)((i * 17 + 5) % 101 - 50); // signed [-50, 50]
    }
    for (int i = 0; i < m * (n / 2); i++) {
        packed_weights[i] = (uint8_t)((i * 23 + 3) & 0xFF); // packed values
    }

    // Run reference calculation
    memset(outputs_ref, 0, m_tokens * m * sizeof(int32_t));
    for (int t = 0; t < m_tokens; t++) {
        for (int i = 0; i < m; i++) {
            int32_t sum = 0;
            const uint8_t* w = &packed_weights[i * (n / 2)];
            const int8_t* act = &activations[t * n];
            for (int j = 0; j < n; j++) {
                uint8_t byte = w[j / 2];
                int8_t val = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                sum += (int32_t)act[j] * (int32_t)(val - 8);
            }
            outputs_ref[t * m + i] = sum;
        }
    }

    // Run blocked SIMD calculation
    qwanto_matmul_blocked(activations, packed_weights, outputs_simd, m_tokens, m, n);

    // Validate outputs
    int ok = 1;
    for (int i = 0; i < m_tokens * m; i++) {
        if (outputs_ref[i] != outputs_simd[i]) {
            printf("  Mismatch at output[%d]: ref = %d, got = %d\n", i, outputs_ref[i], outputs_simd[i]);
            ok = 0;
            break;
        }
    }

    if (ok) {
        printf("  -> Correctness validated (outputs match reference exactly).\n");
    } else {
        printf("  -> FAILED matrix multiplication check!\n");
        exit(1);
    }

    // Speed benchmark
    int iters = 10000;
    double t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        qwanto_matmul_blocked(activations, packed_weights, outputs_simd, m_tokens, m, n);
    }
    double dt = get_time_sec() - t0;
    double ops = 2.0 * m_tokens * m * n * iters;
    printf("  -> Benchmark: %.2f GOPs/s (elapsed time: %.3f s for %d iterations)\n", 
           (ops / dt) / 1e9, dt, iters);

    free(activations);
    free(packed_weights);
    free(outputs_ref);
    free(outputs_simd);
}

// 2. Router Verification
void test_router(void) {
    printf("[Test] Verifying LSH MoE Router...\n");
    int hidden_dim = 4096;
    int n_experts = 256;
    int top_k = 8;

    int8_t* activations = (int8_t*)malloc(hidden_dim * sizeof(int8_t));
    for (int i = 0; i < hidden_dim; i++) {
        activations[i] = (int8_t)((i * 31 + 7) % 256 - 128);
    }

    int expert_ids[8];
    qwanto_route_lsh(activations, hidden_dim, n_experts, top_k, expert_ids);

    printf("  -> Selected experts: ");
    int ok = 1;
    for (int i = 0; i < top_k; i++) {
        printf("%d ", expert_ids[i]);
        if (expert_ids[i] < 0 || expert_ids[i] >= n_experts) {
            ok = 0;
        }
        for (int j = 0; j < i; j++) {
            if (expert_ids[i] == expert_ids[j]) {
                ok = 0; // duplicates found
            }
        }
    }
    printf("\n");

    if (ok) {
        printf("  -> Correctness validated (all expert IDs unique and within bounds).\n");
    } else {
        printf("  -> FAILED LSH router check!\n");
        exit(1);
    }

    // Benchmark routing speed
    int iters = 1000000;
    double t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        qwanto_route_lsh(activations, hidden_dim, n_experts, top_k, expert_ids);
    }
    double dt = get_time_sec() - t0;
    double lat_us = (dt / iters) * 1e6;
    printf("  -> Benchmark: %.3f microseconds per token routing (target: <1.0us)\n", lat_us);
    if (lat_us < 1.0) {
        printf("  -> PASS: Sub-microsecond goal achieved!\n");
    } else {
        printf("  -> WARNING: Routing took longer than 1us.\n");
    }

    free(activations);
}

// 3. Linear Attention Verification
void test_attention(void) {
    printf("[Test] Verifying Linear Attention (Retention)...\n");
    int n_heads = 8;
    int head_dim = 64;
    int seq_len = 16;
    float decay = 0.95f;

    QwantoLinearAttentionState state;
    if (qwanto_linear_attention_init(&state, n_heads, head_dim) != 0) {
        printf("  -> FAILED state initialization!\n");
        exit(1);
    }

    int total_dim = n_heads * head_dim;
    float* Q = (float*)malloc(seq_len * total_dim * sizeof(float));
    float* K = (float*)malloc(seq_len * total_dim * sizeof(float));
    float* V = (float*)malloc(seq_len * total_dim * sizeof(float));
    float* O_simd = (float*)malloc(seq_len * total_dim * sizeof(float));
    float* O_ref = (float*)malloc(seq_len * total_dim * sizeof(float));

    // Initialize test sequence inputs
    for (int i = 0; i < seq_len * total_dim; i++) {
        Q[i] = (float)((i * 13 % 100) - 50) / 100.f;
        K[i] = (float)((i * 17 % 100) - 50) / 100.f;
        V[i] = (float)((i * 19 % 100) - 50) / 100.f;
    }

    // Run reference implementation sequentially
    float* ref_state = (float*)calloc(n_heads * head_dim * head_dim, sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            float* s = &ref_state[h * head_dim * head_dim];
            const float* q_t = &Q[t * total_dim + h * head_dim];
            const float* k_t = &K[t * total_dim + h * head_dim];
            const float* v_t = &V[t * total_dim + h * head_dim];
            float* o_t = &O_ref[t * total_dim + h * head_dim];

            // In-place state update
            for (int i = 0; i < head_dim; i++) {
                for (int j = 0; j < head_dim; j++) {
                    s[i * head_dim + j] = decay * s[i * head_dim + j] + k_t[i] * v_t[j];
                }
            }

            // Output projection
            memset(o_t, 0, head_dim * sizeof(float));
            for (int i = 0; i < head_dim; i++) {
                for (int j = 0; j < head_dim; j++) {
                    o_t[j] += q_t[i] * s[i * head_dim + j];
                }
            }
        }
    }

    // Run SIMD implementation
    qwanto_linear_attention_reset(&state);
    qwanto_linear_attention_prefill(&state, Q, K, V, decay, seq_len, O_simd);

    // Validate correctness
    int ok = 1;
    float max_diff = 0.f;
    for (int i = 0; i < seq_len * total_dim; i++) {
        float diff = fabsf(O_simd[i] - O_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) {
            printf("  Mismatch at outputs[%d]: ref = %.6f, SIMD = %.6f, diff = %.6f\n", 
                   i, O_ref[i], O_simd[i], diff);
            ok = 0;
            break;
        }
    }

    // Validate final state matrix
    for (int i = 0; i < n_heads * head_dim * head_dim; i++) {
        float diff = fabsf(state.state[i] - ref_state[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) {
            printf("  Mismatch in final state[%d]: ref = %.6f, SIMD = %.6f, diff = %.6f\n", 
                   i, ref_state[i], state.state[i], diff);
            ok = 0;
            break;
        }
    }

    if (ok) {
        printf("  -> Correctness validated (max diff = %.9f, matches reference within tolerance).\n", max_diff);
    } else {
        printf("  -> FAILED Linear Attention checks!\n");
        exit(1);
    }

    // Benchmark attention speed
    int iters = 20000;
    double t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        qwanto_linear_attention_reset(&state);
        qwanto_linear_attention_prefill(&state, Q, K, V, decay, seq_len, O_simd);
    }
    double dt = get_time_sec() - t0;
    double total_tokens = (double)seq_len * iters;
    printf("  -> Benchmark: %.2f tokens per second (elapsed: %.3f s for %d sequence runs)\n", 
           total_tokens / dt, dt, iters);

    free(Q);
    free(K);
    free(V);
    free(O_simd);
    free(O_ref);
    free(ref_state);
    qwanto_linear_attention_destroy(&state);
}

int main(void) {
    printf("=====================================================\n");
    printf("[Qwanto Architecture Test Suite Starting]\n");
    printf("=====================================================\n");
    test_matmul();
    test_router();
    test_attention();
    printf("=====================================================\n");
    printf("[ALL TESTS PASSED SUCCESSFULLY!]\n");
    printf("=====================================================\n");
    return 0;
}
