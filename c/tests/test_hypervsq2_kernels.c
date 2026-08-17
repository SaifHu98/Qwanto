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

/* Generate a synthetically valid HyperVSQ-2 block row */
static void generate_synthetic_hypervsq2_row(uint8_t *buf, int K, unsigned int seed) {
    srand(seed);
    int blocks = (K + 255) / 256;
    for (int b = 0; b < blocks; b++) {
        uint8_t *blk = buf + (size_t)b * 74;
        /* FP16 base_scale ~ [0.01 .. 0.5] -> 0x2000 .. 0x3800 */
        uint16_t hs = 0x3000 + (rand() % 0x0800);
        /* FP16 m_base ~ [-0.2 .. 0.2] */
        uint16_t hm = 0x2800 + (rand() % 0x0400);
        if (rand() & 1) hm |= 0x8000; /* signed */

        memcpy(blk, &hs, 2);
        memcpy(blk + 2, &hm, 2);

        /* 8 sub-scales in [1..8] */
        for (int i = 0; i < 4; i++) {
            uint8_t s0 = (rand() % 8) + 1;
            uint8_t s1 = (rand() % 8) + 1;
            blk[4 + i] = (s1 << 4) | (s0 & 0x0F);
        }

        /* 2 reserved bytes */
        blk[8] = 0;
        blk[9] = 0;

        /* 64 bytes of 2-bit codes in [0..3] */
        for (int i = 0; i < 64; i++) {
            blk[10 + i] = (uint8_t)(rand() & 0xFF);
        }
    }
}

static int run_unpack_candidate_test(void) {
    uint8_t packed[8];
    uint8_t shift_mask[32];
    uint8_t lut[32];
    volatile uint64_t checksum = 0;
    unsigned int state = 0x9e3779b9U;

    for (int sample = 0; sample < 10000; sample++) {
        for (int i = 0; i < 8; i++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            packed[i] = (uint8_t)state;
        }
        qwn_hypervsq2_unpack_shift_mask(packed, shift_mask);
        qwn_hypervsq2_unpack_lut(packed, lut);
        if (memcmp(shift_mask, lut, sizeof(shift_mask)) != 0) {
            fprintf(stderr, "[FAIL] HyperVSQ-2 unpack candidates disagree at sample %d\n", sample);
            return -1;
        }
    }

    const int iterations = 1000000;
    for (int iteration = 0; iteration < iterations; iteration++) {
        qwn_hypervsq2_unpack_shift_mask(packed, shift_mask);
        checksum += shift_mask[iteration & 31];
    }
    double shift_start = get_time_sec();
    for (int iteration = 0; iteration < iterations; iteration++) {
        qwn_hypervsq2_unpack_shift_mask(packed, shift_mask);
        checksum += shift_mask[iteration & 31];
    }
    double shift_seconds = get_time_sec() - shift_start;
    double lut_start = get_time_sec();
    for (int iteration = 0; iteration < iterations; iteration++) {
        qwn_hypervsq2_unpack_lut(packed, lut);
        checksum += lut[iteration & 31];
    }
    double lut_seconds = get_time_sec() - lut_start;
    printf("Unpack candidates: shift_mask=%.6f ms, lut=%.6f ms, checksum=%llu\n",
           shift_seconds * 1000.0, lut_seconds * 1000.0,
           (unsigned long long)checksum);
    return 0;
}

static int run_differential_test(int K, int N) {
    const int blocks = (K + 255) / 256;
    const size_t row_bytes = (size_t)blocks * 74;
    const size_t total_weight_bytes = row_bytes * (size_t)N;

    uint8_t *raw_blocks = (uint8_t *)malloc(total_weight_bytes);
    int8_t *q8 = (int8_t *)malloc((size_t)blocks * 256);
    float *out_scalar = (float *)malloc((size_t)N * sizeof(float));
    float *out_precomputed = (float *)malloc((size_t)N * sizeof(float));
    float *out_avx2 = (float *)malloc((size_t)N * sizeof(float));
    float *out_vnni = (float *)malloc((size_t)N * sizeof(float));
    float *out_vnni_delayed = (float *)malloc((size_t)N * sizeof(float));
    float *out_vnni_rows2 = (float *)malloc((size_t)N * sizeof(float));
    float *out_vnni_rows4 = (float *)malloc((size_t)N * sizeof(float));
    int32_t *activation_sums = (int32_t *)calloc((size_t)blocks * 8, sizeof(int32_t));

    if (!raw_blocks || !q8 || !out_scalar || !out_precomputed || !out_avx2 ||
        !out_vnni || !out_vnni_delayed || !out_vnni_rows2 || !out_vnni_rows4 ||
        !activation_sums) {
        fprintf(stderr, "Allocation failed in differential test\n");
        return -1;
    }

    for (int n = 0; n < N; n++) {
        generate_synthetic_hypervsq2_row(raw_blocks + (size_t)n * row_bytes, K, 1000 + n * 17);
    }

    srand(42);
    for (int i = 0; i < blocks * 256; i++) {
        if (i < K) {
            q8[i] = (int8_t)((rand() % 255) - 127);
        } else {
            q8[i] = 0;
        }
    }
    for (int b = 0; b < blocks; b++) {
        int valid = K - b * 256;
        if (valid > 256) valid = 256;
        for (int oct = 0; oct < 8; oct++) {
            int cap = valid - oct * 32;
            if (cap > 32) cap = 32;
            if (cap <= 0) continue;
            for (int i = 0; i < cap; i++)
                activation_sums[b * 8 + oct] += q8[b * 256 + oct * 32 + i];
        }
    }

    float x_scale = 0.0078125f;

    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    qwn_gemv_hypervsq2_scalar(raw_blocks, q8, NULL, x_scale, K, N, row_bytes, out_scalar);
    qwn_gemv_hypervsq2_scalar(raw_blocks, q8, activation_sums, x_scale, K, N,
                               row_bytes, out_precomputed);
    if (cpu->has_avx2) {
        qwn_gemv_hypervsq2_avx2(raw_blocks, q8, activation_sums, x_scale, K, N,
                                row_bytes, out_avx2);
    } else {
        memcpy(out_avx2, out_scalar, (size_t)N * sizeof(float));
    }
    if (cpu->has_vnni) {
        qwn_gemv_hypervsq2_vnni(raw_blocks, q8, activation_sums, x_scale, K, N,
                                row_bytes, out_vnni);
        qwn_gemv_hypervsq2_vnni_delayed(raw_blocks, q8, activation_sums, x_scale, K, N,
                                        row_bytes, out_vnni_delayed);
        qwn_gemv_hypervsq2_vnni_delayed_rows(raw_blocks, q8, activation_sums, x_scale, K, N,
                                             row_bytes, out_vnni_rows2, 2);
        qwn_gemv_hypervsq2_vnni_delayed_rows(raw_blocks, q8, activation_sums, x_scale, K, N,
                                             row_bytes, out_vnni_rows4, 4);
    } else {
        memcpy(out_vnni, out_scalar, (size_t)N * sizeof(float));
        memcpy(out_vnni_delayed, out_scalar, (size_t)N * sizeof(float));
        memcpy(out_vnni_rows2, out_scalar, (size_t)N * sizeof(float));
        memcpy(out_vnni_rows4, out_scalar, (size_t)N * sizeof(float));
    }

    float max_diff_avx2 = 0.0f;
    float max_diff_vnni = 0.0f;
    float max_diff_vnni_delayed = 0.0f;
    float max_diff_vnni_rows2 = 0.0f;
    float max_diff_vnni_rows4 = 0.0f;
    float max_diff_precomputed = 0.0f;

    for (int n = 0; n < N; n++) {
        float diff_precomputed = fabsf(out_scalar[n] - out_precomputed[n]);
        if (diff_precomputed > max_diff_precomputed) max_diff_precomputed = diff_precomputed;
        float diff_a = fabsf(out_scalar[n] - out_avx2[n]);
        if (diff_a > max_diff_avx2) max_diff_avx2 = diff_a;

        float diff_v = fabsf(out_scalar[n] - out_vnni[n]);
        if (diff_v > max_diff_vnni) max_diff_vnni = diff_v;
        float diff_delayed = fabsf(out_scalar[n] - out_vnni_delayed[n]);
        if (diff_delayed > max_diff_vnni_delayed) max_diff_vnni_delayed = diff_delayed;
        float diff_rows2 = fabsf(out_scalar[n] - out_vnni_rows2[n]);
        if (diff_rows2 > max_diff_vnni_rows2) max_diff_vnni_rows2 = diff_rows2;
        float diff_rows4 = fabsf(out_scalar[n] - out_vnni_rows4[n]);
        if (diff_rows4 > max_diff_vnni_rows4) max_diff_vnni_rows4 = diff_rows4;
    }

    int passed = 1;
    if (max_diff_precomputed > 1e-3f) {
        fprintf(stderr, "[FAIL] K=%d N=%d activation-sum max diff: %f\n",
                K, N, max_diff_precomputed);
        passed = 0;
    }
    if (cpu->has_avx2 && max_diff_avx2 > 1e-3f) {
        fprintf(stderr, "[FAIL] K=%d N=%d AVX2 max diff: %f (scalar=%f, avx2=%f)\n",
                K, N, max_diff_avx2, out_scalar[0], out_avx2[0]);
        passed = 0;
    }
    if (cpu->has_vnni && max_diff_vnni > 1e-3f) {
        fprintf(stderr, "[FAIL] K=%d N=%d VNNI max diff: %f (scalar=%f, vnni=%f)\n",
                K, N, max_diff_vnni, out_scalar[0], out_vnni[0]);
        passed = 0;
    }
    if (cpu->has_vnni && max_diff_vnni_delayed > 1e-3f) {
        fprintf(stderr, "[FAIL] K=%d N=%d delayed VNNI max diff: %f\n",
                K, N, max_diff_vnni_delayed);
        passed = 0;
    }
    if (cpu->has_vnni && max_diff_vnni_rows2 > 1e-3f) {
        fprintf(stderr, "[FAIL] K=%d N=%d delayed VNNI 2-row max diff: %f\n",
                K, N, max_diff_vnni_rows2);
        passed = 0;
    }
    if (cpu->has_vnni && max_diff_vnni_rows4 > 1e-3f) {
        fprintf(stderr, "[FAIL] K=%d N=%d delayed VNNI 4-row max diff: %f\n",
                K, N, max_diff_vnni_rows4);
        passed = 0;
    }

    free(raw_blocks);
    free(q8);
    free(out_scalar);
    free(out_precomputed);
    free(out_avx2);
    free(out_vnni);
    free(out_vnni_delayed);
    free(out_vnni_rows2);
    free(out_vnni_rows4);
    free(activation_sums);

    return passed ? 0 : -1;
}

static void benchmark_kernels(int K, int N, int iters) {
    const int blocks = (K + 255) / 256;
    const size_t row_bytes = (size_t)blocks * 74;
    const size_t total_weight_bytes = row_bytes * (size_t)N;

    uint8_t *raw_blocks = (uint8_t *)malloc(total_weight_bytes);
    int8_t *q8 = (int8_t *)malloc((size_t)blocks * 256);
    float *out = (float *)malloc((size_t)N * sizeof(float));

    for (int n = 0; n < N; n++) {
        generate_synthetic_hypervsq2_row(raw_blocks + (size_t)n * row_bytes, K, 5000 + n);
    }
    for (int i = 0; i < blocks * 256; i++) {
        q8[i] = (int8_t)((rand() % 255) - 127);
    }
    float x_scale = 0.0078125f;

    const QwnCpuFeatures *cpu = qwn_get_cpu_features();

    /* Benchmark scalar */
    double t0 = get_time_sec();
    for (int it = 0; it < iters; it++) {
        qwn_gemv_hypervsq2_scalar(raw_blocks, q8, NULL, x_scale, K, N, row_bytes, out);
    }
    double t1 = get_time_sec();
    double time_scalar = (t1 - t0) / iters;
    double gflops_scalar = (2.0 * (double)K * (double)N) / (time_scalar * 1e9);

    /* Benchmark AVX2 */
    double time_avx2 = 0.0;
    double gflops_avx2 = 0.0;
    if (cpu->has_avx2) {
        t0 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            qwn_gemv_hypervsq2_avx2(raw_blocks, q8, NULL, x_scale, K, N, row_bytes, out);
        }
        t1 = get_time_sec();
        time_avx2 = (t1 - t0) / iters;
        gflops_avx2 = (2.0 * (double)K * (double)N) / (time_avx2 * 1e9);
    }

    /* Benchmark VNNI */
    double time_vnni = 0.0;
    double gflops_vnni = 0.0;
    if (cpu->has_vnni) {
        t0 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            qwn_gemv_hypervsq2_vnni(raw_blocks, q8, NULL, x_scale, K, N, row_bytes, out);
        }
        t1 = get_time_sec();
        time_vnni = (t1 - t0) / iters;
        gflops_vnni = (2.0 * (double)K * (double)N) / (time_vnni * 1e9);
    }

    printf("Benchmark K=%-5d N=%-5d | Scalar: %7.2f GFLOPS (%6.2f us) | AVX2: %7.2f GFLOPS (%6.2f us, %5.1fx) | VNNI: %7.2f GFLOPS (%6.2f us, %5.1fx)\n",
           K, N,
           gflops_scalar, time_scalar * 1e6,
           gflops_avx2, time_avx2 * 1e6, time_scalar / (time_avx2 > 0 ? time_avx2 : 1.0),
           gflops_vnni, time_vnni * 1e6, time_scalar / (time_vnni > 0 ? time_vnni : 1.0));

    free(raw_blocks);
    free(q8);
    free(out);
}

int main(void) {
    printf("==================================================\n");
    printf("     Qwanto HyperVSQ-2 Kernel Verification Suite  \n");
    printf("==================================================\n");

    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    printf("Detected CPU Features:\n");
    printf("  AVX2:     %s\n", cpu->has_avx2 ? "YES" : "NO");
    printf("  F16C:     %s\n", cpu->has_f16c ? "YES" : "NO");
    printf("  FMA:      %s\n", cpu->has_fma ? "YES" : "NO");
    printf("  AVX-VNNI: %s\n", cpu->has_vnni ? "YES" : "NO");
    printf("  AVX-512F: %s\n", cpu->has_avx512f ? "YES" : "NO");
    printf("--------------------------------------------------\n");

    if (run_unpack_candidate_test() != 0) return 1;

    int test_k_dims[] = {1, 7, 15, 16, 31, 32, 33, 63, 64, 65, 100, 128, 200, 255, 256, 257, 512, 1024, 2048, 4096};
    int test_n_dims[] = {1, 4, 16, 32, 64, 128, 512};

    int total_tests = 0;
    int failed_tests = 0;

    for (size_t ik = 0; ik < sizeof(test_k_dims)/sizeof(test_k_dims[0]); ik++) {
        for (size_t in = 0; in < sizeof(test_n_dims)/sizeof(test_n_dims[0]); in++) {
            int K = test_k_dims[ik];
            int N = test_n_dims[in];
            total_tests++;
            if (run_differential_test(K, N) != 0) {
                failed_tests++;
            }
        }
    }

    printf("Differential Tests: %d passed / %d total\n", total_tests - failed_tests, total_tests);
    if (failed_tests > 0) {
        printf("FAILED %d differential test cases!\n", failed_tests);
        return 1;
    }
    printf("[SUCCESS] All differential numerical tests passed!\n");

    printf("\n--------------------------------------------------\n");
    printf("           Microkernel Performance Benchmarks      \n");
    printf("--------------------------------------------------\n");

    benchmark_kernels(4096, 4096, 50);
    benchmark_kernels(4096, 8192, 25);
    benchmark_kernels(8192, 4096, 25);
    benchmark_kernels(4096, 14336, 20);

    printf("==================================================\n");
    return 0;
}
