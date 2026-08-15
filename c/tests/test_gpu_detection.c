#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "qwanto_gpu.h"

int main(void) {
    printf("=================================================================\n");
    printf("       Qwanto GPU Dynamic Runtime Detection Test Suite          \n");
    printf("=================================================================\n");

    /* Test 1: Parse backend names */
    assert(qwn_gpu_parse_backend_name("cuda") == QWN_GPU_BACKEND_CUDA);
    assert(qwn_gpu_parse_backend_name("nvidia") == QWN_GPU_BACKEND_CUDA);
    assert(qwn_gpu_parse_backend_name("rocm") == QWN_GPU_BACKEND_ROCM);
    assert(qwn_gpu_parse_backend_name("hip") == QWN_GPU_BACKEND_ROCM);
    assert(qwn_gpu_parse_backend_name("vulkan") == QWN_GPU_BACKEND_VULKAN);
    assert(qwn_gpu_parse_backend_name("metal") == QWN_GPU_BACKEND_METAL);
    assert(qwn_gpu_parse_backend_name("sycl") == QWN_GPU_BACKEND_SYCL);
    assert(qwn_gpu_parse_backend_name("cpu") == QWN_GPU_BACKEND_NONE);
    assert(qwn_gpu_parse_backend_name("auto") == QWN_GPU_BACKEND_AUTO);
    assert(qwn_gpu_parse_backend_name(NULL) == QWN_GPU_BACKEND_AUTO);
    printf("[PASS] Backend string parsing verified.\n");

    /* Test 2: Auto-probing and graceful fallback */
    QwnGPUContext ctx;
    bool ok = qwn_gpu_init(&ctx, QWN_GPU_BACKEND_AUTO);
    assert(ok == true);
    assert(ctx.is_initialized == true);
    printf("[PASS] GPU Context initialization succeeded.\n");
    printf("       Active Backend: %s\n", ctx.backend_name);
    printf("       Hardware Device: %s\n", ctx.device_name);
    printf("       Diagnostic: %s\n", ctx.diagnostic_msg);

    /* Test 3: Print diagnostics */
    qwn_gpu_print_diagnostics(&ctx);

    /* Test 4: Memory management */
    void *buf = qwn_gpu_alloc(&ctx, 1024 * 1024);
    assert(buf != NULL);
    float test_data[16] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(qwn_gpu_memcpy_to_device(&ctx, buf, test_data, sizeof(test_data)) == true);
    float read_data[16] = {0};
    assert(qwn_gpu_memcpy_to_host(&ctx, read_data, buf, sizeof(test_data)) == true);
    assert(read_data[0] == 1.0f && read_data[3] == 4.0f);
    qwn_gpu_free(&ctx, buf);
    printf("[PASS] Unified buffer allocation and memory copy verified.\n");

    /* Test 5: CPU explicit backend */
    QwnGPUContext cpu_ctx;
    ok = qwn_gpu_init(&cpu_ctx, QWN_GPU_BACKEND_NONE);
    assert(ok == true);
    assert(cpu_ctx.active_backend == QWN_GPU_BACKEND_NONE);
    assert(cpu_ctx.is_hardware_accelerated == false);
    printf("[PASS] Explicit CPU backend fallback verified.\n");

    /* Clean up */
    qwn_gpu_shutdown(&ctx);
    qwn_gpu_shutdown(&cpu_ctx);

    printf("=================================================================\n");
    printf("[SUCCESS] All GPU dynamic runtime detection tests passed!\n");
    printf("=================================================================\n");
    return 0;
}
