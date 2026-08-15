#ifndef QWANTO_GPU_H
#define QWANTO_GPU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Qwanto Multi-Vendor GPU Dynamic Runtime Integration & Device Fabric
 * Supports: NVIDIA CUDA, AMD ROCm/HIP, Apple Metal, Intel oneAPI/SYCL, Vulkan
 * ------------------------------------------------------------------------- */

typedef enum {
    QWN_GPU_BACKEND_NONE   = 0,
    QWN_GPU_BACKEND_CUDA   = 1,
    QWN_GPU_BACKEND_ROCM   = 2,
    QWN_GPU_BACKEND_METAL  = 3,
    QWN_GPU_BACKEND_SYCL   = 4,
    QWN_GPU_BACKEND_VULKAN = 5,
    QWN_GPU_BACKEND_AUTO   = 6
} QwnGPUBackendType;

typedef struct {
    QwnGPUBackendType active_backend;
    char backend_name[32];
    char device_name[256];
    int device_index;
    int device_count;
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    int compute_units;
    bool is_initialized;
    bool is_hardware_accelerated;
    
    /* Dynamic library handles */
    void *driver_handle;
    void *runtime_handle;
    void *blas_handle;
    
    /* Diagnostic string */
    char diagnostic_msg[512];
} QwnGPUContext;

/* -------------------------------------------------------------------------
 * GPU Fabric APIs
 * ------------------------------------------------------------------------- */

/* Initialize GPU context with target backend (or QWN_GPU_BACKEND_AUTO) */
bool qwn_gpu_init(QwnGPUContext *ctx, QwnGPUBackendType preferred_backend);

/* Parse backend string: "auto", "cuda", "rocm", "metal", "sycl", "vulkan", "cpu" */
QwnGPUBackendType qwn_gpu_parse_backend_name(const char *name);

/* Print comprehensive diagnostic information */
void qwn_gpu_print_diagnostics(const QwnGPUContext *ctx);

/* Query available GPU device count for a specific backend */
int qwn_gpu_detect_devices(QwnGPUBackendType backend);

/* Allocate unified device buffer */
void *qwn_gpu_alloc(QwnGPUContext *ctx, size_t bytes);

/* Free unified device buffer */
void qwn_gpu_free(QwnGPUContext *ctx, void *ptr);

/* Copy host memory to device */
bool qwn_gpu_memcpy_to_device(QwnGPUContext *ctx, void *dst_device, const void *src_host, size_t bytes);

/* Copy device memory to host */
bool qwn_gpu_memcpy_to_host(QwnGPUContext *ctx, void *dst_host, const void *src_device, size_t bytes);

/* Synchronize pending compute queues */
void qwn_gpu_synchronize(QwnGPUContext *ctx);

/* Clean up and unload dynamic GPU libraries */
void qwn_gpu_shutdown(QwnGPUContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_GPU_H */
