#ifndef QWANTO_GPU_H
#define QWANTO_GPU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "qwanto_bitdecoding.h"

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
    int index;
    QwnGPUBackendType backend;
    char name[256];
    char vendor[64];
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    int compute_major;
    int compute_minor;
    float arch_gen_score;       /* 1.0 Blackwell, 0.9 Hopper, 0.85 Ada, 0.7 Ampere, 0.5 Turing, 0.75 RDNA3, 0.6 RDNA2, 0.8 Apple M */
    bool is_discrete;
    float current_utilization;   /* 0.0 - 1.0 */
    float temperature_c;
    float composite_score;      /* 0.4*VRAM + 0.3*Compute + 0.2*Arch + 0.1*(1.0-util) */
} QwnGPUDeviceInfo;

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
    bool has_tensor_cores;
    QwnTensorCoreArch tc_arch;
    
    /* Dynamic library handles */
    void *driver_handle;
    void *runtime_handle;
    void *blas_handle;
    
    /* Asynchronous streams & double-buffering */
    void *compute_stream;
    void *transfer_stream;
    void *pinned_scratch_buf;
    size_t pinned_scratch_size;
    
    /* BitDecoding Engine */
    QwnBitDecodingEngine bitdec_engine;
    
    /* Selected Device Details */
    QwnGPUDeviceInfo selected_device_info;
    
    /* Diagnostic string */
    char diagnostic_msg[512];
} QwnGPUContext;

/* -------------------------------------------------------------------------
 * Multi-GPU Fabric & Tensor Sharding Context (1, 2, 4, 8 GPUs)
 * ------------------------------------------------------------------------- */
#define QWN_MAX_GPUS 8

typedef struct {
    int num_devices;
    QwnGPUContext devices[QWN_MAX_GPUS];
    bool is_multi_gpu;
    int sharding_dim;               /* Column (0) or Row (1) sharding */
    void *shard_buffers[QWN_MAX_GPUS];
    size_t shard_sizes[QWN_MAX_GPUS];
    float scaling_factor;           /* Linear scaling benchmark metric (e.g. 1.92x on 2 GPUs, 3.75x on 4 GPUs) */
} QwnMultiGPUContext;

/* -------------------------------------------------------------------------
 * GPU Fabric APIs
 * ------------------------------------------------------------------------- */

/* Initialize GPU context with target backend (or QWN_GPU_BACKEND_AUTO with intelligent scoring) */
bool qwn_gpu_init(QwnGPUContext *ctx, QwnGPUBackendType preferred_backend);

/* Initialize specific device index directly */
bool qwn_gpu_init_device(QwnGPUContext *ctx, int device_index);

/* Enumerate all available GPUs and compute their suitability scores */
int qwn_gpu_enumerate_devices(QwnGPUDeviceInfo *devices, int max_devices);

/* Compute composite score for a device (0.4*VRAM + 0.3*Compute + 0.2*Arch + 0.1*(1.0-Util)) */
float qwn_gpu_calculate_score(const QwnGPUDeviceInfo *info);

/* Select highest-scoring device (preferring discrete GPU if VRAM > 4GB) */
int qwn_gpu_select_best_device(const QwnGPUDeviceInfo *devices, int num_devices);

/* List all GPUs with detailed specifications, score breakdown, and recommended selection */
void qwn_gpu_list_all_devices(void);

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

/* Allocate pinned (page-locked) host memory for DMA zero-copy transfers */
void *qwn_gpu_alloc_pinned(QwnGPUContext *ctx, size_t bytes);

/* Free pinned host memory */
void qwn_gpu_free_pinned(QwnGPUContext *ctx, void *ptr);

/* Copy host memory to device */
bool qwn_gpu_memcpy_to_device(QwnGPUContext *ctx, void *dst_device, const void *src_host, size_t bytes);

/* Copy device memory to host */
bool qwn_gpu_memcpy_to_host(QwnGPUContext *ctx, void *dst_host, const void *src_device, size_t bytes);

/* Synchronize pending compute queues */
void qwn_gpu_synchronize(QwnGPUContext *ctx);

/* Clean up and unload dynamic GPU libraries */
void qwn_gpu_shutdown(QwnGPUContext *ctx);

/* -------------------------------------------------------------------------
 * Multi-GPU Management & Sharding APIs
 * ------------------------------------------------------------------------- */
bool qwn_multigpu_init(QwnMultiGPUContext *mgpu, int requested_devices);
bool qwn_multigpu_shard_tensor(QwnMultiGPUContext *mgpu, const void *weights, size_t total_bytes);
bool qwn_multigpu_forward(QwnMultiGPUContext *mgpu, const float *input, float *output);
void qwn_multigpu_shutdown(QwnMultiGPUContext *mgpu);

/* -------------------------------------------------------------------------
 * Unified GPU Accelerated Inference Operations
 * ------------------------------------------------------------------------- */

/* Fused In-Register TurboQuant / BitDecoding Attention Forward Pass */
bool qwn_gpu_attention_forward(
    QwnGPUContext *ctx,
    const float *q_tensor,             /* [n_heads * head_dim] */
    const uint8_t *k_packed_cache,     /* [seq_len * n_heads * (head_dim/2)] */
    const uint8_t *v_packed_cache,     /* [seq_len * n_heads * (head_dim/2)] */
    float *out_context_tensor,         /* [n_heads * head_dim] */
    int n_heads,
    int head_dim,
    int seq_len,
    float sm_scale
);

/* BitDecoding Tensor Core Dedicated Attention Forward */
bool qwn_gpu_bitdecoding_attention_forward(
    QwnGPUContext *ctx,
    const float *q_tensor,
    const uint8_t *k_packed_cache,
    const uint8_t *v_packed_cache,
    float *out_context_tensor,
    int n_heads,
    int head_dim,
    int seq_len,
    float sm_scale
);

/* Vectorized Matrix-Vector Multiplication */
bool qwn_gpu_matmul_forward(
    QwnGPUContext *ctx,
    const void *weights_packed,
    const float *x_vector,
    float *out_y_vector,
    int rows,
    int cols
);

/* Fast RMSNorm Forward Pass */
bool qwn_gpu_rmsnorm_forward(
    QwnGPUContext *ctx,
    const float *input,
    const float *weight,
    float *output,
    int hidden_dim,
    float eps
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_GPU_H */
