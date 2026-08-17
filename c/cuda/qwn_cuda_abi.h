#ifndef QWN_CUDA_ABI_H
#define QWN_CUDA_ABI_H

/*
 * Versioned C ABI between qwnrun and qwn_cuda.dll.
 *
 * This header intentionally contains no CUDA runtime types.  The host can
 * validate and load the DLL without linking against cudart, while the CUDA
 * implementation can use the same fixed-width contract.  Every public
 * structure begins with QwnCudaAbiHeader so newer implementations can append
 * fields without making older qwnrun builds read past the advertised size.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(QWN_CUDA_BUILDING_DLL)
#define QWN_CUDA_ABI_API __declspec(dllexport)
#else
#define QWN_CUDA_ABI_API
#endif

#define QWN_CUDA_ABI_VERSION 1u
#define QWN_CUDA_ABI_NAME "qwn_cuda_hypervsq2"
#define QWN_CUDA_HYPERVSQ2_BLOCK_BYTES 74u
#define QWN_CUDA_HYPERVSQ2_BLOCK_ELEMENTS 256u
#define QWN_CUDA_MAX_RESIDENT_TENSORS 512u

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t reserved[4];
} QwnCudaAbiHeader;

static inline void qwn_cuda_abi_header_init(QwnCudaAbiHeader *header,
                                             uint32_t struct_size) {
    if (!header) return;
    header->struct_size = struct_size;
    header->abi_version = QWN_CUDA_ABI_VERSION;
    header->reserved[0] = 0;
    header->reserved[1] = 0;
    header->reserved[2] = 0;
    header->reserved[3] = 0;
}

enum {
    QWN_CUDA_CAP_HYPERVSQ2_GEMV = 1ull << 0,
    QWN_CUDA_CAP_HYPERVSQ2_GEMM = 1ull << 1,
    QWN_CUDA_CAP_RESIDENT_WEIGHTS = 1ull << 2,
    QWN_CUDA_CAP_TELEMETRY = 1ull << 3,
    QWN_CUDA_CAP_DEVICE_ENUMERATION = 1ull << 4
};

typedef enum {
    QWN_CUDA_STATUS_OK = 0,
    QWN_CUDA_STATUS_INVALID_ARGUMENT = -1,
    QWN_CUDA_STATUS_ABI_MISMATCH = -2,
    QWN_CUDA_STATUS_UNAVAILABLE = -3,
    QWN_CUDA_STATUS_OUT_OF_MEMORY = -4,
    QWN_CUDA_STATUS_UNSUPPORTED = -5,
    QWN_CUDA_STATUS_RUNTIME_ERROR = -6
} QwnCudaStatus;

typedef enum {
    QWN_CUDA_TENSOR_HYPERVSQ2_74 = 1
} QwnCudaTensorDType;

typedef enum {
    QWN_CUDA_INPUT_Q8 = 1
} QwnCudaInputMode;

typedef struct {
    QwnCudaAbiHeader header;
    uint64_t capability_bits;
    uint32_t max_devices;
    uint32_t max_resident_tensors;
    uint32_t hypervsq2_block_bytes;
    uint32_t hypervsq2_block_elements;
    char abi_name[32];
} QwnCudaAbiInfo;

typedef struct {
    QwnCudaAbiHeader header;
    int32_t device_id;
    int32_t compute_major;
    int32_t compute_minor;
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    char name[128];
    char driver[64];
} QwnCudaDeviceInfo;

typedef struct {
    QwnCudaAbiHeader header;
    int32_t device_id;
    uint32_t flags;
    uint64_t memory_budget_bytes;
    uint64_t context_size;
} QwnCudaContextOptions;

typedef struct {
    QwnCudaAbiHeader header;
    void *opaque;
} QwnCudaContextHandle;

typedef struct {
    QwnCudaAbiHeader header;
    void *opaque;
} QwnCudaTensorHandle;

typedef struct {
    QwnCudaAbiHeader header;
    const void *host_data;
    uint64_t data_bytes;
    uint32_t dtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t block_bytes;
} QwnCudaTensorUpload;

typedef struct {
    QwnCudaAbiHeader header;
    QwnCudaTensorHandle tensor;
    const int8_t *input_q8;
    float input_scale;
    const float *input_scales;
    float *output;
    uint32_t batch;
    uint32_t rows;
    uint32_t cols;
    uint32_t input_stride;
    uint32_t output_stride;
    uint32_t input_mode;
} QwnCudaGemmRequest;

typedef struct {
    QwnCudaAbiHeader header;
    uint64_t gpu_matmul_count;
    uint64_t gpu_kernel_launch_count;
    uint64_t gpu_projection_count;
    uint64_t gpu_upload_count;
    uint64_t gpu_upload_bytes;
    uint64_t gpu_resident_bytes;
    uint64_t cpu_matmul_fallback_count;
    uint64_t unsupported_projection_count;
    double gpu_kernel_ms;
    double gpu_transfer_ms;
    double gpu_sync_ms;
    int32_t device_id;
    uint32_t reserved_device;
    char device_name[128];
    char kernel_type[32];
    char dll_hash[65];
} QwnCudaTelemetry;

typedef int (*QwnCudaAbiQueryFn)(QwnCudaAbiInfo *info);
typedef int (*QwnCudaCapabilityQueryFn)(QwnCudaAbiInfo *info);
typedef int (*QwnCudaEnumerateDevicesFn)(QwnCudaDeviceInfo *devices,
                                         uint32_t capacity,
                                         uint32_t *count);
typedef int (*QwnCudaContextCreateFn)(const QwnCudaContextOptions *options,
                                      QwnCudaContextHandle *context);
typedef int (*QwnCudaContextDestroyFn)(QwnCudaContextHandle *context);
typedef int (*QwnCudaTensorUploadFn)(QwnCudaContextHandle *context,
                                     const QwnCudaTensorUpload *upload,
                                     QwnCudaTensorHandle *tensor);
typedef int (*QwnCudaTensorReleaseFn)(QwnCudaContextHandle *context,
                                      QwnCudaTensorHandle *tensor);
typedef int (*QwnCudaGemvFn)(QwnCudaContextHandle *context,
                             const QwnCudaGemmRequest *request,
                             QwnCudaTelemetry *telemetry);
typedef int (*QwnCudaGemmFn)(QwnCudaContextHandle *context,
                             const QwnCudaGemmRequest *request,
                             QwnCudaTelemetry *telemetry);
typedef int (*QwnCudaSynchronizeFn)(QwnCudaContextHandle *context);
typedef int (*QwnCudaTelemetryFn)(QwnCudaContextHandle *context,
                                  QwnCudaTelemetry *telemetry);
typedef int (*QwnCudaLastErrorFn)(char *buffer, uint32_t buffer_size);

QWN_CUDA_ABI_API int qwn_cuda_abi_query(QwnCudaAbiInfo *info);
QWN_CUDA_ABI_API int qwn_cuda_abi_get_capabilities(QwnCudaAbiInfo *info);
QWN_CUDA_ABI_API int qwn_cuda_abi_enumerate_devices(QwnCudaDeviceInfo *devices,
                                                     uint32_t capacity,
                                                     uint32_t *count);
QWN_CUDA_ABI_API int qwn_cuda_abi_context_create(
    const QwnCudaContextOptions *options, QwnCudaContextHandle *context);
QWN_CUDA_ABI_API int qwn_cuda_abi_context_destroy(QwnCudaContextHandle *context);
QWN_CUDA_ABI_API int qwn_cuda_abi_upload_tensor(QwnCudaContextHandle *context,
                                                const QwnCudaTensorUpload *upload,
                                                QwnCudaTensorHandle *tensor);
QWN_CUDA_ABI_API int qwn_cuda_abi_release_tensor(QwnCudaContextHandle *context,
                                                 QwnCudaTensorHandle *tensor);
QWN_CUDA_ABI_API int qwn_cuda_abi_hypervsq2_gemv(QwnCudaContextHandle *context,
                                                const QwnCudaGemmRequest *request,
                                                QwnCudaTelemetry *telemetry);
QWN_CUDA_ABI_API int qwn_cuda_abi_hypervsq2_gemm(QwnCudaContextHandle *context,
                                                const QwnCudaGemmRequest *request,
                                                QwnCudaTelemetry *telemetry);
QWN_CUDA_ABI_API int qwn_cuda_abi_synchronize(QwnCudaContextHandle *context);
QWN_CUDA_ABI_API int qwn_cuda_abi_get_telemetry(QwnCudaContextHandle *context,
                                                QwnCudaTelemetry *telemetry);
QWN_CUDA_ABI_API int qwn_cuda_abi_last_error(char *buffer, uint32_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* QWN_CUDA_ABI_H */
