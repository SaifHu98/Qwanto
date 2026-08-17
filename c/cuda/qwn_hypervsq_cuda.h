#ifndef QWN_HYPERVSQ_CUDA_H
#define QWN_HYPERVSQ_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(QWN_CUDA_BUILDING_DLL)
#define QWN_CUDA_API __declspec(dllexport)
#else
#define QWN_CUDA_API
#endif

typedef struct {
    void *dev_weights;      /* GPU VRAM pointer to quantized weights */
    void *dev_x;            /* GPU VRAM pointer to input activations */
    void *dev_y;            /* GPU VRAM pointer to output activations */
    float *pinned_x;        /* Pinned host memory (cudaHostAlloc) for DMA transfers */
    float *pinned_y;        /* Pinned host memory (cudaHostAlloc) for DMA transfers */
    int K;                  /* In features */
    int N;                  /* Out features */
    int num_layers;
    void *stream_compute;   /* cudaStream_t handle for compute warps */
    void *stream_prefetch;  /* cudaStream_t handle for async DMA transfers */
    size_t weight_bytes;
    size_t x_capacity;
    size_t y_capacity;
    const void *cached_weights;
    int device_id;
} QwnCUDALayerContext;

typedef struct {
    uint64_t matmul_count;
    uint64_t upload_bytes;
    size_t resident_bytes;
    int device_id;
    char kernel[32];
} QwnCudaMetrics;

/* Initialize CUDA device and allocate stream buffers with pinned memory */
QWN_CUDA_API int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id);

/* Execute QWN-HyperVSQ dequantization and GEMV dot product on GPU with zero-copy stream sync */
QWN_CUDA_API int qwn_cuda_hypervsq2_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N);

/* Process-wide API used by qwnrun's dynamic loader. The functions keep the
 * quantized tensor in VRAM between token calls and use separate transfer and
 * compute streams for the activation/output path. */
QWN_CUDA_API int qwn_cuda_init(int gpu_id);
QWN_CUDA_API int qwn_cuda_gemv_hypervsq2(int rows, int cols, const void *weights,
                                        const float *x, float *out);
QWN_CUDA_API int qwn_cuda_gemv_q4_0(int rows, int cols, const void *weights,
                                    const float *x, float *out);
QWN_CUDA_API int qwn_cuda_get_metrics(QwnCudaMetrics *metrics);
QWN_CUDA_API void qwn_cuda_shutdown(void);

/* Allocate page-locked pinned memory on host */
QWN_CUDA_API void *qwn_cuda_host_alloc_pinned(size_t bytes);

/* Free page-locked pinned memory on host */
QWN_CUDA_API void qwn_cuda_host_free_pinned(void *ptr);

/* Free CUDA layer resources */
QWN_CUDA_API void qwn_cuda_layer_free(QwnCUDALayerContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWN_HYPERVSQ_CUDA_H */
