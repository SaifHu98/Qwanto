#ifndef QWN_HYPERVSQ_CUDA_H
#define QWN_HYPERVSQ_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
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
} QwnCUDALayerContext;

/* Initialize CUDA device and allocate stream buffers with pinned memory */
int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id);

/* Execute QWN-HyperVSQ dequantization and GEMV dot product on GPU with zero-copy stream sync */
int qwn_cuda_hypervsq_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N);

/* Allocate page-locked pinned memory on host */
void *qwn_cuda_host_alloc_pinned(size_t bytes);

/* Free page-locked pinned memory on host */
void qwn_cuda_host_free_pinned(void *ptr);

/* Free CUDA layer resources */
void qwn_cuda_layer_free(QwnCUDALayerContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWN_HYPERVSQ_CUDA_H */
