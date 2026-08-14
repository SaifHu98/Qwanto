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
    int K;                  /* In features */
    int N;                  /* Out features */
    int num_layers;
    void *stream;           /* cudaStream_t handle */
} QwnCUDALayerContext;

/* Initialize CUDA device and allocate stream buffers */
int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id);

/* Execute QWN-HyperVSQ dequantization and GEMV dot product on GPU */
int qwn_cuda_hypervsq_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N);

/* Free CUDA layer resources */
void qwn_cuda_layer_free(QwnCUDALayerContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWN_HYPERVSQ_CUDA_H */
