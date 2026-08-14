#include "qwn_hypervsq_cuda.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

/*
 * CUDA Warp-Level Kernel for QWN-HyperVSQ (256 Elements per Octa-Superblock)
 * Each warp of 32 threads processes one or more 32-element octants in parallel.
 */
__global__ void qwn_hypervsq_gemv_kernel(
    const uint8_t * __restrict__ weights,
    const float * __restrict__ x,
    float * __restrict__ y,
    int K, int N)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x; // 0..31
    int blocks = (K + 255) / 256;
    const uint8_t *row_ptr = weights + (size_t)row * blocks * 138;

    float thread_sum = 0.0f;

    for (int b = 0; b < blocks; b++) {
        const uint8_t *blk = row_ptr + b * 138;
        
        // Read superblock base scale and zero-point offset (FP16)
        half hs = *reinterpret_cast<const half*>(blk);
        half hm = *reinterpret_cast<const half*>(blk + 2);
        float d_base = __half2float(hs);
        float m_base = __half2float(hm);

        const uint8_t *sub_scales = blk + 4;
        const uint8_t *qs = blk + 10;

        #pragma unroll
        for (int oct = 0; oct < 8; oct++) {
            int k_idx = b * 256 + oct * 32 + lane;
            if (k_idx < K) {
                // Sub-octant 4-bit scale
                uint8_t sb = sub_scales[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float sub_scale = d_base * ((float)s_val * 0.125f);

                // Unpack 4-bit quantized weight for this lane
                const uint8_t *q_oct = qs + oct * 16;
                uint8_t byte = q_oct[lane >> 1];
                int q = ((lane & 1) ? (byte >> 4) : (byte & 0x0F)) - 8;

                float w_val = (float)q * sub_scale + m_base;
                thread_sum += w_val * x[k_idx];
            }
        }
    }

    // Warp-level butterfly reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    if (lane == 0) {
        y[row] = thread_sum;
    }
}

extern "C" int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id) {
    if (!ctx) return -1;
    cudaSetDevice(device_id);
    ctx->K = K;
    ctx->N = N;
    cudaStream_t s;
    cudaStreamCreate(&s);
    ctx->stream = (void*)s;
    cudaMalloc(&ctx->dev_x, (size_t)K * sizeof(float));
    cudaMalloc(&ctx->dev_y, (size_t)N * sizeof(float));
    return 0;
}

extern "C" int qwn_cuda_hypervsq_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N) {
    if (!ctx || !weights || !x || !y) return -1;
    cudaStream_t s = (cudaStream_t)ctx->stream;
    cudaMemcpyAsync(ctx->dev_x, x, (size_t)K * sizeof(float), cudaMemcpyHostToDevice, s);

    dim3 block(WARP_SIZE, 4);
    dim3 grid((N + block.y - 1) / block.y);

    qwn_hypervsq_gemv_kernel<<<grid, block, 0, s>>>(
        (const uint8_t*)weights, (const float*)ctx->dev_x, (float*)ctx->dev_y, K, N
    );

    cudaMemcpyAsync(y, ctx->dev_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return 0;
}

extern "C" void qwn_cuda_layer_free(QwnCUDALayerContext *ctx) {
    if (!ctx) return;
    if (ctx->dev_x) cudaFree(ctx->dev_x);
    if (ctx->dev_y) cudaFree(ctx->dev_y);
    if (ctx->stream) cudaStreamDestroy((cudaStream_t)ctx->stream);
}

#else
/* Stub implementation for non-CUDA host compilation */
int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id) { (void)ctx; (void)K; (void)N; (void)device_id; return -1; }
int qwn_cuda_hypervsq_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N) { (void)ctx; (void)weights; (void)x; (void)y; (void)K; (void)N; return -1; }
void qwn_cuda_layer_free(QwnCUDALayerContext *ctx) { (void)ctx; }
#endif
