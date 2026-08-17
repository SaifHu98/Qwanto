#include "qwn_hypervsq_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <vector>

#if defined(__NVCC__) || (defined(__CUDACC__) && defined(QWN_CUDA_ENABLED))
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

static QwnCUDALayerContext g_qwn_ctx;
static int g_qwn_initialized = 0;
static int g_qwn_refcount = 0;
static int g_qwn_gpu = -1;
static int g_qwn_use_dp4a = 0;
static uint64_t g_qwn_matmul_count = 0;
static uint64_t g_qwn_upload_bytes = 0;
static size_t g_qwn_resident_bytes = 0;
static char g_qwn_kernel[32] = "none";
static std::mutex g_qwn_mutex;

struct ResidentWeight {
    const void *host;
    void *device;
    size_t bytes;
};
static std::vector<ResidentWeight> g_qwn_resident;

static void *qwn_resident_weight(QwnCUDALayerContext *ctx, const void *host,
                                 size_t bytes, cudaStream_t stream) {
    for (const ResidentWeight &entry : g_qwn_resident)
        if (entry.host == host && entry.bytes == bytes) return entry.device;
    void *device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess) return nullptr;
    if (cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaFree(device);
        return nullptr;
    }
    try {
        g_qwn_resident.push_back({host, device, bytes});
    } catch (...) {
        cudaFree(device);
        return nullptr;
    }
    g_qwn_upload_bytes += bytes;
    g_qwn_resident_bytes += bytes;
    (void)ctx;
    return device;
}

/*
 * CUDA Warp-Level Kernel for the native HyperVSQ-2 layout.
 * Each 256-element block is exactly 74 bytes:
 *   fp16 base scale (2), fp16 offset (2), four packed bytes containing eight
 *   4-bit sub-scales, two reserved bytes, and eight 32-element octants with
 *   two bits per value (64), for 74 bytes total.
 * This is deliberately separate from the older 138-byte HyperVSQ layout.
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
    const uint8_t *row_ptr = weights + (size_t)row * blocks * 74;

    float thread_sum = 0.0f;

    for (int b = 0; b < blocks; b++) {
        const uint8_t *blk = row_ptr + b * 74;
        
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
                // Sub-octant 4-bit scale and native two-bit value.
                uint8_t sb = sub_scales[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float sub_scale = d_base * ((float)s_val * 0.125f);

                const uint8_t *q_oct = qs + oct * 8;
                uint8_t byte = q_oct[lane >> 2];
                int q = (byte >> ((lane & 3) * 2)) & 3;

                float w_val = (float)(q - 1) * sub_scale + m_base;
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

__global__ void qwn_q4_gemv_kernel(
    const uint8_t * __restrict__ weights,
    const float * __restrict__ x,
    float * __restrict__ y,
    int K, int N, int use_dp4a)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x;
    int blocks = (K + 31) / 32;
    const uint8_t *row_ptr = weights + (size_t)row * blocks * 18;
#if __CUDA_ARCH__ >= 610
    if (use_dp4a) {
    float max_abs = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE)
        max_abs = fmaxf(max_abs, fabsf(x[k]));
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xFFFFFFFF, max_abs, offset));
    max_abs = __shfl_sync(0xFFFFFFFF, max_abs, 0);
    float x_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    float total = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const uint8_t *block = row_ptr + (size_t)b * 18;
        float scale = __half2float(*reinterpret_cast<const half *>(block));
        int partial = 0;
        for (int j = lane * 4; j < 32; j += WARP_SIZE * 4) {
            int k = b * 32 + j;
            if (k >= K) break;
            uint8_t p0 = block[2 + (j >> 1)];
            uint8_t p1 = block[2 + ((j + 2) >> 1)];
            int w0 = (p0 & 0x0F) - 8;
            int w1 = (p0 >> 4) - 8;
            int w2 = (p1 & 0x0F) - 8;
            int w3 = (p1 >> 4) - 8;
            int x0 = max(-128, min(127, __float2int_rn(x[k] / x_scale)));
            int x1 = k + 1 < K ? max(-128, min(127, __float2int_rn(x[k + 1] / x_scale))) : 0;
            int x2 = k + 2 < K ? max(-128, min(127, __float2int_rn(x[k + 2] / x_scale))) : 0;
            int x3 = k + 3 < K ? max(-128, min(127, __float2int_rn(x[k + 3] / x_scale))) : 0;
            int wp = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
            int xp = (x0 & 0xFF) | ((x1 & 0xFF) << 8) | ((x2 & 0xFF) << 16) | ((x3 & 0xFF) << 24);
            partial = __dp4a(wp, xp, partial);
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        if (lane == 0) total += (float)partial * scale * x_scale;
    }
    if (lane == 0) y[row] = total;
    return;
    }
#endif
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const uint8_t *block = row_ptr + (size_t)b * 18;
        float scale = __half2float(*reinterpret_cast<const half *>(block));
        for (int i = lane; i < 32; i += WARP_SIZE) {
            int k = b * 32 + i;
            if (k < K) {
                uint8_t packed = block[2 + (i >> 1)];
                int q = ((i & 1) ? packed >> 4 : packed & 0x0F) - 8;
                sum += x[k] * (float)q * scale;
            }
        }
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (lane == 0) y[row] = sum;
}

extern "C" void *qwn_cuda_host_alloc_pinned(size_t bytes) {
    void *ptr = NULL;
    cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocWriteCombined);
    return ptr;
}

extern "C" void qwn_cuda_host_free_pinned(void *ptr) {
    if (ptr) cudaFreeHost(ptr);
}

extern "C" int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    ctx->K = K;
    ctx->N = N;
    ctx->device_id = device_id;
    cudaStream_t s_comp, s_pref;
    if (cudaStreamCreateWithFlags(&s_comp, cudaStreamNonBlocking) != cudaSuccess) return -1;
    if (cudaStreamCreateWithFlags(&s_pref, cudaStreamNonBlocking) != cudaSuccess) {
        cudaStreamDestroy(s_comp);
        return -1;
    }
    ctx->stream_compute = (void*)s_comp;
    ctx->stream_prefetch = (void*)s_pref;

    if (cudaMalloc(&ctx->dev_x, (size_t)K * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ctx->dev_y, (size_t)N * sizeof(float)) != cudaSuccess) {
        qwn_cuda_layer_free(ctx);
        return -1;
    }
    ctx->pinned_x = (float*)qwn_cuda_host_alloc_pinned((size_t)K * sizeof(float));
    ctx->pinned_y = (float*)qwn_cuda_host_alloc_pinned((size_t)N * sizeof(float));
    if (!ctx->pinned_x || !ctx->pinned_y) {
        qwn_cuda_layer_free(ctx);
        return -1;
    }
    ctx->x_capacity = (size_t)K * sizeof(float);
    ctx->y_capacity = (size_t)N * sizeof(float);
    return 0;
}

extern "C" int qwn_cuda_hypervsq2_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N) {
    if (!ctx || !weights || !x || !y) return -1;
    cudaStream_t s_comp = (cudaStream_t)ctx->stream_compute;
    cudaStream_t s_pref = (cudaStream_t)ctx->stream_prefetch;

    if (ctx->x_capacity < (size_t)K * sizeof(float)) {
        if (ctx->dev_x) cudaFree(ctx->dev_x);
        if (ctx->pinned_x) cudaFreeHost(ctx->pinned_x);
        ctx->dev_x = NULL; ctx->pinned_x = NULL;
        if (cudaMalloc(&ctx->dev_x, (size_t)K * sizeof(float)) != cudaSuccess) return -1;
        ctx->pinned_x = (float *)qwn_cuda_host_alloc_pinned((size_t)K * sizeof(float));
        if (!ctx->pinned_x) return -1;
        ctx->x_capacity = (size_t)K * sizeof(float);
    }
    if (ctx->y_capacity < (size_t)N * sizeof(float)) {
        if (ctx->dev_y) cudaFree(ctx->dev_y);
        if (ctx->pinned_y) cudaFreeHost(ctx->pinned_y);
        ctx->dev_y = NULL; ctx->pinned_y = NULL;
        if (cudaMalloc(&ctx->dev_y, (size_t)N * sizeof(float)) != cudaSuccess) return -1;
        ctx->pinned_y = (float *)qwn_cuda_host_alloc_pinned((size_t)N * sizeof(float));
        if (!ctx->pinned_y) return -1;
        ctx->y_capacity = (size_t)N * sizeof(float);
    }

    if (cudaSetDevice(ctx->device_id) != cudaSuccess) return -1;
    size_t weight_bytes = (size_t)N * ((K + 255) / 256) * 74;
    void *weights_device = qwn_resident_weight(ctx, weights, weight_bytes, s_pref);
    if (!weights_device) return -1;

    /* Fast zero-copy memory copy via write-combined pinned buffer */
    if (ctx->pinned_x) {
        memcpy(ctx->pinned_x, x, (size_t)K * sizeof(float));
        if (cudaMemcpyAsync(ctx->dev_x, ctx->pinned_x, (size_t)K * sizeof(float), cudaMemcpyHostToDevice, s_pref) != cudaSuccess) return -1;
    } else {
        if (cudaMemcpyAsync(ctx->dev_x, x, (size_t)K * sizeof(float), cudaMemcpyHostToDevice, s_pref) != cudaSuccess) return -1;
    }

    /* Synchronize prefetch stream before launching compute kernel */
    cudaEvent_t ev;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return -1;
    if (cudaEventRecord(ev, s_pref) != cudaSuccess || cudaStreamWaitEvent(s_comp, ev, 0) != cudaSuccess) {
        cudaEventDestroy(ev);
        return -1;
    }

    dim3 block(WARP_SIZE, 4);
    dim3 grid((N + block.y - 1) / block.y);

    qwn_hypervsq_gemv_kernel<<<grid, block, 0, s_comp>>>(
        (const uint8_t*)weights_device, (const float*)ctx->dev_x, (float*)ctx->dev_y, K, N
    );
    if (cudaGetLastError() != cudaSuccess) {
        cudaEventDestroy(ev);
        return -1;
    }

    /* Asynchronously copy results back to pinned host buffer */
    if (ctx->pinned_y) {
        if (cudaMemcpyAsync(ctx->pinned_y, ctx->dev_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, s_comp) != cudaSuccess) {
            cudaEventDestroy(ev);
            return -1;
        }
        if (cudaStreamSynchronize(s_comp) != cudaSuccess) {
            cudaEventDestroy(ev);
            return -1;
        }
        memcpy(y, ctx->pinned_y, (size_t)N * sizeof(float));
    } else {
        if (cudaMemcpyAsync(y, ctx->dev_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, s_comp) != cudaSuccess ||
            cudaStreamSynchronize(s_comp) != cudaSuccess) {
            cudaEventDestroy(ev);
            return -1;
        }
    }

    cudaEventDestroy(ev);
    g_qwn_matmul_count++;
    strncpy(g_qwn_kernel, "hypervsq2-74", sizeof(g_qwn_kernel) - 1);
    g_qwn_kernel[sizeof(g_qwn_kernel) - 1] = '\0';
    return 0;
}

extern "C" int qwn_cuda_init(int gpu_id) {
    std::lock_guard<std::mutex> lock(g_qwn_mutex);
    if (g_qwn_initialized) {
        if (g_qwn_gpu != gpu_id) return -1;
        g_qwn_refcount++;
        return 0;
    }
    if (qwn_cuda_layer_init(&g_qwn_ctx, 1, 1, gpu_id) != 0) return -1;
    g_qwn_initialized = 1;
    g_qwn_refcount = 1;
    g_qwn_gpu = gpu_id;
    g_qwn_matmul_count = 0;
    g_qwn_upload_bytes = 0;
    g_qwn_resident_bytes = 0;
    strncpy(g_qwn_kernel, "none", sizeof(g_qwn_kernel) - 1);
    g_qwn_kernel[sizeof(g_qwn_kernel) - 1] = '\0';
    const char *dp4a = getenv("QWN_CUDA_INT8");
    g_qwn_use_dp4a = dp4a && strcmp(dp4a, "1") == 0;
    return 0;
}

extern "C" int qwn_cuda_gemv_hypervsq2(int rows, int cols, const void *weights,
                                       const float *x, float *out) {
    std::lock_guard<std::mutex> lock(g_qwn_mutex);
    if (!g_qwn_initialized || rows < 1 || cols < 1) return -1;
    return qwn_cuda_hypervsq2_gemv(&g_qwn_ctx, weights, x, out, cols, rows);
}

extern "C" int qwn_cuda_gemv_q4_0(int rows, int cols, const void *weights,
                                    const float *x, float *out) {
    std::lock_guard<std::mutex> lock(g_qwn_mutex);
    if (!g_qwn_initialized || !weights || !x || !out || rows < 1 || cols < 1) return -1;
    QwnCUDALayerContext *ctx = &g_qwn_ctx;
    if (cudaSetDevice(ctx->device_id) != cudaSuccess) return -1;
    cudaStream_t s_comp = (cudaStream_t)ctx->stream_compute;
    cudaStream_t s_pref = (cudaStream_t)ctx->stream_prefetch;
    if (ctx->x_capacity < (size_t)cols * sizeof(float)) {
        if (ctx->dev_x) cudaFree(ctx->dev_x);
        if (ctx->pinned_x) cudaFreeHost(ctx->pinned_x);
        ctx->dev_x = NULL; ctx->pinned_x = NULL;
        if (cudaMalloc(&ctx->dev_x, (size_t)cols * sizeof(float)) != cudaSuccess) return -1;
        ctx->pinned_x = (float *)qwn_cuda_host_alloc_pinned((size_t)cols * sizeof(float));
        if (!ctx->pinned_x) return -1;
        ctx->x_capacity = (size_t)cols * sizeof(float);
    }
    if (ctx->y_capacity < (size_t)rows * sizeof(float)) {
        if (ctx->dev_y) cudaFree(ctx->dev_y);
        if (ctx->pinned_y) cudaFreeHost(ctx->pinned_y);
        ctx->dev_y = NULL; ctx->pinned_y = NULL;
        if (cudaMalloc(&ctx->dev_y, (size_t)rows * sizeof(float)) != cudaSuccess) return -1;
        ctx->pinned_y = (float *)qwn_cuda_host_alloc_pinned((size_t)rows * sizeof(float));
        if (!ctx->pinned_y) return -1;
        ctx->y_capacity = (size_t)rows * sizeof(float);
    }
    size_t bytes = (size_t)rows * ((cols + 31) / 32) * 18;
    void *weights_device = qwn_resident_weight(ctx, weights, bytes, s_pref);
    if (!weights_device) return -1;
    if (cudaMemcpyAsync(ctx->dev_x, x, (size_t)cols * sizeof(float),
                        cudaMemcpyHostToDevice, s_pref) != cudaSuccess) return -1;
    cudaEvent_t event;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) return -1;
    if (cudaEventRecord(event, s_pref) != cudaSuccess ||
        cudaStreamWaitEvent(s_comp, event, 0) != cudaSuccess) {
        cudaEventDestroy(event);
        return -1;
    }
    qwn_q4_gemv_kernel<<<rows, WARP_SIZE, 0, s_comp>>>(
        (const uint8_t *)weights_device, (const float *)ctx->dev_x,
        (float *)ctx->dev_y, cols, rows, g_qwn_use_dp4a);
    if (cudaGetLastError() != cudaSuccess) {
        cudaEventDestroy(event);
        return -1;
    }
    if (cudaMemcpyAsync(ctx->pinned_y, ctx->dev_y, (size_t)rows * sizeof(float),
                        cudaMemcpyDeviceToHost, s_comp) != cudaSuccess) {
        cudaEventDestroy(event);
        return -1;
    }
    cudaError_t status = cudaStreamSynchronize(s_comp);
    if (status == cudaSuccess) {
        memcpy(out, ctx->pinned_y, (size_t)rows * sizeof(float));
        g_qwn_matmul_count++;
        strncpy(g_qwn_kernel, "q4_0", sizeof(g_qwn_kernel) - 1);
        g_qwn_kernel[sizeof(g_qwn_kernel) - 1] = '\0';
    }
    cudaEventDestroy(event);
    return status == cudaSuccess && cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int qwn_cuda_get_metrics(QwnCudaMetrics *metrics) {
    if (!metrics) return -1;
    std::lock_guard<std::mutex> lock(g_qwn_mutex);
    memset(metrics, 0, sizeof(*metrics));
    metrics->matmul_count = g_qwn_matmul_count;
    metrics->upload_bytes = g_qwn_upload_bytes;
    metrics->resident_bytes = g_qwn_resident_bytes;
    metrics->device_id = g_qwn_gpu;
    strncpy(metrics->kernel, g_qwn_kernel, sizeof(metrics->kernel) - 1);
    return g_qwn_initialized ? 0 : -1;
}

extern "C" void qwn_cuda_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_qwn_mutex);
    if (!g_qwn_initialized) return;
    if (--g_qwn_refcount > 0) return;
    for (const ResidentWeight &entry : g_qwn_resident)
        if (entry.device) cudaFree(entry.device);
    g_qwn_resident.clear();
    qwn_cuda_layer_free(&g_qwn_ctx);
    memset(&g_qwn_ctx, 0, sizeof(g_qwn_ctx));
    g_qwn_initialized = 0;
    g_qwn_gpu = -1;
    g_qwn_use_dp4a = 0;
}

extern "C" void qwn_cuda_layer_free(QwnCUDALayerContext *ctx) {
    if (!ctx) return;
    if (ctx->dev_weights) cudaFree(ctx->dev_weights);
    if (ctx->dev_x) cudaFree(ctx->dev_x);
    if (ctx->dev_y) cudaFree(ctx->dev_y);
    if (ctx->pinned_x) qwn_cuda_host_free_pinned(ctx->pinned_x);
    if (ctx->pinned_y) qwn_cuda_host_free_pinned(ctx->pinned_y);
    if (ctx->stream_compute) cudaStreamDestroy((cudaStream_t)ctx->stream_compute);
    if (ctx->stream_prefetch) cudaStreamDestroy((cudaStream_t)ctx->stream_prefetch);
    memset(ctx, 0, sizeof(*ctx));
}

#else
/* Stub implementation for non-CUDA host compilation */
void *qwn_cuda_host_alloc_pinned(size_t bytes) { return malloc(bytes); }
void qwn_cuda_host_free_pinned(void *ptr) { free(ptr); }
int qwn_cuda_layer_init(QwnCUDALayerContext *ctx, int K, int N, int device_id) { (void)ctx; (void)K; (void)N; (void)device_id; return -1; }
int qwn_cuda_hypervsq2_gemv(QwnCUDALayerContext *ctx, const void *weights, const float *x, float *y, int K, int N) { (void)ctx; (void)weights; (void)x; (void)y; (void)K; (void)N; return -1; }
int qwn_cuda_init(int gpu_id) { (void)gpu_id; return -1; }
int qwn_cuda_gemv_hypervsq2(int rows, int cols, const void *weights, const float *x, float *out) { (void)rows; (void)cols; (void)weights; (void)x; (void)out; return -1; }
int qwn_cuda_gemv_q4_0(int rows, int cols, const void *weights, const float *x, float *out) { (void)rows; (void)cols; (void)weights; (void)x; (void)out; return -1; }
int qwn_cuda_get_metrics(QwnCudaMetrics *metrics) { (void)metrics; return -1; }
void qwn_cuda_shutdown(void) {}
void qwn_cuda_layer_free(QwnCUDALayerContext *ctx) { (void)ctx; }
#endif
