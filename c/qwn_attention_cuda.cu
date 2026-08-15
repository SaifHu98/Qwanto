#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Qwanto CUDA Accelerated Kernels:
 * 1. Fused In-Register TurboQuant 3.5-bit Attention
 * 2. Vectorized TWLA / HyperVSQ-2 GEMV & MatMul
 * 3. Fast RMSNorm
 * ------------------------------------------------------------------------- */

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128

/* -------------------------------------------------------------------------
 * 1. Fused In-Register TurboQuant 3.5-Bit Attention Kernel
 * ------------------------------------------------------------------------- */
__device__ inline float dequantize_turboquant_val(uint8_t code, float scale, float min_val) {
    return (float)code * scale + min_val;
}

__global__ void qwn_attention_turboquant_cuda(
    const uint8_t* __restrict__ k_cache,    /* [seq_len, n_heads, head_dim_bytes] */
    const uint8_t* __restrict__ v_cache,    /* [seq_len, n_heads, head_dim_bytes] */
    const float* __restrict__ q,            /* [n_heads, head_dim] */
    float* __restrict__ output,             /* [n_heads, head_dim] */
    int n_heads,
    int head_dim,
    int seq_len,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (head_idx >= n_heads) return;

    extern __shared__ float s_mem[];
    float* s_q = s_mem;                              /* [head_dim] */
    float* s_scores = s_mem + head_dim;             /* [seq_len] */

    /* Load query into shared memory */
    if (tid < head_dim) {
        s_q[tid] = q[head_idx * head_dim + tid];
    }
    __syncthreads();

    /* Step 1: Compute Q * K^T scores across sequence */
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float score = 0.0f;
        const uint8_t* k_ptr = k_cache + (t * n_heads + head_idx) * (head_dim / 2);

        for (int d = 0; d < head_dim; d += 2) {
            uint8_t byte = k_ptr[d / 2];
            uint8_t code0 = byte & 0x0F;
            uint8_t code1 = (byte >> 4) & 0x0F;

            float k0 = (float)code0 * 0.125f - 1.0f;
            float k1 = (float)code1 * 0.125f - 1.0f;

            score += s_q[d] * k0 + s_q[d + 1] * k1;
        }
        s_scores[t] = score * sm_scale;
    }
    __syncthreads();

    /* Step 2: Softmax across sequence tokens */
    __shared__ float s_max;
    __shared__ float s_sum;

    if (tid == 0) {
        float max_val = -1e20f;
        for (int t = 0; t < seq_len; t++) {
            if (s_scores[t] > max_val) max_val = s_scores[t];
        }
        s_max = max_val;

        float sum_val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            s_scores[t] = expf(s_scores[t] - max_val);
            sum_val += s_scores[t];
        }
        s_sum = sum_val > 0.0f ? (1.0f / sum_val) : 0.0f;
    }
    __syncthreads();

    for (int t = tid; t < seq_len; t += blockDim.x) {
        s_scores[t] *= s_sum;
    }
    __syncthreads();

    /* Step 3: Accumulate Softmax * V */
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float weight = s_scores[t];
            if (weight < 1e-5f) continue;

            const uint8_t* v_ptr = v_cache + (t * n_heads + head_idx) * (head_dim / 2);
            uint8_t byte = v_ptr[tid / 2];
            uint8_t code = (tid % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            float v_val = (float)code * 0.125f - 1.0f;

            acc += weight * v_val;
        }
        output[head_idx * head_dim + tid] = acc;
    }
}

/* -------------------------------------------------------------------------
 * 2. Fast RMSNorm Kernel
 * ------------------------------------------------------------------------- */
__global__ void qwn_rmsnorm_cuda(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_rmsnorm[];

    const float* x = input + row * hidden_dim;
    float* out = output + row * hidden_dim;

    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = x[i];
        thread_sum += val * val;
    }
    s_rmsnorm[tid] = thread_sum;
    __syncthreads();

    /* Parallel block reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rmsnorm[tid] += s_rmsnorm[tid + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(s_rmsnorm[0] / (float)hidden_dim + eps);

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/* -------------------------------------------------------------------------
 * 3. TWLA 1.58-Bit GEMV CUDA Kernel
 * ------------------------------------------------------------------------- */
__global__ void qwn_twla_gemv_cuda(
    const uint8_t* __restrict__ packed_w,  /* [rows, cols / 4] */
    const float* __restrict__ scales_fp16, /* [rows, cols / 256] */
    const float* __restrict__ x,           /* [cols] */
    float* __restrict__ y,                 /* [rows] */
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    if (row >= rows) return;

    float acc = 0.0f;
    int blocks_per_row = cols / 256;

    for (int b = 0; b < blocks_per_row; b++) {
        float scale = scales_fp16[row * blocks_per_row + b];
        const uint8_t* block_w = packed_w + (row * blocks_per_row + b) * 64;

        for (int i = tid; i < 64; i += blockDim.x) {
            uint8_t byte = block_w[i];
            int col_base = b * 256 + i * 4;

            for (int k = 0; k < 4; k++) {
                int code = (byte >> (k * 2)) & 0x03;
                float w_val = (code == 1) ? scale : ((code == 2) ? -scale : 0.0f);
                acc += w_val * x[col_base + k];
            }
        }
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (tid == 0) {
        atomicAdd(&y[row], acc);
    }
}
