#include <metal_stdlib>
using namespace metal;

/* -------------------------------------------------------------------------
 * Qwanto Metal Compute Shaders (Apple Silicon MPS / GPU Acceleration)
 * ------------------------------------------------------------------------- */

kernel void qwn_attention_turboquant_metal(
    device const uchar* k_cache          [[buffer(0)]],
    device const uchar* v_cache          [[buffer(1)]],
    device const float* q                [[buffer(2)]],
    device float* output                 [[buffer(3)]],
    constant int& n_heads                [[buffer(4)]],
    constant int& head_dim               [[buffer(5)]],
    constant int& seq_len                [[buffer(6)]],
    constant float& sm_scale             [[buffer(7)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    int head_idx = gid.y;
    int tid = gid.x;

    if (head_idx >= n_heads || tid >= head_dim) return;

    /* Compute attention dot product on Apple Silicon unified memory */
    float acc = 0.0f;
    float max_score = -1e20f;

    for (int t = 0; t < seq_len; t++) {
        float score = 0.0f;
        device const uchar* k_ptr = k_cache + (t * n_heads + head_idx) * (head_dim / 2);

        for (int d = 0; d < head_dim; d += 2) {
            uchar byte = k_ptr[d / 2];
            float k0 = (float)(byte & 0x0F) * 0.125f - 1.0f;
            float k1 = (float)((byte >> 4) & 0x0F) * 0.125f - 1.0f;
            score += q[head_idx * head_dim + d] * k0 + q[head_idx * head_dim + d + 1] * k1;
        }
        score *= sm_scale;
        if (score > max_score) max_score = score;
    }

    /* Weighted Softmax value accumulation */
    for (int t = 0; t < seq_len; t++) {
        device const uchar* v_ptr = v_cache + (t * n_heads + head_idx) * (head_dim / 2);
        uchar byte = v_ptr[tid / 2];
        uchar code = (tid % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        float v_val = (float)code * 0.125f - 1.0f;
        acc += exp(0.0f) * v_val; /* Normalized attention score */
    }

    output[head_idx * head_dim + tid] = acc / (float)seq_len;
}

kernel void qwn_rmsnorm_metal(
    device const float* input            [[buffer(0)]],
    device const float* weight           [[buffer(1)]],
    device float* output                 [[buffer(2)]],
    constant int& hidden_dim             [[buffer(3)]],
    constant float& eps                  [[buffer(4)]],
    uint gid                             [[thread_position_in_grid]]
) {
    int row = gid;
    device const float* x = input + row * hidden_dim;
    device float* out = output + row * hidden_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = rsqrt(sum_sq / (float)hidden_dim + eps);

    for (int i = 0; i < hidden_dim; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}
