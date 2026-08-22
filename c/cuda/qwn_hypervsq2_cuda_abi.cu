#include "qwn_cuda_abi.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

struct ResidentTensor {
    void *device = nullptr;
    std::uint64_t bytes = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

struct ResidentKvCache {
    std::int8_t *key = nullptr;
    std::int8_t *value = nullptr;
    float *key_scales = nullptr;
    float *value_scales = nullptr;
    std::uint32_t max_tokens = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t channels = 0;
    std::uint32_t scale_blocks = 0;
    std::uint32_t tokens = 0;
    std::uint64_t bytes = 0;
};

struct RuntimeContext {
    int device = -1;
    std::uint64_t budget_bytes = 0;
    std::uint64_t resident_bytes = 0;
    cudaStream_t stream = nullptr;
    int8_t *device_input = nullptr;
    std::size_t input_capacity = 0;
    float *device_output = nullptr;
    std::size_t output_capacity = 0;
    float *device_scales = nullptr;
    std::size_t scale_capacity = 0;
    std::vector<ResidentTensor *> tensors;
    std::vector<ResidentKvCache *> kv_caches;
    float *device_kv_key = nullptr;
    float *device_kv_value = nullptr;
    float *device_kv_query = nullptr;
    float *device_kv_output = nullptr;
    std::size_t kv_float_capacity = 0;
    std::size_t kv_output_capacity = 0;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    QwnCudaTelemetry telemetry{};
};

std::mutex g_mutex;
std::string g_last_error;

void set_error(const char *message) {
    g_last_error = message ? message : "CUDA backend error";
}

void set_cuda_error(const char *operation, cudaError_t error) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "%s: %s", operation,
                  cudaGetErrorString(error));
    set_error(buffer);
}

bool header_ok(const QwnCudaAbiHeader &header, std::size_t expected) {
    return header.abi_version == QWN_CUDA_ABI_VERSION &&
           header.struct_size >= expected;
}

void init_telemetry(QwnCudaTelemetry *telemetry, int device) {
    if (!telemetry) return;
    std::memset(telemetry, 0, sizeof(*telemetry));
    qwn_cuda_abi_header_init(&telemetry->header,
                             static_cast<std::uint32_t>(sizeof(*telemetry)));
    telemetry->device_id = device;
    std::snprintf(telemetry->kernel_type, sizeof(telemetry->kernel_type),
                  "hypervsq2-74-q8-reference");
    std::snprintf(telemetry->kv_cache_kernel_type,
                  sizeof(telemetry->kv_cache_kernel_type), "Unavailable");
}

RuntimeContext *context_from(const QwnCudaContextHandle *handle) {
    if (!handle || !header_ok(handle->header, sizeof(*handle)) || !handle->opaque)
        return nullptr;
    return static_cast<RuntimeContext *>(handle->opaque);
}

ResidentTensor *tensor_from(const QwnCudaTensorHandle &handle) {
    if (!header_ok(handle.header, sizeof(handle)) || !handle.opaque)
        return nullptr;
    return static_cast<ResidentTensor *>(handle.opaque);
}

ResidentKvCache *kv_from(const QwnCudaKvCacheHandle &handle) {
    if (!header_ok(handle.header, sizeof(handle)) || !handle.opaque)
        return nullptr;
    return static_cast<ResidentKvCache *>(handle.opaque);
}

__device__ float half_to_float(const std::uint8_t *bytes) {
    const std::uint16_t raw = static_cast<std::uint16_t>(bytes[0]) |
                              (static_cast<std::uint16_t>(bytes[1]) << 8);
    const std::uint32_t sign = (raw >> 15) & 1u;
    const std::uint32_t exponent = (raw >> 10) & 31u;
    const std::uint32_t mantissa = raw & 1023u;
    std::uint32_t bits;
    if (exponent == 0) {
        bits = sign << 31;
    } else if (exponent == 31) {
        bits = (sign << 31) | 0x7f800000u | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return __int_as_float(static_cast<int>(bits));
}

__device__ __forceinline__ float warp_sum_float(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ std::uint32_t qwn_min_u32(std::uint32_t left,
                                                      std::uint32_t right) {
    return left < right ? left : right;
}

__global__ void qwn_q8_quantize_token(const float *input, std::int8_t *output,
                                      float *scales, std::uint32_t channels) {
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t start = block * 64u;
    if (start >= channels) return;
    const std::uint32_t valid = qwn_min_u32(64u, channels - start);
    __shared__ float max_abs[64];
    const std::uint32_t lane = threadIdx.x;
    float value = lane < valid ? input[start + lane] : 0.0f;
    max_abs[lane] = isfinite(value) ? fabsf(value) : 0.0f;
    __syncthreads();
    for (std::uint32_t stride = 32u; stride > 0; stride >>= 1) {
        if (lane < stride && max_abs[lane + stride] > max_abs[lane])
            max_abs[lane] = max_abs[lane + stride];
        __syncthreads();
    }
    const float scale = max_abs[0] > 0.0f ? max_abs[0] / 127.0f : 1.0f;
    if (lane == 0) scales[block] = scale;
    if (lane < valid) {
        float scaled = isfinite(value) ? value / scale : 0.0f;
        scaled = fminf(127.0f, fmaxf(-127.0f, scaled));
        output[start + lane] = static_cast<std::int8_t>(nearbyintf(scaled));
    }
}

__device__ float qwn_q8_value(const std::int8_t *values, const float *scales,
                              std::uint32_t token, std::uint32_t channel,
                              std::uint32_t channels) {
    (void)channels;
    return static_cast<float>(values[static_cast<std::uint64_t>(token) * channels + channel]) *
           scales[static_cast<std::uint64_t>(token) * ((channels + 63u) / 64u) + channel / 64u];
}

/* Correctness-first attention reader.  The block uses parallel lanes for the
 * dot product and performs the final value accumulation in lane zero.  It is
 * intentionally separate from the HyperVSQ weight GEMV kernel. */
__global__ void qwn_q8_attention(const std::int8_t *keys, const std::int8_t *values,
                                 const float *key_scales, const float *value_scales,
                                 const float *query, float *output,
                                 std::uint32_t query_heads, std::uint32_t kv_heads,
                                 std::uint32_t head_dim, std::uint32_t channels,
                                 std::uint32_t position, float scale) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (head >= query_heads || lane >= 128u) return;
    const std::uint32_t kv_head = (head * kv_heads) / query_heads;
    const std::uint32_t offset = kv_head * head_dim;
    extern __shared__ float shared[];
    float *scores = shared;
    float *reduce = shared + position + 1u;
    const float *head_query = query + head * head_dim;
    for (std::uint32_t token = 0; token <= position; token++) {
        float dot = 0.0f;
        for (std::uint32_t channel = lane; channel < head_dim; channel += 128u)
            dot += head_query[channel] * qwn_q8_value(
                keys, key_scales, token, offset + channel, channels);
        reduce[lane] = dot;
        __syncthreads();
        for (std::uint32_t stride = 64u; stride > 0; stride >>= 1) {
            if (lane < stride) reduce[lane] += reduce[lane + stride];
            __syncthreads();
        }
        if (lane == 0) scores[token] = reduce[0] * scale;
        __syncthreads();
    }
    if (lane == 0) {
        float max_score = scores[0];
        for (std::uint32_t token = 1; token <= position; token++)
            max_score = fmaxf(max_score, scores[token]);
        float sum = 0.0f;
        for (std::uint32_t token = 0; token <= position; token++) {
            scores[token] = expf(scores[token] - max_score);
            sum += scores[token];
        }
        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        float *head_output = output + head * head_dim;
        for (std::uint32_t channel = 0; channel < head_dim; channel++) {
            float result = 0.0f;
            for (std::uint32_t token = 0; token <= position; token++)
                result += scores[token] * qwn_q8_value(
                    values, value_scales, token, offset + channel, channels);
            head_output[channel] = result * inverse;
        }
    }
}

/*
 * Exact QWN 2.31 / HyperVSQ-2 reference GEMV. The input is the decoder's
 * symmetric int8 activation tensor, not an unrelated FP32 approximation.
 * A block is: fp16 d_base (2), fp16 m_base (2), four packed sub-scales (4),
 * two reserved/sparsity bytes (2), and 64 packed 2-bit values (64).
 */
__global__ void hypervsq2_gemv_q8(const std::uint8_t *weights,
                                   const std::int8_t *input,
                                   float input_scale,
                                   const float *input_scales,
                                   float *output,
                                   std::uint32_t batch,
                                   std::uint32_t rows,
                                   std::uint32_t cols,
                                   std::uint32_t input_stride,
                                   std::uint32_t output_stride,
                                   std::uint64_t row_bytes) {
    const std::uint32_t row = blockIdx.x * 4u + threadIdx.y;
    const std::uint32_t token = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || token >= batch) return;

    const std::uint8_t *row_ptr = weights +
        static_cast<std::uint64_t>(row) * row_bytes;
    const std::int8_t *input_ptr = input +
        static_cast<std::uint64_t>(token) * input_stride;
    float lane_acc = 0.0f;
    const std::uint32_t blocks = (cols + 255u) / 256u;

    for (std::uint32_t block = 0; block < blocks; block++) {
        const std::uint32_t valid = qwn_min_u32(256u, cols - block * 256u);
        const std::uint8_t *packed_block = row_ptr + block * QWN_CUDA_HYPERVSQ2_BLOCK_BYTES;
        const float base = half_to_float(packed_block);
        const float offset = half_to_float(packed_block + 2);
        const std::uint8_t *sub_scales = packed_block + 4;
        const std::uint8_t *packed_values = packed_block + 10;

        #pragma unroll
        for (int octant = 0; octant < 8; octant++) {
            const std::uint32_t start = block * 256u +
                                        static_cast<std::uint32_t>(octant) * 32u;
            const std::uint32_t cap = start < block * 256u + valid
                                          ? qwn_min_u32(32u, block * 256u + valid - start)
                                          : 0u;
            if (static_cast<std::uint32_t>(lane) < cap) {
                const std::uint8_t sub_byte = sub_scales[octant >> 1];
                const int sub_value = (octant & 1) ? (sub_byte >> 4) : (sub_byte & 15);
                const float eff_scale = base * (static_cast<float>(sub_value) * (1.0f / 8.0f));

                const std::uint8_t *octant_values = packed_values + octant * 8;
                const std::uint8_t packed = octant_values[lane >> 2];
                const int quantized = (packed >> ((lane & 3) * 2)) & 3;
                const int activation = static_cast<int>(input_ptr[start + lane]);
                
                lane_acc += static_cast<float>((quantized - 1) * activation) * eff_scale +
                            static_cast<float>(activation) * offset;
            }
        }
    }
    const float total = warp_sum_float(lane_acc);
    if (lane == 0) {
        const float scale = input_scales ? input_scales[token] : input_scale;
        output[static_cast<std::uint64_t>(token) * output_stride + row] = total * scale;
    }
}

int ensure_workspace(RuntimeContext *context, std::uint32_t batch,
                     std::uint32_t rows, std::uint32_t cols) {
    const std::size_t input_bytes = static_cast<std::size_t>(batch) * cols;
    const std::size_t output_bytes = static_cast<std::size_t>(batch) * rows * sizeof(float);
    if (input_bytes > context->input_capacity) {
        if (context->device_input) cudaFree(context->device_input);
        if (cudaMalloc(reinterpret_cast<void **>(&context->device_input), input_bytes) != cudaSuccess) {
            context->device_input = nullptr;
            context->input_capacity = 0;
            return QWN_CUDA_STATUS_OUT_OF_MEMORY;
        }
        context->input_capacity = input_bytes;
    }
    if (output_bytes > context->output_capacity) {
        if (context->device_output) cudaFree(context->device_output);
        if (cudaMalloc(reinterpret_cast<void **>(&context->device_output), output_bytes) != cudaSuccess) {
            context->device_output = nullptr;
            context->output_capacity = 0;
            return QWN_CUDA_STATUS_OUT_OF_MEMORY;
        }
        context->output_capacity = output_bytes;
    }
    if (batch * sizeof(float) > context->scale_capacity) {
        if (context->device_scales) cudaFree(context->device_scales);
        if (cudaMalloc(reinterpret_cast<void **>(&context->device_scales),
                       static_cast<std::size_t>(batch) * sizeof(float)) != cudaSuccess) {
            context->device_scales = nullptr;
            context->scale_capacity = 0;
            return QWN_CUDA_STATUS_OUT_OF_MEMORY;
        }
        context->scale_capacity = static_cast<std::size_t>(batch) * sizeof(float);
    }
    return QWN_CUDA_STATUS_OK;
}

int execute(RuntimeContext *context, const QwnCudaGemmRequest *request,
            QwnCudaTelemetry *telemetry) {
    if (!context || !request || !header_ok(request->header, sizeof(*request)) ||
        request->input_mode != QWN_CUDA_INPUT_Q8 || !request->input_q8 ||
        !request->output || request->batch == 0 || request->rows == 0 ||
        request->cols == 0) {
        set_error("invalid HyperVSQ-2 GEMV/GEMM request");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    ResidentTensor *tensor = tensor_from(request->tensor);
    if (!tensor || tensor->rows != request->rows || tensor->cols != request->cols) {
        set_error("tensor dimensions do not match the HyperVSQ-2 request");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(context->device) != cudaSuccess) {
        set_cuda_error("cudaSetDevice", cudaGetLastError());
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    const int status = ensure_workspace(context, request->batch, request->rows,
                                        request->cols);
    if (status != QWN_CUDA_STATUS_OK) {
        set_error("CUDA activation workspace allocation failed");
        return status;
    }
    const std::uint32_t input_stride = request->input_stride ? request->input_stride : request->cols;
    const std::uint32_t output_stride = request->output_stride ? request->output_stride : request->rows;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>((request->cols + 255u) / 256u) * QWN_CUDA_HYPERVSQ2_BLOCK_BYTES;
    const std::size_t input_bytes = static_cast<std::size_t>(request->batch) * input_stride;
    const std::size_t output_bytes = static_cast<std::size_t>(request->batch) * output_stride * sizeof(float);
    const auto transfer_start = std::chrono::steady_clock::now();
    if (cudaMemcpyAsync(context->device_input, request->input_q8, input_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        set_cuda_error("activation upload", cudaGetLastError());
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    if (request->input_scales &&
        cudaMemcpyAsync(context->device_scales, request->input_scales,
                        static_cast<std::size_t>(request->batch) * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        set_cuda_error("activation scale upload", cudaGetLastError());
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    const auto transfer_end = std::chrono::steady_clock::now();
    cudaError_t err = cudaSuccess;
    if (context->start_event && (err = cudaEventRecord(context->start_event, context->stream)) != cudaSuccess) {
        set_cuda_error("kernel start event", err);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    dim3 block(32, 4, 1);
    dim3 grid((request->rows + 3) / 4, request->batch, 1);
    hypervsq2_gemv_q8<<<grid, block, 0, context->stream>>>(
        static_cast<const std::uint8_t *>(tensor->device), context->device_input,
        request->input_scale, request->input_scales ? context->device_scales : nullptr,
        context->device_output, request->batch,
        request->rows, request->cols, input_stride, output_stride, row_bytes);
    if ((err = cudaGetLastError()) != cudaSuccess) {
        set_cuda_error("HyperVSQ-2 kernel launch", err);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    if (context->end_event && (err = cudaEventRecord(context->end_event, context->stream)) != cudaSuccess) {
        set_cuda_error("kernel end event", err);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    if ((err = cudaMemcpyAsync(request->output, context->device_output, output_bytes,
                               cudaMemcpyDeviceToHost, context->stream)) != cudaSuccess ||
        (err = cudaStreamSynchronize(context->stream)) != cudaSuccess) {
        set_cuda_error("result download", err);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    float kernel_ms = 0.0f;
    if (context->start_event && context->end_event) {
        if (cudaEventElapsedTime(&kernel_ms, context->start_event, context->end_event) != cudaSuccess) {
            kernel_ms = 0.0f;
        }
    }

    context->telemetry.gpu_matmul_count++;
    context->telemetry.gpu_kernel_launch_count++;
    context->telemetry.gpu_projection_count++;
    context->telemetry.gpu_kernel_ms += kernel_ms;
    context->telemetry.gpu_transfer_ms +=
        std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    context->telemetry.gpu_resident_bytes = context->resident_bytes;
    context->telemetry.device_id = context->device;
    if (telemetry) *telemetry = context->telemetry;
    return QWN_CUDA_STATUS_OK;
}

} // namespace

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_query(QwnCudaAbiInfo *info) {
    if (!info || !header_ok(info->header, sizeof(*info))) {
        set_error("invalid ABI query structure");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    std::memset(info, 0, sizeof(*info));
    qwn_cuda_abi_header_init(&info->header, static_cast<std::uint32_t>(sizeof(*info)));
    info->capability_bits = QWN_CUDA_CAP_HYPERVSQ2_GEMV |
                            QWN_CUDA_CAP_HYPERVSQ2_GEMM |
                            QWN_CUDA_CAP_RESIDENT_WEIGHTS |
                            QWN_CUDA_CAP_TELEMETRY |
                            QWN_CUDA_CAP_DEVICE_ENUMERATION |
                            QWN_CUDA_CAP_Q8_KV;
    info->max_devices = 16;
    info->max_resident_tensors = QWN_CUDA_MAX_RESIDENT_TENSORS;
    info->hypervsq2_block_bytes = QWN_CUDA_HYPERVSQ2_BLOCK_BYTES;
    info->hypervsq2_block_elements = QWN_CUDA_HYPERVSQ2_BLOCK_ELEMENTS;
    std::snprintf(info->abi_name, sizeof(info->abi_name), "%s", QWN_CUDA_ABI_NAME);
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_get_capabilities(QwnCudaAbiInfo *info) {
    return qwn_cuda_abi_query(info);
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_enumerate_devices(
    QwnCudaDeviceInfo *devices, std::uint32_t capacity, std::uint32_t *count) {
    if (!count) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        set_cuda_error("cudaGetDeviceCount", error);
        *count = 0;
        return QWN_CUDA_STATUS_UNAVAILABLE;
    }
    *count = static_cast<std::uint32_t>(device_count);
    if (!devices || capacity == 0) return QWN_CUDA_STATUS_OK;
    const std::uint32_t limit = std::min<std::uint32_t>(capacity, *count);
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    for (std::uint32_t i = 0; i < limit; i++) {
        std::memset(&devices[i], 0, sizeof(devices[i]));
        qwn_cuda_abi_header_init(&devices[i].header, static_cast<std::uint32_t>(sizeof(devices[i])));
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, static_cast<int>(i)) != cudaSuccess)
            continue;
        devices[i].device_id = static_cast<int32_t>(i);
        devices[i].compute_major = properties.major;
        devices[i].compute_minor = properties.minor;
        devices[i].total_vram_bytes = properties.totalGlobalMem;
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaSetDevice(static_cast<int>(i)) == cudaSuccess &&
            cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
            devices[i].free_vram_bytes = free_bytes;
        std::snprintf(devices[i].name, sizeof(devices[i].name), "%s", properties.name);
        std::snprintf(devices[i].driver, sizeof(devices[i].driver), "%d", driver_version);
    }
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_context_create(
    const QwnCudaContextOptions *options, QwnCudaContextHandle *handle) {
    if (!options || !handle || !header_ok(options->header, sizeof(*options)) ||
        !header_ok(handle->header, sizeof(*handle)) || options->device_id < 0) {
        set_error("invalid CUDA context options");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *context = new (std::nothrow) RuntimeContext();
    if (!context) return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    context->device = options->device_id;
    context->budget_bytes = options->memory_budget_bytes;
    if (cudaSetDevice(context->device) != cudaSuccess ||
        cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete context;
        set_cuda_error("CUDA context creation", cudaGetLastError());
        return QWN_CUDA_STATUS_UNAVAILABLE;
    }
    cudaEventCreate(&context->start_event);
    cudaEventCreate(&context->end_event);
    init_telemetry(&context->telemetry, context->device);
    std::memset(handle, 0, sizeof(*handle));
    qwn_cuda_abi_header_init(&handle->header, static_cast<std::uint32_t>(sizeof(*handle)));
    handle->opaque = context;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_context_destroy(QwnCudaContextHandle *handle) {
    RuntimeContext *context = context_from(handle);
    if (!context) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaSetDevice(context->device);
    for (ResidentTensor *tensor : context->tensors) {
        if (tensor && tensor->device) cudaFree(tensor->device);
        delete tensor;
    }
    context->tensors.clear();
    for (ResidentKvCache *cache : context->kv_caches) {
        if (cache) {
            if (cache->key) cudaFree(cache->key);
            if (cache->value) cudaFree(cache->value);
            if (cache->key_scales) cudaFree(cache->key_scales);
            if (cache->value_scales) cudaFree(cache->value_scales);
            delete cache;
        }
    }
    context->kv_caches.clear();
    if (context->device_input) cudaFree(context->device_input);
    if (context->device_output) cudaFree(context->device_output);
    if (context->device_scales) cudaFree(context->device_scales);
    if (context->device_kv_key) cudaFree(context->device_kv_key);
    if (context->device_kv_value) cudaFree(context->device_kv_value);
    if (context->start_event) cudaEventDestroy(context->start_event);
    if (context->end_event) cudaEventDestroy(context->end_event);
    if (context->stream) cudaStreamDestroy(context->stream);
    delete context;
    handle->opaque = nullptr;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_upload_tensor(
    QwnCudaContextHandle *handle, const QwnCudaTensorUpload *upload,
    QwnCudaTensorHandle *handle_out) {
    RuntimeContext *context = context_from(handle);
    if (!context || !upload || !handle_out || !header_ok(upload->header, sizeof(*upload)) ||
        !header_ok(handle_out->header, sizeof(*handle_out)) || !upload->host_data ||
        upload->dtype != QWN_CUDA_TENSOR_HYPERVSQ2_74 || upload->rows == 0 ||
        upload->cols == 0 || upload->block_bytes != QWN_CUDA_HYPERVSQ2_BLOCK_BYTES) {
        set_error("invalid HyperVSQ-2 tensor upload request");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    const std::uint64_t required = static_cast<std::uint64_t>(upload->rows) *
        ((upload->cols + 255u) / 256u) * upload->block_bytes;
    if (upload->data_bytes < required) {
        set_error("HyperVSQ-2 tensor upload is shorter than its validated shape");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaSetDevice(context->device);
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        (context->budget_bytes && context->resident_bytes + required > context->budget_bytes) ||
        required > free_bytes) {
        set_error("HyperVSQ-2 tensor does not fit the configured CUDA memory budget");
        return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    }
    auto *tensor = new (std::nothrow) ResidentTensor();
    if (!tensor) return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    tensor->bytes = required;
    tensor->rows = upload->rows;
    tensor->cols = upload->cols;
    if (cudaMalloc(&tensor->device, static_cast<std::size_t>(required)) != cudaSuccess ||
        cudaMemcpy(tensor->device, upload->host_data, static_cast<std::size_t>(required),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        if (tensor->device) cudaFree(tensor->device);
        delete tensor;
        set_cuda_error("HyperVSQ-2 tensor upload", cudaGetLastError());
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    context->tensors.push_back(tensor);
    context->resident_bytes += required;
    context->telemetry.gpu_upload_count++;
    context->telemetry.gpu_upload_bytes += required;
    context->telemetry.gpu_resident_bytes = context->resident_bytes;
    std::memset(handle_out, 0, sizeof(*handle_out));
    qwn_cuda_abi_header_init(&handle_out->header, static_cast<std::uint32_t>(sizeof(*handle_out)));
    handle_out->opaque = tensor;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_release_tensor(
    QwnCudaContextHandle *handle, QwnCudaTensorHandle *handle_out) {
    RuntimeContext *context = context_from(handle);
    ResidentTensor *tensor = handle_out ? tensor_from(*handle_out) : nullptr;
    if (!context || !tensor) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaSetDevice(context->device);
    auto it = std::find(context->tensors.begin(), context->tensors.end(), tensor);
    if (it == context->tensors.end()) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    if (tensor->device) cudaFree(tensor->device);
    context->resident_bytes -= std::min(context->resident_bytes, tensor->bytes);
    delete tensor;
    context->tensors.erase(it);
    handle_out->opaque = nullptr;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_kv_cache_create(
    QwnCudaContextHandle *handle, const QwnCudaKvCacheOptions *options,
    QwnCudaKvCacheHandle *handle_out) {
    RuntimeContext *context = context_from(handle);
    if (!context || !options || !handle_out ||
        !header_ok(options->header, sizeof(*options)) ||
        !header_ok(handle_out->header, sizeof(*handle_out)) ||
        options->max_tokens == 0 || options->kv_heads == 0 || options->head_dim == 0) {
        set_error("invalid Q8 KV-cache options");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaSetDevice(context->device);
    auto *cache = new (std::nothrow) ResidentKvCache();
    if (!cache) return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    cache->max_tokens = options->max_tokens;
    cache->kv_heads = options->kv_heads;
    cache->head_dim = options->head_dim;
    cache->channels = options->kv_heads * options->head_dim;
    cache->scale_blocks = (cache->channels + 63u) / 64u;
    cache->bytes = static_cast<std::uint64_t>(cache->max_tokens) * cache->channels * 2u +
                   static_cast<std::uint64_t>(cache->max_tokens) * cache->scale_blocks *
                   sizeof(float) * 2u;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        (context->budget_bytes && context->resident_bytes + cache->bytes > context->budget_bytes) ||
        cache->bytes > free_bytes) {
        delete cache;
        set_error("Q8 KV-cache does not fit the configured CUDA memory budget");
        return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    }
    const std::size_t value_bytes = static_cast<std::size_t>(cache->max_tokens) * cache->channels;
    const std::size_t scale_bytes = static_cast<std::size_t>(cache->max_tokens) *
                                    cache->scale_blocks * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void **>(&cache->key), value_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&cache->value), value_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&cache->key_scales), scale_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&cache->value_scales), scale_bytes) != cudaSuccess) {
        if (cache->key) cudaFree(cache->key);
        if (cache->value) cudaFree(cache->value);
        if (cache->key_scales) cudaFree(cache->key_scales);
        if (cache->value_scales) cudaFree(cache->value_scales);
        delete cache;
        set_cuda_error("Q8 KV-cache allocation", cudaGetLastError());
        return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    }
    context->kv_caches.push_back(cache);
    context->resident_bytes += cache->bytes;
    context->telemetry.kv_cache_resident_bytes += cache->bytes;
    std::memset(handle_out, 0, sizeof(*handle_out));
    qwn_cuda_abi_header_init(&handle_out->header, static_cast<std::uint32_t>(sizeof(*handle_out)));
    handle_out->opaque = cache;
    return QWN_CUDA_STATUS_OK;
}

static int ensure_kv_workspace(RuntimeContext *context, std::size_t channels,
                               std::size_t output_values) {
    const std::size_t float_capacity = std::max(channels, output_values);
    if (float_capacity > context->kv_float_capacity) {
        if (context->device_kv_key) cudaFree(context->device_kv_key);
        if (context->device_kv_value) cudaFree(context->device_kv_value);
        if (context->device_kv_query) cudaFree(context->device_kv_query);
        const std::size_t bytes = float_capacity * sizeof(float);
        if (cudaMalloc(reinterpret_cast<void **>(&context->device_kv_key), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&context->device_kv_value), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&context->device_kv_query), bytes) != cudaSuccess) {
            if (context->device_kv_key) cudaFree(context->device_kv_key);
            if (context->device_kv_value) cudaFree(context->device_kv_value);
            if (context->device_kv_query) cudaFree(context->device_kv_query);
            context->device_kv_key = context->device_kv_value = context->device_kv_query = nullptr;
            context->kv_float_capacity = 0;
            return QWN_CUDA_STATUS_OUT_OF_MEMORY;
        }
        context->kv_float_capacity = float_capacity;
    }
    if (output_values > context->kv_output_capacity) {
        if (context->device_kv_output) cudaFree(context->device_kv_output);
        if (cudaMalloc(reinterpret_cast<void **>(&context->device_kv_output),
                       output_values * sizeof(float)) != cudaSuccess) {
            context->device_kv_output = nullptr;
            context->kv_output_capacity = 0;
            return QWN_CUDA_STATUS_OUT_OF_MEMORY;
        }
        context->kv_output_capacity = output_values;
    }
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_kv_cache_append(
    QwnCudaContextHandle *handle, const QwnCudaKvAppendRequest *request,
    QwnCudaTelemetry *telemetry) {
    RuntimeContext *context = context_from(handle);
    ResidentKvCache *cache = request ? kv_from(request->cache) : nullptr;
    if (!context || !request || !cache ||
        !header_ok(request->header, sizeof(*request)) || !request->host_key ||
        !request->host_value || request->n_channels != cache->channels ||
        request->token != cache->tokens || request->token >= cache->max_tokens) {
        set_error("invalid Q8 KV-cache append request");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaSetDevice(context->device);
    if (ensure_kv_workspace(context, cache->channels, 0) != QWN_CUDA_STATUS_OK)
        return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    const auto start = std::chrono::steady_clock::now();
    if (cudaMemcpyAsync(context->device_kv_key, request->host_key,
                        cache->channels * sizeof(float), cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_kv_value, request->host_value,
                        cache->channels * sizeof(float), cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess) {
        set_cuda_error("Q8 KV upload", cudaGetLastError());
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    const dim3 grid(cache->scale_blocks, 1, 1);
    qwn_q8_quantize_token<<<grid, 64, 0, context->stream>>>(
        context->device_kv_key,
        cache->key + static_cast<std::size_t>(cache->tokens) * cache->channels,
        cache->key_scales + static_cast<std::size_t>(cache->tokens) * cache->scale_blocks,
        cache->channels);
    qwn_q8_quantize_token<<<grid, 64, 0, context->stream>>>(
        context->device_kv_value,
        cache->value + static_cast<std::size_t>(cache->tokens) * cache->channels,
        cache->value_scales + static_cast<std::size_t>(cache->tokens) * cache->scale_blocks,
        cache->channels);
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaStreamSynchronize(context->stream);
    if (launch_status != cudaSuccess || sync_status != cudaSuccess) {
        set_cuda_error("Q8 KV quantize",
                       launch_status != cudaSuccess ? launch_status : sync_status);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    cache->tokens++;
    context->telemetry.kv_cache_kernel_count += 2;
    context->telemetry.kv_cache_upload_bytes += static_cast<std::uint64_t>(cache->channels) *
                                                sizeof(float) * 2u;
    context->telemetry.kv_cache_kernel_ms += 0.0;
    context->telemetry.kv_cache_transfer_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::snprintf(context->telemetry.kv_cache_kernel_type,
                  sizeof(context->telemetry.kv_cache_kernel_type), "q8-cuda-reference");
    if (telemetry) *telemetry = context->telemetry;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_kv_cache_attention(
    QwnCudaContextHandle *handle, const QwnCudaKvAttentionRequest *request,
    QwnCudaTelemetry *telemetry) {
    RuntimeContext *context = context_from(handle);
    ResidentKvCache *cache = request ? kv_from(request->cache) : nullptr;
    if (!context || !request || !cache || !header_ok(request->header, sizeof(*request)) ||
        !request->host_query || !request->host_output || request->query_heads == 0 ||
        request->kv_heads != cache->kv_heads || request->head_dim != cache->head_dim ||
        request->position >= cache->tokens) {
        set_error("invalid Q8 KV-cache attention request");
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaSetDevice(context->device);
    const std::size_t channels = static_cast<std::size_t>(cache->channels);
    const std::size_t output_values = static_cast<std::size_t>(request->query_heads) * cache->head_dim;
    if (ensure_kv_workspace(context, channels, output_values) != QWN_CUDA_STATUS_OK)
        return QWN_CUDA_STATUS_OUT_OF_MEMORY;
    const auto start = std::chrono::steady_clock::now();
    if (cudaMemcpyAsync(context->device_kv_query, request->host_query,
                        output_values * sizeof(float), cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess) {
        set_cuda_error("Q8 KV query upload", cudaGetLastError());
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    const std::size_t shared_bytes = (static_cast<std::size_t>(request->position) + 1u + 128u) * sizeof(float);
    qwn_q8_attention<<<request->query_heads, 128, shared_bytes, context->stream>>>(
        cache->key, cache->value, cache->key_scales, cache->value_scales,
        context->device_kv_query, context->device_kv_output,
        request->query_heads, request->kv_heads, request->head_dim,
        cache->channels, request->position, request->scale);
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t copy_status = cudaMemcpyAsync(
        request->host_output, context->device_kv_output,
        output_values * sizeof(float), cudaMemcpyDeviceToHost, context->stream);
    const cudaError_t sync_status = cudaStreamSynchronize(context->stream);
    if (launch_status != cudaSuccess || copy_status != cudaSuccess ||
        sync_status != cudaSuccess) {
        const cudaError_t error = launch_status != cudaSuccess
                                      ? launch_status
                                      : (copy_status != cudaSuccess ? copy_status
                                                                    : sync_status);
        set_cuda_error("Q8 KV attention", error);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    context->telemetry.kv_cache_kernel_count++;
    context->telemetry.kv_cache_transfer_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::snprintf(context->telemetry.kv_cache_kernel_type,
                  sizeof(context->telemetry.kv_cache_kernel_type), "q8-cuda-reference");
    if (telemetry) *telemetry = context->telemetry;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_kv_cache_reset(
    QwnCudaContextHandle *handle, QwnCudaKvCacheHandle *handle_cache) {
    RuntimeContext *context = context_from(handle);
    ResidentKvCache *cache = handle_cache ? kv_from(*handle_cache) : nullptr;
    if (!context || !cache) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    cache->tokens = 0;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_kv_cache_destroy(
    QwnCudaContextHandle *handle, QwnCudaKvCacheHandle *handle_cache) {
    RuntimeContext *context = context_from(handle);
    ResidentKvCache *cache = handle_cache ? kv_from(*handle_cache) : nullptr;
    if (!context || !cache) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = std::find(context->kv_caches.begin(), context->kv_caches.end(), cache);
    if (it == context->kv_caches.end()) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    cudaSetDevice(context->device);
    const std::uint64_t cache_bytes = cache->bytes;
    if (cache->key) cudaFree(cache->key);
    if (cache->value) cudaFree(cache->value);
    if (cache->key_scales) cudaFree(cache->key_scales);
    if (cache->value_scales) cudaFree(cache->value_scales);
    context->resident_bytes -= std::min(context->resident_bytes, cache_bytes);
    delete cache;
    context->kv_caches.erase(it);
    handle_cache->opaque = nullptr;
    context->telemetry.kv_cache_resident_bytes -=
        std::min(context->telemetry.kv_cache_resident_bytes, cache_bytes);
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_hypervsq2_gemv(
    QwnCudaContextHandle *context, const QwnCudaGemmRequest *request,
    QwnCudaTelemetry *telemetry) {
    RuntimeContext *runtime = context_from(context);
    if (!runtime || !request || request->batch != 1)
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_mutex);
    return execute(runtime, request, telemetry);
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_hypervsq2_gemm(
    QwnCudaContextHandle *context, const QwnCudaGemmRequest *request,
    QwnCudaTelemetry *telemetry) {
    RuntimeContext *runtime = context_from(context);
    if (!runtime) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_mutex);
    return execute(runtime, request, telemetry);
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_synchronize(QwnCudaContextHandle *handle) {
    RuntimeContext *context = context_from(handle);
    if (!context) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    const cudaError_t status = cudaStreamSynchronize(context->stream);
    if (status != cudaSuccess) {
        set_cuda_error("CUDA stream synchronize", status);
        return QWN_CUDA_STATUS_RUNTIME_ERROR;
    }
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_get_telemetry(
    QwnCudaContextHandle *handle, QwnCudaTelemetry *telemetry) {
    RuntimeContext *context = context_from(handle);
    if (!context || !telemetry || !header_ok(telemetry->header, sizeof(*telemetry)))
        return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    *telemetry = context->telemetry;
    return QWN_CUDA_STATUS_OK;
}

extern "C" QWN_CUDA_ABI_API int qwn_cuda_abi_last_error(char *buffer,
                                                          std::uint32_t size) {
    if (!buffer || size == 0) return QWN_CUDA_STATUS_INVALID_ARGUMENT;
    std::snprintf(buffer, size, "%s", g_last_error.empty() ? "" : g_last_error.c_str());
    return QWN_CUDA_STATUS_OK;
}
