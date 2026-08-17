#include "../cuda/qwn_cuda_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static float half_to_float(std::uint16_t raw) {
    const std::uint32_t sign = (raw >> 15) & 1u;
    const std::uint32_t exponent = (raw >> 10) & 31u;
    const std::uint32_t mantissa = raw & 1023u;
    std::uint32_t bits;
    if (exponent == 0) bits = sign << 31;
    else if (exponent == 31) bits = (sign << 31) | 0x7f800000u | (mantissa << 13);
    else bits = (sign << 31) | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static float reference_row(const std::vector<std::uint8_t> &weights,
                           std::size_t row_offset, const std::vector<std::int8_t> &input,
                           float input_scale, std::uint32_t cols) {
    const std::uint32_t blocks = (cols + 255u) / 256u;
    float total = 0.0f;
    for (std::uint32_t block = 0; block < blocks; block++) {
        const std::uint32_t valid = std::min(256u, cols - block * 256u);
        const std::uint8_t *packed = weights.data() + row_offset + block * 74;
        const float base = half_to_float(static_cast<std::uint16_t>(packed[0] | (packed[1] << 8)));
        const float offset = half_to_float(static_cast<std::uint16_t>(packed[2] | (packed[3] << 8)));
        for (int octant = 0; octant < 8; octant++) {
            const std::uint32_t start = block * 256u + static_cast<std::uint32_t>(octant) * 32u;
            const std::uint32_t cap = start < block * 256u + valid
                                          ? std::min(32u, block * 256u + valid - start) : 0u;
            const std::uint8_t sub = packed[4 + octant / 2];
            const int sub_value = (octant & 1) ? (sub >> 4) : (sub & 15);
            int sum_q = 0, sum_a = 0;
            for (std::uint32_t i = 0; i < cap; i++) {
                const std::uint8_t q = packed[10 + octant * 8 + (i >> 2)];
                const int quantized = (q >> ((i & 3u) * 2u)) & 3;
                sum_q += (quantized - 1) * input[start + i];
                sum_a += input[start + i];
            }
            const float scale = base * (static_cast<float>(sub_value) / 8.0f);
            total += (static_cast<float>(sum_q) * scale +
                      static_cast<float>(sum_a) * offset) * input_scale;
        }
    }
    return total;
}

static void init_header(QwnCudaAbiHeader *header, std::size_t size) {
    std::memset(header, 0, size);
    qwn_cuda_abi_header_init(header, static_cast<std::uint32_t>(size));
}

int main() {
    QwnCudaAbiInfo info{};
    init_header(&info.header, sizeof(info));
    if (qwn_cuda_abi_query(&info) != QWN_CUDA_STATUS_OK ||
        info.hypervsq2_block_bytes != 74 || info.hypervsq2_block_elements != 256 ||
        !(info.capability_bits & QWN_CUDA_CAP_HYPERVSQ2_GEMV) ||
        !(info.capability_bits & QWN_CUDA_CAP_HYPERVSQ2_GEMM)) {
        std::fprintf(stderr, "ABI query did not advertise the exact HyperVSQ-2 contract.\n");
        return 1;
    }
    QwnCudaDeviceInfo devices[16]{};
    std::uint32_t device_count = 0;
    if (qwn_cuda_abi_enumerate_devices(devices, 16, &device_count) != QWN_CUDA_STATUS_OK ||
        device_count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA device is available for the ABI test.\n");
        return 77;
    }

    QwnCudaContextOptions options{};
    init_header(&options.header, sizeof(options));
    options.device_id = devices[0].device_id;
    QwnCudaContextHandle context{};
    init_header(&context.header, sizeof(context));
    if (qwn_cuda_abi_context_create(&options, &context) != QWN_CUDA_STATUS_OK) return 1;

    constexpr std::uint32_t rows = 3;
    constexpr std::uint32_t cols = 513;
    constexpr std::uint32_t batch = 2;
    const std::uint32_t blocks = (cols + 255u) / 256u;
    const std::size_t row_bytes = static_cast<std::size_t>(blocks) * 74;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(rows) * row_bytes, 0);
    for (std::uint32_t row = 0; row < rows; row++) {
        for (std::uint32_t block = 0; block < blocks; block++) {
            std::uint8_t *packed = weights.data() + row * row_bytes + block * 74;
            const std::uint16_t base = static_cast<std::uint16_t>(0x3800 + row * 0x200);
            const std::uint16_t offset = static_cast<std::uint16_t>(0x1000 + block * 0x200);
            std::memcpy(packed, &base, 2);
            std::memcpy(packed + 2, &offset, 2);
            for (int i = 0; i < 4; i++) packed[4 + i] = static_cast<std::uint8_t>(0x18 + i * 0x11);
            packed[8] = 0xff;
            packed[9] = 0x55;
            for (int i = 0; i < 64; i++) {
                const std::uint8_t pattern = static_cast<std::uint8_t>((i + row + block) & 3);
                packed[10 + i] = static_cast<std::uint8_t>(pattern * 0x55);
            }
        }
    }
    QwnCudaTensorUpload upload{};
    init_header(&upload.header, sizeof(upload));
    upload.host_data = weights.data();
    upload.data_bytes = weights.size();
    upload.dtype = QWN_CUDA_TENSOR_HYPERVSQ2_74;
    upload.rows = rows;
    upload.cols = cols;
    upload.block_bytes = 74;
    QwnCudaTensorHandle tensor{};
    init_header(&tensor.header, sizeof(tensor));
    if (qwn_cuda_abi_upload_tensor(&context, &upload, &tensor) != QWN_CUDA_STATUS_OK) return 1;

    std::vector<std::int8_t> input(batch * cols);
    for (std::size_t i = 0; i < input.size(); i++) input[i] = static_cast<std::int8_t>((i * 13) % 127 - 63);
    std::vector<float> output(batch * rows, 0.0f);
    QwnCudaGemmRequest request{};
    init_header(&request.header, sizeof(request));
    request.tensor = tensor;
    request.input_q8 = input.data();
    request.input_scale = 0.03125f;
    request.output = output.data();
    request.batch = batch;
    request.rows = rows;
    request.cols = cols;
    request.input_stride = cols;
    request.output_stride = rows;
    request.input_mode = QWN_CUDA_INPUT_Q8;
    QwnCudaTelemetry telemetry{};
    init_header(&telemetry.header, sizeof(telemetry));
    if (qwn_cuda_abi_hypervsq2_gemm(&context, &request, &telemetry) != QWN_CUDA_STATUS_OK ||
        telemetry.gpu_matmul_count == 0 || telemetry.gpu_kernel_launch_count == 0 ||
        telemetry.gpu_upload_count != 1 || telemetry.gpu_resident_bytes != weights.size()) return 1;
    for (std::uint32_t token = 0; token < batch; token++) {
        std::vector<std::int8_t> one(input.begin() + token * cols,
                                     input.begin() + (token + 1) * cols);
        for (std::uint32_t row = 0; row < rows; row++) {
            const float expected = reference_row(weights, row * row_bytes, one,
                                                 request.input_scale, cols);
            if (std::fabs(output[token * rows + row] - expected) > 1e-3f) {
                std::fprintf(stderr, "row %u token %u mismatch: got %.8f expected %.8f\n",
                             row, token, output[token * rows + row], expected);
                return 1;
            }
        }
    }
    if (qwn_cuda_abi_release_tensor(&context, &tensor) != QWN_CUDA_STATUS_OK ||
        qwn_cuda_abi_context_destroy(&context) != QWN_CUDA_STATUS_OK) return 1;
    std::printf("HyperVSQ-2 ABI v%u verified: kernel=%s matmuls=%llu device=%d.\n",
                QWN_CUDA_ABI_VERSION, telemetry.kernel_type,
                static_cast<unsigned long long>(telemetry.gpu_matmul_count),
                telemetry.device_id);
    return 0;
}
