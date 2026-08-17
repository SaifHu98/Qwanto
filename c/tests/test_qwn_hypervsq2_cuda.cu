#include "../cuda/qwn_hypervsq_cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    constexpr int columns = 256;
    std::vector<std::uint8_t> weights(74, 0);
    std::vector<float> input(columns);
    float expected = 0.0f;

    const std::uint16_t one = 0x3c00;
    std::memcpy(weights.data(), &one, sizeof(one));
    for (int index = 0; index < 4; ++index) weights[4 + index] = 0x88;
    for (int index = 0; index < columns; ++index) {
        const int quantized = index % 4;
        weights[10 + (index / 4)] |= static_cast<std::uint8_t>(quantized << ((index % 4) * 2));
        input[index] = static_cast<float>((index % 17) - 8) * 0.125f;
        expected += static_cast<float>(quantized - 1) * input[index];
    }

    if (qwn_cuda_init(0) != 0) {
        std::fprintf(stderr, "CUDA device 0 could not be initialized.\n");
        return 2;
    }
    float actual = 0.0f;
    const int result = qwn_cuda_gemv_hypervsq2(1, columns, weights.data(), input.data(), &actual);
    QwnCudaMetrics metrics{};
    const int metrics_result = qwn_cuda_get_metrics(&metrics);
    qwn_cuda_shutdown();
    if (result != 0 || metrics_result != 0 || metrics.matmul_count < 1 ||
        std::strcmp(metrics.kernel, "hypervsq2-74") != 0) {
        std::fprintf(stderr, "HyperVSQ-2 CUDA dispatch did not report a completed 74-byte kernel.\n");
        return 1;
    }
    if (std::fabs(actual - expected) > 1e-4f) {
        std::fprintf(stderr, "CUDA/CPU reference mismatch: expected %.8f, got %.8f.\n", expected, actual);
        return 1;
    }
    std::printf("HyperVSQ-2 CUDA dispatch verified: kernel=%s matmuls=%llu.\n",
                metrics.kernel, static_cast<unsigned long long>(metrics.matmul_count));
    return 0;
}
