#include "../cuda/qwn_cuda_abi.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int fail(const char *stage) {
    char error[256]{};
    qwn_cuda_abi_last_error(error, sizeof(error));
    std::fprintf(stderr, "CUDA Q8 KV failure at %s: %s\n", stage, error);
    return 1;
}

static float q8_scale(const float *values, std::uint32_t count) {
    float maximum = 0.0f;
    for (std::uint32_t i = 0; i < count; ++i)
        if (std::isfinite(values[i])) maximum = std::max(maximum, std::fabs(values[i]));
    return maximum > 0.0f ? maximum / 127.0f : 1.0f;
}

static void q8_quantize(const float *values, std::int8_t *output, float *scales,
                        std::uint32_t count) {
    const std::uint32_t blocks = (count + 63u) / 64u;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        const std::uint32_t offset = block * 64u;
        const std::uint32_t valid = std::min(64u, count - offset);
        const float scale = q8_scale(values + offset, valid);
        scales[block] = scale;
        for (std::uint32_t i = 0; i < valid; ++i) {
            float scaled = std::isfinite(values[offset + i]) ?
                values[offset + i] / scale : 0.0f;
            scaled = std::max(-127.0f, std::min(127.0f, scaled));
            output[offset + i] = static_cast<std::int8_t>(std::nearbyint(scaled));
        }
    }
}

static void cpu_attention(const std::vector<std::int8_t> &keys,
                          const std::vector<std::int8_t> &values,
                          const std::vector<float> &key_scales,
                          const std::vector<float> &value_scales,
                          const std::vector<float> &query, std::vector<float> &output,
                          std::uint32_t query_heads, std::uint32_t kv_heads,
                          std::uint32_t head_dim, std::uint32_t position, float scale) {
    const std::uint32_t channels = kv_heads * head_dim;
    const std::uint32_t blocks = (channels + 63u) / 64u;
    std::vector<float> scores(position + 1u);
    output.assign(query_heads * head_dim, 0.0f);
    for (std::uint32_t head = 0; head < query_heads; ++head) {
        const std::uint32_t kv_head = (head * kv_heads) / query_heads;
        const std::uint32_t offset = kv_head * head_dim;
        float maximum = -INFINITY;
        for (std::uint32_t token = 0; token <= position; ++token) {
            float dot = 0.0f;
            for (std::uint32_t channel = 0; channel < head_dim; ++channel) {
                const std::uint32_t absolute = offset + channel;
                dot += query[head * head_dim + channel] *
                    static_cast<float>(keys[token * channels + absolute]) *
                    key_scales[token * blocks + absolute / 64u];
            }
            scores[token] = dot * scale;
            maximum = std::max(maximum, scores[token]);
        }
        float sum = 0.0f;
        for (float &score : scores) { score = std::exp(score - maximum); sum += score; }
        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (std::uint32_t channel = 0; channel < head_dim; ++channel) {
            float result = 0.0f;
            for (std::uint32_t token = 0; token <= position; ++token) {
                const std::uint32_t absolute = offset + channel;
                result += scores[token] *
                    static_cast<float>(values[token * channels + absolute]) *
                    value_scales[token * blocks + absolute / 64u];
            }
            output[head * head_dim + channel] = result * inverse;
        }
    }
}

int main() {
    QwnCudaDeviceInfo devices[4]{};
    std::uint32_t count = 0;
    if (qwn_cuda_abi_enumerate_devices(devices, 4, &count) != QWN_CUDA_STATUS_OK || count == 0) {
        std::puts("SKIP: no CUDA device available");
        return 77;
    }
    QwnCudaContextOptions options{};
    qwn_cuda_abi_header_init(&options.header, sizeof(options));
    options.device_id = devices[0].device_id;
    options.context_size = 8;
    QwnCudaContextHandle context{};
    qwn_cuda_abi_header_init(&context.header, sizeof(context));
    if (qwn_cuda_abi_context_create(&options, &context) != QWN_CUDA_STATUS_OK) return fail("context");

    QwnCudaKvCacheOptions cache_options{};
    qwn_cuda_abi_header_init(&cache_options.header, sizeof(cache_options));
    cache_options.max_tokens = 8;
    cache_options.kv_heads = 2;
    cache_options.head_dim = 4;
    QwnCudaKvCacheHandle cache{};
    qwn_cuda_abi_header_init(&cache.header, sizeof(cache));
    if (qwn_cuda_abi_kv_cache_create(&context, &cache_options, &cache) != QWN_CUDA_STATUS_OK)
        return fail("create");

    constexpr std::uint32_t tokens = 2, query_heads = 4, kv_heads = 2, head_dim = 4;
    constexpr std::uint32_t channels = kv_heads * head_dim;
    std::vector<float> key(tokens * channels), value(tokens * channels), query(query_heads * head_dim);
    for (std::uint32_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<float>(static_cast<int>(i) - 3) * 0.17f;
        value[i] = static_cast<float>(static_cast<int>(i) + 2) * -0.11f;
    }
    for (std::uint32_t i = 0; i < query.size(); ++i)
        query[i] = static_cast<float>(static_cast<int>(i) - 5) * 0.09f;
    std::vector<std::int8_t> qkeys(tokens * channels), qvalues(tokens * channels);
    std::vector<float> ks(tokens), vs(tokens);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        q8_quantize(key.data() + token * channels, qkeys.data() + token * channels,
                    ks.data() + token, channels);
        q8_quantize(value.data() + token * channels, qvalues.data() + token * channels,
                    vs.data() + token, channels);
        QwnCudaKvAppendRequest append{};
        qwn_cuda_abi_header_init(&append.header, sizeof(append));
        append.cache = cache;
        append.host_key = key.data() + token * channels;
        append.host_value = value.data() + token * channels;
        append.n_channels = channels;
        append.token = token;
        QwnCudaTelemetry telemetry{};
        qwn_cuda_abi_header_init(&telemetry.header, sizeof(telemetry));
        if (qwn_cuda_abi_kv_cache_append(&context, &append, &telemetry) != QWN_CUDA_STATUS_OK)
            return fail("append");
    }
    std::vector<float> expected, actual(query_heads * head_dim);
    cpu_attention(qkeys, qvalues, ks, vs, query, expected, query_heads, kv_heads, head_dim,
                  tokens - 1, 0.5f);
    QwnCudaKvAttentionRequest attention{};
    qwn_cuda_abi_header_init(&attention.header, sizeof(attention));
    attention.cache = cache;
    attention.host_query = query.data();
    attention.host_output = actual.data();
    attention.query_heads = query_heads;
    attention.kv_heads = kv_heads;
    attention.head_dim = head_dim;
    attention.position = tokens - 1;
    attention.scale = 0.5f;
    QwnCudaTelemetry telemetry{};
    qwn_cuda_abi_header_init(&telemetry.header, sizeof(telemetry));
    if (qwn_cuda_abi_kv_cache_attention(&context, &attention, &telemetry) != QWN_CUDA_STATUS_OK)
        return fail("attention");
    float max_error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i)
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    std::printf("CUDA Q8 KV reference max_abs_error=%.8g kernel_count=%llu resident_bytes=%llu\n",
                max_error, static_cast<unsigned long long>(telemetry.kv_cache_kernel_count),
                static_cast<unsigned long long>(telemetry.kv_cache_resident_bytes));
    const int status = max_error <= 1e-4f && telemetry.kv_cache_kernel_count >= 5 ? 0 : 1;
    if (qwn_cuda_abi_kv_cache_destroy(&context, &cache) != QWN_CUDA_STATUS_OK)
        return fail("destroy");
    QwnCudaTelemetry after_destroy{};
    qwn_cuda_abi_header_init(&after_destroy.header, sizeof(after_destroy));
    if (qwn_cuda_abi_get_telemetry(&context, &after_destroy) != QWN_CUDA_STATUS_OK ||
        after_destroy.kv_cache_resident_bytes != 0) return fail("resident cleanup");
    qwn_cuda_abi_context_destroy(&context);
    return status;
}
