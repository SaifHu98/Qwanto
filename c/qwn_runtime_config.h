#ifndef QWN_RUNTIME_CONFIG_H
#define QWN_RUNTIME_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QWN_RUNTIME_BACKEND_AUTO = 0,
    QWN_RUNTIME_BACKEND_CPU = 1,
    QWN_RUNTIME_BACKEND_CUDA = 2
} QwnRuntimeBackend;

typedef enum {
    QWN_RUNTIME_KV_FP16 = 0,
    QWN_RUNTIME_KV_Q8 = 1,
    QWN_RUNTIME_KV_TURBOQUANT_Q4 = 2,
    QWN_RUNTIME_KV_TURBOQUANT_PAPER = 3,
    QWN_RUNTIME_KV_AUTO = 4
} QwnRuntimeKvCacheMode;

typedef struct {
    QwnRuntimeBackend backend;
    int gpu_device;
    uint64_t gpu_memory_budget_bytes;
    int cpu_threads;
    int context_size;
    int max_tokens;
    int seed;
    char kv_cache_mode[24];
    QwnRuntimeKvCacheMode kv_cache_mode_typed;
    char quantization[32];
    char kernel[32];
    char thinking_mode[16];
    int speculative_decoding;
    char draft_model[1024];
    int speculative_draft_length;
    float minimum_acceptance_rate;
    int adaptive_draft_length;
    int maximum_rollback;
    int fused_kernel;
} QwnRuntimeConfig;

void qwn_runtime_config_default(QwnRuntimeConfig *config);
const char *qwn_runtime_backend_name(QwnRuntimeBackend backend);
const char *qwn_runtime_kv_cache_mode_name(QwnRuntimeKvCacheMode mode);
int qwn_runtime_kv_cache_mode_parse(const char *value,
                                    QwnRuntimeKvCacheMode *mode);
int qwn_runtime_config_parse(QwnRuntimeConfig *config, int argc, char **argv,
                             int start_index, char *error, size_t error_size);
int qwn_runtime_config_validate(const QwnRuntimeConfig *config,
                                char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
