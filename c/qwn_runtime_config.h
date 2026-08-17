#ifndef QWN_RUNTIME_CONFIG_H
#define QWN_RUNTIME_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QWN_RUNTIME_BACKEND_AUTO = 0,
    QWN_RUNTIME_BACKEND_CPU = 1,
    QWN_RUNTIME_BACKEND_CUDA = 2
} QwnRuntimeBackend;

typedef struct {
    QwnRuntimeBackend backend;
    int gpu_device;
    int cpu_threads;
    int context_size;
    int max_tokens;
    int seed;
    char kv_cache_mode[16];
    char quantization[32];
    char kernel[32];
    char thinking_mode[16];
    int speculative_decoding;
    int fused_kernel;
} QwnRuntimeConfig;

void qwn_runtime_config_default(QwnRuntimeConfig *config);
const char *qwn_runtime_backend_name(QwnRuntimeBackend backend);
int qwn_runtime_config_parse(QwnRuntimeConfig *config, int argc, char **argv,
                             int start_index, char *error, size_t error_size);
int qwn_runtime_config_validate(const QwnRuntimeConfig *config,
                                char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
