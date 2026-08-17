#include "qwn_runtime_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) {
        snprintf(error, error_size, "%s", message);
    }
}

void qwn_runtime_config_default(QwnRuntimeConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->backend = QWN_RUNTIME_BACKEND_AUTO;
    config->gpu_device = -1;
    config->cpu_threads = 0;
    config->context_size = 4096;
    config->max_tokens = 256;
    config->seed = 0;
    snprintf(config->kv_cache_mode, sizeof(config->kv_cache_mode), "fp16");
    snprintf(config->quantization, sizeof(config->quantization), "auto");
    snprintf(config->kernel, sizeof(config->kernel), "auto");
    snprintf(config->thinking_mode, sizeof(config->thinking_mode), "medium");
}

const char *qwn_runtime_backend_name(QwnRuntimeBackend backend) {
    switch (backend) {
        case QWN_RUNTIME_BACKEND_CPU: return "cpu";
        case QWN_RUNTIME_BACKEND_CUDA: return "cuda";
        default: return "auto";
    }
}

static int parse_positive(const char *value, int *out, const char *name,
                          char *error, size_t error_size) {
    char *end = NULL;
    long parsed;
    if (!value || !*value) {
        snprintf(error, error_size, "%s requires a value", name);
        return -1;
    }
    parsed = strtol(value, &end, 10);
    if (end == value || *end || parsed <= 0 || parsed > 2147483647L) {
        snprintf(error, error_size, "%s must be a positive integer", name);
        return -1;
    }
    *out = (int)parsed;
    return 0;
}

static int parse_nonnegative(const char *value, int *out, const char *name,
                             char *error, size_t error_size) {
    char *end = NULL;
    long parsed;
    if (!value || !*value) {
        snprintf(error, error_size, "%s requires a value", name);
        return -1;
    }
    parsed = strtol(value, &end, 10);
    if (end == value || *end || parsed < 0 || parsed > 2147483647L) {
        snprintf(error, error_size, "%s must be a non-negative integer", name);
        return -1;
    }
    *out = (int)parsed;
    return 0;
}

static int parse_backend(QwnRuntimeConfig *config, const char *value,
                         char *error, size_t error_size) {
    if (!value) {
        set_error(error, error_size, "--backend requires cpu, cuda, or auto");
        return -1;
    }
    if (strcmp(value, "cpu") == 0) config->backend = QWN_RUNTIME_BACKEND_CPU;
    else if (strcmp(value, "cuda") == 0) config->backend = QWN_RUNTIME_BACKEND_CUDA;
    else if (strcmp(value, "auto") == 0) config->backend = QWN_RUNTIME_BACKEND_AUTO;
    else {
        set_error(error, error_size, "--backend must be cpu, cuda, or auto");
        return -1;
    }
    return 0;
}

int qwn_runtime_config_validate(const QwnRuntimeConfig *config,
                                char *error, size_t error_size) {
    if (!config) {
        set_error(error, error_size, "runtime configuration is missing");
        return -1;
    }
    if (config->gpu_device < -1) {
        set_error(error, error_size, "GPU device index must be -1 or non-negative");
        return -1;
    }
    if (config->cpu_threads < 0 || config->context_size <= 0 || config->max_tokens <= 0) {
        set_error(error, error_size, "threads, context size, and max tokens must be positive");
        return -1;
    }
    if (strcmp(config->kv_cache_mode, "fp16") != 0 &&
        strcmp(config->kv_cache_mode, "auto") != 0) {
        set_error(error, error_size, "only fp16 KV cache is implemented by qwnrun");
        return -1;
    }
    if (strcmp(config->quantization, "auto") != 0 &&
        strcmp(config->quantization, "q4_0") != 0 &&
        strcmp(config->quantization, "hyper_vsq2") != 0 &&
        strcmp(config->quantization, "fp16") != 0 &&
        strcmp(config->quantization, "fp32") != 0) {
        set_error(error, error_size, "quantization must be auto, q4_0, hyper_vsq2, fp16, or fp32");
        return -1;
    }
    if (strcmp(config->kernel, "auto") != 0 &&
        strcmp(config->kernel, "scalar") != 0 &&
        strcmp(config->kernel, "avx2") != 0 &&
        strcmp(config->kernel, "vnni") != 0) {
        set_error(error, error_size, "kernel must be auto, scalar, avx2, or vnni");
        return -1;
    }
    if (strcmp(config->thinking_mode, "none") != 0 &&
        strcmp(config->thinking_mode, "low") != 0 &&
        strcmp(config->thinking_mode, "medium") != 0 &&
        strcmp(config->thinking_mode, "high") != 0) {
        set_error(error, error_size, "thinking mode must be none, low, medium, or high");
        return -1;
    }
    if (config->speculative_decoding || config->fused_kernel) {
        set_error(error, error_size, "speculative decoding and fused kernel flags are unsupported by qwnrun");
        return -1;
    }
    if (config->backend == QWN_RUNTIME_BACKEND_CUDA && config->gpu_device < 0) {
        /* CUDA auto-selection is valid; -1 means the runtime chooses device 0. */
        return 0;
    }
    return 0;
}

int qwn_runtime_config_parse(QwnRuntimeConfig *config, int argc, char **argv,
                             int start_index, char *error, size_t error_size) {
    int i;
    if (!config) return -1;
    qwn_runtime_config_default(config);
    if (error && error_size) error[0] = '\0';
    for (i = start_index; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--backend") == 0 || strcmp(arg, "--gpu-backend") == 0) {
            if (++i >= argc || parse_backend(config, argv[i], error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--gpu") == 0) {
            config->backend = QWN_RUNTIME_BACKEND_CUDA;
        } else if (strcmp(arg, "--gpu-device") == 0) {
            if (++i >= argc || parse_nonnegative(argv[i], &config->gpu_device, "--gpu-device", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--threads") == 0) {
            if (++i >= argc || parse_nonnegative(argv[i], &config->cpu_threads, "--threads", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--ctx-size") == 0) {
            if (++i >= argc || parse_positive(argv[i], &config->context_size, "--ctx-size", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--max-tokens") == 0) {
            if (++i >= argc || parse_positive(argv[i], &config->max_tokens, "--max-tokens", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--seed") == 0) {
            if (++i >= argc || parse_nonnegative(argv[i], &config->seed, "--seed", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--kv-cache") == 0) {
            if (++i >= argc) {
                set_error(error, error_size, "--kv-cache requires a value");
                return -1;
            }
            snprintf(config->kv_cache_mode, sizeof(config->kv_cache_mode), "%s", argv[i]);
        } else if (strcmp(arg, "--quantization") == 0 || strcmp(arg, "--quant") == 0) {
            if (++i >= argc) {
                set_error(error, error_size, "--quantization requires a value");
                return -1;
            }
            snprintf(config->quantization, sizeof(config->quantization), "%s", argv[i]);
        } else if (strcmp(arg, "--kernel") == 0) {
            if (++i >= argc) {
                set_error(error, error_size, "--kernel requires a value");
                return -1;
            }
            snprintf(config->kernel, sizeof(config->kernel), "%s", argv[i]);
        } else if (strcmp(arg, "--thinking") == 0) {
            if (++i >= argc) {
                set_error(error, error_size, "--thinking requires none, low, medium, or high");
                return -1;
            }
            snprintf(config->thinking_mode, sizeof(config->thinking_mode), "%s", argv[i]);
        } else if (strcmp(arg, "--speculative") == 0 || strcmp(arg, "--saguro") == 0 ||
                   strcmp(arg, "--fused") == 0 || strcmp(arg, "--auto-tune") == 0) {
            snprintf(error, error_size, "%s is unsupported by the native decoder", arg);
            return -1;
        } else if (strcmp(arg, "--saguro-draft") == 0 || strcmp(arg, "--saguro-tier") == 0) {
            snprintf(error, error_size, "%s is unsupported by the native decoder", arg);
            return -1;
        }
    }
    return qwn_runtime_config_validate(config, error, error_size);
}
