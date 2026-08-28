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
    config->gpu_memory_budget_bytes = 0;
    config->cpu_threads = 0;
    config->context_size = 4096;
    config->max_tokens = 256;
    config->seed = 0;
    snprintf(config->kv_cache_mode, sizeof(config->kv_cache_mode), "fp16");
    config->kv_cache_mode_typed = QWN_RUNTIME_KV_FP16;
    snprintf(config->quantization, sizeof(config->quantization), "auto");
    snprintf(config->kernel, sizeof(config->kernel), "auto");
    snprintf(config->thinking_mode, sizeof(config->thinking_mode), "medium");
    config->speculative_draft_length = 4;
    config->minimum_acceptance_rate = 0.0f;
    config->adaptive_draft_length = 1;
    config->maximum_rollback = 16;
}

const char *qwn_runtime_backend_name(QwnRuntimeBackend backend) {
    switch (backend) {
        case QWN_RUNTIME_BACKEND_CPU: return "cpu";
        case QWN_RUNTIME_BACKEND_CUDA: return "cuda";
        default: return "auto";
    }
}

const char *qwn_runtime_kv_cache_mode_name(QwnRuntimeKvCacheMode mode) {
    switch (mode) {
        case QWN_RUNTIME_KV_Q8: return "q8";
        case QWN_RUNTIME_KV_TURBOQUANT_Q4: return "turboquant-q4";
        case QWN_RUNTIME_KV_TURBOQUANT_PAPER: return "turboquant-paper";
        case QWN_RUNTIME_KV_AUTO: return "auto";
        default: return "fp16";
    }
}

int qwn_runtime_kv_cache_mode_parse(const char *value,
                                    QwnRuntimeKvCacheMode *mode) {
    if (!value || !mode) return -1;
    if (strcmp(value, "fp16") == 0) *mode = QWN_RUNTIME_KV_FP16;
    else if (strcmp(value, "q8") == 0) *mode = QWN_RUNTIME_KV_Q8;
    else if (strcmp(value, "turboquant-q4") == 0 ||
             strcmp(value, "qwn-q4-kv") == 0)
        *mode = QWN_RUNTIME_KV_TURBOQUANT_Q4;
    else if (strcmp(value, "turboquant-paper") == 0)
        *mode = QWN_RUNTIME_KV_TURBOQUANT_PAPER;
    else if (strcmp(value, "auto") == 0) *mode = QWN_RUNTIME_KV_AUTO;
    else return -1;
    return 0;
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

static int parse_memory_budget_mb(const char *value, uint64_t *out,
                                  char *error, size_t error_size) {
    char *end = NULL;
    unsigned long long parsed;
    if (!value || !*value) {
        snprintf(error, error_size, "--gpu-memory-budget-mb requires a value");
        return -1;
    }
    parsed = strtoull(value, &end, 10);
    if (end == value || *end || parsed == 0 ||
        parsed > UINT64_MAX / (1024ULL * 1024ULL)) {
        snprintf(error, error_size,
                 "--gpu-memory-budget-mb must be a positive integer");
        return -1;
    }
    *out = (uint64_t)parsed * 1024ULL * 1024ULL;
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
        strcmp(config->kv_cache_mode, "q8") != 0 &&
        strcmp(config->kv_cache_mode, "turboquant-q4") != 0 &&
        strcmp(config->kv_cache_mode, "qwn-q4-kv") != 0 &&
        strcmp(config->kv_cache_mode, "turboquant-paper") != 0 &&
        strcmp(config->kv_cache_mode, "auto") != 0) {
        set_error(error, error_size,
                  "kv cache must be fp16, q8, turboquant-q4, turboquant-paper, or auto");
        return -1;
    }
    {
        QwnRuntimeKvCacheMode parsed;
        if (qwn_runtime_kv_cache_mode_parse(config->kv_cache_mode, &parsed) != 0 ||
            parsed != config->kv_cache_mode_typed) {
            /* Callers that construct the struct directly may leave the
             * compatibility string and typed field out of sync.  Reject
             * that ambiguity instead of selecting a different cache. */
            set_error(error, error_size, "typed KV-cache mode does not match kv_cache_mode");
            return -1;
        }
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
    if (config->speculative_draft_length <= 0 || config->speculative_draft_length > 16 ||
        config->maximum_rollback <= 0 || config->maximum_rollback > 1024 ||
        config->minimum_acceptance_rate < 0.0f || config->minimum_acceptance_rate > 1.0f) {
        set_error(error, error_size, "invalid speculative decoding limits");
        return -1;
    }
    if (config->speculative_decoding && config->draft_model[0]) {
        set_error(error, error_size,
                  "native speculative mode uses the model's MTP head and does not accept --draft-model");
        return -1;
    }
    if (config->fused_kernel) {
        set_error(error, error_size, "fused kernels are unsupported");
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
        } else if (strcmp(arg, "--gpu-memory-budget-mb") == 0) {
            if (++i >= argc || parse_memory_budget_mb(argv[i],
                                                       &config->gpu_memory_budget_bytes,
                                                       error, error_size) != 0) return -1;
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
            if (qwn_runtime_kv_cache_mode_parse(config->kv_cache_mode,
                                                &config->kv_cache_mode_typed) != 0) {
                set_error(error, error_size, "kv cache must be fp16, q8, turboquant-q4, turboquant-paper, or auto");
                return -1;
            }
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
        } else if (strcmp(arg, "--draft-model") == 0) {
            if (++i >= argc) {
                set_error(error, error_size, "--draft-model requires a validated .qwn path");
                return -1;
            }
            snprintf(config->draft_model, sizeof(config->draft_model), "%s", argv[i]);
        } else if (strcmp(arg, "--draft-length") == 0) {
            if (++i >= argc || parse_positive(argv[i], &config->speculative_draft_length,
                                               "--draft-length", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--min-acceptance-rate") == 0) {
            if (++i >= argc) { set_error(error, error_size, "--min-acceptance-rate requires a value"); return -1; }
            config->minimum_acceptance_rate = (float)strtod(argv[i], NULL);
        } else if (strcmp(arg, "--maximum-rollback") == 0) {
            if (++i >= argc || parse_positive(argv[i], &config->maximum_rollback,
                                               "--maximum-rollback", error, error_size) != 0) return -1;
        } else if (strcmp(arg, "--no-adaptive-draft-length") == 0) {
            config->adaptive_draft_length = 0;
        } else if (strcmp(arg, "--speculative") == 0 || strcmp(arg, "--saguro") == 0 ||
                   strcmp(arg, "--fused") == 0 || strcmp(arg, "--auto-tune") == 0) {
            if (strcmp(arg, "--speculative") == 0) {
                config->speculative_decoding = 1;
                snprintf(config->thinking_mode, sizeof(config->thinking_mode), "none");
            }
            else {
                snprintf(error, error_size, "%s is unsupported by the native decoder", arg);
                return -1;
            }
        } else if (strcmp(arg, "--saguro-draft") == 0 || strcmp(arg, "--saguro-tier") == 0) {
            snprintf(error, error_size, "%s is unsupported by the native decoder", arg);
            return -1;
        }
    }
    return qwn_runtime_config_validate(config, error, error_size);
}
