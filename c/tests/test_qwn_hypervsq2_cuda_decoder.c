#include "../qwanto_decode.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CUDA_LOGIT_ABS_TOLERANCE 0.1f

static int argmax(const float *values, int count) {
    int best = 0;
    for (int i = 1; i < count; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "../experiments/results/4B_hyper_vsq2.qwn";
    const char *error = NULL;
    QwnRuntimeConfig cpu_config;
    QwnRuntimeConfig cuda_config;
    qwn_runtime_config_default(&cpu_config);
    qwn_runtime_config_default(&cuda_config);
    cpu_config.backend = QWN_RUNTIME_BACKEND_CPU;
    snprintf(cpu_config.kernel, sizeof(cpu_config.kernel), "%s",
             argc > 2 ? argv[2] : "scalar");
    cuda_config.backend = QWN_RUNTIME_BACKEND_CUDA;
    cuda_config.gpu_device = 0;
    cpu_config.context_size = 4096;
    cuda_config.context_size = 4096;

    QwnDecoder cpu;
    QwnDecoder cuda;
    if (qwn_decoder_open_with_config(&cpu, model, &cpu_config, &error) != 0) {
        fprintf(stderr, "CPU decoder open failed: %s\n", error ? error : "unknown");
        return 1;
    }
    error = NULL;
    if (qwn_decoder_open_with_config(&cuda, model, &cuda_config, &error) != 0) {
        fprintf(stderr, "SKIP: CUDA decoder unavailable: %s\n", error ? error : "unknown");
        qwn_decoder_close(&cpu);
        return 77;
    }

    qwn_decoder_reset(&cpu);
    qwn_decoder_reset(&cuda);
    const int token = cpu.cfg.bos_id >= 0 ? cpu.cfg.bos_id : 1;
    const float *cpu_logits = NULL;
    const float *cuda_logits = NULL;
    if (qwn_decoder_forward(&cpu, token, &cpu_logits) != 0 ||
        qwn_decoder_forward(&cuda, token, &cuda_logits) != 0 ||
        !cpu_logits || !cuda_logits) {
        fprintf(stderr, "decoder forward failed\n");
        qwn_decoder_close(&cuda);
        qwn_decoder_close(&cpu);
        return 1;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < cpu.cfg.vocab; i++) {
        const float difference = fabsf(cpu_logits[i] - cuda_logits[i]);
        const float relative = difference / fmaxf(fabsf(cpu_logits[i]), 1e-6f);
        if (difference > max_abs) max_abs = difference;
        if (relative > max_rel) max_rel = relative;
        if (difference > CUDA_LOGIT_ABS_TOLERANCE) mismatches++;
    }
    int cpu_token = argmax(cpu_logits, cpu.cfg.vocab);
    int cuda_token = argmax(cuda_logits, cuda.cfg.vocab);
    int token_agreement = cpu_token == cuda_token;
    for (int step = 0; step < 8 && token_agreement; step++) {
        const float *next_cpu = NULL;
        const float *next_cuda = NULL;
        if (qwn_decoder_forward(&cpu, cpu_token, &next_cpu) != 0 ||
            qwn_decoder_forward(&cuda, cuda_token, &next_cuda) != 0 ||
            !next_cpu || !next_cuda) {
            token_agreement = 0;
            break;
        }
        cpu_token = argmax(next_cpu, cpu.cfg.vocab);
        cuda_token = argmax(next_cuda, cuda.cfg.vocab);
        if (cpu_token != cuda_token) token_agreement = 0;
    }
    const QwnRuntimeMetrics *metrics = qwn_decoder_metrics(&cuda);
    const uint64_t matmuls = metrics ? metrics->cuda_matmul_count : 0;
    const uint64_t fallbacks = metrics ? metrics->cpu_fallback_count : 0;
    printf("HyperVSQ-2 decoder comparison (%s): max_abs=%.9g max_rel=%.9g "
           "tolerance=%.9g mismatches=%d token_agreement=%s "
           "cpu_argmax=%d cuda_argmax=%d gpu_matmuls=%llu "
           "cpu_fallbacks=%llu kernel=%s\n",
           cpu_config.kernel, max_abs, max_rel, CUDA_LOGIT_ABS_TOLERANCE,
           mismatches, token_agreement ? "true" : "false", cpu_token, cuda_token,
           (unsigned long long)matmuls, (unsigned long long)fallbacks,
           metrics && metrics->cuda_kernel_type[0] ? metrics->cuda_kernel_type : "Unavailable");

    const int passed = mismatches == 0 && max_abs <= CUDA_LOGIT_ABS_TOLERANCE &&
                       token_agreement && matmuls > 0 && fallbacks == 0;
    qwn_decoder_close(&cuda);
    qwn_decoder_close(&cpu);
    return passed ? 0 : 1;
}
