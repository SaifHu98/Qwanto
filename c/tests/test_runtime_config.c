#include "../qwn_runtime_config.h"

#include <stdio.h>
#include <string.h>

static int expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

static int parse(int argc, char **argv, QwnRuntimeConfig *config, int must_succeed) {
    char error[256] = {0};
    int rc = qwn_runtime_config_parse(config, argc, argv, 1, error, sizeof(error));
    if ((rc == 0) != must_succeed) {
        fprintf(stderr, "unexpected parse result=%d error=%s\n", rc, error);
        return 1;
    }
    return 0;
}

int main(void) {
    QwnRuntimeConfig config;
    char *default_args[] = {"qwnrun"};
    if (parse(1, default_args, &config, 1) ||
        expect(config.kv_cache_mode_typed == QWN_RUNTIME_KV_FP16,
               "default KV mode is typed FP16") ||
        expect(strcmp(config.kv_cache_mode, "fp16") == 0,
               "default KV compatibility name is FP16")) return 1;

    char *q8_args[] = {"qwnrun", "--kv-cache", "q8", "--threads", "8",
                       "--draft-length", "6", "--maximum-rollback", "12"};
    if (parse((int)(sizeof(q8_args) / sizeof(q8_args[0])), q8_args, &config, 1) ||
        expect(config.kv_cache_mode_typed == QWN_RUNTIME_KV_Q8,
               "q8 selects typed Q8") ||
        expect(config.cpu_threads == 8 && config.speculative_draft_length == 6 &&
                   config.maximum_rollback == 12,
               "typed runtime values retain CLI settings")) return 1;

    char *q4_args[] = {"qwnrun", "--kv-cache", "turboquant-q4"};
    if (parse(3, q4_args, &config, 1) ||
        expect(config.kv_cache_mode_typed == QWN_RUNTIME_KV_TURBOQUANT_Q4,
               "QWN-Q4-KV is explicitly typed")) return 1;

    char *paper_args[] = {"qwnrun", "--kv-cache", "turboquant-paper"};
    if (parse(3, paper_args, &config, 1) ||
        expect(config.kv_cache_mode_typed == QWN_RUNTIME_KV_TURBOQUANT_PAPER,
               "paper TurboQuant is explicitly typed")) return 1;

    char *bad_args[] = {"qwnrun", "--kv-cache", "q3"};
    if (parse(3, bad_args, &config, 0)) return 1;

    char *spec_args[] = {"qwnrun", "--speculative", "--draft-model", "draft.qwn"};
    if (parse(4, spec_args, &config, 0)) return 1;

    puts("typed runtime configuration tests passed");
    return 0;
}
