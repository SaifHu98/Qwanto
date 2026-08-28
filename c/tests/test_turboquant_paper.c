#include "../qwanto_turboquant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static float dot(const float *a, const float *b, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += a[i] * b[i];
    return total;
}

int main(void) {
    enum { D = 16 };
    float key[D], value[D], query[D], output[D];
    for (int i = 0; i < D; i++) {
        key[i] = sinf((float)(i + 1) * 0.23f);
        value[i] = cosf((float)(i + 1) * 0.17f);
        query[i] = sinf((float)(i + 1) * 0.31f);
    }

    QwnTurboQuantPaperCache cache;
    if (qwn_turboquant_paper_cache_init(&cache, 2, 1, D, 3, 7) != 0) return 1;
    if (qwn_turboquant_paper_cache_append(&cache, key, value, D) != 0) return 2;
    if (cache.n_tokens != 1 || cache.vector_bytes == 0 || cache.total_bytes == 0) return 3;
    float estimate = qwn_turboquant_paper_dot_key(&cache, 0, 0, query);
    if (!isfinite(estimate)) return 4;
    memset(output, 0, sizeof(output));
    qwn_turboquant_paper_accum_value(&cache, 0, 0, 0.5f, output);
    for (int i = 0; i < D; i++) if (!isfinite(output[i])) return 5;
    qwn_turboquant_paper_cache_free(&cache);

    /* QJL is stochastic; averaging independent Gaussian projections must
     * recover the original inner product without a systematic bias. */
    float mean = 0.0f;
    for (uint64_t seed = 1; seed <= 96; seed++) {
        if (qwn_turboquant_paper_cache_init(&cache, 1, 1, D, 3, seed) != 0) return 6;
        if (qwn_turboquant_paper_cache_append(&cache, key, value, D) != 0) return 7;
        mean += qwn_turboquant_paper_dot_key(&cache, 0, 0, query);
        qwn_turboquant_paper_cache_free(&cache);
    }
    mean /= 96.0f;
    if (fabsf(mean - dot(key, query, D)) > 0.30f) {
        fprintf(stderr, "QJL mean %f differs from %f\n", mean, dot(key, query, D));
        return 8;
    }
    puts("TurboQuant paper reference tests passed");
    return 0;
}
