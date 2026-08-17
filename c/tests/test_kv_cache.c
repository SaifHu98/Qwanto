#include "../qwanto_turboquant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int near_value(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

int main(void) {
    QwnKvCacheMode mode = QWN_KV_CACHE_FP16;
    QwnKvCacheContract contract;
    float key[130], value[130], query[65], scores[16], context[65];
    QwnQ8Cache cache;

    if (qwn_kv_cache_mode_parse("q8", &mode) != 0 || mode != QWN_KV_CACHE_Q8)
        return 1;
    if (qwn_kv_cache_mode_parse("turboquant-q4", &mode) != 0 ||
        mode != QWN_KV_CACHE_TURBOQUANT_Q4)
        return 2;
    qwn_kv_cache_contract_init(&contract, QWN_KV_CACHE_Q8, 3);
    if (contract.struct_size != sizeof(contract) ||
        contract.abi_version != QWN_KV_CACHE_ABI_VERSION ||
        contract.block_size != 64 || contract.valid_token_count != 3 ||
        contract.alignment != 64)
        return 3;

    for (int i = 0; i < 130; i++) {
        key[i] = (float)(i - 65) * 0.03125f;
        value[i] = sinf((float)i * 0.07f);
    }
    for (int i = 0; i < 65; i++) query[i] = cosf((float)i * 0.03f);
    memset(&cache, 0, sizeof(cache));
    if (qwn_q8_cache_init(&cache, 16, 2, 65) != 0) return 4;
    if (qwn_q8_cache_append(&cache, key, value, 130) != 0) return 5;
    if (cache.n_tokens != 1 || cache.contract.valid_token_count != 1 ||
        cache.total_bytes == 0)
        return 6;
    float dot = qwn_q8_cache_dot_key_scalar(&cache, 0, 0, query, 65);
    if (!isfinite(dot)) return 7;
    memset(context, 0, sizeof(context));
    qwn_q8_cache_accum_value_scalar(&cache, 0, 0, 1.0f, context, 65);
    if (!near_value(context[0], value[0], 0.02f) ||
        !near_value(context[64], value[64], 0.02f))
        return 8;
    memset(context, 0, sizeof(context));
    qwn_q8_cache_attention_head(query, &cache, 0, 0, 1.0f, scores, context);
    for (int i = 0; i < 65; i++) if (!isfinite(context[i])) return 9;
    qwn_q8_cache_reset(&cache);
    if (cache.n_tokens != 0 || cache.contract.valid_token_count != 0) return 10;
    key[0] = NAN;
    key[1] = INFINITY;
    value[0] = -INFINITY;
    value[1] = NAN;
    if (qwn_q8_cache_append(&cache, key, value, 130) != 0) return 11;
    if (!isfinite(cache.scales_k[0]) || !isfinite(cache.scales_v[0]) ||
        cache.packed_k[0] != 0 || cache.packed_k[1] != 0 ||
        cache.packed_v[0] != 0 || cache.packed_v[1] != 0) return 12;
    qwn_q8_cache_free(&cache);
    puts("QWN typed KV cache reference tests passed");
    return 0;
}
