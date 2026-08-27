#include "qwn_speculative.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

uint64_t qwn_speculative_hash_context(const int *tokens, int count) {
    if (!tokens || count <= 0) return 0ULL;
    int start = count > 8 ? count - 8 : 0;
    uint64_t hash = 14695981039346656037ULL;
    for (int i = start; i < count; i++) {
        hash ^= (uint64_t)(unsigned int)tokens[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

int get_optimal_draft_length(const SpeculationCache *cache) {
    /* Four is a configuration default, not an observed acceptance result. */
    if (!cache || cache->total_drafted < 4) return 4;
    if (cache->acceptance_rate > 0.90f) return 8;
    if (cache->acceptance_rate > 0.70f) return 5;
    return 3;
}

int qwn_speculative_engine_init(QwnSpeculativeEngine *engine,
                                QwnDecoder *target, QwnDecoder *draft,
                                int cache_capacity) {
    if (!engine) return -1;
    memset(engine, 0, sizeof(*engine));
    engine->target_decoder = target;
    engine->draft_decoder = draft;
    engine->max_draft_tokens = 4;
    engine->min_acceptance_rate = 0.0f;
    engine->use_bidirectional = false;
    engine->acceptance_rate_avg = 0.0f;
    engine->cache.capacity = cache_capacity > 0 ? cache_capacity : QWN_SPEC_DEFAULT_CACHE_SIZE;
    engine->cache.entries = (SpeculationCacheEntry *)calloc(
        (size_t)engine->cache.capacity, sizeof(SpeculationCacheEntry));
    if (!engine->cache.entries) {
        memset(engine, 0, sizeof(*engine));
        return -1;
    }
    return 0;
}

void qwn_speculative_engine_free(QwnSpeculativeEngine *engine) {
    if (!engine) return;
    free(engine->cache.entries);
    memset(engine, 0, sizeof(*engine));
}

int qwn_speculative_cache_lookup(SpeculationCache *cache, uint64_t hash,
                                 int *out_tokens, float *out_probs, int max_len) {
    if (!cache || !cache->entries || hash == 0 || !out_tokens || max_len <= 0)
        return 0;
    cache->total_lookups++;
    for (int i = 0; i < cache->count; i++) {
        SpeculationCacheEntry *entry = &cache->entries[i];
        if (entry->hash != hash || entry->token_count <= 0) continue;
        entry->lru_timestamp = ++cache->current_lru_clock;
        cache->total_hits++;
        int copied = entry->token_count < max_len ? entry->token_count : max_len;
        for (int k = 0; k < copied; k++) {
            out_tokens[k] = entry->draft_tokens[k];
            if (out_probs) out_probs[k] = entry->draft_probabilities[k];
        }
        return copied;
    }
    return 0;
}

void qwn_speculative_cache_insert(SpeculationCache *cache, uint64_t hash,
                                  const int *tokens, const float *probs, int len) {
    if (!cache || !cache->entries || hash == 0 || !tokens || len <= 0) return;
    if (len > QWN_SPEC_MAX_DRAFT) len = QWN_SPEC_MAX_DRAFT;
    int slot = -1;
    for (int i = 0; i < cache->count; i++) {
        if (cache->entries[i].hash == hash) { slot = i; break; }
    }
    if (slot < 0 && cache->count < cache->capacity) slot = cache->count++;
    if (slot < 0) {
        uint64_t oldest = UINT64_MAX;
        slot = 0;
        for (int i = 0; i < cache->capacity; i++) {
            if (cache->entries[i].lru_timestamp < oldest) {
                oldest = cache->entries[i].lru_timestamp;
                slot = i;
            }
        }
    }
    SpeculationCacheEntry *entry = &cache->entries[slot];
    memset(entry, 0, sizeof(*entry));
    entry->hash = hash;
    entry->token_count = len;
    entry->lru_timestamp = ++cache->current_lru_clock;
    for (int i = 0; i < len; i++) {
        entry->draft_tokens[i] = tokens[i];
        entry->draft_probabilities[i] = probs ? probs[i] : 0.0f;
    }
}

void qwn_speculative_cache_update_rate(SpeculationCache *cache, int drafted,
                                       int accepted) {
    if (!cache || drafted <= 0 || accepted < 0 || accepted > drafted) return;
    cache->total_drafted += (uint64_t)drafted;
    cache->total_accepted += (uint64_t)accepted;
    cache->acceptance_rate = cache->total_drafted > 0 ?
        (float)cache->total_accepted / (float)cache->total_drafted : 0.0f;
}

int qwn_spec_ring_push(QwnSpeculativeEngine *engine, int token) {
    if (!engine) return -1;
    if (engine->ring_count == QWN_SPEC_RING_SIZE) {
        engine->ring_tail = (engine->ring_tail + 1) % QWN_SPEC_RING_SIZE;
        engine->ring_count--;
    }
    engine->ring_buffer[engine->ring_head] = token;
    engine->ring_head = (engine->ring_head + 1) % QWN_SPEC_RING_SIZE;
    engine->ring_count++;
    return 0;
}

int qwn_spec_ring_pop(QwnSpeculativeEngine *engine, int *out_token) {
    if (!engine || !out_token || engine->ring_count <= 0) return -1;
    *out_token = engine->ring_buffer[engine->ring_tail];
    engine->ring_tail = (engine->ring_tail + 1) % QWN_SPEC_RING_SIZE;
    engine->ring_count--;
    return 0;
}

void qwn_spec_ring_clear(QwnSpeculativeEngine *engine) {
    if (!engine) return;
    engine->ring_head = engine->ring_tail = engine->ring_count = 0;
}

/* These legacy entry points deliberately do not invent draft tokens or
 * acceptance results.  The typed QwnSpecContext path is the only execution
 * path and fails closed until a compatible native draft is available. */
void qwn_generate_draft_sequence(QwnDecoder *draft_decoder, const int *context,
                                 int context_len, int *draft_tokens,
                                 float *draft_probs, int n_draft_tokens) {
    (void)draft_decoder; (void)context; (void)context_len;
    if (draft_tokens && n_draft_tokens > 0)
        memset(draft_tokens, 0, (size_t)n_draft_tokens * sizeof(*draft_tokens));
    if (draft_probs && n_draft_tokens > 0)
        memset(draft_probs, 0, (size_t)n_draft_tokens * sizeof(*draft_probs));
}

int qwn_verify_and_accept(QwnSpeculativeEngine *engine, const int *draft_tokens,
                          int n_draft_tokens, const float *draft_probs,
                          int *accepted_tokens, float *target_probs) {
    (void)engine; (void)draft_tokens; (void)n_draft_tokens;
    (void)draft_probs; (void)accepted_tokens; (void)target_probs;
    return 0;
}

int qwn_speculative_forward(QwnSpeculativeEngine *engine, const int *prompt,
                            int prompt_len, int *output, int max_tokens,
                            float temperature) {
    (void)engine; (void)prompt; (void)prompt_len; (void)output;
    (void)max_tokens; (void)temperature;
    return QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL;
}

int qwn_speculative_generate_stream(QwnSpeculativeEngine *engine,
                                    const int *prompt, int prompt_len,
                                    int max_tokens, float temperature,
                                    float top_p,
                                    void (*callback)(const char *, int, void *),
                                    void *opaque) {
    (void)engine; (void)prompt; (void)prompt_len; (void)max_tokens;
    (void)temperature; (void)top_p; (void)callback; (void)opaque;
    return QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL;
}

static double spec_now(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return (double)clock() / (double)CLOCKS_PER_SEC;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

static uint64_t spec_hash_tokenizer(const QwnDecoder *decoder) {
    uint64_t hash = 1469598103934665603ULL;
    if (!decoder || !decoder->tokenizer.id2str || decoder->tokenizer.n_ids <= 0) return 0;
    for (int id = 0; id < decoder->tokenizer.n_ids; id++) {
        const unsigned char *text = (const unsigned char *)decoder->tokenizer.id2str[id];
        if (!text) return 0;
        hash ^= (uint64_t)(id + 1);
        hash *= 1099511628211ULL;
        while (*text) { hash ^= *text++; hash *= 1099511628211ULL; }
        hash ^= 0xffu;
        hash *= 1099511628211ULL;
    }
    return hash;
}

int qwn_speculative_check_compatibility(const QwnDecoder *target,
                                        const QwnDecoder *draft,
                                        char *reason, size_t reason_size) {
    const char *message = "compatible";
    if (!target || !draft) message = "compatible native QWN draft model is required";
    else if (!target->model_sha256[0] || !draft->model_sha256[0] ||
             strcmp(target->model_sha256, "Unavailable") == 0 ||
             strcmp(draft->model_sha256, "Unavailable") == 0)
        message = "target and draft model hashes are unavailable";
    else if (target->cfg.vocab != draft->cfg.vocab ||
             target->tokenizer.n_ids != draft->tokenizer.n_ids)
        message = "target and draft vocabulary sizes differ";
    else if (target->cfg.bos_id != draft->cfg.bos_id ||
             target->cfg.eos_id != draft->cfg.eos_id)
        message = "target and draft special token IDs differ";
    else if (target->cfg.max_ctx != draft->cfg.max_ctx)
        message = "target and draft context policies differ";
    else if (spec_hash_tokenizer(target) == 0 ||
             spec_hash_tokenizer(target) != spec_hash_tokenizer(draft))
        message = "target and draft tokenizer token-ID mapping differs";
    /* Native QWN currently does not expose a validated chat-template identity.
     * Refuse compatibility rather than silently assuming templates match. */
    else message = "chat-template identity is unavailable in native QWN metadata";
    if (reason && reason_size) snprintf(reason, reason_size, "%s", message);
    return strcmp(message, "compatible") == 0 ? QWN_SPEC_OK :
           QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL;
}

int qwn_speculative_init(QwnSpecContext *ctx, QwnDecoder *target,
                         QwnDecoder *draft, int gamma) {
    char reason[160];
    if (!ctx || !target || !draft) return QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL;
    if (qwn_speculative_check_compatibility(target, draft, reason, sizeof(reason)) != QWN_SPEC_OK)
        return QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL;
    memset(ctx, 0, sizeof(*ctx));
    ctx->target = target;
    ctx->draft = draft;
    ctx->gamma = gamma > 0 && gamma <= QWN_SPEC_MAX_GAMMA ? gamma : 4;
    ctx->top_p = 1.0f;
    ctx->rng_state = 0x853c49e6748fea9bULL;
    ctx->history_capacity = target->cfg.max_ctx > 0 ? target->cfg.max_ctx : 4096;
    ctx->history = (int *)malloc((size_t)ctx->history_capacity * sizeof(int));
    if (!ctx->history) return QWN_SPEC_INVALID_ARGUMENT;

    int vocab = target->cfg.vocab;
    ctx->draft_all_capacity = (size_t)QWN_SPEC_MAX_GAMMA * (size_t)vocab;
    ctx->probs_capacity = (size_t)vocab;
    ctx->draft_all = (float *)malloc(ctx->draft_all_capacity * sizeof(float));
    ctx->draft_probs = (float *)malloc(ctx->probs_capacity * sizeof(float));
    ctx->target_probs = (float *)malloc(ctx->probs_capacity * sizeof(float));
    if (!ctx->draft_all || !ctx->draft_probs || !ctx->target_probs) {
        free(ctx->history);
        free(ctx->draft_all);
        free(ctx->draft_probs);
        free(ctx->target_probs);
        memset(ctx, 0, sizeof(*ctx));
        return QWN_SPEC_INVALID_ARGUMENT;
    }
    snprintf(ctx->counters.status, sizeof(ctx->counters.status),
             "IMPLEMENTED_REQUIRES_COMPATIBLE_DRAFT_MODEL");
    return QWN_SPEC_OK;
}

void qwn_speculative_free(QwnSpecContext *ctx) {
    if (!ctx) return;
    free(ctx->history);
    free(ctx->draft_all);
    free(ctx->draft_probs);
    free(ctx->target_probs);
    memset(ctx, 0, sizeof(*ctx));
}

static float spec_random(QwnSpecContext *ctx) {
    uint64_t x = ctx->rng_state ? ctx->rng_state : 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27; ctx->rng_state = x;
    return (float)((x * 2685821657736338717ULL) >> 40) * (1.0f / 16777216.0f);
}

static void spec_softmax(const float *logits, float *probabilities, int vocab,
                         float temperature) {
    float max_value = -INFINITY;
    float divisor = temperature > 0.0f ? temperature : 1.0f;
    for (int i = 0; i < vocab; i++) {
        float value = logits[i] / divisor;
        probabilities[i] = value;
        if (value > max_value) max_value = value;
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        probabilities[i] = expf(probabilities[i] - max_value);
        sum += probabilities[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        memset(probabilities, 0, (size_t)vocab * sizeof(float));
        probabilities[0] = 1.0f;
        return;
    }
    for (int i = 0; i < vocab; i++) probabilities[i] /= sum;
}

static int spec_sample(QwnSpecContext *ctx, const float *probabilities, int vocab) {
    float draw = spec_random(ctx), cumulative = 0.0f;
    for (int i = 0; i < vocab; i++) {
        cumulative += probabilities[i];
        if (draw <= cumulative) return i;
    }
    return vocab - 1;
}

static int spec_append_history(QwnSpecContext *ctx, int token) {
    if (ctx->history_count >= ctx->history_capacity) return QWN_SPEC_CONTEXT_LIMIT;
    ctx->history[ctx->history_count++] = token;
    return QWN_SPEC_OK;
}

static int spec_replay(QwnDecoder *decoder, const int *history, int count,
                       const float **logits) {
    qwn_decoder_reset(decoder);
    for (int i = 0; i < count; i++)
        if (qwn_decoder_forward(decoder, history[i], logits) != 0) return -1;
    return 0;
}

static void spec_emit(const QwnDecoder *decoder, int token,
                      void (*callback)(const char *, int, void *), void *opaque) {
    if (!callback || token < 0 || token >= decoder->tokenizer.n_ids) return;
    if (decoder->tokenizer.id2str && decoder->tokenizer.id2str[token]) {
        const char *text = decoder->tokenizer.id2str[token];
        callback(text, (int)strlen(text), opaque);
    }
}

int qwn_speculative_generate(QwnSpecContext *ctx, const int *prompt,
                             int prompt_count, int max_new_tokens,
                             float temperature, float top_p,
                             void (*callback)(const char *, int, void *),
                             void *opaque) {
    if (!ctx || !ctx->target || !ctx->draft || !prompt || prompt_count <= 0 ||
        max_new_tokens <= 0 || top_p <= 0.0f || top_p > 1.0f)
        return QWN_SPEC_INVALID_ARGUMENT;
    /* Top-p truncation must be applied identically to both distributions.
     * Until the shared sampler contract is exposed, fail closed. */
    if (top_p < 0.999999f) return QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL;
    ctx->temperature = temperature;
    ctx->top_p = top_p;
    ctx->history_count = 0;
    const float *target_logits = NULL, *draft_logits = NULL;
    for (int i = 0; i < prompt_count; i++) {
        if (spec_append_history(ctx, prompt[i]) != QWN_SPEC_OK ||
            qwn_decoder_forward(ctx->target, prompt[i], &target_logits) != 0 ||
            qwn_decoder_forward(ctx->draft, prompt[i], &draft_logits) != 0)
            return QWN_SPEC_CONTEXT_LIMIT;
    }
    int generated = 0;
    int draft_tokens[QWN_SPEC_MAX_GAMMA];
    int accepted_tokens[QWN_SPEC_MAX_GAMMA + 1];
    while (generated < max_new_tokens) {
        int lookahead = ctx->gamma < max_new_tokens - generated ?
                        ctx->gamma : max_new_tokens - generated;
        int vocab = ctx->target->cfg.vocab;
        float *draft_all = ctx->draft_all;
        float *draft_probs = ctx->draft_probs;
        float *target_probs = ctx->target_probs;
        double draft_start = spec_now();
        for (int k = 0; k < lookahead; k++) {
            spec_softmax(draft_logits, draft_all + (size_t)k * (size_t)vocab,
                         vocab, temperature);
            memcpy(draft_probs, draft_all + (size_t)k * (size_t)vocab,
                   (size_t)vocab * sizeof(float));
            draft_tokens[k] = spec_sample(ctx, draft_probs, vocab);
            ctx->counters.proposed_tokens++;
            if (qwn_decoder_forward(ctx->draft, draft_tokens[k], &draft_logits) != 0) {
                return -1;
            }
        }
        ctx->counters.draft_ms += (spec_now() - draft_start) * 1000.0;
        double verify_start = spec_now();
        int accepted = 0, rejected = 0;
        for (int k = 0; k < lookahead; k++) {
            spec_softmax(target_logits, target_probs, vocab, temperature);
            float p = target_probs[draft_tokens[k]];
            float q = draft_all[(size_t)k * (size_t)vocab + draft_tokens[k]];
            float accept_probability = q > 0.0f ? fminf(1.0f, p / q) :
                                       (p > 0.0f ? 1.0f : 0.0f);
            if (spec_random(ctx) <= accept_probability) {
                accepted_tokens[accepted++] = draft_tokens[k];
                ctx->counters.accepted_tokens++;
                ctx->counters.committed_tokens++;
                if (spec_append_history(ctx, draft_tokens[k]) != QWN_SPEC_OK ||
                    qwn_decoder_forward(ctx->target, draft_tokens[k], &target_logits) != 0) {
                    return -1;
                }
            } else {
                rejected = 1;
                ctx->counters.rejected_tokens++;
                double correction_start = spec_now();
                float residual_sum = 0.0f;
                for (int v = 0; v < vocab; v++) {
                    float qv = draft_all[(size_t)k * (size_t)vocab + v];
                    target_probs[v] = fmaxf(0.0f, target_probs[v] - qv);
                    residual_sum += target_probs[v];
                }
                if (residual_sum <= 0.0f) {
                    memset(target_probs, 0, (size_t)vocab * sizeof(float));
                    memcpy(target_probs, draft_all + (size_t)k * (size_t)vocab,
                           (size_t)vocab * sizeof(float));
                } else {
                    for (int v = 0; v < vocab; v++) target_probs[v] /= residual_sum;
                }
                int correction = spec_sample(ctx, target_probs, vocab);
                ctx->counters.correction_sampling_ms +=
                    (spec_now() - correction_start) * 1000.0;
                accepted_tokens[accepted++] = correction;
                ctx->counters.bonus_tokens++;
                ctx->counters.committed_tokens++;
                if (spec_append_history(ctx, correction) != QWN_SPEC_OK ||
                    qwn_decoder_forward(ctx->target, correction, &target_logits) != 0) {
                    return -1;
                }
                break;
            }
        }
        ctx->counters.target_passes++;
        ctx->counters.verification_ms += (spec_now() - verify_start) * 1000.0;
        if (rejected) {
            double rollback_start = spec_now();
            if (spec_replay(ctx->draft, ctx->history, ctx->history_count,
                            &draft_logits) != 0) {
                    return -1;
            }
            ctx->counters.rollback_ms += (spec_now() - rollback_start) * 1000.0;
        }
        /* If every proposal was accepted, sample the target's next-token
         * distribution as the standard speculative bonus token.  The token
         * is committed to both models so their KV positions stay aligned. */
        if (!rejected && accepted == lookahead &&
            generated + accepted < max_new_tokens) {
            spec_softmax(target_logits, target_probs, vocab, temperature);
            const int bonus = spec_sample(ctx, target_probs, vocab);
            accepted_tokens[accepted++] = bonus;
            ctx->counters.bonus_tokens++;
            ctx->counters.committed_tokens++;
            if (spec_append_history(ctx, bonus) != QWN_SPEC_OK ||
                qwn_decoder_forward(ctx->target, bonus, &target_logits) != 0 ||
                qwn_decoder_forward(ctx->draft, bonus, &draft_logits) != 0) {
                return -1;
            }
        }
        for (int i = 0; i < accepted && generated < max_new_tokens; i++) {
            int token = accepted_tokens[i];
            generated++;
            spec_emit(ctx->target, token, callback, opaque);
            if (token == ctx->target->cfg.eos_id) {
                return generated;
            }
        }
    }
    ctx->total_drafted = (int)ctx->counters.proposed_tokens;
    ctx->total_accepted = (int)ctx->counters.accepted_tokens;
    ctx->counters.acceptance_rate = ctx->counters.proposed_tokens > 0 ?
        (double)ctx->counters.accepted_tokens / (double)ctx->counters.proposed_tokens : 0.0;
    ctx->counters.effective_tokens_per_target_pass =
        ctx->counters.target_passes > 0
            ? (double)ctx->counters.committed_tokens /
                  (double)ctx->counters.target_passes
            : 0.0;
    return generated;
}

float qwn_speculative_acceptance_rate(const QwnSpecContext *ctx) {
    return ctx ? (float)ctx->counters.acceptance_rate : 0.0f;
}

const QwnSpecCounters *qwn_speculative_counters(const QwnSpecContext *ctx) {
    return ctx ? &ctx->counters : NULL;
}
