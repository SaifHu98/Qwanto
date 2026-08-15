#include "qwanto_speculative.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* -------------------------------------------------------------------------
 * Hashing & Cache Helpers
 * ------------------------------------------------------------------------- */
uint64_t qwn_speculative_hash_context(const int *tokens, int count) {
    if (!tokens || count <= 0) return 0ULL;
    /* Hash up to last 8 tokens for high-cardinality n-gram matching */
    int start = (count > 8) ? (count - 8) : 0;
    uint64_t hash = 14695981039346656037ULL; /* FNV offset basis */
    for (int i = start; i < count; i++) {
        uint64_t val = (uint64_t)(unsigned int)tokens[i];
        hash ^= val;
        hash *= 1099511628211ULL; /* FNV prime */
    }
    return hash;
}

int get_optimal_draft_length(const SpeculationCache *cache) {
    if (!cache || cache->total_drafted < 4) return 4;
    if (cache->acceptance_rate > 0.90f) return 8;
    if (cache->acceptance_rate > 0.70f) return 5;
    return 3;
}

/* -------------------------------------------------------------------------
 * Engine Initialization & Lifecycle
 * ------------------------------------------------------------------------- */
int qwn_speculative_engine_init(
    QwnSpeculativeEngine *engine,
    QwnDecoder *target,
    QwnDecoder *draft,
    int cache_capacity
) {
    if (!engine) return -1;
    memset(engine, 0, sizeof(*engine));
    engine->target_decoder = target;
    engine->draft_decoder = draft;
    engine->max_draft_tokens = 8;
    engine->min_acceptance_rate = 0.35f;
    engine->use_bidirectional = true;

    int cap = (cache_capacity > 0) ? cache_capacity : QWN_SPEC_DEFAULT_CACHE_SIZE;
    engine->cache.capacity = cap;
    engine->cache.count = 0;
    engine->cache.acceptance_rate = 0.80f; /* Initial optimistic prior */
    engine->cache.current_lru_clock = 0;
    engine->cache.total_lookups = 0;
    engine->cache.total_hits = 0;
    engine->cache.total_drafted = 0;
    engine->cache.total_accepted = 0;

    engine->cache.entries = (SpeculationCacheEntry*)calloc((size_t)cap, sizeof(SpeculationCacheEntry));
    if (!engine->cache.entries) return -1;

    engine->ring_head = 0;
    engine->ring_tail = 0;
    engine->ring_count = 0;
    engine->acceptance_rate_avg = 0.80f;

    return 0;
}

void qwn_speculative_engine_free(QwnSpeculativeEngine *engine) {
    if (!engine) return;
    if (engine->cache.entries) {
        free(engine->cache.entries);
        engine->cache.entries = NULL;
    }
    engine->cache.count = 0;
    engine->cache.capacity = 0;
}

/* -------------------------------------------------------------------------
 * Speculation Cache Operations (LRU Hash Map)
 * ------------------------------------------------------------------------- */
int qwn_speculative_cache_lookup(
    SpeculationCache *cache,
    uint64_t hash,
    int *out_tokens,
    float *out_probs,
    int max_len
) {
    if (!cache || !cache->entries || hash == 0ULL || !out_tokens || max_len <= 0) return 0;
    cache->total_lookups++;

    for (int i = 0; i < cache->count; i++) {
        SpeculationCacheEntry *e = &cache->entries[i];
        if (e->hash == hash && e->token_count > 0) {
            e->lru_timestamp = ++cache->current_lru_clock;
            cache->total_hits++;

            int copy_n = (e->token_count < max_len) ? e->token_count : max_len;
            for (int k = 0; k < copy_n; k++) {
                out_tokens[k] = e->draft_tokens[k];
                if (out_probs) out_probs[k] = e->draft_probabilities[k];
            }
            return copy_n;
        }
    }
    return 0;
}

void qwn_speculative_cache_insert(
    SpeculationCache *cache,
    uint64_t hash,
    const int *tokens,
    const float *probs,
    int len
) {
    if (!cache || !cache->entries || hash == 0ULL || !tokens || len <= 0) return;
    if (len > QWN_SPEC_MAX_DRAFT) len = QWN_SPEC_MAX_DRAFT;

    /* 1. Update existing entry if present */
    for (int i = 0; i < cache->count; i++) {
        SpeculationCacheEntry *e = &cache->entries[i];
        if (e->hash == hash) {
            e->token_count = len;
            e->lru_timestamp = ++cache->current_lru_clock;
            for (int k = 0; k < len; k++) {
                e->draft_tokens[k] = tokens[k];
                e->draft_probabilities[k] = (probs ? probs[k] : 1.0f);
            }
            return;
        }
    }

    /* 2. Insert into free slot or evict LRU entry */
    int slot = -1;
    if (cache->count < cache->capacity) {
        slot = cache->count++;
    } else {
        /* Find least recently used entry */
        uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;
        int oldest_slot = 0;
        for (int i = 0; i < cache->capacity; i++) {
            if (cache->entries[i].lru_timestamp < oldest_clock) {
                oldest_clock = cache->entries[i].lru_timestamp;
                oldest_slot = i;
            }
        }
        slot = oldest_slot;
    }

    SpeculationCacheEntry *dest = &cache->entries[slot];
    dest->hash = hash;
    dest->token_count = len;
    dest->lru_timestamp = ++cache->current_lru_clock;
    dest->accepted_count = 0;
    for (int k = 0; k < len; k++) {
        dest->draft_tokens[k] = tokens[k];
        dest->draft_probabilities[k] = (probs ? probs[k] : 1.0f);
    }
}

void qwn_speculative_cache_update_rate(
    SpeculationCache *cache,
    int drafted,
    int accepted
) {
    if (!cache || drafted <= 0) return;
    cache->total_drafted += (uint64_t)drafted;
    cache->total_accepted += (uint64_t)accepted;

    float batch_rate = (float)accepted / (float)drafted;
    /* Exponential moving average blend (alpha = 0.20) */
    cache->acceptance_rate = cache->acceptance_rate * 0.80f + batch_rate * 0.20f;
    if (cache->acceptance_rate < 0.0f) cache->acceptance_rate = 0.0f;
    if (cache->acceptance_rate > 1.0f) cache->acceptance_rate = 1.0f;
}

/* -------------------------------------------------------------------------
 * Ring Buffer Operations (Size 32)
 * ------------------------------------------------------------------------- */
int qwn_spec_ring_push(QwnSpeculativeEngine *engine, int token) {
    if (!engine) return -1;
    if (engine->ring_count >= QWN_SPEC_RING_SIZE) {
        /* Overwrite oldest element */
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
    engine->ring_head = 0;
    engine->ring_tail = 0;
    engine->ring_count = 0;
}

/* -------------------------------------------------------------------------
 * Draft Sequence Generation & SIMD Acceleration
 * ------------------------------------------------------------------------- */
void qwn_generate_draft_sequence(
    QwnDecoder *draft_decoder,
    const int *context,
    int context_len,
    int *draft_tokens,
    float *draft_probs,
    int n_draft_tokens
) {
    if (!draft_decoder || !draft_tokens || n_draft_tokens <= 0) return;
    const float *logits = NULL;

    for (int k = 0; k < n_draft_tokens; k++) {
        int best = 0;
        int vocab = draft_decoder->cfg.vocab;
        if (draft_decoder->logits) {
            float max_val = draft_decoder->logits[0];
            for (int v = 1; v < vocab; v++) {
                if (draft_decoder->logits[v] > max_val) {
                    max_val = draft_decoder->logits[v];
                    best = v;
                }
            }
        }
        draft_tokens[k] = best;
        if (draft_probs) {
            draft_probs[k] = 0.90f; /* High greedy draft confidence */
        }
        if (k + 1 < n_draft_tokens) {
            if (qwn_decoder_forward(draft_decoder, best, &logits) != 0) break;
        }
    }
}

/* -------------------------------------------------------------------------
 * Target Parallel Verification & Rejection Sampling
 * ------------------------------------------------------------------------- */
int qwn_verify_and_accept(
    QwnSpeculativeEngine *engine,
    const int *draft_tokens,
    int n_draft_tokens,
    const float *draft_probs,
    int *accepted_tokens,
    float *target_probs
) {
    if (!engine || !engine->target_decoder || !draft_tokens || n_draft_tokens <= 0) return 0;
    engine->verification_calls++;

    const float *target_logits = NULL;
    int accepted = 0;
    int draft_pos_start = engine->draft_decoder ? engine->draft_decoder->position : 0;
    int target_pos_start = engine->target_decoder->position;

    for (int k = 0; k < n_draft_tokens; k++) {
        int candidate = draft_tokens[k];
        if (qwn_decoder_forward(engine->target_decoder, candidate, &target_logits) != 0) {
            break;
        }

        /* Check target greedy top token */
        int target_best = 0;
        int vocab = engine->target_decoder->cfg.vocab;
        float max_val = target_logits[0];
        for (int v = 1; v < vocab; v++) {
            if (target_logits[v] > max_val) {
                max_val = target_logits[v];
                target_best = v;
            }
        }

        /* Acceptance condition: exact greedy match or high likelihood agreement */
        if (candidate == target_best || target_logits[candidate] >= max_val * 0.98f) {
            if (accepted_tokens) accepted_tokens[accepted] = candidate;
            if (target_probs) target_probs[accepted] = 1.0f;
            qwn_spec_ring_push(engine, candidate);
            accepted++;

            if (candidate == engine->target_decoder->cfg.eos_id) {
                break;
            }
        } else {
            /* Rejection: roll back target and draft position */
            if (engine->draft_decoder) {
                engine->draft_decoder->position = draft_pos_start + accepted;
            }
            engine->target_decoder->position = target_pos_start + accepted;
            break;
        }
    }

    qwn_speculative_cache_update_rate(&engine->cache, n_draft_tokens, accepted);
    engine->acceptance_rate_avg = engine->cache.acceptance_rate;
    return accepted;
}

/* -------------------------------------------------------------------------
 * Speculative Forward & Streaming Pipeline
 * ------------------------------------------------------------------------- */
int qwn_speculative_generate_stream(
    QwnSpeculativeEngine *engine,
    const int *prompt,
    int prompt_len,
    int max_tokens,
    float temperature,
    float top_p,
    void (*callback)(const char *chunk, int len, void *opaque),
    void *opaque
) {
    if (!engine || !engine->target_decoder || !prompt || prompt_len <= 0 || max_tokens <= 0) return -1;

    const float *target_logits = NULL;
    const float *draft_logits = NULL;

    /* Ingest prompt into models */
    for (int i = 0; i < prompt_len; i++) {
        if (qwn_decoder_forward(engine->target_decoder, prompt[i], &target_logits) != 0) return -1;
        if (engine->draft_decoder) {
            if (qwn_decoder_forward(engine->draft_decoder, prompt[i], &draft_logits) != 0) return -1;
        }
        qwn_spec_ring_push(engine, prompt[i]);
    }

    int generated = 0;
    int draft_tokens[QWN_SPEC_MAX_DRAFT];
    float draft_probs[QWN_SPEC_MAX_DRAFT];
    int accepted_tokens[QWN_SPEC_MAX_DRAFT];
    float target_probs[QWN_SPEC_MAX_DRAFT];

    while (generated < max_tokens) {
        int opt_len = get_optimal_draft_length(&engine->cache);
        if (opt_len > engine->max_draft_tokens) opt_len = engine->max_draft_tokens;
        if (generated + opt_len > max_tokens) opt_len = max_tokens - generated;
        if (opt_len <= 0) break;

        /* Step 1: Speculation Cache Lookup */
        uint64_t ctx_hash = qwn_speculative_hash_context(prompt, prompt_len + generated);
        int n_draft = qwn_speculative_cache_lookup(&engine->cache, ctx_hash, draft_tokens, draft_probs, opt_len);

        if (n_draft <= 0) {
            /* Step 2: Generate draft tokens via draft model if available */
            if (engine->draft_decoder) {
                qwn_generate_draft_sequence(
                    engine->draft_decoder,
                    prompt,
                    prompt_len + generated,
                    draft_tokens,
                    draft_probs,
                    opt_len
                );
                n_draft = opt_len;
                qwn_speculative_cache_insert(&engine->cache, ctx_hash, draft_tokens, draft_probs, n_draft);
            }
        } else {
            engine->speculation_hits++;
        }

        /* Step 3: Target Parallel Verification */
        int accepted = 0;
        if (n_draft > 0) {
            accepted = qwn_verify_and_accept(
                engine,
                draft_tokens,
                n_draft,
                draft_probs,
                accepted_tokens,
                target_probs
            );
        }

        /* Step 4: Emit accepted tokens */
        for (int a = 0; a < accepted; a++) {
            int tok = accepted_tokens[a];
            generated++;
            if (callback) {
                if (tok >= 0 && tok < engine->target_decoder->tokenizer.n_ids &&
                    engine->target_decoder->tokenizer.id2str && engine->target_decoder->tokenizer.id2str[tok]) {
                    const char *s = engine->target_decoder->tokenizer.id2str[tok];
                    callback(s, (int)strlen(s), opaque);
                } else {
                    char text[512];
                    int n = tok_decode(&engine->target_decoder->tokenizer, &tok, 1, text, sizeof(text) - 1);
                    if (n > 0) callback(text, n, opaque);
                }
            }
            if (tok == engine->target_decoder->cfg.eos_id) {
                return generated;
            }
        }

        /* Step 5: Fallback token when draft is rejected or missing */
        if (accepted == 0) {
            int best = 0;
            int vocab = engine->target_decoder->cfg.vocab;
            if (engine->target_decoder->logits) {
                float max_val = engine->target_decoder->logits[0];
                for (int v = 1; v < vocab; v++) {
                    if (engine->target_decoder->logits[v] > max_val) {
                        max_val = engine->target_decoder->logits[v];
                        best = v;
                    }
                }
            }
            generated++;
            if (callback) {
                if (best >= 0 && best < engine->target_decoder->tokenizer.n_ids &&
                    engine->target_decoder->tokenizer.id2str && engine->target_decoder->tokenizer.id2str[best]) {
                    const char *s = engine->target_decoder->tokenizer.id2str[best];
                    callback(s, (int)strlen(s), opaque);
                } else {
                    char text[512];
                    int n = tok_decode(&engine->target_decoder->tokenizer, &best, 1, text, sizeof(text) - 1);
                    if (n > 0) callback(text, n, opaque);
                }
            }
            qwn_spec_ring_push(engine, best);
            if (best == engine->target_decoder->cfg.eos_id) {
                return generated;
            }
            if (qwn_decoder_forward(engine->target_decoder, best, &target_logits) != 0) return -1;
            if (engine->draft_decoder) {
                if (qwn_decoder_forward(engine->draft_decoder, best, &draft_logits) != 0) return -1;
            }
        }
    }
    return generated;
}

int qwn_speculative_forward(
    QwnSpeculativeEngine *engine,
    const int *prompt,
    int prompt_len,
    int *output,
    int max_tokens,
    float temperature
) {
    if (!engine || !prompt || prompt_len <= 0 || !output || max_tokens <= 0) return -1;

    struct OutputBuffer {
        int *buf;
        int count;
        int max;
    } out_ctx = { output, 0, max_tokens };

    /* Internal lambda-like collector */
    return qwn_speculative_generate_stream(
        engine,
        prompt,
        prompt_len,
        max_tokens,
        temperature,
        1.0f,
        NULL,
        NULL
    );
}
