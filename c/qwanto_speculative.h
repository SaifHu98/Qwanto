#ifndef QWANTO_SPECULATIVE_H
#define QWANTO_SPECULATIVE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "qwanto_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QWN_SPEC_MAX_DRAFT = 16,
    QWN_SPEC_RING_SIZE = 32,
    QWN_SPEC_DEFAULT_CACHE_SIZE = 256
};

/* -------------------------------------------------------------------------
 * Speculation Cache Entry & Cache Management
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t hash;                         /* FNV-1a 64-bit context hash key */
    int draft_tokens[QWN_SPEC_MAX_DRAFT];   /* Speculated token sequence */
    float draft_probabilities[QWN_SPEC_MAX_DRAFT]; /* Per-token draft confidence */
    int token_count;                       /* Number of cached tokens in entry */
    uint64_t lru_timestamp;                /* Monotonic LRU counter */
    int accepted_count;                    /* Historical accepted count for entry */
} SpeculationCacheEntry;

typedef struct {
    SpeculationCacheEntry *entries;        /* Dynamic or pre-allocated entry array */
    int capacity;                          /* Max entries (e.g. 64, 128, 256, 512) */
    int count;                             /* Current occupied entries */
    float acceptance_rate;                 /* Observed acceptance rate [0.0, 1.0] */
    uint64_t current_lru_clock;            /* Monotonic LRU clock */
    uint64_t total_lookups;                /* Total cache lookup requests */
    uint64_t total_hits;                   /* Total cache hits */
    uint64_t total_drafted;                /* Total drafted tokens through cache/model */
    uint64_t total_accepted;               /* Total accepted tokens */
} SpeculationCache;

/* -------------------------------------------------------------------------
 * Saguaro Speculative Decoding Engine
 * ------------------------------------------------------------------------- */
typedef struct {
    QwnDecoder *target_decoder;            /* Target high-capacity / precision model */
    QwnDecoder *draft_decoder;             /* Draft fast / compressed model */
    SpeculationCache cache;                /* Speculation LRU cache */
    int max_draft_tokens;                  /* Configured ceiling for draft length (default 8) */
    float min_acceptance_rate;             /* Cutoff to fallback to target-only (default 0.35) */
    bool use_bidirectional;                /* Enable bidirectional forward/backward speculation */

    /* Speculation Ring Buffer (size 32) */
    int ring_buffer[QWN_SPEC_RING_SIZE];
    int ring_head;
    int ring_tail;
    int ring_count;

    /* Live Performance Metrics */
    float token_throughput;                /* Measured tokens per second */
    float acceptance_rate_avg;             /* Exponential moving average acceptance rate */
    uint64_t speculation_hits;             /* Number of successful speculative fast paths */
    uint64_t verification_calls;           /* Number of target verification passes */
} QwnSpeculativeEngine;

/* -------------------------------------------------------------------------
 * API Prototypes
 * ------------------------------------------------------------------------- */

/* Initialize engine, allocating cache and binding decoders */
int qwn_speculative_engine_init(
    QwnSpeculativeEngine *engine,
    QwnDecoder *target,
    QwnDecoder *draft,
    int cache_capacity
);

/* Free allocated engine cache resources */
void qwn_speculative_engine_free(QwnSpeculativeEngine *engine);

/* Dynamic adaptive draft length heuristic based on acceptance history */
int get_optimal_draft_length(const SpeculationCache *cache);

/* 64-bit FNV-1a prefix context hash */
uint64_t qwn_speculative_hash_context(const int *tokens, int count);

/* Speculation Cache operations */
int qwn_speculative_cache_lookup(
    SpeculationCache *cache,
    uint64_t hash,
    int *out_tokens,
    float *out_probs,
    int max_len
);

void qwn_speculative_cache_insert(
    SpeculationCache *cache,
    uint64_t hash,
    const int *tokens,
    const float *probs,
    int len
);

void qwn_speculative_cache_update_rate(
    SpeculationCache *cache,
    int drafted,
    int accepted
);

/* Ring buffer utility operations */
int qwn_spec_ring_push(QwnSpeculativeEngine *engine, int token);
int qwn_spec_ring_pop(QwnSpeculativeEngine *engine, int *out_token);
void qwn_spec_ring_clear(QwnSpeculativeEngine *engine);

/* Draft generation */
void qwn_generate_draft_sequence(
    QwnDecoder *draft_decoder,
    const int *context,
    int context_len,
    int *draft_tokens,
    float *draft_probs,
    int n_draft_tokens
);

/* Target parallel verification and acceptance */
int qwn_verify_and_accept(
    QwnSpeculativeEngine *engine,
    const int *draft_tokens,
    int n_draft_tokens,
    const float *draft_probs,
    int *accepted_tokens,
    float *target_probs
);

/* High-level synchronous batch generation */
int qwn_speculative_forward(
    QwnSpeculativeEngine *engine,
    const int *prompt,
    int prompt_len,
    int *output,
    int max_tokens,
    float temperature
);

/* Streaming generation with callback */
int qwn_speculative_generate_stream(
    QwnSpeculativeEngine *engine,
    const int *prompt,
    int prompt_len,
    int max_tokens,
    float temperature,
    float top_p,
    void (*callback)(const char *chunk, int len, void *opaque),
    void *opaque
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_SPECULATIVE_H */
