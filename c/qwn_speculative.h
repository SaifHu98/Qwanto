#ifndef QWN_SPECULATIVE_H
#define QWN_SPECULATIVE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "qwanto_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QWN_SPEC_MAX_GAMMA 16
#define QWN_SPEC_MAX_DRAFT 16
#define QWN_SPEC_RING_SIZE 32
#define QWN_SPEC_DEFAULT_CACHE_SIZE 256

/* Compatibility cache used by older callers.  It is an ordinary LRU for
 * measured draft prefixes; it is not a speculative execution result and has
 * no optimistic initial acceptance value. */
typedef struct {
    uint64_t hash;
    int draft_tokens[QWN_SPEC_MAX_DRAFT];
    float draft_probabilities[QWN_SPEC_MAX_DRAFT];
    int token_count;
    uint64_t lru_timestamp;
    int accepted_count;
} SpeculationCacheEntry;

typedef struct {
    SpeculationCacheEntry *entries;
    int capacity;
    int count;
    float acceptance_rate;
    uint64_t current_lru_clock;
    uint64_t total_lookups;
    uint64_t total_hits;
    uint64_t total_drafted;
    uint64_t total_accepted;
} SpeculationCache;

typedef struct {
    QwnDecoder *target_decoder;
    QwnDecoder *draft_decoder;
    SpeculationCache cache;
    int max_draft_tokens;
    float min_acceptance_rate;
    bool use_bidirectional;
    int ring_buffer[QWN_SPEC_RING_SIZE];
    int ring_head;
    int ring_tail;
    int ring_count;
    float token_throughput;
    float acceptance_rate_avg;
    uint64_t speculation_hits;
    uint64_t verification_calls;
} QwnSpeculativeEngine;

typedef enum {
    QWN_SPEC_OK = 0,
    QWN_SPEC_INVALID_ARGUMENT = -1,
    QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL = -2,
    QWN_SPEC_CONTEXT_LIMIT = -3
} QwnSpecStatus;

typedef struct {
    uint64_t proposed_tokens;
    uint64_t accepted_tokens;
    uint64_t rejected_tokens;
    uint64_t bonus_tokens;
    uint64_t target_passes;
    uint64_t committed_tokens;
    double effective_tokens_per_target_pass;
    double acceptance_rate;
    double draft_ms;
    double verification_ms;
    double rollback_ms;
    double correction_sampling_ms;
    double baseline_tok_per_sec;
    double speculative_tok_per_sec;
    double net_speedup;
    char status[64];
} QwnSpecCounters;

typedef struct {
    QwnDecoder *target;
    QwnDecoder *draft;
    int native_mtp;
    int gamma;
    int total_drafted;
    int total_accepted;
    float temperature;
    float top_p;
    uint64_t rng_state;
    int *history;
    int history_count;
    int history_capacity;
    float *draft_all;
    float *draft_probs;
    float *target_probs;
    size_t draft_all_capacity;
    size_t probs_capacity;
    float *mtp_state_snapshots;
    size_t mtp_snapshot_capacity;
    QwnSpecCounters counters;
} QwnSpecContext;

int qwn_speculative_check_compatibility(const QwnDecoder *target,
                                        const QwnDecoder *draft,
                                        char *reason, size_t reason_size);
int qwn_speculative_init(QwnSpecContext *ctx, QwnDecoder *target,
                         QwnDecoder *draft, int gamma);
/* Native Qwen NextN/MTP draft path. It has no external draft model and uses
 * transactional MTP-state checkpoints for each proposed token. */
int qwn_speculative_mtp_init(QwnSpecContext *ctx, QwnDecoder *target,
                             int gamma);
void qwn_speculative_free(QwnSpecContext *ctx);
int qwn_speculative_generate(QwnSpecContext *ctx,
                             const int *prompt, int prompt_count,
                             int max_new_tokens, float temperature, float top_p,
                             void (*callback)(const char *, int, void *),
                             void *opaque);
float qwn_speculative_acceptance_rate(const QwnSpecContext *ctx);
const QwnSpecCounters *qwn_speculative_counters(const QwnSpecContext *ctx);

/* Legacy cache/ring symbols remain link-compatible, but execution through
 * this API is deliberately disabled until a compatible native QWN draft is
 * available. */
int qwn_speculative_engine_init(QwnSpeculativeEngine *engine,
                                QwnDecoder *target, QwnDecoder *draft,
                                int cache_capacity);
void qwn_speculative_engine_free(QwnSpeculativeEngine *engine);
int get_optimal_draft_length(const SpeculationCache *cache);
uint64_t qwn_speculative_hash_context(const int *tokens, int count);
int qwn_speculative_cache_lookup(SpeculationCache *cache, uint64_t hash,
                                 int *out_tokens, float *out_probs, int max_len);
void qwn_speculative_cache_insert(SpeculationCache *cache, uint64_t hash,
                                  const int *tokens, const float *probs, int len);
void qwn_speculative_cache_update_rate(SpeculationCache *cache, int drafted,
                                       int accepted);
int qwn_spec_ring_push(QwnSpeculativeEngine *engine, int token);
int qwn_spec_ring_pop(QwnSpeculativeEngine *engine, int *out_token);
void qwn_spec_ring_clear(QwnSpeculativeEngine *engine);
void qwn_generate_draft_sequence(QwnDecoder *draft_decoder, const int *context,
                                 int context_len, int *draft_tokens,
                                 float *draft_probs, int n_draft_tokens);
int qwn_verify_and_accept(QwnSpeculativeEngine *engine, const int *draft_tokens,
                          int n_draft_tokens, const float *draft_probs,
                          int *accepted_tokens, float *target_probs);
int qwn_speculative_forward(QwnSpeculativeEngine *engine, const int *prompt,
                            int prompt_len, int *output, int max_tokens,
                            float temperature);
int qwn_speculative_generate_stream(QwnSpeculativeEngine *engine,
                                     const int *prompt, int prompt_len,
                                     int max_tokens, float temperature,
                                     float top_p,
                                     void (*callback)(const char *, int, void *),
                                     void *opaque);

#ifdef __cplusplus
}
#endif

#endif /* QWN_SPECULATIVE_H */
