#ifndef QWANTO_SAGURO_H
#define QWANTO_SAGURO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Legacy Saguaro compatibility API.
 *
 * This surface is retained for source compatibility with older experiments.
 * It is not the product speculative-decoding path and it must not manufacture
 * acceptance or speedup measurements.
 * ------------------------------------------------------------------------- */

#define QWN_SAGUARO_RING_BUFFER_SIZE 32
#define QWN_SAGUARO_MAX_TIERS 3
#define QWN_SAGUARO_MAX_MODALITIES 2 /* Text + Vision */

typedef enum {
    QWN_SPEC_TIER_1_LIGHTWEIGHT = 0, /* Ultra-compact draft model (e.g. 0.5B) */
    QWN_SPEC_TIER_2_INTERMEDIATE= 1, /* Intermediate verification model (e.g. 1.5B) */
    QWN_SPEC_TIER_3_TARGET      = 2  /* Final full-parameter target model (e.g. 4B/70B) */
} QwnSpecTier;

typedef struct {
    int token_id;
    float confidence;
    int modality;                    /* 0: Text, 1: Vision embedding */
    bool accepted;
} QwnDraftToken;

typedef struct {
    QwnDraftToken ring_buffer[QWN_SAGUARO_RING_BUFFER_SIZE];
    int head;
    int tail;
    int current_draft_len;           /* Adaptive gamma (3..15) */
    float tier_acceptance[QWN_SAGUARO_MAX_TIERS];
    float entropy_threshold;         /* Entropy-adaptive cross-attention threshold */
    uint64_t total_speculated;
    uint64_t total_accepted;
    float baseline_tok_per_sec;
    float speculative_tok_per_sec;
    bool multimodal_enabled;
} QwnSaguaro2Engine;

/* -------------------------------------------------------------------------
 * Saguaro 2.0 APIs
 * ------------------------------------------------------------------------- */

/* Initialize Saguaro 2.0 multi-tier speculative engine */
int qwn_saguaro2_init(
    QwnSaguaro2Engine *engine,
    int initial_draft_length,
    bool enable_multimodal
);

/* Push draft token from lightweight Tier-1 model into ring buffer */
int qwn_saguaro2_push_draft(
    QwnSaguaro2Engine *engine,
    int token_id,
    float confidence,
    int modality
);

/* Run PyramidSD multi-tier verification pass */
int qwn_saguaro2_verify_pyramid(
    QwnSaguaro2Engine *engine,
    const int *target_logits_top1,
    int n_tokens_to_verify,
    int *out_accepted_tokens,
    int *out_accepted_count
);

/* Return a speedup only after a caller records real paired measurements. */
float qwn_saguaro2_measured_speedup(const QwnSaguaro2Engine *engine);

/* Record externally measured baseline/speculative rates.  This does not run
 * speculation; it only makes an independently measured result queryable. */
int qwn_saguaro2_record_measurement(QwnSaguaro2Engine *engine,
                                    float baseline_tok_per_sec,
                                    float speculative_tok_per_sec);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_SAGURO_H */
