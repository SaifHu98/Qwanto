#ifndef QWANTO_SAGURO_H
#define QWANTO_SAGURO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Saguaro 2.0: Multi-Model (PyramidSD) & Multi-Modal (DREAM) Speculative Decoding
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

/* Compute current empirical speedup factor */
float qwn_saguaro2_measured_speedup(const QwnSaguaro2Engine *engine);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_SAGURO_H */
