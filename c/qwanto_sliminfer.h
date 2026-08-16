#ifndef QWANTO_SLIMINFER_H
#define QWANTO_SLIMINFER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * SlimInfer: Dynamic Token Pruning & Long-Context TTFT Acceleration (AAAI 2026)
 * Prunes redundant intermediate-layer tokens based on information diffusion dynamics,
 * achieving 2.53x TTFT speedup and 1.88x end-to-end latency reduction on long contexts.
 * ------------------------------------------------------------------------- */

typedef struct {
    int start_prune_layer;     /* Layer index where pruning begins (e.g. Layer 4) */
    int prune_interval;        /* Layers between pruning steps */
    float target_retention;    /* Percentage of tokens retained (e.g. 0.50 for 50%) */
    int min_retained_tokens;   /* Minimum number of tokens to keep */
    int sink_token_count;      /* Number of initial attention sink tokens preserved */
    bool preserve_recent;      /* Always retain the last N recent tokens */
    int recent_token_count;    /* Number of recent tokens preserved */
} QwnSlimInferConfig;

typedef struct {
    QwnSlimInferConfig cfg;
    int *retained_indices;     /* Array of active retained token indices */
    int retained_count;
    int original_seq_len;
    float *importance_scores;  /* Accumulated salience scores per token */
    
    /* Telemetry & Performance Metrics */
    double measured_ttft_speedup;
    double memory_saved_ratio;
    uint64_t total_tokens_processed;
    uint64_t total_tokens_pruned;
    bool is_initialized;
} QwnSlimInferEngine;

/* -------------------------------------------------------------------------
 * SlimInfer APIs
 * ------------------------------------------------------------------------- */

/* Initialize SlimInfer Engine */
bool qwn_sliminfer_init(
    QwnSlimInferEngine *engine,
    int max_seq_len,
    int start_prune_layer,
    float target_retention
);

/* Compute Dynamic Token Importance Scores from Attention Matrix */
void qwn_sliminfer_compute_salience(
    QwnSlimInferEngine *engine,
    const float *attn_weights, /* [n_heads, seq_len, seq_len] */
    int n_heads,
    int seq_len
);

/* Apply Dynamic Fine-Grained Token Pruning to Hidden States */
int qwn_sliminfer_prune_hidden_states(
    QwnSlimInferEngine *engine,
    const float *in_hidden_states,   /* [seq_len, hidden_dim] */
    float *out_compact_states,       /* [retained_count, hidden_dim] */
    int seq_len,
    int hidden_dim,
    int current_layer_idx
);

/* Gather / Restore Pruned Context into Full Output Projection */
void qwn_sliminfer_scatter_output(
    const QwnSlimInferEngine *engine,
    const float *compact_output,
    float *full_output,
    int hidden_dim
);

/* Reset SlimInfer Engine for a New Sequence */
void qwn_sliminfer_reset(QwnSlimInferEngine *engine, int new_seq_len);

/* Free SlimInfer Resources */
void qwn_sliminfer_free(QwnSlimInferEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_SLIMINFER_H */
