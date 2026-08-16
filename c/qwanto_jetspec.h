#ifndef QWANTO_JETSPEC_H
#define QWANTO_JETSPEC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * JetSpec: Causal Parallel Tree Drafting Engine (UC San Diego Hao AI Lab 2026)
 * Achieves up to 9.64x speculative speedup (2x over Saguaro 2.0) using
 * single-pass causal parallel tree generation and tree-causal mask verification.
 * ------------------------------------------------------------------------- */

#define JETSPEC_MAX_TREE_NODES 64
#define JETSPEC_MAX_TREE_DEPTH 8
#define JETSPEC_RING_BUFFER_SLOTS 32
#define JETSPEC_CACHE_SIZE 1024

/* -------------------------------------------------------------------------
 * Tree Node & Mask Structures
 * ------------------------------------------------------------------------- */
typedef struct {
    int node_id;
    int token_id;
    int parent_id;
    int depth;
    float cumulative_score;
    int child_count;
    int children[8];
} QwnJetSpecNode;

typedef struct {
    QwnJetSpecNode nodes[JETSPEC_MAX_TREE_NODES];
    int node_count;
    int best_path_length;
    int best_path_tokens[JETSPEC_MAX_TREE_DEPTH];
    float best_path_score;
    
    /* Tree-Causal Attention Mask (Flat Matrix [node_count x node_count]) */
    uint8_t tree_mask[JETSPEC_MAX_TREE_NODES * JETSPEC_MAX_TREE_NODES];
} QwnJetSpecTree;

typedef struct {
    uint64_t prompt_hash;
    int token_chain[JETSPEC_MAX_TREE_DEPTH];
    int chain_len;
    float confidence;
} QwnJetSpecCacheEntry;

typedef struct {
    int hidden_dim;
    int num_draft_heads;
    int max_tree_width;
    int max_tree_depth;
    float top_p_threshold;
    float branch_confidence_cutoff;
} QwnJetSpecConfig;

typedef struct {
    QwnJetSpecConfig cfg;
    QwnJetSpecTree active_tree;
    QwnJetSpecCacheEntry cache[JETSPEC_CACHE_SIZE];
    
    /* 32-Slot Speculation Ring Buffer */
    int ring_buffer[JETSPEC_RING_BUFFER_SLOTS];
    int ring_head;
    int ring_tail;
    
    /* Metrics & Telemetry */
    uint64_t total_draft_tokens;
    uint64_t total_accepted_tokens;
    double measured_acceptance_rate;
    double measured_speedup_factor;
    bool is_initialized;
} QwnJetSpecEngine;

/* -------------------------------------------------------------------------
 * JetSpec Engine APIs
 * ------------------------------------------------------------------------- */

/* Initialize JetSpec Causal Parallel Tree Drafting Engine */
bool qwn_jetspec_init(
    QwnJetSpecEngine *engine,
    int hidden_dim,
    int max_tree_depth,
    int max_tree_width
);

/* Generate a Scored Candidate Token Tree in a Single Forward Pass */
bool qwn_jetspec_generate_tree(
    QwnJetSpecEngine *engine,
    const float *fused_hidden_states, /* [hidden_dim] */
    const float *target_logits,        /* [vocab_size] */
    int vocab_size
);

/* Build Tree-Causal 2D Attention Verification Mask */
void qwn_jetspec_build_tree_mask(QwnJetSpecTree *tree);

/* Verify Candidate Token Tree against Target Model Distribution */
int qwn_jetspec_verify_tree(
    QwnJetSpecEngine *engine,
    const float *target_verification_logits, /* [node_count x vocab_size] */
    int vocab_size,
    int *accepted_tokens_out,
    int max_out_tokens
);

/* Query 64-bit FNV-1a Speculation Cache */
bool qwn_jetspec_cache_lookup(
    QwnJetSpecEngine *engine,
    uint64_t prompt_hash,
    int *tokens_out,
    int *chain_len_out
);

/* Insert Verified Path into Cache and Ring Buffer */
void qwn_jetspec_record_acceptance(
    QwnJetSpecEngine *engine,
    uint64_t prompt_hash,
    const int *accepted_tokens,
    int accepted_count
);

/* Reset active speculation tree and statistics */
void qwn_jetspec_reset(QwnJetSpecEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_JETSPEC_H */
