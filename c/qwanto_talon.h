#ifndef QWANTO_TALON_H
#define QWANTO_TALON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Talon: Asynchronous Speculative Decoding & Hybrid Drafting (AAAI 2026)
 * Breaks the synchronous waiting barrier by decoupling drafting from verification,
 * dynamically combining model-based and retrieval-based drafting for 4.04x-6.52x speedup.
 * ------------------------------------------------------------------------- */

typedef enum {
    QWN_TALON_DRAFT_AUTO      = 0,
    QWN_TALON_DRAFT_MODEL     = 1, /* Neural draft head for reasoning/code */
    QWN_TALON_DRAFT_RETRIEVAL = 2, /* Sub-vector / Trie retrieval for knowledge */
    QWN_TALON_DRAFT_HYBRID    = 3  /* Joint model + retrieval concurrent fusion */
} QwnTalonDraftStrategy;

typedef enum {
    QWN_TALON_DOMAIN_CODE        = 0,
    QWN_TALON_DOMAIN_REASONING   = 1,
    QWN_TALON_DOMAIN_KNOWLEDGE   = 2,
    QWN_TALON_DOMAIN_CONVERSATION= 3,
    QWN_TALON_DOMAIN_SUMMARIZE   = 4
} QwnTalonDomain;

#define TALON_QUEUE_CAPACITY 64
#define TALON_RETRIEVAL_TRIE_SIZE 4096

typedef struct {
    int token_chain[16];
    int chain_length;
    float confidence;
    QwnTalonDraftStrategy strategy_used;
    uint64_t timestamp_us;
} QwnTalonDraftItem;

typedef struct {
    QwnTalonDraftItem items[TALON_QUEUE_CAPACITY];
    int head;
    int tail;
    int count;
} QwnTalonAsyncQueue;

typedef struct {
    uint64_t key_hash;
    int candidate_tokens[8];
    int length;
    uint32_t frequency;
} QwnTalonTrieNode;

typedef struct {
    QwnTalonDraftStrategy strategy;
    QwnTalonDomain detected_domain;
    QwnTalonAsyncQueue draft_queue;
    QwnTalonTrieNode retrieval_trie[TALON_RETRIEVAL_TRIE_SIZE];
    
    int max_draft_length;
    float retrieval_threshold;
    bool is_async_active;
    
    /* Telemetry & Acceleration Metrics */
    uint64_t total_drafts_issued;
    uint64_t total_tokens_accepted;
    double measured_acceptance_rate;
    double measured_speedup;
    bool is_initialized;
} QwnTalonEngine;

/* -------------------------------------------------------------------------
 * Talon Engine APIs
 * ------------------------------------------------------------------------- */

/* Initialize Talon Asynchronous Speculative Engine */
bool qwn_talon_init(QwnTalonEngine *engine, int max_draft_length);

/* Domain Classifier: Analyzes prompt context and selects optimal drafting mode */
QwnTalonDomain qwn_talon_classify_domain(const char *prompt_or_context);

/* Enqueue Asynchronous Draft Generation (Model / Retrieval / Hybrid) */
bool qwn_talon_async_draft_step(
    QwnTalonEngine *engine,
    const char *prompt_or_context,
    const float *last_hidden_state,
    int hidden_dim,
    int vocab_size
);

/* Asynchronous Verification Step: Verifies draft queue against target model */
int qwn_talon_async_verify_step(
    QwnTalonEngine *engine,
    const float *target_logits,
    int vocab_size,
    int *accepted_tokens_out,
    int max_out_tokens
);

/* Index text context into Retrieval Trie for rapid Sub-Vector Lookup */
void qwn_talon_index_knowledge(
    QwnTalonEngine *engine,
    const int *token_sequence,
    int seq_len
);

/* Clean up Talon asynchronous engine */
void qwn_talon_shutdown(QwnTalonEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_TALON_H */
