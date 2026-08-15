#ifndef QWANTO_PAGEDEVICTION_H
#define QWANTO_PAGEDEVICTION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * PagedEviction & vToken: Token-Level KV-Cache Virtualization (<6% Memory Waste)
 * ------------------------------------------------------------------------- */

#define QWN_VTOKEN_MAX_CAPACITY 32768
#define QWN_VTOKEN_SINK_TOKENS 4

typedef struct {
    uint32_t token_id;      /* Global sequence index */
    float importance_score; /* Dynamic exponential moving average attention score */
    uint32_t physical_slot; /* Hardware arena index */
    bool is_sink;           /* Protected attention sink token */
    bool is_active;         /* Token is currently referenced */
} QwnVToken;

typedef struct {
    QwnVToken virtual_tokens[QWN_VTOKEN_MAX_CAPACITY];
    uint32_t active_count;
    uint32_t max_capacity;
    uint32_t physical_pool_size;
    float eviction_threshold;
    uint64_t total_evictions;
} QwnPagedEvictionPool;

/* -------------------------------------------------------------------------
 * PagedEviction & vToken APIs
 * ------------------------------------------------------------------------- */

/* Initialize vToken virtualization pool */
int qwn_paged_eviction_init(
    QwnPagedEvictionPool *pool,
    uint32_t max_capacity,
    uint32_t physical_pool_size,
    float eviction_threshold
);

/* Insert a new token into the virtualized KV-cache */
int qwn_paged_eviction_insert(
    QwnPagedEvictionPool *pool,
    uint32_t token_id,
    float initial_importance,
    uint32_t *assigned_physical_slot
);

/* Update token importance based on attention Softmax scores */
void qwn_paged_eviction_update_scores(
    QwnPagedEvictionPool *pool,
    const float *attention_scores,
    uint32_t sequence_length
);

/* Evict lowest-importance transient tokens to fit target budget */
uint32_t qwn_paged_eviction_prune(
    QwnPagedEvictionPool *pool,
    uint32_t target_retained_tokens
);

/* Compute current memory waste percentage */
float qwn_paged_eviction_memory_waste_pct(const QwnPagedEvictionPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_PAGEDEVICTION_H */
