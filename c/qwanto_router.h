#ifndef QWANTO_ROUTER_H
#define QWANTO_ROUTER_H

#include <stdint.h>
#include "aio_compat.h"
#include "st.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sub-microsecond Mixture of Experts (MoE) routing engine using Bitwise LSH.
 * Assigns input tokens to expert IDs using bitwise shifts, XORs, and modulo operations.
 * 
 * @param activations   Pointer to int8_t token activation vector [hidden_dim]
 * @param hidden_dim    Hidden dimension size
 * @param n_experts     Total number of experts available
 * @param top_k         Number of experts to route to per token
 * @param expert_ids    Output array of size [top_k] to store selected expert IDs
 */
void qwanto_route_lsh(const int8_t* activations, int hidden_dim, int n_experts, int top_k, int* expert_ids);

/* Prefetcher structures */
typedef struct {
    ColiAioRequest req_g;
    ColiAioRequest req_u;
    ColiAioRequest req_d;
    int layer;
    int eid;
    int submitted;
} QwantoPrefetchJob;

typedef struct {
    st_tensor *tg;
    st_tensor *tu;
    st_tensor *td;
} QwantoExpertTensors;

typedef struct {
    ColiAioContext aio_ctx;
    QwantoPrefetchJob jobs[16];
    int max_jobs;
    int active_jobs;
    QwantoExpertTensors expert_cache[64][128]; /* Cached tensor pointers per [layer][eid] */
} QwantoPrefetcher;

/**
 * Initialize the prefetcher context.
 * 
 * @param pf            Pointer to prefetcher context
 * @param queue_depth   Maximum number of concurrent I/O operations
 * @return 0 on success, -1 on failure
 */
int qwanto_prefetcher_init(QwantoPrefetcher* pf, int queue_depth);

/**
 * Submit an asynchronous prefetch request for an expert's weights.
 * 
 * @param pf            Pointer to prefetcher context
 * @param S             Pointer to safetensors shards database
 * @param layer         Layer index
 * @param eid           Expert ID
 * @param buf_g         Buffer to load gate_proj weights
 * @param buf_u         Buffer to load up_proj weights
 * @param buf_d         Buffer to load down_proj weights
 * @return 0 on success, -1 on failure/queue full
 */
int qwanto_prefetcher_submit(QwantoPrefetcher* pf, shards* S, int layer, int eid, 
                             void* buf_g, void* buf_u, void* buf_d);

/**
 * Wait for all pending prefetch I/O requests to complete.
 * 
 * @param pf            Pointer to prefetcher context
 * @return 0 on success, -1 on failure
 */
int qwanto_prefetcher_wait_all(QwantoPrefetcher* pf);

/**
 * Destroy the prefetcher and release resources.
 * 
 * @param pf            Pointer to prefetcher context
 */
void qwanto_prefetcher_destroy(QwantoPrefetcher* pf);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_ROUTER_H */
