#ifndef QWN_PAGED_KV_H
#define QWN_PAGED_KV_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QWN_PAGE_BLOCK_SIZE 16  /* Tokens per physical block page */

typedef struct {
    uint16_t *key_data;    /* [block_capacity, layers, kv_heads, block_size, head_dim] */
    uint16_t *val_data;    /* [block_capacity, layers, kv_heads, block_size, head_dim] */
    int *free_stack;       /* Stack of available physical block IDs */
    int free_top;
    int total_blocks;
    int layers;
    int kv_heads;
    int head_dim;
    size_t block_bytes;    /* Bytes per physical block across all layers */
} QwnKVBlockPool;

typedef struct {
    int req_id;
    int num_tokens;
    int max_tokens;
    int *block_ids;        /* Array of physical block indices */
    int block_count;
    int block_capacity;
} QwnBlockTable;

/* Initialize global physical block pool */
int qwn_kv_pool_init(QwnKVBlockPool *pool, int total_blocks, int layers, int kv_heads, int head_dim);

/* Free physical block pool */
void qwn_kv_pool_free(QwnKVBlockPool *pool);

/* Allocate physical block from pool */
int qwn_kv_pool_alloc_block(QwnKVBlockPool *pool);

/* Return physical block to pool */
void qwn_kv_pool_free_block(QwnKVBlockPool *pool, int block_id);

/* Initialize per-request block table */
int qwn_block_table_init(QwnBlockTable *bt, int req_id, int initial_capacity);

/* Append token to request's paged KV cache, allocating new physical block if needed */
int qwn_block_table_append_token(QwnKVBlockPool *pool, QwnBlockTable *bt);

/* Free request block table and return all physical blocks to pool */
void qwn_block_table_free(QwnKVBlockPool *pool, QwnBlockTable *bt);

/* Write new K and V embeddings for a given layer at the current sequence position */
int qwn_paged_kv_write(QwnKVBlockPool *pool, const QwnBlockTable *bt, int layer,
                       int token_pos, const float *k, const float *v);

/* Compute PagedAttention context vector for a single attention head */
void qwn_paged_attention_head(const QwnKVBlockPool *pool, const QwnBlockTable *bt,
                              int layer, int head_idx, int kv_head_idx,
                              const float *q_head, float *scores_scratch,
                              float *out_ctx_head);

#ifdef __cplusplus
}
#endif

#endif /* QWN_PAGED_KV_H */
