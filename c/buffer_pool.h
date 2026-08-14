#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include <stdint.h>
#include <stdlib.h>
#include "compat.h"

#define MAX_POOL_SLABS 128

typedef struct {
    uint8_t *slab;
    size_t slab_cap;
    float *fslab;
    size_t fslab_cap;
    int in_use;
} PoolEntry;

typedef struct {
    PoolEntry entries[MAX_POOL_SLABS];
    int count;
    size_t default_slab_cap;
    size_t default_fslab_cap;
} EBufferPool;

static inline void buffer_pool_init(EBufferPool *pool, int count, size_t slab_cap, size_t fslab_cap) {
    pool->count = count > MAX_POOL_SLABS ? MAX_POOL_SLABS : count;
    pool->default_slab_cap = slab_cap;
    pool->default_fslab_cap = fslab_cap;
    
    for (int i = 0; i < pool->count; i++) {
        pool->entries[i].slab = NULL;
        pool->entries[i].slab_cap = slab_cap;
        pool->entries[i].fslab = NULL;
        pool->entries[i].fslab_cap = fslab_cap;
        pool->entries[i].in_use = 0;
    }
}

static inline int buffer_pool_lease(EBufferPool *pool, uint8_t **slab_out, size_t *slab_cap_out, float **fslab_out, size_t *fslab_cap_out) {
    for (int i = 0; i < pool->count; i++) {
        // Atomic compare and swap to lease lock-free
        int expected = 0;
        if (__atomic_compare_exchange_n(&pool->entries[i].in_use, &expected, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            // Lazy allocation: allocate memory only on first lease
            if (!pool->entries[i].slab && pool->entries[i].slab_cap > 0) {
                posix_memalign((void**)&pool->entries[i].slab, 16384, pool->entries[i].slab_cap);
            }
            if (!pool->entries[i].fslab && pool->entries[i].fslab_cap > 0) {
                posix_memalign((void**)&pool->entries[i].fslab, 16384, pool->entries[i].fslab_cap);
            }
            *slab_out = pool->entries[i].slab;
            *slab_cap_out = pool->entries[i].slab_cap;
            *fslab_out = pool->entries[i].fslab;
            *fslab_cap_out = pool->entries[i].fslab_cap;
            return i; // return lease ID
        }
    }
    return -1; // No free buffers
}

static inline void buffer_pool_release(EBufferPool *pool, int lease_id) {
    if (lease_id >= 0 && lease_id < pool->count) {
        __atomic_store_n(&pool->entries[lease_id].in_use, 0, __ATOMIC_RELEASE);
    }
}

static inline void buffer_pool_destroy(EBufferPool *pool) {
    for (int i = 0; i < pool->count; i++) {
        if (pool->entries[i].slab) compat_aligned_free(pool->entries[i].slab);
        if (pool->entries[i].fslab) compat_aligned_free(pool->entries[i].fslab);
    }
    pool->count = 0;
}

#endif /* BUFFER_POOL_H */
