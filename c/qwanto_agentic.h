#ifndef QWANTO_AGENTIC_H
#define QWANTO_AGENTIC_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "qwanto_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QWN_AGENT_DEFAULT_CACHE_SIZE = 512,
    QWN_AGENT_MAX_PARALLEL_WORKERS = 8,
    QWN_AGENT_DEFAULT_TTL = 3600 /* 1 hour */
};

/* -------------------------------------------------------------------------
 * Tool Cache Entry & LRU Cache with TTL
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t key_hash;          /* FNV-1a 64-bit hash of tool_name + args_json */
    char *tool_name;            /* Identifier of the tool */
    char *args_json;            /* Exact normalized arguments JSON */
    char *result_data;          /* Cached result payload */
    size_t data_len;            /* Byte length of result */
    uint64_t timestamp;         /* Insertion unix timestamp for TTL checking */
    uint64_t ttl_seconds;       /* Time to live in seconds */
    int access_count;           /* Cumulative access count */
    uint64_t lru_clock;         /* Monotonic LRU clock */
} ToolCacheEntry;

typedef struct {
    ToolCacheEntry *entries;    /* Pre-allocated or dynamic array of entries */
    int capacity;               /* Max cache capacity */
    int count;                  /* Occupied entries */
    uint64_t current_lru_clock; /* Monotonic clock for LRU eviction */
    uint64_t total_lookups;     /* Lookup counter */
    uint64_t total_hits;        /* Hit counter */
} ToolCache;

/* -------------------------------------------------------------------------
 * Multi-Turn Session Context Reuse
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t session_id;        /* Unique session identifier */
    int *tokens;                /* Accumulated session tokens */
    int n_tokens;               /* Number of valid tokens stored */
    int capacity;               /* Allocated token capacity */
    int frozen_prefix_tokens;   /* Number of tokens in frozen immutable prefix */
    bool is_frozen;             /* Whether prefix is frozen for zero-recompute */
} SessionContext;

/* -------------------------------------------------------------------------
 * Tool Execution Result
 * ------------------------------------------------------------------------- */
typedef struct {
    char id[64];
    char tool_name[64];
    char *data;
    bool from_cache;
    double elapsed_seconds;
} ToolResult;

/* -------------------------------------------------------------------------
 * Agentic Optimization Engine
 * ------------------------------------------------------------------------- */
typedef struct {
    QwnDecoder *decoder;        /* Target neural decoder */
    ToolCache tool_cache;       /* High-speed LRU tool cache with TTL */
    SessionContext session_ctx; /* Persistent session context for zero-copy multi-turn */
    int max_parallel_tools;     /* Max parallel worker threads (e.g. 8) */
    bool use_cache;             /* Enable/disable tool result caching */
    bool use_context_reuse;     /* Enable/disable prefix context reuse */
    uint64_t total_tasks;       /* Total agentic tasks processed */
    double cumulative_latency;  /* Total wall-clock time in agentic execution */
} QwnAgenticEngine;

/* -------------------------------------------------------------------------
 * API Prototypes
 * ------------------------------------------------------------------------- */

/* Compute 64-bit FNV-1a hash over tool name and arguments */
uint64_t qwn_tool_hash(const char *tool_name, const char *args_json);

/* Initialize agentic engine */
int qwn_agentic_engine_init(
    QwnAgenticEngine *engine,
    QwnDecoder *decoder,
    int cache_capacity
);

/* Free agentic engine and cache memory */
void qwn_agentic_engine_free(QwnAgenticEngine *engine);

/* Tool cache operations */
const char *qwn_get_cached_tool(
    ToolCache *cache,
    uint64_t hash,
    uint64_t current_time
);

void qwn_cache_tool_result(
    ToolCache *cache,
    uint64_t hash,
    const char *tool_name,
    const char *args_json,
    const char *result_data,
    uint64_t ttl_seconds,
    uint64_t current_time
);

/* Session context reuse operations */
int qwn_session_init(SessionContext *ctx, uint64_t session_id, int capacity);
void qwn_session_free(SessionContext *ctx);
int qwn_reuse_context(SessionContext *ctx, const int *new_tokens, int n_new_tokens);
void qwn_freeze_session(SessionContext *ctx);

/* Agentic multi-step execution pipeline */
char *qwn_agentic_forward(
    QwnAgenticEngine *engine,
    const char *task,
    const char *tools_json,
    int max_steps
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_AGENTIC_H */
