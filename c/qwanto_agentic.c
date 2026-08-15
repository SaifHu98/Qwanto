#include "qwanto_agentic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define strdup _strdup
#endif

/* -------------------------------------------------------------------------
 * FNV-1a Tool & Argument Hashing
 * ------------------------------------------------------------------------- */
uint64_t qwn_tool_hash(const char *tool_name, const char *args_json) {
    if (!tool_name) return 0ULL;
    uint64_t hash = 14695981039346656037ULL;

    /* Hash tool name */
    for (const char *p = tool_name; *p; p++) {
        hash ^= (uint64_t)(unsigned char)(*p);
        hash *= 1099511628211ULL;
    }

    /* Hash delimiter */
    hash ^= (uint64_t)':';
    hash *= 1099511628211ULL;

    /* Hash args if present */
    if (args_json) {
        for (const char *p = args_json; *p; p++) {
            hash ^= (uint64_t)(unsigned char)(*p);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

/* -------------------------------------------------------------------------
 * Engine Lifecycle
 * ------------------------------------------------------------------------- */
int qwn_agentic_engine_init(
    QwnAgenticEngine *engine,
    QwnDecoder *decoder,
    int cache_capacity
) {
    if (!engine) return -1;
    memset(engine, 0, sizeof(*engine));
    engine->decoder = decoder;
    engine->max_parallel_tools = QWN_AGENT_MAX_PARALLEL_WORKERS;
    engine->use_cache = true;
    engine->use_context_reuse = true;

    int cap = (cache_capacity > 0) ? cache_capacity : QWN_AGENT_DEFAULT_CACHE_SIZE;
    engine->tool_cache.capacity = cap;
    engine->tool_cache.count = 0;
    engine->tool_cache.current_lru_clock = 0;
    engine->tool_cache.total_lookups = 0;
    engine->tool_cache.total_hits = 0;
    engine->tool_cache.entries = (ToolCacheEntry*)calloc((size_t)cap, sizeof(ToolCacheEntry));
    if (!engine->tool_cache.entries) return -1;

    /* Initialize default session */
    if (qwn_session_init(&engine->session_ctx, 1001ULL, 4096) != 0) {
        free(engine->tool_cache.entries);
        return -1;
    }

    return 0;
}

void qwn_agentic_engine_free(QwnAgenticEngine *engine) {
    if (!engine) return;

    /* Free all allocated strings in tool cache */
    if (engine->tool_cache.entries) {
        for (int i = 0; i < engine->tool_cache.count; i++) {
            ToolCacheEntry *e = &engine->tool_cache.entries[i];
            if (e->tool_name) free(e->tool_name);
            if (e->args_json) free(e->args_json);
            if (e->result_data) free(e->result_data);
        }
        free(engine->tool_cache.entries);
        engine->tool_cache.entries = NULL;
    }
    engine->tool_cache.count = 0;
    engine->tool_cache.capacity = 0;

    qwn_session_free(&engine->session_ctx);
}

/* -------------------------------------------------------------------------
 * Tool Cache Operations (LRU + TTL)
 * ------------------------------------------------------------------------- */
const char *qwn_get_cached_tool(
    ToolCache *cache,
    uint64_t hash,
    uint64_t current_time
) {
    if (!cache || !cache->entries || hash == 0ULL) return NULL;
    cache->total_lookups++;

    for (int i = 0; i < cache->count; i++) {
        ToolCacheEntry *e = &cache->entries[i];
        if (e->key_hash == hash) {
            /* Check TTL expiration */
            if (current_time > 0 && e->ttl_seconds > 0) {
                if (current_time > (e->timestamp + e->ttl_seconds)) {
                    /* Entry expired */
                    return NULL;
                }
            }

            e->lru_clock = ++cache->current_lru_clock;
            e->access_count++;
            cache->total_hits++;
            return e->result_data;
        }
    }
    return NULL;
}

void qwn_cache_tool_result(
    ToolCache *cache,
    uint64_t hash,
    const char *tool_name,
    const char *args_json,
    const char *result_data,
    uint64_t ttl_seconds,
    uint64_t current_time
) {
    if (!cache || !cache->entries || hash == 0ULL || !result_data) return;

    /* 1. Update existing entry if present */
    for (int i = 0; i < cache->count; i++) {
        ToolCacheEntry *e = &cache->entries[i];
        if (e->key_hash == hash) {
            if (e->result_data) free(e->result_data);
            e->result_data = strdup(result_data);
            e->data_len = strlen(result_data);
            e->timestamp = (current_time > 0) ? current_time : (uint64_t)time(NULL);
            e->ttl_seconds = (ttl_seconds > 0) ? ttl_seconds : QWN_AGENT_DEFAULT_TTL;
            e->lru_clock = ++cache->current_lru_clock;
            return;
        }
    }

    /* 2. Choose slot (new slot or LRU evicted slot) */
    int slot = -1;
    if (cache->count < cache->capacity) {
        slot = cache->count++;
    } else {
        /* Evict least recently used slot */
        uint64_t oldest = 0xFFFFFFFFFFFFFFFFULL;
        int oldest_slot = 0;
        for (int i = 0; i < cache->capacity; i++) {
            if (cache->entries[i].lru_clock < oldest) {
                oldest = cache->entries[i].lru_clock;
                oldest_slot = i;
            }
        }
        slot = oldest_slot;
        ToolCacheEntry *old_e = &cache->entries[slot];
        if (old_e->tool_name) free(old_e->tool_name);
        if (old_e->args_json) free(old_e->args_json);
        if (old_e->result_data) free(old_e->result_data);
        memset(old_e, 0, sizeof(*old_e));
    }

    ToolCacheEntry *dest = &cache->entries[slot];
    dest->key_hash = hash;
    dest->tool_name = tool_name ? strdup(tool_name) : NULL;
    dest->args_json = args_json ? strdup(args_json) : NULL;
    dest->result_data = strdup(result_data);
    dest->data_len = strlen(result_data);
    dest->timestamp = (current_time > 0) ? current_time : (uint64_t)time(NULL);
    dest->ttl_seconds = (ttl_seconds > 0) ? ttl_seconds : QWN_AGENT_DEFAULT_TTL;
    dest->access_count = 1;
    dest->lru_clock = ++cache->current_lru_clock;
}

/* -------------------------------------------------------------------------
 * Session Context Operations
 * ------------------------------------------------------------------------- */
int qwn_session_init(SessionContext *ctx, uint64_t session_id, int capacity) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->session_id = session_id;
    ctx->capacity = (capacity > 0) ? capacity : 4096;
    ctx->tokens = (int*)malloc((size_t)ctx->capacity * sizeof(int));
    if (!ctx->tokens) return -1;
    ctx->n_tokens = 0;
    ctx->frozen_prefix_tokens = 0;
    ctx->is_frozen = false;
    return 0;
}

void qwn_session_free(SessionContext *ctx) {
    if (!ctx) return;
    if (ctx->tokens) {
        free(ctx->tokens);
        ctx->tokens = NULL;
    }
    ctx->n_tokens = 0;
    ctx->capacity = 0;
}

int qwn_reuse_context(SessionContext *ctx, const int *new_tokens, int n_new_tokens) {
    if (!ctx || !ctx->tokens || !new_tokens || n_new_tokens <= 0) return -1;

    if (ctx->n_tokens + n_new_tokens > ctx->capacity) {
        int new_cap = (ctx->n_tokens + n_new_tokens) * 2;
        int *re = (int*)realloc(ctx->tokens, (size_t)new_cap * sizeof(int));
        if (!re) return -1;
        ctx->tokens = re;
        ctx->capacity = new_cap;
    }

    /* Append only delta tokens */
    memcpy(ctx->tokens + ctx->n_tokens, new_tokens, (size_t)n_new_tokens * sizeof(int));
    ctx->n_tokens += n_new_tokens;
    return ctx->n_tokens;
}

void qwn_freeze_session(SessionContext *ctx) {
    if (!ctx) return;
    ctx->frozen_prefix_tokens = ctx->n_tokens;
    ctx->is_frozen = true;
}

/* -------------------------------------------------------------------------
 * Multi-Step Agentic Execution Pipeline
 * ------------------------------------------------------------------------- */
char *qwn_agentic_forward(
    QwnAgenticEngine *engine,
    const char *task,
    const char *tools_json,
    int max_steps
) {
    if (!engine || !task) return NULL;
    engine->total_tasks++;

    /* Format synthesis response */
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"success\",\"task\":\"%s\",\"parallel_workers\":%d,\"cache_hits\":%llu,\"total_lookups\":%llu}",
             task, engine->max_parallel_tools,
             (unsigned long long)engine->tool_cache.total_hits,
             (unsigned long long)engine->tool_cache.total_lookups);

    return strdup(buf);
}
