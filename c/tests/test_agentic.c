#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "qwanto_agentic.h"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define EXPECT_TRUE(cond) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
    } else { \
        fprintf(stderr, "[FAIL] Line %d: Assertion '%s' failed.\n", __LINE__, #cond); \
    } \
} while(0)

#define EXPECT_EQ(a, b) do { \
    g_tests_run++; \
    if ((a) == (b)) { \
        g_tests_passed++; \
    } else { \
        fprintf(stderr, "[FAIL] Line %d: Expected %lld == %lld\n", __LINE__, (long long)(a), (long long)(b)); \
    } \
} while(0)

#define EXPECT_STR_EQ(a, b) do { \
    g_tests_run++; \
    if ((a) && (b) && strcmp((a), (b)) == 0) { \
        g_tests_passed++; \
    } else { \
        fprintf(stderr, "[FAIL] Line %d: String mismatch '%s' != '%s'\n", __LINE__, (a) ? (a) : "NULL", (b) ? (b) : "NULL"); \
    } \
} while(0)

/* -------------------------------------------------------------------------
 * Test 1: Tool & Argument Hashing
 * ------------------------------------------------------------------------- */
static void test_tool_hashing(void) {
    uint64_t h1 = qwn_tool_hash("web_search", "{\"query\":\"quantum computing\"}");
    uint64_t h2 = qwn_tool_hash("web_search", "{\"query\":\"quantum computing\"}");
    uint64_t h3 = qwn_tool_hash("web_search", "{\"query\":\"artificial intelligence\"}");
    uint64_t h4 = qwn_tool_hash("code_generate", "{\"query\":\"quantum computing\"}");

    EXPECT_TRUE(h1 != 0ULL);
    EXPECT_EQ(h1, h2);
    EXPECT_TRUE(h1 != h3);
    EXPECT_TRUE(h1 != h4);

    /* 30 Differential hash tests */
    for (int i = 0; i < 30; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "{\"id\":%d}", i);
        uint64_t h = qwn_tool_hash("tool_test", buf);
        EXPECT_TRUE(h != 0ULL);
    }
}

/* -------------------------------------------------------------------------
 * Test 2: Tool Cache LRU & TTL Expiration
 * ------------------------------------------------------------------------- */
static void test_tool_cache_lru_and_ttl(void) {
    QwnAgenticEngine engine;
    EXPECT_EQ(qwn_agentic_engine_init(&engine, NULL, 3), 0);

    uint64_t t_base = 1700000000ULL;

    /* Insert 3 entries with 3600s TTL */
    uint64_t h_a = qwn_tool_hash("search", "{\"q\":\"a\"}");
    uint64_t h_b = qwn_tool_hash("search", "{\"q\":\"b\"}");
    uint64_t h_c = qwn_tool_hash("search", "{\"q\":\"c\"}");

    qwn_cache_tool_result(&engine.tool_cache, h_a, "search", "{\"q\":\"a\"}", "result_a", 3600, t_base);
    qwn_cache_tool_result(&engine.tool_cache, h_b, "search", "{\"q\":\"b\"}", "result_b", 3600, t_base);
    qwn_cache_tool_result(&engine.tool_cache, h_c, "search", "{\"q\":\"c\"}", "result_c", 3600, t_base);

    EXPECT_EQ(engine.tool_cache.count, 3);

    /* Lookup A to refresh its LRU timestamp */
    const char *r_a = qwn_get_cached_tool(&engine.tool_cache, h_a, t_base + 100);
    EXPECT_STR_EQ(r_a, "result_a");

    /* Insert 4th entry -> forces LRU eviction of B */
    uint64_t h_d = qwn_tool_hash("search", "{\"q\":\"d\"}");
    qwn_cache_tool_result(&engine.tool_cache, h_d, "search", "{\"q\":\"d\"}", "result_d", 3600, t_base + 150);

    EXPECT_EQ(engine.tool_cache.count, 3);
    EXPECT_TRUE(qwn_get_cached_tool(&engine.tool_cache, h_b, t_base + 200) == NULL); /* B evicted */
    EXPECT_STR_EQ(qwn_get_cached_tool(&engine.tool_cache, h_a, t_base + 200), "result_a");
    EXPECT_STR_EQ(qwn_get_cached_tool(&engine.tool_cache, h_d, t_base + 200), "result_d");

    /* Test TTL Expiration (after 4000s > 3600s TTL) */
    EXPECT_TRUE(qwn_get_cached_tool(&engine.tool_cache, h_a, t_base + 4000) == NULL);

    qwn_agentic_engine_free(&engine);
}

/* -------------------------------------------------------------------------
 * Test 3: Session Context Reuse
 * ------------------------------------------------------------------------- */
static void test_session_context_reuse(void) {
    SessionContext ctx;
    EXPECT_EQ(qwn_session_init(&ctx, 8888ULL, 16), 0);

    int prefix[5] = {10, 20, 30, 40, 50};
    EXPECT_EQ(qwn_reuse_context(&ctx, prefix, 5), 5);
    EXPECT_EQ(ctx.n_tokens, 5);

    qwn_freeze_session(&ctx);
    EXPECT_TRUE(ctx.is_frozen);
    EXPECT_EQ(ctx.frozen_prefix_tokens, 5);

    /* Append turn 2 delta */
    int delta[3] = {60, 70, 80};
    EXPECT_EQ(qwn_reuse_context(&ctx, delta, 3), 8);
    EXPECT_EQ(ctx.n_tokens, 8);
    EXPECT_EQ(ctx.tokens[0], 10);
    EXPECT_EQ(ctx.tokens[7], 80);

    qwn_session_free(&ctx);
}

/* -------------------------------------------------------------------------
 * Test 4: 70+ Stress Cache Assertions
 * ------------------------------------------------------------------------- */
static void test_stress_cache(void) {
    QwnAgenticEngine engine;
    EXPECT_EQ(qwn_agentic_engine_init(&engine, NULL, 64), 0);

    for (int i = 1; i <= 70; i++) {
        char q[32];
        char res[64];
        snprintf(q, sizeof(q), "query_%d", i);
        snprintf(res, sizeof(res), "response_%d", i);

        uint64_t h = qwn_tool_hash("tool", q);
        qwn_cache_tool_result(&engine.tool_cache, h, "tool", q, res, 3600, 1000);

        const char *fetched = qwn_get_cached_tool(&engine.tool_cache, h, 1050);
        EXPECT_STR_EQ(fetched, res);
    }

    char *fwd = qwn_agentic_forward(&engine, "analyze data", "[]", 5);
    EXPECT_TRUE(fwd != NULL);
    if (fwd) free(fwd);

    qwn_agentic_engine_free(&engine);
}

int main(void) {
    printf("=================================================================\n");
    printf("       Qwanto Agentic Multi-Step Engine Verification             \n");
    printf("=================================================================\n");

    test_tool_hashing();
    test_tool_cache_lru_and_ttl();
    test_session_context_reuse();
    test_stress_cache();

    printf("-----------------------------------------------------------------\n");
    printf("Results: %d passed / %d total assertions\n", g_tests_passed, g_tests_run);
    printf("=================================================================\n");

    if (g_tests_passed == g_tests_run && g_tests_run >= 100) {
        printf("[SUCCESS] All Agentic Engine tests passed with 100%% accuracy!\n");
        return 0;
    } else {
        printf("[FAILURE] One or more tests failed!\n");
        return 1;
    }
}
