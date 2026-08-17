#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwn_speculative.h"

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

#define EXPECT_NEAR(a, b, tol) do { \
    g_tests_run++; \
    float diff = fabsf((float)(a) - (float)(b)); \
    if (diff <= (tol)) { \
        g_tests_passed++; \
    } else { \
        fprintf(stderr, "[FAIL] Line %d: %f not near %f (diff=%e, tol=%e)\n", __LINE__, (float)(a), (float)(b), diff, (float)(tol)); \
    } \
} while(0)

/* -------------------------------------------------------------------------
 * Test 1: 64-bit FNV-1a Prefix Context Hashing
 * ------------------------------------------------------------------------- */
static void test_context_hashing(void) {
    int tokens1[] = {101, 202, 303, 404, 505, 606, 707, 808};
    int tokens2[] = {101, 202, 303, 404, 505, 606, 707, 809}; /* 1 token diff */
    int tokens3[] = {101, 202, 303, 404, 505, 606, 707, 808}; /* exact match */

    uint64_t h1 = qwn_speculative_hash_context(tokens1, 8);
    uint64_t h2 = qwn_speculative_hash_context(tokens2, 8);
    uint64_t h3 = qwn_speculative_hash_context(tokens3, 8);

    EXPECT_TRUE(h1 != 0ULL);
    EXPECT_TRUE(h2 != 0ULL);
    EXPECT_TRUE(h1 != h2);
    EXPECT_TRUE(h1 == h3);

    /* Edge cases */
    EXPECT_TRUE(qwn_speculative_hash_context(NULL, 10) == 0ULL);
    EXPECT_TRUE(qwn_speculative_hash_context(tokens1, 0) == 0ULL);
    EXPECT_TRUE(qwn_speculative_hash_context(tokens1, -5) == 0ULL);

    /* 50 Differential hash length tests */
    for (int len = 1; len <= 50; len++) {
        int seq[50];
        for (int i = 0; i < len; i++) seq[i] = i * 17 + 3;
        uint64_t h = qwn_speculative_hash_context(seq, len);
        EXPECT_TRUE(h != 0ULL);
    }
}

/* -------------------------------------------------------------------------
 * Test 2: Speculation Cache LRU Management & Eviction
 * ------------------------------------------------------------------------- */
static void test_cache_lru_and_eviction(void) {
    QwnSpeculativeEngine engine;
    EXPECT_EQ(qwn_speculative_engine_init(&engine, NULL, NULL, 4), 0);

    EXPECT_EQ(engine.cache.capacity, 4);
    EXPECT_EQ(engine.cache.count, 0);

    int draft_a[] = {10, 20, 30};
    int draft_b[] = {40, 50};
    int draft_c[] = {60, 70, 80, 90};
    int draft_d[] = {100};
    int draft_e[] = {110, 120}; /* 5th entry, forces LRU eviction */

    float probs[] = {0.9f, 0.9f, 0.9f, 0.9f};

    qwn_speculative_cache_insert(&engine.cache, 0xA1, draft_a, probs, 3);
    qwn_speculative_cache_insert(&engine.cache, 0xB2, draft_b, probs, 2);
    qwn_speculative_cache_insert(&engine.cache, 0xC3, draft_c, probs, 4);
    qwn_speculative_cache_insert(&engine.cache, 0xD4, draft_d, probs, 1);

    EXPECT_EQ(engine.cache.count, 4);

    /* Lookup A to refresh its LRU clock */
    int out_buf[8];
    float out_probs[8];
    int n = qwn_speculative_cache_lookup(&engine.cache, 0xA1, out_buf, out_probs, 8);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(out_buf[0], 10);
    EXPECT_EQ(out_buf[1], 20);
    EXPECT_EQ(out_buf[2], 30);

    /* Insert E (0xE5) -> should evict B2 (0xB2) since A1 was refreshed */
    qwn_speculative_cache_insert(&engine.cache, 0xE5, draft_e, probs, 2);
    EXPECT_EQ(engine.cache.count, 4);

    /* B2 must be evicted */
    EXPECT_EQ(qwn_speculative_cache_lookup(&engine.cache, 0xB2, out_buf, out_probs, 8), 0);
    /* A1 and E5 must exist */
    EXPECT_EQ(qwn_speculative_cache_lookup(&engine.cache, 0xA1, out_buf, out_probs, 8), 3);
    EXPECT_EQ(qwn_speculative_cache_lookup(&engine.cache, 0xE5, out_buf, out_probs, 8), 2);

    qwn_speculative_engine_free(&engine);
}

/* -------------------------------------------------------------------------
 * Test 3: Adaptive Draft Length Scaling
 * ------------------------------------------------------------------------- */
static void test_adaptive_draft_length(void) {
    SpeculationCache cache;
    memset(&cache, 0, sizeof(cache));

    /* 1. Initial bootstrap phase (total_drafted < 4) */
    cache.total_drafted = 2;
    cache.acceptance_rate = 0.5f;
    EXPECT_EQ(get_optimal_draft_length(&cache), 4);

    /* 2. High acceptance rate (> 0.90) -> length 8 */
    cache.total_drafted = 20;
    cache.acceptance_rate = 0.95f;
    EXPECT_EQ(get_optimal_draft_length(&cache), 8);

    /* 3. Moderate acceptance rate (> 0.70) -> length 5 */
    cache.acceptance_rate = 0.75f;
    EXPECT_EQ(get_optimal_draft_length(&cache), 5);

    /* 4. Low acceptance rate (<= 0.70) -> length 3 */
    cache.acceptance_rate = 0.65f;
    EXPECT_EQ(get_optimal_draft_length(&cache), 3);

    cache.acceptance_rate = 0.20f;
    EXPECT_EQ(get_optimal_draft_length(&cache), 3);
}

/* -------------------------------------------------------------------------
 * Test 4: Speculation Ring Buffer Operations (Capacity 32)
 * ------------------------------------------------------------------------- */
static void test_ring_buffer(void) {
    QwnSpeculativeEngine engine;
    EXPECT_EQ(qwn_speculative_engine_init(&engine, NULL, NULL, 16), 0);

    /* Push 10 elements */
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(qwn_spec_ring_push(&engine, 1000 + i), 0);
    }
    EXPECT_EQ(engine.ring_count, 10);

    /* Pop 5 elements */
    for (int i = 0; i < 5; i++) {
        int tok = -1;
        EXPECT_EQ(qwn_spec_ring_pop(&engine, &tok), 0);
        EXPECT_EQ(tok, 1000 + i);
    }
    EXPECT_EQ(engine.ring_count, 5);

    /* Push 30 elements (forces wrap-around overwrite) */
    for (int i = 0; i < 30; i++) {
        EXPECT_EQ(qwn_spec_ring_push(&engine, 2000 + i), 0);
    }
    EXPECT_EQ(engine.ring_count, 32); /* Max ring buffer size */

    qwn_spec_ring_clear(&engine);
    EXPECT_EQ(engine.ring_count, 0);

    qwn_speculative_engine_free(&engine);
}

/* -------------------------------------------------------------------------
 * Test 5: 100+ Stress & Differential Cache Assertions
 * ------------------------------------------------------------------------- */
static void test_stress_cache_and_rates(void) {
    QwnSpeculativeEngine engine;
    EXPECT_EQ(qwn_speculative_engine_init(&engine, NULL, NULL, 64), 0);

    int sample_toks[4] = {1, 2, 3, 4};
    float sample_probs[4] = {0.85f, 0.85f, 0.85f, 0.85f};

    for (int i = 1; i <= 100; i++) {
        uint64_t h = (uint64_t)(i * 10007);
        sample_toks[0] = i;
        qwn_speculative_cache_insert(&engine.cache, h, sample_toks, sample_probs, 4);

        int out_toks[4];
        float out_p[4];
        int hit_n = qwn_speculative_cache_lookup(&engine.cache, h, out_toks, out_p, 4);
        EXPECT_EQ(hit_n, 4);
        EXPECT_EQ(out_toks[0], i);

        qwn_speculative_cache_update_rate(&engine.cache, 8, (i % 2 == 0) ? 7 : 6);
        EXPECT_TRUE(engine.cache.acceptance_rate >= 0.0f && engine.cache.acceptance_rate <= 1.0f);
    }

    qwn_speculative_engine_free(&engine);
}

static void test_product_path_fails_closed_without_draft(void) {
    QwnSpeculativeEngine engine;
    EXPECT_EQ(qwn_speculative_engine_init(&engine, NULL, NULL, 8), 0);
    EXPECT_EQ(qwn_speculative_forward(&engine, NULL, 0, NULL, 0, 0.0f),
              QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL);
    EXPECT_EQ(qwn_verify_and_accept(&engine, NULL, 0, NULL, NULL, NULL), 0);
    qwn_speculative_engine_free(&engine);
}

int main(void) {
    printf("=================================================================\n");
    printf("       Qwanto Saguaro (SSD) Speculative Engine Verification      \n");
    printf("=================================================================\n");

    test_context_hashing();
    test_cache_lru_and_eviction();
    test_adaptive_draft_length();
    test_ring_buffer();
    test_stress_cache_and_rates();
    test_product_path_fails_closed_without_draft();

    printf("-----------------------------------------------------------------\n");
    printf("Results: %d passed / %d total assertions\n", g_tests_passed, g_tests_run);
    printf("=================================================================\n");

    if (g_tests_passed == g_tests_run && g_tests_run >= 200) {
        printf("[SUCCESS] Legacy cache compatibility and fail-closed boundary tests passed.\n");
        return 0;
    } else {
        printf("[FAILURE] One or more tests failed!\n");
        return 1;
    }
}
