#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_thinking.h"

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
 * Test 1: Configuration Profiles & Defaults
 * ------------------------------------------------------------------------- */
static void test_config_profiles(void) {
    QwnThinkingConfig low = qwn_thinking_default_config(QWN_THINK_LOW);
    EXPECT_TRUE(low.level == QWN_THINK_LOW);
    EXPECT_TRUE(low.n_layers_max == 4);
    EXPECT_TRUE(low.early_exit_threshold == 80);
    EXPECT_TRUE(low.max_speculative_tokens == 0);
    EXPECT_TRUE(low.use_turboquant == 0);

    QwnThinkingConfig med = qwn_thinking_default_config(QWN_THINK_MEDIUM);
    EXPECT_TRUE(med.level == QWN_THINK_MEDIUM);
    EXPECT_TRUE(med.early_exit_threshold == 80);
    EXPECT_TRUE(med.max_speculative_tokens == 3);
    EXPECT_TRUE(med.use_turboquant == 1);

    QwnThinkingConfig high = qwn_thinking_default_config(QWN_THINK_HIGH);
    EXPECT_TRUE(high.level == QWN_THINK_HIGH);
    EXPECT_TRUE(high.early_exit_threshold == 100);
    EXPECT_TRUE(high.max_speculative_tokens == 10);
    EXPECT_TRUE(high.use_turboquant == 1);
}

/* -------------------------------------------------------------------------
 * Test 2: Level Parsing & String Names
 * ------------------------------------------------------------------------- */
static void test_level_parsing(void) {
    EXPECT_TRUE(qwn_thinking_parse_level("low") == QWN_THINK_LOW);
    EXPECT_TRUE(qwn_thinking_parse_level("fast") == QWN_THINK_LOW);
    EXPECT_TRUE(qwn_thinking_parse_level("0") == QWN_THINK_LOW);

    EXPECT_TRUE(qwn_thinking_parse_level("medium") == QWN_THINK_MEDIUM);
    EXPECT_TRUE(qwn_thinking_parse_level("balanced") == QWN_THINK_MEDIUM);
    EXPECT_TRUE(qwn_thinking_parse_level("1") == QWN_THINK_MEDIUM);

    EXPECT_TRUE(qwn_thinking_parse_level("high") == QWN_THINK_HIGH);
    EXPECT_TRUE(qwn_thinking_parse_level("deep") == QWN_THINK_HIGH);
    EXPECT_TRUE(qwn_thinking_parse_level("cot") == QWN_THINK_HIGH);
    EXPECT_TRUE(qwn_thinking_parse_level("2") == QWN_THINK_HIGH);

    EXPECT_TRUE(strcmp(qwn_thinking_level_name(QWN_THINK_LOW), "low") == 0);
    EXPECT_TRUE(strcmp(qwn_thinking_level_name(QWN_THINK_MEDIUM), "medium") == 0);
    EXPECT_TRUE(strcmp(qwn_thinking_level_name(QWN_THINK_HIGH), "high") == 0);
}

/* -------------------------------------------------------------------------
 * Test 3: Mathematical Confidence Estimation
 * ------------------------------------------------------------------------- */
static void test_confidence_computation(void) {
    /* 1. Sharp single peak distribution: 1.0 confidence */
    float sharp[8] = {100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float c_sharp = qwn_thinking_compute_confidence(sharp, 8, 1.0f);
    EXPECT_NEAR(c_sharp, 1.0f, 1e-3f);

    /* 2. Perfectly uniform distribution: low confidence */
    float uniform[8] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float c_uniform = qwn_thinking_compute_confidence(uniform, 8, 1.0f);
    /* Uniform with 8 elements: p = 1/8 = 0.125, margin = 0 -> conf = 0.125 * 0.7 = 0.0875 */
    EXPECT_NEAR(c_uniform, 0.0875f, 1e-3f);

    /* 3. Moderate separation (top1=10.0, top2=8.0) */
    float mod[4] = {10.0f, 8.0f, 0.0f, 0.0f};
    float c_mod = qwn_thinking_compute_confidence(mod, 4, 1.0f);
    EXPECT_TRUE(c_mod > 0.6f && c_mod < 1.0f);

    /* 4. Temperature sensitivity: higher temp -> lower confidence */
    float c_mod_cold = qwn_thinking_compute_confidence(mod, 4, 0.5f);
    float c_mod_hot  = qwn_thinking_compute_confidence(mod, 4, 2.0f);
    EXPECT_TRUE(c_mod_cold > c_mod);
    EXPECT_TRUE(c_mod_hot < c_mod);

    /* 5. Edge cases: vocab=1, null pointer, negative temp */
    float single[1] = {42.0f};
    EXPECT_NEAR(qwn_thinking_compute_confidence(single, 1, 1.0f), 1.0f, 1e-3f);
    EXPECT_NEAR(qwn_thinking_compute_confidence(NULL, 10, 1.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(qwn_thinking_compute_confidence(sharp, 0, 1.0f), 0.0f, 1e-5f);
}

/* -------------------------------------------------------------------------
 * Test 4: 100+ Parameterized Differential Confidence Tests
 * ------------------------------------------------------------------------- */
static void test_differential_confidence_suite(void) {
    const int vocab_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 32000};
    const int n_vocabs = sizeof(vocab_sizes) / sizeof(vocab_sizes[0]);

    for (int v = 0; v < n_vocabs; v++) {
        int vocab = vocab_sizes[v];
        float *logits = (float*)malloc(vocab * sizeof(float));
        assert(logits != NULL);

        for (int step = 0; step < 10; step++) {
            /* Generate synthetic distribution with controlled gap */
            float gap = (float)step * 1.5f;
            for (int i = 0; i < vocab; i++) {
                logits[i] = (float)(rand() % 100) / 100.0f;
            }
            logits[0] += gap;

            float conf = qwn_thinking_compute_confidence(logits, vocab, 1.0f);
            EXPECT_TRUE(conf >= 0.0f && conf <= 1.0f);

            if (gap >= 12.0f && vocab <= 1024) {
                EXPECT_TRUE(conf >= 0.90f);
            }
        }
        free(logits);
    }
}

int main(void) {
    printf("=================================================================\n");
    printf("       Qwanto Configurable Thinking Engine Verification Suite    \n");
    printf("=================================================================\n");

    test_config_profiles();
    test_level_parsing();
    test_confidence_computation();
    test_differential_confidence_suite();

    printf("-----------------------------------------------------------------\n");
    printf("Results: %d passed / %d total assertions\n", g_tests_passed, g_tests_run);
    printf("=================================================================\n");

    if (g_tests_passed == g_tests_run && g_tests_run >= 100) {
        printf("[SUCCESS] All Configurable Thinking tests passed with 100%% accuracy!\n");
        return 0;
    } else {
        printf("[FAILURE] One or more tests failed!\n");
        return 1;
    }
}
