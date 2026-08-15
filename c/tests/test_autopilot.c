#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_autopilot.h"

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
 * Test 1: Hardware Capability Detection
 * ------------------------------------------------------------------------- */
static void test_hardware_detection(void) {
    QwnAutoPilotConfig hw = qwn_autopilot_detect_hardware();
    EXPECT_TRUE(hw.auto_detect);
    EXPECT_TRUE(hw.max_parallel_tools >= 2 && hw.max_parallel_tools <= 16);
    EXPECT_TRUE(hw.speculative_draft_length >= 3 && hw.speculative_draft_length <= 12);
    EXPECT_TRUE(hw.use_turboquant);
    EXPECT_TRUE(hw.use_speculative);
    EXPECT_TRUE(hw.use_agentic_opt);
}

/* -------------------------------------------------------------------------
 * Test 2: Unified Optimization Matrix Mapping
 * ------------------------------------------------------------------------- */
static void test_matrix_mapping(void) {
    /* 1. Simple Q&A */
    QwnAutoPilotConfig qa = qwn_autopilot_select_config(QWN_MODE_BALANCED, QWN_TASK_SIMPLE_QA);
    EXPECT_EQ(qa.thinking_level, QWN_THINK_LOW);
    EXPECT_TRUE(qa.use_turboquant);
    EXPECT_TRUE(!qa.use_speculative);
    EXPECT_TRUE(!qa.use_agentic_opt);
    EXPECT_NEAR(qa.speedup_target, 8.0f, 0.01f);

    /* 2. Code Generation */
    QwnAutoPilotConfig code = qwn_autopilot_select_config(QWN_MODE_BALANCED, QWN_TASK_CODE_GEN);
    EXPECT_EQ(code.thinking_level, QWN_THINK_MEDIUM);
    EXPECT_TRUE(code.use_turboquant);
    EXPECT_TRUE(code.use_speculative);
    EXPECT_TRUE(!code.use_agentic_opt);
    EXPECT_NEAR(code.speedup_target, 5.0f, 0.01f);

    /* 3. Complex Reasoning */
    QwnAutoPilotConfig rzn = qwn_autopilot_select_config(QWN_MODE_BALANCED, QWN_TASK_REASONING);
    EXPECT_EQ(rzn.thinking_level, QWN_THINK_HIGH);
    EXPECT_TRUE(rzn.use_turboquant);
    EXPECT_TRUE(rzn.use_speculative);
    EXPECT_TRUE(!rzn.use_agentic_opt);
    EXPECT_NEAR(rzn.speedup_target, 3.0f, 0.01f);

    /* 4. Multi-Turn Agentic */
    QwnAutoPilotConfig agt = qwn_autopilot_select_config(QWN_MODE_BALANCED, QWN_TASK_AGENTIC);
    EXPECT_EQ(agt.thinking_level, QWN_THINK_MEDIUM);
    EXPECT_TRUE(agt.use_turboquant);
    EXPECT_TRUE(!agt.use_speculative);
    EXPECT_TRUE(agt.use_agentic_opt);
    EXPECT_NEAR(agt.speedup_target, 6.0f, 0.01f);

    /* 5. Tool-Intensive */
    QwnAutoPilotConfig tool = qwn_autopilot_select_config(QWN_MODE_BALANCED, QWN_TASK_TOOL_INTENSIVE);
    EXPECT_EQ(tool.thinking_level, QWN_THINK_LOW);
    EXPECT_TRUE(tool.use_turboquant);
    EXPECT_TRUE(!tool.use_speculative);
    EXPECT_TRUE(tool.use_agentic_opt);
    EXPECT_NEAR(tool.speedup_target, 10.0f, 0.01f);

    /* 6. Batch Processing */
    QwnAutoPilotConfig batch = qwn_autopilot_select_config(QWN_MODE_BALANCED, QWN_TASK_BATCH);
    EXPECT_EQ(batch.thinking_level, QWN_THINK_LOW);
    EXPECT_TRUE(batch.use_turboquant);
    EXPECT_TRUE(batch.use_speculative);
    EXPECT_TRUE(batch.use_agentic_opt);
    EXPECT_NEAR(batch.speedup_target, 12.0f, 0.01f);
}

/* -------------------------------------------------------------------------
 * Test 3: Mode Overrides
 * ------------------------------------------------------------------------- */
static void test_mode_overrides(void) {
    /* Max Performance Mode Override */
    QwnAutoPilotConfig max_perf = qwn_autopilot_select_config(QWN_MODE_MAX_PERFORMANCE, QWN_TASK_REASONING);
    EXPECT_EQ(max_perf.thinking_level, QWN_THINK_LOW);
    EXPECT_TRUE(max_perf.use_turboquant);
    EXPECT_TRUE(max_perf.use_speculative);
    EXPECT_TRUE(max_perf.use_agentic_opt);
    EXPECT_NEAR(max_perf.speedup_target, 10.0f, 0.01f);
    EXPECT_NEAR(max_perf.quality_threshold, 0.85f, 0.01f);

    /* Max Quality Mode Override */
    QwnAutoPilotConfig max_qual = qwn_autopilot_select_config(QWN_MODE_MAX_QUALITY, QWN_TASK_SIMPLE_QA);
    EXPECT_EQ(max_qual.thinking_level, QWN_THINK_HIGH);
    EXPECT_TRUE(!max_qual.use_turboquant);
    EXPECT_TRUE(!max_qual.use_speculative);
    EXPECT_TRUE(!max_qual.use_agentic_opt);
    EXPECT_NEAR(max_qual.speedup_target, 1.0f, 0.01f);
    EXPECT_NEAR(max_qual.quality_threshold, 0.99f, 0.01f);
}

/* -------------------------------------------------------------------------
 * Test 4: Task String Parsing & Descriptions
 * ------------------------------------------------------------------------- */
static void test_task_parsing_and_descriptions(void) {
    EXPECT_EQ(qwn_autopilot_parse_task("simple_qa"), QWN_TASK_SIMPLE_QA);
    EXPECT_EQ(qwn_autopilot_parse_task("code_generation"), QWN_TASK_CODE_GEN);
    EXPECT_EQ(qwn_autopilot_parse_task("math_reasoning"), QWN_TASK_REASONING);
    EXPECT_EQ(qwn_autopilot_parse_task("agentic_chat"), QWN_TASK_AGENTIC);
    EXPECT_EQ(qwn_autopilot_parse_task("tool_use"), QWN_TASK_TOOL_INTENSIVE);
    EXPECT_EQ(qwn_autopilot_parse_task("batch_eval"), QWN_TASK_BATCH);

    EXPECT_TRUE(strstr(qwn_autopilot_describe_mode(QWN_MODE_MAX_PERFORMANCE), "10x") != NULL);
    EXPECT_TRUE(strstr(qwn_autopilot_describe_mode(QWN_MODE_BALANCED), "5x") != NULL);
    EXPECT_TRUE(strstr(qwn_autopilot_describe_mode(QWN_MODE_MAX_QUALITY), "1x") != NULL);
}

/* -------------------------------------------------------------------------
 * Test 5: 100+ Matrix Boundary & Stress Assertions
 * ------------------------------------------------------------------------- */
static void test_stress_autopilot_matrix(void) {
    for (int mode = 0; mode < 3; mode++) {
        for (int task = 0; task < 6; task++) {
            QwnAutoPilotConfig cfg = qwn_autopilot_select_config((QwnPerformanceMode)mode, (QwnTaskType)task);
            EXPECT_TRUE(cfg.speedup_target >= 1.0f && cfg.speedup_target <= 12.0f);
            EXPECT_TRUE(cfg.quality_threshold >= 0.80f && cfg.quality_threshold <= 1.00f);
            EXPECT_TRUE(cfg.max_parallel_tools > 0);
            EXPECT_TRUE(cfg.speculative_draft_length > 0);

            char out_buf[512];
            int rc = qwn_autopilot_forward(&cfg, NULL, "sample prompt", "code", out_buf, sizeof(out_buf));
            EXPECT_EQ(rc, 0);
            EXPECT_TRUE(strlen(out_buf) > 10);
        }
    }
}

int main(void) {
    printf("=================================================================\n");
    printf("       Qwanto Performance Autopilot Engine Verification         \n");
    printf("=================================================================\n");

    test_hardware_detection();
    test_matrix_mapping();
    test_mode_overrides();
    test_task_parsing_and_descriptions();
    test_stress_autopilot_matrix();

    printf("-----------------------------------------------------------------\n");
    printf("Results: %d passed / %d total assertions\n", g_tests_passed, g_tests_run);
    printf("=================================================================\n");

    if (g_tests_passed == g_tests_run && g_tests_run >= 150) {
        printf("[SUCCESS] All Performance Autopilot tests passed with 100%% accuracy!\n");
        return 0;
    } else {
        printf("[FAILURE] One or more tests failed!\n");
        return 1;
    }
}
