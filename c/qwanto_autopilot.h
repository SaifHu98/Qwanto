#ifndef QWANTO_AUTOPILOT_H
#define QWANTO_AUTOPILOT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "qwanto_decode.h"
#include "qwanto_thinking.h"
#include "qwanto_turboquant.h"
#include "qwanto_speculative.h"
#include "qwanto_agentic.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Performance Modes & Task Types
 * ------------------------------------------------------------------------- */
typedef enum {
    QWN_MODE_MAX_PERFORMANCE = 0,   /* 10x-12x speedup target, maximum throughput */
    QWN_MODE_BALANCED        = 1,   /* 5x speedup target, 95%+ quality retention */
    QWN_MODE_MAX_QUALITY     = 2    /* 1x baseline, 100% full precision reasoning */
} QwnPerformanceMode;

typedef enum {
    QWN_TASK_SIMPLE_QA       = 0,   /* Short factual Q&A, sentiment, classification */
    QWN_TASK_CODE_GEN        = 1,   /* Code synthesis, refactoring, linting */
    QWN_TASK_REASONING       = 2,   /* Multi-step logic, math, chain-of-thought */
    QWN_TASK_AGENTIC         = 3,   /* Multi-turn conversation with context reuse */
    QWN_TASK_TOOL_INTENSIVE  = 4,   /* Parallel tool calls, API integrations */
    QWN_TASK_BATCH           = 5    /* High-throughput batch inference */
} QwnTaskType;

/* -------------------------------------------------------------------------
 * Unified AutoPilot Configuration
 * ------------------------------------------------------------------------- */
typedef struct {
    QwnPerformanceMode mode;        /* Selected performance profile */
    QwnTaskType task_type;          /* Identified task domain */
    bool auto_detect;               /* Auto-configure based on CPU/GPU hardware */
    int max_parallel_tools;         /* Tool worker thread count (e.g. 4, 8, 16) */
    int speculative_draft_length;   /* Saguaro SSD draft length gamma (e.g. 0, 3, 5, 8, 10) */
    QwnThinkingLevel thinking_level;/* Reasoning depth (LOW, MEDIUM, HIGH) */
    bool use_turboquant;            /* 3.5-bit asymmetric KV-Cache quantization */
    bool use_speculative;           /* Saguaro SSD speculative decoding */
    bool use_agentic_opt;           /* LRU Tool Cache + Context Reuse */
    float quality_threshold;        /* Minimum acceptable quality score [0.0, 1.0] */
    float speedup_target;           /* Expected speedup factor (e.g. 5.0x, 10.0x) */
} QwnAutoPilotConfig;

/* -------------------------------------------------------------------------
 * API Prototypes
 * ------------------------------------------------------------------------- */

/* Auto-detect underlying CPU ISA (AVX-512, AVX-VNNI, AVX2) and memory tiers */
QwnAutoPilotConfig qwn_autopilot_detect_hardware(void);

/* Generate optimal configuration based on unified optimization matrix */
QwnAutoPilotConfig qwn_autopilot_select_config(
    QwnPerformanceMode mode,
    QwnTaskType task_type
);

/* Parse task type string into enum */
QwnTaskType qwn_autopilot_parse_task(const char *task_str);

/* String representation helpers */
const char *qwn_autopilot_describe_mode(QwnPerformanceMode mode);
const char *qwn_autopilot_describe_task(QwnTaskType task_type);

/* Unified forward execution dispatching optimal combination of kernels */
int qwn_autopilot_forward(
    QwnAutoPilotConfig *config,
    QwnDecoder *decoder,
    const char *prompt,
    const char *task_type,
    char *output,
    int max_tokens
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_AUTOPILOT_H */
