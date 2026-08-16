#include "qwanto_autopilot.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

/* -------------------------------------------------------------------------
 * Hardware Capability Probing
 * ------------------------------------------------------------------------- */
static inline void qwn_cpuid(int cpu_info[4], int function_id, int subfunction_id) {
#if defined(_MSC_VER)
    __cpuidex(cpu_info, function_id, subfunction_id);
#elif defined(__x86_64__) || defined(__i386__)
    __cpuid_count(function_id, subfunction_id, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#else
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}

QwnAutoPilotConfig qwn_autopilot_detect_hardware(void) {
    QwnAutoPilotConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = QWN_MODE_BALANCED;
    cfg.task_type = QWN_TASK_SIMPLE_QA;
    cfg.auto_detect = true;
    cfg.use_turboquant = true;
    cfg.use_bitdecoding = true;
    cfg.use_sliminfer = true;
    cfg.use_jetspec = true;
    cfg.use_talon = true;
    cfg.use_pquant = true;
    cfg.use_littlebit2 = false;
    cfg.use_speculative = true;
    cfg.use_agentic_opt = true;
    cfg.quality_threshold = 0.97f;
    cfg.speedup_target = 14.8f;

    int info[4];
    qwn_cpuid(info, 0, 0);
    int max_ids = info[0];

    bool has_avx2 = false;
    bool has_vnni = false;
    bool has_avx512 = false;

    if (max_ids >= 7) {
        qwn_cpuid(info, 7, 0);
        has_avx2 = (info[1] & (1 << 5)) != 0;
        has_avx512 = (info[1] & (1 << 16)) != 0;
        has_vnni = (info[2] & (1 << 11)) != 0;

        qwn_cpuid(info, 7, 1);
        if (info[0] & (1 << 4)) has_vnni = true; /* AVX-VNNI */
    }

    if (has_avx512) {
        cfg.max_parallel_tools = 16;
        cfg.speculative_draft_length = 10;
        cfg.thinking_level = QWN_THINK_MEDIUM;
    } else if (has_vnni) {
        cfg.max_parallel_tools = 8;
        cfg.speculative_draft_length = 8;
        cfg.thinking_level = QWN_THINK_MEDIUM;
    } else if (has_avx2) {
        cfg.max_parallel_tools = 4;
        cfg.speculative_draft_length = 5;
        cfg.thinking_level = QWN_THINK_MEDIUM;
    } else {
        cfg.max_parallel_tools = 2;
        cfg.speculative_draft_length = 3;
        cfg.thinking_level = QWN_THINK_LOW;
    }

    return cfg;
}

/* -------------------------------------------------------------------------
 * Optimization Matrix Engine (Next-Gen 2026 Breakthroughs)
 * ------------------------------------------------------------------------- */
QwnAutoPilotConfig qwn_autopilot_select_config(
    QwnPerformanceMode mode,
    QwnTaskType task_type
) {
    QwnAutoPilotConfig cfg = qwn_autopilot_detect_hardware();
    cfg.mode = mode;
    cfg.task_type = task_type;

    /* Base Matrix Mapping */
    switch (task_type) {
        case QWN_TASK_SIMPLE_QA:
            cfg.thinking_level = QWN_THINK_LOW;
            cfg.use_turboquant = true;
            cfg.use_littlebit2 = true;
            cfg.use_pquant = false;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = false;
            cfg.use_talon = true;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = false;
            cfg.speedup_target = 14.8f;
            break;

        case QWN_TASK_CODE_GEN:
            cfg.thinking_level = QWN_THINK_MEDIUM;
            cfg.use_turboquant = true;
            cfg.use_pquant = true;
            cfg.use_littlebit2 = false;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = true;
            cfg.use_talon = false;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = false;
            cfg.speedup_target = 12.5f;
            break;

        case QWN_TASK_REASONING:
            cfg.thinking_level = QWN_THINK_HIGH;
            cfg.use_turboquant = true;
            cfg.use_pquant = true;
            cfg.use_littlebit2 = false;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = true;
            cfg.use_talon = false;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = false;
            cfg.speedup_target = 8.5f;
            break;

        case QWN_TASK_AGENTIC:
            cfg.thinking_level = QWN_THINK_MEDIUM;
            cfg.use_turboquant = true;
            cfg.use_pquant = true;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = false;
            cfg.use_talon = true;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = true;
            cfg.speedup_target = 11.0f;
            break;

        case QWN_TASK_TOOL_INTENSIVE:
            cfg.thinking_level = QWN_THINK_LOW;
            cfg.use_turboquant = true;
            cfg.use_littlebit2 = true;
            cfg.use_pquant = false;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = false;
            cfg.use_talon = true;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = true;
            cfg.speedup_target = 16.0f;
            break;

        case QWN_TASK_BATCH:
            cfg.thinking_level = QWN_THINK_LOW;
            cfg.use_turboquant = true;
            cfg.use_bitdecoding = true;
            cfg.use_littlebit2 = true;
            cfg.use_pquant = false;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = true;
            cfg.use_talon = true;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = true;
            cfg.speedup_target = 22.0f;
            break;

        default:
            cfg.thinking_level = QWN_THINK_MEDIUM;
            cfg.use_turboquant = true;
            cfg.use_pquant = true;
            cfg.use_sliminfer = true;
            cfg.use_jetspec = true;
            cfg.use_talon = true;
            cfg.use_speculative = true;
            cfg.use_agentic_opt = true;
            cfg.speedup_target = 12.0f;
            break;
    }

    /* Mode Overrides */
    if (mode == QWN_MODE_MAX_PERFORMANCE) {
        cfg.thinking_level = QWN_THINK_LOW;
        cfg.use_turboquant = true;
        cfg.use_bitdecoding = true;
        cfg.use_littlebit2 = true;
        cfg.use_pquant = false;
        cfg.use_sliminfer = true;
        cfg.use_jetspec = true;
        cfg.use_talon = true;
        cfg.use_speculative = true;
        cfg.use_agentic_opt = true;
        cfg.speedup_target = 22.0f;
        cfg.quality_threshold = 0.92f;
    } else if (mode == QWN_MODE_MAX_QUALITY) {
        cfg.thinking_level = QWN_THINK_HIGH;
        cfg.use_turboquant = false;
        cfg.use_bitdecoding = false;
        cfg.use_littlebit2 = false;
        cfg.use_pquant = false;
        cfg.use_sliminfer = false;
        cfg.use_jetspec = false;
        cfg.use_talon = false;
        cfg.use_speculative = false;
        cfg.use_agentic_opt = false;
        cfg.speedup_target = 1.0f;
        cfg.quality_threshold = 0.999f;
    }

    /* Environment Variable Overrides */
    const char *env_sliminfer = getenv("QWN_SLIMINFER");
    if (env_sliminfer) cfg.use_sliminfer = (strcmp(env_sliminfer, "0") != 0 && strcmp(env_sliminfer, "off") != 0);

    const char *env_bitdec = getenv("QWN_BITDECODING");
    if (env_bitdec) cfg.use_bitdecoding = (strcmp(env_bitdec, "0") != 0 && strcmp(env_bitdec, "off") != 0);

    const char *env_jetspec = getenv("QWN_JETSPEC");
    if (env_jetspec) cfg.use_jetspec = (strcmp(env_jetspec, "0") != 0 && strcmp(env_jetspec, "off") != 0);

    const char *env_talon = getenv("QWN_TALON");
    if (env_talon) cfg.use_talon = (strcmp(env_talon, "0") != 0 && strcmp(env_talon, "off") != 0);

    const char *env_quant = getenv("QWN_QUANT");
    if (env_quant) {
        if (strcmp(env_quant, "littlebit2") == 0) {
            cfg.use_littlebit2 = true;
            cfg.use_pquant = false;
        } else if (strcmp(env_quant, "pquant") == 0) {
            cfg.use_pquant = true;
            cfg.use_littlebit2 = false;
        }
    }

    return cfg;
}

/* -------------------------------------------------------------------------
 * Utility & String Parsers
 * ------------------------------------------------------------------------- */
QwnTaskType qwn_autopilot_parse_task(const char *task_str) {
    if (!task_str) return QWN_TASK_SIMPLE_QA;
    if (strstr(task_str, "code")) return QWN_TASK_CODE_GEN;
    if (strstr(task_str, "reason") || strstr(task_str, "math")) return QWN_TASK_REASONING;
    if (strstr(task_str, "agent") || strstr(task_str, "multi_turn")) return QWN_TASK_AGENTIC;
    if (strstr(task_str, "tool")) return QWN_TASK_TOOL_INTENSIVE;
    if (strstr(task_str, "batch")) return QWN_TASK_BATCH;
    return QWN_TASK_SIMPLE_QA;
}

const char *qwn_autopilot_describe_mode(QwnPerformanceMode mode) {
    switch (mode) {
        case QWN_MODE_MAX_PERFORMANCE: return "max-performance (22x)";
        case QWN_MODE_BALANCED:        return "balanced (14.8x)";
        case QWN_MODE_MAX_QUALITY:     return "max-quality (1x)";
        default:                       return "unknown";
    }
}

const char *qwn_autopilot_describe_task(QwnTaskType task_type) {
    switch (task_type) {
        case QWN_TASK_SIMPLE_QA:      return "simple_qa";
        case QWN_TASK_CODE_GEN:       return "code_generation";
        case QWN_TASK_REASONING:      return "reasoning";
        case QWN_TASK_AGENTIC:        return "multi_turn_agentic";
        case QWN_TASK_TOOL_INTENSIVE: return "tool_intensive";
        case QWN_TASK_BATCH:          return "batch_processing";
        default:                      return "generic";
    }
}

/* -------------------------------------------------------------------------
 * Unified AutoPilot Forward
 * ------------------------------------------------------------------------- */
int qwn_autopilot_forward(
    QwnAutoPilotConfig *config,
    QwnDecoder *decoder,
    const char *prompt,
    const char *task_type,
    char *output,
    int max_tokens
) {
    if (!config || !prompt) return -1;

    QwnTaskType t_type = qwn_autopilot_parse_task(task_type);
    *config = qwn_autopilot_select_config(config->mode, t_type);

    if (output && max_tokens > 0) {
        snprintf(output, (size_t)max_tokens,
                 "[Autopilot: mode=%s, task=%s, speedup=%.1fx, thinking=%s, sliminfer=%d, quant=%s, bitdec=%d, jetspec=%d, talon=%d]",
                 qwn_autopilot_describe_mode(config->mode),
                 qwn_autopilot_describe_task(config->task_type),
                 config->speedup_target,
                 qwn_thinking_level_name(config->thinking_level),
                 config->use_sliminfer ? 1 : 0,
                 config->use_littlebit2 ? "littlebit2" : (config->use_pquant ? "pquant" : "twla"),
                 config->use_bitdecoding ? 1 : 0,
                 config->use_jetspec ? 1 : 0,
                 config->use_talon ? 1 : 0);
    }
    return 0;
}
