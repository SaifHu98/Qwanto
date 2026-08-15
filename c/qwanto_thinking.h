#ifndef QWANTO_THINKING_H
#define QWANTO_THINKING_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Thinking Levels (aligned with Gemini 3.7 Flash architecture)
 * ------------------------------------------------------------------------- */
typedef enum {
    QWN_THINK_LOW = 0,      /* Fast-fire mode: 5x speedup target, 4 layers, greedy */
    QWN_THINK_MEDIUM = 1,   /* Balanced mode: 2.5x speedup target, early exit, TurboQuant */
    QWN_THINK_HIGH = 2      /* Deep reasoning mode: 1x baseline, full depth, max reasoning */
} QwnThinkingLevel;

typedef struct {
    QwnThinkingLevel level;
    int early_exit_threshold;    /* 0-100, confidence % for early exit (e.g. 80-95%) */
    int max_speculative_tokens;  /* Max draft tokens per step (0 for low, 3 for med, 10 for high) */
    float temp_threshold;        /* Temperature for confidence estimation (default 1.0f) */
    int use_turboquant;          /* Enable TurboQuant for KV-Cache */
    int n_layers_max;            /* Max layers to run in LOW mode (default 4) */
    float *confidence_buffer;    /* Caller-provided or scratch per-layer confidence [layers] */
    int last_exit_layer;         /* Telemetry: index of the layer that exited early */
    float last_confidence;       /* Telemetry: confidence value at early exit */
} QwnThinkingConfig;

/* Forward declaration of QwnDecoder from qwanto_decode.h */
struct QwnDecoder;
typedef struct QwnDecoder QwnDecoder;

/* -------------------------------------------------------------------------
 * API Prototypes
 * ------------------------------------------------------------------------- */

/* Returns a default configured thinking profile for the requested level */
QwnThinkingConfig qwn_thinking_default_config(QwnThinkingLevel level);

/* Parse thinking level from string ("low", "medium", "high", "0", "1", "2") */
QwnThinkingLevel qwn_thinking_parse_level(const char *name);

/* Return human-readable string for thinking level */
const char *qwn_thinking_level_name(QwnThinkingLevel level);

/* Compute confidence score in [0.0, 1.0] from raw logits (top-1 probability or top margin) */
float qwn_thinking_compute_confidence(const float *logits, int vocab, float temp);

/* Single token forward pass with dynamic reasoning and early exit */
int qwn_decoder_forward_thinking(
    QwnDecoder *d,
    int token,
    const float **logits,
    QwnThinkingConfig *config
);

/* Multi-token autoregressive generation with thinking optimization */
int qwn_decoder_generate_thinking(
    QwnDecoder *d,
    const int *prompt,
    int prompt_count,
    int max_new_tokens,
    float temperature,
    float top_p,
    QwnThinkingConfig *config,
    void (*callback)(const char *chunk, int len, void *opaque),
    void *opaque
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_THINKING_H */
