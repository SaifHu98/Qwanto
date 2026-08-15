#include "qwanto_thinking.h"
#include "qwanto_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Default Configurations
 * ------------------------------------------------------------------------- */
QwnThinkingConfig qwn_thinking_default_config(QwnThinkingLevel level) {
    QwnThinkingConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.level = level;
    cfg.temp_threshold = 1.0f;
    cfg.last_exit_layer = -1;
    cfg.last_confidence = 0.0f;

    switch (level) {
        case QWN_THINK_LOW:
            cfg.early_exit_threshold = 80;
            cfg.max_speculative_tokens = 0;
            cfg.use_turboquant = 0;
            cfg.n_layers_max = 4;
            break;

        case QWN_THINK_MEDIUM:
            cfg.early_exit_threshold = 80;
            cfg.max_speculative_tokens = 3;
            cfg.use_turboquant = 1;
            cfg.n_layers_max = 999;
            break;

        case QWN_THINK_HIGH:
        default:
            cfg.level = QWN_THINK_HIGH;
            cfg.early_exit_threshold = 100;
            cfg.max_speculative_tokens = 10;
            cfg.use_turboquant = 1;
            cfg.n_layers_max = 999;
            break;
    }
    return cfg;
}

QwnThinkingLevel qwn_thinking_parse_level(const char *name) {
    if (!name || !*name) return QWN_THINK_MEDIUM;
    if (strcmp(name, "low") == 0 || strcmp(name, "fast") == 0 || strcmp(name, "0") == 0) {
        return QWN_THINK_LOW;
    }
    if (strcmp(name, "high") == 0 || strcmp(name, "deep") == 0 || strcmp(name, "cot") == 0 || strcmp(name, "2") == 0) {
        return QWN_THINK_HIGH;
    }
    return QWN_THINK_MEDIUM;
}

const char *qwn_thinking_level_name(QwnThinkingLevel level) {
    switch (level) {
        case QWN_THINK_LOW:    return "low";
        case QWN_THINK_MEDIUM: return "medium";
        case QWN_THINK_HIGH:   return "high";
        default:               return "unknown";
    }
}

/* -------------------------------------------------------------------------
 * Mathematical Confidence Estimation
 * ------------------------------------------------------------------------- */
float qwn_thinking_compute_confidence(const float *logits, int vocab, float temp) {
    if (!logits || vocab <= 0) return 0.0f;
    if (temp <= 0.0f) temp = 1.0f;

    /* Find top-1 and top-2 logits */
    float max1 = -1e30f;
    float max2 = -1e30f;
    int idx1 = -1;

    for (int i = 0; i < vocab; i++) {
        float val = logits[i];
        if (val > max1) {
            max2 = max1;
            max1 = val;
            idx1 = i;
        } else if (val > max2) {
            max2 = val;
        }
    }

    if (idx1 < 0) return 0.0f;

    /* Scaled Softmax probability calculation */
    float sum_exp = 0.0f;
    float top1_exp = 1.0f; /* exp((max1 - max1)/T) */
    float inv_temp = 1.0f / temp;

    for (int i = 0; i < vocab; i++) {
        float diff = (logits[i] - max1) * inv_temp;
        if (diff > -16.0f) {
            sum_exp += expf(diff);
        }
    }

    if (sum_exp <= 0.0f) return 0.0f;
    float p_top1 = top1_exp / sum_exp;
    float p_top2 = (max2 > -1e20f) ? (expf((max2 - max1) * inv_temp) / sum_exp) : 0.0f;

    /* Margin confidence combining top probability and separation from runner-up */
    float margin = p_top1 - p_top2;
    float confidence = p_top1 * 0.7f + margin * 0.3f;
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    return confidence;
}

/* -------------------------------------------------------------------------
 * Generation Dispatcher
 * ------------------------------------------------------------------------- */
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
) {
    if (!d || !prompt || prompt_count <= 0 || max_new_tokens <= 0) return -1;
    QwnThinkingConfig default_cfg;
    if (!config) {
        default_cfg = qwn_thinking_default_config(QWN_THINK_MEDIUM);
        config = &default_cfg;
    }

    const float *logits = NULL;
    /* Prompt ingestion with thinking forward */
    for (int i = 0; i < prompt_count; i++) {
        if (qwn_decoder_forward_thinking(d, prompt[i], &logits, config) != 0) {
            return -1;
        }
    }

    int generated = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        /* Greedy sampling for LOW mode; temperature/top_p sampling for others */
        int token = -1;
        if (config->level == QWN_THINK_LOW || temperature <= 0.0f) {
            /* Greedy argmax */
            int best = 0;
            float max_val = logits[0];
            for (int i = 1; i < d->cfg.vocab; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    best = i;
                }
            }
            token = best;
        } else {
            /* Full distribution sampling */
            enum { K = 256 };
            float val[K];
            int id[K], n = 0;
            for (int t = 0; t < d->cfg.vocab; t++) {
                float x = logits[t] / temperature;
                int p;
                if (n < K) p = n++;
                else { if (x <= val[K-1]) continue; p = K-1; }
                while (p > 0 && x > val[p-1]) {
                    if (p < K) { val[p] = val[p-1]; id[p] = id[p-1]; }
                    p--;
                }
                val[p] = x; id[p] = t;
            }
            float peak = val[0], sum = 0.0f;
            for (int i = 0; i < n; i++) { val[i] = expf(val[i] - peak); sum += val[i]; }
            if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
            float cutoff = sum * top_p, cumulative = 0.0f;
            int keep = n;
            for (int i = 0; i < n; i++) {
                cumulative += val[i];
                if (cumulative >= cutoff) { keep = i + 1; break; }
            }
            float kept = 0.0f;
            for (int i = 0; i < keep; i++) kept += val[i];
            uint64_t *rng = &d->rng_state;
            *rng ^= *rng >> 12; *rng ^= *rng << 25; *rng ^= *rng >> 27;
            float r = (float)((*rng * 2685821657736338717ULL) >> 40) * (1.0f / 16777216.0f);
            float target = r * kept;
            token = id[keep - 1];
            for (int i = 0; i < keep; i++) {
                target -= val[i];
                if (target <= 0) { token = id[i]; break; }
            }
        }

        if (token == d->cfg.eos_id) break;

        /* Emit token via callback */
        if (callback) {
            if (token >= 0 && token < d->tokenizer.n_ids && d->tokenizer.id2str && d->tokenizer.id2str[token]) {
                const char *s = d->tokenizer.id2str[token];
                callback(s, (int)strlen(s), opaque);
            } else {
                char text[512];
                int n = tok_decode(&d->tokenizer, &token, 1, text, sizeof(text) - 1);
                if (n > 0) callback(text, n, opaque);
            }
        }

        generated++;
        if (qwn_decoder_forward_thinking(d, token, &logits, config) != 0) {
            return -1;
        }
    }
    return generated;
}
