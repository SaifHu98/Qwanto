#include "qwanto_saguro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int qwn_saguaro2_init(
    QwnSaguaro2Engine *engine,
    int initial_draft_length,
    bool enable_multimodal
) {
    if (!engine) return -1;

    memset(engine, 0, sizeof(*engine));
    engine->current_draft_len = (initial_draft_length >= 3 && initial_draft_length <= 15) ? initial_draft_length : 8;
    engine->entropy_threshold = 0.45f;
    engine->multimodal_enabled = enable_multimodal;

    return 0;
}

int qwn_saguaro2_push_draft(
    QwnSaguaro2Engine *engine,
    int token_id,
    float confidence,
    int modality
) {
    if (!engine) return -1;

    int next_tail = (engine->tail + 1) % QWN_SAGUARO_RING_BUFFER_SIZE;
    if (next_tail == engine->head) {
        /* Ring buffer full: advance head */
        engine->head = (engine->head + 1) % QWN_SAGUARO_RING_BUFFER_SIZE;
    }

    QwnDraftToken *tok = &engine->ring_buffer[engine->tail];
    tok->token_id = token_id;
    tok->confidence = confidence;
    tok->modality = modality;
    tok->accepted = false;

    engine->tail = next_tail;
    engine->total_speculated++;

    return 0;
}

int qwn_saguaro2_verify_pyramid(
    QwnSaguaro2Engine *engine,
    const int *target_logits_top1,
    int n_tokens_to_verify,
    int *out_accepted_tokens,
    int *out_accepted_count
) {
    if (!engine || !target_logits_top1 || !out_accepted_tokens || !out_accepted_count) return -1;

    int accepted_count = 0;
    int curr = engine->head;

    for (int i = 0; i < n_tokens_to_verify && curr != engine->tail; i++) {
        QwnDraftToken *tok = &engine->ring_buffer[curr];

        /* Multi-modal entropy check if enabled */
        bool pass_entropy = true;
        if (engine->multimodal_enabled && tok->modality > 0) {
            float entropy = -tok->confidence * logf(tok->confidence > 1e-6f ? tok->confidence : 1e-6f);
            if (entropy > engine->entropy_threshold) pass_entropy = false;
        }

        if (tok->token_id == target_logits_top1[i] && pass_entropy) {
            tok->accepted = true;
            out_accepted_tokens[accepted_count++] = tok->token_id;
            engine->total_accepted++;
        } else {
            /* Mismatch: stop sequential verification */
            break;
        }
        curr = (curr + 1) % QWN_SAGUARO_RING_BUFFER_SIZE;
    }

    *out_accepted_count = accepted_count;
    engine->head = curr;

    /* Dynamic draft length adaptation based on rolling acceptance */
    float rate = (float)accepted_count / (float)(n_tokens_to_verify > 0 ? n_tokens_to_verify : 1);
    if (rate >= 0.85f && engine->current_draft_len < 15) {
        engine->current_draft_len++;
    } else if (rate < 0.60f && engine->current_draft_len > 3) {
        engine->current_draft_len--;
    }

    return 0;
}

float qwn_saguaro2_measured_speedup(const QwnSaguaro2Engine *engine) {
    if (!engine || engine->baseline_tok_per_sec <= 0.0f ||
        engine->speculative_tok_per_sec < 0.0f) return 0.0f;
    return engine->speculative_tok_per_sec / engine->baseline_tok_per_sec;
}

int qwn_saguaro2_record_measurement(QwnSaguaro2Engine *engine,
                                    float baseline_tok_per_sec,
                                    float speculative_tok_per_sec) {
    if (!engine || !isfinite(baseline_tok_per_sec) ||
        !isfinite(speculative_tok_per_sec) || baseline_tok_per_sec <= 0.0f ||
        speculative_tok_per_sec < 0.0f) return -1;
    engine->baseline_tok_per_sec = baseline_tok_per_sec;
    engine->speculative_tok_per_sec = speculative_tok_per_sec;
    return 0;
}
