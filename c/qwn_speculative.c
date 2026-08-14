#include "qwn_speculative.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float spec_rand_float(void) {
    static uint64_t seed = 0x853c49e6748fea9bULL;
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    return (float)((seed * 2685821657736338717ULL) >> 40) * (1.0f / 16777216.0f);
}

int qwn_speculative_init(QwnSpecContext *ctx, QwnDecoder *target, QwnDecoder *draft, int gamma) {
    if (!ctx || !target || !draft) return -1;
    ctx->target = target;
    ctx->draft = draft;
    ctx->gamma = (gamma > 0 && gamma <= 16) ? gamma : 4;
    ctx->total_drafted = 0;
    ctx->total_accepted = 0;
    ctx->temperature = 0.0f;
    ctx->top_p = 1.0f;
    return 0;
}

int qwn_speculative_generate(QwnSpecContext *ctx,
                            const int *prompt, int prompt_count,
                            int max_new_tokens,
                            float temperature, float top_p,
                            void(*callback)(const char*, int, void*),
                            void *opaque) {
    if (!ctx || !ctx->target || !ctx->draft || !prompt || prompt_count <= 0 || max_new_tokens <= 0)
        return -1;

    const float *target_logits = NULL;
    const float *draft_logits = NULL;

    /* Ingest prompt into both target and draft models */
    for (int i = 0; i < prompt_count; i++) {
        if (qwn_decoder_forward(ctx->target, prompt[i], &target_logits) != 0) return -1;
        if (qwn_decoder_forward(ctx->draft, prompt[i], &draft_logits) != 0) return -1;
    }

    int generated = 0;
    int draft_tokens[16];

    while (generated < max_new_tokens) {
        int lookahead = ctx->gamma;
        if (generated + lookahead > max_new_tokens)
            lookahead = max_new_tokens - generated;

        int draft_pos_start = ctx->draft->position;
        int target_pos_start = ctx->target->position;

        /* Step 1: Draft model generates lookahead speculative tokens */
        for (int k = 0; k < lookahead; k++) {
            /* Greedy / Argmax or Temperature sample from draft */
            int best = 0;
            int vocab = ctx->draft->cfg.vocab;
            for (int v = 1; v < vocab; v++) {
                if (draft_logits[v] > draft_logits[best]) best = v;
            }
            draft_tokens[k] = best;
            ctx->total_drafted++;
            if (k + 1 < lookahead) {
                if (qwn_decoder_forward(ctx->draft, best, &draft_logits) != 0) break;
            }
        }

        /* Step 2: Target model verifies draft tokens in parallel */
        int accepted_in_round = 0;
        for (int k = 0; k < lookahead; k++) {
            int tok = draft_tokens[k];
            if (qwn_decoder_forward(ctx->target, tok, &target_logits) != 0) break;

            int target_best = 0;
            int vocab = ctx->target->cfg.vocab;
            for (int v = 1; v < vocab; v++) {
                if (target_logits[v] > target_logits[target_best]) target_best = v;
            }

            /* Verification condition: greedy match or probability threshold */
            if (tok == target_best || target_logits[tok] >= target_logits[target_best] * 0.95f) {
                accepted_in_round++;
                ctx->total_accepted++;
                generated++;

                if (callback) {
                    if (tok >= 0 && tok < ctx->target->tokenizer.n_ids && ctx->target->tokenizer.id2str && ctx->target->tokenizer.id2str[tok]) {
                        const char *s = ctx->target->tokenizer.id2str[tok];
                        callback(s, (int)strlen(s), opaque);
                    }
                }

                if (tok == ctx->target->cfg.eos_id) {
                    return generated;
                }
            } else {
                /* Rejection: roll back draft position to match target acceptance */
                ctx->draft->position = draft_pos_start + accepted_in_round;
                ctx->target->position = target_pos_start + accepted_in_round;
                break;
            }
        }

        /* If none were accepted, emit target token directly */
        if (accepted_in_round == 0) {
            int best = 0;
            int vocab = ctx->target->cfg.vocab;
            for (int v = 1; v < vocab; v++) {
                if (target_logits[v] > target_logits[best]) best = v;
            }
            generated++;
            if (callback && best >= 0 && best < ctx->target->tokenizer.n_ids && ctx->target->tokenizer.id2str && ctx->target->tokenizer.id2str[best]) {
                const char *s = ctx->target->tokenizer.id2str[best];
                callback(s, (int)strlen(s), opaque);
            }
            if (qwn_decoder_forward(ctx->target, best, &target_logits) != 0) return -1;
            if (qwn_decoder_forward(ctx->draft, best, &draft_logits) != 0) return -1;
        }
    }

    return generated;
}

float qwn_speculative_acceptance_rate(const QwnSpecContext *ctx) {
    if (!ctx || ctx->total_drafted <= 0) return 0.0f;
    return ((float)ctx->total_accepted / (float)ctx->total_drafted) * 100.0f;
}
