#include "qwanto_sparsity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int qwn_sparsity_init(
    QwnAdaptiveSparsityContext *ctx,
    int n_heads,
    float initial_sparsity_ratio
) {
    if (!ctx || n_heads <= 0 || n_heads > QWN_SPARSITY_MAX_HEADS) return -1;

    memset(ctx, 0, sizeof(*ctx));
    ctx->n_heads = n_heads;
    ctx->sparsity_ratio = (initial_sparsity_ratio >= 0.0f && initial_sparsity_ratio <= 0.8f) ? initial_sparsity_ratio : 0.25f;
    ctx->energy_threshold = 0.01f;

    for (int i = 0; i < n_heads; i++) {
        ctx->head_active_mask[i] = 1;
        ctx->head_importance[i] = 1.0f;
    }
    ctx->active_heads_count = n_heads;

    return 0;
}

int qwn_sparsity_prune_heads(
    QwnAdaptiveSparsityContext *ctx,
    const float *head_activations,
    int head_dim
) {
    if (!ctx || !head_activations || head_dim <= 0) return -1;

    int n_heads = ctx->n_heads;
    float max_energy = 0.0f;

    /* Compute L2 energy per head */
    for (int h = 0; h < n_heads; h++) {
        const float *ha = head_activations + h * head_dim;
        float energy = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            energy += ha[d] * ha[d];
        }
        ctx->head_importance[h] = energy;
        if (energy > max_energy) max_energy = energy;
    }

    /* Prune heads below dynamic threshold */
    float cutoff = max_energy * ctx->sparsity_ratio;
    int active = 0;

    for (int h = 0; h < n_heads; h++) {
        if (ctx->head_importance[h] >= cutoff) {
            ctx->head_active_mask[h] = 1;
            active++;
        } else {
            ctx->head_active_mask[h] = 0;
        }
    }

    /* Ensure at least 1 head remains active */
    if (active == 0) {
        ctx->head_active_mask[0] = 1;
        active = 1;
    }

    ctx->active_heads_count = active;
    return active;
}

int qwn_sparsity_prune_mlp_neurons(
    const float *mlp_intermediate,
    float *out_sparse_intermediate,
    int intermediate_dim,
    float threshold,
    int *out_active_neuron_count
) {
    if (!mlp_intermediate || !out_sparse_intermediate || intermediate_dim <= 0) return -1;

    int active = 0;
    for (int i = 0; i < intermediate_dim; i++) {
        float val = mlp_intermediate[i];
        if (fabsf(val) > threshold) {
            out_sparse_intermediate[i] = val;
            active++;
        } else {
            out_sparse_intermediate[i] = 0.0f;
        }
    }

    if (out_active_neuron_count) *out_active_neuron_count = active;
    return 0;
}
