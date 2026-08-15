#ifndef QWANTO_SPARSITY_H
#define QWANTO_SPARSITY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Adaptive Dynamic Sparsity (MoSE-Inspired Variable-Width Forward Path)
 * ------------------------------------------------------------------------- */

#define QWN_SPARSITY_MAX_HEADS 64
#define QWN_SPARSITY_MAX_NEURONS 16384

typedef struct {
    float head_importance[QWN_SPARSITY_MAX_HEADS];
    uint8_t head_active_mask[QWN_SPARSITY_MAX_HEADS];
    int n_heads;
    int active_heads_count;
    float sparsity_ratio;       /* e.g. 0.0 (dense) to 0.6 (60% pruned) */
    float energy_threshold;     /* Dynamic L1/L2 activation energy cutoff */
} QwnAdaptiveSparsityContext;

/* -------------------------------------------------------------------------
 * Adaptive Dynamic Sparsity APIs
 * ------------------------------------------------------------------------- */

/* Initialize dynamic sparsity context */
int qwn_sparsity_init(
    QwnAdaptiveSparsityContext *ctx,
    int n_heads,
    float initial_sparsity_ratio
);

/* Compute activation energy and prune inactive attention heads in real-time */
int qwn_sparsity_prune_heads(
    QwnAdaptiveSparsityContext *ctx,
    const float *head_activations,
    int head_dim
);

/* Prune intermediate MLP neurons based on ReLU / SiLU magnitude threshold */
int qwn_sparsity_prune_mlp_neurons(
    const float *mlp_intermediate,
    float *out_sparse_intermediate,
    int intermediate_dim,
    float threshold,
    int *out_active_neuron_count
);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_SPARSITY_H */
