#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "qwanto_sliminfer.h"

int main(void) {
    printf("=================================================================\n");
    printf("     Qwanto SlimInfer Dynamic Token Pruning Test Suite           \n");
    printf("                 (AAAI 2026 Breakthrough)                       \n");
    printf("=================================================================\n");

    const int max_seq_len = 128;
    const int hidden_dim = 64;
    const int n_heads = 4;

    /* Test 1: Initialize SlimInfer Engine */
    QwnSlimInferEngine engine;
    bool init_ok = qwn_sliminfer_init(&engine, max_seq_len, 4, 0.50f);
    assert(init_ok == true);
    assert(engine.is_initialized == true);
    assert(engine.cfg.start_prune_layer == 4);
    printf("[PASS] SlimInfer engine initialization verified.\n");

    /* Test 2: Compute Salience Scores */
    float *attn_mat = (float *)malloc((size_t)n_heads * max_seq_len * max_seq_len * sizeof(float));
    assert(attn_mat != NULL);
    for (int i = 0; i < n_heads * max_seq_len * max_seq_len; i++) attn_mat[i] = 0.01f;

    qwn_sliminfer_compute_salience(&engine, attn_mat, n_heads, max_seq_len);
    /* Verify Sink tokens received high boost */
    assert(engine.importance_scores[0] > 1000.0f);
    assert(engine.importance_scores[1] > 1000.0f);
    printf("[PASS] Information diffusion salience scores and attention sink protection verified.\n");

    /* Test 3: Prune Hidden States at Intermediate Layer 4 */
    float *in_hidden = (float *)malloc((size_t)max_seq_len * hidden_dim * sizeof(float));
    float *compact_hidden = (float *)malloc((size_t)max_seq_len * hidden_dim * sizeof(float));
    assert(in_hidden != NULL && compact_hidden != NULL);

    for (int i = 0; i < max_seq_len * hidden_dim; i++) in_hidden[i] = 1.0f;

    int retained = qwn_sliminfer_prune_hidden_states(
        &engine, in_hidden, compact_hidden, max_seq_len, hidden_dim, 4
    );
    assert(retained == 64); /* 50% retention */
    assert(engine.retained_count == 64);
    assert(engine.retained_indices[0] == 0); /* Sink preserved */
    printf("[PASS] Dynamic fine-grained token pruning verified (128 -> %d tokens retained, 50%% reduction).\n", retained);

    /* Test 4: Scatter Compact Output */
    float *full_out = (float *)malloc((size_t)max_seq_len * hidden_dim * sizeof(float));
    assert(full_out != NULL);
    qwn_sliminfer_scatter_output(&engine, compact_hidden, full_out, hidden_dim);
    assert(full_out[0] == 1.0f);
    printf("[PASS] Token scattering and output alignment verified.\n");

    /* Cleanup */
    qwn_sliminfer_free(&engine);
    free(attn_mat);
    free(in_hidden);
    free(compact_hidden);
    free(full_out);

    printf("=================================================================\n");
    printf("[SUCCESS] All SlimInfer Dynamic Token Pruning tests passed!\n");
    printf("=================================================================\n");
    return 0;
}
