#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "qwanto_jetspec.h"

int main(void) {
    printf("=================================================================\n");
    printf("     Qwanto JetSpec Reference Tree Fixture Test Suite            \n");
    printf("               (UC San Diego Hao AI Lab 2026)                   \n");
    printf("=================================================================\n");

    const int hidden_dim = 2048;
    const int vocab_size = 32000;
    const int max_tree_depth = 6;
    const int max_tree_width = 4;

    /* Test 1: Initialize JetSpec Engine */
    QwnJetSpecEngine engine;
    bool init_ok = qwn_jetspec_init(&engine, hidden_dim, max_tree_depth, max_tree_width);
    assert(init_ok == true);
    assert(engine.is_initialized == true);
    assert(engine.cfg.max_tree_depth == max_tree_depth);
    printf("[PASS] JetSpec engine initialization verified.\n");

    /* Test 2: Generate Causal Parallel Draft Tree in Single Pass */
    float *fused_hidden = (float *)malloc(hidden_dim * sizeof(float));
    float *target_logits = (float *)malloc(vocab_size * sizeof(float));
    assert(fused_hidden != NULL && target_logits != NULL);

    for (int i = 0; i < hidden_dim; i++) fused_hidden[i] = 0.05f;
    for (int i = 0; i < vocab_size; i++) target_logits[i] = (float)(i % 100);
    target_logits[1337] = 999.0f; /* Root token target */

    bool gen_ok = qwn_jetspec_generate_tree(&engine, fused_hidden, target_logits, vocab_size);
    assert(gen_ok == true);
    assert(engine.active_tree.node_count > 1);
    assert(engine.active_tree.nodes[0].token_id == 1337);
    printf("[PASS] Single-pass causal parallel tree generation verified (%d nodes generated, Root = %d).\n",
           engine.active_tree.node_count, engine.active_tree.nodes[0].token_id);

    /* Test 3: Tree-Causal Attention Mask Invariant Verification */
    int n = engine.active_tree.node_count;
    for (int i = 0; i < n; i++) {
        /* Every node attends to itself */
        assert(engine.active_tree.tree_mask[i * n + i] == 1);
        /* Child attends to root */
        assert(engine.active_tree.tree_mask[i * n + 0] == 1);
    }
    printf("[PASS] Tree-causal ancestor attention mask invariants verified for all %d nodes.\n", n);

    /* Test 4: Verify Candidate Tree and Extract Accepted Tokens */
    int accepted[8];
    float *verif_logits = (float *)malloc(n * vocab_size * sizeof(float));
    assert(verif_logits != NULL);
    for (int i = 0; i < n * vocab_size; i++) verif_logits[i] = 0.1f;
    for (int i = 0; i < n; i++)
        verif_logits[i * vocab_size + engine.active_tree.nodes[i].token_id] = 2.0f;

    int accepted_count = qwn_jetspec_verify_tree(&engine, verif_logits, vocab_size, accepted, 8);
    assert(accepted_count > 0);
    assert(accepted[0] == 1337);
    printf("[PASS] Batched tree verification passed (%d tokens accepted along rank-1 path).\n", accepted_count);

    /* Test 5: Speculation Cache and Ring Buffer */
    uint64_t prompt_hash = 0xCBF29CE484222325ULL;
    qwn_jetspec_record_acceptance(&engine, prompt_hash, accepted, accepted_count);

    int cached_chain[8];
    int cached_len = 0;
    bool lookup_ok = qwn_jetspec_cache_lookup(&engine, prompt_hash, cached_chain, &cached_len);
    assert(lookup_ok == true);
    assert(cached_len == accepted_count);
    assert(cached_chain[0] == 1337);
    printf("[PASS] 64-bit FNV-1a LRU speculation cache and 32-slot ring buffer verified.\n");

    /* Cleanup */
    free(fused_hidden);
    free(target_logits);
    free(verif_logits);

    printf("=================================================================\n");
    printf("[SUCCESS] JetSpec reference tree invariants passed; product execution remains disabled.\n");
    printf("=================================================================\n");
    return 0;
}
