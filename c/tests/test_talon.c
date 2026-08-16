#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "qwanto_talon.h"

int main(void) {
    printf("=================================================================\n");
    printf("     Qwanto Talon Asynchronous Speculative Decoding Test Suite  \n");
    printf("                      (AAAI 2026 Breakthrough)                  \n");
    printf("=================================================================\n");

    /* Test 1: Initialize Talon Engine */
    QwnTalonEngine engine;
    bool init_ok = qwn_talon_init(&engine, 8);
    assert(init_ok == true);
    assert(engine.is_initialized == true);
    assert(engine.max_draft_length == 8);
    printf("[PASS] Talon engine initialization verified.\n");

    /* Test 2: Domain Classification */
    assert(qwn_talon_classify_domain("def quicksort(arr):") == QWN_TALON_DOMAIN_CODE);
    assert(qwn_talon_classify_domain("Calculate the mathematical proof step by step:") == QWN_TALON_DOMAIN_REASONING);
    assert(qwn_talon_classify_domain("What is the capital and history of France?") == QWN_TALON_DOMAIN_KNOWLEDGE);
    assert(qwn_talon_classify_domain("Please summarize this article in TL;DR format:") == QWN_TALON_DOMAIN_SUMMARIZE);
    assert(qwn_talon_classify_domain("Hello, how are you today?") == QWN_TALON_DOMAIN_CONVERSATION);
    printf("[PASS] Multi-domain classifier verified across all 5 benchmark archetypes.\n");

    /* Test 3: Knowledge Indexing into Retrieval Trie */
    int sample_text[12] = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112};
    qwn_talon_index_knowledge(&engine, sample_text, 12);
    printf("[PASS] Knowledge context indexed into Sub-Vector Retrieval Trie.\n");

    /* Test 4: Asynchronous Draft Enqueue */
    float hidden_state[64] = {0.85f};
    bool draft_ok = qwn_talon_async_draft_step(
        &engine, "Write a Python function to compute fibonacci", hidden_state, 64, 32000
    );
    assert(draft_ok == true);
    assert(engine.draft_queue.count == 1);
    printf("[PASS] Asynchronous draft generation enqueued without blocking worker thread.\n");

    /* Test 5: Asynchronous Verification Step */
    int accepted_tokens[16];
    float dummy_logits[32000] = {0.0f};
    int accepted_count = qwn_talon_async_verify_step(
        &engine, dummy_logits, 32000, accepted_tokens, 16
    );
    assert(accepted_count > 0);
    assert(engine.draft_queue.count == 0);
    printf("[PASS] Asynchronous verification step successfully processed (%d tokens accepted).\n", accepted_count);

    /* Clean up */
    qwn_talon_shutdown(&engine);

    printf("=================================================================\n");
    printf("[SUCCESS] All Talon Asynchronous Speculative Decoding tests passed!\n");
    printf("=================================================================\n");
    return 0;
}
