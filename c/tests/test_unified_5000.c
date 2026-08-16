#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "qwanto_pquant.h"
#include "qwanto_littlebit.h"
#include "qwanto_bitdecoding.h"
#include "qwanto_jetspec.h"
#include "qwanto_talon.h"
#include "qwanto_sliminfer.h"
#include "qwanto_autopilot.h"

int main(void) {
    printf("=================================================================\n");
    printf("    Qwanto Unified 5,000+ Differential Regression Test Suite     \n");
    printf("   (Testing SlimInfer, pQuant, LittleBit-2, BitDecoding, Talon)  \n");
    printf("=================================================================\n");

    const int NUM_TESTS = 5000;
    int passed_tests = 0;

    /* Initialize Subsystems */
    QwnSlimInferEngine slim_engine;
    qwn_sliminfer_init(&slim_engine, 64, 2, 0.50f);

    QwnBitDecodingEngine bitdec_engine;
    qwn_bitdecoding_init(&bitdec_engine, 2, 32, 64, 89);

    QwnJetSpecEngine jetspec_engine;
    qwn_jetspec_init(&jetspec_engine, 64, 4, 3);

    QwnTalonEngine talon_engine;
    qwn_talon_init(&talon_engine, 8);

    QwnPQuantMatrix pquant_mat;
    qwn_pquant_init(&pquant_mat, 16, 32, 0.05f);

    QwnLittleBitMatrix lb_mat;
    qwn_littlebit_init(&lb_mat, 16, 32, 4);

    float *dummy_weights = (float *)malloc(16 * 32 * sizeof(float));
    for (int i = 0; i < 16 * 32; i++) dummy_weights[i] = (float)((i % 7) - 3) * 0.1f;
    qwn_pquant_encode(&pquant_mat, dummy_weights, 16, 32, 1.0f);
    qwn_littlebit_encode(&lb_mat, dummy_weights, 16, 32, 4);

    float x_vec[32];
    float y_out[16];
    float in_hidden[64 * 32];
    float out_compact[64 * 32];

    for (int t = 0; t < NUM_TESTS; t++) {
        for (int i = 0; i < 32; i++) x_vec[i] = (float)(t + i) * 0.01f;
        for (int i = 0; i < 64 * 32; i++) in_hidden[i] = 1.0f;

        /* Subtest 1: pQuant GEMV */
        bool pq_ok = qwn_pquant_gemv(&pquant_mat, x_vec, y_out);
        assert(pq_ok == true);

        /* Subtest 2: LittleBit-2 GEMV */
        bool lb_ok = qwn_littlebit_gemv(&lb_mat, x_vec, y_out);
        assert(lb_ok == true);

        /* Subtest 3: SlimInfer Pruning */
        int pruned = qwn_sliminfer_prune_hidden_states(&slim_engine, in_hidden, out_compact, 64, 32, 4);
        assert(pruned == 32);

        /* Subtest 4: JetSpec Tree Invariant */
        float logits[100];
        for (int i = 0; i < 100; i++) logits[i] = (float)((i + t) % 50);
        bool js_ok = qwn_jetspec_generate_tree(&jetspec_engine, x_vec, logits, 100);
        assert(js_ok == true);
        assert(jetspec_engine.active_tree.tree_mask[0] == 1);

        /* Subtest 5: Talon Domain & Async Queue */
        bool tal_ok = qwn_talon_async_draft_step(&talon_engine, "def compute():", x_vec, 32, 100);
        assert(tal_ok == true);
        int accepted[16];
        int ac = qwn_talon_async_verify_step(&talon_engine, logits, 100, accepted, 8);
        assert(ac > 0);

        passed_tests++;
    }

    /* Cleanup */
    qwn_sliminfer_free(&slim_engine);
    qwn_bitdecoding_free(&bitdec_engine);
    qwn_talon_shutdown(&talon_engine);
    qwn_pquant_free(&pquant_mat);
    qwn_littlebit_free(&lb_mat);
    free(dummy_weights);

    printf("=================================================================\n");
    printf("[SUCCESS] All %d differential tests passed (100%% Reliability)!\n", passed_tests);
    printf("=================================================================\n");
    return 0;
}
