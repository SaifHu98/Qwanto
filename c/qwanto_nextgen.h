#ifndef QWANTO_NEXTGEN_H
#define QWANTO_NEXTGEN_H

#include "qwanto_twla.h"
#include "qwanto_turboquant.h"
#include "qwanto_spectral.h"
#include "qwanto_pagedeviction.h"
#include "qwanto_saguro.h"
#include "qwanto_sparsity.h"
#include "qwanto_fused.h"
#include "qwn_container.h"
#include "qwanto_autopilot.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Qwanto Next-Gen Master Unified Engine Architecture
 * ------------------------------------------------------------------------- */

typedef struct {
    QwnAutoPilotConfig autopilot;
    QwnSpectralRouter moe_router;
    QwnPagedEvictionPool kv_virtual_pool;
    QwnSaguaro2Engine speculative_engine;
    QwnAdaptiveSparsityContext sparsity_ctx;
    QwnContainer container;
    bool is_initialized;
    double measured_throughput_tps;
    double measured_memory_gb;
} QwnNextGenEngine;

/* Initialize Next-Gen Unified Engine */
int qwn_nextgen_init(QwnNextGenEngine *engine, const char *model_path);

/* Execute hardware-saturated forward step */
int qwn_nextgen_forward_step(
    QwnNextGenEngine *engine,
    const float *input_hidden_state,
    int seq_len,
    int hidden_dim,
    float *out_logits
);

/* Clean up Next-Gen Unified Engine */
void qwn_nextgen_free(QwnNextGenEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_NEXTGEN_H */
