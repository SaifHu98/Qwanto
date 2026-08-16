#ifndef QWANTO_BITDECODING_H
#define QWANTO_BITDECODING_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * BitDecoding Tensor Core Low-Bit KV-Cache Engine (HPCA 2026)
 * Unlocks hardware Tensor Cores (wgmma / mma.sync / WMMA) for sub-4-bit KV caches.
 * Achieves 7.5x decoding speedup over FP16 FlashDecoding and 2x over CUDA cores.
 * ------------------------------------------------------------------------- */

typedef enum {
    QWN_TC_ARCH_GENERIC  = 0,
    QWN_TC_ARCH_AMPERE   = 1, /* SM80, SM86 */
    QWN_TC_ARCH_ADA      = 2, /* SM89 */
    QWN_TC_ARCH_HOPPER   = 3, /* SM90 - WGMMA async copy */
    QWN_TC_ARCH_BLACKWELL= 4  /* SM100 / SM120 - NVFP4 / FP4 Tensor Cores */
} QwnTensorCoreArch;

typedef enum {
    QWN_BITDEC_LAYOUT_LINEAR    = 0,
    QWN_BITDEC_LAYOUT_SWIZZLED  = 1, /* 16x16 / 16x32 WMMA swizzled tile */
    QWN_BITDEC_LAYOUT_NVFP4     = 2  /* Blackwell NVFP4 native tensor layout */
} QwnBitDecLayoutType;

typedef struct {
    QwnTensorCoreArch tc_arch;
    QwnBitDecLayoutType layout_type;
    int warp_size;
    int mma_tile_m;
    int mma_tile_n;
    int mma_tile_k;
    bool has_wgmma_support;
    bool has_nvfp4_support;
    uint32_t sm_version;
} QwnBitDecodingConfig;

typedef struct {
    QwnBitDecodingConfig cfg;
    void *swizzled_k_cache;
    void *swizzled_v_cache;
    size_t k_cache_bytes;
    size_t v_cache_bytes;
    int max_seq_len;
    int n_heads;
    int head_dim;
    bool is_initialized;
} QwnBitDecodingEngine;

/* -------------------------------------------------------------------------
 * BitDecoding APIs
 * ------------------------------------------------------------------------- */

/* Detect GPU architecture and initialize BitDecoding configuration */
bool qwn_bitdecoding_init(
    QwnBitDecodingEngine *engine,
    int n_heads,
    int head_dim,
    int max_seq_len,
    uint32_t sm_version
);

/* Swizzle TurboQuant/Low-bit linear KV cache into Tensor Core MMA tiled layout */
bool qwn_bitdecoding_pack_kv(
    QwnBitDecodingEngine *engine,
    const uint8_t *linear_k_packed,
    const uint8_t *linear_v_packed,
    int seq_len
);

/* Execute hardware Tensor Core accelerated attention decoding step */
bool qwn_bitdecoding_attention_step(
    QwnBitDecodingEngine *engine,
    const float *q_query_heads,     /* [n_heads, head_dim] */
    float *out_context_heads,       /* [n_heads, head_dim] */
    int seq_len,
    float sm_scale
);

/* Free allocated BitDecoding structures and GPU buffers */
void qwn_bitdecoding_free(QwnBitDecodingEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_BITDECODING_H */
