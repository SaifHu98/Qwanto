#ifndef QWANTO_DECODE_H
#define QWANTO_DECODE_H

#include "qwanto_native.h"
#include "qwanto_kernels.h"
#include "qwn_paged_kv.h"
#include "tok.h"
#include "qwanto_thinking.h"
#include "qwn_runtime_config.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

typedef struct {
    int hidden, intermediate, layers, heads, kv_heads, head_dim;
    int q_head_dim, k_head_dim, v_head_dim;
    int vocab, max_ctx, bos_id, eos_id;
    float rms_eps, rope_theta;
    int tie_embeddings;
} QwnConfig;

/* Per-layer tensor descriptor cache — resolved once at load, zero lookup cost at runtime.
 * Per-layer output dimensions (q_out, kv_out) are derived from the actual
 * tensor shapes so models with variable head_dim per layer
 * (Qwen3.5 hybrid attention/SSM, some GQA variants) load and run
 * correctly.  ``head_dim`` in QwnConfig is treated as a fallback when
 * per-layer dims are unavailable.
 */
typedef struct {
    const QwnTensorDesc *q_proj, *k_proj, *v_proj, *o_proj;
    const QwnTensorDesc *q_bias, *k_bias, *v_bias, *o_bias;
    const QwnTensorDesc *q_norm, *k_norm;
    const QwnTensorDesc *input_norm, *post_norm;
    const QwnTensorDesc *gate_proj, *up_proj, *down_proj;
    int q_out;            /* = shape[1] of q_proj, 0 if no q_proj */
    int k_out;            /* = shape[1] of k_proj, 0 if no k_proj */
    int v_out;            /* = shape[1] of v_proj, 0 if no v_proj */
    int q_head_dim;
    int k_head_dim;
    int v_head_dim;
    int ffn_in;           /* = shape[0] of up_proj, 0 if missing */
    int ffn_out;          /* = shape[1] of gate_proj / up_proj, 0 if missing */
    int ffn_down_out;     /* = shape[1] of down_proj, 0 if missing */
    int is_moe;           /* 1 if this layer has routed experts */
    int is_ssm;           /* 1 if this layer is SSM (skip attention entirely) */
} QwnLayerTensors;

typedef int (*QwnCudaInitFn)(int gpu_id);
typedef int (*QwnCudaGemvFn)(int rows, int cols, const void *weights,
                            const float *x, float *out);
typedef void (*QwnCudaShutdownFn)(void);
typedef struct {
    uint64_t matmul_count;
    uint64_t upload_bytes;
    size_t resident_bytes;
    int device_id;
    char kernel[32];
} QwnCudaMetricsSnapshot;
typedef int (*QwnCudaGetMetricsFn)(QwnCudaMetricsSnapshot *metrics);

#include "qwanto_turboquant.h"

typedef struct {
    void *handle;
    QwnCudaInitFn init;
    QwnCudaGemvFn gemv_hypervsq2;
    QwnCudaGemvFn gemv_q4_0;
    QwnCudaGetMetricsFn get_metrics;
    QwnCudaShutdownFn shutdown;
    int available;
} QwnCudaRuntime;

typedef struct {
    uint64_t cuda_matmul_count;
    uint64_t cpu_fallback_count;
    uint64_t cuda_upload_bytes;
    size_t cuda_resident_bytes;
    int cuda_device;
    int requested_cpu_threads;
    int active_cpu_threads;
    int openmp_runtime_loaded;
    uint64_t hypervsq2_matmul_count;
    uint64_t hypervsq2_worker_participations;
    int hypervsq2_last_active_threads;
    int hypervsq2_max_active_threads;
    char dispatch_reason[128];
    char backend[16];
    char kernel[32];
    char cuda_dll_hash[65];
} QwnRuntimeMetrics;

typedef struct {
    int prompt_tokens;
    int generated_tokens;
    double prefill_ms;
    double first_token_ms;
    double decode_wall_ms;
    double sampling_ms;
} QwnGenerationMetrics;

typedef struct {
    double model_load_ms;
    double file_open_ms;
    double mmap_ms;
    double metadata_parse_ms;
    double tokenizer_init_ms;
    double kv_cache_alloc_ms;
    double advisory_preload_ms;
    double first_tensor_touch_ms;
    double first_real_forward_ms;
} QwnStartupMetrics;

int qwn_sha256_file_hex(const char *path, char output[65]);

typedef struct QwnDecoder {
    QwnModel model;
    QwnConfig cfg;
    Tok tokenizer;
    QwnScratch scratch;
    void *arena;
    size_t arena_bytes;
    uint16_t *key_cache;
    uint16_t *value_cache;
    void *kv_allocation;
    float *x, *xb, *q, *k, *v, *att, *ctx, *gate, *up, *hidden, *logits;
    float *norm_weights;
    int position;
    QwnLayerTensors *layer_cache; /* resolved at load time */
    const QwnTensorDesc *embed_weight;
    const QwnTensorDesc *lm_head_weight;
    const QwnTensorDesc *final_norm_weight;
    QwnResidencyPlan residency;
    QwnPlacement *residency_items;
    uint64_t prefetch_calls;
    QwnCudaRuntime qwn_cuda;
    QwnKVBlockPool paged_kv;
    QwnBlockTable paged_table;
    uint16_t *kv_gather_key;
    uint16_t *kv_gather_value;
    size_t kv_gather_stride;
    int use_paged_kv;
    TurboQuantCache *turboquant_layers;
    int use_turboquant;
    float *rope_cos_cache;
    float *rope_sin_cache;
    int rope_cache_ctx;
    int rope_cache_half;
    uint64_t rng_state;
    QwnRuntimeConfig runtime_config;
    QwnRuntimeMetrics runtime_metrics;
    QwnGenerationMetrics generation_metrics;
    QwnStartupMetrics startup_metrics;
    double startup_started_seconds;
#ifdef COLI_CUDA
    int cuda_devices[COLI_CUDA_MAX_DEVICES];
    int cuda_device_count;
    int cuda_enabled;
    size_t cuda_resident_bytes[COLI_CUDA_MAX_DEVICES];
    size_t cuda_budget_bytes[COLI_CUDA_MAX_DEVICES];
    struct { const QwnTensorDesc *desc; ColiCudaTensor *tensor; int device; } cuda_weights[128];
    int cuda_weight_count;
#endif
} QwnDecoder;

int qwn_decoder_open(QwnDecoder *d, const char *path, int ctx_size,
                     const char **error);
int qwn_decoder_open_with_config(QwnDecoder *d, const char *path,
                                 const QwnRuntimeConfig *config,
                                 const char **error);
const QwnRuntimeMetrics *qwn_decoder_metrics(const QwnDecoder *d);
void qwn_decoder_refresh_runtime_metrics(QwnDecoder *d);
const QwnGenerationMetrics *qwn_decoder_generation_metrics(const QwnDecoder *d);
const QwnStartupMetrics *qwn_decoder_startup_metrics(const QwnDecoder *d);
void qwn_decoder_close(QwnDecoder *d);
void qwn_decoder_reset(QwnDecoder *d);

/* Consume one token and return logits predicting the next token. */
int qwn_decoder_forward(QwnDecoder *d, int token, const float **logits);

/* Forward with configurable thinking depth and early exit */
int qwn_decoder_forward_thinking(QwnDecoder *d, int token, const float **logits,
                                QwnThinkingConfig *config);

/* Greedy decode. callback receives each decoded byte chunk. */
int qwn_decoder_generate(QwnDecoder *d, const int *prompt, int prompt_count,
                         int max_new_tokens, float temperature, float top_p,
                         void (*callback)(const char *, int, void *), void *opaque);

#endif
