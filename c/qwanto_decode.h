#ifndef QWANTO_DECODE_H
#define QWANTO_DECODE_H

#include "qwanto_native.h"
#include "qwanto_kernels.h"
#include "qwn_paged_kv.h"
#include "tok.h"
#include "qwanto_thinking.h"
#include "qwn_runtime_config.h"
#include "cuda/qwn_cuda_abi.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

typedef struct {
    int hidden, intermediate, layers, heads, kv_heads, head_dim;
    int q_head_dim, k_head_dim, v_head_dim;
    int vocab, max_ctx, bos_id, eos_id;
    float rms_eps, rope_theta;
    int tie_embeddings;
    int is_qwen35;
    int total_layers, mtp_layers, mtp_layer_start;
    int ssm_inner, ssm_state, ssm_groups, ssm_dt_rank, ssm_conv_kernel;
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
    const QwnTensorDesc *ssm_qkv, *ssm_gate;
    const QwnTensorDesc *ssm_alpha, *ssm_beta, *ssm_a, *ssm_dt;
    const QwnTensorDesc *ssm_conv1d, *ssm_norm, *ssm_out;
    int q_out;            /* = shape[1] of q_proj, 0 if no q_proj */
    int q_proj_out;       /* raw q projection width (Qwen3.5 includes Q+gate) */
    int q_gate_out;       /* gate width paired with q_out, normally q_out */
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

#include "qwanto_turboquant.h"

typedef struct {
    void *handle;
    QwnCudaAbiQueryFn query;
    QwnCudaCapabilityQueryFn get_capabilities;
    QwnCudaEnumerateDevicesFn enumerate_devices;
    QwnCudaContextCreateFn context_create;
    QwnCudaContextDestroyFn context_destroy;
    QwnCudaTensorUploadFn upload_tensor;
    QwnCudaTensorReleaseFn release_tensor;
    QwnCudaKvCreateFn kv_cache_create;
    QwnCudaKvDestroyFn kv_cache_destroy;
    QwnCudaKvAppendFn kv_cache_append;
    QwnCudaKvAttentionFn kv_cache_attention;
    QwnCudaKvResetFn kv_cache_reset;
    QwnCudaGemvFn gemv_hypervsq2;
    QwnCudaGemmFn gemm_hypervsq2;
    QwnCudaSynchronizeFn synchronize;
    QwnCudaTelemetryFn get_telemetry;
    QwnCudaLastErrorFn last_error;
    QwnCudaAbiInfo capabilities;
    QwnCudaContextHandle context;
    struct {
        const QwnTensorDesc *desc;
        QwnCudaTensorHandle tensor;
    } weights[QWN_CUDA_MAX_RESIDENT_TENSORS];
    uint32_t weight_count;
    QwnCudaKvCacheHandle kv_caches[QWN_CUDA_MAX_RESIDENT_KV_CACHES];
    uint32_t kv_cache_count;
    int available;
} QwnCudaRuntime;

typedef struct {
    uint64_t cuda_matmul_count;
    uint64_t cpu_fallback_count;
    uint64_t cuda_upload_bytes;
    size_t cuda_resident_bytes;
    int cuda_device;
    uint64_t gpu_kernel_launch_count;
    uint64_t gpu_projection_count;
    uint64_t gpu_upload_count;
    uint64_t unsupported_projection_count;
    double gpu_kernel_ms;
    double gpu_transfer_ms;
    double gpu_sync_ms;
    char cuda_kernel_type[32];
    char cuda_backend_reason[128];
    int requested_cpu_threads;
    int active_cpu_threads;
    int openmp_runtime_loaded;
    uint64_t hypervsq2_matmul_count;
    uint64_t hypervsq2_worker_participations;
    int hypervsq2_last_active_threads;
    int hypervsq2_max_active_threads;
    uint64_t final_lm_head_calls;
    uint64_t intermediate_lm_head_calls;
    double final_lm_head_ms;
    double intermediate_lm_head_ms;
    uint64_t early_exit_decisions;
    uint64_t layers_skipped;
    uint64_t tokens_saved;
    uint64_t hypervsq2_logical_weight_bytes;
    uint64_t hypervsq2_logical_flops;
    uint64_t hypervsq2_delayed_reduction_invocation_count;
    uint64_t hypervsq2_row_block_invocation_count;
    uint64_t logical_tensor_visits;
    uint64_t logical_repeated_tensor_accesses;
    uint64_t logical_tensors_skipped;
    uint64_t logical_embedding_bytes;
    uint64_t logical_attention_bytes;
    uint64_t logical_ffn_bytes;
    uint64_t logical_lm_head_bytes;
    uint64_t logical_other_weight_bytes;
    uint64_t logical_kv_bytes;
    uint64_t logical_activation_bytes;
    uint64_t logical_temporary_bytes;
    int kv_cache_active;
    uint64_t kv_cache_append_count;
    uint64_t kv_cache_attention_reads;
    char kv_cache_kernel[32];
    char kv_cache_algorithm[32];
    char kv_cache_mode_actual[32];
    uint64_t kv_cache_allocated_bytes;
    uint64_t kv_cache_kernel_count;
    uint64_t kv_cache_upload_bytes;
    double kv_cache_kernel_ms;
    double kv_cache_transfer_ms;
    uint64_t kv_cache_cpu_fallback_count;
    double hypervsq2_kernel_ms;
    int hypervsq2_reductions_per_row;
    char hypervsq2_reduction_mode[32];
    uint64_t swiglu_calls;
    uint64_t swiglu_elements;
    double swiglu_ms;
    char dispatch_reason[256];
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
    char model_sha256[65];
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
    float *ssm_states;
    float *ssm_conv_states;
    size_t ssm_state_layer_stride;
    size_t ssm_conv_layer_stride;
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
    QwnQ8Cache *q8_layers;
    int use_q8_kv;
    int use_cuda_q8_kv;
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
