#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "qwanto_decode.h"
#include "qwanto_gpu.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#define qwn_process_id() ((unsigned long)GetCurrentProcessId())
#else
#include <unistd.h>
#define qwn_process_id() ((unsigned long)getpid())
#endif

#define QWN_STR_IMPL(...) #__VA_ARGS__
#define QWN_STR(...) QWN_STR_IMPL(__VA_ARGS__)

#ifndef QWN_BUILD_OPT_FLAGS
#define QWN_OPT_FLAGS_STR "Unavailable"
#else
#define QWN_OPT_FLAGS_STR QWN_STR(QWN_BUILD_OPT_FLAGS)
#endif

static void emit(const char *s,int n,void *opaque){(void)opaque;fwrite(s,1,(size_t)n,stdout);fflush(stdout);}
static double wall_seconds(void);
static const char *g_executable_path = NULL;

typedef struct {
    double started;
    double first_token;
    int callbacks;
} TimedOutput;

static void emit_timed(const char *s, int n, void *opaque) {
    TimedOutput *timed = (TimedOutput *)opaque;
    if (timed->first_token == 0.0) timed->first_token = wall_seconds();
    timed->callbacks++;
    emit(s, n, NULL);
}

static double wall_seconds(void) {
#ifdef _WIN32
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (double)clock() / CLOCKS_PER_SEC;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

static void print_build_info(const QwnRuntimeConfig *config) {
    const char *compiler = "unknown";
    const char *compiler_version = "Unavailable";
#if defined(__clang__)
    compiler = "clang";
    compiler_version = QWN_STR(__clang_major__) "." QWN_STR(__clang_minor__) "." QWN_STR(__clang_patchlevel__);
#elif defined(__GNUC__)
    compiler = "gcc";
    compiler_version = QWN_STR(__GNUC__) "." QWN_STR(__GNUC_MINOR__) "." QWN_STR(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    compiler = "msvc";
    compiler_version = QWN_STR(_MSC_VER);
#endif
    const QwnRuntimeConfig default_config = {0};
    if (!config) config = &default_config;
    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    char binary_sha256[65] = "Unavailable";
    if (!g_executable_path || qwn_sha256_file_hex(g_executable_path, binary_sha256) != 0)
        snprintf(binary_sha256, sizeof(binary_sha256), "Unavailable");
    int runtime_loaded = 0;
#if defined(_OPENMP)
    if (config->cpu_threads > 0) omp_set_num_threads(config->cpu_threads);
    runtime_loaded = omp_get_max_threads() > 0;
#endif
    fprintf(stderr, "qwnrun build: compiler=%s compiler_version=%s "
            "optimization_flags=%s openmp_enabled=%s openmp_runtime_loaded=%s "
            "openmp_version=%s omp_max_threads=%d requested_threads=%d "
             "active_threads=Unavailable actual_executed_kernel=Unavailable "
             "preferred_kernel_candidate=%s "
             "delayed_reduction_compiled=true delayed_reduction_executed=false "
             "compiled_kernels=avx2:%s,vnni:%s "
            "detected_cpu_features=avx2:%s,f16c:%s,fma:%s,vnni:%s,avx512f:%s "
            "binary_sha256=%s "
            "model_dtype=Unavailable backend_requested=%s backend_actual=Unavailable "
             "gpu_kernel_coverage=versioned-qwn-cuda-abi-source-runtime-dll-check-required gpu_matmul_count=0 "
            "cpu_fallback_count=0 pid=%lu\n",
            compiler, compiler_version, QWN_OPT_FLAGS_STR,
#if defined(_OPENMP)
            "true", runtime_loaded ? "true" : "false", QWN_STR(_OPENMP),
            omp_get_max_threads(),
#else
            "false", "false", "Unavailable", 1,
#endif
            config->cpu_threads, qwn_cpu_kernel_name(),
            qwn_cpu_avx2_kernel_compiled() ? "true" : "false",
            qwn_cpu_vnni_kernel_compiled() ? "true" : "false",
            cpu->has_avx2 ? "true" : "false", cpu->has_f16c ? "true" : "false",
            cpu->has_fma ? "true" : "false", cpu->has_vnni ? "true" : "false",
            cpu->has_avx512f ? "true" : "false",
            binary_sha256,
            qwn_runtime_backend_name(config->backend), qwn_process_id());
}

static void print_build_info_json(const QwnRuntimeConfig *config) {
    const char *compiler = "unknown";
    const char *compiler_version = "Unavailable";
#if defined(__clang__)
    compiler = "clang";
    compiler_version = QWN_STR(__clang_major__) "." QWN_STR(__clang_minor__) "." QWN_STR(__clang_patchlevel__);
#elif defined(__GNUC__)
    compiler = "gcc";
    compiler_version = QWN_STR(__GNUC__) "." QWN_STR(__GNUC_MINOR__) "." QWN_STR(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    compiler = "msvc";
    compiler_version = QWN_STR(_MSC_VER);
#endif
    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    char binary_sha256[65] = "Unavailable";
    if (!g_executable_path || qwn_sha256_file_hex(g_executable_path, binary_sha256) != 0)
        snprintf(binary_sha256, sizeof(binary_sha256), "Unavailable");
    int runtime_loaded = 0;
#if defined(_OPENMP)
    if (config && config->cpu_threads > 0) omp_set_num_threads(config->cpu_threads);
    runtime_loaded = omp_get_max_threads() > 0;
#endif
    const QwnRuntimeConfig *cfg = config;
    QwnRuntimeConfig default_config;
    if (!cfg) { qwn_runtime_config_default(&default_config); cfg = &default_config; }
    printf("{\"compiler\":\"%s\",\"compiler_version\":\"%s\","
           "\"optimization_flags\":\"%s\",\"openmp_compiled\":%s,"
           "\"openmp_runtime_loaded\":%s,\"openmp_version\":\"%s\","
           "\"requested_threads\":%d,\"active_threads\":\"Unavailable\","
           "\"cpu_features\":{\"avx2\":%s,\"f16c\":%s,\"fma\":%s,\"vnni\":%s,\"avx512f\":%s},"
           "\"binary_avx2_kernel\":%s,\"binary_vnni_kernel\":%s,"
           "\"compiled_kernels\":{\"avx2\":%s,\"vnni\":%s},"
           "\"detected_cpu_features\":{\"avx2\":%s,\"f16c\":%s,\"fma\":%s,\"vnni\":%s,\"avx512f\":%s},"
           "\"preferred_kernel_candidate\":\"%s\","
           "\"delayed_reduction_compiled\":true,\"delayed_reduction_executed\":false,"
           "\"actual_executed_kernel\":\"Unavailable\","
           "\"selected_isa_kernel\":\"Unavailable\",\"binary_sha256\":\"%s\","
           "\"backend_requested\":\"%s\",\"backend_actual\":\"Unavailable\","
           "\"gpu_matmul_count\":0,\"cpu_fallback_count\":0,"
           "\"gpu_kernel_coverage\":\"versioned-qwn-cuda-abi-source-runtime-dll-check-required\","
           "\"model_dtype\":\"Unavailable\",\"thinking_mode\":\"%s\","
           "\"kv_cache_mode\":\"%s\",\"quantization\":\"%s\",\"kernel_requested\":\"%s\","
           "\"pid\":%lu}\n",
           compiler, compiler_version, QWN_OPT_FLAGS_STR,
#if defined(_OPENMP)
           "true", runtime_loaded ? "true" : "false", QWN_STR(_OPENMP),
#else
           "false", "false", "Unavailable",
#endif
           cfg->cpu_threads,
           cpu->has_avx2 ? "true" : "false", cpu->has_f16c ? "true" : "false",
           cpu->has_fma ? "true" : "false", cpu->has_vnni ? "true" : "false",
           cpu->has_avx512f ? "true" : "false",
           qwn_cpu_avx2_kernel_compiled() ? "true" : "false",
           qwn_cpu_vnni_kernel_compiled() ? "true" : "false",
           qwn_cpu_avx2_kernel_compiled() ? "true" : "false",
           qwn_cpu_vnni_kernel_compiled() ? "true" : "false",
           cpu->has_avx2 ? "true" : "false", cpu->has_f16c ? "true" : "false",
           cpu->has_fma ? "true" : "false", cpu->has_vnni ? "true" : "false",
           cpu->has_avx512f ? "true" : "false",
           qwn_cpu_kernel_name(), binary_sha256,
           qwn_runtime_backend_name(cfg->backend), cfg->thinking_mode,
           cfg->kv_cache_mode, cfg->quantization, cfg->kernel, qwn_process_id());
    fflush(stdout);
}

static void print_runtime_info(const QwnDecoder *decoder) {
    qwn_decoder_refresh_runtime_metrics((QwnDecoder *)decoder);
    const QwnRuntimeMetrics *metrics = qwn_decoder_metrics(decoder);
    const char *dtype = decoder->embed_weight ? qwn_dtype_name(decoder->embed_weight->dtype) : "Unavailable";
    const char *hot_kernel = metrics && metrics->hypervsq2_matmul_count > 0 ?
                             metrics->kernel : "Unavailable";
    const char *backend_actual = metrics && strcmp(metrics->backend, "cuda-pending") == 0 ?
                                 "Unavailable" : (metrics ? metrics->backend : "Unavailable");
    fprintf(stderr, "qwnrun runtime detail: qwn_cuda_dll_loaded=%s model_dtype=%s "
            "preferred_kernel_candidate=%s actual_executed_kernel=%s "
            "hot_path_isa_kernel=%s "
            "backend_requested=%s backend_actual=%s backend=%s kernel=%s "
            "gpu_matmul_count=%llu cpu_fallback_count=%llu gpu_device=%d "
            "cuda_upload_bytes=%llu cuda_resident_bytes=%llu cuda_dll_sha256=%s "
            "requested_threads=%d active_threads=%d openmp_runtime_loaded=%s "
            "hypervsq2_matmul_count=%llu hypervsq2_worker_participations=%llu "
            "hypervsq2_last_active_threads=%d hypervsq2_max_active_threads=%d "
            "activation_sum_precompute_calls=%llu activation_sum_reuse_count=%llu "
            "activation_sum_recompute_count=%llu activation_sum_mode=%s "
             "hypervsq2_logical_weight_bytes=%llu hypervsq2_logical_flops=%llu "
             "hypervsq2_kernel_ms=%.3f swiglu_calls=%llu swiglu_elements=%llu swiglu_ms=%.3f "
             "hypervsq2_reductions_per_row=%d hypervsq2_reduction_mode=%s "
             "delayed_reduction_invocation_count=%llu "
             "row_block_invocation_count=%llu "
             "logical_tensor_visits=%llu logical_repeated_tensor_accesses=%llu "
             "logical_tensors_skipped=%llu logical_embedding_bytes=%llu "
             "logical_attention_bytes=%llu logical_ffn_bytes=%llu "
             "logical_lm_head_bytes=%llu logical_other_weight_bytes=%llu "
             "logical_kv_bytes=%llu logical_activation_bytes=%llu logical_temporary_bytes=%llu "
             "final_lm_head_calls=%llu intermediate_lm_head_calls=%llu "
            "final_lm_head_ms=%.3f intermediate_lm_head_ms=%.3f "
            "early_exit_decisions=%llu layers_skipped=%llu tokens_saved=%llu\n",
            decoder->qwn_cuda.available ? "true" : "false", dtype,
            qwn_cpu_kernel_name(), hot_kernel, hot_kernel,
            qwn_runtime_backend_name(decoder->runtime_config.backend), backend_actual,
            backend_actual,
            metrics ? metrics->kernel : "unknown",
            (unsigned long long)(metrics ? metrics->cuda_matmul_count : 0),
            (unsigned long long)(metrics ? metrics->cpu_fallback_count : 0),
            metrics ? metrics->cuda_device : -1,
            (unsigned long long)(metrics ? metrics->cuda_upload_bytes : 0),
            (unsigned long long)(metrics ? metrics->cuda_resident_bytes : 0),
            metrics ? metrics->cuda_dll_hash : "Unavailable",
            metrics ? metrics->requested_cpu_threads : 0,
            metrics ? metrics->active_cpu_threads : 0,
            metrics && metrics->openmp_runtime_loaded ? "true" : "false",
            (unsigned long long)(metrics ? metrics->hypervsq2_matmul_count : 0),
            (unsigned long long)(metrics ? metrics->hypervsq2_worker_participations : 0),
            metrics ? metrics->hypervsq2_last_active_threads : 0,
            metrics ? metrics->hypervsq2_max_active_threads : 0,
            (unsigned long long)(decoder->scratch.activation_sum_precompute_calls),
            (unsigned long long)(decoder->scratch.activation_sum_reuse_count),
            (unsigned long long)(decoder->scratch.activation_sum_recompute_count),
            decoder->scratch.activation_sum_enabled ? "precomputed" : "recomputed",
            (unsigned long long)(metrics ? metrics->hypervsq2_logical_weight_bytes : 0),
            (unsigned long long)(metrics ? metrics->hypervsq2_logical_flops : 0),
            metrics ? metrics->hypervsq2_kernel_ms : 0.0,
            (unsigned long long)(metrics ? metrics->swiglu_calls : 0),
             (unsigned long long)(metrics ? metrics->swiglu_elements : 0),
             metrics ? metrics->swiglu_ms : 0.0,
             metrics ? metrics->hypervsq2_reductions_per_row : 0,
             metrics ? metrics->hypervsq2_reduction_mode : "Unavailable",
             (unsigned long long)(metrics ? metrics->hypervsq2_delayed_reduction_invocation_count : 0),
             (unsigned long long)(metrics ? metrics->hypervsq2_row_block_invocation_count : 0),
             (unsigned long long)(metrics ? metrics->logical_tensor_visits : 0),
             (unsigned long long)(metrics ? metrics->logical_repeated_tensor_accesses : 0),
             (unsigned long long)(metrics ? metrics->logical_tensors_skipped : 0),
             (unsigned long long)(metrics ? metrics->logical_embedding_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_attention_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_ffn_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_lm_head_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_other_weight_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_kv_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_activation_bytes : 0),
             (unsigned long long)(metrics ? metrics->logical_temporary_bytes : 0),
             (unsigned long long)(metrics ? metrics->final_lm_head_calls : 0),
            (unsigned long long)(metrics ? metrics->intermediate_lm_head_calls : 0),
            metrics ? metrics->final_lm_head_ms : 0.0,
            metrics ? metrics->intermediate_lm_head_ms : 0.0,
            (unsigned long long)(metrics ? metrics->early_exit_decisions : 0),
            (unsigned long long)(metrics ? metrics->layers_skipped : 0),
            (unsigned long long)(metrics ? metrics->tokens_saved : 0));
    fprintf(stderr, "qwnrun dispatch detail: reason=%s\n",
            metrics && metrics->dispatch_reason[0] ? metrics->dispatch_reason : "Unavailable");
    const QwnStartupMetrics *startup = qwn_decoder_startup_metrics(decoder);
    if (startup) {
        fprintf(stderr, "qwnrun startup detail: model_load_ms=%.3f file_open_ms=%.3f "
                "mmap_ms=%.3f metadata_parse_ms=%.3f tokenizer_init_ms=%.3f "
                "kv_cache_alloc_ms=%.3f advisory_preload_ms=%.3f "
                "first_tensor_touch_ms=%.3f first_real_forward_ms=%.3f\n",
                startup->model_load_ms, startup->file_open_ms, startup->mmap_ms,
                startup->metadata_parse_ms, startup->tokenizer_init_ms,
                startup->kv_cache_alloc_ms, startup->advisory_preload_ms,
                startup->first_tensor_touch_ms, startup->first_real_forward_ms);
    }
    fprintf(stderr, "qwnrun runtime: backend=%s cuda_compiled=versioned-abi-source "
            "cuda_dll_loaded=%s memory_backend=mmap prefetch_enabled=true "
            "planned_gpu_bytes=%llu planned_ram_bytes=%llu "
            "planned_nvme_bytes=%llu prefetch_calls=%llu gpu_matmul_count=%llu "
            "cpu_fallback_count=%llu gpu_resident_bytes=%llu\n",
            metrics && metrics->backend[0] ? metrics->backend : "CPU",
            decoder->qwn_cuda.available ? "true" : "false",
            (unsigned long long)decoder->residency.gpu_bytes,
            (unsigned long long)decoder->residency.ram_bytes,
            (unsigned long long)decoder->residency.nvme_bytes,
            (unsigned long long)decoder->prefetch_calls,
            (unsigned long long)(metrics ? metrics->cuda_matmul_count : 0),
            (unsigned long long)(metrics ? metrics->cpu_fallback_count : 0),
            (unsigned long long)(metrics ? metrics->cuda_resident_bytes : 0));
}

typedef struct { const char *id; } ServeOut;
static void emit_mux(const char *s,int n,void *opaque){
    ServeOut *o=(ServeOut*)opaque;
    printf("DATA %s %d\n",o->id,n);fwrite(s,1,(size_t)n,stdout);putchar('\n');fflush(stdout);
}

static int serve_mode(const char *model, const QwnRuntimeConfig *runtime_config){
    QwnRuntimeConfig config;
    if (runtime_config) config = *runtime_config;
    else qwn_runtime_config_default(&config);
    if (getenv("CTX") && atoi(getenv("CTX")) > 0) config.context_size = atoi(getenv("CTX"));
    if (getenv("NGEN") && atoi(getenv("NGEN")) > 0) config.max_tokens = atoi(getenv("NGEN"));
    print_build_info(&config);
    double serve_started = wall_seconds();
    double model_load_started = serve_started;
    QwnDecoder d;const char *error=NULL;
    if(qwn_decoder_open_with_config(&d,model,&config,&error)!=0){fprintf(stderr,"qwnrun: %s\n",error?error:"open");return 1;}
    double model_loaded = wall_seconds();
    print_runtime_info(&d);
    if(getenv("SERVE")){
        printf("\x01\x01READY\x01\x01\nSTAT 0 0.000 0.0 0.0 0 0 "
               "model_load_ms=%.3f runtime_ready_ms=%.3f file_open_ms=%.3f "
               "mmap_ms=%.3f metadata_parse_ms=%.3f tokenizer_init_ms=%.3f "
               "kv_cache_alloc_ms=%.3f advisory_preload_ms=%.3f "
               "first_tensor_touch_ms=%.3f first_real_forward_ms=Unavailable "
               "pid=%lu backend_requested=%s thinking_mode=%s context_size=%d "
               "max_tokens=%d seed=%d kv_cache_mode=%s quantization=%s kernel_requested=%s\n",
               (model_loaded - model_load_started) * 1000.0,
               (wall_seconds() - serve_started) * 1000.0,
               d.startup_metrics.file_open_ms, d.startup_metrics.mmap_ms,
               d.startup_metrics.metadata_parse_ms, d.startup_metrics.tokenizer_init_ms,
               d.startup_metrics.kv_cache_alloc_ms, d.startup_metrics.advisory_preload_ms,
               d.startup_metrics.first_tensor_touch_ms,
               qwn_process_id(),
               qwn_runtime_backend_name(config.backend), config.thinking_mode,
               config.context_size, config.max_tokens, config.seed,
               config.kv_cache_mode, config.quantization, config.kernel);
        fflush(stdout);
    }
    char line[512];
    while(fgets(line,sizeof(line),stdin)){
        char id[64];int slot=0,bytes=0,max_tokens=config.max_tokens;float temp=0,top_p=1;int token_fwd=0;
        if(strncmp(line,"PING",4)==0){
            printf("PONG\n");fflush(stdout);
        }else if(strncmp(line,"CONFIG",6)==0){
            printf("CONFIG dim=%d vocab=%d layers=%d\n",d.cfg.hidden,d.cfg.vocab,d.cfg.layers);fflush(stdout);
        }else if(sscanf(line,"FORWARD %d",&token_fwd)==1){
            const float *l_out=NULL;
            qwn_decoder_forward(&d,token_fwd,&l_out);
            printf("LOGITS %d\n",d.cfg.vocab);
            for(int i=0;i<d.cfg.vocab;i++){
                printf("%f\n",l_out[i]);
            }
            fflush(stdout);
        }else if(sscanf(line,"SUBMIT %63s %d %d %d %f %f",id,&slot,&bytes,&max_tokens,&temp,&top_p)==6){
            double request_started = wall_seconds();
            (void)slot;
            if(bytes<0||bytes>(16<<20)){printf("ERROR %s invalid-prompt-size\n",id);fflush(stdout);continue;}
            if(!isfinite(temp)||!isfinite(top_p)||temp<0.0f||top_p<0.0f||top_p>1.0f){
                printf("ERROR %s invalid-sampling-options\n",id);fflush(stdout);continue;
            }
            char *prompt=(char*)malloc((size_t)bytes+1);if(!prompt){printf("ERROR %s out-of-memory\n",id);fflush(stdout);continue;}
            if(fread(prompt,1,(size_t)bytes,stdin)!=(size_t)bytes){free(prompt);break;}
            prompt[bytes]=0;if(fgetc(stdin)!='\n'){free(prompt);break;}
            int effective_ctx = d.cfg.max_ctx;
            int *ids=(int*)malloc((size_t)effective_ctx*sizeof(int));if(!ids){free(prompt);printf("ERROR %s out-of-memory\n",id);fflush(stdout);continue;}
            int count=tok_encode(&d.tokenizer,prompt,bytes,ids,effective_ctx-1);
            if (count <= 0) {
                count = bytes < effective_ctx - 1 ? bytes : effective_ctx - 1;
                for (int i = 0; i < count; i++) ids[i] = (unsigned char)prompt[i];
            }
            free(prompt);
            if (count <= 0) {
                free(ids);
                printf("ERROR %s empty-prompt\n", id); fflush(stdout);
                continue;
            }
            if(d.cfg.bos_id>=0&&count<effective_ctx){memmove(ids+1,ids,(size_t)count*sizeof(int));ids[0]=d.cfg.bos_id;count++;}
            double reset_started = wall_seconds();
            qwn_decoder_reset(&d);
            double kv_reset_ms = (wall_seconds() - reset_started) * 1000.0;
            ServeOut out={id};
            if (max_tokens <= 0 || max_tokens > config.max_tokens) max_tokens = config.max_tokens;
            int generated;
            if (strcmp(config.thinking_mode, "none") == 0) {
                generated=qwn_decoder_generate(&d,ids,count,max_tokens,temp,top_p,emit_mux,&out);
            } else {
                QwnThinkingLevel level = qwn_thinking_parse_level(config.thinking_mode);
                QwnThinkingConfig thinking = qwn_thinking_default_config(level);
                generated=qwn_decoder_generate_thinking(&d,ids,count,max_tokens,temp,top_p,
                                                        &thinking,emit_mux,&out);
            }
            free(ids);
            if(generated<0){
                fprintf(stderr, "qwnrun result: status=error tokens=0\n");
                printf("ERROR %s generation-failed\n",id);fflush(stdout);continue;
            }
            qwn_decoder_refresh_runtime_metrics(&d);
            const QwnGenerationMetrics *generation = qwn_decoder_generation_metrics(&d);
            double prefill_tps = generation && generation->prefill_ms > 0.0 ?
                (double)generation->prompt_tokens / (generation->prefill_ms / 1000.0) : 0.0;
            double decode_tps = generation && generation->decode_wall_ms > 0.0 ?
                (double)generation->generated_tokens / (generation->decode_wall_ms / 1000.0) : 0.0;
            printf("DONE %s STAT %d %.6f 0 0 %d %d "
                   "prefill_ms=%.3f prefill_tok_per_sec=%.6f "
                   "first_token_ms=%.3f decode_wall_ms=%.3f "
                   "decode_tok_per_sec=%.6f sampling_ms=%.3f kv_reset_ms=%.3f pid=%lu actual_device=%d backend_actual=%s "
                   "kernel=%s cuda_dll_sha256=%s gpu_matmul_count=%llu cpu_fallback_count=%llu "
                   "gpu_kernel_launch_count=%llu gpu_projection_count=%llu "
                   "gpu_upload_count=%llu gpu_upload_bytes=%llu gpu_resident_bytes=%llu "
                   "unsupported_projection_count=%llu "
                   "gpu_kernel_ms=%.3f gpu_transfer_ms=%.3f gpu_sync_ms=%.3f "
                   "cuda_kernel_type=%s cuda_backend_reason=%s "
                   "active_threads=%d dispatch_reason=%s model_dtype=%s "
                   "final_lm_head_calls=%llu intermediate_lm_head_calls=%llu "
                   "final_lm_head_ms=%.3f intermediate_lm_head_ms=%.3f "
                   "early_exit_decisions=%llu layers_skipped=%llu tokens_saved=%llu "
                   "activation_sum_precompute_calls=%llu activation_sum_reuse_count=%llu "
                   "activation_sum_recompute_count=%llu activation_sum_mode=%s "
                   "hypervsq2_logical_weight_bytes=%llu hypervsq2_logical_flops=%llu "
                   "hypervsq2_kernel_ms=%.3f swiglu_calls=%llu swiglu_elements=%llu swiglu_ms=%.3f "
             "hypervsq2_reductions_per_row=%d hypervsq2_reduction_mode=%s "
             "delayed_reduction_invocation_count=%llu "
              "logical_tensor_visits=%llu logical_repeated_tensor_accesses=%llu "
                   "logical_tensors_skipped=%llu logical_embedding_bytes=%llu "
                   "logical_attention_bytes=%llu logical_ffn_bytes=%llu "
                   "logical_lm_head_bytes=%llu logical_other_weight_bytes=%llu "
                   "logical_kv_bytes=%llu logical_activation_bytes=%llu logical_temporary_bytes=%llu "
                   "thinking_mode=%s decode_function=%s config_backend=%s context_size=%d "
                   "max_tokens=%d seed=%d kv_cache_mode=%s kv_cache_mode_actual=%s "
                   "kv_cache_active=%d kv_cache_algorithm=%s kv_cache_kernel=%s "
                   "kv_cache_allocated_bytes=%llu kv_cache_kernel_count=%llu "
                   "kv_cache_upload_bytes=%llu kv_cache_kernel_ms=%.3f "
                   "kv_cache_transfer_ms=%.3f kv_cache_append_count=%llu "
                   "kv_cache_attention_reads=%llu quantization=%s kernel_requested=%s "
                   "temperature=%.8g top_p=%.8g first_real_forward_ms=%.3f "
                   "total_end_to_end_ms=%.3f\n",
                   id, generated, decode_tps, count, generated>=max_tokens,
                   generation ? generation->prefill_ms : 0.0, prefill_tps,
                   generation ? generation->first_token_ms : 0.0,
                   generation ? generation->decode_wall_ms : 0.0, decode_tps,
                   generation ? generation->sampling_ms : 0.0, kv_reset_ms,
                   qwn_process_id(),
                   strcmp(d.runtime_metrics.backend, "cuda") == 0 ?
                       d.runtime_metrics.cuda_device : -1,
                   d.runtime_metrics.backend[0] ? d.runtime_metrics.backend : "Unavailable",
                   d.runtime_metrics.kernel[0] ? d.runtime_metrics.kernel : "Unavailable",
                   d.runtime_metrics.cuda_dll_hash[0] ? d.runtime_metrics.cuda_dll_hash : "Unavailable",
                   (unsigned long long)d.runtime_metrics.cuda_matmul_count,
                   (unsigned long long)d.runtime_metrics.cpu_fallback_count,
                   (unsigned long long)d.runtime_metrics.gpu_kernel_launch_count,
                   (unsigned long long)d.runtime_metrics.gpu_projection_count,
                   (unsigned long long)d.runtime_metrics.gpu_upload_count,
                   (unsigned long long)d.runtime_metrics.cuda_upload_bytes,
                   (unsigned long long)d.runtime_metrics.cuda_resident_bytes,
                   (unsigned long long)d.runtime_metrics.unsupported_projection_count,
                   d.runtime_metrics.gpu_kernel_ms,
                   d.runtime_metrics.gpu_transfer_ms,
                   d.runtime_metrics.gpu_sync_ms,
                   d.runtime_metrics.cuda_kernel_type[0] ? d.runtime_metrics.cuda_kernel_type : "Unavailable",
                   d.runtime_metrics.cuda_backend_reason[0] ? d.runtime_metrics.cuda_backend_reason : "Unavailable",
                   d.runtime_metrics.active_cpu_threads,
                   d.runtime_metrics.dispatch_reason[0] ? d.runtime_metrics.dispatch_reason : "Unavailable",
                   d.embed_weight ? qwn_dtype_name(d.embed_weight->dtype) : "Unavailable",
                   (unsigned long long)d.runtime_metrics.final_lm_head_calls,
                   (unsigned long long)d.runtime_metrics.intermediate_lm_head_calls,
                   d.runtime_metrics.final_lm_head_ms,
                   d.runtime_metrics.intermediate_lm_head_ms,
                   (unsigned long long)d.runtime_metrics.early_exit_decisions,
                   (unsigned long long)d.runtime_metrics.layers_skipped,
                   (unsigned long long)d.runtime_metrics.tokens_saved,
                   (unsigned long long)d.scratch.activation_sum_precompute_calls,
                   (unsigned long long)d.scratch.activation_sum_reuse_count,
                   (unsigned long long)d.scratch.activation_sum_recompute_count,
                   d.scratch.activation_sum_enabled ? "precomputed" : "recomputed",
                   (unsigned long long)d.runtime_metrics.hypervsq2_logical_weight_bytes,
                   (unsigned long long)d.runtime_metrics.hypervsq2_logical_flops,
                   d.runtime_metrics.hypervsq2_kernel_ms,
                   (unsigned long long)d.runtime_metrics.swiglu_calls,
                   (unsigned long long)d.runtime_metrics.swiglu_elements,
                   d.runtime_metrics.swiglu_ms,
                   d.runtime_metrics.hypervsq2_reductions_per_row,
                   d.runtime_metrics.hypervsq2_reduction_mode,
                   (unsigned long long)d.runtime_metrics.hypervsq2_delayed_reduction_invocation_count,
                   (unsigned long long)d.runtime_metrics.logical_tensor_visits,
                   (unsigned long long)d.runtime_metrics.logical_repeated_tensor_accesses,
                   (unsigned long long)d.runtime_metrics.logical_tensors_skipped,
                   (unsigned long long)d.runtime_metrics.logical_embedding_bytes,
                   (unsigned long long)d.runtime_metrics.logical_attention_bytes,
                   (unsigned long long)d.runtime_metrics.logical_ffn_bytes,
                   (unsigned long long)d.runtime_metrics.logical_lm_head_bytes,
                   (unsigned long long)d.runtime_metrics.logical_other_weight_bytes,
                   (unsigned long long)d.runtime_metrics.logical_kv_bytes,
                   (unsigned long long)d.runtime_metrics.logical_activation_bytes,
                   (unsigned long long)d.runtime_metrics.logical_temporary_bytes,
                   config.thinking_mode,
                   strcmp(config.thinking_mode, "none") == 0 ? "qwn_decoder_generate" : "qwn_decoder_generate_thinking",
                   qwn_runtime_backend_name(config.backend), config.context_size, config.max_tokens,
                   config.seed, config.kv_cache_mode,
                   d.runtime_metrics.kv_cache_mode_actual,
                   d.runtime_metrics.kv_cache_active,
                   d.runtime_metrics.kv_cache_algorithm,
                   d.runtime_metrics.kv_cache_kernel,
                   (unsigned long long)d.runtime_metrics.kv_cache_allocated_bytes,
                   (unsigned long long)d.runtime_metrics.kv_cache_kernel_count,
                   (unsigned long long)d.runtime_metrics.kv_cache_upload_bytes,
                   d.runtime_metrics.kv_cache_kernel_ms,
                   d.runtime_metrics.kv_cache_transfer_ms,
                   (unsigned long long)d.runtime_metrics.kv_cache_append_count,
                   (unsigned long long)d.runtime_metrics.kv_cache_attention_reads,
                   config.quantization, config.kernel, temp, top_p,
                   d.startup_metrics.first_real_forward_ms,
                   (wall_seconds() - request_started) * 1000.0);
            fflush(stdout);
            print_runtime_info(&d);
        }else if(sscanf(line,"CANCEL %63s",id)==1){
            printf("ERROR %s CANCELLED\n",id);fflush(stdout);
        }
    }
    qwn_decoder_close(&d);return 0;
}

int main(int argc,char **argv){
    double process_started = wall_seconds();
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);
    _setmode(_fileno(stdout),_O_BINARY);
#endif
    g_executable_path = argc > 0 ? argv[0] : NULL;
    if (argc >= 2 && strcmp(argv[1], "--build-info") == 0) {
        QwnRuntimeConfig config;
        char config_error[256];
        if (qwn_runtime_config_parse(&config, argc, argv, 2,
                                     config_error, sizeof(config_error)) != 0) {
            fprintf(stderr, "qwnrun: %s\n", config_error);
            return 2;
        }
        int json = 0;
        for (int i = 2; i < argc; i++) if (strcmp(argv[i], "--json") == 0) json = 1;
        if (json) print_build_info_json(&config);
        else print_build_info(&config);
        return 0;
    }
    if (argc >= 2 && (strcmp(argv[1], "--list-gpus") == 0 || strcmp(argv[1], "-l") == 0)) {
        qwn_gpu_list_all_devices();
        return 0;
    }
    if (argc >= 3 && strcmp(argv[2], "--serve") == 0) {
        QwnRuntimeConfig config; char config_error[256];
        if (qwn_runtime_config_parse(&config, argc, argv, 2, config_error, sizeof(config_error)) != 0) {
            fprintf(stderr, "qwnrun: %s\n", config_error); return 2;
        }
        return serve_mode(argv[1], &config);
    }
    if (argc >= 2 && strcmp(argv[1], "--serve") == 0) {
        const char *model = getenv("SNAP");
        if (!model || !*model) { fprintf(stderr, "SNAP missing\n"); return 2; }
        QwnRuntimeConfig config; char config_error[256];
        if (qwn_runtime_config_parse(&config, argc, argv, 2, config_error, sizeof(config_error)) != 0) {
            fprintf(stderr, "qwnrun: %s\n", config_error); return 2;
        }
        return serve_mode(model, &config);
    }

    if(getenv("SERVE")){
        const char *model=getenv("SNAP");if(!model||!*model){fprintf(stderr,"SNAP missing\n");return 2;}
        QwnRuntimeConfig config; char config_error[256];
        if (qwn_runtime_config_parse(&config, argc, argv, 1, config_error, sizeof(config_error)) != 0) {
            fprintf(stderr, "qwnrun: %s\n", config_error); return 2;
        }
        return serve_mode(model, &config);
    }
    if(argc<3){
        fprintf(stderr,"usage: qwnrun model.qwn 'prompt' [max_tokens] [ctx] [--backend cpu|cuda|auto] [--gpu-device N] [--threads N] [--ctx-size N] [--max-tokens N] [--kv-cache fp16|q8|turboquant-q4|auto] [--quantization auto|q4_0|hyper_vsq2|fp16|fp32] [--kernel auto|scalar|avx2|vnni] [--seed N]\n");
        return 2;
    }

    const char *model_path = argv[1];
    const char *prompt_str = argv[2];
    QwnRuntimeConfig runtime_config;
    char config_error[256];
    if (qwn_runtime_config_parse(&runtime_config, argc, argv, 3,
                                 config_error, sizeof(config_error)) != 0) {
        fprintf(stderr, "qwnrun: %s\n", config_error);
        return 2;
    }
    bool has_max_flag = false;
    bool has_ctx_flag = false;
    for (int a = 3; a < argc; a++) {
        if (strcmp(argv[a], "--max-tokens") == 0) has_max_flag = true;
        if (strcmp(argv[a], "--ctx-size") == 0) has_ctx_flag = true;
        if (strcmp(argv[a], "--list-gpus") == 0) {
            qwn_gpu_list_all_devices();
            return 0;
        }
    }
    if (!has_max_flag && argc > 3 && argv[3][0] != '-') runtime_config.max_tokens = atoi(argv[3]);
    if (!has_ctx_flag && argc > 4 && argv[4][0] != '-') runtime_config.context_size = atoi(argv[4]);
    if (qwn_runtime_config_validate(&runtime_config, config_error, sizeof(config_error)) != 0) {
        fprintf(stderr, "qwnrun: %s\n", config_error);
        return 2;
    }
    const char *think_text = getenv("QWN_THINKING_LEVEL");
    if (!think_text || !*think_text) think_text = getenv("THINKING");
    if (think_text && *think_text) {
        snprintf(runtime_config.thinking_mode, sizeof(runtime_config.thinking_mode), "%s", think_text);
        if (qwn_runtime_config_validate(&runtime_config, config_error, sizeof(config_error)) != 0) {
            fprintf(stderr, "qwnrun: %s\n", config_error);
            return 2;
        }
    }
    print_build_info(&runtime_config);
    int max_tokens = runtime_config.max_tokens;

    QwnDecoder decoder;const char *error=NULL;
    if(qwn_decoder_open_with_config(&decoder,model_path,&runtime_config,&error)!=0){
        fprintf(stderr,"qwnrun open error: %s\n",error?error:"open failed");return 1;
    }
    print_runtime_info(&decoder);
    int max_prompt=decoder.cfg.max_ctx>8?decoder.cfg.max_ctx-8:decoder.cfg.max_ctx;
    int *ids=(int*)malloc((size_t)max_prompt*sizeof(int));if(!ids){fprintf(stderr,"qwnrun: malloc failed\n");return 1;}
    int count=tok_encode(&decoder.tokenizer,prompt_str,(int)strlen(prompt_str),ids,max_prompt);
    if(count<=0){
        fprintf(stderr,"qwnrun: prompt encoded to zero tokens (vocab size %d)\n", decoder.tokenizer.n_ids);
        /* Fallback: use raw byte tokens */
        count = (int)strlen(prompt_str);
        if(count > max_prompt) count = max_prompt;
        for(int i=0; i<count; i++) ids[i] = (unsigned char)prompt_str[i];
    }
    if(decoder.cfg.bos_id>=0&&count<max_prompt){memmove(ids+1,ids,(size_t)count*sizeof(int));ids[0]=decoder.cfg.bos_id;count++;}
    int valid_count = 0;
    for(int i=0; i<count; i++) {
        if(ids[i] >= 0 && ids[i] < decoder.cfg.vocab) {
            ids[valid_count++] = ids[i];
        }
    }
    if(valid_count == 0) { ids[0] = 1; valid_count = 1; }
    printf("Prompt tokens: %d, generating up to %d tokens...\n", valid_count, max_tokens); fflush(stdout);
    TimedOutput timing = {wall_seconds(), 0.0, 0};
    const char *temp_text = getenv("QWANTO_TEMP");
    const char *top_text = getenv("QWANTO_TOP_P");
    if (!top_text || !*top_text) top_text = getenv("TOPP");
    QwnThinkingLevel think_lvl = qwn_thinking_parse_level(runtime_config.thinking_mode);
    QwnThinkingConfig think_cfg = qwn_thinking_default_config(think_lvl);

    float temperature = temp_text && *temp_text ? strtof(temp_text, NULL) : 0.0f;
    float top_p = top_text && *top_text ? strtof(top_text, NULL) : 1.0f;
    if (!isfinite(temperature) || temperature < 0.0f || !isfinite(top_p) ||
        top_p < 0.0f || top_p > 1.0f) {
        fprintf(stderr, "qwnrun: invalid sampling environment\n");
        free(ids); qwn_decoder_close(&decoder); return 1;
    }
    int rc;
    if (strcmp(runtime_config.thinking_mode, "none") == 0) {
        rc=qwn_decoder_generate(&decoder,ids,valid_count,max_tokens,temperature,top_p,emit_timed,&timing);
    } else {
        rc=qwn_decoder_generate_thinking(&decoder,ids,valid_count,max_tokens,temperature,top_p,
                                         &think_cfg,emit_timed,&timing);
    }
    double elapsed = wall_seconds() - timing.started;
    double total_end_to_end_ms = (wall_seconds() - process_started) * 1000.0;
    putchar('\n');

    if(rc < 0) fprintf(stderr,"qwnrun result: status=error tokens=0\nqwnrun: generate failed (rc=%d)\n", rc);
    else {
        const QwnGenerationMetrics *generation = qwn_decoder_generation_metrics(&decoder);
        double ttft = generation && generation->first_token_ms > 0.0 ?
                      generation->first_token_ms : (timing.first_token > 0.0 ?
                      (timing.first_token - timing.started) * 1000.0 : 0.0);
        double tps = elapsed > 0.0 ? (double)rc / elapsed : 0.0;
        fprintf(stderr,"qwnrun result: status=ok tokens=%d wall_seconds=%.6f "
                "ttft_ms=%.3f tok_per_sec=%.6f thinking_level=%s\n", rc, elapsed, ttft,
                tps, runtime_config.thinking_mode);

        const QwnRuntimeMetrics *metrics = qwn_decoder_metrics(&decoder);
        const QwnStartupMetrics *startup = qwn_decoder_startup_metrics(&decoder);
        fprintf(stderr, "qwnrun result detail: backend=%s kernel=%s gpu_matmul_count=%llu "
                "cpu_fallback_count=%llu active_threads=%d dispatch_reason=%s "
                "decode_function=%s thinking_mode=%s prompt_tokens=%d "
                "prefill_ms=%.3f decode_wall_ms=%.3f sampling_ms=%.3f "
                "prefill_tok_per_sec=%.6f decode_tok_per_sec=%.6f generation_wall_ms=%.3f "
                "process_create_ms=Unavailable file_open_ms=%.3f mmap_ms=%.3f "
                "metadata_parse_ms=%.3f tokenizer_init_ms=%.3f kv_cache_alloc_ms=%.3f "
                "advisory_preload_ms=%.3f first_tensor_touch_ms=%.3f "
                "first_real_forward_ms=%.3f total_end_to_end_ms=%.3f "
                "final_lm_head_calls=%llu intermediate_lm_head_calls=%llu "
                "final_lm_head_ms=%.3f intermediate_lm_head_ms=%.3f "
                "early_exit_decisions=%llu layers_skipped=%llu tokens_saved=%llu "
                "activation_sum_precompute_calls=%llu activation_sum_reuse_count=%llu "
                "activation_sum_recompute_count=%llu activation_sum_mode=%s "
                "hypervsq2_logical_weight_bytes=%llu hypervsq2_logical_flops=%llu "
                "hypervsq2_kernel_ms=%.3f swiglu_calls=%llu swiglu_elements=%llu swiglu_ms=%.3f "
                "hypervsq2_reductions_per_row=%d hypervsq2_reduction_mode=%s "
                "delayed_reduction_invocation_count=%llu "
                "logical_tensor_visits=%llu logical_repeated_tensor_accesses=%llu "
                "logical_tensors_skipped=%llu logical_embedding_bytes=%llu "
                "logical_attention_bytes=%llu logical_ffn_bytes=%llu "
                "logical_lm_head_bytes=%llu logical_other_weight_bytes=%llu "
                "logical_kv_bytes=%llu logical_activation_bytes=%llu logical_temporary_bytes=%llu "
                "config_backend=%s context_size=%d max_tokens=%d seed=%d "
                "kv_cache_mode=%s kv_cache_mode_actual=%s kv_cache_active=%d "
                "kv_cache_algorithm=%s kv_cache_kernel=%s kv_cache_allocated_bytes=%llu "
                "kv_cache_kernel_count=%llu kv_cache_upload_bytes=%llu "
                "kv_cache_kernel_ms=%.3f kv_cache_transfer_ms=%.3f "
                "quantization=%s kernel_requested=%s temperature=%.8g top_p=%.8g\n",
                metrics ? metrics->backend : "unknown",
                metrics ? metrics->kernel : "unknown",
                (unsigned long long)(metrics ? metrics->cuda_matmul_count : 0),
                (unsigned long long)(metrics ? metrics->cpu_fallback_count : 0),
                metrics ? metrics->active_cpu_threads : 0,
                metrics && metrics->dispatch_reason[0] ? metrics->dispatch_reason : "Unavailable",
                strcmp(runtime_config.thinking_mode, "none") == 0 ? "qwn_decoder_generate" : "qwn_decoder_generate_thinking",
                runtime_config.thinking_mode, generation ? generation->prompt_tokens : 0,
                generation ? generation->prefill_ms : 0.0, generation ? generation->decode_wall_ms : 0.0,
                generation ? generation->sampling_ms : 0.0,
                generation && generation->prefill_ms > 0.0 ?
                    (double)generation->prompt_tokens / (generation->prefill_ms / 1000.0) : 0.0,
                generation && generation->decode_wall_ms > 0.0 ?
                    (double)generation->generated_tokens / (generation->decode_wall_ms / 1000.0) : 0.0,
                elapsed * 1000.0,
                startup ? startup->file_open_ms : 0.0, startup ? startup->mmap_ms : 0.0,
                startup ? startup->metadata_parse_ms : 0.0, startup ? startup->tokenizer_init_ms : 0.0,
                startup ? startup->kv_cache_alloc_ms : 0.0, startup ? startup->advisory_preload_ms : 0.0,
                startup ? startup->first_tensor_touch_ms : 0.0,
                startup ? startup->first_real_forward_ms : 0.0, total_end_to_end_ms,
                (unsigned long long)(metrics ? metrics->final_lm_head_calls : 0),
                (unsigned long long)(metrics ? metrics->intermediate_lm_head_calls : 0),
                metrics ? metrics->final_lm_head_ms : 0.0,
                metrics ? metrics->intermediate_lm_head_ms : 0.0,
                (unsigned long long)(metrics ? metrics->early_exit_decisions : 0),
                (unsigned long long)(metrics ? metrics->layers_skipped : 0),
                (unsigned long long)(metrics ? metrics->tokens_saved : 0),
                (unsigned long long)decoder.scratch.activation_sum_precompute_calls,
                (unsigned long long)decoder.scratch.activation_sum_reuse_count,
                (unsigned long long)decoder.scratch.activation_sum_recompute_count,
                decoder.scratch.activation_sum_enabled ? "precomputed" : "recomputed",
                (unsigned long long)(metrics ? metrics->hypervsq2_logical_weight_bytes : 0),
                (unsigned long long)(metrics ? metrics->hypervsq2_logical_flops : 0),
                metrics ? metrics->hypervsq2_kernel_ms : 0.0,
                (unsigned long long)(metrics ? metrics->swiglu_calls : 0),
                (unsigned long long)(metrics ? metrics->swiglu_elements : 0),
                metrics ? metrics->swiglu_ms : 0.0,
                metrics ? metrics->hypervsq2_reductions_per_row : 0,
            metrics ? metrics->hypervsq2_reduction_mode : "Unavailable",
            (unsigned long long)(metrics ? metrics->hypervsq2_delayed_reduction_invocation_count : 0),
             (unsigned long long)(metrics ? metrics->logical_tensor_visits : 0),
                (unsigned long long)(metrics ? metrics->logical_repeated_tensor_accesses : 0),
                (unsigned long long)(metrics ? metrics->logical_tensors_skipped : 0),
                (unsigned long long)(metrics ? metrics->logical_embedding_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_attention_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_ffn_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_lm_head_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_other_weight_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_kv_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_activation_bytes : 0),
                (unsigned long long)(metrics ? metrics->logical_temporary_bytes : 0),
                qwn_runtime_backend_name(runtime_config.backend), runtime_config.context_size,
                runtime_config.max_tokens, runtime_config.seed, runtime_config.kv_cache_mode,
                metrics ? metrics->kv_cache_mode_actual : "Unavailable",
                metrics ? metrics->kv_cache_active : 0,
                metrics ? metrics->kv_cache_algorithm : "Unavailable",
                metrics ? metrics->kv_cache_kernel : "Unavailable",
                (unsigned long long)(metrics ? metrics->kv_cache_allocated_bytes : 0),
                (unsigned long long)(metrics ? metrics->kv_cache_kernel_count : 0),
                (unsigned long long)(metrics ? metrics->kv_cache_upload_bytes : 0),
                metrics ? metrics->kv_cache_kernel_ms : 0.0,
                metrics ? metrics->kv_cache_transfer_ms : 0.0,
                runtime_config.quantization, runtime_config.kernel, temperature, top_p);
    }
    free(ids);qwn_decoder_close(&decoder);return rc<0?1:0;
}
