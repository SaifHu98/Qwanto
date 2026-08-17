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
#endif

static void emit(const char *s,int n,void *opaque){(void)opaque;fwrite(s,1,(size_t)n,stdout);fflush(stdout);}
static double wall_seconds(void);

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

static int runtime_active_threads(void) {
#if defined(_OPENMP)
    int active = 1;
    #pragma omp parallel
    {
        #pragma omp single
        active = omp_get_num_threads();
    }
    return active;
#else
    return 1;
#endif
}

static void print_build_info(void) {
    const char *compiler = "unknown";
#if defined(__clang__)
    compiler = "clang";
#elif defined(__GNUC__)
    compiler = "gcc";
#elif defined(_MSC_VER)
    compiler = "msvc";
#endif
    const char *optimization_flags = "unknown";
#if defined(__OPTIMIZE__)
    optimization_flags = "optimized";
#endif
#if defined(_OPENMP)
    fprintf(stderr, "qwnrun build: compiler=%s openmp_enabled=true "
            "openmp_runtime=%d omp_max_threads=%d active_threads=%d "
            "optimization_flags=%s selected_isa_kernel=%s model_dtype=Unavailable "
            "backend=unselected gpu_kernel_coverage=hypervsq2-74,q4_0 cpu_fallback_count=0\n",
            compiler, _OPENMP, omp_get_max_threads(), runtime_active_threads(),
            optimization_flags, qwn_cpu_kernel_name());
#else
    fprintf(stderr, "qwnrun build: compiler=%s openmp_enabled=false "
            "openmp_runtime=none omp_max_threads=1 active_threads=1 "
            "optimization_flags=%s selected_isa_kernel=%s model_dtype=Unavailable "
            "backend=unselected gpu_kernel_coverage=hypervsq2-74,q4_0 cpu_fallback_count=0\n",
            compiler, optimization_flags, qwn_cpu_kernel_name());
#endif
}

static void print_runtime_info(const QwnDecoder *decoder) {
    const QwnRuntimeMetrics *metrics = qwn_decoder_metrics(decoder);
    const char *dtype = decoder->embed_weight ? qwn_dtype_name(decoder->embed_weight->dtype) : "Unavailable";
    fprintf(stderr, "qwnrun runtime detail: qwn_cuda_dll_loaded=%s model_dtype=%s "
            "selected_isa_kernel=%s backend=%s kernel=%s gpu_matmul_count=%llu "
            "cpu_fallback_count=%llu gpu_device=%d cuda_upload_bytes=%llu "
            "cuda_resident_bytes=%llu cuda_dll_sha256=%s\n",
            decoder->qwn_cuda.available ? "true" : "false", dtype,
            qwn_cpu_kernel_name(), metrics ? metrics->backend : "unknown",
            metrics ? metrics->kernel : "unknown",
            (unsigned long long)(metrics ? metrics->cuda_matmul_count : 0),
            (unsigned long long)(metrics ? metrics->cpu_fallback_count : 0),
            metrics ? metrics->cuda_device : -1,
            (unsigned long long)(metrics ? metrics->cuda_upload_bytes : 0),
            (unsigned long long)(metrics ? metrics->cuda_resident_bytes : 0),
            metrics ? metrics->cuda_dll_hash : "Unavailable");
#ifdef COLI_CUDA
    if (decoder->cuda_enabled) {
        fprintf(stderr, "qwnrun runtime: backend=CUDA cuda_compiled=true "
                "cuda_dll_loaded=true cuda_devices=%d memory_backend=mmap "
                "prefetch_enabled=true planned_gpu_bytes=%llu "
                "planned_ram_bytes=%llu planned_nvme_bytes=%llu "
                "prefetch_calls=%llu\n", decoder->cuda_device_count,
                (unsigned long long)decoder->residency.gpu_bytes,
                (unsigned long long)decoder->residency.ram_bytes,
                (unsigned long long)decoder->residency.nvme_bytes,
                (unsigned long long)decoder->prefetch_calls);
        return;
    }
    fprintf(stderr, "qwnrun runtime: backend=CPU cuda_compiled=true "
            "cuda_dll_loaded=false memory_backend=mmap prefetch_enabled=true "
            "planned_gpu_bytes=%llu planned_ram_bytes=%llu "
            "planned_nvme_bytes=%llu prefetch_calls=%llu\n",
            (unsigned long long)decoder->residency.gpu_bytes,
            (unsigned long long)decoder->residency.ram_bytes,
            (unsigned long long)decoder->residency.nvme_bytes,
            (unsigned long long)decoder->prefetch_calls);
#else
    fprintf(stderr, "qwnrun runtime: backend=CPU cuda_compiled=false "
            "cuda_dll_loaded=false memory_backend=mmap prefetch_enabled=true "
            "planned_gpu_bytes=%llu planned_ram_bytes=%llu "
            "planned_nvme_bytes=%llu prefetch_calls=%llu\n",
            (unsigned long long)decoder->residency.gpu_bytes,
            (unsigned long long)decoder->residency.ram_bytes,
            (unsigned long long)decoder->residency.nvme_bytes,
            (unsigned long long)decoder->prefetch_calls);
#endif
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
    QwnDecoder d;const char *error=NULL;
    if(qwn_decoder_open_with_config(&d,model,&config,&error)!=0){fprintf(stderr,"qwnrun: %s\n",error?error:"open");return 1;}
    print_runtime_info(&d);
    if(getenv("SERVE")){
        printf("\x01\x01READY\x01\x01\nSTAT 0 0 0 0\n");fflush(stdout);
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
            qwn_decoder_reset(&d);ServeOut out={id};double start=wall_seconds();
            if (max_tokens <= 0 || max_tokens > config.max_tokens) max_tokens = config.max_tokens;
            int generated=qwn_decoder_generate(&d,ids,count,max_tokens,temp,top_p,emit_mux,&out);free(ids);
            if(generated<0){
                fprintf(stderr, "qwnrun result: status=error tokens=0\n");
                printf("ERROR %s generation-failed\n",id);fflush(stdout);continue;
            }
            double sec=wall_seconds()-start;
            double tps=sec>0?generated/sec:0;
            printf("DONE %s STAT %d %.3f 0 0 %d %d\n",id,generated,tps,count,generated>=max_tokens);fflush(stdout);
        }else if(sscanf(line,"CANCEL %63s",id)==1){
            printf("ERROR %s CANCELLED\n",id);fflush(stdout);
        }
    }
    qwn_decoder_close(&d);return 0;
}

int main(int argc,char **argv){
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);
    _setmode(_fileno(stdout),_O_BINARY);
#endif
    if (argc == 2 && strcmp(argv[1], "--build-info") == 0) {
        print_build_info();
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

    print_build_info();
    if(getenv("SERVE")){
        const char *model=getenv("SNAP");if(!model||!*model){fprintf(stderr,"SNAP missing\n");return 2;}
        QwnRuntimeConfig config; char config_error[256];
        if (qwn_runtime_config_parse(&config, argc, argv, 1, config_error, sizeof(config_error)) != 0) {
            fprintf(stderr, "qwnrun: %s\n", config_error); return 2;
        }
        return serve_mode(model, &config);
    }
    if(argc<3){
        fprintf(stderr,"usage: qwnrun model.qwn 'prompt' [max_tokens] [ctx] [--backend cpu|cuda|auto] [--gpu-device N] [--threads N] [--ctx-size N] [--max-tokens N] [--kv-cache fp16] [--quantization auto|q4_0|hyper_vsq2|fp16|fp32] [--kernel auto|scalar|avx2|vnni] [--seed N]\n");
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
    if (!temp_text || !*temp_text) temp_text = getenv("TEMP");
    const char *top_text = getenv("QWANTO_TOP_P");
    if (!top_text || !*top_text) top_text = getenv("TOPP");
    const char *think_text = getenv("QWN_THINKING_LEVEL");
    if (!think_text || !*think_text) think_text = getenv("THINKING");

    QwnThinkingLevel think_lvl = QWN_THINK_MEDIUM;
    if (think_text) think_lvl = qwn_thinking_parse_level(think_text);
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--thinking") == 0 && a + 1 < argc) {
            think_lvl = qwn_thinking_parse_level(argv[a+1]);
        }
    }
    QwnThinkingConfig think_cfg = qwn_thinking_default_config(think_lvl);

    float temperature = temp_text && *temp_text ? strtof(temp_text, NULL) : 0.0f;
    float top_p = top_text && *top_text ? strtof(top_text, NULL) : 1.0f;
    if (!isfinite(temperature) || temperature < 0.0f || !isfinite(top_p) ||
        top_p < 0.0f || top_p > 1.0f) {
        fprintf(stderr, "qwnrun: invalid sampling environment\n");
        free(ids); qwn_decoder_close(&decoder); return 1;
    }
    int rc=qwn_decoder_generate_thinking(&decoder,ids,valid_count,max_tokens,temperature,top_p,&think_cfg,emit_timed,&timing);
    double elapsed = wall_seconds() - timing.started;
    putchar('\n');

    if(rc < 0) fprintf(stderr,"qwnrun result: status=error tokens=0\nqwnrun: generate failed (rc=%d)\n", rc);
    else {
        double ttft = timing.first_token > 0.0 ?
                      (timing.first_token - timing.started) * 1000.0 : 0.0;
        double tps = elapsed > 0.0 ? (double)rc / elapsed : 0.0;
        fprintf(stderr,"qwnrun result: status=ok tokens=%d wall_seconds=%.6f "
                "ttft_ms=%.3f tok_per_sec=%.6f thinking_level=%s\n", rc, elapsed, ttft,
                tps, qwn_thinking_level_name(think_lvl));

        const QwnRuntimeMetrics *metrics = qwn_decoder_metrics(&decoder);
        fprintf(stderr, "qwnrun result detail: backend=%s kernel=%s gpu_matmul_count=%llu "
                "cpu_fallback_count=%llu\n",
                metrics ? metrics->backend : "unknown",
                metrics ? metrics->kernel : "unknown",
                (unsigned long long)(metrics ? metrics->cuda_matmul_count : 0),
                (unsigned long long)(metrics ? metrics->cpu_fallback_count : 0));
    }
    free(ids);qwn_decoder_close(&decoder);return rc<0?1:0;
}
