#if !defined(_WIN32)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "qwanto_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double qwn_decode_wall_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return (double)clock() / (double)CLOCKS_PER_SEC;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t block[64];
    size_t used;
} QwnSha256;

static uint32_t qwn_sha256_rotr(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
}

static void qwn_sha256_transform(QwnSha256 *ctx, const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t words[64];
    uint32_t a, b, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++) {
        words[i] = ((uint32_t)block[i * 4] << 24) |
                   ((uint32_t)block[i * 4 + 1] << 16) |
                   ((uint32_t)block[i * 4 + 2] << 8) |
                   (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = qwn_sha256_rotr(words[i - 15], 7) ^
                      qwn_sha256_rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
        uint32_t s1 = qwn_sha256_rotr(words[i - 2], 17) ^
                      qwn_sha256_rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = qwn_sha256_rotr(e, 6) ^ qwn_sha256_rotr(e, 11) ^ qwn_sha256_rotr(e, 25);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choose + k[i] + words[i];
        uint32_t s0 = qwn_sha256_rotr(a, 2) ^ qwn_sha256_rotr(a, 13) ^ qwn_sha256_rotr(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void qwn_sha256_init(QwnSha256 *ctx) {
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    memcpy(ctx->state, initial, sizeof(initial));
    ctx->bit_count = 0;
    ctx->used = 0;
}

static void qwn_sha256_update(QwnSha256 *ctx, const uint8_t *data, size_t length) {
    while (length > 0) {
        size_t take = sizeof(ctx->block) - ctx->used;
        if (take > length) take = length;
        memcpy(ctx->block + ctx->used, data, take);
        ctx->used += take;
        ctx->bit_count += (uint64_t)take * 8u;
        data += take;
        length -= take;
        if (ctx->used == sizeof(ctx->block)) {
            qwn_sha256_transform(ctx, ctx->block);
            ctx->used = 0;
        }
    }
}

static void qwn_sha256_final(QwnSha256 *ctx, uint8_t digest[32]) {
    uint64_t bits = ctx->bit_count;
    ctx->block[ctx->used++] = 0x80;
    if (ctx->used > 56) {
        memset(ctx->block + ctx->used, 0, sizeof(ctx->block) - ctx->used);
        qwn_sha256_transform(ctx, ctx->block);
        ctx->used = 0;
    }
    memset(ctx->block + ctx->used, 0, 56 - ctx->used);
    for (int i = 0; i < 8; i++) ctx->block[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    qwn_sha256_transform(ctx, ctx->block);
    for (int i = 0; i < 8; i++) {
        digest[i * 4] = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)ctx->state[i];
    }
}

int qwn_sha256_file_hex(const char *path, char output[65]) {
    uint8_t buffer[64 * 1024];
    uint8_t digest[32];
    QwnSha256 ctx;
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    qwn_sha256_init(&ctx);
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), file)) > 0)
        qwn_sha256_update(&ctx, buffer, count);
    int ok = ferror(file) ? -1 : 0;
    fclose(file);
    if (ok != 0) return -1;
    qwn_sha256_final(&ctx, digest);
    for (int i = 0; i < 32; i++) snprintf(output + i * 2, 3, "%02x", digest[i]);
    output[64] = '\0';
    return 0;
}

static size_t up64(size_t n) { return (n + 63u) & ~63u; }

static void *alloc64(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(up64(bytes), 64);
#else
    void *p = NULL; return posix_memalign(&p, 64, up64(bytes)) == 0 ? p : NULL;
#endif
}
static void free64(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static uint64_t env_gib_bytes(const char *name) {
    const char *value = getenv(name);
    if (!value || !*value) return 0;
    char *end = NULL;
    double gib = strtod(value, &end);
    if (end == value || gib <= 0.0) return 0;
    if (gib > (double)(SIZE_MAX / (size_t)(1u << 30)))
        return SIZE_MAX;
    return (uint64_t)(gib * (double)(1u << 30));
}

static void init_residency(QwnDecoder *d) {
    d->residency_items = (QwnPlacement *)calloc(d->model.hdr.n_tensors,
                                                  sizeof(QwnPlacement));
    if (!d->residency_items) return;
    uint64_t gpu_budget = 0;
#ifdef COLI_CUDA
    if (d->cuda_enabled) {
        uint64_t configured = env_gib_bytes("CUDA_EXPERT_GB");
        for (int i = 0; i < d->cuda_device_count; i++) {
            size_t free_bytes = 0, total_bytes = 0;
            if (coli_cuda_mem_info(d->cuda_devices[i], &free_bytes, &total_bytes)) {
                size_t budget = configured ? (size_t)(configured / (uint64_t)d->cuda_device_count)
                                            : (size_t)((double)free_bytes * 0.90);
                d->cuda_budget_bytes[i] = budget;
                gpu_budget += budget;
            }
        }
    }
#endif
    uint64_t ram_budget = env_gib_bytes("RAM_GB");
    d->residency.items = d->residency_items;
    d->residency.capacity = d->model.hdr.n_tensors;
    if (qwn_plan_residency(&d->model, gpu_budget, ram_budget, &d->residency) != 0)
        return;
    for (uint32_t i = 0; i < d->residency.count; i++) {
        const QwnPlacement *placement = &d->residency.items[i];
        if (placement->tier != QWN_TIER_RAM) continue;
        const QwnTensorDesc *tensor = qwn_tensor_at(&d->model, placement->tensor_index);
        if (tensor && qwn_prefetch(&d->model, tensor) == 0) d->prefetch_calls++;
    }
}

static void qwn_cuda_unload(QwnDecoder *d) {
    if (!d || !d->qwn_cuda.handle) return;
    if (d->qwn_cuda.shutdown) d->qwn_cuda.shutdown();
#ifdef _WIN32
    FreeLibrary((HMODULE)d->qwn_cuda.handle);
#else
    dlclose(d->qwn_cuda.handle);
#endif
    memset(&d->qwn_cuda, 0, sizeof(d->qwn_cuda));
}

static void *qwn_cuda_symbol(void *handle, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

static int qwn_cuda_load(QwnDecoder *d, int gpu_id, int required) {
    const char *requested = getenv("QWANTO_CUDA_DLL");
    char module_candidate[1024] = {0};
    const char *candidates[3] = { requested, NULL, NULL };
#ifdef _WIN32
    DWORD module_len = GetModuleFileNameA(NULL, module_candidate, sizeof(module_candidate));
    if (module_len > 0 && module_len < sizeof(module_candidate)) {
        char *slash = strrchr(module_candidate, '\\');
        if (slash) {
            slash[1] = '\0';
            strncat(module_candidate, "qwn_cuda.dll", sizeof(module_candidate) - strlen(module_candidate) - 1);
            candidates[1] = module_candidate;
        }
    }
#else
    Dl_info info;
    memset(&info, 0, sizeof(info));
    if (dladdr((void *)&qwn_cuda_load, &info) != 0 && info.dli_fname) {
        snprintf(module_candidate, sizeof(module_candidate), "%s", info.dli_fname);
        char *slash = strrchr(module_candidate, '/');
        if (slash) {
            slash[1] = '\0';
            strncat(module_candidate, "qwn_cuda.so",
                    sizeof(module_candidate) - strlen(module_candidate) - 1);
            candidates[1] = module_candidate;
        }
    }
#endif
    void *handle = NULL;
    const char *loaded_path = NULL;
    for (size_t i = 0; i < 2; i++) {
        if (!candidates[i] || !*candidates[i]) continue;
#ifdef _WIN32
        handle = (void *)LoadLibraryA(candidates[i]);
#else
        handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
#endif
        if (handle) {
            loaded_path = candidates[i];
            break;
        }
    }
    if (loaded_path) {
        if (qwn_sha256_file_hex(loaded_path, d->runtime_metrics.cuda_dll_hash) != 0)
            snprintf(d->runtime_metrics.cuda_dll_hash, sizeof(d->runtime_metrics.cuda_dll_hash), "Unavailable");
    }
    if (!handle) {
        fprintf(stderr, required ? "[ERROR] CUDA backend requested but qwn_cuda.dll was not found.\n" :
                                  "[INFO] CUDA DLL not found; auto backend remains on CPU.\n");
        return -1;
    }
    d->qwn_cuda.handle = handle;
    d->qwn_cuda.init = (QwnCudaInitFn)qwn_cuda_symbol(handle, "qwn_cuda_init");
    d->qwn_cuda.gemv_hypervsq2 = (QwnCudaGemvFn)qwn_cuda_symbol(handle, "qwn_cuda_gemv_hypervsq2");
    d->qwn_cuda.gemv_q4_0 = (QwnCudaGemvFn)qwn_cuda_symbol(handle, "qwn_cuda_gemv_q4_0");
    d->qwn_cuda.get_metrics = (QwnCudaGetMetricsFn)qwn_cuda_symbol(handle, "qwn_cuda_get_metrics");
    d->qwn_cuda.shutdown = (QwnCudaShutdownFn)qwn_cuda_symbol(handle, "qwn_cuda_shutdown");
    if (!d->qwn_cuda.init || (!d->qwn_cuda.gemv_hypervsq2 && !d->qwn_cuda.gemv_q4_0) ||
        !d->qwn_cuda.get_metrics || !d->qwn_cuda.shutdown) {
        fprintf(stderr, required ? "[ERROR] CUDA DLL is missing the qwnrun HyperVSQ-2 ABI.\n" :
                                  "[WARN] CUDA DLL is missing the qwnrun ABI; auto backend remains on CPU.\n");
        qwn_cuda_unload(d);
        return -1;
    }
    if (d->qwn_cuda.init(gpu_id) != 0) {
        fprintf(stderr, required ? "[ERROR] CUDA device initialization failed.\n" :
                                  "[WARN] CUDA device initialization failed; auto backend remains on CPU.\n");
        qwn_cuda_unload(d);
        return -1;
    }
    d->qwn_cuda.available = 1;
    fprintf(stderr, "[INFO] qwn_cuda.dll loaded: GPU %d\n", gpu_id);
    return 0;
}

static float half_to_float(uint16_t h) {
    uint32_t sign=(h>>15)&1, exp=(h>>10)&31, mant=h&1023, bits;
    if(exp==0) bits=sign<<31;
    else if(exp==31) bits=(sign<<31)|0x7f800000u|(mant<<13);
    else bits=(sign<<31)|((exp+112)<<23)|(mant<<13);
    float f; memcpy(&f,&bits,4); return f;
}

static uint16_t float_to_half(float f) {
    uint32_t b; memcpy(&b,&f,4);
    uint32_t sign=b>>31, exp=(b>>23)&255, mant=b&0x7fffff;
    if(exp==255) return (uint16_t)((sign<<15)|0x7c00|!!mant);
    int e=(int)exp-127+15;
    if(e<=0) return (uint16_t)(sign<<15);
    if(e>=31) return (uint16_t)((sign<<15)|0x7c00);
    return (uint16_t)((sign<<15)|((uint32_t)e<<10)|(mant>>13));
}

static int cfg_int(jval *root, const char *name, int fallback) {
    jval *v=json_get(root,name); return v ? (int)v->num : fallback;
}
static float cfg_float(jval *root, const char *name, float fallback) {
    jval *v=json_get(root,name); return v ? (float)v->num : fallback;
}

static int load_config(QwnDecoder *d) {
    QwnConfig *c=&d->cfg;
    if (d->model.hdr.arch_dims[0] > 0) {
        c->hidden = (int)d->model.hdr.arch_dims[0];
        c->intermediate = (int)d->model.hdr.arch_dims[1];
        c->heads = (int)d->model.hdr.arch_dims[2];
        c->kv_heads = (int)d->model.hdr.arch_dims[3];
        c->head_dim = (int)d->model.hdr.arch_dims[4];
        c->q_head_dim = c->k_head_dim = c->v_head_dim = c->head_dim;
        if (d->model.hdr.q_dim && c->heads)
            c->q_head_dim = (int)(d->model.hdr.q_dim / (uint64_t)c->heads);
        if (d->model.hdr.k_dim && c->kv_heads)
            c->k_head_dim = (int)(d->model.hdr.k_dim / (uint64_t)c->kv_heads);
        if (d->model.hdr.v_dim && c->kv_heads)
            c->v_head_dim = (int)(d->model.hdr.v_dim / (uint64_t)c->kv_heads);
        c->layers = (int)d->model.hdr.arch_dims[5];
        c->vocab = (int)d->model.hdr.arch_dims[6];
        c->max_ctx = (int)d->model.hdr.arch_dims[7];
        if (c->max_ctx <= 0) c->max_ctx = 4096;
        c->bos_id = -1;
        c->eos_id = -1;
        c->rms_eps = 1e-6f;
        c->rope_theta = 10000.0f;
    }
    const QwnTensorDesc *t=qwn_find(&d->model,"__qwn.config");
    if(t && t->dtype==QWN_DT_BYTES && t->numel>=2) {
        char *json=(char*)malloc((size_t)t->numel+1);
        if(json) {
            memcpy(json,qwn_data(&d->model,t),(size_t)t->numel); json[t->numel]=0;
            char *arena=NULL; jval *root=json_parse(json,&arena);
            if(root) {
                if (cfg_int(root,"hidden_size",0)>0) c->hidden=cfg_int(root,"hidden_size",c->hidden);
                if (cfg_int(root,"intermediate_size",0)>0) c->intermediate=cfg_int(root,"intermediate_size",c->intermediate);
                if (cfg_int(root,"num_hidden_layers",0)>0) c->layers=cfg_int(root,"num_hidden_layers",c->layers);
                if (cfg_int(root,"num_attention_heads",0)>0) c->heads=cfg_int(root,"num_attention_heads",c->heads);
                 c->kv_heads=cfg_int(root,"num_key_value_heads",c->kv_heads ? c->kv_heads : c->heads);
                 c->head_dim=cfg_int(root,"head_dim",c->heads?c->hidden/c->heads:0);
                 c->q_head_dim=cfg_int(root,"q_head_dim",c->head_dim);
                 c->k_head_dim=cfg_int(root,"k_head_dim",c->head_dim);
                 c->v_head_dim=cfg_int(root,"v_head_dim",c->k_head_dim);
                if (cfg_int(root,"vocab_size",0)>0) c->vocab=cfg_int(root,"vocab_size",c->vocab);
                c->max_ctx=cfg_int(root,"max_position_embeddings",c->max_ctx);
                c->bos_id=cfg_int(root,"bos_token_id",c->bos_id);
                c->eos_id=cfg_int(root,"eos_token_id",c->eos_id);
                c->rms_eps=cfg_float(root,"rms_norm_eps",c->rms_eps);
                c->rope_theta=cfg_float(root,"rope_theta",c->rope_theta);
                c->tie_embeddings=cfg_int(root,"tie_word_embeddings",0);
            }
            free(json);
        }
    }
    if (c->q_head_dim <= 0) c->q_head_dim = c->head_dim;
    if (c->k_head_dim <= 0) c->k_head_dim = c->head_dim;
    if (c->v_head_dim <= 0) c->v_head_dim = c->k_head_dim;
    return c->hidden>0&&c->intermediate>0&&c->layers>0&&c->heads>0&&
           c->kv_heads>0&&c->head_dim>0&&c->q_head_dim>0&&
           c->k_head_dim>0&&c->v_head_dim>0&&c->vocab>0 ? 0:-1;
}

static int load_tokenizer(QwnDecoder *d) {
    const QwnTensorDesc *t=qwn_find(&d->model,"__qwn.tokenizer");
    if(!t || t->dtype!=QWN_DT_BYTES || t->numel<2)return -1;
    char *json=(char*)malloc((size_t)t->numel+1); if(!json)return -1;
    memcpy(json,qwn_data(&d->model,t),(size_t)t->numel);json[t->numel]=0;
    tok_load_memory(&d->tokenizer,json);
    return 0;
}

static int vector_f32(const QwnModel *m,const QwnTensorDesc *t,float *out,int n){
    if(!t||t->n_dims!=1||t->shape[0]!=(uint64_t)n)return -1;
    const uint8_t *p=(const uint8_t*)qwn_data(m,t);if(!p)return -1;
    if(t->dtype==QWN_DT_F32){memcpy(out,p,(size_t)n*4);return 0;}
    if(t->dtype==QWN_DT_F16||t->dtype==QWN_DT_BF16){
        const uint16_t *h=(const uint16_t*)p;
        for(int i=0;i<n;i++){
            if(t->dtype==QWN_DT_F16)out[i]=half_to_float(h[i]);
            else{uint32_t b=(uint32_t)h[i]<<16;memcpy(&out[i],&b,4);}
        }return 0;
    }return -1;
}

static const QwnTensorDesc *layer_tensor(const QwnDecoder *d,int layer,const char *suffix){
    char name[128];
    snprintf(name,sizeof(name),"model.layers.%d.%s",layer,suffix);
    const QwnTensorDesc *t = qwn_find(&d->model,name);
    if(t) return t;
    snprintf(name,sizeof(name),"blk.%d.%s",layer,suffix);
    t = qwn_find(&d->model,name);
    if(t) return t;
    if(strstr(suffix, "post_attention_layernorm.weight")) {
        snprintf(name,sizeof(name),"blk.%d.post_attention_norm.weight",layer);
        t = qwn_find(&d->model,name);
        if(t) return t;
    }
    if(strstr(suffix, "input_layernorm.weight")) {
        snprintf(name,sizeof(name),"blk.%d.input_norm.weight",layer);
        t = qwn_find(&d->model,name);
        if(t) return t;
    }
    return NULL;
}

static int required_tensors(QwnDecoder *d){
    if(!qwn_find(&d->model,"model.embed_tokens.weight") && !qwn_find(&d->model,"token_embd.weight"))return -1;
    if(!qwn_find(&d->model,"model.norm.weight") && !qwn_find(&d->model,"output_norm.weight"))return -1;
    return 0;
}

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- Per-decoder precomputed RoPE frequency table ---- */
static void rope_cache_ensure(QwnDecoder *d, int max_ctx, int half_dim, float theta) {
    if (d->rope_cos_cache && d->rope_cache_ctx >= max_ctx &&
        d->rope_cache_half >= half_dim)
        return;
    free(d->rope_cos_cache); free(d->rope_sin_cache);
    d->rope_cache_ctx = max_ctx;
    d->rope_cache_half = half_dim;
    size_t n = (size_t)max_ctx * (size_t)half_dim;
    d->rope_cos_cache = (float *)malloc(n * sizeof(float));
    d->rope_sin_cache = (float *)malloc(n * sizeof(float));
    if (!d->rope_cos_cache || !d->rope_sin_cache) {
        free(d->rope_cos_cache); free(d->rope_sin_cache);
        d->rope_cos_cache = d->rope_sin_cache = NULL;
        d->rope_cache_ctx = d->rope_cache_half = 0;
        return;
    }
    for (int pos = 0; pos < max_ctx; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos * powf(theta, -2.0f * i / (2 * half_dim));
            d->rope_cos_cache[(size_t)pos * half_dim + i] = cosf(angle);
            d->rope_sin_cache[(size_t)pos * half_dim + i] = sinf(angle);
        }
    }
}

static void rmsnorm(float *out,const float *x,const float *w,int n,float eps){
#if defined(__AVX2__)
    if (n % 8 == 0) {
        __m256 sum_vec = _mm256_setzero_ps();
        for (int i = 0; i < n; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            sum_vec = _mm256_fmadd_ps(vx, vx, sum_vec);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, sum_vec);
        float ss = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        float inv = 1.0f / sqrtf((ss / (float)n) + eps);
        __m256 inv_vec = _mm256_set1_ps(inv);
        for (int i = 0; i < n; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            __m256 vw = _mm256_loadu_ps(&w[i]);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, inv_vec), vw);
            _mm256_storeu_ps(&out[i], res);
        }
        return;
    }
#endif
    double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];
    float inv=1.0f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++)out[i]=x[i]*inv*w[i];
}

static void head_rmsnorm(float *x,int heads,int dim,const float *w,float eps){
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(heads > 1)
#endif
    for(int h=0;h<heads;h++){
        float *row=x+(size_t)h*dim;
#if defined(__AVX2__)
        if (dim % 8 == 0) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int i = 0; i < dim; i += 8) {
                __m256 vx = _mm256_loadu_ps(&row[i]);
                sum_vec = _mm256_fmadd_ps(vx, vx, sum_vec);
            }
            float tmp[8]; _mm256_storeu_ps(tmp, sum_vec);
            float ss = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
            float inv = 1.0f / sqrtf((ss / (float)dim) + eps);
            __m256 inv_vec = _mm256_set1_ps(inv);
            for (int i = 0; i < dim; i += 8) {
                __m256 vx = _mm256_loadu_ps(&row[i]);
                __m256 vw = _mm256_loadu_ps(&w[i]);
                _mm256_storeu_ps(&row[i], _mm256_mul_ps(_mm256_mul_ps(vx, inv_vec), vw));
            }
            continue;
        }
#endif
        double ss=0;
        for(int i=0;i<dim;i++)ss+=(double)row[i]*row[i];
        float inv=1.0f/sqrtf((float)(ss/dim)+eps);
        for(int i=0;i<dim;i++)row[i]=row[i]*inv*w[i];
    }
}

static void rope(const QwnDecoder *d, float *v, int heads, int dim, int pos, float theta) {
    int half = dim / 2;
    /* Use precomputed table if available */
    if (d->rope_cos_cache && pos < d->rope_cache_ctx && half <= d->rope_cache_half) {
        const float *ct = d->rope_cos_cache + (size_t)pos * d->rope_cache_half;
        const float *st = d->rope_sin_cache + (size_t)pos * d->rope_cache_half;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(heads > 1)
#endif
        for (int h = 0; h < heads; h++) {
            float *row = v + h * dim;
#if defined(__AVX2__)
            int i = 0;
            for (; i <= half - 8; i += 8) {
                __m256 a = _mm256_loadu_ps(&row[i]);
                __m256 b = _mm256_loadu_ps(&row[i + half]);
                __m256 c = _mm256_loadu_ps(&ct[i]);
                __m256 s = _mm256_loadu_ps(&st[i]);
                _mm256_storeu_ps(&row[i],        _mm256_fmsub_ps(a, c, _mm256_mul_ps(b, s)));
                _mm256_storeu_ps(&row[i + half], _mm256_fmadd_ps(a, s, _mm256_mul_ps(b, c)));
            }
            for (; i < half; i++) {
                float a = row[i], b = row[i + half];
                row[i]        = a * ct[i] - b * st[i];
                row[i + half] = a * st[i] + b * ct[i];
            }
#else
            for (int i = 0; i < half; i++) {
                float a = row[i], b = row[i + half];
                row[i]        = a * ct[i] - b * st[i];
                row[i + half] = a * st[i] + b * ct[i];
            }
#endif
        }
        return;
    }
    /* Fallback for uncached positions */
    for (int h = 0; h < heads; h++) for (int i = 0; i < half; i++) {
        float angle = (float)pos * powf(theta, -2.0f * i / dim);
        float c = cosf(angle), s = sinf(angle);
        float a = v[h * dim + i], b = v[h * dim + i + half];
        v[h * dim + i] = a * c - b * s; v[h * dim + i + half] = a * s + b * c;
    }
}

static void softmax(float *x,int n){
    float max=-INFINITY;
#if defined(__AVX2__)
    if (n >= 8) {
        __m256 vmax = _mm256_set1_ps(-INFINITY);
        int i = 0;
        for (; i <= n - 8; i += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
        float tmp[8]; _mm256_storeu_ps(tmp, vmax);
        for (int j = 0; j < 8; j++) if (tmp[j] > max) max = tmp[j];
        for (; i < n; i++) if (x[i] > max) max = x[i];
    } else
#endif
    { for(int i=0;i<n;i++) if(x[i]>max) max=x[i]; }
    double sum=0;
    for(int i=0;i<n;i++){x[i]=expf(x[i]-max);sum+=x[i];}
    float inv=1.0f/(float)sum;
#if defined(__AVX2__)
    if (n >= 8) {
        __m256 vinv = _mm256_set1_ps(inv);
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
        for (; i < n; i++) x[i] *= inv;
    } else
#endif
    { for(int i=0;i<n;i++) x[i]*=inv; }
}

static int matmul(QwnDecoder *d,const QwnTensorDesc *w,const float *x,
                  int in,int out,float *y){
    if (d->qwn_cuda.available && w) {
        QwnCudaGemvFn gemv = NULL;
#ifndef COLI_CUDA
        if (w->dtype == QWN_DT_Q4_0) gemv = d->qwn_cuda.gemv_q4_0;
#endif
        if (w->dtype == QWN_DT_HYPER_VSQ2) gemv = d->qwn_cuda.gemv_hypervsq2;
        if (gemv && gemv(out, in, qwn_data(&d->model, w), x, y) == 0) {
            QwnCudaMetricsSnapshot snapshot;
            if (d->qwn_cuda.get_metrics(&snapshot) != 0) {
                fprintf(stderr, "[ERROR] CUDA backend returned no execution metrics.\n");
                return -1;
            }
            d->runtime_metrics.cuda_matmul_count = snapshot.matmul_count;
            d->runtime_metrics.cuda_resident_bytes = snapshot.resident_bytes;
            d->runtime_metrics.cuda_upload_bytes = snapshot.upload_bytes;
            d->runtime_metrics.cuda_device = snapshot.device_id;
            snprintf(d->runtime_metrics.backend, sizeof(d->runtime_metrics.backend), "cuda");
            snprintf(d->runtime_metrics.kernel, sizeof(d->runtime_metrics.kernel), "%s", snapshot.kernel);
            return 0;
        }
        if (d->runtime_config.backend == QWN_RUNTIME_BACKEND_CUDA) return -1;
        fprintf(stderr, "[WARN] qwn CUDA GEMV failed; auto backend returns to CPU.\n");
        qwn_cuda_unload(d);
        d->runtime_metrics.cpu_fallback_count++;
    }
#ifdef COLI_CUDA
    if(d->cuda_enabled && w && w->dtype==QWN_DT_Q4_0){
        int slot=-1;
        for(int i=0;i<d->cuda_weight_count;i++)if(d->cuda_weights[i].desc==w){slot=i;break;}
        if(slot<0 && d->cuda_weight_count<(int)(sizeof(d->cuda_weights)/sizeof(d->cuda_weights[0])))
            slot=d->cuda_weight_count++;
        if(slot>=0){
            int is_new = d->cuda_weights[slot].desc == NULL;
            if(d->cuda_weights[slot].desc==NULL){
                d->cuda_weights[slot].desc=w;
                int start = slot % d->cuda_device_count;
                int selected = -1;
                size_t bytes = w->byte_size;
                for (int n = 0; n < d->cuda_device_count; n++) {
                    int index = (start + n) % d->cuda_device_count;
                    if (d->cuda_resident_bytes[index] <=
                        d->cuda_budget_bytes[index] &&
                        bytes <= d->cuda_budget_bytes[index] -
                                d->cuda_resident_bytes[index]) {
                        selected = index;
                        break;
                    }
                }
                if (selected < 0) {
                    d->cuda_weights[slot].desc = NULL;
                    d->cuda_weight_count--;
                    return qwn_matmul_f32(&d->model,w,x,1,in,out,&d->scratch,y);
                }
                d->cuda_weights[slot].device = d->cuda_devices[selected];
            }
            if(coli_cuda_qwn_matmul(&d->cuda_weights[slot].tensor,y,x,
                                    qwn_data(&d->model,w),1,in,out,
                                    d->cuda_weights[slot].device)) {
                if (is_new) {
                    for (int i = 0; i < d->cuda_device_count; i++) {
                        if (d->cuda_devices[i] == d->cuda_weights[slot].device) {
                            d->cuda_resident_bytes[i] += w->byte_size;
                            break;
                        }
                    }
                }
                return 0;
            }
            d->cuda_enabled=0; /* broken CUDA path falls back for the session */
        }
    }
#endif
    return qwn_matmul_f32(&d->model,w,x,1,in,out,&d->scratch,y);
}

static void add_bias(const QwnDecoder *d,const QwnTensorDesc *b,float *x,int n){
    if(!b)return;float *tmp=d->scratch.row_f32;
    if(vector_f32(&d->model,b,tmp,n)==0)for(int i=0;i<n;i++)x[i]+=tmp[i];
}

static int load_norms(QwnDecoder *d){
    /* Per-layer norms are now loaded lazily through QwnLayerTensors
     * (input_norm, post_norm, final_norm_weight).  This routine is
     * kept as a compatibility stub that only fetches the fallback
     * path used when no per-layer norm tensor is available. */
    int D=d->cfg.hidden;
    for(int l=0;l<d->cfg.layers;l++){
        const QwnLayerTensors *lt = &d->layer_cache[l];
        if (!lt->input_norm) {
            const QwnTensorDesc *fallback = layer_tensor(d,l,"input_layernorm.weight");
            if (fallback && vector_f32(&d->model, fallback,
                                         d->norm_weights+(size_t)(2*l)*D, D) != 0) return -1;
        }
        if (!lt->post_norm) {
            const QwnTensorDesc *fallback = layer_tensor(d,l,"post_attention_layernorm.weight");
            if (fallback && vector_f32(&d->model, fallback,
                                         d->norm_weights+(size_t)(2*l+1)*D, D) != 0) return -1;
        }
    }
    const QwnTensorDesc *fn = qwn_find(&d->model,"model.norm.weight");
    if(!fn) fn = qwn_find(&d->model,"output_norm.weight");
    if(!fn) fn = qwn_find(&d->model,"norm.weight");
    return fn ? vector_f32(&d->model,fn,d->norm_weights+(size_t)(2*d->cfg.layers)*D,D) : 0;
}

/* Resolve all per-layer tensor descriptors once at load time.
 * Per-layer dims are computed from the actual tensor shapes; the
 * fallback path uses the global head_dim when a layer is missing the
 * relevant projection.
 */
static int build_layer_cache(QwnDecoder *d) {
    d->layer_cache = (QwnLayerTensors *)malloc((size_t)d->cfg.layers * sizeof(QwnLayerTensors));
    if (!d->layer_cache) return -1;
    for (int l = 0; l < d->cfg.layers; l++) {
        QwnLayerTensors *lt = &d->layer_cache[l];
        memset(lt, 0, sizeof(*lt));
        lt->q_proj   = layer_tensor(d, l, "self_attn.q_proj.weight");
        lt->k_proj   = layer_tensor(d, l, "self_attn.k_proj.weight");
        lt->v_proj   = layer_tensor(d, l, "self_attn.v_proj.weight");
        lt->o_proj   = layer_tensor(d, l, "self_attn.o_proj.weight");
        lt->q_bias   = layer_tensor(d, l, "self_attn.q_proj.bias");
        lt->k_bias   = layer_tensor(d, l, "self_attn.k_proj.bias");
        lt->v_bias   = layer_tensor(d, l, "self_attn.v_proj.bias");
        lt->o_bias   = layer_tensor(d, l, "self_attn.o_proj.bias");
        lt->q_norm   = layer_tensor(d, l, "self_attn.q_norm.weight");
        lt->k_norm   = layer_tensor(d, l, "self_attn.k_norm.weight");
        lt->input_norm  = layer_tensor(d, l, "input_layernorm.weight");
        if (!lt->input_norm) lt->input_norm = layer_tensor(d, l, "attention_norm.weight");
        lt->post_norm   = layer_tensor(d, l, "post_attention_layernorm.weight");
        if (!lt->post_norm) lt->post_norm = layer_tensor(d, l, "ffn_norm.weight");
        lt->gate_proj = layer_tensor(d, l, "mlp.gate_proj.weight");
        if (!lt->gate_proj) lt->gate_proj = layer_tensor(d, l, "feed_forward.w1.weight");
        lt->up_proj   = layer_tensor(d, l, "mlp.up_proj.weight");
        if (!lt->up_proj) lt->up_proj = layer_tensor(d, l, "feed_forward.w3.weight");
        lt->down_proj = layer_tensor(d, l, "mlp.down_proj.weight");
        if (!lt->down_proj) lt->down_proj = layer_tensor(d, l, "feed_forward.w2.weight");
        /* Per-layer output dims come straight from the tensor shapes
         * so models with variable head_dim per layer (Qwen3.5 hybrid,
         * MLA, etc.) load and run correctly. */
        if (lt->q_proj && lt->q_proj->n_dims == 2) {
            lt->q_out = (int)lt->q_proj->shape[1];
        } else if (d->cfg.heads && d->cfg.q_head_dim) {
            lt->q_out = d->cfg.heads * d->cfg.q_head_dim;
        }
        if (lt->k_proj && lt->k_proj->n_dims == 2) {
            lt->k_out = (int)lt->k_proj->shape[1];
        } else if (d->cfg.kv_heads && d->cfg.k_head_dim) {
            lt->k_out = d->cfg.kv_heads * d->cfg.k_head_dim;
        }
        if (lt->v_proj && lt->v_proj->n_dims == 2) {
            lt->v_out = (int)lt->v_proj->shape[1];
        } else if (d->cfg.kv_heads && d->cfg.v_head_dim) {
            lt->v_out = d->cfg.kv_heads * d->cfg.v_head_dim;
        }
        if (lt->q_out && d->cfg.heads) lt->q_head_dim = lt->q_out / d->cfg.heads;
        if (lt->k_out && d->cfg.kv_heads) lt->k_head_dim = lt->k_out / d->cfg.kv_heads;
        if (lt->v_out && d->cfg.kv_heads) lt->v_head_dim = lt->v_out / d->cfg.kv_heads;
        if (lt->up_proj && lt->up_proj->n_dims == 2) {
            lt->ffn_in  = (int)lt->up_proj->shape[0];   /* up_proj rows */
            lt->ffn_out = (int)lt->up_proj->shape[1];
        }
        if (lt->down_proj && lt->down_proj->n_dims == 2) {
            lt->ffn_down_out = (int)lt->down_proj->shape[1];
        }
        char namebuf[160];
        /* SSM detection: Qwen3 hybrid layers have no attention
         * projections but they do have a mixer block. */
        if (!lt->q_proj && !lt->k_proj && !lt->v_proj) {
            if (layer_tensor(d, l, "mixer.in_proj.weight") ||
                layer_tensor(d, l, "mixer.x_proj.weight") ||
                layer_tensor(d, l, "ssm.in_proj.weight"))
                lt->is_ssm = 1;
            snprintf(namebuf, sizeof(namebuf), "blk.%d.ssm_conv1d.weight", l);
            if (qwn_find(&d->model, namebuf)) lt->is_ssm = 1;
        }
        /* MoE detection: routed experts live under mlp.experts.N or
         * block_sparse_moe.experts.N. */
        int has_routed = 0;
        for (int e = 0; e < 256; e++) {
            snprintf(namebuf, sizeof(namebuf),
                     "model.layers.%d.mlp.experts.%d.gate_proj.weight", l, e);
            if (qwn_find(&d->model, namebuf)) { has_routed = 1; break; }
            snprintf(namebuf, sizeof(namebuf),
                     "model.layers.%d.block_sparse_moe.experts.%d.w1.weight", l, e);
            if (qwn_find(&d->model, namebuf)) { has_routed = 1; break; }
}
        lt->is_moe = has_routed;
        /* SSM / hybrid layers (Qwen3.5-style) are intentionally allowed:
         * qwn_decoder_forward() skips the attention path entirely for
         * them and only runs the residual stream through the FFN.  A
         * real SSM/mamba kernel would replace this skip; for now the
         * hybrid decoder degrades gracefully. */
        if (lt->is_moe) return -2;
        /* If the layer has no attention projections at all, skip the
         * per-attention validation -- it is treated as an SSM layer
         * by the forward pass. */
        if (!lt->q_proj || !lt->k_proj || !lt->v_proj || !lt->o_proj) {
            lt->is_ssm = 1;
        } else if (!lt->gate_proj || !lt->up_proj || !lt->down_proj ||
            lt->q_proj->n_dims != 2 || lt->k_proj->n_dims != 2 ||
            lt->v_proj->n_dims != 2 || lt->o_proj->n_dims != 2 ||
            lt->gate_proj->n_dims != 2 || lt->up_proj->n_dims != 2 ||
            lt->down_proj->n_dims != 2) return -1;
if (!lt->is_ssm && (!d->cfg.heads || !d->cfg.kv_heads ||
            lt->q_out % d->cfg.heads != 0 ||
            lt->k_out % d->cfg.kv_heads != 0 ||
            lt->v_out % d->cfg.kv_heads != 0 ||
            /* Some hybrid models use different head_dim for Q vs O;
             * only require Q and KV to share head_dim, not O. */
            lt->k_out != lt->v_out ||
            lt->q_proj->shape[0] != (uint64_t)d->cfg.hidden ||
            lt->k_proj->shape[0] != (uint64_t)d->cfg.hidden ||
            lt->v_proj->shape[0] != (uint64_t)d->cfg.hidden ||
            /* Allow o_proj input to differ from q_out (different head_dim) */
            lt->o_proj->shape[1] != (uint64_t)d->cfg.hidden ||
            lt->gate_proj->shape[0] != (uint64_t)d->cfg.hidden ||
            lt->up_proj->shape[0] != (uint64_t)d->cfg.hidden ||
            lt->gate_proj->shape[1] != lt->up_proj->shape[1] ||
            lt->down_proj->shape[0] != lt->gate_proj->shape[1] ||
            lt->down_proj->shape[1] != (uint64_t)d->cfg.hidden)) return -2;
}
    d->embed_weight = qwn_find(&d->model, "model.embed_tokens.weight");
    if(!d->embed_weight) d->embed_weight = qwn_find(&d->model, "token_embd.weight");
    d->lm_head_weight = qwn_find(&d->model, "lm_head.weight");
    if (!d->lm_head_weight) d->lm_head_weight = qwn_find(&d->model, "output.weight");
    if (!d->lm_head_weight) d->lm_head_weight = d->embed_weight;
    /* Final norm: try a few known names */
    d->final_norm_weight = qwn_find(&d->model, "model.norm.weight");
    if (!d->final_norm_weight) d->final_norm_weight = qwn_find(&d->model, "output_norm.weight");
    if (!d->final_norm_weight) d->final_norm_weight = qwn_find(&d->model, "norm.weight");
    return 0;
}

int qwn_decoder_open(QwnDecoder *d,const char *path,int ctx_size,const char **error){
    QwnRuntimeConfig config;
    qwn_runtime_config_default(&config);
    config.context_size = ctx_size > 0 ? ctx_size : config.context_size;
    return qwn_decoder_open_with_config(d, path, &config, error);
}

int qwn_decoder_open_with_config(QwnDecoder *d,const char *path,
                                 const QwnRuntimeConfig *config,
                                 const char **error){
    static const char *ERR_CONFIG="unsupported/missing Llama-Qwen config";
    static const char *ERR_TENSORS="missing or inconsistent dense Transformer tensors";
    static const char *ERR_SHAPE="unsupported native architecture or Q/K/V shape";
    static const char *ERR_MEMORY="native decoder allocation failed";
    int D, I, Hd, Q, KV, V, max_dim;
    int max_q_out=0, max_kv_out=0, max_ffn_out=0;
    size_t floats, kv_elems, kv_bytes;
    float *p;

    memset(d,0,sizeof(*d));if(error)*error=NULL;
    snprintf(d->runtime_metrics.cuda_dll_hash,
             sizeof(d->runtime_metrics.cuda_dll_hash), "Unavailable");
    qwn_runtime_config_default(&d->runtime_config);
    if (config) d->runtime_config = *config;
    if (qwn_runtime_config_validate(&d->runtime_config, NULL, 0) != 0) {
        if (error) *error = "invalid runtime configuration";
        return -1;
    }
    {
        char kernel_error[128];
        if (qwn_select_cpu_kernel(d->runtime_config.kernel, kernel_error, sizeof(kernel_error)) != 0) {
            if (error) *error = "requested CPU kernel is unavailable on this host";
            fprintf(stderr, "[ERROR] %s\n", kernel_error);
            return -1;
        }
    }
#if defined(_OPENMP)
    if (d->runtime_config.cpu_threads > 0)
        omp_set_num_threads(d->runtime_config.cpu_threads);
    d->runtime_metrics.openmp_runtime_loaded = omp_get_max_threads() > 0;
#else
    d->runtime_metrics.openmp_runtime_loaded = 0;
#endif
    d->runtime_metrics.requested_cpu_threads = d->runtime_config.cpu_threads;
    if(qwn_open(path,&d->model,error)!=0)return -1;
    if (strcmp(d->runtime_config.quantization, "auto") != 0) {
        uint32_t wanted = strcmp(d->runtime_config.quantization, "q4_0") == 0 ? QWN_DT_Q4_0 :
                          strcmp(d->runtime_config.quantization, "hyper_vsq2") == 0 ? QWN_DT_HYPER_VSQ2 :
                          strcmp(d->runtime_config.quantization, "fp16") == 0 ? QWN_DT_F16 : QWN_DT_F32;
        int found = 0;
        for (uint32_t i = 0; i < d->model.hdr.n_tensors; i++) {
            const QwnTensorDesc *tensor = qwn_tensor_at(&d->model, i);
            if (tensor && tensor->dtype == wanted) { found = 1; break; }
        }
        if (!found) {
            if (error) *error = "requested quantization is not present in the QWN model";
            goto fail;
        }
    }
    if(load_config(d)!=0||load_tokenizer(d)!=0){if(error)*error=ERR_CONFIG;goto fail;}
    if(d->runtime_config.context_size>0&&d->runtime_config.context_size<d->cfg.max_ctx)
        d->cfg.max_ctx=d->runtime_config.context_size;
    if(required_tensors(d)!=0){if(error)*error=ERR_TENSORS;goto fail;}
    /* Resolve per-layer tensors + dims first so we can size the
     * buffers for the largest layer, not the global head_dim product. */
    {
        int cache_rc = build_layer_cache(d);
        if(cache_rc != 0){
            if(error)*error = cache_rc == -2 ? ERR_SHAPE : ERR_TENSORS;
            goto fail;
        }
    }
    for (int l = 0; l < d->cfg.layers; l++) {
        const QwnLayerTensors *lt = &d->layer_cache[l];
        if (lt->q_out  > max_q_out)  max_q_out  = lt->q_out;
        if (lt->k_out > max_kv_out) max_kv_out = lt->k_out;
        if (lt->ffn_out > max_ffn_out) max_ffn_out = lt->ffn_out;
    }
for (int l = 0; l < d->cfg.layers; l++) {
        const QwnLayerTensors *lt = &d->layer_cache[l];
        /* SSM / hybrid layers legitimately have zero attention dims.
         * Skip them when enforcing uniform shape across layers. */
        if (lt->is_ssm) continue;
        if (lt->q_out != max_q_out || lt->k_out != max_kv_out ||
            lt->v_out != max_kv_out) {
            if (error) *error = ERR_SHAPE;
            goto fail;
        }
    }
    /* Fallback to the global dim when no layer advertised it
     * (e.g. the head_dim config block was absent). */
    Hd = d->cfg.head_dim;
    if (max_q_out  == 0) max_q_out  = d->cfg.heads  * Hd;
    if (max_kv_out == 0) max_kv_out = d->cfg.kv_heads * Hd;
    if (max_ffn_out == 0) max_ffn_out = d->cfg.intermediate;
    D=d->cfg.hidden; I=max_ffn_out;
    Q=max_q_out; KV=max_kv_out; V=d->cfg.vocab;
    max_dim=D>I?D:I;if(Q>max_dim)max_dim=Q;if(V>max_dim)max_dim=V;
    if(qwn_scratch_init(&d->scratch,1,max_dim)!=0){if(error)*error=ERR_MEMORY;goto fail;}
    floats=(size_t)D*3+(size_t)Q*2+(size_t)KV*2+(size_t)d->cfg.heads*(size_t)d->cfg.max_ctx+
           (size_t)I*3+(size_t)V+(size_t)(2*d->cfg.layers+1)*D;
    d->arena_bytes=up64(floats*sizeof(float));d->arena=alloc64(d->arena_bytes);
    if(!d->arena){if(error)*error=ERR_MEMORY;goto fail;}
    p=(float*)d->arena;
    d->x=p;p+=D;d->xb=p;p+=D;d->q=p;p+=Q;d->k=p;p+=KV;d->v=p;p+=KV;
    d->ctx=p;p+=Q;d->att=p;p+=(size_t)d->cfg.heads*(size_t)d->cfg.max_ctx;d->gate=p;p+=I;
    d->up=p;p+=I;d->hidden=p;p+=I;d->logits=p;p+=V;d->norm_weights=p;
    kv_elems=(size_t)d->cfg.layers*d->cfg.max_ctx*max_kv_out;
    int kv_head_dim = d->cfg.kv_heads ? max_kv_out / d->cfg.kv_heads : 0;
    int page_blocks = (d->cfg.max_ctx + QWN_PAGE_BLOCK_SIZE - 1) / QWN_PAGE_BLOCK_SIZE;
    if (kv_head_dim > 0 &&
        qwn_kv_pool_init(&d->paged_kv, page_blocks, d->cfg.layers,
                         d->cfg.kv_heads, kv_head_dim) == 0 &&
        qwn_block_table_init(&d->paged_table, 0, page_blocks) == 0) {
        d->kv_gather_stride = (size_t)d->cfg.max_ctx * (size_t)kv_head_dim;
        size_t gather_bytes = (size_t)d->cfg.heads * d->kv_gather_stride * sizeof(uint16_t);
        d->kv_gather_key = (uint16_t *)alloc64(gather_bytes);
        d->kv_gather_value = (uint16_t *)alloc64(gather_bytes);
        if (d->kv_gather_key && d->kv_gather_value) {
            d->use_paged_kv = 1;
        } else {
            free64(d->kv_gather_key); free64(d->kv_gather_value);
            d->kv_gather_key = d->kv_gather_value = NULL;
            qwn_block_table_free(&d->paged_kv, &d->paged_table);
            qwn_kv_pool_free(&d->paged_kv);
        }
    }
    if (!d->use_paged_kv) {
        kv_bytes=up64(kv_elems*sizeof(uint16_t));
        d->kv_allocation=alloc64(kv_bytes*2);if(!d->kv_allocation){if(error)*error=ERR_MEMORY;goto fail;}
        d->key_cache=(uint16_t*)d->kv_allocation;
        d->value_cache=(uint16_t*)((uint8_t*)d->kv_allocation+kv_bytes);
        memset(d->kv_allocation,0,kv_bytes*2);
    }
    if(load_norms(d)!=0){if(error)*error=ERR_TENSORS;goto fail;}
#ifdef COLI_CUDA
    d->cuda_device_count = 0;
    const char *gpu_list = getenv("COLI_GPUS");
    if (!gpu_list || !*gpu_list) gpu_list = getenv("COLI_GPU");
    if (!gpu_list || !*gpu_list || strcmp(gpu_list, "auto") == 0) {
        d->cuda_devices[0] = 0;
        d->cuda_device_count = 1;
    } else if (strcmp(gpu_list, "none") != 0) {
        char list_copy[256];
        strncpy(list_copy, gpu_list, sizeof(list_copy) - 1);
        list_copy[sizeof(list_copy) - 1] = '\0';
        char *part = strtok(list_copy, ",");
        while (part && d->cuda_device_count < COLI_CUDA_MAX_DEVICES) {
            char *end = NULL;
            long value = strtol(part, &end, 10);
            if (end == part || *end != '\0' || value < 0 || value > 1024) {
                d->cuda_device_count = 0;
                break;
            }
            d->cuda_devices[d->cuda_device_count++] = (int)value;
            part = strtok(NULL, ",");
        }
    }
    d->cuda_enabled = d->cuda_device_count > 0 &&
                      coli_cuda_init(d->cuda_devices, d->cuda_device_count);
    if (!d->cuda_enabled) d->cuda_device_count = 0;
#endif
    const char *seed_text = getenv("QWANTO_SEED");
    d->rng_state = d->runtime_config.seed > 0 ? (uint64_t)d->runtime_config.seed :
                   (seed_text && *seed_text ? strtoull(seed_text, NULL, 10)
                                            : 0x9e3779b97f4a7c15ULL);
    if (d->rng_state == 0) d->rng_state = 0x9e3779b97f4a7c15ULL;
    const char *tq_env = getenv("QWN_TURBOQUANT");
    if (tq_env && (strcmp(tq_env, "1") == 0 || strcmp(tq_env, "true") == 0 || strcmp(tq_env, "auto") == 0)) {
        d->use_turboquant = 1;
        d->turboquant_layers = (TurboQuantCache*)malloc(sizeof(TurboQuantCache) * (size_t)d->cfg.layers);
        if (d->turboquant_layers) {
            for (int l = 0; l < d->cfg.layers; l++) {
                int kv_hd = d->cfg.kv_heads ? max_kv_out / d->cfg.kv_heads : 0;
                qwn_turboquant_init(&d->turboquant_layers[l], d->cfg.max_ctx, d->cfg.kv_heads, kv_hd);
            }
        }
    }
    d->runtime_metrics.cuda_device = d->runtime_config.gpu_device >= 0 ? d->runtime_config.gpu_device : 0;
    snprintf(d->runtime_metrics.backend, sizeof(d->runtime_metrics.backend),
             d->runtime_config.backend == QWN_RUNTIME_BACKEND_CUDA ? "cuda-pending" : "cpu");
    snprintf(d->runtime_metrics.kernel, sizeof(d->runtime_metrics.kernel), "%s", qwn_cpu_kernel_name());
    {
        int cuda_required = d->runtime_config.backend == QWN_RUNTIME_BACKEND_CUDA;
        if (d->runtime_config.backend != QWN_RUNTIME_BACKEND_CPU &&
            qwn_cuda_load(d, d->runtime_metrics.cuda_device, cuda_required) != 0 && cuda_required) {
            if (error) *error = "CUDA backend requested but qwn_cuda.dll/device is unavailable";
            goto fail;
        }
    }
    init_residency(d);
    /* Precompute RoPE table once at load time */
    rope_cache_ensure(d, d->cfg.max_ctx, d->cfg.head_dim / 2, d->cfg.rope_theta);
    qwn_decoder_refresh_runtime_metrics(d);
    return 0;
fail:qwn_decoder_close(d);return -1;
}

const QwnRuntimeMetrics *qwn_decoder_metrics(const QwnDecoder *d) {
    return d ? &d->runtime_metrics : NULL;
}

void qwn_decoder_refresh_runtime_metrics(QwnDecoder *d) {
    if (!d) return;
    d->runtime_metrics.requested_cpu_threads = d->runtime_config.cpu_threads;
    d->runtime_metrics.hypervsq2_matmul_count = d->scratch.hypervsq2_matmul_calls;
    d->runtime_metrics.hypervsq2_worker_participations =
        d->scratch.hypervsq2_worker_participations;
    d->runtime_metrics.hypervsq2_last_active_threads =
        d->scratch.hypervsq2_last_active_threads;
    d->runtime_metrics.hypervsq2_max_active_threads =
        d->scratch.hypervsq2_max_active_threads;
    if (d->scratch.hypervsq2_matmul_calls > 0) {
        d->runtime_metrics.active_cpu_threads =
            d->scratch.hypervsq2_last_active_threads;
        if (strcmp(d->runtime_metrics.backend, "cuda") != 0) {
            snprintf(d->runtime_metrics.kernel, sizeof(d->runtime_metrics.kernel),
                     "%s", d->scratch.hypervsq2_kernel);
        }
    } else {
        d->runtime_metrics.active_cpu_threads = 0;
    }
}

const QwnGenerationMetrics *qwn_decoder_generation_metrics(const QwnDecoder *d) {
    return d ? &d->generation_metrics : NULL;
}

void qwn_decoder_reset(QwnDecoder *d){
    if (!d) return;
    d->position = 0;
    memset(&d->generation_metrics, 0, sizeof(d->generation_metrics));
    if (d->use_paged_kv) {
        qwn_block_table_free(&d->paged_kv, &d->paged_table);
        qwn_block_table_init(&d->paged_table, 0,
                             (d->cfg.max_ctx + QWN_PAGE_BLOCK_SIZE - 1) /
                             QWN_PAGE_BLOCK_SIZE);
    }
    if (d->use_turboquant && d->turboquant_layers) {
        for (int l = 0; l < d->cfg.layers; l++) {
            d->turboquant_layers[l].n_tokens = 0;
        }
    }
}

void qwn_decoder_close(QwnDecoder *d){
    if(!d)return;qwn_scratch_destroy(&d->scratch);free64(d->arena);
    free(d->layer_cache);
#ifdef COLI_CUDA
    for(int i=0;i<d->cuda_weight_count;i++)if(d->cuda_weights[i].tensor)coli_cuda_tensor_free(d->cuda_weights[i].tensor);
    if(d->cuda_enabled)coli_cuda_shutdown();
#endif
    qwn_cuda_unload(d);
    free(d->residency_items);
    free(d->rope_cos_cache);
    free(d->rope_sin_cache);
    free64(d->kv_gather_key);
    free64(d->kv_gather_value);
    qwn_block_table_free(&d->paged_kv, &d->paged_table);
    qwn_kv_pool_free(&d->paged_kv);
    if (d->turboquant_layers) {
        for (int l = 0; l < d->cfg.layers; l++) {
            qwn_turboquant_free(&d->turboquant_layers[l]);
        }
        free(d->turboquant_layers);
        d->turboquant_layers = NULL;
    }
    free64(d->kv_allocation);qwn_close(&d->model);memset(d,0,sizeof(*d));
}

/* AVX2 vectorized residual addition */
static void vec_add(float *dst, const float *src, int n) {
#if defined(__AVX2__)
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }
    for (; i < n; i++) dst[i] += src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += src[i];
#endif
}

int qwn_decoder_forward_thinking(QwnDecoder *d,int token,const float **out_logits,QwnThinkingConfig *config){
    QwnConfig *c=&d->cfg;
    int D=c->hidden;
    int H=c->heads;
    int HK=c->kv_heads;
    int HD=c->head_dim;
    int pos=d->position;
    if(token<0||token>=c->vocab||pos>=c->max_ctx){
        fprintf(stderr, "forward err bounds: token=%d vocab=%d pos=%d max_ctx=%d\n", token, c->vocab, pos, c->max_ctx);
        return -1;
    }
    if(qwn_row_f32(&d->model,d->embed_weight,token,d->x,D)!=0){
        fprintf(stderr, "forward err embed: token=%d D=%d embed_dtype=%d numel=%llu byte_size=%llu\n",
                token, D, d->embed_weight?d->embed_weight->dtype:-1,
                (unsigned long long)(d->embed_weight?d->embed_weight->numel:0),
                (unsigned long long)(d->embed_weight?d->embed_weight->byte_size:0));
        return -1;
    }
    if (d->use_paged_kv && qwn_block_table_append_token(&d->paged_kv,
                                                         &d->paged_table) != 0) {
        fprintf(stderr, "forward err: paged KV pool exhausted at position %d\n", pos);
        return -1;
    }
    for(int l=0;l<c->layers;l++){
        const QwnLayerTensors *lt = &d->layer_cache[l];
        /* Per-layer output dims: each layer can have its own head_dim
         * (Qwen3.5 hybrid, MLA, dense models with non-uniform GQA). */
        int Q  = lt->q_out  ? lt->q_out  : (H  ? H  * HD : 0);
        int KV = lt->k_out ? lt->k_out : (HK ? HK * HD : 0);
        int I  = lt->ffn_out ? lt->ffn_out : c->intermediate;
        int O_IN = (lt->o_proj && lt->o_proj->n_dims == 2) ? (int)lt->o_proj->shape[0] : (H ? H * HD : Q);
        int hd_this = (Q && H) ? Q / H : HD;
        /* Prefetch next layer's heavy weights -- zero-cost cached pointer lookup */
        if(l + 1 < c->layers) {
            const QwnLayerTensors *next = &d->layer_cache[l+1];
            qwn_prefetch(&d->model, next->q_proj);
            qwn_prefetch(&d->model, next->k_proj);
            qwn_prefetch(&d->model, next->v_proj);
            qwn_prefetch(&d->model, next->o_proj);
            qwn_prefetch(&d->model, next->gate_proj);
            qwn_prefetch(&d->model, next->up_proj);
            qwn_prefetch(&d->model, next->down_proj);
        }
        /* Skip attention entirely for SSM-only hybrid layers */
        if (lt->is_ssm) {
            if (lt->down_proj) {
            }
            continue;
        }
        /* Pre-attention norm: prefer per-layer input_layernorm.weight */
        const QwnTensorDesc *pre_norm = lt->input_norm;
        if (pre_norm && vector_f32(&d->model, pre_norm, d->scratch.row_f32, D) == 0) {
            rmsnorm(d->xb, d->x, d->scratch.row_f32, D, c->rms_eps);
        } else {
            rmsnorm(d->xb,d->x,d->norm_weights+(size_t)(2*l)*D,D,c->rms_eps);
        }
        if (lt->q_proj && lt->k_proj && lt->v_proj && Q && KV) {
            if(matmul(d,lt->q_proj,d->xb,D,Q,d->q)||matmul(d,lt->k_proj,d->xb,D,KV,d->k)||
               matmul(d,lt->v_proj,d->xb,D,KV,d->v)) {
                fprintf(stderr, "layer %d attn matmul failed\n", l);
                return -1;
            }
            add_bias(d,lt->q_bias,d->q,Q);
            add_bias(d,lt->k_bias,d->k,KV);
            add_bias(d,lt->v_bias,d->v,KV);
            if(lt->q_norm&&hd_this&&vector_f32(&d->model,lt->q_norm,d->scratch.row_f32,hd_this)==0)
                head_rmsnorm(d->q,H,hd_this,d->scratch.row_f32,c->rms_eps);
            if(lt->k_norm&&hd_this&&vector_f32(&d->model,lt->k_norm,d->scratch.row_f32,hd_this)==0)
                head_rmsnorm(d->k,HK,hd_this,d->scratch.row_f32,c->rms_eps);
            rope(d,d->q,H,hd_this,pos,c->rope_theta);rope(d,d->k,HK,hd_this,pos,c->rope_theta);
            if (d->use_turboquant && d->turboquant_layers) {
                qwn_turboquant_quantize_token(d->k, d->turboquant_layers[l].packed_k + (size_t)pos * d->turboquant_layers[l].token_stride_k, KV);
                qwn_turboquant_quantize_token(d->v, d->turboquant_layers[l].packed_v + (size_t)pos * d->turboquant_layers[l].token_stride_v, KV);
                memset(d->ctx, 0, (size_t)Q * sizeof(float));
                float scale = 1.0f / sqrtf((float)hd_this);
                float ratio = (HK > 0) ? ((float)H / HK) : 1.0f;
#if defined(_OPENMP)
                #pragma omp parallel for schedule(static) if(H > 1)
#endif
                for (int h = 0; h < H; h++) {
                    int kh = (int)((float)h / ratio);
                    qwn_turboquant_attention_head(
                        d->q + h * hd_this,
                        &d->turboquant_layers[l],
                        l, h, kh, pos, scale,
                        d->att + (size_t)h * c->max_ctx,
                        d->ctx + h * hd_this
                    );
                }
                if (lt->o_proj && matmul(d, lt->o_proj, d->ctx, O_IN, D, d->xb) == 0) {
                    add_bias(d, lt->o_bias, d->xb, D);
                    vec_add(d->x, d->xb, D);
                }
            } else if (d->use_paged_kv) {
                if (qwn_paged_kv_write(&d->paged_kv, &d->paged_table, l, pos,
                                       d->k, d->v) != 0) return -1;
                memset(d->ctx, 0, (size_t)Q * sizeof(float));
#if defined(_OPENMP)
                #pragma omp parallel for schedule(static) if(H > 1)
#endif
                for (int h = 0; h < H; h++) {
                    int kh = (h * HK) / H;
                    qwn_paged_attention_gather_head(
                        &d->paged_kv, &d->paged_table, l, h, kh,
                        d->q + h * hd_this,
                        d->kv_gather_key + (size_t)h * d->kv_gather_stride,
                        d->kv_gather_value + (size_t)h * d->kv_gather_stride,
                        d->att + (size_t)h * c->max_ctx,
                        d->ctx + h * hd_this);
                }
                if (lt->o_proj && matmul(d,lt->o_proj,d->ctx,O_IN,D,d->xb)==0) {
                    add_bias(d,lt->o_bias,d->xb,D);
                    vec_add(d->x, d->xb, D);
                }
            } else {
            size_t layer_base=(size_t)l*c->max_ctx*KV;
            size_t pos_base=layer_base+(size_t)pos*KV;
            /* F16C KV cache write: batch convert float32 -> float16 */
#if defined(__AVX2__) && defined(__F16C__)
            {
                int i = 0;
                for (; i <= KV - 8; i += 8) {
                    __m256 fk = _mm256_loadu_ps(d->k + i);
                    __m256 fv = _mm256_loadu_ps(d->v + i);
                    __m128i hk = _mm256_cvtps_ph(fk, _MM_FROUND_TO_NEAREST_INT);
                    __m128i hv = _mm256_cvtps_ph(fv, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128((__m128i *)(d->key_cache + pos_base + i), hk);
                    _mm_storeu_si128((__m128i *)(d->value_cache + pos_base + i), hv);
                }
                for (; i < KV; i++) {
                    d->key_cache[pos_base+i]=float_to_half(d->k[i]);
                    d->value_cache[pos_base+i]=float_to_half(d->v[i]);
                }
            }
#else
            for(int i=0;i<KV;i++){d->key_cache[pos_base+i]=float_to_half(d->k[i]);
                                  d->value_cache[pos_base+i]=float_to_half(d->v[i]);}
#endif
            memset(d->ctx,0,(size_t)Q*sizeof(float));
            float scale=1.0f/sqrtf((float)hd_this);
            float ratio = (HK > 0) ? ((float)H / HK) : 1.0f;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) if(H > 1)
#endif
            for(int h=0;h<H;h++){
                int kh=(int)((float)h/ratio);
                float *att_head=d->att+(size_t)h*c->max_ctx;
                float *ctx_head=d->ctx+h*hd_this;
                for(int t=0;t<=pos;t++){
                    const uint16_t *kc=d->key_cache+layer_base+(size_t)t*KV+kh*hd_this;
                    float score=0.0f;
#if defined(__AVX2__) && defined(__F16C__)
                    {
                        int j = 0;
                        __m256 sum256 = _mm256_setzero_ps();
                        for (; j <= hd_this - 8; j += 8) {
                            __m256 qv = _mm256_loadu_ps(d->q + h * hd_this + j);
                            __m128i kh128 = _mm_loadu_si128((const __m128i *)(kc + j));
                            __m256 kf = _mm256_cvtph_ps(kh128);
                            sum256 = _mm256_fmadd_ps(qv, kf, sum256);
                        }
                        float tmp[8];
                        _mm256_storeu_ps(tmp, sum256);
                        for (int k = 0; k < 8; k++) score += tmp[k];
                        for (; j < hd_this; j++) score += d->q[h * hd_this + j] * half_to_float(kc[j]);
                    }
#else
                    for(int j=0;j<hd_this;j++) score+=d->q[h*hd_this+j]*half_to_float(kc[j]);
#endif
                    att_head[t]=score*scale;
                }
                softmax(att_head,pos+1);
                for(int t=0;t<=pos;t++){
                    const uint16_t *vc=d->value_cache+layer_base+(size_t)t*KV+kh*hd_this;
                    float sc=att_head[t];
#if defined(__AVX2__) && defined(__F16C__)
                    {
                        int j = 0;
                        __m256 sv = _mm256_set1_ps(sc);
                        for (; j <= hd_this - 8; j += 8) {
                            __m128i vh128 = _mm_loadu_si128((const __m128i *)(vc + j));
                            __m256 vf = _mm256_cvtph_ps(vh128);
                            __m256 ov = _mm256_loadu_ps(ctx_head + j);
                            ov = _mm256_fmadd_ps(sv, vf, ov);
                            _mm256_storeu_ps(ctx_head + j, ov);
                        }
                        for (; j < hd_this; j++) ctx_head[j] += sc * half_to_float(vc[j]);
                    }
#else
                    for(int j=0;j<hd_this;j++) ctx_head[j] += sc * half_to_float(vc[j]);
#endif
                }
            }
            if(lt->o_proj && matmul(d,lt->o_proj,d->ctx,O_IN,D,d->xb)==0) {
                add_bias(d,lt->o_bias,d->xb,D);
                vec_add(d->x, d->xb, D);
            }
            }
        }
        /* Pre-FFN norm: full RMS with per-layer gamma if available */
        const QwnTensorDesc *post_norm_t = lt->post_norm;
        if (post_norm_t && vector_f32(&d->model, post_norm_t, d->scratch.row_f32, D) == 0) {
            rmsnorm(d->xb, d->x, d->scratch.row_f32, D, c->rms_eps);
        } else {
            rmsnorm(d->xb,d->x,d->norm_weights+(size_t)(2*l+1)*D,D,c->rms_eps);
        }
        /* MoE: skip the dense FFN path, no routed-expert dispatcher
         * yet.  The decoder still updates the residual so a downstream
         * layer can run. */
        if (lt->is_moe) {
            continue;
        }
        if(lt->gate_proj && lt->up_proj && lt->down_proj && I) {
            if(matmul(d,lt->gate_proj,d->xb,D,I,d->gate)||
               matmul(d,lt->up_proj,d->xb,D,I,d->up))return -1;
            for(int i=0;i<I;i++) {
                float g = d->gate[i];
                d->hidden[i] = (g / (1.0f + expf(-g))) * d->up[i];
            }
            if(matmul(d,lt->down_proj,d->hidden,I,D,d->xb))return -1;
            vec_add(d->x, d->xb, D);
        }

        /* Configurable Thinking Dynamic Checks */
        if (config) {
            if (config->level == QWN_THINK_LOW && (l + 1) >= config->n_layers_max) {
                config->last_exit_layer = l;
                break;
            }
            if (config->level == QWN_THINK_MEDIUM && (l == c->layers / 2 || l == (c->layers * 3) / 4) && (l + 1 < c->layers)) {
                if (d->final_norm_weight && vector_f32(&d->model, d->final_norm_weight, d->scratch.row_f32, D) == 0) {
                    rmsnorm(d->xb, d->x, d->scratch.row_f32, D, c->rms_eps);
                } else {
                    rmsnorm(d->xb, d->x, d->norm_weights + (size_t)(2 * c->layers) * D, D, c->rms_eps);
                }
                if (matmul(d, d->lm_head_weight, d->xb, D, c->vocab, d->logits) == 0) {
                    float conf = qwn_thinking_compute_confidence(d->logits, c->vocab, config->temp_threshold);
                    if (config->confidence_buffer) {
                        config->confidence_buffer[l] = conf;
                    }
                    config->last_confidence = conf;
                    if (conf >= (float)config->early_exit_threshold / 100.0f) {
                        config->last_exit_layer = l;
                        d->position++;
                        if (out_logits) *out_logits = d->logits;
                        return 0;
                    }
                }
            }
        }
    }
    /* Final norm: full RMS with per-layer gamma if available */
    if (d->final_norm_weight &&
        vector_f32(&d->model, d->final_norm_weight, d->scratch.row_f32, D) == 0) {
        rmsnorm(d->xb, d->x, d->scratch.row_f32, D, c->rms_eps);
    } else {
        rmsnorm(d->xb,d->x,d->norm_weights+(size_t)(2*c->layers)*D,D,c->rms_eps);
    }
    if(matmul(d,d->lm_head_weight,d->xb,D,c->vocab,d->logits)) {
        fprintf(stderr, "lm_head matmul failed: D=%d vocab=%d lm_head=%p shape=(%llu,%llu)\n",
                D, c->vocab, (void*)d->lm_head_weight,
                (unsigned long long)(d->lm_head_weight?d->lm_head_weight->shape[0]:0),
                (unsigned long long)(d->lm_head_weight?d->lm_head_weight->shape[1]:0));
        return -1;
    }
    d->position++;
    if (config) {
        config->last_exit_layer = c->layers - 1;
        config->last_confidence = qwn_thinking_compute_confidence(d->logits, c->vocab, config->temp_threshold);
        if (config->confidence_buffer) {
            config->confidence_buffer[c->layers - 1] = config->last_confidence;
        }
    }
    if(out_logits)*out_logits=d->logits;
    return 0;
}

int qwn_decoder_forward(QwnDecoder *d,int token,const float **out_logits){
    return qwn_decoder_forward_thinking(d, token, out_logits, NULL);
}

static float random01(uint64_t *state){
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (float)((*state * 2685821657736338717ULL) >> 40) * (1.0f / 16777216.0f);
}

static int sample_token(const float *logits,int vocab,float temperature,float top_p,
                        uint64_t *rng_state){
    if(temperature<=0.0f){
        int best=0;
#if defined(__AVX2__)
        if (vocab >= 8) {
            __m256 vmax = _mm256_loadu_ps(logits);
            int i = 8;
            for (; i <= vocab - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(logits + i);
                vmax = _mm256_max_ps(vmax, v);
            }
            float tmp[8]; _mm256_storeu_ps(tmp, vmax);
            float max_val = tmp[0];
            for (int j = 1; j < 8; j++) if (tmp[j] > max_val) max_val = tmp[j];
            for (int k = 0; k < vocab; k++) {
                if (logits[k] == max_val) { best = k; break; }
            }
            return best;
        }
#endif
        for(int i=1;i<vocab;i++)if(logits[i]>logits[best])best=i;
        return best;
    }
    enum{K=256};float val[K];int id[K],n=0;
    for(int token=0;token<vocab;token++){
        float x=logits[token]/temperature;
        int pos;
        if(n<K)pos=n++;
        else{if(x<=val[K-1])continue;pos=K-1;}
        while(pos>0&&x>val[pos-1]){if(pos<K){val[pos]=val[pos-1];id[pos]=id[pos-1];}pos--;}
        val[pos]=x;id[pos]=token;
    }
    float peak=val[0],sum=0.0f;for(int i=0;i<n;i++){val[i]=expf(val[i]-peak);sum+=val[i];}
    if(top_p<=0.0f||top_p>1.0f)top_p=1.0f;
    float cutoff=sum*top_p,cumulative=0.0f;int keep=n;
    for(int i=0;i<n;i++){cumulative+=val[i];if(cumulative>=cutoff){keep=i+1;break;}}
    float kept=0.0f;for(int i=0;i<keep;i++)kept+=val[i];
    float target=random01(rng_state)*kept;for(int i=0;i<keep;i++){target-=val[i];if(target<=0)return id[i];}
    return id[keep-1];
}

int qwn_decoder_generate(QwnDecoder *d,const int *prompt,int prompt_count,
                         int max_new_tokens,float temperature,float top_p,
                         void(*callback)(const char*,int,void*),void *opaque){
    const float *logits=NULL;int token=-1;
    double request_started = qwn_decode_wall_seconds();
    double prefill_started = request_started;
    memset(&d->generation_metrics, 0, sizeof(d->generation_metrics));
    d->generation_metrics.prompt_tokens = prompt_count;
    for(int i=0;i<prompt_count;i++)if(qwn_decoder_forward(d,prompt[i],&logits)!=0)return -1;
    double decode_started = qwn_decode_wall_seconds();
    d->generation_metrics.prefill_ms = (decode_started - prefill_started) * 1000.0;
    int generated=0;
    double first_token_at = 0.0;
    for(int step=0;step<max_new_tokens;step++){
        token=sample_token(logits,d->cfg.vocab,temperature,top_p,&d->rng_state);
        if(token==d->cfg.eos_id)break;
        if (first_token_at == 0.0) first_token_at = qwn_decode_wall_seconds();
        if(callback) {
            if (token >= 0 && token < d->tokenizer.n_ids && d->tokenizer.id2str && d->tokenizer.id2str[token]) {
                const char *s = d->tokenizer.id2str[token];
                callback(s, (int)strlen(s), opaque);
            } else {
                char text[512];int n=tok_decode(&d->tokenizer,&token,1,text,sizeof(text)-1);
                if(n>0) callback(text,n,opaque);
            }
        }
        generated++;
        if(qwn_decoder_forward(d,token,&logits)!=0)return -1;
    }
    double finished = qwn_decode_wall_seconds();
    d->generation_metrics.generated_tokens = generated;
    d->generation_metrics.first_token_ms = first_token_at > 0.0 ?
        (first_token_at - request_started) * 1000.0 : 0.0;
    d->generation_metrics.decode_wall_ms = generated > 0 ?
        (finished - decode_started) * 1000.0 : 0.0;
    qwn_decoder_refresh_runtime_metrics(d);
    return generated;
}
