#include "qwanto_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
                c->kv_heads=cfg_int(root,"num_key_value_heads",c->heads);
                c->head_dim=cfg_int(root,"head_dim",c->heads?c->hidden/c->heads:0);
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
    return c->hidden>0&&c->intermediate>0&&c->layers>0&&c->heads>0&&
           c->kv_heads>0&&c->head_dim>0&&c->vocab>0 ? 0:-1;
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

/* ---- Precomputed RoPE frequency table ---- */
static float *rope_cos_cache = NULL;
static float *rope_sin_cache = NULL;
static int    rope_cache_ctx = 0;
static int    rope_cache_half = 0;

static void rope_cache_ensure(int max_ctx, int half_dim, float theta) {
    if (rope_cos_cache && rope_cache_ctx >= max_ctx && rope_cache_half >= half_dim)
        return;
    free(rope_cos_cache); free(rope_sin_cache);
    rope_cache_ctx = max_ctx;
    rope_cache_half = half_dim;
    size_t n = (size_t)max_ctx * (size_t)half_dim;
    rope_cos_cache = (float *)malloc(n * sizeof(float));
    rope_sin_cache = (float *)malloc(n * sizeof(float));
    if (!rope_cos_cache || !rope_sin_cache) return;
    for (int pos = 0; pos < max_ctx; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos * powf(theta, -2.0f * i / (2 * half_dim));
            rope_cos_cache[(size_t)pos * half_dim + i] = cosf(angle);
            rope_sin_cache[(size_t)pos * half_dim + i] = sinf(angle);
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

static void rope(float *v, int heads, int dim, int pos, float theta) {
    int half = dim / 2;
    /* Use precomputed table if available */
    if (rope_cos_cache && pos < rope_cache_ctx && half <= rope_cache_half) {
        const float *ct = rope_cos_cache + (size_t)pos * rope_cache_half;
        const float *st = rope_sin_cache + (size_t)pos * rope_cache_half;
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
#ifdef COLI_CUDA
    if(d->cuda_enabled && w && w->dtype==QWN_DT_Q4_0){
        int slot=-1;
        for(int i=0;i<d->cuda_weight_count;i++)if(d->cuda_weights[i].desc==w){slot=i;break;}
        if(slot<0 && d->cuda_weight_count<(int)(sizeof(d->cuda_weights)/sizeof(d->cuda_weights[0])))
            slot=d->cuda_weight_count++;
        if(slot>=0){
            if(d->cuda_weights[slot].desc==NULL)d->cuda_weights[slot].desc=w;
            if(coli_cuda_qwn_matmul(&d->cuda_weights[slot].tensor,y,x,qwn_data(&d->model,w),1,in,out,d->cuda_device))return 0;
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
    int D=d->cfg.hidden;
    for(int l=0;l<d->cfg.layers;l++){
        const QwnTensorDesc *in_norm = layer_tensor(d,l,"input_layernorm.weight");
        if(in_norm && vector_f32(&d->model,in_norm,d->norm_weights+(size_t)(2*l)*D,D)!=0)return -1;
        const QwnTensorDesc *post_norm = layer_tensor(d,l,"post_attention_layernorm.weight");
        if(post_norm && vector_f32(&d->model,post_norm,d->norm_weights+(size_t)(2*l+1)*D,D)!=0)return -1;
    }
    const QwnTensorDesc *fn = qwn_find(&d->model,"model.norm.weight");
    if(!fn) fn = qwn_find(&d->model,"output_norm.weight");
    if(!fn) fn = qwn_find(&d->model,"norm.weight");
    return fn ? vector_f32(&d->model,fn,d->norm_weights+(size_t)(2*d->cfg.layers)*D,D) : 0;
}

/* Resolve all per-layer tensor descriptors once at load time */
static int build_layer_cache(QwnDecoder *d) {
    d->layer_cache = (QwnLayerTensors *)malloc((size_t)d->cfg.layers * sizeof(QwnLayerTensors));
    if (!d->layer_cache) return -1;
    for (int l = 0; l < d->cfg.layers; l++) {
        QwnLayerTensors *lt = &d->layer_cache[l];
        lt->q_proj = layer_tensor(d, l, "self_attn.q_proj.weight");
        lt->k_proj = layer_tensor(d, l, "self_attn.k_proj.weight");
        lt->v_proj = layer_tensor(d, l, "self_attn.v_proj.weight");
        lt->o_proj = layer_tensor(d, l, "self_attn.o_proj.weight");
        lt->q_bias = layer_tensor(d, l, "self_attn.q_proj.bias");
        lt->k_bias = layer_tensor(d, l, "self_attn.k_proj.bias");
        lt->v_bias = layer_tensor(d, l, "self_attn.v_proj.bias");
        lt->o_bias = layer_tensor(d, l, "self_attn.o_proj.bias");
        lt->q_norm = layer_tensor(d, l, "self_attn.q_norm.weight");
        lt->k_norm = layer_tensor(d, l, "self_attn.k_norm.weight");
        lt->gate_proj = layer_tensor(d, l, "mlp.gate_proj.weight");
        lt->up_proj = layer_tensor(d, l, "mlp.up_proj.weight");
        lt->down_proj = layer_tensor(d, l, "mlp.down_proj.weight");
    }
    d->embed_weight = qwn_find(&d->model, "model.embed_tokens.weight");
    if(!d->embed_weight) d->embed_weight = qwn_find(&d->model, "token_embd.weight");
    d->lm_head_weight = qwn_find(&d->model, "lm_head.weight");
    if (!d->lm_head_weight) d->lm_head_weight = qwn_find(&d->model, "output.weight");
    if (!d->lm_head_weight) d->lm_head_weight = d->embed_weight;
    return 0;
}

int qwn_decoder_open(QwnDecoder *d,const char *path,int ctx_size,const char **error){
    static const char *ERR_CONFIG="unsupported/missing Llama-Qwen config";
    static const char *ERR_TENSORS="missing Llama-Qwen tensors";
    static const char *ERR_MEMORY="native decoder allocation failed";
    memset(d,0,sizeof(*d));if(error)*error=NULL;
    if(qwn_open(path,&d->model,error)!=0)return -1;
    if(load_config(d)!=0||load_tokenizer(d)!=0){if(error)*error=ERR_CONFIG;goto fail;}
    if(ctx_size>0&&ctx_size<d->cfg.max_ctx)d->cfg.max_ctx=ctx_size;
    if(required_tensors(d)!=0){if(error)*error=ERR_TENSORS;goto fail;}
    int D=d->cfg.hidden,I=d->cfg.intermediate,Hd=d->cfg.head_dim;
    int Q=d->cfg.heads*Hd,KV=d->cfg.kv_heads*Hd,V=d->cfg.vocab;
    int max_dim=D>I?D:I;if(Q>max_dim)max_dim=Q;if(V>max_dim)max_dim=V;
    if(qwn_scratch_init(&d->scratch,1,max_dim)!=0){if(error)*error=ERR_MEMORY;goto fail;}
    size_t floats=(size_t)D*5+(size_t)Q+(size_t)KV*2+(size_t)d->cfg.heads*(size_t)d->cfg.max_ctx+
                  (size_t)I*3+(size_t)V+(size_t)(2*d->cfg.layers+1)*D;
    d->arena_bytes=up64(floats*sizeof(float));d->arena=alloc64(d->arena_bytes);
    if(!d->arena){if(error)*error=ERR_MEMORY;goto fail;}
    float *p=(float*)d->arena;
    d->x=p;p+=D;d->xb=p;p+=D;d->q=p;p+=Q;d->k=p;p+=KV;d->v=p;p+=KV;
    d->ctx=p;p+=D;d->att=p;p+=(size_t)d->cfg.heads*(size_t)d->cfg.max_ctx;d->gate=p;p+=I;
    d->up=p;p+=I;d->hidden=p;p+=I;d->logits=p;p+=V;d->norm_weights=p;
    size_t kv_elems=(size_t)d->cfg.layers*d->cfg.max_ctx*d->cfg.kv_heads*Hd;
    size_t kv_bytes=up64(kv_elems*sizeof(uint16_t));
    d->kv_allocation=alloc64(kv_bytes*2);if(!d->kv_allocation){if(error)*error=ERR_MEMORY;goto fail;}
    d->key_cache=(uint16_t*)d->kv_allocation;
    d->value_cache=(uint16_t*)((uint8_t*)d->kv_allocation+kv_bytes);
    memset(d->kv_allocation,0,kv_bytes*2);
    if(load_norms(d)!=0){if(error)*error=ERR_TENSORS;goto fail;}
    if(build_layer_cache(d)!=0){if(error)*error=ERR_MEMORY;goto fail;}
#ifdef COLI_CUDA
    d->cuda_device=0;
    d->cuda_enabled=coli_cuda_init(&d->cuda_device,1);
#endif
    /* Precompute RoPE table once at load time */
    rope_cache_ensure(d->cfg.max_ctx, d->cfg.head_dim / 2, d->cfg.rope_theta);
    return 0;
fail:qwn_decoder_close(d);return -1;
}

void qwn_decoder_reset(QwnDecoder *d){d->position=0;}

void qwn_decoder_close(QwnDecoder *d){
    if(!d)return;qwn_scratch_destroy(&d->scratch);free64(d->arena);
    free(d->layer_cache);
#ifdef COLI_CUDA
    for(int i=0;i<d->cuda_weight_count;i++)if(d->cuda_weights[i].tensor)coli_cuda_tensor_free(d->cuda_weights[i].tensor);
    if(d->cuda_enabled)coli_cuda_shutdown();
#endif
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

int qwn_decoder_forward(QwnDecoder *d,int token,const float **out_logits){
    QwnConfig *c=&d->cfg;int D=c->hidden,I=c->intermediate,H=c->heads;
    int HK=c->kv_heads,HD=c->head_dim,Q=H*HD,KV=HK*HD,pos=d->position;
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
    for(int l=0;l<c->layers;l++){
        const QwnLayerTensors *lt = &d->layer_cache[l];
        /* Prefetch next layer's heavy weights — zero-cost cached pointer lookup */
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
        rmsnorm(d->xb,d->x,d->norm_weights+(size_t)(2*l)*D,D,c->rms_eps);
        if (lt->q_proj && lt->k_proj && lt->v_proj) {
            if(matmul(d,lt->q_proj,d->xb,D,Q,d->q)||matmul(d,lt->k_proj,d->xb,D,KV,d->k)||
               matmul(d,lt->v_proj,d->xb,D,KV,d->v)) {
                fprintf(stderr, "layer %d attn matmul failed\n", l);
                return -1;
            }
            add_bias(d,lt->q_bias,d->q,Q);
            add_bias(d,lt->k_bias,d->k,KV);
            add_bias(d,lt->v_bias,d->v,KV);
            if(lt->q_norm&&vector_f32(&d->model,lt->q_norm,d->scratch.row_f32,HD)==0)
                head_rmsnorm(d->q,H,HD,d->scratch.row_f32,c->rms_eps);
            if(lt->k_norm&&vector_f32(&d->model,lt->k_norm,d->scratch.row_f32,HD)==0)
                head_rmsnorm(d->k,HK,HD,d->scratch.row_f32,c->rms_eps);
            rope(d->q,H,HD,pos,c->rope_theta);rope(d->k,HK,HD,pos,c->rope_theta);
            size_t layer_base=(size_t)l*c->max_ctx*HK*HD;
            size_t pos_base=layer_base+(size_t)pos*HK*HD;
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
            memset(d->ctx,0,(size_t)D*sizeof(float));
            float scale=1.0f/sqrtf((float)HD),ratio=(float)H/HK;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) if(H > 1)
#endif
            for(int h=0;h<H;h++){
                int kh=(int)((float)h/ratio);
                const float *q_head=d->q+h*HD;
                float *scores=d->att+(size_t)h*c->max_ctx;
                for(int t=0;t<=pos;t++){
                    size_t base=layer_base+(size_t)t*HK*HD+(size_t)kh*HD;
                    const uint16_t *kc=d->key_cache+base;
                    float sum=0;
#if defined(__AVX2__) && defined(__F16C__)
                    {
                        int j = 0;
                        __m256 acc = _mm256_setzero_ps();
                        for (; j <= HD - 8; j += 8) {
                            __m128i h8 = _mm_loadu_si128((const __m128i *)(kc + j));
                            __m256 kf = _mm256_cvtph_ps(h8);
                            __m256 qf = _mm256_loadu_ps(q_head + j);
                            acc = _mm256_fmadd_ps(qf, kf, acc);
                        }
                        float tmp[8]; _mm256_storeu_ps(tmp, acc);
                        sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                        for (; j < HD; j++) sum += q_head[j] * half_to_float(kc[j]);
                    }
#else
                    for(int j=0;j<HD;j++) sum += q_head[j] * half_to_float(kc[j]);
#endif
                    scores[t]=sum*scale;
                }
                softmax(scores,pos+1);
                float *ctx_head = d->ctx + h * HD;
                for(int t=0;t<=pos;t++){
                    size_t base=layer_base+(size_t)t*HK*HD+(size_t)kh*HD;
                    const uint16_t *vc = d->value_cache + base;
                    float sc = scores[t];
#if defined(__AVX2__) && defined(__F16C__)
                    {
                        __m256 sv = _mm256_set1_ps(sc);
                        int j = 0;
                        for (; j <= HD - 8; j += 8) {
                            __m128i h8 = _mm_loadu_si128((const __m128i *)(vc + j));
                            __m256 vf = _mm256_cvtph_ps(h8);
                            __m256 ov = _mm256_loadu_ps(ctx_head + j);
                            ov = _mm256_fmadd_ps(sv, vf, ov);
                            _mm256_storeu_ps(ctx_head + j, ov);
                        }
                        for (; j < HD; j++) ctx_head[j] += sc * half_to_float(vc[j]);
                    }
#else
                    for(int j=0;j<HD;j++) ctx_head[j] += sc * half_to_float(vc[j]);
#endif
                }
            }
            if(lt->o_proj && matmul(d,lt->o_proj,d->ctx,Q,D,d->xb)==0) {
                add_bias(d,lt->o_bias,d->xb,D);
                vec_add(d->x, d->xb, D);
            }
        }
        rmsnorm(d->xb,d->x,d->norm_weights+(size_t)(2*l+1)*D,D,c->rms_eps);
        if(lt->gate_proj && lt->up_proj && lt->down_proj) {
            if(matmul(d,lt->gate_proj,d->xb,D,I,d->gate)||
               matmul(d,lt->up_proj,d->xb,D,I,d->up))return -1;
            for(int i=0;i<I;i++) {
                float g = d->gate[i];
                d->hidden[i] = (g / (1.0f + expf(-g))) * d->up[i];
            }
            if(matmul(d,lt->down_proj,d->hidden,I,D,d->xb))return -1;
            vec_add(d->x, d->xb, D);
        }
    }
    rmsnorm(d->xb,d->x,d->norm_weights+(size_t)(2*c->layers)*D,D,c->rms_eps);
    if(matmul(d,d->lm_head_weight,d->xb,D,c->vocab,d->logits)) {
        fprintf(stderr, "lm_head matmul failed: D=%d vocab=%d lm_head=%p shape=(%llu,%llu)\n",
                D, c->vocab, (void*)d->lm_head_weight,
                (unsigned long long)(d->lm_head_weight?d->lm_head_weight->shape[0]:0),
                (unsigned long long)(d->lm_head_weight?d->lm_head_weight->shape[1]:0));
        return -1;
    }
    d->position++;if(out_logits)*out_logits=d->logits;return 0;
}

static uint64_t sample_rng=0x9e3779b97f4a7c15ULL;
static float random01(void){
    sample_rng^=sample_rng>>12;sample_rng^=sample_rng<<25;sample_rng^=sample_rng>>27;
    return (float)((sample_rng*2685821657736338717ULL)>>40)*(1.0f/16777216.0f);
}

static int sample_token(const float *logits,int vocab,float temperature,float top_p){
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
    float target=random01()*kept;for(int i=0;i<keep;i++){target-=val[i];if(target<=0)return id[i];}
    return id[keep-1];
}

int qwn_decoder_generate(QwnDecoder *d,const int *prompt,int prompt_count,
                         int max_new_tokens,float temperature,float top_p,
                         void(*callback)(const char*,int,void*),void *opaque){
    const float *logits=NULL;int token=-1;
    for(int i=0;i<prompt_count;i++)if(qwn_decoder_forward(d,prompt[i],&logits)!=0)return -1;
    int generated=0;
    for(int step=0;step<max_new_tokens;step++){
        token=sample_token(logits,d->cfg.vocab,temperature,top_p);
        if(token==d->cfg.eos_id)break;
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
    }return generated;
}
