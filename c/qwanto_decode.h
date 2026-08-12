#ifndef QWANTO_DECODE_H
#define QWANTO_DECODE_H

#include "qwanto_native.h"
#include "qwanto_kernels.h"
#include "tok.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

typedef struct {
    int hidden, intermediate, layers, heads, kv_heads, head_dim;
    int vocab, max_ctx, bos_id, eos_id;
    float rms_eps, rope_theta;
    int tie_embeddings;
} QwnConfig;

typedef struct {
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
#ifdef COLI_CUDA
    int cuda_device;
    int cuda_enabled;
    struct { const QwnTensorDesc *desc; ColiCudaTensor *tensor; } cuda_weights[128];
    int cuda_weight_count;
#endif
} QwnDecoder;

int qwn_decoder_open(QwnDecoder *d, const char *path, int ctx_size,
                     const char **error);
void qwn_decoder_close(QwnDecoder *d);
void qwn_decoder_reset(QwnDecoder *d);

/* Consume one token and return logits predicting the next token. */
int qwn_decoder_forward(QwnDecoder *d, int token, const float **logits);

/* Greedy decode. callback receives each decoded byte chunk. */
int qwn_decoder_generate(QwnDecoder *d, const int *prompt, int prompt_count,
                         int max_new_tokens, float temperature, float top_p,
                         void (*callback)(const char *, int, void *), void *opaque);

#endif
