#include "qwanto_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static size_t round_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

static void *aligned_malloc64(size_t bytes) {
    bytes = round_up(bytes, 64);
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
#endif
}

static void aligned_free64(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

int qwn_scratch_init(QwnScratch *s, int max_tokens, int max_k) {
    if (!s || max_tokens < 1 || max_k < 1) return -1;
    memset(s, 0, sizeof(*s));
    int padded_k = (max_k + 31) & ~31;
    size_t q8_bytes = round_up((size_t)max_tokens * (size_t)padded_k, 64);
    size_t scale_bytes = round_up((size_t)max_tokens * sizeof(float), 64);
    size_t row_bytes = round_up((size_t)padded_k * sizeof(float), 64);
    size_t total = q8_bytes + scale_bytes + row_bytes;
    uint8_t *p = (uint8_t *)aligned_malloc64(total);
    if (!p) return -1;
    memset(p, 0, total);
    s->allocation = p;
    s->q8 = (int8_t *)p;
    s->token_scales = (float *)(p + q8_bytes);
    s->row_f32 = (float *)(p + q8_bytes + scale_bytes);
    s->bytes = total;
    s->max_tokens = max_tokens;
    s->padded_k = padded_k;
    return 0;
}

void qwn_scratch_destroy(QwnScratch *s) {
    if (!s) return;
    aligned_free64(s->allocation);
    memset(s, 0, sizeof(*s));
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            int e = -14;
            while ((mant & 0x400) == 0) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = (sign << 31) | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float bf16_to_float(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float out; memcpy(&out, &bits, sizeof(out)); return out;
}

#if defined(_OPENMP)
#include <omp.h>
#endif

static void quantize_tokens(const float *x, int M, int K, QwnScratch *s) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(M > 4)
#endif
    for (int t = 0; t < M; t++) {
        const float *row = x + (size_t)t * K;
        int8_t *q = s->q8 + (size_t)t * s->padded_k;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / scale;
        s->token_scales[t] = scale;
        for (int k = 0; k < K; k++) {
            float v = row[k] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q[k] = (int8_t)lrintf(v);
        }
        memset(q + K, 0, (size_t)(s->padded_k - K));
    }
}

static int32_t dot_q4_q8_block(const uint8_t *packed, const int8_t *q8,
                               int valid) {
#if defined(__AVX2__)
    if (valid == 32) {
        const __m128i p = _mm_loadu_si128((const __m128i *)packed);
        const __m128i mask = _mm_set1_epi8(0x0f);
        const __m128i lo = _mm_and_si128(p, mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(p, 4), mask);
        __m256i q4 = _mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi));
        q4 = _mm256_inserti128_si256(q4, _mm_unpackhi_epi8(lo, hi), 1);
        const __m256i q8v = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i pair_dot = _mm256_maddubs_epi16(q4, q8v);
        const __m256i dot32 = _mm256_madd_epi16(pair_dot, ones16);
        const __m256i pair_sum = _mm256_maddubs_epi16(ones8, q8v);
        const __m256i sum32 = _mm256_madd_epi16(pair_sum, ones16);
        int32_t dot_lane[8], sum_lane[8];
        _mm256_storeu_si256((__m256i *)dot_lane, dot32);
        _mm256_storeu_si256((__m256i *)sum_lane, sum32);
        int32_t dot = 0, qsum = 0;
        for (int i = 0; i < 8; i++) { dot += dot_lane[i]; qsum += sum_lane[i]; }
        return dot - 8 * qsum;
    }
#endif
    int32_t sum = 0;
    for (int i = 0; i < valid; i++) {
        uint8_t byte = packed[i >> 1];
        int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
        sum += w * (int32_t)q8[i];
    }
    return sum;
}

int qwn_matmul_q4_0_f32(const QwnModel *m,
                        const QwnTensorDesc *weights,
                        const float *x, int M, int K, int N,
                        QwnScratch *scratch,
                        float *y) {
    if (!m || !weights || !x || !scratch || !y || M < 1 || K < 1 || N < 1)
        return -1;
    if (weights->dtype != QWN_DT_Q4_0 || weights->n_dims != 2 ||
        weights->shape[0] != (uint64_t)K || weights->shape[1] != (uint64_t)N)
        return -1;
    if ((weights->byte_offset & 63ULL) != 0 || M > scratch->max_tokens ||
        ((K + 31) & ~31) > scratch->padded_k)
        return -1;

    const int blocks = (K + 31) / 32;
    const uint64_t row_bytes = (uint64_t)blocks * 18ULL;
    const uint64_t raw_bytes = row_bytes * (uint64_t)N;
    if (weights->byte_offset > m->file_size ||
        raw_bytes > m->file_size - weights->byte_offset ||
        raw_bytes > weights->byte_size)
        return -1;

    quantize_tokens(x, M, K, scratch);
    const uint8_t *raw = m->base + weights->byte_offset;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(N > 16)
#endif
    for (int n = 0; n < N; n++) {
        const uint8_t *row = raw + (uint64_t)n * row_bytes;
        for (int t = 0; t < M; t++) {
            const int8_t *q8 = scratch->q8 + (size_t)t * scratch->padded_k;
            float x_scale = scratch->token_scales[t];
            float sum = 0.0f;
            for (int b = 0; b < blocks; b++) {
                const uint8_t *block = row + (size_t)b * 18;
                uint16_t hs;
                memcpy(&hs, block, sizeof(hs));
                float w_scale = half_to_float(hs);
                int valid = K - b * 32;
                if (valid > 32) valid = 32;
                sum += (float)dot_q4_q8_block(block + 2,
                                             q8 + b * 32, valid) *
                       (x_scale * w_scale);
            }
            y[(size_t)t * N + n] = sum;
        }
    }
    return 0;
}

int qwn_row_f32(const QwnModel *m, const QwnTensorDesc *t,
                int row, float *out, int width) {
    if (!m || !t || !out || t->n_dims != 2 || row < 0 ||
        row >= (int)t->shape[1] || width != (int)t->shape[0]) return -1;
    const uint8_t *raw = (const uint8_t *)qwn_data(m, t);
    if (!raw) return -1;
    if (t->dtype == QWN_DT_Q4_0) {
        int blocks = (width + 31) / 32;
        const uint8_t *p = raw + (size_t)row * blocks * 18;
        for (int b = 0; b < blocks; b++) {
            uint16_t hs; memcpy(&hs, p + b * 18, 2);
            float scale = half_to_float(hs);
            for (int i = 0; i < 32 && b * 32 + i < width; i++) {
                uint8_t byte = p[b * 18 + 2 + (i >> 1)];
                int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                out[b * 32 + i] = q * scale;
            }
        }
        return 0;
    }
    if (t->dtype == QWN_DT_F32) {
        memcpy(out, raw + (size_t)row * width * 4, (size_t)width * 4);
        return 0;
    }
    if (t->dtype == QWN_DT_F16 || t->dtype == QWN_DT_BF16) {
        const uint16_t *p = (const uint16_t *)raw + (size_t)row * width;
        for (int i = 0; i < width; i++)
            out[i] = t->dtype == QWN_DT_F16 ? half_to_float(p[i]) : bf16_to_float(p[i]);
        return 0;
    }
    return -1;
}

int qwn_matmul_f32(const QwnModel *m, const QwnTensorDesc *w,
                   const float *x, int M, int K, int N,
                   QwnScratch *scratch, float *y) {
    if (!w || w->n_dims != 2 || w->shape[0] != (uint64_t)K ||
        w->shape[1] != (uint64_t)N) return -1;
    if (w->dtype == QWN_DT_Q4_0)
        return qwn_matmul_q4_0_f32(m,w,x,M,K,N,scratch,y);
    if (!scratch || K > scratch->padded_k) return -1;
    float *row = scratch->row_f32;
    for (int n = 0; n < N; n++) {
        if (qwn_row_f32(m,w,n,row,K) != 0) return -1;
        for (int token = 0; token < M; token++) {
            float sum = 0.0f;
            const float *xr = x + (size_t)token * K;
            for (int k = 0; k < K; k++) sum += xr[k] * row[k];
            y[(size_t)token * N + n] = sum;
        }
    }
    return 0;
}
