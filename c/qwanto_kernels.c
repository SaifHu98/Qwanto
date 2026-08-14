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
#if defined(__AVX2__)
        {
            __m256 vmax = _mm256_setzero_ps();
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 v = _mm256_loadu_ps(&row[k]);
                __m256 av = _mm256_andnot_ps(sign_mask, v);
                vmax = _mm256_max_ps(vmax, av);
            }
            float tmp[8]; _mm256_storeu_ps(tmp, vmax);
            for (int i = 0; i < 8; i++) if (tmp[i] > amax) amax = tmp[i];
            for (; k < K; k++) {
                float a = fabsf(row[k]);
                if (a > amax) amax = a;
            }
        }
#else
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
#endif
        float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / scale;
        s->token_scales[t] = scale;
#if defined(__AVX2__)
        {
            __m256 vinv = _mm256_set1_ps(inv);
            __m256 vhi = _mm256_set1_ps(127.0f);
            __m256 vlo = _mm256_set1_ps(-127.0f);
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(&row[k]), vinv);
                v = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
                __m256i vi = _mm256_cvtps_epi32(v);
                /* Pack 8 int32 -> 8 int8 */
                __m128i lo = _mm256_castsi256_si128(vi);
                __m128i hi = _mm256_extracti128_si256(vi, 1);
                __m128i packed16 = _mm_packs_epi32(lo, hi);
                __m128i packed8  = _mm_packs_epi16(packed16, packed16);
                /* Store 8 bytes */
                *(int64_t *)(q + k) = _mm_cvtsi128_si64(packed8);
            }
            for (; k < K; k++) {
                float v = row[k] * inv;
                if (v > 127.0f) v = 127.0f;
                if (v < -127.0f) v = -127.0f;
                q[k] = (int8_t)lrintf(v);
            }
        }
#else
        for (int k = 0; k < K; k++) {
            float v = row[k] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q[k] = (int8_t)lrintf(v);
        }
#endif
        memset(q + K, 0, (size_t)(s->padded_k - K));
    }
}

#if defined(__AVX2__)
static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    __m128i tmp = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2));
    sum128 = _mm_add_epi32(sum128, tmp);
    tmp = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
    sum128 = _mm_add_epi32(sum128, tmp);
    return _mm_cvtsi128_si32(sum128);
}
#endif

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
        return hsum_epi32_avx2(dot32) - 8 * hsum_epi32_avx2(sum32);
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
    if ((weights->dtype != QWN_DT_Q4_0 && weights->dtype != QWN_DT_VSQ && weights->dtype != QWN_DT_VSQ_ULTRA && weights->dtype != QWN_DT_HYPER_VSQ) || weights->n_dims != 2 ||
        weights->shape[0] != (uint64_t)K || weights->shape[1] != (uint64_t)N)
        return -1;
    if ((weights->byte_offset & 63ULL) != 0 || M > scratch->max_tokens ||
        ((K + 255) & ~255) > scratch->padded_k)
        return -1;

    const int is_hyper = (weights->dtype == QWN_DT_HYPER_VSQ);
    const int is_vsq_ultra = (weights->dtype == QWN_DT_VSQ_ULTRA);
    const int is_vsq = (weights->dtype == QWN_DT_VSQ);
    const int block_elems = is_hyper ? 256 : (is_vsq_ultra ? 128 : (is_vsq ? 64 : 32));
    const int block_bytes = is_hyper ? 138 : (is_vsq_ultra ? 70 : (is_vsq ? 36 : 18));
    const int blocks = (K + block_elems - 1) / block_elems;
    const uint64_t row_bytes = (uint64_t)blocks * (uint64_t)block_bytes;
    const uint64_t raw_bytes = row_bytes * (uint64_t)N;
    if (weights->byte_offset > m->file_size ||
        raw_bytes > m->file_size - weights->byte_offset ||
        raw_bytes > weights->byte_size)
        return -1;

    quantize_tokens(x, M, K, scratch);
    const uint8_t *raw = m->base + weights->byte_offset;

#if defined(__AVX2__)
    const __m128i mask128 = _mm_set1_epi8(0x0f);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);
#endif

    for (int t = 0; t < M; t++) {
        const int8_t *q8 = scratch->q8 + (size_t)t * scratch->padded_k;
        float x_scale = scratch->token_scales[t];
        
        int n = 0;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(N > 16)
#endif
        for (n = 0; n <= N - 4; n += 4) {
            const uint8_t *r0 = raw + (uint64_t)n * row_bytes;
            const uint8_t *r1 = raw + (uint64_t)(n + 1) * row_bytes;
            const uint8_t *r2 = raw + (uint64_t)(n + 2) * row_bytes;
            const uint8_t *r3 = raw + (uint64_t)(n + 3) * row_bytes;
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (int b = 0; b < blocks; b++) {
                int valid = K - b * 32;
                if (valid > 32) valid = 32;
                const uint8_t *b0 = r0 + (size_t)b * 18;
                const uint8_t *b1 = r1 + (size_t)b * 18;
                const uint8_t *b2 = r2 + (size_t)b * 18;
                const uint8_t *b3 = r3 + (size_t)b * 18;
                uint16_t h0; memcpy(&h0, b0, 2); float ws0 = half_to_float(h0) * x_scale;
                uint16_t h1; memcpy(&h1, b1, 2); float ws1 = half_to_float(h1) * x_scale;
                uint16_t h2; memcpy(&h2, b2, 2); float ws2 = half_to_float(h2) * x_scale;
                uint16_t h3; memcpy(&h3, b3, 2); float ws3 = half_to_float(h3) * x_scale;

#if defined(__AVX2__)
                if (valid == 32) {
                    __m256i q8v = _mm256_loadu_si256((const __m256i *)(q8 + b * 32));
                    const __m256i psum = _mm256_maddubs_epi16(ones8, q8v);
                    const int32_t qsum = hsum_epi32_avx2(_mm256_madd_epi16(psum, ones16));
                    
                    // Row 0
                    __m128i p0 = _mm_loadu_si128((const __m128i *)(b0 + 2));
                    __m128i lo0 = _mm_and_si128(p0, mask128);
                    __m128i hi0 = _mm_and_si128(_mm_srli_epi16(p0, 4), mask128);
                    __m256i q4_0 = _mm256_castsi128_si256(_mm_unpacklo_epi8(lo0, hi0));
                    q4_0 = _mm256_inserti128_si256(q4_0, _mm_unpackhi_epi8(lo0, hi0), 1);
                    __m256i pdot0 = _mm256_maddubs_epi16(q4_0, q8v);
                    int32_t dot0 = hsum_epi32_avx2(_mm256_madd_epi16(pdot0, ones16));
                    sum0 += (float)(dot0 - 8 * qsum) * ws0;

                    // Row 1
                    __m128i p1 = _mm_loadu_si128((const __m128i *)(b1 + 2));
                    __m128i lo1 = _mm_and_si128(p1, mask128);
                    __m128i hi1 = _mm_and_si128(_mm_srli_epi16(p1, 4), mask128);
                    __m256i q4_1 = _mm256_castsi128_si256(_mm_unpacklo_epi8(lo1, hi1));
                    q4_1 = _mm256_inserti128_si256(q4_1, _mm_unpackhi_epi8(lo1, hi1), 1);
                    __m256i pdot1 = _mm256_maddubs_epi16(q4_1, q8v);
                    int32_t dot1 = hsum_epi32_avx2(_mm256_madd_epi16(pdot1, ones16));
                    sum1 += (float)(dot1 - 8 * qsum) * ws1;
                    
                    // Row 2
                    __m128i p2 = _mm_loadu_si128((const __m128i *)(b2 + 2));
                    __m128i lo2 = _mm_and_si128(p2, mask128);
                    __m128i hi2 = _mm_and_si128(_mm_srli_epi16(p2, 4), mask128);
                    __m256i q4_2 = _mm256_castsi128_si256(_mm_unpacklo_epi8(lo2, hi2));
                    q4_2 = _mm256_inserti128_si256(q4_2, _mm_unpackhi_epi8(lo2, hi2), 1);
                    __m256i pdot2 = _mm256_maddubs_epi16(q4_2, q8v);
                    int32_t dot2 = hsum_epi32_avx2(_mm256_madd_epi16(pdot2, ones16));
                    sum2 += (float)(dot2 - 8 * qsum) * ws2;

                    // Row 3
                    __m128i p3 = _mm_loadu_si128((const __m128i *)(b3 + 2));
                    __m128i lo3 = _mm_and_si128(p3, mask128);
                    __m128i hi3 = _mm_and_si128(_mm_srli_epi16(p3, 4), mask128);
                    __m256i q4_3 = _mm256_castsi128_si256(_mm_unpacklo_epi8(lo3, hi3));
                    q4_3 = _mm256_inserti128_si256(q4_3, _mm_unpackhi_epi8(lo3, hi3), 1);
                    __m256i pdot3 = _mm256_maddubs_epi16(q4_3, q8v);
                    int32_t dot3 = hsum_epi32_avx2(_mm256_madd_epi16(pdot3, ones16));
                    sum3 += (float)(dot3 - 8 * qsum) * ws3;
                } else
#endif
                {
                    sum0 += (float)dot_q4_q8_block(b0 + 2, q8 + b * 32, valid) * ws0;
                    sum1 += (float)dot_q4_q8_block(b1 + 2, q8 + b * 32, valid) * ws1;
                    sum2 += (float)dot_q4_q8_block(b2 + 2, q8 + b * 32, valid) * ws2;
                    sum3 += (float)dot_q4_q8_block(b3 + 2, q8 + b * 32, valid) * ws3;
                }
            }
            y[(size_t)t * N + n + 0] = sum0;
            y[(size_t)t * N + n + 1] = sum1;
            y[(size_t)t * N + n + 2] = sum2;
            y[(size_t)t * N + n + 3] = sum3;
        }
        // Cleanup loop
        for (; n < N; n++) {
            const uint8_t *row = raw + (uint64_t)n * row_bytes;
            float sum = 0.0f;
            for (int b = 0; b < blocks; b++) {
                const uint8_t *block = row + (size_t)b * 18;
                uint16_t hs; memcpy(&hs, block, sizeof(hs));
                float ws = half_to_float(hs) * x_scale;
                int valid = K - b * 32;
                if (valid > 32) valid = 32;
                sum += (float)dot_q4_q8_block(block + 2, q8 + b * 32, valid) * ws;
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
    if (t->dtype == QWN_DT_VSQ) {
        int blocks = (width + 63) / 64;
        const uint8_t *p = raw + (size_t)row * blocks * 36;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = p + b * 36;
            uint16_t hs; memcpy(&hs, blk, 2);
            float base = half_to_float(hs);
            float s0 = base * ((float)blk[2] * (1.0f / 128.0f));
            float s1 = base * ((float)blk[3] * (1.0f / 128.0f));
            const uint8_t *qs = blk + 4;
            for (int i = 0; i < 32 && b * 64 + i < width; i++) {
                uint8_t byte = qs[i >> 1];
                int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                out[b * 64 + i] = q * s0;
            }
            for (int i = 0; i < 32 && b * 64 + 32 + i < width; i++) {
                uint8_t byte = qs[16 + (i >> 1)];
                int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                out[b * 64 + 32 + i] = q * s1;
            }
        }
        return 0;
    }
    if (t->dtype == QWN_DT_VSQ_ULTRA) {
        int blocks = (width + 127) / 128;
        const uint8_t *p = raw + (size_t)row * blocks * 70;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = p + b * 70;
            uint16_t hs, hm;
            memcpy(&hs, blk, 2);
            memcpy(&hm, blk + 2, 2);
            float base = half_to_float(hs);
            float offset = half_to_float(hm);
            uint8_t sub_byte0 = blk[4];
            uint8_t sub_byte1 = blk[5];
            float s[4];
            s[0] = base * ((float)(sub_byte0 & 0x0F) * (1.0f / 8.0f));
            s[1] = base * ((float)(sub_byte0 >> 4)   * (1.0f / 8.0f));
            s[2] = base * ((float)(sub_byte1 & 0x0F) * (1.0f / 8.0f));
            s[3] = base * ((float)(sub_byte1 >> 4)   * (1.0f / 8.0f));
            const uint8_t *qs = blk + 6;
            for (int quad = 0; quad < 4; quad++) {
                const uint8_t *q_quad = qs + quad * 16;
                float sq = s[quad];
                int base_idx = b * 128 + quad * 32;
                for (int i = 0; i < 32 && base_idx + i < width; i++) {
                    uint8_t byte = q_quad[i >> 1];
                    int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                    out[base_idx + i] = q * sq + offset;
                }
            }
        }
        return 0;
    }
    if (t->dtype == QWN_DT_HYPER_VSQ) {
        int blocks = (width + 255) / 256;
        const uint8_t *p = raw + (size_t)row * blocks * 138;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = p + b * 138;
            uint16_t hs, hm;
            memcpy(&hs, blk, 2);
            memcpy(&hm, blk + 2, 2);
            float base = half_to_float(hs);
            float offset = half_to_float(hm);
            const uint8_t *sub_bytes = blk + 4;
            float s[8];
            for (int oct = 0; oct < 8; oct++) {
                uint8_t sb = sub_bytes[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                s[oct] = base * ((float)s_val * (1.0f / 8.0f));
            }
            const uint8_t *qs = blk + 10;
            for (int oct = 0; oct < 8; oct++) {
                const uint8_t *q_oct = qs + oct * 16;
                float sq = s[oct];
                int base_idx = b * 256 + oct * 32;
                for (int i = 0; i < 32 && base_idx + i < width; i++) {
                    uint8_t byte = q_oct[i >> 1];
                    int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                    out[base_idx + i] = q * sq + offset;
                }
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
        int i = 0;
#if defined(__AVX2__)
        if (t->dtype == QWN_DT_F16) {
#if defined(__F16C__)
            for (; i <= width - 8; i += 8) {
                __m128i h8 = _mm_loadu_si128((const __m128i *)(p + i));
                __m256 f8 = _mm256_cvtph_ps(h8);
                _mm256_storeu_ps(out + i, f8);
            }
#endif
        } else if (t->dtype == QWN_DT_BF16) {
            for (; i <= width - 8; i += 8) {
                __m128i h8 = _mm_loadu_si128((const __m128i *)(p + i));
                __m256i u32 = _mm256_cvtepu16_epi32(h8);
                __m256i f32bits = _mm256_slli_epi32(u32, 16);
                _mm256_storeu_si256((__m256i *)(out + i), f32bits);
            }
        }
#endif
        for (; i < width; i++)
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

    if (w->dtype == QWN_DT_F32) {
        const float *raw = (const float *)qwn_data(m, w);
        if (!raw) return -1;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(N > 16)
#endif
        for (int n = 0; n < N; n++) {
            const float *row_p = raw + (size_t)n * K;
            for (int token = 0; token < M; token++) {
                const float *xr = x + (size_t)token * K;
                float sum = 0.0f;
#if defined(__AVX2__)
                __m256 sum_vec = _mm256_setzero_ps();
                int k = 0;
                for (; k <= K - 8; k += 8) {
                    __m256 vx = _mm256_loadu_ps(xr + k);
                    __m256 vw = _mm256_loadu_ps(row_p + k);
                    sum_vec = _mm256_fmadd_ps(vx, vw, sum_vec);
                }
                float tmp[8]; _mm256_storeu_ps(tmp, sum_vec);
                sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                for (; k < K; k++) sum += xr[k] * row_p[k];
#else
                for (int k = 0; k < K; k++) sum += xr[k] * row_p[k];
#endif
                y[(size_t)token * N + n] = sum;
            }
        }
        return 0;
    }

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
