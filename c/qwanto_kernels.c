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

static float qwn_packed_value(const uint8_t *raw, uint32_t dtype,
                              size_t row_bytes, int row, int index) {
    int block_elems = dtype == QWN_DT_VSQ ? 64 :
                      dtype == QWN_DT_VSQ_ULTRA ? 128 : 256;
    int block_bytes = dtype == QWN_DT_VSQ ? 36 :
                      dtype == QWN_DT_VSQ_ULTRA ? 70 :
                      dtype == QWN_DT_HYPER_VSQ ? 138 : 74;
    int block = index / block_elems;
    int local = index % block_elems;
    const uint8_t *p = raw + (size_t)row * row_bytes + (size_t)block * block_bytes;
    uint16_t hs, hm;
    memcpy(&hs, p, 2);
    float base = half_to_float(hs);
    if (dtype == QWN_DT_VSQ) {
        int sub = local / 32;
        int q = ((p[4 + sub * 16 + (local % 32) / 2] >> ((local & 1) * 4)) & 0x0F) - 8;
        return q * base * ((float)p[2 + sub] / 128.0f);
    }
    memcpy(&hm, p + 2, 2);
    float offset = half_to_float(hm);
    int sub = local / 32;
    int sub_byte = p[4 + (sub >> 1)];
    int sub_scale = (sub & 1) ? (sub_byte >> 4) : (sub_byte & 0x0F);
    if (dtype == QWN_DT_HYPER_VSQ2) {
        const uint8_t *q = p + 10 + sub * 8 + (local % 32) / 4;
        int value = ((*q >> ((local & 3) * 2)) & 3) - 1;
        return value * base * ((float)sub_scale / 8.0f) + offset;
    }
    const uint8_t *q = p + (dtype == QWN_DT_VSQ_ULTRA ? 6 : 10) +
                       sub * (dtype == QWN_DT_VSQ_ULTRA ? 16 : 16) +
                       (local % 32) / 2;
    int value = ((local & 1) ? (*q >> 4) : (*q & 0x0F)) - 8;
    return value * base * ((float)sub_scale / 8.0f) + offset;
}

static int qwn_matmul_packed_f32(const QwnModel *m, const QwnTensorDesc *w,
                                 const float *x, int M, int K, int N,
                                 float *y) {
    const uint8_t *raw = (const uint8_t *)qwn_data(m, w);
    if (!raw) return -1;
    int block_elems = w->dtype == QWN_DT_VSQ ? 64 :
                      w->dtype == QWN_DT_VSQ_ULTRA ? 128 : 256;
    int block_bytes = w->dtype == QWN_DT_VSQ ? 36 :
                      w->dtype == QWN_DT_VSQ_ULTRA ? 70 :
                      w->dtype == QWN_DT_HYPER_VSQ ? 138 : 74;
    size_t row_bytes = (size_t)((K + block_elems - 1) / block_elems) * block_bytes;
    if (row_bytes * (size_t)N > w->byte_size) return -1;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(N > 16)
#endif
    for (int n = 0; n < N; n++) {
        for (int token = 0; token < M; token++) {
            const float *xr = x + (size_t)token * K;
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += xr[k] * qwn_packed_value(raw, w->dtype, row_bytes, n, k);
            y[(size_t)token * N + n] = sum;
        }
    }
    return 0;
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

/* Unpack 8 packed 2-bit values (8 bytes, 32 weights) into a __m256i of
 * 32 signed bytes in [0..3] (the encoded range).  Each input byte holds
 * 4 weights in its low/high 2-bit pairs; the output lays them out in
 * order so output[0] = weights[0..3] and so on.
 *
 * The trick: AND with 0x03, then shift each nibble to its position
 * (no shift, >>2, >>4, >>6), then interleave 4-byte chunks via two
 * levels of unpacklo_epi8 / unpacklo_epi16.
 */
static inline __m256i unpack_8x4_2bit_avx2(__m128i in8) {
    const __m256i mask03 = _mm256_set1_epi8(0x03);
    __m256i in256 = _mm256_set_m128i(in8, in8);
    __m256i w0 = _mm256_and_si256(in256, mask03);
    __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(in256, 2), mask03);
    __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(in256, 4), mask03);
    __m256i w3 = _mm256_and_si256(_mm256_srli_epi16(in256, 6), mask03);
    __m256i pack01 = _mm256_unpacklo_epi8(w0, w1);
    __m256i pack23 = _mm256_unpacklo_epi8(w2, w3);
    return _mm256_unpacklo_epi16(pack01, pack23);
}

/* Convert packed [0..3] bytes to signed [-1..2] in place using a
 * single signed subtract-by-one.  After this, _mm256_maddubs_epi16
 * treats the bytes as signed (Q4_0-style) and produces the correct
 * int16 partial products.
 */
static inline __m256i signed_minus_one_avx2(__m256i v) {
    return _mm256_sub_epi8(v, _mm256_set1_epi8(1));
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

/* Format-specific dot products.  Each block carries its own scale(s)
 * in the header; the q8 row is sign-symmetric int8 in [-127, 127].
 *
 * NOTE: VSQ / VSQ_ULTRA / HYPER_VSQ / HYPER_VSQ2 currently ship without
 * AVX2/AVX-512 SIMD kernels in this release -- the scalar fallbacks
 * below run correctly on x86-64.  The Q4_0 SIMD path above is
 * unchanged and remains the hot path for production workloads.
 */
static int32_t dot_vsq_block(const uint8_t *blk, const int8_t *q8, int valid) {
    uint16_t hs; memcpy(&hs, blk, 2);
    float base = half_to_float(hs);
    float s0 = base * ((float)blk[2] * (1.0f / 128.0f));
    float s1 = base * ((float)blk[3] * (1.0f / 128.0f));
    const uint8_t *qs = blk + 4;
    int32_t sum = 0;
    /* Half 0 (32 elements) */
    int half0 = valid < 32 ? valid : 32;
    for (int i = 0; i < half0; i++) {
        uint8_t byte = qs[i >> 1];
        int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
        sum += (int32_t)((float)w * s0 * (float)q8[i]);
    }
    /* Half 1 (32 elements) */
    int half1 = valid - 32; if (half1 > 32) half1 = 32;
    for (int i = 0; i < half1; i++) {
        uint8_t byte = qs[16 + (i >> 1)];
        int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
        sum += (int32_t)((float)w * s1 * (float)q8[32 + i]);
    }
    return sum;
}

static int32_t dot_vsq_ultra_block(const uint8_t *blk, const int8_t *q8,
                                   int valid) {
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
    int32_t sum = 0;
    for (int quad = 0; quad < 4; quad++) {
        const uint8_t *q_quad = qs + quad * 16;
        float sq = s[quad];
        int base_idx = quad * 32;
        int cap = valid - base_idx; if (cap > 32) cap = 32;
        for (int i = 0; i < cap; i++) {
            uint8_t byte = q_quad[i >> 1];
            int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
            sum += (int32_t)(((float)w * sq + offset) * (float)q8[base_idx + i]);
        }
    }
    return sum;
}

static int32_t dot_hyper_vsq_block(const uint8_t *blk, const int8_t *q8,
                                   int valid) {
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
    int32_t sum = 0;
    for (int oct = 0; oct < 8; oct++) {
        const uint8_t *q_oct = qs + oct * 16;
        float sq = s[oct];
        int base_idx = oct * 32;
        int cap = valid - base_idx; if (cap > 32) cap = 32;
        for (int i = 0; i < cap; i++) {
            uint8_t byte = q_oct[i >> 1];
            int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
            sum += (int32_t)(((float)w * sq + offset) * (float)q8[base_idx + i]);
        }
    }
    return sum;
}

/* Vectorized HyperVSQ-2 dot product: 256-element block, 74-byte packed.
 *
 * Each block has 8 octants (32 weights each).  Per octant the effective
 * scale is `eff_scale = base_scale * (sub_scale / 8)` and the offset is
 * the global base_offset.  Each weight is reconstructed as
 *      w = (q - 1) * eff_scale + offset            (q in {0,1,2,3})
 *
 * So the per-octant dot product expands to
 *      sum(w * q8) = eff_scale * sum((q-1) * q8) + offset * sum(q8)
 *
 * The 32 packed 2-bit weights fit in 8 bytes which we unpack in-register
 * to 32 int8 values, sign-subtract 1 to get the (q-1) range, then use
 * _mm256_maddubs_epi16 + _mm256_madd_epi16 to compute the int8 dot
 * product (universal AVX2 path).  When AVX-VNNI is available we skip
 * the madd chain and use _mm256_dpbusd_epi32 directly (single-instruction
 * int8 dot product with int32 accumulator).
 */
static int32_t dot_hyper_vsq2_block_simd(const uint8_t *blk, const int8_t *q8,
                                         int valid) {
    const uint16_t hs = *(const uint16_t *)(blk);
    const uint16_t hm = *(const uint16_t *)(blk + 2);
    const float base_scale = half_to_float(hs) * 0.5f;          /* fp16 / 2 */
    const float offset    = half_to_float(hm);
    const uint8_t *sub_bytes = blk + 4;                          /* 4 bytes, 8 nibbles */
    const uint8_t *qs       = blk + 10;                          /* 8 bytes per octant */
    const int cap_oct = valid / 32;                              /* number of full octants */
    int32_t sum = 0;

#if defined(__AVX2__)
    /* Pre-compute the per-octant scale as float, then for each octant
     * process the 32 weights vectorially. */
    for (int oct = 0; oct < cap_oct; oct++) {
        const int sb_idx = oct >> 1;
        const int sub_nibble = (oct & 1) ? 4 : 0;
        const int sub_int = (sub_bytes[sb_idx] >> sub_nibble) & 0x0F;
        const float eff_scale = base_scale * ((float)sub_int / 8.0f);

        /* Load 8 packed bytes for this octant (32 weights). */
        __m128i packed = _mm_loadl_epi64((const __m128i *)(qs + oct * 8));
        __m256i w_vec = signed_minus_one_avx2(unpack_8x4_2bit_avx2(packed));
        __m256i a_vec = _mm256_loadu_si256((const __m256i *)(q8 + oct * 32));

#if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
        /* Hardware VNNI path: single instruction int8 dot product. */
        __m256i dot32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_vec, a_vec);
#else
        /* Universal AVX2 path: int8 -> int16 -> int32. */
        __m256i prod16 = _mm256_maddubs_epi16(w_vec, a_vec);   /* 16 int16 */
        __m256i dot32  = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)); /* 8 int32 */
#endif
        const int32_t dot_w = hsum_epi32_avx2(dot32);

        /* q8 sum contribution (for the offset term).  We add it inside
         * the same YMM by zeroing the weight vector: with weights = 1
         * the dot becomes a sum of q8 in [-127..127].  Cheaper than a
         * separate horizontal sum. */
        __m256i ones8 = _mm256_set1_epi8(1);
        __m256i sum32 = _mm256_maddubs_epi16(ones8, a_vec);
        __m256i sum32_32 = _mm256_madd_epi16(sum32, _mm256_set1_epi16(1));
        const int32_t dot_q8 = hsum_epi32_avx2(sum32_32);

        sum += (int32_t)((float)dot_w * eff_scale + (float)dot_q8 * offset);
    }
#endif

    /* Scalar tail: any remaining octants (when K is not a multiple of
     * 32) and any non-AVX2 fallback. */
    for (int oct = cap_oct; oct < 8; oct++) {
        const int sb_idx = oct >> 1;
        const int sub_nibble = (oct & 1) ? 4 : 0;
        const int sub_int = (sub_bytes[sb_idx] >> sub_nibble) & 0x0F;
        const float eff_scale = base_scale * ((float)sub_int / 8.0f);
        const uint8_t *q_oct = qs + oct * 8;
        const int base_idx = oct * 32;
        int cap = valid - base_idx; if (cap > 32) cap = 32;
        for (int i = 0; i < cap; i++) {
            const uint8_t byte = q_oct[i >> 2];
            const int shift = (i & 3) * 2;
            const int32_t w = ((byte >> shift) & 3) - 1;
            sum += (int32_t)(((float)w * eff_scale + offset) * (float)q8[base_idx + i]);
        }
    }
    return sum;
}

static int32_t dot_hyper_vsq2_block(const uint8_t *blk, const int8_t *q8,
                                    int valid) {
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
    int32_t sum = 0;
    for (int oct = 0; oct < 8; oct++) {
        const uint8_t *q_oct = qs + oct * 8;
        float sq = s[oct];
        int base_idx = oct * 32;
        int cap = valid - base_idx; if (cap > 32) cap = 32;
        for (int i = 0; i < cap; i++) {
            uint8_t byte = q_oct[i >> 2];
            int shift = (i & 3) * 2;
            int32_t w = ((byte >> shift) & 3) - 1;
            sum += (int32_t)(((float)w * sq + offset) * (float)q8[base_idx + i]);
        }
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
    if (weights->dtype == QWN_DT_BYTES && weights->n_dims == 2 &&
        weights->shape[0] == (uint64_t)K && weights->shape[1] == (uint64_t)N) {
        const uint8_t *raw = m->base + weights->byte_offset;
        size_t row_b = (size_t)weights->byte_size / (size_t)N;
        const int is_q4 = (row_b == (size_t)((K + 31) / 32) * 18);
        for (int t = 0; t < M; t++) {
            const float *xt = x + (size_t)t * K;
            float *yt = y + (size_t)t * N;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static) if(N > 128)
#endif
            for (int r = 0; r < N; r++) {
                float sum = 0.0f;
                if (is_q4) {
                    const uint8_t *p = raw + (size_t)r * row_b;
                    int blocks = (K + 31) / 32;
                    for (int b = 0; b < blocks; b++) {
                        uint16_t hs; memcpy(&hs, p + b * 18, 2);
                        float scale = half_to_float(hs);
                        const float *xb = xt + b * 32;
                        const uint8_t *qs = p + b * 18 + 2;
                        for (int i = 0; i < 32 && b * 32 + i < K; i++) {
                            uint8_t byte = qs[i >> 1];
                            int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                            sum += (float)q * scale * xb[i];
                        }
                    }
                } else if (row_b >= (size_t)K * 4) {
                    const float *p = (const float *)(raw + (size_t)r * row_b);
                    int k = 0;
#if defined(__AVX2__)
                    __m256 vsum = _mm256_setzero_ps();
                    for (; k <= K - 8; k += 8) {
                        __m256 vx = _mm256_loadu_ps(xt + k);
                        __m256 vw = _mm256_loadu_ps(p + k);
                        vsum = _mm256_fmadd_ps(vx, vw, vsum);
                    }
                    float tmp[8]; _mm256_storeu_ps(tmp, vsum);
                    sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#endif
                    for (; k < K; k++) sum += xt[k] * p[k];
                } else {
                    const uint16_t *p = (const uint16_t *)(raw + (size_t)r * row_b);
                    for (int k = 0; k < K; k++) {
                        uint32_t b = (uint32_t)p[k] << 16;
                        float f; memcpy(&f, &b, 4);
                        sum += xt[k] * f;
                    }
                }
                yt[r] = sum;
            }
        }
        return 0;
    }

    if ((weights->dtype != QWN_DT_Q4_0 && weights->dtype != QWN_DT_VSQ && weights->dtype != QWN_DT_VSQ_ULTRA && weights->dtype != QWN_DT_HYPER_VSQ && weights->dtype != QWN_DT_HYPER_VSQ2) || weights->n_dims != 2 ||
        weights->shape[0] != (uint64_t)K || weights->shape[1] != (uint64_t)N)
        return -1;
    if ((weights->byte_offset & 63ULL) != 0 || M > scratch->max_tokens ||
        ((K + 255) & ~255) > scratch->padded_k)
        return -1;

    const int is_hyper2 = (weights->dtype == QWN_DT_HYPER_VSQ2);
    const int is_hyper = (weights->dtype == QWN_DT_HYPER_VSQ);
    const int is_vsq_ultra = (weights->dtype == QWN_DT_VSQ_ULTRA);
    const int is_vsq = (weights->dtype == QWN_DT_VSQ);
    const int block_elems = (is_hyper || is_hyper2) ? 256 : (is_vsq_ultra ? 128 : (is_vsq ? 64 : 32));
    const int block_bytes = is_hyper2 ? 74 : (is_hyper ? 138 : (is_vsq_ultra ? 70 : (is_vsq ? 36 : 18)));
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

    /* Dispatch the per-block dot product by dtype.  The Q4_0 path is
     * hot -- keep it SIMD-accelerated.  The other formats use scalar
     * dot functions that are correct and memory-safe; future work
     * can add VNNI SIMD for them. */
    typedef int32_t (*dot_fn_t)(const uint8_t *, const int8_t *, int);
    dot_fn_t dot_fn = NULL;
    int q4_simd = 0;
    if (weights->dtype == QWN_DT_Q4_0) { dot_fn = dot_q4_q8_block; q4_simd = 1; }
    else if (weights->dtype == QWN_DT_VSQ) dot_fn = dot_vsq_block;
    else if (weights->dtype == QWN_DT_VSQ_ULTRA) dot_fn = dot_vsq_ultra_block;
    else if (weights->dtype == QWN_DT_HYPER_VSQ) dot_fn = dot_hyper_vsq_block;
    else if (weights->dtype == QWN_DT_HYPER_VSQ2) {
#if defined(__AVX2__)
        dot_fn = dot_hyper_vsq2_block_simd;
#else
        dot_fn = dot_hyper_vsq2_block;
#endif
    }
    if (!dot_fn) return -1;

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
                int valid = K - b * block_elems;
                if (valid > block_elems) valid = block_elems;
                const uint8_t *b0 = r0 + (size_t)b * block_bytes;
                const uint8_t *b1 = r1 + (size_t)b * block_bytes;
                const uint8_t *b2 = r2 + (size_t)b * block_bytes;
                const uint8_t *b3 = r3 + (size_t)b * block_bytes;

                /* Per-format dot product.  For Q4_0 we use the SIMD
                 * dot function directly; for VSQ/VSQ_ULTRA/HYPER_VSQ/
                 * HYPER_VSQ2 we use the scalar dot with the q8 sum
                 * separately accumulated (their block layouts include
                 * their own scale/offset inside the block header). */
                int32_t dot0 = dot_fn(b0, q8 + b * block_elems, valid);
                int32_t dot1 = dot_fn(b1, q8 + b * block_elems, valid);
                int32_t dot2 = dot_fn(b2, q8 + b * block_elems, valid);
                int32_t dot3 = dot_fn(b3, q8 + b * block_elems, valid);

                /* Apply per-block scale.  Q4_0 dot returns the
                 * zero-centered contribution (-8..7) * sum(q8) which
                 * needs the float scale * x_scale.  The VSQ/Hyper
                 * dot functions already bake the float scale in, so
                 * only the q8 scale (x_scale) remains. */
                if (q4_simd) {
                    sum0 += (float)dot0 * x_scale;
                    sum1 += (float)dot1 * x_scale;
                    sum2 += (float)dot2 * x_scale;
                    sum3 += (float)dot3 * x_scale;
                } else {
                    sum0 += (float)dot0;
                    sum1 += (float)dot1;
                    sum2 += (float)dot2;
                    sum3 += (float)dot3;
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
                int valid = K - b * block_elems;
                if (valid > block_elems) valid = block_elems;
                const uint8_t *blk = row + (size_t)b * block_bytes;
                int32_t d = dot_fn(blk, q8 + b * block_elems, valid);
                if (q4_simd) sum += (float)d * x_scale;
                else sum += (float)d;
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
    if (t->dtype == QWN_DT_Q8_0) {
        int blocks = (width + 31) / 32;
        const uint8_t *p = raw + (size_t)row * blocks * 34;
        for (int b = 0; b < blocks; b++) {
            uint16_t hs;
            memcpy(&hs, p + (size_t)b * 34, 2);
            float scale = half_to_float(hs);
            const int8_t *q = (const int8_t *)(p + (size_t)b * 34 + 2);
            int valid = width - b * 32;
            if (valid > 32) valid = 32;
            for (int i = 0; i < valid; i++)
                out[b * 32 + i] = (float)q[i] * scale;
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
    if (t->dtype == QWN_DT_HYPER_VSQ2) {
        int blocks = (width + 255) / 256;
        const uint8_t *p = raw + (size_t)row * blocks * 74;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = p + b * 74;
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
                const uint8_t *q_oct = qs + oct * 8;
                float sq = s[oct];
                int base_idx = b * 256 + oct * 32;
                for (int i = 0; i < 32 && base_idx + i < width; i++) {
                    uint8_t byte = q_oct[i >> 2];
                    int shift = (i & 3) * 2;
                    int q = ((byte >> shift) & 3) - 1;
                    out[base_idx + i] = (float)q * sq + offset;
                }
            }
        }
        return 0;
    }
    if (t->dtype == QWN_DT_BYTES) {
        size_t total_b = (size_t)t->byte_size;
        size_t rows = t->shape[1] ? (size_t)t->shape[1] : 1;
        size_t row_b = total_b / rows;
        if (row_b >= (size_t)((width + 31) / 32) * 18) {
            int blocks = (width + 31) / 32;
            const uint8_t *p = raw + (size_t)row * row_b;
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
        } else if (row_b >= (size_t)width * 2) {
            const uint16_t *p = (const uint16_t *)(raw + (size_t)row * row_b);
            for (int i = 0; i < width; i++) out[i] = half_to_float(p[i]);
            return 0;
        }
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
    if (w->dtype == QWN_DT_Q8_0) {
        const uint8_t *raw = (const uint8_t *)qwn_data(m, w);
        if (!raw) return -1;
        int blocks = (K + 31) / 32;
        size_t row_bytes = (size_t)blocks * 34;
        if (row_bytes * (size_t)N > w->byte_size) return -1;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(N > 16)
#endif
        for (int n = 0; n < N; n++) {
            const uint8_t *row = raw + (size_t)n * row_bytes;
            for (int token = 0; token < M; token++) {
                const float *xr = x + (size_t)token * K;
                float sum = 0.0f;
                for (int b = 0; b < blocks; b++) {
                    uint16_t hs;
                    memcpy(&hs, row + (size_t)b * 34, 2);
                    float scale = half_to_float(hs);
                    const int8_t *q = (const int8_t *)(row + (size_t)b * 34 + 2);
                    int valid = K - b * 32;
                    if (valid > 32) valid = 32;
                    for (int k = 0; k < valid; k++)
                        sum += xr[b * 32 + k] * (float)q[k] * scale;
                }
                y[(size_t)token * N + n] = sum;
            }
        }
        return 0;
    }
    if (w->dtype == QWN_DT_VSQ || w->dtype == QWN_DT_VSQ_ULTRA ||
        w->dtype == QWN_DT_HYPER_VSQ || w->dtype == QWN_DT_HYPER_VSQ2)
        return qwn_matmul_packed_f32(m, w, x, M, K, N, y);
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
