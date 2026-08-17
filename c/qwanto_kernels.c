#include "qwanto_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
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

static QwnCpuFeatures g_cpu_features;
static int g_cpu_features_initialized = 0;

int qwn_cpu_avx2_kernel_compiled(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int qwn_cpu_vnni_kernel_compiled(void) {
#if defined(__AVX2__) && (defined(__clang__) || defined(__GNUC__) || defined(__AVX_VNNI__))
    return 1;
#else
    return 0;
#endif
}

static uint64_t qwn_xgetbv0(void) {
#if defined(_MSC_VER)
    return (uint64_t)_xgetbv(0);
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
}

static void qwn_init_cpu_features(void) {
    if (g_cpu_features_initialized) return;
    memset(&g_cpu_features, 0, sizeof(g_cpu_features));

#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    int max_ids = info[0];
    int os_avx = 0;
    int os_avx512 = 0;
    if (max_ids >= 1) {
        __cpuid(info, 1);
        os_avx = (info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0;
        os_avx = os_avx && ((qwn_xgetbv0() & 0x6) == 0x6);
        g_cpu_features.has_f16c = os_avx && (info[2] & (1 << 29)) != 0;
        g_cpu_features.has_fma  = os_avx && (info[2] & (1 << 12)) != 0;
    }
    if (max_ids >= 7) {
        __cpuidex(info, 7, 0);
        g_cpu_features.has_avx2    = os_avx && (info[1] & (1 << 5)) != 0;
        os_avx512 = os_avx && ((qwn_xgetbv0() & 0xE0) == 0xE0);
        g_cpu_features.has_avx512f = os_avx512 && (info[1] & (1 << 16)) != 0;
        if (os_avx512 && (info[2] & (1 << 11))) g_cpu_features.has_vnni = 1; /* AVX512_VNNI */
        __cpuidex(info, 7, 1);
        if (g_cpu_features.has_avx2 && (info[0] & (1 << 4))) g_cpu_features.has_vnni = 1;  /* AVX_VNNI */
    }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    int os_avx = 0;
    int os_avx512 = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        os_avx = (ecx & (1 << 27)) != 0 && (ecx & (1 << 28)) != 0;
        os_avx = os_avx && ((qwn_xgetbv0() & 0x6) == 0x6);
        g_cpu_features.has_f16c = os_avx && (ecx & (1 << 29)) != 0;
        g_cpu_features.has_fma  = os_avx && (ecx & (1 << 12)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        g_cpu_features.has_avx2    = os_avx && (ebx & (1 << 5)) != 0;
        os_avx512 = os_avx && ((qwn_xgetbv0() & 0xE0) == 0xE0);
        g_cpu_features.has_avx512f = os_avx512 && (ebx & (1 << 16)) != 0;
        if (os_avx512 && (ecx & (1 << 11))) g_cpu_features.has_vnni = 1; /* AVX512_VNNI */
    }
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        if (g_cpu_features.has_avx2 && (eax & (1 << 4))) g_cpu_features.has_vnni = 1;  /* AVX_VNNI */
    }
#endif

    const char *force_scalar = getenv("QWN_FORCE_SCALAR");
    const char *force_avx2   = getenv("QWN_FORCE_AVX2");
    const char *force_vnni   = getenv("QWN_FORCE_VNNI");

    if (force_scalar && *force_scalar && strcmp(force_scalar, "0") != 0) {
        g_cpu_features.forced_mode = 1;
    } else if (force_vnni && *force_vnni && strcmp(force_vnni, "0") != 0) {
        if (g_cpu_features.has_vnni) {
            g_cpu_features.forced_mode = 3;
        } else {
            fprintf(stderr, "[WARN] QWN_FORCE_VNNI set but CPU lacks AVX-VNNI support; falling back.\n");
            g_cpu_features.forced_mode = g_cpu_features.has_avx2 ? 2 : 1;
        }
    } else if (force_avx2 && *force_avx2 && strcmp(force_avx2, "0") != 0) {
        if (g_cpu_features.has_avx2) {
            g_cpu_features.forced_mode = 2;
        } else {
            fprintf(stderr, "[WARN] QWN_FORCE_AVX2 set but CPU lacks AVX2 support; falling back to scalar.\n");
            g_cpu_features.forced_mode = 1;
        }
    }

    g_cpu_features_initialized = 1;
}

const QwnCpuFeatures *qwn_get_cpu_features(void) {
    if (!g_cpu_features_initialized) qwn_init_cpu_features();
    return &g_cpu_features;
}

const char *qwn_cpu_kernel_name(void) {
    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    if (cpu->forced_mode == 1) return "scalar-forced";
    if (cpu->forced_mode == 2 && cpu->has_avx2 && qwn_cpu_avx2_kernel_compiled())
        return "avx2-fma-f16c-forced";
    if (cpu->forced_mode == 3 && cpu->has_vnni && qwn_cpu_vnni_kernel_compiled())
        return "vnni-forced";
    if (cpu->has_vnni && qwn_cpu_vnni_kernel_compiled()) return "vnni";
    if (cpu->has_avx2 && qwn_cpu_avx2_kernel_compiled() && cpu->has_fma && cpu->has_f16c)
        return "avx2-fma-f16c";
    if (cpu->has_avx2 && qwn_cpu_avx2_kernel_compiled()) return "avx2";
    return "scalar";
}

int qwn_select_cpu_kernel(const char *kernel, char *error, size_t error_size) {
    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    if (!kernel || strcmp(kernel, "auto") == 0) return 0;
    if (strcmp(kernel, "scalar") == 0) {
        g_cpu_features.forced_mode = 1;
        return 0;
    }
    if (strcmp(kernel, "avx2") == 0) {
        if (!cpu->has_avx2 || !qwn_cpu_avx2_kernel_compiled()) {
            if (error && error_size) snprintf(error, error_size,
                "AVX2 kernel requested but CPU support or compiled AVX2 code is unavailable");
            return -1;
        }
        g_cpu_features.forced_mode = 2;
        return 0;
    }
    if (strcmp(kernel, "vnni") == 0) {
        if (!cpu->has_vnni || !qwn_cpu_vnni_kernel_compiled()) {
            if (error && error_size) snprintf(error, error_size,
                "VNNI kernel requested but CPU support or compiled AVX2/VNNI code is unavailable");
            return -1;
        }
        g_cpu_features.forced_mode = 3;
        return 0;
    }
    if (error && error_size) snprintf(error, error_size, "unsupported CPU kernel selection: %s", kernel);
    return -1;
}

int qwn_scratch_init(QwnScratch *s, int max_tokens, int max_k) {
    if (!s || max_tokens < 1 || max_k < 1) return -1;
    memset(s, 0, sizeof(*s));
    int padded_k = (max_k + 255) & ~255;
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
        return (float)value * base * ((float)sub_scale / 8.0f) + offset;
    }
    const uint8_t *q = p + (dtype == QWN_DT_VSQ_ULTRA ? 6 : 10) +
                       sub * 16 +
                       (local % 32) / 2;
    int value = ((local & 1) ? (*q >> 4) : (*q & 0x0F)) - 8;
    return (float)value * base * ((float)sub_scale / 8.0f) + offset;
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

/* Unpack 8 packed 2-bit bytes (32 quaternary codes) in-register into
 * a __m256i containing 32 unsigned uint8 values in [0..3].
 * Layout: byte 0 has weights 0..3, byte 1 has weights 4..7, ..., byte 7 has weights 28..31.
 */
static inline __m256i unpack_32x2bit_avx2(const uint8_t *qs) {
    int64_t raw8;
    memcpy(&raw8, qs, 8);
    int32_t lo4 = (int32_t)raw8;
    int32_t hi4 = (int32_t)(raw8 >> 32);
    __m256i v = _mm256_set_epi32(hi4, hi4, hi4, hi4, lo4, lo4, lo4, lo4);
    __m256i shuf_mask = _mm256_setr_epi8(
        0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3,
        0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3
    );
    __m256i rep = _mm256_shuffle_epi8(v, shuf_mask);
    __m256i w0 = _mm256_and_si256(rep, _mm256_set1_epi32(0x00000003));
    __m256i w1 = _mm256_and_si256(_mm256_srli_epi32(rep, 2), _mm256_set1_epi32(0x00000300));
    __m256i w2 = _mm256_and_si256(_mm256_srli_epi32(rep, 4), _mm256_set1_epi32(0x00030000));
    __m256i w3 = _mm256_and_si256(_mm256_srli_epi32(rep, 6), _mm256_set1_epi32(0x03000000));
    return _mm256_or_si256(_mm256_or_si256(w0, w1), _mm256_or_si256(w2, w3));
}
#endif

static float dot_q4_q8_block(const uint8_t *blk, const int8_t *q8,
                              int valid) {
    uint16_t hs; memcpy(&hs, blk, 2);
    float scale = half_to_float(hs);
    const uint8_t *packed = blk + 2;
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
        int32_t raw_sum = hsum_epi32_avx2(dot32) - 8 * hsum_epi32_avx2(sum32);
        return (float)raw_sum * scale;
    }
#endif
    int32_t sum = 0;
    for (int i = 0; i < valid; i++) {
        uint8_t byte = packed[i >> 1];
        int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
        sum += w * (int32_t)q8[i];
    }
    return (float)sum * scale;
}

static float dot_vsq_block(const uint8_t *blk, const int8_t *q8, int valid) {
    uint16_t hs; memcpy(&hs, blk, 2);
    float base = half_to_float(hs);
    float s0 = base * ((float)blk[2] * (1.0f / 128.0f));
    float s1 = base * ((float)blk[3] * (1.0f / 128.0f));
    const uint8_t *qs = blk + 4;
    float sum = 0.0f;
    int half0 = valid < 32 ? valid : 32;
    for (int i = 0; i < half0; i++) {
        uint8_t byte = qs[i >> 1];
        int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
        sum += (float)w * s0 * (float)q8[i];
    }
    int half1 = valid - 32; if (half1 > 32) half1 = 32;
    for (int i = 0; i < half1; i++) {
        uint8_t byte = qs[16 + (i >> 1)];
        int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
        sum += (float)w * s1 * (float)q8[32 + i];
    }
    return sum;
}

static float dot_vsq_ultra_block(const uint8_t *blk, const int8_t *q8,
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
    float sum = 0.0f;
    for (int quad = 0; quad < 4; quad++) {
        const uint8_t *q_quad = qs + quad * 16;
        float sq = s[quad];
        int base_idx = quad * 32;
        int cap = valid - base_idx; if (cap > 32) cap = 32;
        for (int i = 0; i < cap; i++) {
            uint8_t byte = q_quad[i >> 1];
            int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
            sum += ((float)w * sq + offset) * (float)q8[base_idx + i];
        }
    }
    return sum;
}

static float dot_hyper_vsq_block(const uint8_t *blk, const int8_t *q8,
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
    float sum = 0.0f;
    for (int oct = 0; oct < 8; oct++) {
        const uint8_t *q_oct = qs + oct * 16;
        float sq = s[oct];
        int base_idx = oct * 32;
        int cap = valid - base_idx; if (cap > 32) cap = 32;
        for (int i = 0; i < cap; i++) {
            uint8_t byte = q_oct[i >> 1];
            int32_t w = ((i & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
            sum += ((float)w * sq + offset) * (float)q8[base_idx + i];
        }
    }
    return sum;
}

/* Scalar Golden Reference GEMV for HyperVSQ-2 */
void qwn_gemv_hypervsq2_scalar(const uint8_t *raw_blocks, const int8_t *q8,
                              float x_scale, int K, int N,
                              size_t row_bytes, float *out) {
    int blocks = (K + 255) / 256;
    for (int n = 0; n < N; n++) {
        const uint8_t *row = raw_blocks + (size_t)n * row_bytes;
        float row_sum = 0.0f;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = row + (size_t)b * 74;
            int valid = K - b * 256;
            if (valid > 256) valid = 256;
            if (valid <= 0) break;

            uint16_t hs, hm;
            memcpy(&hs, blk, 2);
            memcpy(&hm, blk + 2, 2);
            float base_scale = half_to_float(hs);
            float offset     = half_to_float(hm);
            const uint8_t *sub_bytes = blk + 4;
            const uint8_t *qs = blk + 10;

            for (int oct = 0; oct < 8; oct++) {
                int base_idx = b * 256 + oct * 32;
                int cap = valid - oct * 32;
                if (cap > 32) cap = 32;
                if (cap <= 0) break;

                uint8_t sb = sub_bytes[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float eff_scale = base_scale * ((float)s_val * (1.0f / 8.0f));
                const uint8_t *q_oct = qs + oct * 8;

                int32_t sum_q = 0;
                int32_t sum_a = 0;
                for (int i = 0; i < cap; i++) {
                    uint8_t byte = q_oct[i >> 2];
                    int shift = (i & 3) * 2;
                    int q = (byte >> shift) & 3;
                    int8_t a = q8[base_idx + i];
                    sum_q += (q - 1) * (int32_t)a;
                    sum_a += (int32_t)a;
                }
                row_sum += (float)sum_q * (eff_scale * x_scale) + (float)sum_a * (offset * x_scale);
            }
        }
        out[n] = row_sum;
    }
}

/* AVX2 Accelerated HyperVSQ-2 GEMV */
void qwn_gemv_hypervsq2_avx2(const uint8_t *raw_blocks, const int8_t *q8,
                            float x_scale, int K, int N,
                            size_t row_bytes, float *out) {
#if defined(__AVX2__)
    int blocks = (K + 255) / 256;
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8  = _mm256_set1_epi8(1);

    for (int n = 0; n < N; n++) {
        const uint8_t *row = raw_blocks + (size_t)n * row_bytes;
        float row_sum = 0.0f;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = row + (size_t)b * 74;
            int valid = K - b * 256;
            if (valid > 256) valid = 256;
            if (valid <= 0) break;

            uint16_t hs, hm;
            memcpy(&hs, blk, 2);
            memcpy(&hm, blk + 2, 2);
            float base_scale = half_to_float(hs);
            float offset     = half_to_float(hm);
            const uint8_t *sub_bytes = blk + 4;
            const uint8_t *qs = blk + 10;
            int cap_oct = valid / 32;
            if (cap_oct > 8) cap_oct = 8;

            for (int oct = 0; oct < cap_oct; oct++) {
                int base_idx = b * 256 + oct * 32;
                uint8_t sb = sub_bytes[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float eff_scale = base_scale * ((float)s_val * (1.0f / 8.0f));

                __m256i q_unp = unpack_32x2bit_avx2(qs + oct * 8);
                __m256i a_vec = _mm256_loadu_si256((const __m256i *)(q8 + base_idx));

                __m256i p16 = _mm256_maddubs_epi16(q_unp, a_vec);
                __m256i dot32 = _mm256_madd_epi16(p16, ones16);

                __m256i sa16 = _mm256_maddubs_epi16(ones8, a_vec);
                __m256i sum_a32 = _mm256_madd_epi16(sa16, ones16);

                __m256i diff32 = _mm256_sub_epi32(dot32, sum_a32);

                int32_t dot_centered = hsum_epi32_avx2(diff32);
                int32_t sum_a = hsum_epi32_avx2(sum_a32);

                row_sum += (float)dot_centered * (eff_scale * x_scale) + (float)sum_a * (offset * x_scale);
            }

            for (int oct = cap_oct; oct < 8; oct++) {
                int base_idx = b * 256 + oct * 32;
                int cap = valid - oct * 32;
                if (cap > 32) cap = 32;
                if (cap <= 0) break;

                uint8_t sb = sub_bytes[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float eff_scale = base_scale * ((float)s_val * (1.0f / 8.0f));
                const uint8_t *q_oct = qs + oct * 8;

                int32_t sum_q = 0;
                int32_t sum_a = 0;
                for (int i = 0; i < cap; i++) {
                    uint8_t byte = q_oct[i >> 2];
                    int shift = (i & 3) * 2;
                    int q = (byte >> shift) & 3;
                    int8_t a = q8[base_idx + i];
                    sum_q += (q - 1) * (int32_t)a;
                    sum_a += (int32_t)a;
                }
                row_sum += (float)sum_q * (eff_scale * x_scale) + (float)sum_a * (offset * x_scale);
            }
        }
        out[n] = row_sum;
    }
#else
    qwn_gemv_hypervsq2_scalar(raw_blocks, q8, x_scale, K, N, row_bytes, out);
#endif
}

/* AVX-VNNI Accelerated HyperVSQ-2 GEMV */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
void qwn_gemv_hypervsq2_vnni(const uint8_t *raw_blocks, const int8_t *q8,
                            float x_scale, int K, int N,
                            size_t row_bytes, float *out) {
#if defined(__AVX2__)
    int blocks = (K + 255) / 256;
    const __m256i ones8 = _mm256_set1_epi8(1);

    for (int n = 0; n < N; n++) {
        const uint8_t *row = raw_blocks + (size_t)n * row_bytes;
        float row_sum = 0.0f;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *blk = row + (size_t)b * 74;
            int valid = K - b * 256;
            if (valid > 256) valid = 256;
            if (valid <= 0) break;

            uint16_t hs, hm;
            memcpy(&hs, blk, 2);
            memcpy(&hm, blk + 2, 2);
            float base_scale = half_to_float(hs);
            float offset     = half_to_float(hm);
            const uint8_t *sub_bytes = blk + 4;
            const uint8_t *qs = blk + 10;
            int cap_oct = valid / 32;
            if (cap_oct > 8) cap_oct = 8;

            for (int oct = 0; oct < cap_oct; oct++) {
                int base_idx = b * 256 + oct * 32;
                uint8_t sb = sub_bytes[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float eff_scale = base_scale * ((float)s_val * (1.0f / 8.0f));

                __m256i q_unp = unpack_32x2bit_avx2(qs + oct * 8);
                __m256i a_vec = _mm256_loadu_si256((const __m256i *)(q8 + base_idx));

                __m256i dot32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_unp, a_vec);
                __m256i sum_a32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones8, a_vec);
                __m256i diff32 = _mm256_sub_epi32(dot32, sum_a32);

                int32_t dot_centered = hsum_epi32_avx2(diff32);
                int32_t sum_a = hsum_epi32_avx2(sum_a32);

                row_sum += (float)dot_centered * (eff_scale * x_scale) + (float)sum_a * (offset * x_scale);
            }

            for (int oct = cap_oct; oct < 8; oct++) {
                int base_idx = b * 256 + oct * 32;
                int cap = valid - oct * 32;
                if (cap > 32) cap = 32;
                if (cap <= 0) break;

                uint8_t sb = sub_bytes[oct >> 1];
                int s_val = (oct & 1) ? (sb >> 4) : (sb & 0x0F);
                float eff_scale = base_scale * ((float)s_val * (1.0f / 8.0f));
                const uint8_t *q_oct = qs + oct * 8;

                int32_t sum_q = 0;
                int32_t sum_a = 0;
                for (int i = 0; i < cap; i++) {
                    uint8_t byte = q_oct[i >> 2];
                    int shift = (i & 3) * 2;
                    int q = (byte >> shift) & 3;
                    int8_t a = q8[base_idx + i];
                    sum_q += (q - 1) * (int32_t)a;
                    sum_a += (int32_t)a;
                }
                row_sum += (float)sum_q * (eff_scale * x_scale) + (float)sum_a * (offset * x_scale);
            }
        }
        out[n] = row_sum;
    }
#else
    qwn_gemv_hypervsq2_scalar(raw_blocks, q8, x_scale, K, N, row_bytes, out);
#endif
}

/* Full matrix multiplication for HyperVSQ-2 with runtime CPUID dispatch */
int qwn_matmul_hypervsq2_f32(const QwnModel *m,
                             const QwnTensorDesc *weights,
                             const float *x, int M, int K, int N,
                             QwnScratch *scratch,
                             float *y) {
    if (!m || !weights || !x || !scratch || !y || M < 1 || K < 1 || N < 1)
        return -1;
    if (weights->dtype != QWN_DT_HYPER_VSQ2 || weights->n_dims != 2 ||
        weights->shape[0] != (uint64_t)K || weights->shape[1] != (uint64_t)N)
        return -1;
    if ((weights->byte_offset & 63ULL) != 0 || M > scratch->max_tokens ||
        ((K + 255) & ~255) > scratch->padded_k)
        return -1;

    const int blocks = (K + 255) / 256;
    const size_t row_bytes = (size_t)blocks * 74;
    const uint64_t raw_bytes = (uint64_t)row_bytes * (uint64_t)N;
    if (weights->byte_offset > m->file_size ||
        raw_bytes > m->file_size - weights->byte_offset ||
        raw_bytes > weights->byte_size)
        return -1;

    quantize_tokens(x, M, K, scratch);
    const uint8_t *raw = m->base + weights->byte_offset;

    const QwnCpuFeatures *cpu = qwn_get_cpu_features();
    static int s_logged = 0;
    if (!s_logged) {
        const char *backend = "scalar";
        if (cpu->forced_mode == 1) backend = "forced scalar";
        else if (cpu->forced_mode == 2) backend = "forced avx2";
        else if (cpu->forced_mode == 3) backend = "forced vnni";
        else if (cpu->has_vnni) backend = "avx-vnni";
        else if (cpu->has_avx2) backend = "avx2";
        fprintf(stderr, "[INFO] HyperVSQ-2 kernel selected: %s\n", backend);
        s_logged = 1;
    }

    typedef void (*gemv_fn_t)(const uint8_t *, const int8_t *, float, int, int, size_t, float *);
    gemv_fn_t gemv_fn = qwn_gemv_hypervsq2_scalar;
    if (cpu->forced_mode == 1) {
        gemv_fn = qwn_gemv_hypervsq2_scalar;
    } else if (cpu->forced_mode == 2 && cpu->has_avx2 && qwn_cpu_avx2_kernel_compiled()) {
        gemv_fn = qwn_gemv_hypervsq2_avx2;
    } else if (cpu->forced_mode == 3 && cpu->has_vnni && qwn_cpu_vnni_kernel_compiled()) {
        gemv_fn = qwn_gemv_hypervsq2_vnni;
    } else if (cpu->has_vnni && qwn_cpu_vnni_kernel_compiled()) {
        gemv_fn = qwn_gemv_hypervsq2_vnni;
    } else if (cpu->has_avx2 && qwn_cpu_avx2_kernel_compiled()) {
        gemv_fn = qwn_gemv_hypervsq2_avx2;
    }

    if (scratch->hypervsq2_matmul_calls == 0) {
        snprintf(scratch->hypervsq2_kernel, sizeof(scratch->hypervsq2_kernel),
                 "%s", qwn_cpu_kernel_name());
        if (gemv_fn == qwn_gemv_hypervsq2_vnni) {
            snprintf(scratch->hypervsq2_dispatch_reason,
                     sizeof(scratch->hypervsq2_dispatch_reason),
                     "cpu_vnni=yes;binary_vnni=yes;dtype=hypervsq2-74;selected=vnni");
        } else if (gemv_fn == qwn_gemv_hypervsq2_avx2) {
            snprintf(scratch->hypervsq2_dispatch_reason,
                     sizeof(scratch->hypervsq2_dispatch_reason),
                     "cpu_avx2=yes;binary_avx2=yes;dtype=hypervsq2-74;selected=avx2");
        } else if (cpu->forced_mode == 1) {
            snprintf(scratch->hypervsq2_dispatch_reason,
                     sizeof(scratch->hypervsq2_dispatch_reason),
                     "requested=scalar;selected=scalar");
        } else {
            snprintf(scratch->hypervsq2_dispatch_reason,
                     sizeof(scratch->hypervsq2_dispatch_reason),
                     "cpu_avx2=%s;binary_avx2=%s;cpu_vnni=%s;binary_vnni=%s;selected=scalar",
                     cpu->has_avx2 ? "yes" : "no",
                     qwn_cpu_avx2_kernel_compiled() ? "yes" : "no",
                     cpu->has_vnni ? "yes" : "no",
                     qwn_cpu_vnni_kernel_compiled() ? "yes" : "no");
        }
    }

    for (int t = 0; t < M; t++) {
        const int8_t *q8 = scratch->q8 + (size_t)t * scratch->padded_k;
        float x_scale = scratch->token_scales[t];
        float *yt = y + (size_t)t * N;

        int active_threads = 1;
        int participating_threads = 0;
#if defined(_OPENMP)
        #pragma omp parallel if(N > 16)
        {
            int participated = 0;
            #pragma omp single
            active_threads = omp_get_num_threads();
            #pragma omp for schedule(static)
            for (int n = 0; n < N; n += 64) {
                participated = 1;
                int chunk = N - n;
                if (chunk > 64) chunk = 64;
                gemv_fn(raw + (size_t)n * row_bytes, q8, x_scale, K, chunk, row_bytes, yt + n);
            }
            if (participated) {
                #pragma omp atomic
                participating_threads += 1;
            }
        }
#else
        for (int n = 0; n < N; n += 64) {
            int chunk = N - n;
            if (chunk > 64) chunk = 64;
            gemv_fn(raw + (size_t)n * row_bytes, q8, x_scale, K, chunk, row_bytes, yt + n);
        }
        participating_threads = 1;
#endif
        scratch->hypervsq2_matmul_calls++;
        scratch->hypervsq2_worker_participations += (uint64_t)participating_threads;
        scratch->hypervsq2_last_active_threads = participating_threads;
        if (participating_threads > scratch->hypervsq2_max_active_threads)
            scratch->hypervsq2_max_active_threads = participating_threads;
    }
    return 0;
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

    if ((weights->dtype != QWN_DT_Q4_0 && weights->dtype != QWN_DT_VSQ && weights->dtype != QWN_DT_VSQ_ULTRA && weights->dtype != QWN_DT_HYPER_VSQ) || weights->n_dims != 2 ||
        weights->shape[0] != (uint64_t)K || weights->shape[1] != (uint64_t)N)
        return -1;
    if ((weights->byte_offset & 63ULL) != 0 || M > scratch->max_tokens ||
        ((K + 31) & ~31) > scratch->padded_k)
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

    typedef float (*dot_fn_t)(const uint8_t *, const int8_t *, int);
    dot_fn_t dot_fn = NULL;
    if (weights->dtype == QWN_DT_Q4_0) dot_fn = dot_q4_q8_block;
    else if (weights->dtype == QWN_DT_VSQ) dot_fn = dot_vsq_block;
    else if (weights->dtype == QWN_DT_VSQ_ULTRA) dot_fn = dot_vsq_ultra_block;
    else if (weights->dtype == QWN_DT_HYPER_VSQ) dot_fn = dot_hyper_vsq_block;
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

                float dot0 = dot_fn(b0, q8 + b * block_elems, valid);
                float dot1 = dot_fn(b1, q8 + b * block_elems, valid);
                float dot2 = dot_fn(b2, q8 + b * block_elems, valid);
                float dot3 = dot_fn(b3, q8 + b * block_elems, valid);

                sum0 += dot0 * x_scale;
                sum1 += dot1 * x_scale;
                sum2 += dot2 * x_scale;
                sum3 += dot3 * x_scale;
            }
            y[(size_t)t * N + n + 0] = sum0;
            y[(size_t)t * N + n + 1] = sum1;
            y[(size_t)t * N + n + 2] = sum2;
            y[(size_t)t * N + n + 3] = sum3;
        }
        for (; n < N; n++) {
            const uint8_t *row = raw + (uint64_t)n * row_bytes;
            float sum = 0.0f;
            for (int b = 0; b < blocks; b++) {
                int valid = K - b * block_elems;
                if (valid > block_elems) valid = block_elems;
                const uint8_t *blk = row + (size_t)b * block_bytes;
                float d = dot_fn(blk, q8 + b * block_elems, valid);
                sum += d * x_scale;
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
                out[b * 32 + i] = (float)q * scale;
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
                out[b * 64 + i] = (float)q * s0;
            }
            for (int i = 0; i < 32 && b * 64 + 32 + i < width; i++) {
                uint8_t byte = qs[16 + (i >> 1)];
                int q = ((i & 1) ? byte >> 4 : byte & 15) - 8;
                out[b * 64 + 32 + i] = (float)q * s1;
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
                    out[base_idx + i] = (float)q * sq + offset;
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
                    out[base_idx + i] = (float)q * sq + offset;
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
                    out[b * 32 + i] = (float)q * scale;
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
        return qwn_matmul_q4_0_f32(m, w, x, M, K, N, scratch, y);
    if (w->dtype == QWN_DT_HYPER_VSQ2)
        return qwn_matmul_hypervsq2_f32(m, w, x, M, K, N, scratch, y);
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
        w->dtype == QWN_DT_HYPER_VSQ)
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
