/*
 * Qwanto Core Engine
 * 
 * Based on the Colibri unified inference runtime (https://github.com/JustVugg/colibri).
 * A unified inference engine that utilizes CPU, GPU, RAM, and NVMe to run models larger than memory.
 */
#include "qwanto_core.h"
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// Helper to sum elements of a __m256i vector of 32-bit integers
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

// Helper to sum elements of a __m512i vector of 32-bit integers
#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline int32_t hsum_epi32_avx512(__m512i v) {
    __m256i lo = _mm512_castsi512_si256(v);
    __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    __m256i sum256 = _mm256_add_epi32(lo, hi);
    return hsum_epi32_avx2(sum256);
}
#endif

void qwanto_matmul_core_avx2(const int8_t* activations, const uint8_t* packed_weights, int32_t* outputs, int m, int n) {
#if defined(__AVX2__)
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);

    // Unroll outer loop by 4
    int i = 0;
    for (; i + 3 <= m; i += 4) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        const uint8_t* w0 = &packed_weights[i * (n / 2)];
        const uint8_t* w1 = &packed_weights[(i + 1) * (n / 2)];
        const uint8_t* w2 = &packed_weights[(i + 2) * (n / 2)];
        const uint8_t* w3 = &packed_weights[(i + 3) * (n / 2)];

        for (int j = 0; j < n; j += 32) {
            // Load 32 activations (32 bytes)
            __m256i act = _mm256_loadu_si256((const __m256i*)&activations[j]);
            __m256i act_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(act));
            __m256i act_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(act, 1));

            // Load and unpack weights for row 0
            __m128i raw_w0 = _mm_loadu_si128((const __m128i*)&w0[j / 2]);
            __m128i lo0 = _mm_and_si128(raw_w0, m4);
            __m128i hi0 = _mm_and_si128(_mm_srli_epi16(raw_w0, 4), m4);
            __m128i u_w0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo0, hi0), b8);
            __m256i w_vec0 = _mm256_insertf128_si256(_mm256_castsi128_si256(u_w0), _mm_sub_epi8(_mm_unpackhi_epi8(lo0, hi0), b8), 1);
            __m256i w0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec0));
            __m256i w0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec0, 1));
            sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(act_lo, w0_lo));
            sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(act_hi, w0_hi));

            // Load and unpack weights for row 1
            __m128i raw_w1 = _mm_loadu_si128((const __m128i*)&w1[j / 2]);
            __m128i lo1 = _mm_and_si128(raw_w1, m4);
            __m128i hi1 = _mm_and_si128(_mm_srli_epi16(raw_w1, 4), m4);
            __m128i u_w1 = _mm_sub_epi8(_mm_unpacklo_epi8(lo1, hi1), b8);
            __m256i w_vec1 = _mm256_insertf128_si256(_mm256_castsi128_si256(u_w1), _mm_sub_epi8(_mm_unpackhi_epi8(lo1, hi1), b8), 1);
            __m256i w1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec1));
            __m256i w1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec1, 1));
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(act_lo, w1_lo));
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(act_hi, w1_hi));

            // Load and unpack weights for row 2
            __m128i raw_w2 = _mm_loadu_si128((const __m128i*)&w2[j / 2]);
            __m128i lo2 = _mm_and_si128(raw_w2, m4);
            __m128i hi2 = _mm_and_si128(_mm_srli_epi16(raw_w2, 4), m4);
            __m128i u_w2 = _mm_sub_epi8(_mm_unpacklo_epi8(lo2, hi2), b8);
            __m256i w_vec2 = _mm256_insertf128_si256(_mm256_castsi128_si256(u_w2), _mm_sub_epi8(_mm_unpackhi_epi8(lo2, hi2), b8), 1);
            __m256i w2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec2));
            __m256i w2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec2, 1));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(act_lo, w2_lo));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(act_hi, w2_hi));

            // Load and unpack weights for row 3
            __m128i raw_w3 = _mm_loadu_si128((const __m128i*)&w3[j / 2]);
            __m128i lo3 = _mm_and_si128(raw_w3, m4);
            __m128i hi3 = _mm_and_si128(_mm_srli_epi16(raw_w3, 4), m4);
            __m128i u_w3 = _mm_sub_epi8(_mm_unpacklo_epi8(lo3, hi3), b8);
            __m256i w_vec3 = _mm256_insertf128_si256(_mm256_castsi128_si256(u_w3), _mm_sub_epi8(_mm_unpackhi_epi8(lo3, hi3), b8), 1);
            __m256i w3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec3));
            __m256i w3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec3, 1));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(act_lo, w3_lo));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(act_hi, w3_hi));
        }

        outputs[i]     += hsum_epi32_avx2(sum0);
        outputs[i + 1] += hsum_epi32_avx2(sum1);
        outputs[i + 2] += hsum_epi32_avx2(sum2);
        outputs[i + 3] += hsum_epi32_avx2(sum3);
    }

    // Cleanup loop for rows
    for (; i < m; i++) {
        __m256i sum = _mm256_setzero_si256();
        const uint8_t* w = &packed_weights[i * (n / 2)];

        for (int j = 0; j < n; j += 32) {
            __m256i act = _mm256_loadu_si256((const __m256i*)&activations[j]);
            __m256i act_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(act));
            __m256i act_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(act, 1));

            __m128i raw_w = _mm_loadu_si128((const __m128i*)&w[j / 2]);
            __m128i lo = _mm_and_si128(raw_w, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw_w, 4), m4);
            __m128i u_w = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), b8);
            __m256i w_vec = _mm256_insertf128_si256(_mm256_castsi128_si256(u_w), _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), b8), 1);

            __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec));
            __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1));

            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(act_lo, w_lo));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(act_hi, w_hi));
        }
        outputs[i] += hsum_epi32_avx2(sum);
    }
#else
    // Fallback scalar path
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        const uint8_t* w = &packed_weights[i * (n / 2)];
        for (int j = 0; j < n; j++) {
            uint8_t byte = w[j / 2];
            int8_t val = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            sum += (int32_t)activations[j] * (int32_t)(val - 8);
        }
        outputs[i] += sum;
    }
#endif
}

void qwanto_matmul_core_avx512(const int8_t* activations, const uint8_t* packed_weights, int32_t* outputs, int m, int n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i b8_256 = _mm256_set1_epi8(8);

    int i = 0;
    // Unroll by 4
    for (; i + 3 <= m; i += 4) {
        __m512i sum0 = _mm512_setzero_si512();
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();

        const uint8_t* w0 = &packed_weights[i * (n / 2)];
        const uint8_t* w1 = &packed_weights[(i + 1) * (n / 2)];
        const uint8_t* w2 = &packed_weights[(i + 2) * (n / 2)];
        const uint8_t* w3 = &packed_weights[(i + 3) * (n / 2)];

        for (int j = 0; j < n; j += 64) {
            // Load 64 activations
            __m512i act = _mm512_loadu_si512((const __m512i*)&activations[j]);
            __m512i act_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(act));
            __m512i act_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(act, 1));

            // Row 0
            __m128i raw_w0_0 = _mm_loadu_si128((const __m128i*)&w0[j / 2]);
            __m128i lo0_0 = _mm_and_si128(raw_w0_0, m4);
            __m128i hi0_0 = _mm_and_si128(_mm_srli_epi16(raw_w0_0, 4), m4);
            __m256i w_vec0 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo0_0, hi0_0)), _mm_unpackhi_epi8(lo0_0, hi0_0), 1);
            
            __m128i raw_w0_1 = _mm_loadu_si128((const __m128i*)&w0[j / 2 + 16]);
            __m128i lo0_1 = _mm_and_si128(raw_w0_1, m4);
            __m128i hi0_1 = _mm_and_si128(_mm_srli_epi16(raw_w0_1, 4), m4);
            __m256i w_vec0_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo0_1, hi0_1)), _mm_unpackhi_epi8(lo0_1, hi0_1), 1);
            
            __m256i w_v0_final = _mm256_sub_epi8(w_vec0, b8_256);
            __m256i w_v0_final_1 = _mm256_sub_epi8(w_vec0_1, b8_256);
            __m512i w512_0 = _mm512_inserti64x4(_mm512_castsi256_si512(w_v0_final), w_v0_final_1, 1);
            __m512i w0_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(w512_0));
            __m512i w0_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(w512_0, 1));
            sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(act_lo, w0_lo));
            sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(act_hi, w0_hi));

            // Row 1
            __m128i raw_w1_0 = _mm_loadu_si128((const __m128i*)&w1[j / 2]);
            __m128i lo1_0 = _mm_and_si128(raw_w1_0, m4);
            __m128i hi1_0 = _mm_and_si128(_mm_srli_epi16(raw_w1_0, 4), m4);
            __m256i w_vec1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo1_0, hi1_0)), _mm_unpackhi_epi8(lo1_0, hi1_0), 1);
            
            __m128i raw_w1_1 = _mm_loadu_si128((const __m128i*)&w1[j / 2 + 16]);
            __m128i lo1_1 = _mm_and_si128(raw_w1_1, m4);
            __m128i hi1_1 = _mm_and_si128(_mm_srli_epi16(raw_w1_1, 4), m4);
            __m256i w_vec1_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo1_1, hi1_1)), _mm_unpackhi_epi8(lo1_1, hi1_1), 1);
            
            __m256i w_v1_final = _mm256_sub_epi8(w_vec1, b8_256);
            __m256i w_v1_final_1 = _mm256_sub_epi8(w_vec1_1, b8_256);
            __m512i w512_1 = _mm512_inserti64x4(_mm512_castsi256_si512(w_v1_final), w_v1_final_1, 1);
            __m512i w1_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(w512_1));
            __m512i w1_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(w512_1, 1));
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(act_lo, w1_lo));
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(act_hi, w1_hi));

            // Row 2
            __m128i raw_w2_0 = _mm_loadu_si128((const __m128i*)&w2[j / 2]);
            __m128i lo2_0 = _mm_and_si128(raw_w2_0, m4);
            __m128i hi2_0 = _mm_and_si128(_mm_srli_epi16(raw_w2_0, 4), m4);
            __m256i w_vec2 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo2_0, hi2_0)), _mm_unpackhi_epi8(lo2_0, hi2_0), 1);
            
            __m128i raw_w2_1 = _mm_loadu_si128((const __m128i*)&w2[j / 2 + 16]);
            __m128i lo2_1 = _mm_and_si128(raw_w2_1, m4);
            __m128i hi2_1 = _mm_and_si128(_mm_srli_epi16(raw_w2_1, 4), m4);
            __m256i w_vec2_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo2_1, hi2_1)), _mm_unpackhi_epi8(lo2_1, hi2_1), 1);
            
            __m256i w_v2_final = _mm256_sub_epi8(w_vec2, b8_256);
            __m256i w_v2_final_1 = _mm256_sub_epi8(w_vec2_1, b8_256);
            __m512i w512_2 = _mm512_inserti64x4(_mm512_castsi256_si512(w_v2_final), w_v2_final_1, 1);
            __m512i w2_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(w512_2));
            __m512i w2_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(w512_2, 1));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(act_lo, w2_lo));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(act_hi, w2_hi));

            // Row 3
            __m128i raw_w3_0 = _mm_loadu_si128((const __m128i*)&w3[j / 2]);
            __m128i lo3_0 = _mm_and_si128(raw_w3_0, m4);
            __m128i hi3_0 = _mm_and_si128(_mm_srli_epi16(raw_w3_0, 4), m4);
            __m256i w_vec3 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo3_0, hi3_0)), _mm_unpackhi_epi8(lo3_0, hi3_0), 1);
            
            __m128i raw_w3_1 = _mm_loadu_si128((const __m128i*)&w3[j / 2 + 16]);
            __m128i lo3_1 = _mm_and_si128(raw_w3_1, m4);
            __m128i hi3_1 = _mm_and_si128(_mm_srli_epi16(raw_w3_1, 4), m4);
            __m256i w_vec3_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo3_1, hi3_1)), _mm_unpackhi_epi8(lo3_1, hi3_1), 1);
            
            __m256i w_v3_final = _mm256_sub_epi8(w_vec3, b8_256);
            __m256i w_v3_final_1 = _mm256_sub_epi8(w_vec3_1, b8_256);
            __m512i w512_3 = _mm512_inserti64x4(_mm512_castsi256_si512(w_v3_final), w_v3_final_1, 1);
            __m512i w3_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(w512_3));
            __m512i w3_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(w512_3, 1));
            sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(act_lo, w3_lo));
            sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(act_hi, w3_hi));
        }

        outputs[i]     += hsum_epi32_avx512(sum0);
        outputs[i + 1] += hsum_epi32_avx512(sum1);
        outputs[i + 2] += hsum_epi32_avx512(sum2);
        outputs[i + 3] += hsum_epi32_avx512(sum3);
    }

    // Cleanup loop for rows
    for (; i < m; i++) {
        __m512i sum = _mm512_setzero_si512();
        const uint8_t* w = &packed_weights[i * (n / 2)];

        for (int j = 0; j < n; j += 64) {
            __m512i act = _mm512_loadu_si512((const __m512i*)&activations[j]);
            __m512i act_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(act));
            __m512i act_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(act, 1));

            __m128i raw_w_0 = _mm_loadu_si128((const __m128i*)&w[j / 2]);
            __m128i lo_0 = _mm_and_si128(raw_w_0, m4);
            __m128i hi_0 = _mm_and_si128(_mm_srli_epi16(raw_w_0, 4), m4);
            __m256i w_vec = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo_0, hi_0)), _mm_unpackhi_epi8(lo_0, hi_0), 1);
            
            __m128i raw_w_1 = _mm_loadu_si128((const __m128i*)&w[j / 2 + 16]);
            __m128i lo_1 = _mm_and_si128(raw_w_1, m4);
            __m128i hi_1 = _mm_and_si128(_mm_srli_epi16(raw_w_1, 4), m4);
            __m256i w_vec_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo_1, hi_1)), _mm_unpackhi_epi8(lo_1, hi_1), 1);
            
            __m256i w_v_final = _mm256_sub_epi8(w_vec, b8_256);
            __m256i w_v_final_1 = _mm256_sub_epi8(w_vec_1, b8_256);
            __m512i w512 = _mm512_inserti64x4(_mm512_castsi256_si512(w_v_final), w_v_final_1, 1);
            __m512i w_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(w512));
            __m512i w_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(w512, 1));
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(act_lo, w_lo));
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(act_hi, w_hi));
        }
        outputs[i] += hsum_epi32_avx512(sum);
    }
#else
    // Fallback to AVX2 if AVX-512 compile target is not set
    qwanto_matmul_core_avx2(activations, packed_weights, outputs, m, n);
#endif
}

#if defined(_OPENMP)
#include <omp.h>
#endif

void qwanto_matmul_blocked(const int8_t* activations, const uint8_t* packed_weights, int32_t* outputs, 
                           int m_tokens, int n_out, int k_in) {
    // Clear outputs initially
    memset(outputs, 0, m_tokens * n_out * sizeof(int32_t));

    // Choose optimization path
    int use_avx512 = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    use_avx512 = 1;
#endif

    // Cache blocking boundaries
    const int BLOCK_K = 1024;
    const int BLOCK_M = 256;

    for (int kk = 0; kk < k_in; kk += BLOCK_K) {
        int k_len = (kk + BLOCK_K > k_in) ? (k_in - kk) : BLOCK_K;
        
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2) schedule(static) if(n_out > 32)
#endif
        for (int ii = 0; ii < n_out; ii += BLOCK_M) {
            for (int t = 0; t < m_tokens; t++) {
                int m_len = (ii + BLOCK_M > n_out) ? (n_out - ii) : BLOCK_M;
                const int8_t* act_ptr = &activations[t * k_in + kk];
                const uint8_t* w_ptr = &packed_weights[ii * (k_in / 2) + (kk / 2)];
                int32_t* out_ptr = &outputs[t * n_out + ii];

                if (use_avx512 && k_len % 64 == 0) {
                    qwanto_matmul_core_avx512(act_ptr, w_ptr, out_ptr, m_len, k_len);
                } else if (k_len % 32 == 0) {
                    qwanto_matmul_core_avx2(act_ptr, w_ptr, out_ptr, m_len, k_len);
                } else {
                    // Raw scalar fallback block if k_len is not a multiple of 32
                    for (int i = 0; i < m_len; i++) {
                        int32_t sum = 0;
                        const uint8_t* w = &w_ptr[i * (k_in / 2)];
                        for (int j = 0; j < k_len; j++) {
                            uint8_t byte = w[j / 2];
                            int8_t val = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                            sum += (int32_t)act_ptr[j] * (int32_t)(val - 8);
                        }
                        out_ptr[i] += sum;
                    }
                }
            }
        }
    }
}
