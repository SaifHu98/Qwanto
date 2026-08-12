#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../qwanto_native.h"
#include "../qwanto_kernels.h"

static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) & ~(a - 1); }

static uint16_t f16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t sign = b >> 31, exp = (b >> 23) & 255, mant = b & 0x7fffff;
    if (exp == 255) return (uint16_t)((sign << 15) | 0x7c00 | !!mant);
    int e = (int)exp - 127 + 15;
    if (e <= 0) return (uint16_t)(sign << 15);
    if (e >= 31) return (uint16_t)((sign << 15) | 0x7c00);
    return (uint16_t)((sign << 15) | ((uint32_t)e << 10) | (mant >> 13));
}

static float from_f16(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1, exp = (h >> 10) & 31;
    uint32_t mant = h & 1023, bits;
    if (exp == 0) bits = sign << 31;
    else if (exp == 31) bits = (sign << 31) | 0x7f800000u | (mant << 13);
    else bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v; memcpy(&v, &bits, 4); return v;
}

static void q4_row(const float *x, int k, uint8_t *out) {
    int blocks = (k + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            int col = b * 32 + i;
            float a = col < k ? fabsf(x[col]) : 0.0f;
            if (a > amax) amax = a;
        }
        float scale = amax > 0 ? amax / 7.0f : 1.0f;
        uint16_t hs = f16(scale);
        memcpy(out + b * 18, &hs, 2);
        for (int i = 0; i < 16; i++) {
            int c0 = b * 32 + 2 * i, c1 = c0 + 1;
            int q0 = c0 < k ? (int)lrintf(x[c0] / scale) : 0;
            int q1 = c1 < k ? (int)lrintf(x[c1] / scale) : 0;
            if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
            if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
            out[b * 18 + 2 + i] = (uint8_t)(((q0 + 8) & 15) | (((q1 + 8) & 15) << 4));
        }
    }
}

static int make_model(const char *path) {
    enum { K = 37, N = 3 };
    float weights[N * K];
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            weights[n * K + k] = (float)((n * 5 + k) % 9 - 4) * 0.25f;

    QwnHeader header; memset(&header, 0, sizeof(header));
    memcpy(header.magic, QWN_MAGIC, QWN_MAGIC_LEN);
    header.version = 1; header.arch_code = 1;
    header.n_tensors = 1; header.inline_count = 1;
    header.n_params = N * K;
    QwnTensorDesc *d = &header.inline_index[0];
    strcpy(d->name, "weight"); d->name_len = 6; d->dtype = QWN_DT_Q4_0;
    d->n_dims = 2; d->shape[0] = K; d->shape[1] = N; d->numel = N * K;
    d->byte_offset = QWN_HEADER_SIZE;
    d->byte_size = align_up((uint64_t)N * ((K + 31) / 32) * 18, 64);
    d->block_q = 32;

    uint64_t tail = align_up(d->byte_offset + d->byte_size, QWN_ALIGN);
    uint64_t file_size = tail + sizeof(QwnOverflowHeader) + sizeof(uint64_t);
    uint8_t *file = (uint8_t *)calloc(1, (size_t)file_size);
    if (!file) return -1;
    memcpy(file, &header, sizeof(header));
    uint8_t *payload = file + d->byte_offset;
    int row_bytes = ((K + 31) / 32) * 18;
    for (int n = 0; n < N; n++) q4_row(weights + n * K, K, payload + n * row_bytes);
    QwnOverflowHeader oh = {0, sizeof(QwnTensorDesc), tail + sizeof(oh),
                            tail + sizeof(oh), 0};
    memcpy(file + tail, &oh, sizeof(oh));
    memcpy(file + file_size - sizeof(uint64_t), &tail, sizeof(tail));
    FILE *f = fopen(path, "wb");
    if (!f) { free(file); return -1; }
    int ok = fwrite(file, 1, (size_t)file_size, f) == file_size;
    fclose(f); free(file); return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        QwnModel probe; const char *probe_err = NULL;
        if (qwn_open(argv[1], &probe, &probe_err) != 0) {
            fprintf(stderr, "overflow probe: %s\n", probe_err ? probe_err : "unknown");
            return 1;
        }
        const QwnTensorDesc *last = qwn_find(&probe, "tensor.45");
        int ok = last && last->byte_offset % 4096 == 0;
        qwn_close(&probe);
        return ok ? 0 : 1;
    }
    const char *path = "test_qwanto_native.qwn";
    if (make_model(path) != 0) return 1;
    QwnModel model; const char *err = NULL;
    if (qwn_open(path, &model, &err) != 0) {
        fprintf(stderr, "qwn_open: %s\n", err ? err : "unknown"); return 1;
    }
    const QwnTensorDesc *w = qwn_find(&model, "weight");
    if (!w || w->byte_offset % 4096 || w->byte_offset % 64) return 1;

    enum { M = 2, K = 37, N = 3 };
    float x[M * K], y[M * N];
    for (int i = 0; i < M * K; i++) x[i] = (float)(i % 13 - 6) * 0.125f;
    QwnScratch scratch;
    if (qwn_scratch_init(&scratch, M, K) != 0) return 1;
    if (((uintptr_t)scratch.q8 & 63u) || ((uintptr_t)scratch.token_scales & 63u)) return 1;
    if (qwn_matmul_q4_0_f32(&model, w, x, M, K, N, &scratch, y) != 0) return 1;
    const uint8_t *raw = model.base + w->byte_offset;
    int blocks = (K + 31) / 32, row_bytes = blocks * 18;
    for (int t = 0; t < M; t++) {
        for (int n = 0; n < N; n++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++) {
                const uint8_t *block = raw + n * row_bytes + (k / 32) * 18;
                float scale = from_f16(*(const uint16_t *)block);
                uint8_t packed = block[2 + (k % 32) / 2];
                int q = ((k & 1) ? packed >> 4 : packed & 15) - 8;
                ref += x[t * K + k] * (float)q * scale;
            }
            float tolerance = 0.03f * (fabsf(ref) + 1.0f);
            if (!isfinite(y[t * N + n]) || fabsf(y[t * N + n] - ref) > tolerance) {
                fprintf(stderr, "matmul mismatch: got=%f ref=%f tol=%f\n",
                        y[t * N + n], ref, tolerance);
                return 1;
            }
        }
    }
    QwnPlacement placements[1];
    QwnResidencyPlan plan = { placements, 0, 1, 0, 0, 0 };
    if (qwn_plan_residency(&model, w->byte_size, 0, &plan) != 0 ||
        plan.count != 1 || plan.items[0].tier != QWN_TIER_GPU ||
        plan.gpu_bytes != w->byte_size) return 1;
    if (qwn_prefetch(&model, w) != 0 || qwn_drop_pages(&model, w) != 0) return 1;
    qwn_scratch_destroy(&scratch);
    qwn_close(&model);
    remove(path);
    puts("Qwanto native: alignment + K-tail + arena OK");
    return 0;
}
