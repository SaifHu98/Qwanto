#include "qwanto_native.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

static double qwn_native_wall_seconds(void) {
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

/* Platform-specific mmap. compat.h already provides CreateFileMapping on
 * Windows, but to keep this module self-contained (and to avoid pulling
 * every transitive header into the C tests) we re-implement the OS calls
 * directly with the same semantics. */
#ifdef _WIN32
#include <windows.h>
typedef intptr_t qwn_fd_t;
#define QWN_BAD_FD ((intptr_t)-1)
static int qwn_open_fd(const char *path, qwn_fd_t *out) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    *out = (intptr_t)h; return 0;
}
static int qwn_size_fd(qwn_fd_t h, size_t *out) {
    LARGE_INTEGER sz; if (!GetFileSizeEx((HANDLE)h, &sz)) return -1;
    *out = (size_t)sz.QuadPart; return 0;
}
static int qwn_mmap(qwn_fd_t h, size_t sz, uint8_t **out) {
    (void)sz;
    HANDLE m = CreateFileMappingA((HANDLE)h, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m) return -1;
    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (!p) return -1;
    *out = (uint8_t *)p; return 0;
}
static void qwn_unmap(uint8_t *p, size_t sz) { (void)sz; UnmapViewOfFile(p); }
static void qwn_close_fd(qwn_fd_t h) { CloseHandle((HANDLE)h); }
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
typedef intptr_t qwn_fd_t;
#define QWN_BAD_FD -1
static int qwn_open_fd(const char *path, qwn_fd_t *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    *out = fd; return 0;
}
static int qwn_size_fd(qwn_fd_t h, size_t *out) {
    struct stat st; if (fstat((int)h, &st) < 0) return -1;
    *out = (size_t)st.st_size; return 0;
}
static int qwn_mmap(qwn_fd_t h, size_t sz, uint8_t **out) {
    void *p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, (int)h, 0);
    if (p == MAP_FAILED) return -1;
    *out = (uint8_t *)p; return 0;
}
static void qwn_unmap(uint8_t *p, size_t sz) { munmap(p, sz); }
static void qwn_close_fd(qwn_fd_t h) { close((int)h); }
#endif

uint64_t qwn_hash_name(const char *name) {
    /* FNV-1a 64 */
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        h ^= (uint64_t)*p;
        h *= 0x100000001b3ULL;
    }
    return h;
}

const char *qwn_dtype_name(uint32_t dt) {
    switch (dt) {
        case QWN_DT_F32:  return "F32";
        case QWN_DT_F16:  return "F16";
        case QWN_DT_Q4_0: return "Q4_0";
        case QWN_DT_Q8_0: return "Q8_0";
        case QWN_DT_BF16: return "BF16";
        case QWN_DT_BYTES:return "BYTES";
        case QWN_DT_VSQ: return "VSQ";
        case QWN_DT_VSQ_ULTRA: return "VSQ_ULTRA";
        case QWN_DT_HYPER_VSQ: return "HYPER_VSQ";
        case QWN_DT_HYPER_VSQ2: return "HYPER_VSQ2";
        default: return "?";
    }
}

uint64_t qwn_block_bytes(uint32_t dt, uint64_t numel) {
    switch (dt) {
        case QWN_DT_F32:  return numel * 4;
        case QWN_DT_F16:  return numel * 2;
        case QWN_DT_BF16: return numel * 2;
        case QWN_DT_BYTES:return numel;
        case QWN_DT_Q4_0: {
            uint64_t blocks = (numel + 31) / 32;
            return blocks * (2 + 16); /* f16 scale + 16 data bytes */
        }
        case QWN_DT_Q8_0: {
            uint64_t blocks = (numel + 31) / 32;
            return blocks * (2 + 32);
        }
        case QWN_DT_VSQ: return ((numel + 63) / 64) * 36;
        case QWN_DT_VSQ_ULTRA: return ((numel + 127) / 128) * 70;
        case QWN_DT_HYPER_VSQ: return ((numel + 255) / 256) * 138;
        case QWN_DT_HYPER_VSQ2: return ((numel + 255) / 256) * 74;
        default: return 0;
    }
}

static const char *ERR_BAD_MAGIC = "not a Qwanto .qwn file (bad magic)";
static const char *ERR_TRUNC     = "file truncated below header";
static const char *ERR_VERSION   = "unsupported .qwn version";
static const char *ERR_NO_TAIL   = "file too small for overflow index";
static const char *ERR_DTYPE     = "unsupported or inconsistent tensor dtype";
static const char *ERR_IO        = "I/O error";

static uint64_t qwn_expected_payload(const QwnTensorDesc *t) {
    if (!t) return 0;
    uint64_t n = t->numel;
    switch (t->dtype) {
        case QWN_DT_F32: return n * 4;
        case QWN_DT_F16:
        case QWN_DT_BF16: return n * 2;
        case QWN_DT_BYTES: return n;
        case QWN_DT_Q4_0: return ((n + 31) / 32) * 18;
        case QWN_DT_Q8_0: return ((n + 31) / 32) * 34;
        case QWN_DT_VSQ: return ((n + 63) / 64) * 36;
        case QWN_DT_VSQ_ULTRA: return ((n + 127) / 128) * 70;
        case QWN_DT_HYPER_VSQ: return ((n + 255) / 256) * 138;
        case QWN_DT_HYPER_VSQ2: return ((n + 255) / 256) * 74;
        default: return 0;
    }
}

static int qwn_validate_desc(const QwnTensorDesc *t, size_t file_size) {
    if (!t || t->name_len == 0 || t->name_len > 63 ||
        t->name[t->name_len] != '\0' || t->n_dims == 0 || t->n_dims > 4 ||
        t->numel == 0 || t->byte_size == 0 ||
        (t->byte_size & 63ULL) != 0 ||
        t->byte_offset < QWN_HEADER_SIZE ||
        (t->byte_offset & (QWN_ALIGN - 1)) != 0 ||
        t->byte_offset > file_size ||
        t->byte_size > file_size - t->byte_offset) return -1;
    uint64_t numel = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        if (t->shape[i] == 0 || numel > UINT64_MAX / t->shape[i]) return -1;
        numel *= t->shape[i];
    }
    if (numel != t->numel) return -1;
    uint64_t expected = qwn_expected_payload(t);
    if (expected == 0 || t->byte_size < expected) return -1;
    return 0;
}

int qwn_open(const char *path, QwnModel *m, const char **err) {
    double file_open_started;
    double mmap_started;
    double metadata_started;
    if (err) *err = NULL;
    memset(m, 0, sizeof(*m));
    m->fd = QWN_BAD_FD;
    file_open_started = qwn_native_wall_seconds();
    if (qwn_open_fd(path, &m->fd) != 0) {
        if (err) *err = ERR_IO;
        return -1;
    }
    if (qwn_size_fd(m->fd, &m->file_size) != 0 ||
        m->file_size < QWN_HEADER_SIZE + sizeof(uint64_t)) {
        qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
        if (err) *err = ERR_TRUNC;
        return -1;
    }
    m->open_metrics.file_open_ms = (qwn_native_wall_seconds() - file_open_started) * 1000.0;
    mmap_started = qwn_native_wall_seconds();
    if (qwn_mmap(m->fd, m->file_size, &m->base) != 0) {
        qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
        if (err) *err = ERR_IO;
        return -1;
    }
    m->open_metrics.mmap_ms = (qwn_native_wall_seconds() - mmap_started) * 1000.0;
    metadata_started = qwn_native_wall_seconds();
    /* Footer is always read first. It is the only way to locate the tail
     * index without assumptions about model size or payload padding. */
    memcpy(&m->tail_offset, m->base + m->file_size - sizeof(uint64_t),
           sizeof(m->tail_offset));
    if (m->tail_offset < QWN_HEADER_SIZE ||
        m->tail_offset > m->file_size - sizeof(uint64_t)) {
        qwn_unmap(m->base, m->file_size); m->base = NULL;
        qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
        if (err) *err = ERR_NO_TAIL;
        return -1;
    }
    /* Validate + copy the fixed header (cannot take a pointer into the mmap
     * because the caller may unmap; copy out into m->hdr). */
    const QwnHeader *h0 = (const QwnHeader *)m->base;
    if (memcmp(h0->magic, QWN_MAGIC, QWN_MAGIC_LEN) != 0) {
        qwn_unmap(m->base, m->file_size); m->base = NULL;
        qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
        if (err) *err = ERR_BAD_MAGIC;
        return -1;
    }
    if (h0->version != 1) {
        qwn_unmap(m->base, m->file_size); m->base = NULL;
        qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
        if (err) *err = ERR_VERSION;
        return -1;
    }
    m->hdr = *h0;
    /* Bounds-check the inline count. */
    if (m->hdr.inline_count > QWN_INLINE_MAX) m->hdr.inline_count = QWN_INLINE_MAX;
    for (uint32_t i = 0; i < m->hdr.inline_count; i++) {
        QwnTensorDesc *t = &m->hdr.inline_index[i];
        if (qwn_validate_desc(t, m->file_size) != 0 ||
            t->byte_offset >= m->tail_offset) {
            qwn_unmap(m->base, m->file_size); m->base = NULL;
            qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
            if (err) *err = ERR_DTYPE;
            return -1;
        }
        m->inline_hashes[i] = qwn_hash_name(t->name);
    }
    if (m->hdr.n_tensors > m->hdr.inline_count) {
        /* Footer contains the absolute tail offset. */
        if (m->file_size < QWN_HEADER_SIZE + sizeof(uint64_t)) {
            qwn_unmap(m->base, m->file_size); m->base = NULL;
            qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
            if (err) *err = ERR_NO_TAIL;
            return -1;
        }
        if (sizeof(QwnOverflowHeader) >
            m->file_size - sizeof(uint64_t) - m->tail_offset) {
            qwn_unmap(m->base, m->file_size); m->base = NULL;
            qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
            if (err) *err = ERR_NO_TAIL;
            return -1;
        }
        const QwnOverflowHeader *oh = (const QwnOverflowHeader *)(m->base + m->tail_offset);
        uint32_t n = oh->count;
        if (oh->desc_size != sizeof(QwnTensorDesc) ||
            oh->desc_offset < m->tail_offset + sizeof(QwnOverflowHeader) ||
            oh->index_offset < oh->desc_offset + (uint64_t)n * sizeof(QwnTensorDesc) ||
            oh->index_offset > m->file_size - sizeof(uint64_t) ||
            (uint64_t)n * 16 > m->file_size - sizeof(uint64_t) - oh->index_offset ||
            n != m->hdr.n_tensors - m->hdr.inline_count) {
            qwn_unmap(m->base, m->file_size); m->base = NULL;
            qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
            if (err) *err = ERR_TRUNC;
            return -1;
        }
        m->overflow_count = n;
        m->overflow_hashes = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
        m->overflow_offsets = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
        m->overflow_descs = (QwnTensorDesc *)malloc((size_t)n * sizeof(QwnTensorDesc));
        if (n && (!m->overflow_hashes || !m->overflow_offsets || !m->overflow_descs)) {
            qwn_unmap(m->base, m->file_size); m->base = NULL;
            qwn_close_fd(m->fd); m->fd = QWN_BAD_FD;
            free(m->overflow_hashes); free(m->overflow_offsets);
            free(m->overflow_descs);
            if (err) *err = ERR_IO;
            return -1;
        }
        const uint8_t *p = m->base + oh->index_offset;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t h, off;
            memcpy(&h, p, 8); p += 8;
            memcpy(&off, p, 8); p += 8;
            m->overflow_hashes[i] = h;
            m->overflow_offsets[i] = off;
            if (off < oh->desc_offset ||
                off + sizeof(QwnTensorDesc) > oh->index_offset) {
                qwn_close(m);
                if (err) *err = ERR_TRUNC;
                return -1;
            }
            memcpy(&m->overflow_descs[i], m->base + off, sizeof(QwnTensorDesc));
            const QwnTensorDesc *t = &m->overflow_descs[i];
            if (qwn_validate_desc(t, m->file_size) != 0 ||
                t->byte_offset >= m->tail_offset) {
                qwn_close(m);
                if (err) *err = ERR_DTYPE;
                return -1;
            }
        }
    }
    m->open_metrics.metadata_parse_ms =
        (qwn_native_wall_seconds() - metadata_started) * 1000.0;
    return 0;
}

void qwn_close(QwnModel *m) {
    if (!m) return;
    if (m->base) qwn_unmap(m->base, m->file_size);
    if (m->fd != QWN_BAD_FD) qwn_close_fd(m->fd);
    free(m->overflow_hashes);
    free(m->overflow_offsets);
    free(m->overflow_descs);
    memset(m, 0, sizeof(*m));
}

const QwnTensorDesc *qwn_find(const QwnModel *m, const char *name) {
    if (!m || !name) return NULL;
    uint64_t h = qwn_hash_name(name);
    /* Inline: linear scan of up to 64 entries is cache-friendly. */
    for (uint32_t i = 0; i < m->hdr.inline_count; i++) {
        if (m->inline_hashes[i] == h &&
            strcmp(m->hdr.inline_index[i].name, name) == 0) {
            return &m->hdr.inline_index[i];
        }
    }
    /* Overflow: sorted by hash, binary search. */
    if (m->overflow_count == 0) return NULL;
    int32_t lo = 0, hi = (int32_t)m->overflow_count - 1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) >> 1;
        uint64_t hm = m->overflow_hashes[mid];
        if (hm < h) lo = mid + 1;
        else if (hm > h) hi = mid - 1;
        else {
            if (strcmp(m->overflow_descs[mid].name, name) == 0)
                return &m->overflow_descs[mid];
            /* Continue scanning in case of duplicate hashes. */
            for (int32_t j = mid - 1; j >= 0 && m->overflow_hashes[j] == h; j--) {
                if (strcmp(m->overflow_descs[j].name, name) == 0)
                    return &m->overflow_descs[j];
            }
            for (uint32_t j = mid + 1; j < m->overflow_count && m->overflow_hashes[j] == h; j++) {
                if (strcmp(m->overflow_descs[j].name, name) == 0)
                    return &m->overflow_descs[j];
            }
            return NULL;
        }
    }
    return NULL;
}

int64_t qwn_read(const QwnModel *m, const QwnTensorDesc *t, void *dst, size_t dst_bytes) {
    if (!m || !t || !dst) return -1;
    if (t->byte_offset + t->byte_size > m->file_size) return -1;
    if (dst_bytes < t->byte_size) return -1;
    memcpy(dst, m->base + t->byte_offset, (size_t)t->byte_size);
    return (int64_t)t->byte_size;
}

const void *qwn_data(const QwnModel *m, const QwnTensorDesc *t) {
    if (!m || !t || t->byte_offset > m->file_size ||
        t->byte_size > m->file_size - t->byte_offset) return NULL;
    return m->base + t->byte_offset;
}

const QwnTensorDesc *qwn_tensor_at(const QwnModel *m, uint32_t i) {
    if (i < m->hdr.inline_count) return &m->hdr.inline_index[i];
    i -= m->hdr.inline_count;
    return i < m->overflow_count ? &m->overflow_descs[i] : NULL;
}

static uint8_t tensor_priority(const char *name) {
    /* Dense/token-hot weights outrank sparse experts. */
    if (strstr(name, "embed") || strstr(name, "lm_head")) return 255;
    if (strstr(name, "norm")) return 245;
    if (strstr(name, "q_proj") || strstr(name, "k_proj") ||
        strstr(name, "v_proj") || strstr(name, "o_proj") ||
        strstr(name, "attn")) return 230;
    if (strstr(name, "gate") && !strstr(name, "expert")) return 210;
    if (strstr(name, "expert") || strstr(name, "experts")) return 80;
    return 160;
}

static int placement_cmp(const void *a, const void *b) {
    const QwnPlacement *x = (const QwnPlacement *)a;
    const QwnPlacement *y = (const QwnPlacement *)b;
    if (x->priority != y->priority) return x->priority > y->priority ? -1 : 1;
    if (x->byte_size != y->byte_size) return x->byte_size < y->byte_size ? -1 : 1;
    return x->tensor_index < y->tensor_index ? -1 : 1;
}

int qwn_plan_residency(const QwnModel *m, uint64_t gpu_budget,
                       uint64_t ram_budget, QwnResidencyPlan *plan) {
    if (!m || !plan || !plan->items || plan->capacity < m->hdr.n_tensors)
        return -1;
    memset(plan->items, 0, (size_t)plan->capacity * sizeof(QwnPlacement));
    plan->count = m->hdr.n_tensors;
    plan->gpu_bytes = plan->ram_bytes = plan->nvme_bytes = 0;
    for (uint32_t i = 0; i < plan->count; i++) {
        const QwnTensorDesc *t = qwn_tensor_at(m, i);
        if (!t) return -1;
        QwnPlacement *p = &plan->items[i];
        p->name_hash = qwn_hash_name(t->name);
        p->byte_offset = t->byte_offset;
        p->byte_size = t->byte_size;
        p->tensor_index = i;
        p->priority = tensor_priority(t->name);
        p->tier = QWN_TIER_NVME;
    }
    qsort(plan->items, plan->count, sizeof(QwnPlacement), placement_cmp);
    for (uint32_t i = 0; i < plan->count; i++) {
        QwnPlacement *p = &plan->items[i];
        if (plan->gpu_bytes <= gpu_budget &&
            p->byte_size <= gpu_budget - plan->gpu_bytes) {
            p->tier = QWN_TIER_GPU;
            plan->gpu_bytes += p->byte_size;
        } else if (plan->ram_bytes <= ram_budget &&
                   p->byte_size <= ram_budget - plan->ram_bytes) {
            p->tier = QWN_TIER_RAM;
            plan->ram_bytes += p->byte_size;
        } else {
            p->tier = QWN_TIER_NVME;
            plan->nvme_bytes += p->byte_size;
        }
    }
    return 0;
}

int qwn_prefetch(const QwnModel *m, const QwnTensorDesc *t) {
    if (!m || !t || t->byte_offset + t->byte_size > m->file_size) return -1;
#ifdef _WIN32
    typedef BOOL (WINAPI *prefetch_fn)(HANDLE, ULONG_PTR,
                                      PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static prefetch_fn cached_fn = NULL;
    static int resolved = 0;
    if (!resolved) {
        cached_fn = (prefetch_fn)(void *)(uintptr_t)GetProcAddress(GetModuleHandleA("kernel32.dll"),
                                                                    "PrefetchVirtualMemory");
        resolved = 1;
    }
    if (!cached_fn) return 0;
    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = m->base + t->byte_offset;
    range.NumberOfBytes = (SIZE_T)t->byte_size;
    return cached_fn(GetCurrentProcess(), 1, &range, 0) ? 0 : -1;
#else
    uintptr_t addr = (uintptr_t)(m->base + t->byte_offset);
    uintptr_t page_addr = addr & ~((uintptr_t)4095);
    size_t len = (size_t)(addr + t->byte_size - page_addr);
    madvise((void *)page_addr, len, MADV_WILLNEED);
    return 0;
#endif
}

int qwn_prefetch_batch(const QwnModel *m, const QwnTensorDesc *const *tensors, uint32_t count) {
    if (!m || !tensors || count == 0) return 0;
#ifdef _WIN32
    typedef BOOL (WINAPI *prefetch_fn)(HANDLE, ULONG_PTR,
                                      PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static prefetch_fn cached_fn = NULL;
    static int resolved = 0;
    if (!resolved) {
        cached_fn = (prefetch_fn)(void *)(uintptr_t)GetProcAddress(GetModuleHandleA("kernel32.dll"),
                                                                    "PrefetchVirtualMemory");
        resolved = 1;
    }
    if (!cached_fn) return 0;
    WIN32_MEMORY_RANGE_ENTRY ranges[16];
    ULONG valid_count = 0;
    for (uint32_t i = 0; i < count && valid_count < 16; i++) {
        const QwnTensorDesc *t = tensors[i];
        if (t && t->byte_offset + t->byte_size <= m->file_size) {
            ranges[valid_count].VirtualAddress = m->base + t->byte_offset;
            ranges[valid_count].NumberOfBytes = (SIZE_T)t->byte_size;
            valid_count++;
        }
    }
    if (valid_count == 0) return 0;
    return cached_fn(GetCurrentProcess(), valid_count, ranges, 0) ? 0 : -1;
#else
    for (uint32_t i = 0; i < count; i++) {
        const QwnTensorDesc *t = tensors[i];
        if (t && t->byte_offset + t->byte_size <= m->file_size) {
            uintptr_t addr = (uintptr_t)(m->base + t->byte_offset);
            uintptr_t page_addr = addr & ~((uintptr_t)4095);
            size_t len = (size_t)(addr + t->byte_size - page_addr);
            madvise((void *)page_addr, len, MADV_WILLNEED);
        }
    }
    return 0;
#endif
}

int qwn_drop_pages(const QwnModel *m, const QwnTensorDesc *t) {
    if (!m || !t || t->byte_offset + t->byte_size > m->file_size) return -1;
#ifdef _WIN32
    /* File-backed pages are naturally evicted by the Windows memory manager. */
    return 0;
#else
    uintptr_t addr = (uintptr_t)(m->base + t->byte_offset);
    uintptr_t page_addr = addr & ~((uintptr_t)4095);
    size_t len = (size_t)(addr + t->byte_size - page_addr);
    madvise((void *)page_addr, len, MADV_DONTNEED);
    return 0;
#endif
}
