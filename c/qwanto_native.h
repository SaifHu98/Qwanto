#ifndef QWANTO_NATIVE_H
#define QWANTO_NATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All on-disk structures use byte-level packing (no compiler-inserted
 * padding) so the Python writer and the C reader agree on offsets. */

/* Qwanto Native on-disk format (.qwn)
 * ----------------------------------
 * Layout:
 *   [0 .. 0x1000)                  fixed 4 KiB header
 *     magic "QWANTO_NATIVE_V1"     16 bytes
 *     version, flags, dtype policy  fixed scalars
 *     inline index for first 29 tensors (fast path: header-only open)
 *   [aligned, +4 KiB padding)      tensor payloads
 *     each tensor: header table entry says dtype/shape/offset/bytes
 *     offset is 4 KiB aligned
 *     length is multiple of 64 (cache-line friendly)
 *   [tail_offset .. EOF-8)         overflow descriptors + sorted index
 *   [EOF-8 .. EOF)                 uint64 tail_offset
 *     Runtime reads the final 8 bytes first; tail location never depends on
 *     model size or alignment padding.
 *     entries are sorted by (name_hash) so lookup is O(log N) with one
 *     page fault for the overflow region on cold access.
 *
 * Quantization:
 *   F32 : raw little-endian float, length = 4*numel
 *   F16 : raw half,    length = 2*numel
 *   Q4_0: 32 elements per block: scale (f16) + 16 bytes (32 * 4-bit, low nibble first)
 *          length = numel/32 * (2 + 16) = numel/32 * 18
 *
 * Header is 4 KiB so it fits in a single NVMe read unit; index is sorted by
 * a 64-bit FNV-1a hash of the tensor name so lookup is O(log N) with one
 * disk page fault for the overflow region on cold access.
 */

#define QWN_MAGIC          "QWANTO_NATIVE_V1"
#define QWN_MAGIC_LEN      16
#define QWN_HEADER_SIZE    0x1000
#define QWN_ALIGN          0x1000
#define QWN_TAIL_INDEX     0x1000
/* Header is 4 KiB; with the prefix (112 bytes) and 136-byte TensorDescs, the
 * inline index fits at most 29 tensors. More than that uses the tail index. */
#define QWN_INLINE_MAX     29

/* Dtype codes (matches GGUF naming for familiarity) */
#define QWN_DT_F32         0
#define QWN_DT_F16         1
#define QWN_DT_Q4_0        2
#define QWN_DT_Q8_0        3
#define QWN_DT_BF16        4
#define QWN_DT_BYTES       5
#define QWN_DT_VSQ         6
#define QWN_DT_VSQ_ULTRA   7
#define QWN_DT_HYPER_VSQ   8
#define QWN_DT_HYPER_VSQ2  9

#if defined(_MSC_VER)
#define QWN_PACKED
#pragma pack(push, 1)
#else
#define QWN_PACKED __attribute__((packed))
#endif

/* Qwanto Vector-Superblock Quantization (64 elements / 36 bytes) */
typedef struct QWN_PACKED {
    uint16_t d_base;              /* FP16 base scale */
    uint8_t  d_sub0;              /* 4-bit/8-bit sub-block scale 0 */
    uint8_t  d_sub1;              /* 4-bit/8-bit sub-block scale 1 */
    uint8_t  qs[32];              /* 64 x 4-bit nibbles (bias 8) */
} QwnBlockVSQ;

/* Qwanto Vector-Superblock Ultra Quantization (128 elements / 70 bytes) */
typedef struct QWN_PACKED {
    uint16_t d_base;              /* FP16 base scale */
    uint16_t m_base;              /* FP16 zero-point offset */
    uint8_t  d_subs[2];           /* 4 x 4-bit sub-quadrant scales */
    uint8_t  qs[64];              /* 128 x 4-bit nibbles (bias 8) */
} QwnBlockVSQUltra;

/* Qwanto Hyper-Vector Superblock Quantization (256 elements / 138 bytes) */
typedef struct QWN_PACKED {
    uint16_t d_base;              /* FP16 base scale */
    uint16_t m_base;              /* FP16 zero-point offset */
    uint8_t  d_subs[4];           /* 8 x 4-bit sub-octant scale multipliers */
    uint16_t sparse_mask;         /* 16-bit outlier/sparsity mask */
    uint8_t  qs[128];             /* 256 x 4-bit nibbles (bias 8) */
} QwnBlockHyperVSQ;

/* Qwanto Super-Sub-2-bit Hyper-Vector Superblock Quantization (256 elements / 74 bytes = 2.31 bpw) */
typedef struct QWN_PACKED {
    uint16_t d_base;              /* FP16 base scale */
    uint16_t m_base;              /* FP16 zero-point offset */
    uint8_t  d_subs[4];           /* 8 x 4-bit sub-octant scale multipliers */
    uint16_t sparse_mask;         /* 16-bit outlier/sparsity mask */
    uint8_t  qs[64];              /* 256 x 2-bit quaternary weights (4 per byte) */
} QwnBlockHyperVSQ2;

typedef struct QWN_PACKED {
    char     name[64];
    uint32_t name_len;            /* strlen of name, 0..63 */
    uint32_t dtype;               /* QWN_DT_* */
    uint32_t n_dims;
    uint64_t shape[4];            /* shape[0] is fastest-varying */
    uint64_t numel;
    uint64_t byte_offset;         /* from start of file, 4 KiB aligned */
    uint64_t byte_size;           /* padded to 64 */
    uint32_t block_q;             /* block size in elements (0 for non-quant) */
} QwnTensorDesc;

typedef struct QWN_PACKED {
    char     magic[QWN_MAGIC_LEN];
    uint32_t version;             /* currently 1 */
    uint32_t flags;
    uint32_t arch_code;           /* 0=unknown 1=llama-like 2=moe */
    uint32_t n_tensors;
    uint32_t inline_count;        /* number of entries present in the inline index */
    uint32_t reserved0;
    uint64_t n_params;            /* sum of numel across all tensors */
    uint64_t arch_dims[8];        /* hidden,intermediate,heads,kv_heads,head_dim,layers,vocab,ctx */
    QwnTensorDesc inline_index[QWN_INLINE_MAX];
} QwnHeader;

_Static_assert(sizeof(QwnTensorDesc) == 136, "QwnTensorDesc disk ABI changed");
_Static_assert(sizeof(QwnHeader) <= QWN_HEADER_SIZE, "QwnHeader exceeds 4 KiB");

typedef struct QWN_PACKED {
    uint32_t count;
    uint32_t desc_size;
    uint64_t desc_offset;
    uint64_t index_offset;
    uint64_t reserved;
} QwnOverflowHeader;

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

typedef struct QwnModel {
    intptr_t  fd;                 /* HANDLE cast on Windows; fd on POSIX */
    uint8_t  *base;               /* mmap base, may be MAP_FAILED on error */
    size_t    file_size;
    uint64_t  tail_offset;
    QwnHeader hdr;
    /* Runtime-side overflow index (built once on open):
     * sorted array of (name_hash, file_offset) where file_offset points to
     * the tensor's serialized QwnTensorDesc in the tail block. */
    uint64_t *overflow_hashes;
    uint64_t *overflow_offsets;
    QwnTensorDesc *overflow_descs;
    uint32_t  overflow_count;
    uint64_t  inline_hashes[QWN_INLINE_MAX]; /* precomputed at open time */
} QwnModel;

enum { QWN_TIER_GPU = 0, QWN_TIER_RAM = 1, QWN_TIER_NVME = 2 };

typedef struct {
    uint64_t name_hash;
    uint64_t byte_offset;
    uint64_t byte_size;
    uint32_t tensor_index;
    uint8_t  tier;
    uint8_t  priority;
    uint16_t reserved;
} QwnPlacement;

typedef struct {
    QwnPlacement *items;          /* caller-owned array [capacity] */
    uint32_t count;
    uint32_t capacity;
    uint64_t gpu_bytes;
    uint64_t ram_bytes;
    uint64_t nvme_bytes;
} QwnResidencyPlan;

/* Open a .qwn file. Returns 0 on success, -1 on failure.
 * The file is mmap'd read-only and never written to.
 * `err` (optional) receives a static error string. */
int  qwn_open(const char *path, QwnModel *m, const char **err);
void qwn_close(QwnModel *m);

/* Find a tensor by name. Returns NULL if not found.
 * Search order: inline index (no I/O), then sorted overflow index
 * (one binary search, one page fault on cold). */
const QwnTensorDesc *qwn_find(const QwnModel *m, const char *name);

/* Read tensor payload into a caller-provided buffer.
 * For F32/F16: exact copy of byte_size.
 * For Q4_0/Q8_0: same — caller interprets the blocks.
 * Returns bytes copied, or -1 on size mismatch. */
int64_t qwn_read(const QwnModel *m, const QwnTensorDesc *t, void *dst, size_t dst_bytes);
const void *qwn_data(const QwnModel *m, const QwnTensorDesc *t);

/* FNV-1a 64-bit hash, exposed so the converter writes the same hashes. */
uint64_t qwn_hash_name(const char *name);

/* Info helpers for diagnostics / CLI inspect */
const char *qwn_dtype_name(uint32_t dt);
uint64_t qwn_block_bytes(uint32_t dt, uint64_t numel);

/* Hot tensor placement. GPU budget fills first by priority, then RAM;
 * everything else stays file-backed and is prefetched from NVMe on demand. */
int qwn_plan_residency(const QwnModel *m, uint64_t gpu_budget,
                       uint64_t ram_budget, QwnResidencyPlan *plan);

/* Ask the OS to prefetch/drop the pages of one tensor. */
int qwn_prefetch(const QwnModel *m, const QwnTensorDesc *t);
int qwn_drop_pages(const QwnModel *m, const QwnTensorDesc *t);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_NATIVE_H */
