#ifndef QWN_CONTAINER_H
#define QWN_CONTAINER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Enhanced .qwn Container Invariants:
 * - 4 KiB Header Alignment
 * - 64-Byte Payload Padding for SIMD
 * - Zero-Copy Memory Mapping & Layer-Ahead Prefetching
 * ------------------------------------------------------------------------- */

#define QWN_HEADER_MAGIC 0x51574E32 /* "QWN2" */
#define QWN_HEADER_ALIGNMENT 4096
#define QWN_PAYLOAD_PADDING 64
#define QWN_MAX_TENSORS 512

typedef enum {
    QWN_DTYPE_FP32        = 0,
    QWN_DTYPE_FP16        = 1,
    QWN_DTYPE_Q4_0        = 2,
    QWN_DTYPE_HYPER_VSQ2  = 3,  /* 2.3125 bpw */
    QWN_DTYPE_TWLA_158    = 4,  /* 1.58 bpw ternary */
    QWN_DTYPE_TURBOQUANT  = 5   /* 3.5 bpw KV */
} QwnDataType;

typedef struct {
    char name[64];
    uint32_t dtype;
    uint32_t n_dims;
    uint64_t shape[4];
    uint64_t offset_bytes;
    uint64_t size_bytes;
} QwnTensorEntry;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tensors;
    uint32_t n_layers;
    uint64_t total_payload_bytes;
    uint8_t reserved[4096 - 24]; /* Padded to exactly 4 KiB */
} QwnContainerHeader;

typedef struct {
    QwnContainerHeader header;
    QwnTensorEntry entries[QWN_MAX_TENSORS];
    void *mmap_base;
    uint64_t mmap_size;
    int fd;
    bool is_mmapped;
} QwnContainer;

/* -------------------------------------------------------------------------
 * Container APIs
 * ------------------------------------------------------------------------- */

/* Open and mmap a .qwn container file */
int qwn_container_open(QwnContainer *container, const char *file_path);

/* Locate a tensor by name */
const QwnTensorEntry *qwn_container_find_tensor(const QwnContainer *container, const char *tensor_name);

/* Get direct zero-copy pointer to tensor payload */
const void *qwn_container_tensor_data(const QwnContainer *container, const QwnTensorEntry *entry);

/* Asynchronous layer-ahead prefetch into page cache */
void qwn_container_prefetch_layer(const QwnContainer *container, int layer_idx);

/* Close container and unmap memory */
void qwn_container_close(QwnContainer *container);

#ifdef __cplusplus
}
#endif

#endif /* QWN_CONTAINER_H */
