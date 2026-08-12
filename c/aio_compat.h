#ifndef AIO_COMPAT_H
#define AIO_COMPAT_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#elif defined(__linux__)
#include "uring.h"
#endif

typedef void (*ColiAioCallback)(void *user_data, int status, size_t bytes_transferred);

/* Unified Async Request Structure */
typedef struct ColiAioRequest {
    int fd;
    void *buffer;
    size_t length;
    int64_t offset;
    ColiAioCallback callback;
    void *user_data;
    int status;           /* 0 = pending, 1 = success, -1 = error */
    int cancelled;

#ifdef _WIN32
    OVERLAPPED overlapped;
    HANDLE file_handle;
#elif defined(__linux__)
    int sqe_idx;
#endif
} ColiAioRequest;

/* Unified Async Context Structure */
typedef struct ColiAioContext {
    int max_queue_depth;
    int active_requests;

#ifdef _WIN32
    HANDLE iocp;
#elif defined(__linux__)
    ColiUring ring;
#endif
} ColiAioContext;

#ifdef __cplusplus
extern "C" {
#endif

int coli_aio_init(ColiAioContext *ctx, int queue_depth);
int coli_aio_submit_read(ColiAioContext *ctx, ColiAioRequest *req);
int coli_aio_wait_all(ColiAioContext *ctx);
void coli_aio_cancel(ColiAioContext *ctx, ColiAioRequest *req);
void coli_aio_destroy(ColiAioContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* AIO_COMPAT_H */
