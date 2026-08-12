#include "aio_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32

int coli_aio_init(ColiAioContext *ctx, int queue_depth) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->max_queue_depth = queue_depth;
    ctx->iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    if (ctx->iocp == NULL) {
        fprintf(stderr, "CreateIoCompletionPort failed: %lu\n", GetLastError());
        return -1;
    }
    return 0;
}

int coli_aio_submit_read(ColiAioContext *ctx, ColiAioRequest *req) {
    if (ctx->active_requests >= ctx->max_queue_depth) {
        return -1; // Queue full
    }

    HANDLE file_handle = req->file_handle;
    if (!file_handle || file_handle == INVALID_HANDLE_VALUE) {
        file_handle = (HANDLE)(intptr_t)_get_osfhandle(req->fd);
        if (file_handle == INVALID_HANDLE_VALUE) {
            return -1;
        }
        req->file_handle = file_handle;
        CreateIoCompletionPort(file_handle, ctx->iocp, (ULONG_PTR)req, 0);
    }

    memset(&req->overlapped, 0, sizeof(OVERLAPPED));
    req->overlapped.Offset = (DWORD)(req->offset & 0xFFFFFFFF);
    req->overlapped.OffsetHigh = (DWORD)((req->offset >> 32) & 0xFFFFFFFF);

    req->status = 0;
    req->cancelled = 0;

    DWORD bytes_read = 0;
    BOOL result = ReadFile(file_handle, req->buffer, (DWORD)req->length, &bytes_read, &req->overlapped);

    if (!result && GetLastError() != ERROR_IO_PENDING) {
        req->status = -1;
        return -1;
    }

    ctx->active_requests++;
    return 0;
}

int coli_aio_wait_all(ColiAioContext *ctx) {
    while (ctx->active_requests > 0) {
        DWORD bytes_transferred = 0;
        ULONG_PTR completion_key = 0;
        OVERLAPPED *overlapped = NULL;

        BOOL result = GetQueuedCompletionStatus(ctx->iocp, &bytes_transferred, &completion_key, &overlapped, INFINITE);

        if (overlapped != NULL) {
            ColiAioRequest *req = (ColiAioRequest *)completion_key;
            if (req) {
                if (!result) {
                    req->status = -1;
                } else if (req->cancelled) {
                    req->status = -1; // Mark cancelled requests as errors
                } else {
                    req->status = 1;
                }

                if (req->callback) {
                    req->callback(req->user_data, req->status, bytes_transferred);
                }

                ctx->active_requests--;
            }
        } else {
            // An error occurred and no overlapped structure was returned.
            return -1;
        }
    }
    return 0;
}

void coli_aio_cancel(ColiAioContext *ctx, ColiAioRequest *req) {
    if (req && req->status == 0) {
        CancelIoEx(req->file_handle, &req->overlapped);
        req->cancelled = 1;
    }
}

void coli_aio_destroy(ColiAioContext *ctx) {
    if (ctx->iocp) {
        CloseHandle(ctx->iocp);
        ctx->iocp = NULL;
    }
}

#elif defined(__linux__)

int coli_aio_init(ColiAioContext *ctx, int queue_depth) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->max_queue_depth = queue_depth;
    if (coli_uring_init(&ctx->ring, queue_depth) != 0) {
        fprintf(stderr, "coli_uring_init failed\n");
        return -1;
    }
    return 0;
}

int coli_aio_submit_read(ColiAioContext *ctx, ColiAioRequest *req) {
    if (ctx->active_requests >= ctx->max_queue_depth) {
        return -1; // Queue full
    }

    if (coli_uring_prep_read(&ctx->ring, req->fd, req->buffer, req->length, req->offset, (uint64_t)(uintptr_t)req) != 0) {
        return -1;
    }

    req->status = 0;
    req->cancelled = 0;
    ctx->active_requests++;
    return 0;
}

int coli_aio_wait_all(ColiAioContext *ctx) {
    // Submit all pending reads
    if (coli_uring_enter(&ctx->ring, ctx->active_requests) < 0) {
        return -1;
    }

    while (ctx->active_requests > 0) {
        struct io_uring_cqe cqe;
        // Wait for at least one completion
        if (coli_uring_enter(&ctx->ring, 1) < 0) {
             return -1;
        }

        while (coli_uring_peek(&ctx->ring, &cqe)) {
            ColiAioRequest *req = (ColiAioRequest *)(uintptr_t)cqe.user_data;
            if (req) {
                if (cqe.res < 0 || req->cancelled) {
                    req->status = -1;
                } else {
                    req->status = 1;
                }

                if (req->callback) {
                    req->callback(req->user_data, req->status, cqe.res >= 0 ? cqe.res : 0);
                }
                ctx->active_requests--;
            }
        }
    }
    return 0;
}

void coli_aio_cancel(ColiAioContext *ctx, ColiAioRequest *req) {
    // Basic io_uring cancellation requires prep_cancel which we haven't exposed in uring.h yet.
    // For now, mark as cancelled so wait_all throws it away.
    if (req) {
        req->cancelled = 1;
    }
}

void coli_aio_destroy(ColiAioContext *ctx) {
    coli_uring_close(&ctx->ring);
}

#else
// Fallback stub for macOS/unsupported OS
int coli_aio_init(ColiAioContext *ctx, int queue_depth) { return -1; }
int coli_aio_submit_read(ColiAioContext *ctx, ColiAioRequest *req) { return -1; }
int coli_aio_wait_all(ColiAioContext *ctx) { return -1; }
void coli_aio_cancel(ColiAioContext *ctx, ColiAioRequest *req) {}
void coli_aio_destroy(ColiAioContext *ctx) {}
#endif
