#include "qwn_container.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

int qwn_container_open(QwnContainer *container, const char *file_path) {
    if (!container || !file_path) return -1;

    memset(container, 0, sizeof(*container));

#ifdef _WIN32
    HANDLE hFile = CreateFileA(file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return -2;

    LARGE_INTEGER size;
    GetFileSizeEx(hFile, &size);
    container->mmap_size = (uint64_t)size.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return -3;
    }

    container->mmap_base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);

    if (!container->mmap_base) return -4;
    container->is_mmapped = true;
#else
    int fd = open(file_path, O_RDONLY);
    if (fd < 0) return -2;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return -3;
    }
    container->mmap_size = (uint64_t)st.st_size;
    container->mmap_base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    container->fd = fd;

    if (container->mmap_base == MAP_FAILED) {
        close(fd);
        return -4;
    }
    container->is_mmapped = true;
#endif

    /* Read and validate 4 KiB header */
    if (container->mmap_size >= sizeof(QwnContainerHeader)) {
        memcpy(&container->header, container->mmap_base, sizeof(QwnContainerHeader));
    }

    return 0;
}

const QwnTensorEntry *qwn_container_find_tensor(const QwnContainer *container, const char *tensor_name) {
    if (!container || !tensor_name) return NULL;

    for (uint32_t i = 0; i < container->header.n_tensors && i < QWN_MAX_TENSORS; i++) {
        if (strcmp(container->entries[i].name, tensor_name) == 0) {
            return &container->entries[i];
        }
    }
    return NULL;
}

const void *qwn_container_tensor_data(const QwnContainer *container, const QwnTensorEntry *entry) {
    if (!container || !entry || !container->mmap_base) return NULL;
    if (entry->offset_bytes + entry->size_bytes > container->mmap_size) return NULL;

    return (const void *)((const uint8_t *)container->mmap_base + entry->offset_bytes);
}

void qwn_container_prefetch_layer(const QwnContainer *container, int layer_idx) {
    (void)layer_idx;
    if (!container || !container->mmap_base) return;

#if defined(_WIN32)
    WIN32_MEMORY_RANGE_ENTRY entry;
    entry.VirtualAddress = container->mmap_base;
    entry.NumberOfBytes = (SIZE_T)(container->mmap_size > (64 * 1024 * 1024) ? (64 * 1024 * 1024) : container->mmap_size);
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#elif defined(MADV_WILLNEED)
    madvise(container->mmap_base, (size_t)(container->mmap_size > (64 * 1024 * 1024) ? (64 * 1024 * 1024) : container->mmap_size), MADV_WILLNEED);
#endif
}

void qwn_container_close(QwnContainer *container) {
    if (!container || !container->is_mmapped) return;

#ifdef _WIN32
    if (container->mmap_base) {
        UnmapViewOfFile(container->mmap_base);
        container->mmap_base = NULL;
    }
#else
    if (container->mmap_base && container->mmap_base != MAP_FAILED) {
        munmap(container->mmap_base, (size_t)container->mmap_size);
        container->mmap_base = NULL;
    }
    if (container->fd >= 0) {
        close(container->fd);
        container->fd = -1;
    }
#endif
    container->is_mmapped = false;
}
