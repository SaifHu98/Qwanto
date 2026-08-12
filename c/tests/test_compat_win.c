#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include "../compat.h"

static int fail(const char *msg){
    fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

#define TMPF "test_win_compat.tmp"

int main() {
    printf("Testing Windows compat layer...\n");

    /* 1. Monotonic timing */
    double t1 = compat_now_s();
    Sleep(50);
    double t2 = compat_now_s();
    if (t2 <= t1) return fail("compat_now_s did not advance monotonically");
    if (t2 - t1 < 0.04 || t2 - t1 > 0.5) return fail("compat_now_s interval seems grossly incorrect");

    /* 2. File creation and compat_wpath (used in compat_open_direct / rename) */
    FILE *f = fopen(TMPF, "wb");
    if (!f) return fail("fopen failed");
    
    // Write 10MB of data
    size_t sz = 10 * 1024 * 1024;
    char *buf = malloc(sz);
    for (size_t i = 0; i < sz; i++) buf[i] = (char)(i & 0xFF);
    fwrite(buf, 1, sz, f);
    fclose(f);
    free(buf);

    /* 3. Overlapped Pread (compat_pread) */
    int fd = _open(TMPF, _O_RDONLY | _O_BINARY);
    if (fd < 0) return fail("open failed");
    
    char *rbuf = _aligned_malloc(4096, 4096);
    // Read at offset 4096
    ssize_t rd = compat_pread(fd, rbuf, 4096, 4096);
    if (rd != 4096) {
        fprintf(stderr, "compat_pread read %lld, errno %d, GetLastError %d\n", (long long)rd, errno, GetLastError());
        return fail("compat_pread did not read exact size");
    }
    for (int i = 0; i < 4096; i++) {
        if (rbuf[i] != (char)((4096 + i) & 0xFF)) return fail("compat_pread read wrong data");
    }

    /* 4. mmap with unaligned offset */
    // Map 4096 bytes starting at offset 13 (unaligned!)
    void *m = compat_mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 13);
    if (m == MAP_FAILED) return fail("compat_mmap with unaligned offset failed");
    
    char *cm = (char*)m;
    for (int i = 0; i < 4096; i++) {
        if (cm[i] != (char)((13 + i) & 0xFF)) return fail("compat_mmap mapped wrong data for unaligned offset");
    }
    
    if (compat_munmap(m, 4096) != 0) return fail("compat_munmap failed for unaligned address");
    
    /* 5. Repeated mmap/munmap stress test (check for handle leaks) */
    for (int i = 0; i < 100; i++) {
        void *p = compat_mmap(NULL, 1024, PROT_READ, MAP_SHARED, fd, i * 333);
        if (p == MAP_FAILED) return fail("compat_mmap failed during stress test");
        if (compat_munmap(p, 1024) != 0) return fail("compat_munmap failed during stress test");
    }

    _close(fd);
    _aligned_free(rbuf);
    
    /* 6. Rename replace existing */
    FILE *f2 = fopen("test_win_compat_2.tmp", "wb");
    fputs("hello", f2);
    fclose(f2);
    
    if (compat_rename(TMPF, "test_win_compat_2.tmp") != 0) return fail("compat_rename replace failed");
    
    remove("test_win_compat_2.tmp");
    
    printf("ALL TESTS PASSED.\n");
    return 0;
}
#else
int main() {
    printf("Not Windows, skipping test.\n");
    return 0;
}
#endif
