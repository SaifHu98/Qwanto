#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mock types to include sched.h */
typedef struct {
    int I, O, fmt;
} QT;

/* Stub out external dependencies */
int g_cuda_enabled = 1;
int g_metal_enabled = 0;

/* Include the scheduler under test */
#include "../sched.h"

/* Define the global instance */
SchedState g_sched = {0};

static int fail(const char *msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    printf("Testing Scheduler...\n");

    /* Initialize with test baseline */
    memset(&g_sched, 0, sizeof(g_sched));
    g_sched.force_cpu = 0;
    g_sched.accelerator_failed = 0;
    g_sched.s_min_gpu = 16;
    g_sched.expert_cpu_ms = 1.0;
    g_sched.expert_transfer_ms = 3.5;

    /* 1. Test Dense Projections */
    /* Small batch S=1, should be CPU */
    if (sched_route_dense(1, 4096, 4096, 0, 1) != TIER_CPU) return fail("Dense S=1 should be CPU");

    /* Large batch S=64, should be GPU */
    if (sched_route_dense(64, 4096, 4096, 0, 1) != TIER_GPU) return fail("Dense S=64 should be GPU");

    /* Non-resident tensor, always CPU */
    if (sched_route_dense(64, 4096, 4096, 0, 0) != TIER_CPU) return fail("Non-resident dense should be CPU");

    /* 2. Test Routed Experts */
    /* Pinned expert (resident=1) behaves like dense */
    if (sched_route_expert(1, 4096, 4096, 2, 1) != TIER_CPU) return fail("Pinned expert S=1 should be CPU");
    if (sched_route_expert(64, 4096, 4096, 2, 1) != TIER_GPU) return fail("Pinned expert S=64 should be GPU");

    /* Unpinned expert (resident=0, fmt=2 int4).
     * For S=1: CPU is faster due to PCIe transfer latency overhead.
     * For S=8: GPU compute advantage amortizes transfer cost. */
    if (sched_route_expert(1, 4096, 4096, 2, 0) != TIER_CPU) return fail("Unpinned expert S=1 should be CPU");
    if (sched_route_expert(8, 4096, 4096, 2, 0) != TIER_GPU) return fail("Unpinned expert S=8 should be GPU");

    /* 3. Fallback logic */
    sched_mark_fallback();
    if (sched_route_dense(64, 4096, 4096, 0, 1) != TIER_CPU) return fail("Failed accelerator should force CPU");
    if (sched_route_expert(64, 4096, 4096, 0, 1) != TIER_CPU) return fail("Failed accelerator should force CPU for experts");

    printf("ALL TESTS PASSED.\n");
    return 0;
}
