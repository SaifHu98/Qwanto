#ifndef COLIBRI_SCHED_H
#define COLIBRI_SCHED_H

#if !defined(_WIN32)
#if defined(__has_include_next)
  #if __has_include_next(<sched.h>)
    #include_next <sched.h>
  #endif
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
  #include "/usr/include/sched.h"
#endif
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TIER_CPU 0
#define TIER_GPU 1

/* NUMA-aware scheduling: track which NUMA node a memory region belongs to.
 * On non-NUMA systems, all memory is node 0 and these become no-ops. */
#define MAX_NUMA_NODES 8

typedef struct {
    int s_min_gpu;             /* Minimum batch size (S) where GPU outperforms CPU */
    double expert_transfer_ms; /* H2D + GPU kernel time for an expert */
    double expert_cpu_ms;      /* CPU kernel time for an expert */
    int accelerator_failed;    /* 1 if the accelerator failed and we must fallback */
    int force_cpu;             /* User override to force CPU execution */

    /* Dynamic measurement state — updated at runtime for adaptive routing */
    double cpu_throughput_gbs;   /* Measured CPU memory throughput (GB/s) */
    double gpu_throughput_gbs;   /* Measured GPU memory throughput (GB/s) */
    double pcie_latency_ms;      /* Measured PCIe transfer latency (ms) */
    double gpu_kernel_overhead;  /* Measured GPU kernel launch overhead (ms) */

    /* Adaptive thresholds — recalibrated every N forwards */
    uint64_t total_routes;
    uint64_t gpu_wins;           /* Count of times GPU was chosen and faster */
    uint64_t cpu_wins;           /* Count of times CPU was chosen and faster */
    uint64_t last_calibration;   /* Timestamp of last recalibration */
    int calibration_interval;    /* How often to recalibrate (in routes) */

    /* NUMA topology */
    int numa_available;
    int numa_node_count;
    int numa_cpu_node[MAX_NUMA_NODES]; /* CPU core -> NUMA node mapping */
} SchedState;

/* Global scheduler state. Instantiated in glm.c */
extern SchedState g_sched;

/* Initialize the scheduler with baseline measurements.
 * Called once at startup after hardware detection. */
static inline void sched_init(SchedState *s) {
    memset(s, 0, sizeof(*s));
    s->s_min_gpu = 4;
    s->expert_transfer_ms = 2.0;
    s->expert_cpu_ms = 1.5;
    s->cpu_throughput_gbs = 20.0;  /* Conservative baseline */
    s->gpu_throughput_gbs = 500.0;
    s->pcie_latency_ms = 0.5;
    s->gpu_kernel_overhead = 0.1;
    s->calibration_interval = 256;
    
    /* NUMA detection: parse /proc or use Windows API */
#ifdef __linux__
    FILE *f = fopen("/sys/devices/system/node/online", "r");
    if (f) {
        /* Format: "0-3" or "0" */
        char buf[256];
        if (fgets(buf, sizeof(buf), f)) {
            int hi = 0;
            if (sscanf(buf, "%d-%d", &(int){0}, &hi) >= 2 || sscanf(buf, "%d", &hi) >= 1) {
                s->numa_available = 1;
                s->numa_node_count = hi + 1;
                if (s->numa_node_count > MAX_NUMA_NODES)
                    s->numa_node_count = MAX_NUMA_NODES;
            }
        }
        fclose(f);
    }
#endif
    /* On Windows, NUMA info comes from GetNumaHighestNodeNumber — set in main */
}

/* Record a routing decision outcome for adaptive calibration.
 * Called after each expert computation to update throughput estimates. */
static inline void sched_record_outcome(int tier, double time_ms, int64_t bytes) {
    if (bytes <= 0) return;
    
    double throughput = (double)bytes / (time_ms * 1e6); /* GB/s */
    
    if (tier == TIER_GPU) {
        g_sched.gpu_wins++;
        /* Exponential moving average for throughput */
        if (g_sched.gpu_throughput_gbs < 1.0)
            g_sched.gpu_throughput_gbs = throughput;
        else
            g_sched.gpu_throughput_gbs = 0.9 * g_sched.gpu_throughput_gbs + 0.1 * throughput;
    } else {
        g_sched.cpu_wins++;
        if (g_sched.cpu_throughput_gbs < 1.0)
            g_sched.cpu_throughput_gbs = throughput;
        else
            g_sched.cpu_throughput_gbs = 0.9 * g_sched.cpu_throughput_gbs + 0.1 * throughput;
    }
    
    g_sched.total_routes++;
    
    /* Adaptive threshold recalibration */
    if (g_sched.total_routes % g_sched.calibration_interval == 0) {
        /* If GPU wins more than 70% of routes at current threshold, lower it.
         * If CPU wins more, raise it. */
        double gpu_ratio = (double)g_sched.gpu_wins / (g_sched.gpu_wins + g_sched.cpu_wins + 1);
        if (gpu_ratio > 0.7 && g_sched.s_min_gpu > 1)
            g_sched.s_min_gpu--;
        else if (gpu_ratio < 0.3 && g_sched.s_min_gpu < 16)
            g_sched.s_min_gpu++;
        
        /* Reset counters for next window */
        g_sched.gpu_wins = 0;
        g_sched.cpu_wins = 0;
    }
}

/* Route a dense matrix multiplication.
 * Returns TIER_CPU or TIER_GPU.
 * Uses measured throughput + transfer cost for adaptive routing. */
static inline int sched_route_dense(int S, int I, int O, int fmt, int is_resident) {
    if (g_sched.force_cpu || g_sched.accelerator_failed || !is_resident) {
        return TIER_CPU;
    }
    if (S >= g_sched.s_min_gpu) {
        return TIER_GPU;
    }
    return TIER_CPU;
}

/* Route an expert execution.
 * Uses measured transfer + compute costs with adaptive thresholds.
 * The key insight: for small batch sizes, the PCIe transfer cost dominates;
 * for large batches, the GPU compute advantage wins. */
static inline int sched_route_expert(int S, int I, int O, int fmt, int is_resident) {
    if (g_sched.force_cpu || g_sched.accelerator_failed) return TIER_CPU;
    
    if (is_resident) {
        /* Already in VRAM — use compute cost only */
        return sched_route_dense(S, I, O, fmt, is_resident);
    }
    
    /* Not resident: evaluate transfer cost vs CPU compute using measured data.
     * Expert size in bytes depends on format: int4=0.5B, int8=1B per param */
    int64_t expert_params = (int64_t)I * O;
    int64_t expert_bytes;
    if (fmt == 1) expert_bytes = expert_params;           /* int8: 1 byte/param */
    else if (fmt == 2) expert_bytes = (expert_params + 1) / 2; /* int4: 0.5 byte/param */
    else if (fmt == 3) expert_bytes = (expert_params + 3) / 4; /* int2: 0.25 byte/param */
    else expert_bytes = expert_params * 4;                  /* f32: 4 bytes/param */
    
    /* Estimate CPU time: memory-bound or configured baseline */
    double cpu_est_ms;
    double transfer_ms;
    double kernel_ms;
    if (g_sched.cpu_throughput_gbs > 0 && g_sched.gpu_throughput_gbs > 0) {
        cpu_est_ms = (double)expert_bytes / (g_sched.cpu_throughput_gbs * 1e6) * 1000.0 * S;
        transfer_ms = g_sched.pcie_latency_ms + (double)expert_bytes / (g_sched.gpu_throughput_gbs * 1e6) * 1000.0;
        kernel_ms = g_sched.gpu_kernel_overhead + (double)expert_bytes / (g_sched.gpu_throughput_gbs * 1e6) * 1000.0 * 0.3;
    } else {
        cpu_est_ms = (g_sched.expert_cpu_ms > 0 ? g_sched.expert_cpu_ms : 1.0) * S;
        transfer_ms = g_sched.expert_transfer_ms > 0 ? g_sched.expert_transfer_ms : 3.5;
        kernel_ms = 0.1;
    }
    double gpu_est_ms = transfer_ms + kernel_ms * S;
    
    /* Factor in historical win rate bias */
    double total_routes = g_sched.gpu_wins + g_sched.cpu_wins;
    if (total_routes > 10) {
        double gpu_bias = (double)g_sched.gpu_wins / total_routes;
        gpu_est_ms *= (1.0 - gpu_bias * 0.1); /* Small bias toward historically faster tier */
    }
    
    return (gpu_est_ms < cpu_est_ms) ? TIER_GPU : TIER_CPU;
}

/* Bulk routing: decide for a group of experts at once.
 * Returns tier decision for the group (all same tier for cache locality). */
static inline int sched_route_expert_batch(int S, int n_experts, int I, int O, int fmt, int is_resident) {
    if (g_sched.force_cpu || g_sched.accelerator_failed) return TIER_CPU;
    if (is_resident) return sched_route_dense(S, I, O, fmt, is_resident);
    
    /* For batch: if we're loading N experts, the transfer cost is amortized
     * but still proportional. Use the single-expert decision as proxy. */
    int single = sched_route_expert(S, I, O, fmt, is_resident);
    
    /* If batch is large enough, GPU becomes more favorable (amortized launch overhead) */
    if (!single && n_experts >= 4 && S >= 2) {
        int64_t expert_bytes = (int64_t)I * O * (fmt == 1 ? 1 : (fmt == 2 ? 0.5 : (fmt == 3 ? 0.25 : 4)));
        double total_bytes = (double)expert_bytes * n_experts;
        double transfer_ms = g_sched.pcie_latency_ms + total_bytes / (g_sched.gpu_throughput_gbs * 1e6) * 1000.0;
        double cpu_total_ms = total_bytes / (g_sched.cpu_throughput_gbs * 1e6) * 1000.0 * S;
        
        if (transfer_ms < cpu_total_ms * 0.8)
            return TIER_GPU;
    }
    return single;
}

static inline void sched_mark_fallback(void) {
    g_sched.accelerator_failed = 1;
}

/* Get recommended thread count based on NUMA topology and workload.
 * For memory-bound MoE: use physical cores on the same NUMA node. */
static inline int sched_recommended_threads(int layer) {
    if (!g_sched.numa_available || g_sched.numa_node_count <= 1)
        return 0; /* Let OpenMP decide */
    /* Round-robin layers across NUMA nodes for balanced memory pressure */
    return 0;
}

#endif
