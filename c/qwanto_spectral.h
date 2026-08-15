#ifndef QWANTO_SPECTRAL_H
#define QWANTO_SPECTRAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * SpectralAI: O(N log N) MoE Routing via Spatial Bounding Volume Hierarchy
 * ------------------------------------------------------------------------- */

#define QWN_SPECTRAL_MAX_EXPERTS 256
#define QWN_SPECTRAL_MAX_TOPK 8
#define QWN_SPECTRAL_DIM 128

typedef struct {
    float min_bounds[QWN_SPECTRAL_DIM]; /* Hyper-dimensional AABB lower bounds */
    float max_bounds[QWN_SPECTRAL_DIM]; /* Hyper-dimensional AABB upper bounds */
    int left_child;                     /* Index of left child (-1 if leaf) */
    int right_child;                    /* Index of right child (-1 if leaf) */
    int expert_id;                      /* Expert index if leaf node (-1 otherwise) */
    float centroid[QWN_SPECTRAL_DIM];   /* Expert centroid vector */
} QwnBVHNode;

typedef struct {
    QwnBVHNode nodes[QWN_SPECTRAL_MAX_EXPERTS * 2];
    int node_count;
    int root_index;
    int n_experts;
    int dim;
    bool rt_cores_enabled;
} QwnSpectralRouter;

/* -------------------------------------------------------------------------
 * SpectralAI APIs
 * ------------------------------------------------------------------------- */

/* Initialize BVH Spatial Router for N experts with D dimensions */
int qwn_spectral_router_init(
    QwnSpectralRouter *router,
    const float *expert_centroids,
    int n_experts,
    int dim
);

/* Execute O(N log N) Top-K routing via hierarchical BVH tree traversal */
int qwn_spectral_route_topk(
    const QwnSpectralRouter *router,
    const float *token_hidden_state,
    int top_k,
    int *selected_experts,
    float *routing_weights
);

/* Query routing metrics */
double qwn_spectral_last_latency_us(const QwnSpectralRouter *router);

#ifdef __cplusplus
}
#endif

#endif /* QWANTO_SPECTRAL_H */
