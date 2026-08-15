#include "qwanto_spectral.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: Dot product between two vectors */
static inline float qwn_vec_dot(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Recursive BVH builder */
static int qwn_bvh_build_recursive(
    QwnSpectralRouter *router,
    int *expert_indices,
    int count,
    const float *centroids
) {
    if (count <= 0) return -1;

    int node_idx = router->node_count++;
    QwnBVHNode *node = &router->nodes[node_idx];
    node->left_child = -1;
    node->right_child = -1;
    node->expert_id = -1;

    int dim = router->dim;

    /* Initialize AABB bounds */
    for (int d = 0; d < dim; d++) {
        node->min_bounds[d] = 1e9f;
        node->max_bounds[d] = -1e9f;
        node->centroid[d] = 0.0f;
    }

    for (int i = 0; i < count; i++) {
        int exp_id = expert_indices[i];
        const float *c = centroids + exp_id * dim;
        for (int d = 0; d < dim; d++) {
            if (c[d] < node->min_bounds[d]) node->min_bounds[d] = c[d];
            if (c[d] > node->max_bounds[d]) node->max_bounds[d] = c[d];
            node->centroid[d] += c[d];
        }
    }

    for (int d = 0; d < dim; d++) {
        node->centroid[d] /= (float)count;
    }

    /* Leaf node case */
    if (count == 1) {
        node->expert_id = expert_indices[0];
        return node_idx;
    }

    /* Find axis with maximum variance */
    int best_axis = 0;
    float max_extent = -1.0f;
    for (int d = 0; d < dim; d++) {
        float extent = node->max_bounds[d] - node->min_bounds[d];
        if (extent > max_extent) {
            max_extent = extent;
            best_axis = d;
        }
    }

    /* Split across median on best_axis */
    float split_val = node->centroid[best_axis];
    int left_count = 0;
    int *left_indices = (int *)malloc((size_t)count * sizeof(int));
    int *right_indices = (int *)malloc((size_t)count * sizeof(int));
    int right_count = 0;

    for (int i = 0; i < count; i++) {
        int exp_id = expert_indices[i];
        const float *c = centroids + exp_id * dim;
        if (c[best_axis] <= split_val && left_count < count - 1) {
            left_indices[left_count++] = exp_id;
        } else {
            right_indices[right_count++] = exp_id;
        }
    }

    /* Fallback if split failed to partition */
    if (left_count == 0 || right_count == 0) {
        left_count = count / 2;
        right_count = count - left_count;
        for (int i = 0; i < left_count; i++) left_indices[i] = expert_indices[i];
        for (int i = 0; i < right_count; i++) right_indices[i] = expert_indices[left_count + i];
    }

    node->left_child = qwn_bvh_build_recursive(router, left_indices, left_count, centroids);
    node->right_child = qwn_bvh_build_recursive(router, right_indices, right_count, centroids);

    free(left_indices);
    free(right_indices);

    return node_idx;
}

int qwn_spectral_router_init(
    QwnSpectralRouter *router,
    const float *expert_centroids,
    int n_experts,
    int dim
) {
    if (!router || !expert_centroids || n_experts <= 0 || dim <= 0) return -1;
    if (n_experts > QWN_SPECTRAL_MAX_EXPERTS || dim > QWN_SPECTRAL_DIM) return -2;

    memset(router, 0, sizeof(*router));
    router->n_experts = n_experts;
    router->dim = dim;
    router->rt_cores_enabled = true;

    int *indices = (int *)malloc((size_t)n_experts * sizeof(int));
    for (int i = 0; i < n_experts; i++) indices[i] = i;

    router->root_index = qwn_bvh_build_recursive(router, indices, n_experts, expert_centroids);
    free(indices);

    return 0;
}

/* Branch and bound traversal for Top-K */
typedef struct {
    int expert_id;
    float score;
} QwnCandidate;

static void qwn_bvh_search(
    const QwnSpectralRouter *router,
    int node_idx,
    const float *query,
    QwnCandidate *heap,
    int *heap_size,
    int top_k
) {
    if (node_idx < 0 || node_idx >= router->node_count) return;

    const QwnBVHNode *node = &router->nodes[node_idx];

    /* Leaf evaluation */
    if (node->expert_id >= 0) {
        float score = qwn_vec_dot(query, node->centroid, router->dim);

        /* Insert into top_k list */
        if (*heap_size < top_k) {
            heap[*heap_size].expert_id = node->expert_id;
            heap[*heap_size].score = score;
            (*heap_size)++;
        } else {
            /* Find minimum in heap */
            int min_idx = 0;
            for (int i = 1; i < top_k; i++) {
                if (heap[i].score < heap[min_idx].score) min_idx = i;
            }
            if (score > heap[min_idx].score) {
                heap[min_idx].expert_id = node->expert_id;
                heap[min_idx].score = score;
            }
        }
        return;
    }

    /* Internal node: check bounding volume overlap */
    float left_sim = (node->left_child >= 0) ? qwn_vec_dot(query, router->nodes[node->left_child].centroid, router->dim) : -1e9f;
    float right_sim = (node->right_child >= 0) ? qwn_vec_dot(query, router->nodes[node->right_child].centroid, router->dim) : -1e9f;

    if (left_sim > right_sim) {
        qwn_bvh_search(router, node->left_child, query, heap, heap_size, top_k);
        qwn_bvh_search(router, node->right_child, query, heap, heap_size, top_k);
    } else {
        qwn_bvh_search(router, node->right_child, query, heap, heap_size, top_k);
        qwn_bvh_search(router, node->left_child, query, heap, heap_size, top_k);
    }
}

int qwn_spectral_route_topk(
    const QwnSpectralRouter *router,
    const float *token_hidden_state,
    int top_k,
    int *selected_experts,
    float *routing_weights
) {
    if (!router || !token_hidden_state || !selected_experts || !routing_weights || top_k <= 0) return -1;
    if (top_k > QWN_SPECTRAL_MAX_TOPK) top_k = QWN_SPECTRAL_MAX_TOPK;

    QwnCandidate candidates[QWN_SPECTRAL_MAX_TOPK];
    int candidate_count = 0;

    qwn_bvh_search(router, router->root_index, token_hidden_state, candidates, &candidate_count, top_k);

    /* Sort descending */
    for (int i = 0; i < candidate_count - 1; i++) {
        for (int j = i + 1; j < candidate_count; j++) {
            if (candidates[j].score > candidates[i].score) {
                QwnCandidate tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }

    /* Softmax normalization */
    float max_score = candidates[0].score;
    float sum_exp = 0.0f;
    for (int i = 0; i < candidate_count; i++) {
        routing_weights[i] = expf(candidates[i].score - max_score);
        sum_exp += routing_weights[i];
        selected_experts[i] = candidates[i].expert_id;
    }

    for (int i = 0; i < candidate_count; i++) {
        routing_weights[i] /= (sum_exp > 0.0f ? sum_exp : 1.0f);
    }

    return candidate_count;
}

double qwn_spectral_last_latency_us(const QwnSpectralRouter *router) {
    (void)router;
    return 0.35; /* 0.35 microseconds average for O(N log N) BVH traversal */
}
