#include "qwanto_router.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void qwanto_route_lsh(const int8_t* activations, int hidden_dim, int n_experts, int top_k, int* expert_ids) {
    // 1. Calculate a 32-bit sign signature from activations
    uint32_t sig = 0;
    for (int k = 0; k < 32 && k < hidden_dim; k++) {
        if (activations[k] > 0) {
            sig |= (1u << k);
        }
    }

    // 2. Generate top_k unique expert IDs
    for (int kk = 0; kk < top_k; kk++) {
        uint32_t hash = sig ^ (kk * 0x9e3779b9);
        hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
        hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
        hash = (hash >> 16) ^ hash;
        
        int eid = (int)(hash % (uint32_t)n_experts);

        // Resolve collision to ensure unique expert IDs
        int unique = 0;
        while (!unique) {
            unique = 1;
            for (int j = 0; j < kk; j++) {
                if (expert_ids[j] == eid) {
                    eid = (eid + 1) % n_experts;
                    unique = 0;
                    break;
                }
            }
        }
        expert_ids[kk] = eid;
    }
}

int qwanto_prefetcher_init(QwantoPrefetcher* pf, int queue_depth) {
    memset(pf, 0, sizeof(*pf));
    pf->max_jobs = 16;
    if (coli_aio_init(&pf->aio_ctx, queue_depth) != 0) {
        return -1;
    }
    return 0;
}

int qwanto_prefetcher_submit(QwantoPrefetcher* pf, shards* S, int layer, int eid, 
                             void* buf_g, void* buf_u, void* buf_d) {
    if (pf->active_jobs >= pf->max_jobs) {
        return -1; // Queue full
    }

    // Locate the tensors in the shards database
    char name_g[256], name_u[256], name_d[256];
    snprintf(name_g, sizeof(name_g), "model.layers.%d.mlp.experts.%d.gate_proj.weight", layer, eid);
    snprintf(name_u, sizeof(name_u), "model.layers.%d.mlp.experts.%d.up_proj.weight", layer, eid);
    snprintf(name_d, sizeof(name_d), "model.layers.%d.mlp.experts.%d.down_proj.weight", layer, eid);

    st_tensor *tg = st_find(S, name_g);
    st_tensor *tu = st_find(S, name_u);
    st_tensor *td = st_find(S, name_d);

    if (!tg || !tu || !td) {
        return -1; // Tensors not found
    }

    // Find the next free job slot
    int job_idx = -1;
    for (int i = 0; i < pf->max_jobs; i++) {
        if (!pf->jobs[i].submitted) {
            job_idx = i;
            break;
        }
    }
    if (job_idx == -1) return -1;

    QwantoPrefetchJob *job = &pf->jobs[job_idx];
    memset(job, 0, sizeof(*job));
    job->layer = layer;
    job->eid = eid;
    job->submitted = 1;

    // Set up and submit request for gate_proj
    job->req_g.fd = tg->fd;
    job->req_g.buffer = buf_g;
    job->req_g.length = (size_t)tg->nbytes;
    job->req_g.offset = tg->off;
    job->req_g.callback = NULL;
    job->req_g.user_data = NULL;
    if (coli_aio_submit_read(&pf->aio_ctx, &job->req_g) != 0) {
        job->submitted = 0;
        return -1;
    }

    // Set up and submit request for up_proj
    job->req_u.fd = tu->fd;
    job->req_u.buffer = buf_u;
    job->req_u.length = (size_t)tu->nbytes;
    job->req_u.offset = tu->off;
    job->req_u.callback = NULL;
    job->req_u.user_data = NULL;
    if (coli_aio_submit_read(&pf->aio_ctx, &job->req_u) != 0) {
        coli_aio_cancel(&pf->aio_ctx, &job->req_g);
        job->submitted = 0;
        return -1;
    }

    // Set up and submit request for down_proj
    job->req_d.fd = td->fd;
    job->req_d.buffer = buf_d;
    job->req_d.length = (size_t)td->nbytes;
    job->req_d.offset = td->off;
    job->req_d.callback = NULL;
    job->req_d.user_data = NULL;
    if (coli_aio_submit_read(&pf->aio_ctx, &job->req_d) != 0) {
        coli_aio_cancel(&pf->aio_ctx, &job->req_g);
        coli_aio_cancel(&pf->aio_ctx, &job->req_u);
        job->submitted = 0;
        return -1;
    }

    pf->active_jobs++;
    return 0;
}

int qwanto_prefetcher_wait_all(QwantoPrefetcher* pf) {
    if (pf->active_jobs == 0) return 0;
    int res = coli_aio_wait_all(&pf->aio_ctx);
    
    // Reset jobs
    for (int i = 0; i < pf->max_jobs; i++) {
        pf->jobs[i].submitted = 0;
    }
    pf->active_jobs = 0;
    
    return res;
}

void qwanto_prefetcher_destroy(QwantoPrefetcher* pf) {
    qwanto_prefetcher_wait_all(pf);
    coli_aio_destroy(&pf->aio_ctx);
}
