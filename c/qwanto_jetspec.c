#include "qwanto_jetspec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool qwn_jetspec_init(
    QwnJetSpecEngine *engine,
    int hidden_dim,
    int max_tree_depth,
    int max_tree_width
) {
    if (!engine) return false;
    memset(engine, 0, sizeof(*engine));

    engine->cfg.hidden_dim = hidden_dim > 0 ? hidden_dim : 2048;
    engine->cfg.max_tree_depth = (max_tree_depth > 0 && max_tree_depth <= JETSPEC_MAX_TREE_DEPTH) ? max_tree_depth : 6;
    engine->cfg.max_tree_width = (max_tree_width > 0 && max_tree_width <= 8) ? max_tree_width : 4;
    engine->cfg.top_p_threshold = 0.90f;
    engine->cfg.branch_confidence_cutoff = 0.15f;
    engine->cfg.num_draft_heads = 4;

    engine->ring_head = 0;
    engine->ring_tail = 0;
    engine->measured_acceptance_rate = 0.0;
    engine->measured_speedup_factor = 0.0;
    engine->product_enabled = false;
    engine->is_initialized = true;

    return true;
}

void qwn_jetspec_build_tree_mask(QwnJetSpecTree *tree) {
    if (!tree || tree->node_count <= 0) return;
    int n = tree->node_count;
    memset(tree->tree_mask, 0, (size_t)n * n * sizeof(uint8_t));

    for (int i = 0; i < n; i++) {
        /* Every node attends to itself */
        tree->tree_mask[i * n + i] = 1;

        /* Trace ancestors back to root */
        int curr = tree->nodes[i].parent_id;
        while (curr >= 0 && curr < n) {
            tree->tree_mask[i * n + curr] = 1;
            curr = tree->nodes[curr].parent_id;
        }
    }
}

bool qwn_jetspec_generate_tree(
    QwnJetSpecEngine *engine,
    const float *fused_hidden_states,
    const float *target_logits,
    int vocab_size
) {
    if (!engine || !engine->is_initialized || !fused_hidden_states || !target_logits || vocab_size <= 0) {
        return false;
    }

    QwnJetSpecTree *tree = &engine->active_tree;
    memset(tree, 0, sizeof(*tree));

    /* This fixture path ranks tokens from supplied logits.  It is deliberately
     * reference-only: a production caller must provide proposals from a
     * compatible QWN draft model or validated MTP heads. */
    int root_tok = 0;
    float max_logit = target_logits[0];
    for (int v = 1; v < vocab_size; v++) {
        if (target_logits[v] > max_logit) {
            max_logit = target_logits[v];
            root_tok = v;
        }
    }

    /* Initialize Root Node (Depth 0) */
    tree->nodes[0].node_id = 0;
    tree->nodes[0].token_id = root_tok;
    tree->nodes[0].parent_id = -1;
    tree->nodes[0].depth = 0;
    tree->nodes[0].cumulative_score = max_logit;
    tree->nodes[0].child_count = 0;
    tree->node_count = 1;

    int current_nodes[16];
    int current_count = 1;
    current_nodes[0] = 0;

    /* Expand a deterministic ranked-logit fixture.  Tokens are selected from
     * actual supplied logits and duplicates are removed; no token arithmetic
     * or fabricated confidence is permitted. */
    for (int depth = 1; depth < engine->cfg.max_tree_depth; depth++) {
        int next_nodes[16];
        int next_count = 0;

        for (int i = 0; i < current_count; i++) {
            int parent_idx = current_nodes[i];
            int branches = (depth == 1) ? engine->cfg.max_tree_width : 2;

            for (int b = 0; b < branches; b++) {
                if (tree->node_count >= JETSPEC_MAX_TREE_NODES) break;
                int draft_tok = -1;
                float draft_score = -INFINITY;
                for (int v = 0; v < vocab_size; v++) {
                    int duplicate = 0;
                    for (int n = 0; n < tree->node_count; n++) {
                        if (tree->nodes[n].token_id == v) { duplicate = 1; break; }
                    }
                    if (!duplicate && target_logits[v] > draft_score) {
                        draft_tok = v;
                        draft_score = target_logits[v];
                    }
                }
                if (draft_tok < 0) break;
                int node_id = tree->node_count++;

                tree->nodes[node_id].node_id = node_id;
                tree->nodes[node_id].token_id = draft_tok;
                tree->nodes[node_id].parent_id = parent_idx;
                tree->nodes[node_id].depth = depth;
                tree->nodes[node_id].cumulative_score =
                    tree->nodes[parent_idx].cumulative_score + draft_score;
                tree->nodes[node_id].child_count = 0;

                if (tree->nodes[parent_idx].child_count < 8) {
                    tree->nodes[parent_idx].children[tree->nodes[parent_idx].child_count++] = node_id;
                }

                if (next_count < 16) {
                    next_nodes[next_count++] = node_id;
                }
            }
        }
        if (next_count == 0) break;
        current_count = next_count;
        memcpy(current_nodes, next_nodes, (size_t)current_count * sizeof(int));
    }

    /* Build 2D Tree-Causal Attention Mask */
    qwn_jetspec_build_tree_mask(tree);

    /* Populate Rank-1 Best Path */
    tree->best_path_length = 0;
    int curr = 0;
    while (curr >= 0 && tree->best_path_length < JETSPEC_MAX_TREE_DEPTH) {
        tree->best_path_tokens[tree->best_path_length++] = tree->nodes[curr].token_id;
        if (tree->nodes[curr].child_count > 0) {
            curr = tree->nodes[curr].children[0];
        } else {
            break;
        }
    }
    tree->best_path_score = tree->nodes[curr >= 0 ? curr : 0].cumulative_score;
    engine->tree_nodes_proposed += (uint64_t)tree->node_count;
    engine->wasted_nodes += tree->node_count > tree->best_path_length ?
                            (uint64_t)(tree->node_count - tree->best_path_length) : 0;
    return true;
}

int qwn_jetspec_verify_tree(
    QwnJetSpecEngine *engine,
    const float *target_verification_logits,
    int vocab_size,
    int *accepted_tokens_out,
    int max_out_tokens
) {
    if (!engine || !engine->is_initialized || !target_verification_logits || !accepted_tokens_out || max_out_tokens <= 0) {
        return 0;
    }

    QwnJetSpecTree *tree = &engine->active_tree;
    if (tree->node_count <= 0) return 0;

    int accepted_count = 0;
    int curr_node = 0;

    while (curr_node >= 0 && curr_node < tree->node_count && accepted_count < max_out_tokens) {
        int candidate_tok = tree->nodes[curr_node].token_id;
        const float *row = target_verification_logits +
                           (size_t)curr_node * (size_t)vocab_size;
        int target_best = 0;
        for (int v = 1; v < vocab_size; v++)
            if (row[v] > row[target_best]) target_best = v;
        engine->branches_verified++;
        if (target_best != candidate_tok) break;
        accepted_tokens_out[accepted_count++] = candidate_tok;

        if (tree->nodes[curr_node].child_count > 0) {
            curr_node = tree->nodes[curr_node].children[0];
        } else {
            break;
        }
    }

    engine->total_draft_tokens += tree->node_count;
    engine->total_accepted_tokens += accepted_count;
    if (engine->total_draft_tokens > 0) {
        engine->measured_acceptance_rate = (double)engine->total_accepted_tokens / (double)engine->total_draft_tokens;
    }
    engine->accepted_path_tokens += (uint64_t)accepted_count;
    return accepted_count;
}

bool qwn_jetspec_cache_lookup(
    QwnJetSpecEngine *engine,
    uint64_t prompt_hash,
    int *tokens_out,
    int *chain_len_out
) {
    if (!engine || !engine->is_initialized || !tokens_out || !chain_len_out) return false;

    uint32_t idx = (uint32_t)(prompt_hash % JETSPEC_CACHE_SIZE);
    if (engine->cache[idx].prompt_hash == prompt_hash && engine->cache[idx].chain_len > 0) {
        *chain_len_out = engine->cache[idx].chain_len;
        memcpy(tokens_out, engine->cache[idx].token_chain, (size_t)engine->cache[idx].chain_len * sizeof(int));
        return true;
    }
    return false;
}

void qwn_jetspec_record_acceptance(
    QwnJetSpecEngine *engine,
    uint64_t prompt_hash,
    const int *accepted_tokens,
    int accepted_count
) {
    if (!engine || !engine->is_initialized || !accepted_tokens || accepted_count <= 0) return;

    uint32_t idx = (uint32_t)(prompt_hash % JETSPEC_CACHE_SIZE);
    engine->cache[idx].prompt_hash = prompt_hash;
    engine->cache[idx].chain_len = accepted_count < JETSPEC_MAX_TREE_DEPTH ? accepted_count : JETSPEC_MAX_TREE_DEPTH;
    memcpy(engine->cache[idx].token_chain, accepted_tokens, (size_t)engine->cache[idx].chain_len * sizeof(int));
    /* Confidence is unavailable here because the API receives accepted IDs,
     * not their probabilities.  Never present a fabricated confidence. */
    engine->cache[idx].confidence = 0.0f;

    /* Push into ring buffer */
    for (int i = 0; i < accepted_count; i++) {
        engine->ring_buffer[engine->ring_head] = accepted_tokens[i];
        engine->ring_head = (engine->ring_head + 1) % JETSPEC_RING_BUFFER_SLOTS;
    }
}

void qwn_jetspec_reset(QwnJetSpecEngine *engine) {
    if (!engine) return;
    memset(&engine->active_tree, 0, sizeof(engine->active_tree));
    engine->ring_head = 0;
    engine->ring_tail = 0;
}
