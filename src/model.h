/* model.h - LFM2 / LFM2-MoE hybrid (short-conv + GQA attention + sparse MoE) forward pass. */
#ifndef PITHOS_MODEL_H
#define PITHOS_MODEL_H

#include <stdio.h>
#include "gguf.h"
#include "platform.h"
#include "quant.h"

typedef struct {
    uint32_t type;
    const uint8_t *data;
    int rows, cols;
    size_t row_bytes;
    size_t expert_stride;   /* bytes between experts for 3D expert tensors */
} Mat;

typedef struct {
    int is_attn, is_moe;
    int attn_idx, conv_idx;
    const float *attn_norm, *ffn_norm;
    /* short conv */
    Mat in_proj, out_proj;
    const float *conv_w;            /* [dim][l_cache] */
    /* attention */
    Mat wq, wk, wv, wo;
    const float *q_norm, *k_norm;
    /* dense ffn */
    Mat w_gate, w_up, w_down;
    /* moe */
    Mat router;                     /* [n_expert][dim], F32 */
    const float *exp_bias;          /* [n_expert] or NULL */
    Mat gate_exps, up_exps, down_exps;
} Layer;

typedef struct {
    GGUF gguf;
    char arch[32];
    int n_layer, dim, n_head, n_kv_head, head_dim, n_vocab;
    int ff_dense, ff_exp, n_expert, n_expert_used, l_cache, n_ctx_train;
    int gating;                     /* 1 softmax, 2 sigmoid */
    int norm_topk;
    float eps, rope_base;
    Mat embed, output;
    const float *final_norm;
    Layer *layers;
    int n_attn, n_conv;
    uint64_t bytes_per_token;       /* estimate of weight bytes touched per decoded token */
} Model;

typedef struct Context Context;

enum { CAT_ATTN, CAT_CONV, CAT_DENSE, CAT_ROUTER, CAT_EXP_GU, CAT_EXP_DOWN, CAT_LOGITS, N_CAT };

typedef struct {
    double t_matmul, t_attn, t_moe_route;
    double t_cat[N_CAT], bytes_cat[N_CAT];
    uint64_t calls_cat[N_CAT];
    double t_forward;
    uint64_t pred_hits, pred_total;     /* next-layer expert prediction accuracy */
    uint64_t gpu_experts;               /* expert evaluations served by the GPU */
    double t_gpu_wait;                  /* time spent waiting for GPU results */
    uint64_t expert_hits[64 * 64];  /* [layer][expert] usage counters (first 64 layers) */
} Stats;

int  model_load(Model *m, const char *path, char *err, int errlen);
void model_free(Model *m);

/* n_ctx: KV capacity in tokens; n_batch: max tokens per forward call chunk. */
Context *ctx_create(Model *m, ThreadPool *pool, int n_ctx, int n_batch);
void     ctx_free(Context *c);
int      ctx_pos(const Context *c);
int      ctx_n_ctx(const Context *c);
void     ctx_reset(Context *c);
void     ctx_set_prefetch(Context *c, int on);
struct Gpu;
void     ctx_set_gpu(Context *c, struct Gpu *g);

/* Snapshot / restore the recurrent state + position (KV entries beyond pos are ignored). */
typedef struct { int pos; float *conv; } CtxSnapshot;
void ctx_snapshot(const Context *c, CtxSnapshot *s);
void ctx_restore(Context *c, const CtxSnapshot *s);
void ctx_snapshot_free(CtxSnapshot *s);
size_t ctx_conv_bytes(const Context *c);

/* Serialize the KV cache rows [0, n) (prompt reuse across processes). The recurrent state
 * travels separately as a CtxSnapshot; ctx_kv_read does not change the position.
 * ctx_kv_read loads the first n rows of a block written with n_file rows. */
size_t ctx_kv_bytes(const Context *c, int n);
int    ctx_kv_write(const Context *c, int n, FILE *f);
int    ctx_kv_read(Context *c, int n, int n_file, FILE *f);

/* Feed tokens; returns logits (n_vocab floats, owned by ctx) for the last token,
 * or NULL if the context is full. */
float *ctx_forward(Context *c, const int *tokens, int n);
/* Like ctx_forward but writes logits for every token into out[n][n_vocab]. */
int ctx_forward_all(Context *c, const int *tokens, int n, float *out);

Stats *ctx_stats(Context *c);

#endif
