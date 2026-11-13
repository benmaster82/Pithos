/* model.h - LFM2 / LFM2-MoE / LFM3 hybrid (short-conv + GQA attention + sparse MoE) forward pass. */
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
    int is_attn, is_moe, is_swa;    /* is_swa: sliding-window attention (LFM3) */
    int attn_idx, conv_idx;
    const float *attn_norm, *ffn_norm;
    const float *post_op_norm, *post_ffn_norm;  /* LFM3 sandwich norms, or NULL */
    /* short conv */
    Mat in_proj, out_proj;
    const float *conv_w;            /* [dim][l_cache] */
    /* attention */
    Mat wq, wk, wv, wo;
    const float *q_norm, *k_norm;
    Mat attn_gate;                  /* LFM3: per-head sigmoid output gate [n_head][dim], or no data */
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
    float eps, rope_base, rope_base_swa;
    const float *rope_freqs;        /* per-frequency divisors of the full-attention layers, or NULL */
    int swa_window;                 /* sliding-window size of is_swa layers (tokens, current one included) */
    int add_bos;                    /* tokenizer.ggml.add_bos_token (default 1) */
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
/* the speech encoder of an LFM3-Audio mmproj file, as a Model without vocabulary */
int  model_load_audio_encoder(Model *m, const char *path, char *err, int errlen);
/* a tensor of m's file as a matrix / as F32[n]; 0 / NULL with err if missing or of another type */
int  model_mat(Model *m, const char *name, Mat *mat, char *err, int errlen);
const float *model_f32(Model *m, const char *name, int n, char *err, int errlen);
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
struct Dx;
/* info: whether single tokens run entirely on the GPU (returns 1), or why not */
int      ctx_set_dx(Context *c, struct Dx *d, char *info, int infolen);
struct Vulkan;
/* matmuls on the Vulkan GPU; info: whether single tokens run entirely there (returns 1), or why not */
int      ctx_set_vk(Context *c, struct Vulkan *v, char *info, int infolen);

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
 * or NULL if the context is full or was lost with a failing GPU (then reset, see ctx_lost). */
float *ctx_forward(Context *c, const int *tokens, int n);
/* Like ctx_forward but writes logits for every token into out[n][n_vocab]. */
int ctx_forward_all(Context *c, const int *tokens, int n, float *out);
/* Token ids -1, -2... in ctx_forward stand for rows 0, 1... of rows[n][dim] (audio embeddings);
 * the caller keeps them alive while they are fed. NULL clears them. */
void ctx_set_embeds(Context *c, const float *rows, int n);
/* The layers over input rows x[n][dim] instead of tokens (CPU); hidden[n][dim] receives the output
 * after the final norm. Returns 0 if the context is full. */
int ctx_forward_embd(Context *c, const float *x, int n, float *hidden);
/* out[n][w->rows] = x[n][w->cols] times w, on the context's threads (w->cols a multiple of 256 if quantized) */
void ctx_matmul(Context *c, const Mat *w, const float *x, int n, float *out);
/* How many times the context was lost with a failing GPU (it then goes on on the CPU). */
int ctx_lost(const Context *c);

Stats *ctx_stats(Context *c);

#endif
