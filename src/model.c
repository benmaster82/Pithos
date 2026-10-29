/* model.c - LFM2 / LFM2-MoE / LFM3 inference on CPU.
 *
 * Layer l:  h = rmsnorm(x);  x += is_attn ? attention(h) : shortconv(h)
 *           h = rmsnorm(x);  x += is_moe ? moe(h) : swiglu_ffn(h)
 * Final:    logits = output(rmsnorm(x, token_embd_norm))
 *
 * shortconv: [B,C,xx] = in_proj(h); bx = B*xx; y = C * causal_depthwise_conv(bx); out_proj(y)
 * attention: GQA with per-head RMSNorm on q and k, NEOX RoPE.
 * moe:       sigmoid router (+ selection bias), top-k, normalized weights, SwiGLU experts.
 *
 * LFM3 adds, each one only where the model has it: an RMSNorm on both block outputs before
 * they enter the residual stream (sandwich norm), a sigmoid gate per head on the attention
 * output (computed from h), sliding-window attention on some layers with their own RoPE base,
 * and partial rotation on the full-attention layers (rope_freqs divisors). */
#include "model.h"
#include "gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#define PREFETCH(p) __builtin_prefetch((p), 0, 3)
#endif

#define MAX_JOBS 160
#define MAX_TASKS 32768
#define TOK_TILE 16

/* ------------------------------------------------------------------ loading */

static int load_mat(Model *m, Mat *mat, const char *name, int required, char *err, int errlen) {
    memset(mat, 0, sizeof(*mat));
    const GGUFTensor *t = gguf_tensor(&m->gguf, name);
    if (!t) {
        if (required) snprintf(err, errlen, "missing tensor %s", name);
        return 0;
    }
    if (!type_supported(t->type)) {
        snprintf(err, errlen, "tensor %s: unsupported type %s", name, ggml_type_name(t->type));
        return -1;
    }
    mat->type = t->type;
    mat->data = t->data;
    mat->cols = (int)t->ne[0];
    mat->rows = (int)t->ne[1];
    mat->row_bytes = row_bytes(t->type, mat->cols);
    mat->expert_stride = mat->row_bytes * (size_t)mat->rows;
    if (t->type != GGML_F32 && t->type != GGML_F16 && (mat->cols % QK_K) != 0) {
        snprintf(err, errlen, "tensor %s: row length %d not a multiple of 256", name, mat->cols);
        return -1;
    }
    return 1;
}

static const float *load_f32(Model *m, const char *name, int n, int required, char *err, int errlen) {
    const GGUFTensor *t = gguf_tensor(&m->gguf, name);
    if (!t) {
        if (required) snprintf(err, errlen, "missing tensor %s", name);
        return NULL;
    }
    if (t->type != GGML_F32 || (n > 0 && (int64_t)n != t->ne[0] * t->ne[1] * t->ne[2])) {
        snprintf(err, errlen, "tensor %s: expected F32[%d]", name, n);
        return NULL;
    }
    return (const float *)t->data;
}

#define REQ(expr) do { if ((expr) <= 0) goto fail; } while (0)
#define REQP(expr) do { if (!(expr)) goto fail; } while (0)

int model_load(Model *m, const char *path, char *err, int errlen) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gguf, path, err, errlen)) return 0;
    GGUF *g = &m->gguf;
    uint64_t al = 0;
    const char *arch = gguf_get_str(g, "general.architecture", &al);
    if (!arch || al >= sizeof(m->arch)) { snprintf(err, errlen, "no architecture"); goto fail; }
    memcpy(m->arch, arch, al);
    m->arch[al] = 0;
    if (strcmp(m->arch, "lfm2") && strcmp(m->arch, "lfm2moe") && strcmp(m->arch, "lfm3")) {
        snprintf(err, errlen, "unsupported architecture '%s' (need lfm2 / lfm2moe / lfm3)", m->arch);
        goto fail;
    }
    char key[128];
#define KEY(s) (snprintf(key, sizeof(key), "%s." s, m->arch), key)
    m->n_layer = (int)gguf_get_int(g, KEY("block_count"), 0);
    m->dim = (int)gguf_get_int(g, KEY("embedding_length"), 0);
    m->n_head = (int)gguf_get_int(g, KEY("attention.head_count"), 0);
    m->ff_dense = (int)gguf_get_int(g, KEY("feed_forward_length"), 0);
    m->ff_exp = (int)gguf_get_int(g, KEY("expert_feed_forward_length"), 0);
    m->n_expert = (int)gguf_get_int(g, KEY("expert_count"), 0);
    m->n_expert_used = (int)gguf_get_int(g, KEY("expert_used_count"), 0);
    m->l_cache = (int)gguf_get_int(g, KEY("shortconv.l_cache"), 3);
    m->n_ctx_train = (int)gguf_get_int(g, KEY("context_length"), 4096);
    m->gating = (int)gguf_get_int(g, KEY("expert_gating_func"), 2);
    m->norm_topk = 1;
    m->eps = (float)gguf_get_float(g, KEY("attention.layer_norm_rms_epsilon"), 1e-5);
    m->rope_base = (float)gguf_get_float(g, KEY("rope.freq_base"), 1e6);
    m->rope_base_swa = (float)gguf_get_float(g, KEY("rope.freq_base_swa"), m->rope_base);
    m->swa_window = (int)gguf_get_int(g, KEY("attention.sliding_window"), 0);
    const GGUFKV *kvh = gguf_find(g, KEY("attention.head_count_kv"));
    const GGUFKV *swa = gguf_find(g, KEY("attention.sliding_window_pattern"));
    const int key_len = (int)gguf_get_int(g, KEY("attention.key_length"), 0);
#undef KEY
    m->add_bos = (int)gguf_get_int(g, "tokenizer.ggml.add_bos_token", 1);
    if (m->n_layer <= 0 || m->dim <= 0 || m->n_head <= 0 || !kvh) { snprintf(err, errlen, "bad hyperparameters"); goto fail; }
    m->head_dim = key_len > 0 ? key_len : m->dim / m->n_head;
    if (m->head_dim > 256 || m->head_dim % 2) { snprintf(err, errlen, "head size %d not supported", m->head_dim); goto fail; }
    if (m->n_expert > 64) { snprintf(err, errlen, "too many experts (%d)", m->n_expert); goto fail; }

    REQ(load_mat(m, &m->embed, "token_embd.weight", 1, err, errlen));
    m->n_vocab = m->embed.rows;
    if (load_mat(m, &m->output, "output.weight", 0, err, errlen) < 0) goto fail;
    if (!m->output.data) m->output = m->embed;
    m->final_norm = load_f32(m, "token_embd_norm.weight", m->dim, 0, err, errlen);
    if (!m->final_norm) REQP(m->final_norm = load_f32(m, "output_norm.weight", m->dim, 1, err, errlen));
    if (gguf_tensor(g, "rope_freqs.weight"))
        REQP(m->rope_freqs = load_f32(m, "rope_freqs.weight", m->head_dim / 2, 1, err, errlen));

    m->layers = (Layer *)calloc((size_t)m->n_layer, sizeof(Layer));
    char nm[128];
#define T(s) (snprintf(nm, sizeof(nm), "blk.%d." s, l), nm)
    for (int l = 0; l < m->n_layer; l++) {
        Layer *L = &m->layers[l];
        int nkv = kvh->type == GGUF_T_ARRAY ? (int)gguf_arr_int(kvh, (uint64_t)l) : (int)gguf_get_int(g, "x", 0);
        L->is_attn = nkv > 0;
        L->is_swa = L->is_attn && swa && swa->type == GGUF_T_ARRAY && gguf_arr_int(swa, (uint64_t)l) != 0;
        if (L->is_swa && m->swa_window <= 0) { snprintf(err, errlen, "sliding-window layer without a window size"); goto fail; }
        REQP(L->attn_norm = load_f32(m, T("attn_norm.weight"), m->dim, 1, err, errlen));
        REQP(L->ffn_norm = load_f32(m, T("ffn_norm.weight"), m->dim, 1, err, errlen));
        /* sandwich norms: both or neither */
        L->post_op_norm = load_f32(m, T("post_attention_norm.weight"), m->dim, 0, err, errlen);
        L->post_ffn_norm = load_f32(m, T("post_ffw_norm.weight"), m->dim, 0, err, errlen);
        if (!L->post_op_norm != !L->post_ffn_norm) { snprintf(err, errlen, "layer %d: only one post norm", l); goto fail; }
        if (L->is_attn) {
            m->n_kv_head = nkv;
            L->attn_idx = m->n_attn++;
            REQ(load_mat(m, &L->wq, T("attn_q.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wk, T("attn_k.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wv, T("attn_v.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wo, T("attn_output.weight"), 1, err, errlen));
            REQP(L->q_norm = load_f32(m, T("attn_q_norm.weight"), m->head_dim, 1, err, errlen));
            REQP(L->k_norm = load_f32(m, T("attn_k_norm.weight"), m->head_dim, 1, err, errlen));
            if (load_mat(m, &L->attn_gate, T("attn_gate.weight"), 0, err, errlen) < 0) goto fail;
            if (L->attn_gate.data && (L->attn_gate.rows != m->n_head || L->attn_gate.cols != m->dim)) {
                snprintf(err, errlen, "layer %d: attn_gate must be [%d][%d]", l, m->n_head, m->dim);
                goto fail;
            }
        } else {
            L->conv_idx = m->n_conv++;
            REQ(load_mat(m, &L->in_proj, T("shortconv.in_proj.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->out_proj, T("shortconv.out_proj.weight"), 1, err, errlen));
            REQP(L->conv_w = load_f32(m, T("shortconv.conv.weight"), m->dim * m->l_cache, 1, err, errlen));
        }
        int r = load_mat(m, &L->router, T("ffn_gate_inp.weight"), 0, err, errlen);
        if (r < 0) goto fail;
        if (r > 0) {
            L->is_moe = 1;
            if (L->router.type != GGML_F32 && L->router.type != GGML_F16) { snprintf(err, errlen, "router must be F32/F16"); goto fail; }
            L->exp_bias = load_f32(m, T("exp_probs_b.bias"), m->n_expert, 0, err, errlen);
            REQ(load_mat(m, &L->gate_exps, T("ffn_gate_exps.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->up_exps, T("ffn_up_exps.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->down_exps, T("ffn_down_exps.weight"), 1, err, errlen));
        } else {
            REQ(load_mat(m, &L->w_gate, T("ffn_gate.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->w_up, T("ffn_up.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->w_down, T("ffn_down.weight"), 1, err, errlen));
        }
    }
#undef T
    /* bytes touched per decoded token (weights) */
    uint64_t b = (uint64_t)m->output.row_bytes * m->n_vocab;
    for (int l = 0; l < m->n_layer; l++) {
        Layer *L = &m->layers[l];
        if (L->is_attn) b += (uint64_t)L->wq.expert_stride + L->wk.expert_stride + L->wv.expert_stride + L->wo.expert_stride +
                             L->attn_gate.expert_stride;
        else b += (uint64_t)L->in_proj.expert_stride + L->out_proj.expert_stride;
        if (L->is_moe) b += (uint64_t)m->n_expert_used * (L->gate_exps.expert_stride + L->up_exps.expert_stride + L->down_exps.expert_stride);
        else b += (uint64_t)L->w_gate.expert_stride + L->w_up.expert_stride + L->w_down.expert_stride;
    }
    m->bytes_per_token = b;
    if (!m->n_kv_head) m->n_kv_head = m->n_head;
    return 1;
fail:
    model_free(m);
    return 0;
}

void model_free(Model *m) {
    free(m->layers);
    gguf_close(&m->gguf);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ context */

typedef struct {
    uint32_t type;
    const uint8_t *w;
    size_t row_bytes;
    int rows, K;
    const block_q8_K *xq; int xstride;
    const float *xf; int xfstride;
    int nt;
    float *out; int ostride;
} MatJob;

struct Context {
    Model *m;
    ThreadPool *pool;
    int n_threads, n_ctx, n_batch, pos, prefetch_rows;
    Prefetcher *pfq;            /* background read-ahead (NULL = off) */
    Gpu *gpu;                   /* optional expert co-processor (NULL = CPU only) */
    /* dense decode path: single tokens run on the GPU, which keeps its own copy of the KV cache
     * and of the convolution state. Rows [0, kv_cpu_n) / [0, kv_gpu_n) are valid on each side. */
    int gpu_dense, kv_cpu_n, kv_gpu_n, conv_cpu_ok, conv_gpu_ok;
    int *on_gpu, *ga_e, *ga_t;  /* [n_batch * K] GPU assignment bookkeeping */
    float *gy;                  /* [n_batch * K][dim] GPU expert outputs */
    int pred[64], pred_n, pred_layer;
    uint32_t tok_counter;       /* tokens processed since creation (never reset) */
    uint32_t last_used[64 * 64];/* [layer][expert] -> tok_counter at last use */
    f16_t *kc, *vc;
    float *conv;
    float inv_freq[256], inv_freq_swa[256];
    float *gate;                /* [n_batch][n_head] attention output gates (LFM3) */
    /* scratch (sized for n_batch tokens) */
    float *x, *h, *big, *tmp, *out, *logits;
    block_q8_K *hq, *tq;
    float *fg, *fu; block_q8_K *fq;
    float *rlog, *selw; int *sel;
    int *a_tok, *a_slot, *tok_a;
    block_q8_K *xg; float *eg, *eu; block_q8_K *eq; float *eo;
    float *scores;   /* [n_threads][n_ctx] */
    /* matmul scheduling */
    MatJob jobs[MAX_JOBS]; int njobs;
    int *task_job, *task_r0, *task_r1; int ntasks;
    Stats stats;
};

static void *amalloc(size_t n) {
    void *p = aligned_malloc(n, 64);
    if (p) memset(p, 0, n);
    return p;
}

Context *ctx_create(Model *m, ThreadPool *pool, int n_ctx, int n_batch) {
    Context *c = (Context *)calloc(1, sizeof(Context));
    c->m = m;
    c->pool = pool;
    c->n_threads = pool_size(pool);
    c->n_ctx = n_ctx;
    c->n_batch = n_batch < 1 ? 1 : n_batch;
    c->prefetch_rows = getenv("PITHOS_PF") ? atoi(getenv("PITHOS_PF")) : 4;
    c->tok_counter = 1;
    c->pred_layer = -1;
    int B = c->n_batch, D = m->dim, kvd = m->n_kv_head * m->head_dim;
    int nbd = D / QK_K;
    size_t kvn = (size_t)m->n_attn * n_ctx * kvd;
    c->kc = (f16_t *)aligned_malloc(kvn * sizeof(f16_t) + 64, 64);
    c->vc = (f16_t *)aligned_malloc(kvn * sizeof(f16_t) + 64, 64);
    c->conv = (float *)amalloc(sizeof(float) * (size_t)(m->n_conv ? m->n_conv : 1) * (m->l_cache - 1) * D);
    for (int i = 0; i < m->head_dim / 2; i++) {
        c->inv_freq[i] = (float)pow((double)m->rope_base, -2.0 * i / m->head_dim);
        if (m->rope_freqs) c->inv_freq[i] /= m->rope_freqs[i]; /* a huge divisor leaves the pair unrotated */
        c->inv_freq_swa[i] = (float)pow((double)m->rope_base_swa, -2.0 * i / m->head_dim);
    }
    int bigw = 3 * D;
    if (D + 2 * kvd > bigw) bigw = D + 2 * kvd;
    c->x = (float *)amalloc(sizeof(float) * B * D);
    c->h = (float *)amalloc(sizeof(float) * B * D);
    c->big = (float *)amalloc(sizeof(float) * B * bigw);
    c->tmp = (float *)amalloc(sizeof(float) * B * D);
    c->out = (float *)amalloc(sizeof(float) * B * D);
    c->gate = (float *)amalloc(sizeof(float) * B * m->n_head);
    c->logits = (float *)amalloc(sizeof(float) * m->n_vocab);
    c->hq = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * nbd);
    c->tq = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * nbd);
    int ffd = m->ff_dense > 0 ? m->ff_dense : QK_K;
    c->fg = (float *)amalloc(sizeof(float) * B * ffd);
    c->fu = (float *)amalloc(sizeof(float) * B * ffd);
    c->fq = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * (ffd / QK_K));
    int ne = m->n_expert ? m->n_expert : 1, k = m->n_expert_used ? m->n_expert_used : 1;
    int ffe = m->ff_exp > 0 ? m->ff_exp : QK_K;
    c->rlog = (float *)amalloc(sizeof(float) * B * ne);
    c->selw = (float *)amalloc(sizeof(float) * B * k);
    c->sel = (int *)amalloc(sizeof(int) * B * k);
    c->a_tok = (int *)amalloc(sizeof(int) * B * k);
    c->a_slot = (int *)amalloc(sizeof(int) * B * k);
    c->tok_a = (int *)amalloc(sizeof(int) * B * k);
    c->xg = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * k * nbd);
    c->eg = (float *)amalloc(sizeof(float) * B * k * ffe);
    c->eu = (float *)amalloc(sizeof(float) * B * k * ffe);
    c->eq = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * k * (ffe / QK_K));
    c->eo = (float *)amalloc(sizeof(float) * B * k * D);
    c->gy = (float *)amalloc(sizeof(float) * B * k * D);
    c->on_gpu = (int *)amalloc(sizeof(int) * B * k);
    c->ga_e = (int *)amalloc(sizeof(int) * B * k);
    c->ga_t = (int *)amalloc(sizeof(int) * B * k);
    c->scores = (float *)amalloc(sizeof(float) * c->n_threads * (size_t)n_ctx);
    c->task_job = (int *)malloc(sizeof(int) * MAX_TASKS);
    c->task_r0 = (int *)malloc(sizeof(int) * MAX_TASKS);
    c->task_r1 = (int *)malloc(sizeof(int) * MAX_TASKS);
    if (!c->kc || !c->vc || !c->scores || !c->eo) { ctx_free(c); return NULL; }
    return c;
}

void ctx_free(Context *c) {
    if (!c) return;
    if (c->pfq) prefetcher_destroy(c->pfq);
    void *a[] = { c->kc, c->vc, c->conv, c->x, c->h, c->big, c->tmp, c->out, c->gate, c->logits, c->hq, c->tq,
                  c->fg, c->fu, c->fq, c->rlog, c->selw, c->sel, c->a_tok, c->a_slot, c->tok_a, c->xg,
                  c->eg, c->eu, c->eq, c->eo, c->scores, c->gy, c->on_gpu, c->ga_e, c->ga_t };
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); i++) aligned_free(a[i]);
    free(c->task_job); free(c->task_r0); free(c->task_r1);
    free(c);
}

void ctx_set_gpu(Context *c, struct Gpu *g) {
    c->gpu = g;
    const char *env = getenv("PITHOS_GPU_DENSE");
    c->gpu_dense = g && gpu_dense_on(g) && !(env && env[0] == '0');
    c->kv_gpu_n = 0;
    c->conv_gpu_ok = 0;
}

/* make the host copy of the state current (before CPU work or a snapshot) */
static void sync_to_cpu(Context *c, int kv_rows) {
    if (!c->gpu_dense) return;
    if (c->kv_cpu_n < kv_rows) {
        gpu_dense_kv(c->gpu, c->kc, c->vc, c->kv_cpu_n, kv_rows, 0);
        c->kv_cpu_n = kv_rows;
    }
    if (!c->conv_cpu_ok) {
        gpu_dense_conv(c->gpu, c->conv, 0);
        c->conv_cpu_ok = 1;
    }
}
int ctx_pos(const Context *c) { return c->pos; }
int ctx_n_ctx(const Context *c) { return c->n_ctx; }
void ctx_set_prefetch(Context *c, int on) {
    if (on && !c->pfq) c->pfq = prefetcher_create();
    if (!on && c->pfq) { prefetcher_destroy(c->pfq); c->pfq = NULL; }
    c->pred_layer = -1;
}
Stats *ctx_stats(Context *c) { return &c->stats; }

void ctx_reset(Context *c) {
    c->pos = 0;
    c->kv_cpu_n = c->kv_gpu_n = 0;
    c->conv_cpu_ok = 1;
    c->conv_gpu_ok = 0;
    memset(c->conv, 0, sizeof(float) * (size_t)(c->m->n_conv ? c->m->n_conv : 1) * (c->m->l_cache - 1) * c->m->dim);
}

static size_t conv_state_size(const Context *c) {
    return sizeof(float) * (size_t)(c->m->n_conv ? c->m->n_conv : 1) * (c->m->l_cache - 1) * c->m->dim;
}

void ctx_snapshot(const Context *cc, CtxSnapshot *s) {
    Context *c = (Context *)cc;
    sync_to_cpu(c, c->pos);
    s->pos = c->pos;
    s->conv = (float *)malloc(conv_state_size(c));
    memcpy(s->conv, c->conv, conv_state_size(c));
}

void ctx_restore(Context *c, const CtxSnapshot *s) {
    c->pos = s->pos;
    if (c->kv_cpu_n > s->pos) c->kv_cpu_n = s->pos;
    if (c->kv_gpu_n > s->pos) c->kv_gpu_n = s->pos;
    c->conv_cpu_ok = 1;
    c->conv_gpu_ok = 0;
    memcpy(c->conv, s->conv, conv_state_size(c));
}

void ctx_snapshot_free(CtxSnapshot *s) {
    free(s->conv);
    s->conv = NULL;
}

size_t ctx_conv_bytes(const Context *c) { return conv_state_size(c); }

size_t ctx_kv_bytes(const Context *c, int n) {
    return 2 * sizeof(f16_t) * (size_t)c->m->n_attn * n * c->m->n_kv_head * c->m->head_dim;
}

/* KV rows [0, n) of every attention layer: all K, then all V. */
int ctx_kv_write(const Context *cc, int n, FILE *f) {
    Context *c = (Context *)cc;
    sync_to_cpu(c, n);
    const size_t kvd = (size_t)c->m->n_kv_head * c->m->head_dim;
    for (int w = 0; w < 2; w++)
        for (int l = 0; l < c->m->n_attn; l++) {
            const f16_t *src = (w ? c->vc : c->kc) + (size_t)l * c->n_ctx * kvd;
            if (fwrite(src, sizeof(f16_t) * kvd, (size_t)n, f) != (size_t)n) return 0;
        }
    return 1;
}

int ctx_kv_read(Context *c, int n, int n_file, FILE *f) {
    if (n > c->n_ctx || n > n_file) return 0;
    const size_t kvd = (size_t)c->m->n_kv_head * c->m->head_dim;
    for (int w = 0; w < 2; w++)
        for (int l = 0; l < c->m->n_attn; l++) {
            f16_t *dst = (w ? c->vc : c->kc) + (size_t)l * c->n_ctx * kvd;
            if (fread(dst, sizeof(f16_t) * kvd, (size_t)n, f) != (size_t)n) return 0;
            if (n < n_file && fseek(f, (long)(sizeof(f16_t) * kvd * (size_t)(n_file - n)), SEEK_CUR)) return 0;
        }
    c->kv_cpu_n = n;
    c->kv_gpu_n = 0;
    return 1;
}

/* ------------------------------------------------------------------ parallel matmul */

static void job_clear(Context *c) { c->njobs = 0; }

static void job_add_q(Context *c, const Mat *w, const uint8_t *wdata, const block_q8_K *xq, int xstride, int nt,
                      float *out, int ostride) {
    MatJob *j = &c->jobs[c->njobs++];
    j->type = w->type;
    j->w = wdata ? wdata : w->data;
    j->row_bytes = w->row_bytes;
    j->rows = w->rows;
    j->K = w->cols;
    j->xq = xq; j->xstride = xstride;
    j->xf = NULL; j->xfstride = 0;
    j->nt = nt;
    j->out = out; j->ostride = ostride;
}

static void job_add_f(Context *c, const Mat *w, const float *xf, int xfstride, int nt, float *out, int ostride) {
    MatJob *j = &c->jobs[c->njobs++];
    j->type = w->type;
    j->w = w->data;
    j->row_bytes = w->row_bytes;
    j->rows = w->rows;
    j->K = w->cols;
    j->xq = NULL; j->xstride = 0;
    j->xf = xf; j->xfstride = xfstride;
    j->nt = nt;
    j->out = out; j->ostride = ostride;
}

static void mm_task(void *ctx, int task, int tid) {
    (void)tid;
    Context *c = (Context *)ctx;
    const MatJob *j = &c->jobs[c->task_job[task]];
    const int r0 = c->task_r0[task], r1 = c->task_r1[task];
    if (j->type == GGML_F32 || j->type == GGML_F16) {
        for (int r = r0; r < r1; r++) {
            const uint8_t *wr = j->w + (size_t)r * j->row_bytes;
            for (int t = 0; t < j->nt; t++) {
                const float *xv = j->xf + (size_t)t * j->xfstride;
                j->out[(size_t)t * j->ostride + r] = j->type == GGML_F32 ? KS.dot_f32((const float *)wr, xv, j->K)
                                                                         : KS.dot_f16_f32((const f16_t *)wr, xv, j->K);
            }
        }
        return;
    }
    dot_rows_fn fn = KS.dot[j->type];
    for (int t0 = 0; t0 < j->nt; t0 += TOK_TILE) {
        const int tn = j->nt - t0 < TOK_TILE ? j->nt - t0 : TOK_TILE;
        const block_q8_K *xq = j->xq + (size_t)t0 * j->xstride;
        float *o = j->out + (size_t)t0 * j->ostride;
        for (int r = r0; r < r1; r++) {
            if (c->prefetch_rows && r + c->prefetch_rows < r1) {
                /* software prefetch of a row ahead: keeps more memory requests in flight */
                const char *pf = (const char *)j->w + (size_t)(r + c->prefetch_rows) * j->row_bytes;
                for (size_t off = 0; off < j->row_bytes; off += 64) PREFETCH(pf + off);
            }
            fn(j->K, j->w + (size_t)r * j->row_bytes, xq, j->xstride, tn, o + r, j->ostride);
        }
    }
}

static void job_run(Context *c, int cat) {
    double t0 = time_sec();
    int nt = 0;
    for (int ji = 0; ji < c->njobs; ji++) {
        const MatJob *j = &c->jobs[ji];
        /* ~48 KB of weights per task, at least 4 rows */
        int chunk = (int)(49152 / (j->row_bytes ? j->row_bytes : 1));
        if (j->nt > 1) chunk = chunk * 2 / (j->nt < 4 ? j->nt : 4);
        if (chunk < 4) chunk = 4;
        if (chunk > 512) chunk = 512;
        for (int r = 0; r < j->rows && nt < MAX_TASKS; r += chunk) {
            c->task_job[nt] = ji;
            c->task_r0[nt] = r;
            c->task_r1[nt] = r + chunk < j->rows ? r + chunk : j->rows;
            nt++;
        }
    }
    c->ntasks = nt;
    pool_run(c->pool, mm_task, c, nt);
    const double dt = time_sec() - t0;
    c->stats.t_matmul += dt;
    c->stats.t_cat[cat] += dt;
    for (int ji = 0; ji < c->njobs; ji++) c->stats.bytes_cat[cat] += (double)c->jobs[ji].rows * c->jobs[ji].row_bytes;
    c->stats.calls_cat[cat]++;
}

/* ------------------------------------------------------------------ small ops */

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) o[i] = x[i] * s * w[i];
}

static void quantize_rows(const float *x, int n, int K, block_q8_K *q) {
    for (int t = 0; t < n; t++) KS.quantize_q8_K(x + (size_t)t * K, q + (size_t)t * (K / QK_K), K);
}


static void rope_neox(float *v, int hd, int pos, const float *inv_freq) {
    const int half = hd / 2;
    for (int i = 0; i < half; i++) {
        const double a = (double)pos * inv_freq[i];
        const float cs = (float)cos(a), sn = (float)sin(a);
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * cs - x1 * sn;
        v[i + half] = x0 * sn + x1 * cs;
    }
}

/* ------------------------------------------------------------------ attention */

typedef struct {
    Context *c;
    const Layer *L;
    int n, pos0;
} AttnArgs;

static void attn_task(void *ctx, int task, int tid) {
    AttnArgs *a = (AttnArgs *)ctx;
    Context *c = a->c;
    const Model *m = c->m;
    const int hd = m->head_dim, nh = m->n_head, kvd = m->n_kv_head * hd;
    const int t = task / nh, h = task % nh;
    const int kvh = h / (nh / m->n_kv_head);
    const int qkvw = m->dim + 2 * kvd;
    const float *q = c->big + (size_t)t * qkvw + h * hd;
    const int npos = a->pos0 + t + 1;
    /* sliding-window layers see the last swa_window positions, the current one included */
    const int j0 = a->L->is_swa && npos > m->swa_window ? npos - m->swa_window : 0;
    const size_t lbase = (size_t)a->L->attn_idx * c->n_ctx * kvd;
    const f16_t *kc = c->kc + lbase + kvh * hd;
    const f16_t *vc = c->vc + lbase + kvh * hd;
    float *s = c->scores + (size_t)tid * c->n_ctx;
    const float scale = 1.f / sqrtf((float)hd);
    float mx = -1e30f;
    for (int j = j0; j < npos; j++) {
        const float v = KS.dot_f16_f32(kc + (size_t)j * kvd, q, hd) * scale;
        s[j - j0] = v;
        if (v > mx) mx = v;
    }
    const float sum = KS.exp_sum(s, npos - j0, mx);
    float inv = 1.f / sum;
    if (a->L->attn_gate.data) inv *= c->gate[(size_t)t * nh + h]; /* LFM3 output gate */
    float *o = c->tmp + (size_t)t * m->dim + h * hd;
    float acc[512];
    memset(acc, 0, sizeof(float) * hd);
    for (int j = j0; j < npos; j++) KS.axpy_f16(s[j - j0] * inv, vc + (size_t)j * kvd, acc, hd);
    memcpy(o, acc, sizeof(float) * hd);
}

static void attention_block(Context *c, const Layer *L, int n) {
    const Model *m = c->m;
    const int D = m->dim, hd = m->head_dim, kvd = m->n_kv_head * hd, nbd = D / QK_K;
    const int qkvw = D + 2 * kvd;
    quantize_rows(c->h, n, D, c->hq);
    job_clear(c);
    job_add_q(c, &L->wq, NULL, c->hq, nbd, n, c->big, qkvw);
    job_add_q(c, &L->wk, NULL, c->hq, nbd, n, c->big + D, qkvw);
    job_add_q(c, &L->wv, NULL, c->hq, nbd, n, c->big + D + kvd, qkvw);
    if (L->attn_gate.data) job_add_q(c, &L->attn_gate, NULL, c->hq, nbd, n, c->gate, m->n_head);
    job_run(c, CAT_ATTN);

    double t0 = time_sec();
    if (L->attn_gate.data)
        for (int i = 0; i < n * m->n_head; i++) c->gate[i] = 1.f / (1.f + expf(-c->gate[i]));
    const float *inv_freq = L->is_swa ? c->inv_freq_swa : c->inv_freq;
    const size_t lbase = (size_t)L->attn_idx * c->n_ctx * kvd;
    for (int t = 0; t < n; t++) {
        float *q = c->big + (size_t)t * qkvw;
        float *k = q + D, *v = q + D + kvd;
        const int pos = c->pos + t;
        for (int h = 0; h < m->n_head; h++) {
            rmsnorm(q + h * hd, q + h * hd, L->q_norm, hd, m->eps);
            rope_neox(q + h * hd, hd, pos, inv_freq);
        }
        for (int h = 0; h < m->n_kv_head; h++) {
            rmsnorm(k + h * hd, k + h * hd, L->k_norm, hd, m->eps);
            rope_neox(k + h * hd, hd, pos, inv_freq);
        }
        KS.f32_to_f16_row(k, c->kc + lbase + (size_t)pos * kvd, kvd);
        KS.f32_to_f16_row(v, c->vc + lbase + (size_t)pos * kvd, kvd);
    }
    AttnArgs a = { c, L, n, c->pos };
    pool_run(c->pool, attn_task, &a, n * m->n_head);
    c->stats.t_attn += time_sec() - t0;

    quantize_rows(c->tmp, n, D, c->tq);
    job_clear(c);
    job_add_q(c, &L->wo, NULL, c->tq, nbd, n, c->out, D);
    job_run(c, CAT_ATTN);
}

/* ------------------------------------------------------------------ short conv */

static void shortconv_block(Context *c, const Layer *L, int n) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, lc = m->l_cache;
    quantize_rows(c->h, n, D, c->hq);
    job_clear(c);
    job_add_q(c, &L->in_proj, NULL, c->hq, nbd, n, c->big, 3 * D);
    job_run(c, CAT_CONV);

    float *st = c->conv + (size_t)L->conv_idx * (lc - 1) * D;   /* [lc-1][D], oldest first */
    for (int t = 0; t < n; t++) {
        const float *bcx = c->big + (size_t)t * 3 * D;
        float *y = c->tmp + (size_t)t * D;
        for (int ch = 0; ch < D; ch++) {
            const float bx = bcx[ch] * bcx[2 * D + ch];
            const float *w = L->conv_w + (size_t)ch * lc;
            float acc = w[lc - 1] * bx;
            for (int k = 0; k < lc - 1; k++) acc += w[k] * st[(size_t)k * D + ch];
            for (int k = 0; k < lc - 2; k++) st[(size_t)k * D + ch] = st[(size_t)(k + 1) * D + ch];
            st[(size_t)(lc - 2) * D + ch] = bx;
            y[ch] = bcx[D + ch] * acc;
        }
    }
    quantize_rows(c->tmp, n, D, c->tq);
    job_clear(c);
    job_add_q(c, &L->out_proj, NULL, c->tq, nbd, n, c->out, D);
    job_run(c, CAT_CONV);
}

/* ------------------------------------------------------------------ feed-forward */

static void dense_ffn(Context *c, const Layer *L, int n) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, F = L->w_gate.rows;
    quantize_rows(c->h, n, D, c->hq);
    job_clear(c);
    job_add_q(c, &L->w_gate, NULL, c->hq, nbd, n, c->fg, F);
    job_add_q(c, &L->w_up, NULL, c->hq, nbd, n, c->fu, F);
    job_run(c, CAT_DENSE);
    KS.silu_mul(c->fg, c->fu, n * F);
    quantize_rows(c->fg, n, F, c->fq);
    job_clear(c);
    job_add_q(c, &L->w_down, NULL, c->fq, F / QK_K, n, c->out, D);
    job_run(c, CAT_DENSE);
}

static void prefetch_expert(Context *c, const Layer *L, int e) {
    prefetcher_push(c->pfq, L->gate_exps.data + (size_t)e * L->gate_exps.expert_stride, L->gate_exps.expert_stride);
    prefetcher_push(c->pfq, L->up_exps.data + (size_t)e * L->up_exps.expert_stride, L->up_exps.expert_stride);
    prefetcher_push(c->pfq, L->down_exps.data + (size_t)e * L->down_exps.expert_stride, L->down_exps.expert_stride);
}

/* Decode-time speculation: the residual stream changes slowly between layers, so the next
 * MoE layer's router applied to the current state predicts most of its experts. The predicted
 * experts are read ahead by the background thread while this layer computes. */
static void predict_next_experts(Context *c, int layer) {
    const Model *m = c->m;
    int ln = layer + 1;
    c->pred_layer = -1;
    if (ln >= m->n_layer || ln >= 64 || !m->layers[ln].is_moe) return;
    const Layer *Ln = &m->layers[ln];
    const int D = m->dim, E = m->n_expert;
    float hn[8192], sc[64];
    if (D > 8192 || Ln->router.type != GGML_F32) return;
    rmsnorm(hn, c->x, Ln->ffn_norm, D, m->eps);
    for (int e = 0; e < E; e++) {
        const float lg = KS.dot_f32((const float *)(Ln->router.data + (size_t)e * Ln->router.row_bytes), hn, D);
        const float p = m->gating == 1 ? lg : 1.f / (1.f + expf(-lg));
        sc[e] = p + (Ln->exp_bias && m->gating != 1 ? Ln->exp_bias[e] : 0.f);
    }
    int np = m->n_expert_used + 2;
    if (np > E) np = E;
    prefetcher_clear(c->pfq);
    for (int k = 0; k < np; k++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            int used = 0;
            for (int j = 0; j < k; j++) used |= c->pred[j] == e;
            if (!used && (best < 0 || sc[e] > sc[best])) best = e;
        }
        c->pred[k] = best;
        /* experts used in the last few dozen tokens are almost surely resident: skip them */
        if (c->gpu && gpu_resident(c->gpu, ln, best)) continue; /* served from VRAM */
        if (c->tok_counter - c->last_used[ln * 64 + best] > 48 || !c->last_used[ln * 64 + best]) prefetch_expert(c, Ln, best);
    }
    c->pred_n = np;
    c->pred_layer = ln;
}

static void moe_ffn(Context *c, const Layer *L, int layer, int n) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, E = m->n_expert, K = m->n_expert_used, F = m->ff_exp;
    const int nbf = F / QK_K;
    double t0 = time_sec();

    /* router (f32) */
    job_clear(c);
    job_add_f(c, &L->router, c->h, D, n, c->rlog, E);
    job_run(c, CAT_ROUTER);

    int cnt[64] = {0}, off[65];
    for (int t = 0; t < n; t++) {
        float *lg = c->rlog + (size_t)t * E;
        float p[64], sc[64];
        if (m->gating == 1) {
            float mx = -1e30f, s = 0;
            for (int e = 0; e < E; e++) if (lg[e] > mx) mx = lg[e];
            for (int e = 0; e < E; e++) { p[e] = expf(lg[e] - mx); s += p[e]; }
            for (int e = 0; e < E; e++) p[e] /= s;
        } else {
            for (int e = 0; e < E; e++) p[e] = 1.f / (1.f + expf(-lg[e]));
        }
        for (int e = 0; e < E; e++) sc[e] = p[e] + (L->exp_bias ? L->exp_bias[e] : 0.f);
        int *sel = c->sel + (size_t)t * K;
        float *w = c->selw + (size_t)t * K;
        float wsum = 0;
        for (int k = 0; k < K; k++) {
            int best = -1;
            for (int e = 0; e < E; e++) {
                int used = 0;
                for (int j = 0; j < k; j++) used |= sel[j] == e;
                if (!used && (best < 0 || sc[e] > sc[best])) best = e;
            }
            sel[k] = best;
            w[k] = p[best];
            wsum += p[best];
            if (c->gpu) gpu_note_usage(c->gpu, layer, best);
            if (layer < 64) {
                c->stats.expert_hits[layer * 64 + best]++;
                c->last_used[layer * 64 + best] = c->tok_counter;
            }
        }
        if (m->norm_topk && wsum > 0)
            for (int k = 0; k < K; k++) w[k] /= wsum;
    }

    /* (token, expert) pairs whose expert is cached in VRAM run on the GPU, concurrently
     * with the CPU computing the others */
    int ng = 0;
    if (c->gpu) {
        for (int i = 0; i < n * K; i++) {
            c->on_gpu[i] = gpu_resident(c->gpu, layer, c->sel[i]);
            if (c->on_gpu[i]) {
                c->ga_e[ng] = c->sel[i];
                c->ga_t[ng] = i / K;
                c->tok_a[i] = -1 - ng;
                ng++;
            }
        }
        if (ng) gpu_moe_begin(c->gpu, layer, c->h, n, c->ga_e, c->ga_t, ng);
        c->stats.gpu_experts += (uint64_t)ng;
    } else {
        memset(c->on_gpu, 0, sizeof(int) * (size_t)n * K);
    }
    for (int i = 0; i < n * K; i++)
        if (!c->on_gpu[i]) cnt[c->sel[i]]++;
    off[0] = 0;
    for (int e = 0; e < E; e++) off[e + 1] = off[e] + cnt[e];

    if (c->pfq && n == 1) {
        /* score the prediction made one layer earlier, then predict the next layer */
        if (c->pred_layer == layer) {
            for (int k = 0; k < K; k++) {
                int hit = 0;
                for (int j = 0; j < c->pred_n; j++) hit |= c->pred[j] == c->sel[k];
                c->stats.pred_hits += hit;
                c->stats.pred_total++;
            }
        }
        predict_next_experts(c, layer);
    }

    /* assignment list grouped by expert; gather quantized inputs */
    quantize_rows(c->h, n, D, c->hq);
    int fill[64];
    memcpy(fill, off, sizeof(int) * E);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < K; k++) {
            if (c->on_gpu[(size_t)t * K + k]) continue;
            int e = c->sel[(size_t)t * K + k];
            int a = fill[e]++;
            c->a_tok[a] = t;
            c->tok_a[(size_t)t * K + k] = a;
            memcpy(c->xg + (size_t)a * nbd, c->hq + (size_t)t * nbd, sizeof(block_q8_K) * nbd);
        }
    c->stats.t_moe_route += time_sec() - t0;

    job_clear(c);
    for (int e = 0; e < E; e++) {
        if (!cnt[e]) continue;
        const block_q8_K *xq = c->xg + (size_t)off[e] * nbd;
        job_add_q(c, &L->gate_exps, L->gate_exps.data + (size_t)e * L->gate_exps.expert_stride, xq, nbd, cnt[e],
                  c->eg + (size_t)off[e] * F, F);
        job_add_q(c, &L->up_exps, L->up_exps.data + (size_t)e * L->up_exps.expert_stride, xq, nbd, cnt[e],
                  c->eu + (size_t)off[e] * F, F);
    }
    job_run(c, CAT_EXP_GU);

    const int na = off[E];
    KS.silu_mul(c->eg, c->eu, na * F);
    quantize_rows(c->eg, na, F, c->eq);

    job_clear(c);
    for (int e = 0; e < E; e++) {
        if (!cnt[e]) continue;
        job_add_q(c, &L->down_exps, L->down_exps.data + (size_t)e * L->down_exps.expert_stride,
                  c->eq + (size_t)off[e] * nbf, nbf, cnt[e], c->eo + (size_t)off[e] * D, D);
    }
    job_run(c, CAT_EXP_DOWN);

    for (int t = 0; t < n; t++) {
        float *o = c->out + (size_t)t * D;
        memset(o, 0, sizeof(float) * D);
        for (int k = 0; k < K; k++) {
            const int a = c->tok_a[(size_t)t * K + k];
            if (a < 0) continue;
            const float w = c->selw[(size_t)t * K + k];
            const float *eo = c->eo + (size_t)a * D;
            for (int i = 0; i < D; i++) o[i] += w * eo[i];
        }
    }
    if (ng) {
        double tw = time_sec();
        gpu_moe_end(c->gpu, c->gy);
        c->stats.t_gpu_wait += time_sec() - tw;
        for (int i = 0; i < n * K; i++) {
            if (!c->on_gpu[i]) continue;
            const float w = c->selw[i];
            const float *gy = c->gy + (size_t)(-1 - c->tok_a[i]) * D;
            float *o = c->out + (size_t)(i / K) * D;
            for (int j = 0; j < D; j++) o[j] += w * gy[j];
        }
    }
}

/* ------------------------------------------------------------------ dense GPU decode */

/* The experts of one token that are not in VRAM; y = their weighted sum. Returns 0 if none. */
static int cpu_experts_one(Context *c, const Layer *L, const int *sel, const int *slot, const float *w, float *y) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, K = m->n_expert_used, F = m->ff_exp, nbf = F / QK_K;
    int idx[8], na = 0;
    for (int k = 0; k < K; k++)
        if (slot[k] < 0) idx[na++] = k;
    if (!na) return 0;
    quantize_rows(c->h, 1, D, c->hq);
    job_clear(c);
    for (int a = 0; a < na; a++) {
        const int e = sel[idx[a]];
        job_add_q(c, &L->gate_exps, L->gate_exps.data + (size_t)e * L->gate_exps.expert_stride, c->hq, nbd, 1,
                  c->eg + (size_t)a * F, F);
        job_add_q(c, &L->up_exps, L->up_exps.data + (size_t)e * L->up_exps.expert_stride, c->hq, nbd, 1,
                  c->eu + (size_t)a * F, F);
    }
    job_run(c, CAT_EXP_GU);
    KS.silu_mul(c->eg, c->eu, na * F);
    quantize_rows(c->eg, na, F, c->eq);
    job_clear(c);
    for (int a = 0; a < na; a++) {
        const int e = sel[idx[a]];
        job_add_q(c, &L->down_exps, L->down_exps.data + (size_t)e * L->down_exps.expert_stride, c->eq + (size_t)a * nbf, nbf, 1,
                  c->eo + (size_t)a * D, D);
    }
    job_run(c, CAT_EXP_DOWN);
    memset(y, 0, sizeof(float) * D);
    for (int a = 0; a < na; a++) {
        const float wk = w[idx[a]];
        const float *eo = c->eo + (size_t)a * D;
        for (int i = 0; i < D; i++) y[i] += wk * eo[i];
    }
    return 1;
}

/* One token through every layer on the GPU; the CPU only computes the non-resident experts. */
static void forward_token_gpu(Context *c, int token, float *all_logits) {
    const Model *m = c->m;
    const int D = m->dim, K = m->n_expert_used, hd = m->head_dim, pos = c->pos;
    Gpu *g = c->gpu;
    if (c->kv_gpu_n < pos) {
        gpu_dense_kv(g, c->kc, c->vc, c->kv_gpu_n, pos, 1);
        c->kv_gpu_n = pos;
    }
    if (!c->conv_gpu_ok) {
        gpu_dense_conv(g, c->conv, 1);
        c->conv_gpu_ok = 1;
    }
    if (token < 0 || token >= m->n_vocab) token = 0;
    dequant_row(m->embed.type, m->embed.data + (size_t)token * m->embed.row_bytes, c->x, D);
    float rope[512], rope_swa[512];
    for (int i = 0; i < hd / 2; i++) {
        const double a = (double)pos * c->inv_freq[i], as = (double)pos * c->inv_freq_swa[i];
        rope[i] = (float)cos(a);
        rope[hd / 2 + i] = (float)sin(a);
        rope_swa[i] = (float)cos(as);
        rope_swa[hd / 2 + i] = (float)sin(as);
    }
    gpu_dense_begin(g, c->x, rope, m->swa_window > 0 ? rope_swa : NULL, pos);
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        if (!L->is_moe) continue;
        const float *gx, *gh;
        const int *ret;
        const double tw = time_sec();
        gpu_dense_wait(g, l, &gx, &gh, &ret);
        c->stats.t_gpu_wait += time_sec() - tw;
        memcpy(c->x, gx, sizeof(float) * D);
        memcpy(c->h, gh, sizeof(float) * D);
        int sel[8], slot[8];
        float w[8];
        memcpy(sel, ret, sizeof(int) * K);
        memcpy(slot, ret + K, sizeof(int) * K);
        memcpy(w, ret + 2 * K, sizeof(float) * K);
        for (int k = 0; k < K; k++) {
            gpu_note_usage(g, l, sel[k]);
            if (l < 64) {
                c->stats.expert_hits[l * 64 + sel[k]]++;
                c->last_used[l * 64 + sel[k]] = c->tok_counter;
            }
            c->stats.gpu_experts += slot[k] >= 0;
        }
        if (c->pfq) {
            if (c->pred_layer == l)
                for (int k = 0; k < K; k++) {
                    int hit = 0;
                    for (int j = 0; j < c->pred_n; j++) hit |= c->pred[j] == sel[k];
                    c->stats.pred_hits += hit;
                    c->stats.pred_total++;
                }
            predict_next_experts(c, l);
        }
        const int any = cpu_experts_one(c, L, sel, slot, w, c->out);
        gpu_dense_moe_finish(g, l, any ? c->out : NULL);
    }
    gpu_dense_logits(g, c->logits);
    if (all_logits) memcpy(all_logits, c->logits, sizeof(float) * (size_t)m->n_vocab);
    c->pos++;
    c->tok_counter++;
    c->kv_gpu_n = pos + 1;
    if (c->kv_cpu_n > pos) c->kv_cpu_n = pos;
    c->conv_cpu_ok = 0;
}

/* ------------------------------------------------------------------ forward */

static void forward_chunk(Context *c, const int *tokens, int n, int want_logits, float *all_logits) {
    const Model *m = c->m;
    const int D = m->dim;
    if (c->gpu_dense && n == 1) {
        forward_token_gpu(c, tokens[0], all_logits);
        return;
    }
    const int pos0 = c->pos;
    sync_to_cpu(c, pos0);
    for (int t = 0; t < n; t++) {
        int id = tokens[t];
        if (id < 0 || id >= m->n_vocab) id = 0;
        dequant_row(m->embed.type, m->embed.data + (size_t)id * m->embed.row_bytes, c->x + (size_t)t * D, D);
    }
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        if (c->pfq && n >= 8 && l + 1 < m->n_layer && m->layers[l + 1].is_moe) {
            /* prompt batches touch nearly every expert: stream the next layer, expert by expert
             * (small requests so a newer layer can cancel what is still pending) */
            const Layer *Ln = &m->layers[l + 1];
            prefetcher_clear(c->pfq);
            for (int e = 0; e < m->n_expert; e++)
                if (!c->gpu || !gpu_resident(c->gpu, l + 1, e)) prefetch_expert(c, Ln, e);
        }
        for (int t = 0; t < n; t++) rmsnorm(c->h + (size_t)t * D, c->x + (size_t)t * D, L->attn_norm, D, m->eps);
        if (L->is_attn) attention_block(c, L, n);
        else shortconv_block(c, L, n);
        if (L->post_op_norm)
            for (int t = 0; t < n; t++) rmsnorm(c->out + (size_t)t * D, c->out + (size_t)t * D, L->post_op_norm, D, m->eps);
        for (size_t i = 0; i < (size_t)n * D; i++) c->x[i] += c->out[i];
        for (int t = 0; t < n; t++) rmsnorm(c->h + (size_t)t * D, c->x + (size_t)t * D, L->ffn_norm, D, m->eps);
        if (L->is_moe) moe_ffn(c, L, l, n);
        else dense_ffn(c, L, n);
        if (L->post_ffn_norm)
            for (int t = 0; t < n; t++) rmsnorm(c->out + (size_t)t * D, c->out + (size_t)t * D, L->post_ffn_norm, D, m->eps);
        for (size_t i = 0; i < (size_t)n * D; i++) c->x[i] += c->out[i];
    }
    c->pos += n;
    c->tok_counter += (uint32_t)n;
    c->kv_cpu_n = c->pos;
    if (c->kv_gpu_n > pos0) c->kv_gpu_n = pos0;
    c->conv_gpu_ok = 0;
    if (all_logits) {
        for (int t = 0; t < n; t++) rmsnorm(c->h + (size_t)t * D, c->x + (size_t)t * D, m->final_norm, D, m->eps);
        job_clear(c);
        if (m->output.type == GGML_F32 || m->output.type == GGML_F16) {
            job_add_f(c, &m->output, c->h, D, n, all_logits, m->n_vocab);
        } else {
            quantize_rows(c->h, n, D, c->hq);
            job_add_q(c, &m->output, NULL, c->hq, D / QK_K, n, all_logits, m->n_vocab);
        }
        job_run(c, CAT_LOGITS);
        memcpy(c->logits, all_logits + (size_t)(n - 1) * m->n_vocab, sizeof(float) * m->n_vocab);
    } else if (want_logits) {
        const float *xl = c->x + (size_t)(n - 1) * D;
        rmsnorm(c->h, xl, m->final_norm, D, m->eps);
        job_clear(c);
        if (m->output.type == GGML_F32 || m->output.type == GGML_F16) {
            job_add_f(c, &m->output, c->h, D, 1, c->logits, m->n_vocab);
        } else {
            quantize_rows(c->h, 1, D, c->hq);
            job_add_q(c, &m->output, NULL, c->hq, D / QK_K, 1, c->logits, m->n_vocab);
        }
        job_run(c, CAT_LOGITS);
    }
}

float *ctx_forward(Context *c, const int *tokens, int n) {
    if (n <= 0) return c->logits;
    if (c->pos + n > c->n_ctx) return NULL;
    for (int i = 0; i < n; i += c->n_batch) {
        int nb = n - i < c->n_batch ? n - i : c->n_batch;
        double t0 = time_sec();
        forward_chunk(c, tokens + i, nb, i + nb == n, NULL);
        c->stats.t_forward += time_sec() - t0;
        if (c->gpu) gpu_rebalance(c->gpu, nb == 1 ? 2 : 16);
    }
    return c->logits;
}

int ctx_forward_all(Context *c, const int *tokens, int n, float *out) {
    if (c->pos + n > c->n_ctx) return 0;
    for (int i = 0; i < n; i += c->n_batch) {
        int nb = n - i < c->n_batch ? n - i : c->n_batch;
        forward_chunk(c, tokens + i, nb, 0, out + (size_t)i * c->m->n_vocab);
    }
    return 1;
}
