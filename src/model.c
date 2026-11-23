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
 * and partial rotation on the full-attention layers (rope_freqs divisors).
 *
 * Qwen3.5 / 3.6 MoE (qwen35moe): three Gated DeltaNet layers to one full-attention layer, every
 * FFN an MoE (softmax router, top-k renormalized) plus a shared expert gated by sigmoid(w . h).
 * DeltaNet: qkv = silu(causal_conv(attn_qkv h)), q and k L2-normalized, per V head j (K head
 * j % n_k) S = S exp(g) + beta k (v - S^T k)^T, o = S^T q / sqrt(d), then rmsnorm(o) silu(z).
 * Attention: attn_q gives q and a gate per head, RoPE on the first n_rot dims only, the output
 * multiplied by sigmoid(gate) elementwise. The converter folded the +1 of Qwen's norms into the
 * weights and stored ssm_a as -exp(A_log). */
#include "model.h"
#include "gpu.h"
#include "gpu_dx.h"
#include "gpu_vk.h"
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

#define MAX_JOBS (2 * MAX_EXPERT + 32)   /* gate and up of every expert of a prompt chunk, at most */
#define DX_MIN_TOKENS 1      /* default PITHOS_DX_MIN: matmuls over fewer tokens stay on the CPU */
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
    /* Inkling's 1152-long expert rows: NVFP4 and Q8_0 rows may end in a partial Q8_K block */
    const int blk = !m->ik ? QK_K : t->type == GGML_NVFP4 ? QK_NVFP4 : t->type == GGML_Q8_0 ? QK8_0 : QK_K;
    if (t->type != GGML_F32 && t->type != GGML_F16 && (mat->cols % blk) != 0) {
        snprintf(err, errlen, "tensor %s: row length %d not a multiple of %d", name, mat->cols, blk);
        return -1;
    }
    return 1;
}

/* an F32 or BF16 matrix as F32: BF16 is converted into memory the model owns */
static int load_mat_f32(Model *m, Mat *mat, const char *name, char *err, int errlen) {
    const GGUFTensor *t = gguf_tensor(&m->gguf, name);
    if (!t) { snprintf(err, errlen, "missing tensor %s", name); return 0; }
    if (t->type == GGML_F32) return load_mat(m, mat, name, 1, err, errlen);
    if (t->type != GGML_BF16 || t->ne[2] * t->ne[3] != 1) { snprintf(err, errlen, "tensor %s: expected BF16 or F32", name); return 0; }
    const size_t n = (size_t)t->ne[0] * (size_t)t->ne[1];
    float *f = (float *)malloc(sizeof(float) * n);
    void **o = (void **)realloc(m->owned, sizeof(void *) * (size_t)(m->n_owned + 1));
    if (!f || !o) { free(f); snprintf(err, errlen, "out of memory"); return 0; }
    m->owned = o;
    m->owned[m->n_owned++] = f;
    const uint16_t *s = (const uint16_t *)t->data;
    for (size_t i = 0; i < n; i++) {
        const uint32_t u = (uint32_t)s[i] << 16;
        memcpy(f + i, &u, 4);
    }
    memset(mat, 0, sizeof(*mat));
    mat->type = GGML_F32;
    mat->data = (const uint8_t *)f;
    mat->cols = (int)t->ne[0];
    mat->rows = (int)t->ne[1];
    mat->row_bytes = sizeof(float) * (size_t)mat->cols;
    mat->expert_stride = mat->row_bytes * (size_t)mat->rows;
    return 1;
}

static const float *load_f32(Model *m, const char *name, int n, int required, char *err, int errlen) {
    const GGUFTensor *t = gguf_tensor(&m->gguf, name);
    if (!t) {
        if (required) snprintf(err, errlen, "missing tensor %s", name);
        return NULL;
    }
    if (t->type != GGML_F32 || (n > 0 && (int64_t)n != t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3])) {
        snprintf(err, errlen, "tensor %s: expected F32[%d]", name, n);
        return NULL;
    }
    return (const float *)t->data;
}

#define REQ(expr) do { if ((expr) <= 0) goto fail; } while (0)
#define REQP(expr) do { if (!(expr)) goto fail; } while (0)

/* weight bytes one decoded token touches */
static uint64_t token_bytes(const Model *m) {
    uint64_t b = (uint64_t)m->output.row_bytes * m->n_vocab;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        if (L->is_attn) b += (uint64_t)L->wq.expert_stride + L->wk.expert_stride + L->wv.expert_stride + L->wo.expert_stride +
                             L->attn_gate.expert_stride;
        else if (L->is_delta) b += (uint64_t)L->ssm_qkv.expert_stride + L->ssm_z.expert_stride + L->ssm_alpha.expert_stride +
                                   L->ssm_beta.expert_stride + L->ssm_out.expert_stride;
        else b += (uint64_t)L->in_proj.expert_stride + L->out_proj.expert_stride;
        if (L->is_moe) b += (uint64_t)m->n_expert_used * (L->gate_exps.expert_stride + L->up_exps.expert_stride + L->down_exps.expert_stride) +
                            L->sh_gate.expert_stride + L->sh_up.expert_stride + L->sh_down.expert_stride;
        else b += (uint64_t)L->w_gate.expert_stride + L->w_up.expert_stride + L->w_down.expert_stride;
    }
    return b;
}

/* Qwen3.5 / 3.6 MoE, the GGUF of llama.cpp's converter (arch qwen35moe); m->gguf is open */
static int load_qwen35(Model *m, char *err, int errlen) {
    GGUF *g = &m->gguf;
#define KEY(s) "qwen35moe." s
    m->n_layer = (int)gguf_get_int(g, KEY("block_count"), 0);
    m->dim = (int)gguf_get_int(g, KEY("embedding_length"), 0);
    m->n_head = (int)gguf_get_int(g, KEY("attention.head_count"), 0);
    m->n_kv_head = (int)gguf_get_int(g, KEY("attention.head_count_kv"), 0);
    m->head_dim = (int)gguf_get_int(g, KEY("attention.key_length"), 0);
    m->n_rot = (int)gguf_get_int(g, KEY("rope.dimension_count"), m->head_dim);
    m->ff_exp = (int)gguf_get_int(g, KEY("expert_feed_forward_length"), 0);
    m->ff_shexp = (int)gguf_get_int(g, KEY("expert_shared_feed_forward_length"), 0);
    m->n_expert = (int)gguf_get_int(g, KEY("expert_count"), 0);
    m->n_expert_used = (int)gguf_get_int(g, KEY("expert_used_count"), 0);
    m->n_ctx_train = (int)gguf_get_int(g, KEY("context_length"), 4096);
    m->eps = (float)gguf_get_float(g, KEY("attention.layer_norm_rms_epsilon"), 1e-6);
    m->rope_base = m->rope_base_swa = (float)gguf_get_float(g, KEY("rope.freq_base"), 1e7);
    m->ssm_conv_k = (int)gguf_get_int(g, KEY("ssm.conv_kernel"), 0);
    m->ssm_d = (int)gguf_get_int(g, KEY("ssm.state_size"), 0);
    m->ssm_kh = (int)gguf_get_int(g, KEY("ssm.group_count"), 0);
    m->ssm_vh = (int)gguf_get_int(g, KEY("ssm.time_step_rank"), 0);
    const int inner = (int)gguf_get_int(g, KEY("ssm.inner_size"), 0);
    const int interval = (int)gguf_get_int(g, KEY("full_attention_interval"), 4);
    const GGUFKV *rec = gguf_find(g, KEY("attention.recurrent_layers"));
#undef KEY
    m->gating = 1;      /* softmax over all experts, then the top k renormalized */
    m->norm_topk = 1;
    m->q_gate = 1;
    m->l_cache = 1;     /* no short conv: no state of its own */
    m->add_bos = (int)gguf_get_int(g, "tokenizer.ggml.add_bos_token", 0);
    if (m->n_layer <= 0 || m->dim <= 0 || m->n_head <= 0 || m->n_kv_head <= 0 || m->n_head % m->n_kv_head ||
        m->head_dim <= 0 || m->head_dim > 256 || m->n_rot <= 0 || m->n_rot > m->head_dim || m->n_rot % 2 ||
        m->ssm_conv_k < 2 || m->ssm_d <= 0 || m->ssm_d > 256 || m->ssm_kh <= 0 || m->ssm_vh % m->ssm_kh ||
        inner != m->ssm_vh * m->ssm_d || m->n_expert <= 0 || m->n_expert > MAX_EXPERT || m->n_expert_used <= 0 ||
        m->n_expert_used > 8 || m->ff_exp % QK_K || m->ff_shexp % QK_K) {
        snprintf(err, errlen, "qwen35moe: unsupported hyperparameters");
        return 0;
    }
    REQ(load_mat(m, &m->embed, "token_embd.weight", 1, err, errlen));
    m->n_vocab = m->embed.rows;
    if (load_mat(m, &m->output, "output.weight", 0, err, errlen) < 0) return 0;
    if (!m->output.data) m->output = m->embed;
    REQP(m->final_norm = load_f32(m, "output_norm.weight", m->dim, 1, err, errlen));

    const int qkv = 2 * m->ssm_kh * m->ssm_d + inner;
    m->layers = (Layer *)calloc((size_t)m->n_layer, sizeof(Layer));
    char nm[128];
#define T(s) (snprintf(nm, sizeof(nm), "blk.%d." s, l), nm)
    for (int l = 0; l < m->n_layer; l++) {
        Layer *L = &m->layers[l];
        L->is_delta = rec && rec->type == GGUF_T_ARRAY ? gguf_arr_int(rec, (uint64_t)l) != 0 : (l + 1) % interval != 0;
        L->is_attn = !L->is_delta;
        REQP(L->attn_norm = load_f32(m, T("attn_norm.weight"), m->dim, 1, err, errlen));
        REQP(L->ffn_norm = load_f32(m, T("post_attention_norm.weight"), m->dim, 1, err, errlen));  /* the pre-FFN norm */
        if (L->is_delta) {
            L->delta_idx = m->n_delta++;
            REQ(load_mat(m, &L->ssm_qkv, T("attn_qkv.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->ssm_z, T("attn_gate.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->ssm_alpha, T("ssm_alpha.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->ssm_beta, T("ssm_beta.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->ssm_out, T("ssm_out.weight"), 1, err, errlen));
            REQP(L->ssm_conv = load_f32(m, T("ssm_conv1d.weight"), qkv * m->ssm_conv_k, 1, err, errlen));
            REQP(L->ssm_a = load_f32(m, T("ssm_a"), m->ssm_vh, 1, err, errlen));
            REQP(L->ssm_dt = load_f32(m, T("ssm_dt.bias"), m->ssm_vh, 1, err, errlen));
            REQP(L->ssm_norm = load_f32(m, T("ssm_norm.weight"), m->ssm_d, 1, err, errlen));
            if (L->ssm_qkv.rows != qkv || L->ssm_z.rows != inner || L->ssm_alpha.rows != m->ssm_vh ||
                L->ssm_beta.rows != m->ssm_vh || L->ssm_out.cols != inner || L->ssm_out.rows != m->dim) {
                snprintf(err, errlen, "layer %d: DeltaNet tensors of unexpected shape", l);
                return 0;
            }
        } else {
            L->attn_idx = m->n_attn++;
            REQ(load_mat(m, &L->wq, T("attn_q.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wk, T("attn_k.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wv, T("attn_v.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wo, T("attn_output.weight"), 1, err, errlen));
            REQP(L->q_norm = load_f32(m, T("attn_q_norm.weight"), m->head_dim, 1, err, errlen));
            REQP(L->k_norm = load_f32(m, T("attn_k_norm.weight"), m->head_dim, 1, err, errlen));
            if (L->wq.rows != 2 * m->n_head * m->head_dim || L->wk.rows != m->n_kv_head * m->head_dim ||
                L->wo.cols != m->n_head * m->head_dim) {
                snprintf(err, errlen, "layer %d: attention tensors of unexpected shape", l);
                return 0;
            }
        }
        L->is_moe = 1;
        REQ(load_mat(m, &L->router, T("ffn_gate_inp.weight"), 1, err, errlen));
        if (L->router.type != GGML_F32 || L->router.rows != m->n_expert) { snprintf(err, errlen, "router must be F32"); return 0; }
        REQ(load_mat(m, &L->gate_exps, T("ffn_gate_exps.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->up_exps, T("ffn_up_exps.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->down_exps, T("ffn_down_exps.weight"), 1, err, errlen));
        if (m->ff_shexp) {
            REQ(load_mat(m, &L->sh_gate, T("ffn_gate_shexp.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->sh_up, T("ffn_up_shexp.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->sh_down, T("ffn_down_shexp.weight"), 1, err, errlen));
            REQP(L->sh_inp = load_f32(m, T("ffn_gate_inp_shexp.weight"), m->dim, 1, err, errlen));
        }
    }
#undef T
    m->bytes_per_token = token_bytes(m);
    return 1;
fail:
    return 0;
}

/* Inkling 2 Mini (inkling2_loop, the layout of llama.cpp's mainline converter); m->gguf is open */
static int load_inkling2(Model *m, char *err, int errlen) {
    GGUF *g = &m->gguf;
    m->ik = 1;
#define KEY(s) "inkling2_loop." s
    const GGUFKV *nh = gguf_find(g, KEY("attention.head_count")), *nkv = gguf_find(g, KEY("attention.head_count_kv"));
    const GGUFKV *swa = gguf_find(g, KEY("attention.sliding_window_pattern"));
    m->n_layer = (int)gguf_get_int(g, KEY("block_count"), 0);
    m->dim = (int)gguf_get_int(g, KEY("embedding_length"), 0);
    m->n_head = nh && nh->type == GGUF_T_ARRAY ? (int)gguf_arr_int(nh, 0) : (int)gguf_get_int(g, KEY("attention.head_count"), 0);
    m->n_kv_head = nkv && nkv->type == GGUF_T_ARRAY ? (int)gguf_arr_int(nkv, 0) : (int)gguf_get_int(g, KEY("attention.head_count_kv"), 0);
    m->head_dim = (int)gguf_get_int(g, KEY("attention.key_length"), 0);
    m->ff_exp = (int)gguf_get_int(g, KEY("expert_feed_forward_length"), 0);
    m->n_expert = (int)gguf_get_int(g, KEY("expert_count"), 0);
    m->n_expert_used = (int)gguf_get_int(g, KEY("expert_used_count"), 0);
    m->n_ctx_train = (int)gguf_get_int(g, KEY("context_length"), 4096);
    m->eps = (float)gguf_get_float(g, KEY("attention.layer_norm_rms_epsilon"), 1e-6);
    m->swa_window = (int)gguf_get_int(g, KEY("attention.sliding_window"), 0);
    m->l_cache = (int)gguf_get_int(g, KEY("shortconv.l_cache"), 0);
    m->res_block = (int)gguf_get_int(g, KEY("attn_res.block_size"), 0);
    m->n_rel = (int)gguf_get_int(g, KEY("attention.relative_length"), 0);
    m->rel_extent = (int)gguf_get_int(g, KEY("attention.relative_extent"), 0);
    m->qk_clip = (float)gguf_get_float(g, KEY("attention.qk_clip"), 0);
    m->rel_clip = (float)gguf_get_float(g, KEY("attention.relative_clip"), 0);
    m->log_floor = (int)gguf_get_int(g, KEY("attention.log_scale_floor"), 0);
    m->log_alpha = (float)gguf_get_float(g, KEY("attention.log_scale_alpha"), 0);
    m->exp_scale = (float)gguf_get_float(g, KEY("expert_weights_scale"), 1);
    m->situ_beta = (float)gguf_get_float(g, KEY("activation.situ_beta"), 0);
    m->situ_lb = (float)gguf_get_float(g, KEY("activation.situ_linear_beta"), 0);
    m->softcap = (float)gguf_get_float(g, KEY("final_logit_softcapping"), 0);
    m->loop_start = (int)gguf_get_int(g, KEY("loop.start"), 0);
    m->loop_end = (int)gguf_get_int(g, KEY("loop.end"), 0);
    m->n_pass = (int)gguf_get_int(g, KEY("loop.count"), 0);
    m->gate_rank = (int)gguf_get_int(g, KEY("loop.gate_rank"), 0);
    const int varying = (int)gguf_get_int(g, KEY("loop.varying"), 0);
    const int vocab_padded = (int)gguf_get_int(g, KEY("vocab_size"), 0);
    const int vocab = (int)gguf_get_int(g, KEY("vocab_size_unpadded"), 0);
#undef KEY
    m->gating = 2;      /* sigmoid router with a selection bias, the top k renormalized, times exp_scale */
    m->norm_topk = 1;
    m->add_bos = (int)gguf_get_int(g, "tokenizer.ggml.add_bos_token", 0);
    const int nblk = m->res_block > 0 ? m->n_layer / m->res_block : 0;
    if (m->n_layer <= 0 || m->n_layer > 64 || m->dim <= 0 || m->dim % QK_K || m->dim > 8192 || m->n_head <= 0 ||
        m->n_kv_head <= 0 || m->n_head % m->n_kv_head || m->head_dim <= 0 || m->head_dim > 256 || m->n_expert <= 0 ||
        m->n_expert > MAX_EXPERT || m->n_expert_used <= 0 || m->n_expert_used > 8 || m->ff_exp <= 0 || m->ff_exp % QK_NVFP4 ||
        m->swa_window <= 0 || m->l_cache < 2 || m->l_cache > 8 || m->res_block <= 0 || m->n_layer % m->res_block || nblk > 8 ||
        m->n_rel <= 0 || m->n_rel > 64 || m->rel_extent <= 0 || m->log_floor <= 0 || !(m->qk_clip > 0) || !(m->rel_clip > 0) ||
        !(m->situ_beta > 0) || m->loop_start <= 0 || m->loop_start >= m->loop_end || m->loop_end > m->n_layer ||
        m->loop_start % m->res_block || m->loop_end % m->res_block || m->n_pass < 2 || varying || m->gate_rank <= 0 ||
        !swa || swa->type != GGUF_T_ARRAY || swa->arr_len != (uint64_t)m->n_layer || vocab <= 0 || vocab > vocab_padded) {
        snprintf(err, errlen, "inkling2_loop: unsupported hyperparameters");
        return 0;
    }
    for (int l = 0; l < m->n_layer; l++)
        if ((nh && nh->type == GGUF_T_ARRAY && gguf_arr_int(nh, (uint64_t)l) != m->n_head) ||
            (nkv && nkv->type == GGUF_T_ARRAY && gguf_arr_int(nkv, (uint64_t)l) != m->n_kv_head)) {
            snprintf(err, errlen, "inkling2_loop: layers with different head counts");
            return 0;
        }
    m->n_slot = m->n_layer + (m->n_pass - 1) * (m->loop_end - m->loop_start);
    m->n_attn = m->n_slot;          /* a KV cache per slot */
    const int D = m->dim, hd = m->head_dim, kvd = m->n_kv_head * hd, qd = m->n_head * hd, lc = m->l_cache;
    m->ik_ch = 2 * kvd + 2 * D;

    REQ(load_mat(m, &m->embed, "token_embd.weight", 1, err, errlen));
    REQ(load_mat(m, &m->output, "output.weight", 1, err, errlen));
    if (m->embed.rows != vocab_padded || m->output.rows != vocab_padded || m->embed.cols != D || m->output.cols != D) {
        snprintf(err, errlen, "inkling2_loop: embedding or output of unexpected shape");
        return 0;
    }
    m->n_vocab = vocab;
    m->output.rows = vocab;         /* the rows past the vocabulary are padding */
    REQP(m->embd_norm = load_f32(m, "token_embd_norm.weight", D, 1, err, errlen));
    REQP(m->final_norm = load_f32(m, "output_norm.weight", D, 1, err, errlen));
    REQP(m->out_query = load_f32(m, "output_res_query.weight", D, 1, err, errlen));

    m->layers = (Layer *)calloc((size_t)m->n_layer, sizeof(Layer));
    char nm[128];
#define T(s) (snprintf(nm, sizeof(nm), "blk.%d." s, l), nm)
    for (int l = 0; l < m->n_layer; l++) {
        Layer *L = &m->layers[l];
        L->is_attn = L->is_moe = 1;
        L->is_swa = gguf_arr_int(swa, (uint64_t)l) != 0;
        L->attn_idx = l;
        const int extent = L->is_swa ? m->swa_window : m->rel_extent;
        REQ(load_mat(m, &L->wq, T("attn_q.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->wk, T("attn_k.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->wv, T("attn_v.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->wo, T("attn_output.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->ik_gate, T("attn_gate.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->ik_rel, T("attn_rel.weight"), 1, err, errlen));
        REQ(load_mat_f32(m, &L->ik_sink, T("attn_sink_proj.weight"), err, errlen));
        REQP(L->ik_q_clip = load_f32(m, T("attn_q_clip.weight"), qd, 1, err, errlen));
        REQP(L->ik_rel_bias = load_f32(m, T("attn_rel_bias.weight"), m->n_rel * extent, 1, err, errlen));
        REQP(L->ik_rel_clip = load_f32(m, T("attn_rel_clip.weight"), m->n_rel * m->n_head, 1, err, errlen));
        REQP(L->ik_k_conv = load_f32(m, T("attn_k_conv.weight"), lc * kvd, 1, err, errlen));
        REQP(L->ik_v_conv = load_f32(m, T("attn_v_conv.weight"), lc * kvd, 1, err, errlen));
        REQP(L->ik_o_conv = load_f32(m, T("attn_post_conv.weight"), lc * D, 1, err, errlen));
        REQP(L->ik_f_conv = load_f32(m, T("ffn_post_conv.weight"), lc * D, 1, err, errlen));
        if (L->is_swa) {
            REQP(L->attn_norm = load_f32(m, T("attn_norm.weight"), D, 1, err, errlen));
        } else {
            REQP(L->ik_q_norm = load_f32(m, T("attn_q_inp_norm.weight"), D, 1, err, errlen));
            REQP(L->ik_k_norm = load_f32(m, T("attn_k_inp_norm.weight"), D, 1, err, errlen));
            REQP(L->ik_v_norm = load_f32(m, T("attn_v_inp_norm.weight"), D, 1, err, errlen));
            REQ(load_mat_f32(m, &L->ik_kdyn, T("attn_k_conv_proj.weight"), err, errlen));
            REQ(load_mat_f32(m, &L->ik_vdyn, T("attn_v_conv_proj.weight"), err, errlen));
            if (L->ik_kdyn.rows != lc * m->n_kv_head || L->ik_vdyn.rows != lc * m->n_kv_head || L->ik_kdyn.cols != D) {
                snprintf(err, errlen, "layer %d: conv taps of unexpected shape", l);
                return 0;
            }
        }
        if (l > 0) REQP(L->ik_attn_query = load_f32(m, T("attn_res_query.weight"), D * (L->is_swa ? 1 : 3), 1, err, errlen));
        REQP(L->ik_ffn_query = load_f32(m, T("ffn_res_query.weight"), D, 1, err, errlen));
        REQP(L->ffn_norm = load_f32(m, T("ffn_norm.weight"), D, 1, err, errlen));
        REQ(load_mat(m, &L->router, T("ffn_gate_inp.weight"), 1, err, errlen));
        L->exp_bias = load_f32(m, T("exp_probs_b.bias"), m->n_expert, 0, err, errlen);
        REQ(load_mat(m, &L->gate_exps, T("ffn_gate_up_exps.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->down_exps, T("ffn_down_exps.weight"), 1, err, errlen));
        L->ik_gu_scale = load_f32(m, T("ffn_gate_up_exps.scale"), m->n_expert, 0, err, errlen);
        L->ik_down_scale = load_f32(m, T("ffn_down_exps.scale"), m->n_expert, 0, err, errlen);
        if (L->wq.rows != qd || L->wk.rows != kvd || L->wv.rows != kvd || L->wo.cols != qd || L->wo.rows != D ||
            L->ik_gate.rows != qd || L->ik_rel.rows != m->n_rel * m->n_head || L->ik_sink.rows != m->n_head ||
            L->ik_sink.cols != D || L->router.type != GGML_F32 || L->router.rows != m->n_expert ||
            L->gate_exps.rows != 2 * m->ff_exp || L->gate_exps.cols != D || L->down_exps.cols != m->ff_exp ||
            L->down_exps.rows != D) {
            snprintf(err, errlen, "layer %d: tensors of unexpected shape", l);
            return 0;
        }
    }
    m->ik_loop = (IkLoop *)calloc((size_t)(m->loop_end - m->loop_start), sizeof(IkLoop));
    static const char *ctl[4] = { "q", "k", "v", "ffn" };
    for (int l = m->loop_start; l < m->loop_end; l++) {
        IkLoop *lp = &m->ik_loop[l - m->loop_start];
        REQP(lp->msg_query = load_f32(m, T("loop_msg_query.weight"), 4 * D, 1, err, errlen));
        for (int k = 0; k < 4; k++) {
            char base[64];
            snprintf(base, sizeof(base), "blk.%d.loop_%s_ctrl_", l, ctl[k]);
            snprintf(nm, sizeof(nm), "%sdown.weight", base);
            REQ(load_mat(m, &lp->down[k], nm, 1, err, errlen));
            snprintf(nm, sizeof(nm), "%sgain.weight", base);
            REQ(load_mat(m, &lp->gain[k], nm, 1, err, errlen));
            snprintf(nm, sizeof(nm), "%sforget.weight", base);
            REQ(load_mat(m, &lp->forget[k], nm, 1, err, errlen));
            if (lp->down[k].type != GGML_F32 || lp->down[k].rows != m->gate_rank || lp->down[k].cols != 2 * D ||
                lp->gain[k].type != GGML_F32 || lp->gain[k].rows != D || lp->gain[k].cols != m->gate_rank ||
                lp->forget[k].type != GGML_F32 || lp->forget[k].rows != D || lp->forget[k].cols != m->gate_rank) {
                snprintf(err, errlen, "layer %d: loop controller of unexpected shape", l);
                return 0;
            }
        }
    }
#undef T
    /* every layer once, the core again for each extra pass */
    uint64_t b = (uint64_t)m->output.row_bytes * m->n_vocab;
    for (int s = 0; s < m->n_slot; s++) {
        const Layer *L = &m->layers[s < m->n_layer ? s : m->loop_start + (s - m->n_layer) % (m->loop_end - m->loop_start)];
        b += (uint64_t)L->wq.expert_stride + L->wk.expert_stride + L->wv.expert_stride + L->wo.expert_stride +
             L->ik_gate.expert_stride + L->ik_rel.expert_stride +
             (uint64_t)m->n_expert_used * (L->gate_exps.expert_stride + L->down_exps.expert_stride);
    }
    m->bytes_per_token = b;
    return 1;
fail:
    return 0;
}

int model_load(Model *m, const char *path, char *err, int errlen) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gguf, path, err, errlen)) return 0;
    GGUF *g = &m->gguf;
    uint64_t al = 0;
    const char *arch = gguf_get_str(g, "general.architecture", &al);
    if (!arch || al >= sizeof(m->arch)) { snprintf(err, errlen, "no architecture"); goto fail; }
    memcpy(m->arch, arch, al);
    m->arch[al] = 0;
    if (!strcmp(m->arch, "qwen35moe")) {
        if (load_qwen35(m, err, errlen)) return 1;
        goto fail;
    }
    if (!strcmp(m->arch, "inkling2_loop")) {
        if (load_inkling2(m, err, errlen)) return 1;
        goto fail;
    }
    if (strcmp(m->arch, "lfm2") && strcmp(m->arch, "lfm2moe") && strcmp(m->arch, "lfm3")) {
        snprintf(err, errlen, "unsupported architecture '%s' (need lfm2 / lfm2moe / lfm3 / qwen35moe / inkling2_loop)", m->arch);
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
    m->n_rot = m->head_dim;
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
    m->bytes_per_token = token_bytes(m);
    if (!m->n_kv_head) m->n_kv_head = m->n_head;
    return 1;
fail:
    model_free(m);
    return 0;
}

int model_mat(Model *m, const char *name, Mat *mat, char *err, int errlen) {
    return load_mat(m, mat, name, 1, err, errlen) > 0;
}

const float *model_f32(Model *m, const char *name, int n, char *err, int errlen) {
    return load_f32(m, name, n, 1, err, errlen);
}

/* The speech encoder of an LFM3-Audio projector (mmproj, projector type lfm3a): 17 LFM2 blocks,
 * short-conv or attention, every attention layer causal with a sliding window, full NEOX RoPE. */
int model_load_audio_encoder(Model *m, const char *path, char *err, int errlen) {
    GGUF g;
    if (!gguf_open(&g, path, err, errlen)) return 0;
    uint64_t pl = 0;
    const char *pt = gguf_get_str(&g, "clip.audio.projector_type", &pl);
    const int ok = pt && pl == 5 && !memcmp(pt, "lfm3a", 5);
    const int window = (int)gguf_get_int(&g, "clip.audio.window_size", 0);
    gguf_close(&g);
    if (!ok || window <= 0) { snprintf(err, errlen, "%s: not an LFM3-Audio projector", path); return 0; }
    return model_load_lfm_part(m, path, "a.blk.", "a.post_ln.weight", window, 0, err, errlen);
}

int model_load_lfm_part(Model *m, const char *path, const char *prefix, const char *final_norm, int window, int bidir,
                        char *err, int errlen) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gguf, path, err, errlen)) return 0;
    GGUF *g = &m->gguf;
    char nm[160];
#define T(s) (snprintf(nm, sizeof(nm), "%s%d." s, prefix, l), nm)
    int l = 0;
    while (gguf_tensor(g, T("ln1.weight"))) l++;
    m->n_layer = l;
    l = 0;
    const GGUFTensor *ln = gguf_tensor(g, T("ln1.weight"));
    snprintf(m->arch, sizeof(m->arch), "lfm3a");
    m->dim = ln ? (int)ln->ne[0] : 0;
    m->eps = 1e-5f;
    m->rope_base = m->rope_base_swa = 1e6f;   /* fixed, as in llama.cpp (checked by its converter) */
    m->swa_window = window;
    m->bidir = bidir;
    for (l = 0; l < m->n_layer; l++) {        /* the sizes, from the first layer that has each block */
        const GGUFTensor *qn = gguf_tensor(g, T("attn_q_norm.weight")), *q = gguf_tensor(g, T("attn_q.weight"));
        const GGUFTensor *k = gguf_tensor(g, T("attn_k.weight")), *cw = gguf_tensor(g, T("conv_dw.weight"));
        const GGUFTensor *fg = gguf_tensor(g, T("ffn_gate.weight"));
        if (qn && q && k && !m->n_head) {
            m->head_dim = (int)qn->ne[0];
            m->n_head = (int)q->ne[1] / m->head_dim;
            m->n_kv_head = (int)k->ne[1] / m->head_dim;
        }
        if (cw && !m->l_cache) m->l_cache = (int)cw->ne[0];
        if (fg && !m->ff_dense) m->ff_dense = (int)fg->ne[1];
    }
    if (!m->l_cache) m->l_cache = 3;
    m->n_rot = m->head_dim;
    if (m->n_layer <= 0 || m->dim <= 0 || m->n_head <= 0 || m->head_dim <= 0 || m->head_dim > 256) {
        snprintf(err, errlen, "%s: no usable %s* blocks", path, prefix);
        goto fail;
    }
    REQP(m->final_norm = load_f32(m, final_norm, m->dim, 1, err, errlen));
    m->layers = (Layer *)calloc((size_t)m->n_layer, sizeof(Layer));
    for (l = 0; l < m->n_layer; l++) {
        Layer *L = &m->layers[l];
        L->is_attn = gguf_tensor(g, T("attn_q.weight")) != NULL;
        L->is_swa = L->is_attn && window > 0;
        REQP(L->attn_norm = load_f32(m, T("ln1.weight"), m->dim, 1, err, errlen));
        REQP(L->ffn_norm = load_f32(m, T("ln2.weight"), m->dim, 1, err, errlen));
        L->post_op_norm = load_f32(m, T("attn_post_norm.weight"), m->dim, 0, err, errlen);
        L->post_ffn_norm = load_f32(m, T("ffn_post_norm.weight"), m->dim, 0, err, errlen);
        if (L->is_attn) {
            L->attn_idx = m->n_attn++;
            REQ(load_mat(m, &L->wq, T("attn_q.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wk, T("attn_k.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wv, T("attn_v.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->wo, T("attn_out.weight"), 1, err, errlen));
            REQP(L->q_norm = load_f32(m, T("attn_q_norm.weight"), m->head_dim, 1, err, errlen));
            REQP(L->k_norm = load_f32(m, T("attn_k_norm.weight"), m->head_dim, 1, err, errlen));
            if (load_mat(m, &L->attn_gate, T("attn_gate.weight"), 0, err, errlen) < 0) goto fail;
        } else {
            L->conv_idx = m->n_conv++;
            REQ(load_mat(m, &L->in_proj, T("conv_pw1.weight"), 1, err, errlen));
            REQ(load_mat(m, &L->out_proj, T("conv_pw2.weight"), 1, err, errlen));
            REQP(L->conv_w = load_f32(m, T("conv_dw.weight"), m->dim * m->l_cache, 1, err, errlen));
        }
        REQ(load_mat(m, &L->w_gate, T("ffn_gate.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->w_up, T("ffn_up.weight"), 1, err, errlen));
        REQ(load_mat(m, &L->w_down, T("ffn_down.weight"), 1, err, errlen));
    }
#undef T
    return 1;
fail:
    model_free(m);
    return 0;
}

void model_free(Model *m) {
    for (int i = 0; i < m->n_owned; i++) free(m->owned[i]);
    free(m->owned);
    free(m->ik_loop);
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
    Dx *dx;                     /* optional Direct3D 12 GPU for prompt matmuls */
    int dx_min;                 /* tokens per batch from which matmuls go to dx */
    int dx_dense;               /* single tokens entirely on dx (same KV / conv bookkeeping as gpu_dense) */
    int dx_chunks;              /* prompt chunks too */
    Vulkan *vk;                 /* optional Vulkan GPU for matmuls (instead of dx) */
    int vk_min;                 /* tokens per batch from which matmuls go to vk */
    int vk_exp;                 /* the Vulkan pass keeps the most used experts in VRAM */
    int vk_dense, vk_chunks;    /* as dx_dense / dx_chunks, on vk */
    int gpu_lost;               /* the whole-pass GPU failed with the KV cache and state on it */
    int n_lost;                 /* how many times that happened (ctx_lost) */
    float *rope;                /* [2][n_batch][head_dim] RoPE tables of a chunk's positions */
    /* dense decode path: single tokens run on the GPU, which keeps its own copy of the KV cache
     * and of the convolution state. Rows [0, kv_cpu_n) / [0, kv_gpu_n) are valid on each side. */
    int gpu_dense, kv_cpu_n, kv_gpu_n, conv_cpu_ok, conv_gpu_ok;
    int *on_gpu, *ga_e, *ga_t;  /* [n_batch * K] GPU assignment bookkeeping */
    float *gy;                  /* [n_batch * K][dim] GPU expert outputs */
    int pred[64], pred_n, pred_layer;
    int rlog_ready;             /* moe_ffn: rlog already holds the router's logits (the GPU's) */
    const float *pred_lg;       /* predict_next_experts: the next router's logits (the GPU's), or NULL */
    uint32_t tok_counter;       /* tokens processed since creation (never reset) */
    uint32_t *last_used;        /* [layer][expert] -> tok_counter at last use */
    f16_t *kc, *vc;
    /* the recurrent state: [n_conv][l_cache-1][dim] short-conv inputs, then per DeltaNet layer
     * its last ssm_conv_k-1 qkv rows and the S matrices [ssm_vh][ssm_d][ssm_d] */
    float *conv;
    size_t delta_off, delta_size;   /* in floats: where the DeltaNet part starts, one layer's share */
    float *qg;                  /* [n_batch][2 * n_head * head_dim] attn_q output with the gates (q_gate) */
    float inv_freq[256], inv_freq_swa[256];
    float *gate;                /* [n_batch][n_head] attention output gates (LFM3) */
    const float *ext;           /* [n_ext][dim] input rows of the token ids -1, -2... (ctx_set_embeds) */
    int hidden_ok;              /* c->h holds the last token's final-norm state (ctx_hidden) */
    int need_hidden;            /* ctx_need_hidden */
    int n_ext;
    /* scratch (sized for n_batch tokens) */
    float *x, *h, *big, *tmp, *out, *logits;
    block_q8_K *hq, *tq;
    float *fg, *fu; block_q8_K *fq;
    float *rlog, *selw; int *sel;
    int *a_tok, *a_slot, *tok_a;
    block_q8_K *xg; float *eg, *eu; block_q8_K *eq; float *eo;
    float *scores;   /* [n_threads][n_ctx] */
    /* Inkling 2 scratch, n_batch rows each (ik_layers): the prelude output, two buffers per depth
     * block (the core's alternate between passes), the previous pass's output, the messages
     * [4][rows][dim], the attention inputs [3][rows][dim], projections, controllers, experts */
    struct {
        float *arena, *pre, *blk[16], *bank_out, *msg, *in, *q, *k, *v, *r, *g, *s, *kd, *vd;
        float *z, *cg, *gain, *forget, *gu, *act, *tab, *mix;
        block_q8_K *qin, *eq;
    } ik;
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
    int B = c->n_batch, D = m->dim, kvd = m->n_kv_head * m->head_dim, qd = m->n_head * m->head_dim;
    int nbd = D / QK_K;
    size_t kvn = (size_t)m->n_attn * n_ctx * kvd;
    c->kc = (f16_t *)aligned_malloc(kvn * sizeof(f16_t) + 64, 64);
    c->vc = (f16_t *)aligned_malloc(kvn * sizeof(f16_t) + 64, 64);
    const int vd = m->ssm_vh * m->ssm_d, qkv = 2 * m->ssm_kh * m->ssm_d + vd;
    c->delta_off = (size_t)(m->n_conv ? m->n_conv : 1) * (m->l_cache - 1) * D;
    c->delta_size = m->n_delta ? (size_t)(m->ssm_conv_k - 1) * qkv + (size_t)vd * m->ssm_d : 0;
    if (m->ik) c->delta_off = (size_t)m->n_slot * (m->l_cache - 1) * m->ik_ch;   /* the conv inputs of every slot */
    c->conv = (float *)amalloc(sizeof(float) * (c->delta_off + (size_t)m->n_delta * c->delta_size));
    for (int i = 0; i < m->n_rot / 2; i++) {
        c->inv_freq[i] = (float)pow((double)m->rope_base, -2.0 * i / m->n_rot);
        if (m->rope_freqs) c->inv_freq[i] /= m->rope_freqs[i]; /* a huge divisor leaves the pair unrotated */
        c->inv_freq_swa[i] = (float)pow((double)m->rope_base_swa, -2.0 * i / m->n_rot);
    }
    /* widest rows: short conv [B|C|x], attention [q|k|v], DeltaNet [qkv|z|alpha|beta] */
    int bigw = 3 * D;
    if (qd + 2 * kvd > bigw) bigw = qd + 2 * kvd;
    if (m->n_delta && qkv + vd + 2 * m->ssm_vh > bigw) bigw = qkv + vd + 2 * m->ssm_vh;
    int tw = D;                 /* the input of the output projections */
    if (qd > tw) tw = qd;
    if (vd > tw) tw = vd;
    c->x = (float *)amalloc(sizeof(float) * B * D);
    c->h = (float *)amalloc(sizeof(float) * B * D);
    c->big = (float *)amalloc(sizeof(float) * B * bigw);
    c->tmp = (float *)amalloc(sizeof(float) * B * tw);
    c->out = (float *)amalloc(sizeof(float) * B * D);
    c->gate = (float *)amalloc(sizeof(float) * B * (m->q_gate ? qd : m->n_head));
    if (m->q_gate) c->qg = (float *)amalloc(sizeof(float) * B * 2 * qd);
    c->rope = (float *)amalloc(sizeof(float) * 2 * B * m->head_dim);
    c->logits = (float *)amalloc(sizeof(float) * m->n_vocab);
    c->hq = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * nbd);
    c->tq = (block_q8_K *)amalloc(sizeof(block_q8_K) * B * (tw / QK_K));
    c->last_used = (uint32_t *)calloc((size_t)m->n_layer * (m->n_expert ? m->n_expert : 1), sizeof(uint32_t));
    int ffd = m->ff_dense > 0 ? m->ff_dense : QK_K;
    if (m->ff_shexp > ffd) ffd = m->ff_shexp;
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
    if (m->ik) {
        const size_t BD = (size_t)B * D, F = (size_t)m->ff_exp, FP = (F + QK_K - 1) / QK_K * QK_K, BK = (size_t)B * k;
        const int nblk = 2 * (m->n_layer / m->res_block), lc = m->l_cache;
        const size_t ext = (size_t)(m->rel_extent > m->swa_window ? m->rel_extent : m->swa_window) + 1;
        const size_t sz[] = { BD, BD * nblk, BD, 4 * BD, 3 * BD, (size_t)B * qd, (size_t)B * kvd, (size_t)B * kvd,
                              (size_t)B * m->n_head * m->n_rel, (size_t)B * qd, (size_t)B * m->n_head,
                              (size_t)B * lc * m->n_kv_head, (size_t)B * lc * m->n_kv_head, 2 * BD,
                              (size_t)B * m->gate_rank, BD, BD, BK * 2 * F, BK * FP, (size_t)c->n_threads * ext, BD };
        size_t tot = 0;
        for (size_t i = 0; i < sizeof(sz) / sizeof(sz[0]); i++) tot += (sz[i] + 15) / 16 * 16;
        float *p = c->ik.arena = (float *)amalloc(sizeof(float) * tot);
        if (p) {
            float **dst[] = { &c->ik.pre, NULL, &c->ik.bank_out, &c->ik.msg, &c->ik.in, &c->ik.q, &c->ik.k, &c->ik.v,
                              &c->ik.r, &c->ik.g, &c->ik.s, &c->ik.kd, &c->ik.vd, &c->ik.z, &c->ik.cg, &c->ik.gain,
                              &c->ik.forget, &c->ik.gu, &c->ik.act, &c->ik.tab, &c->ik.mix };
            for (size_t i = 0; i < sizeof(sz) / sizeof(sz[0]); i++) {
                if (dst[i]) *dst[i] = p;
                else for (int b = 0; b < nblk; b++) c->ik.blk[b] = p + (size_t)b * BD;
                p += (sz[i] + 15) / 16 * 16;
            }
        }
        c->ik.qin = (block_q8_K *)amalloc(sizeof(block_q8_K) * 3 * B * nbd);
        c->ik.eq = (block_q8_K *)amalloc(sizeof(block_q8_K) * BK * (FP / QK_K));
        if (!c->ik.arena || !c->ik.qin || !c->ik.eq) { ctx_free(c); return NULL; }
    }
    if (!c->kc || !c->vc || !c->scores || !c->eo) { ctx_free(c); return NULL; }
    return c;
}

void ctx_free(Context *c) {
    if (!c) return;
    if (c->pfq) prefetcher_destroy(c->pfq);
    void *a[] = { c->kc, c->vc, c->conv, c->x, c->h, c->big, c->tmp, c->out, c->gate, c->rope, c->logits, c->hq, c->tq,
                  c->fg, c->fu, c->fq, c->rlog, c->selw, c->sel, c->a_tok, c->a_slot, c->tok_a, c->xg,
                  c->eg, c->eu, c->eq, c->eo, c->scores, c->gy, c->on_gpu, c->ga_e, c->ga_t, c->qg,
                  c->ik.arena, c->ik.qin, c->ik.eq };
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); i++) aligned_free(a[i]);
    free(c->task_job); free(c->task_r0); free(c->task_r1); free(c->last_used);
    free(c);
}

void ctx_set_gpu(Context *c, struct Gpu *g) {
    c->gpu = g;
    const char *env = getenv("PITHOS_GPU_DENSE");
    c->gpu_dense = g && gpu_dense_on(g) && !(env && env[0] == '0');
    c->kv_gpu_n = 0;
    c->conv_gpu_ok = 0;
}

int ctx_set_dx(Context *c, struct Dx *d, char *info, int infolen) {
    c->dx = d;
    const char *env = getenv("PITHOS_DX_MIN"), *dense = getenv("PITHOS_DX_DENSE"), *chunks = getenv("PITHOS_DX_CHUNKS");
    c->dx_min = env && atoi(env) > 0 ? atoi(env) : DX_MIN_TOKENS;
    info[0] = 0;
    c->dx_dense = d && !c->gpu_dense && !(dense && dense[0] == '0') &&
                  dx_dense_init(d, c->m, c->n_ctx, c->n_batch, info, infolen);
    c->dx_chunks = c->dx_dense && !(chunks && chunks[0] == '0');   /* =0: prompts as before (matmuls only) */
    c->kv_gpu_n = 0;
    c->conv_gpu_ok = 0;
    return c->dx_dense;
}

int ctx_set_vk(Context *c, struct Vulkan *v, char *info, int infolen) {
    c->vk = v;
    const char *env = getenv("PITHOS_VK_MIN"), *dense = getenv("PITHOS_VK_DENSE"), *chunks = getenv("PITHOS_VK_CHUNKS");
    c->vk_min = env && atoi(env) > 0 ? atoi(env) : DX_MIN_TOKENS;
    info[0] = 0;
    c->vk_dense = v && !c->gpu_dense && !c->dx_dense && !(dense && dense[0] == '0') &&
                  vk_dense_init(v, c->m, c->n_ctx, c->n_batch, info, infolen);
    c->vk_chunks = c->vk_dense && !(chunks && chunks[0] == '0');   /* =0: prompts through the matmuls only */
    c->vk_exp = c->vk_dense && vk_experts_on(v);
    c->kv_gpu_n = 0;
    c->conv_gpu_ok = 0;
    return c->vk_dense;
}

/* make the host copy of the state current (before CPU work or a snapshot) */
static void sync_to_cpu(Context *c, int kv_rows) {
    if (!c->gpu_dense && !c->dx_dense && !c->vk_dense) return;
    if (c->kv_cpu_n < kv_rows) {
        if (c->gpu_dense) gpu_dense_kv(c->gpu, c->kc, c->vc, c->kv_cpu_n, kv_rows, 0);
        else if (c->dx_dense) dx_dense_kv(c->dx, c->kc, c->vc, c->kv_cpu_n, kv_rows, 0);
        else vk_dense_kv(c->vk, c->kc, c->vc, c->kv_cpu_n, kv_rows, 0);
        c->kv_cpu_n = kv_rows;
    }
    if (!c->conv_cpu_ok) {
        if (c->gpu_dense) gpu_dense_conv(c->gpu, c->conv, 0);
        else if (c->dx_dense) dx_dense_conv(c->dx, c->conv, 0);
        else vk_dense_conv(c->vk, c->conv, 0);
        c->conv_cpu_ok = 1;
    }
}
int ctx_pos(const Context *c) { return c->pos; }
int ctx_lost(const Context *c) { return c->n_lost; }
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
    memset(c->conv, 0, sizeof(float) * (c->delta_off + (size_t)c->m->n_delta * c->delta_size));
}

static size_t conv_state_size(const Context *c) {
    return sizeof(float) * (c->delta_off + (size_t)c->m->n_delta * c->delta_size);
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

/* the whole batch on the Direct3D 12 GPU, if every job is a quantized matmul it holds over enough tokens */
static int dx_run(Context *c) {
    DxJob dj[16];
    if (!c->dx || c->njobs > 16) return 0;
    for (int ji = 0; ji < c->njobs; ji++) {
        const MatJob *j = &c->jobs[ji];
        if (!j->xq || j->nt < c->dx_min || !dx_has(c->dx, j->w)) return 0;
        DxJob d = {j->w, j->xq, j->xstride, j->nt, j->out, j->ostride};
        dj[ji] = d;
    }
    return dx_matmul(c->dx, dj, c->njobs);
}

/* the same on the Vulkan GPU */
static int vk_run(Context *c) {
    VulkanJob vj[16];
    if (!c->vk || c->njobs > 16) return 0;
    for (int ji = 0; ji < c->njobs; ji++) {
        const MatJob *j = &c->jobs[ji];
        if (!j->xq || j->nt < c->vk_min || !vk_has(c->vk, j->w)) return 0;
        VulkanJob v = {j->w, j->xq, j->xstride, j->nt, j->out, j->ostride};
        vj[ji] = v;
    }
    return vk_matmul(c->vk, vj, c->njobs);
}

static void cpu_run(Context *c);

/* the same through OpenCL, on the dense weights its decode path keeps in VRAM */
static int cl_run(Context *c) {
    GpuMatJob gj[16];
    if (!c->gpu || c->njobs > 16) return 0;
    for (int ji = 0; ji < c->njobs; ji++) {
        const MatJob *j = &c->jobs[ji];
        if (!j->xq || !gpu_mm_has(c->gpu, j->w, j->nt)) return 0;
        GpuMatJob g = {j->w, j->xq, j->xstride, j->nt, j->out, j->ostride};
        gj[ji] = g;
    }
    if (!gpu_matmul(c->gpu, gj, c->njobs)) return 0;
    if (getenv("PITHOS_CL_CHECK")) { /* debugging: redo the jobs on the CPU (its results stay) and report differences */
        float *keep[16];
        for (int ji = 0; ji < c->njobs; ji++) {
            const MatJob *j = &c->jobs[ji];
            keep[ji] = (float *)malloc(sizeof(float) * (size_t)j->rows * j->nt);
            for (int t = 0; t < j->nt && keep[ji]; t++) memcpy(keep[ji] + (size_t)t * j->rows, j->out + (size_t)t * j->ostride, sizeof(float) * j->rows);
        }
        cpu_run(c);
        for (int ji = 0; ji < c->njobs; ji++) {
            const MatJob *j = &c->jobs[ji];
            double num = 0, den = 0;
            for (int t = 0; t < j->nt && keep[ji]; t++)
                for (int r = 0; r < j->rows; r++) {
                    const double a = keep[ji][(size_t)t * j->rows + r], b = j->out[(size_t)t * j->ostride + r];
                    num += (a - b) * (a - b);
                    den += b * b;
                }
            if (!(num <= 1e-10 * den)) /* NaN too */
                fprintf(stderr, "PITHOS_CL_CHECK: type %d %dx%d, %d tokens: rel err %.2e\n", (int)j->type, j->rows, j->K, j->nt,
                        sqrt(num / (den + 1e-30)));
            free(keep[ji]);
        }
    }
    return 1;
}

/* split the jobs into row ranges for the thread pool */
static void cpu_run(Context *c) {
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
}

static void job_run(Context *c, int cat) {
    double t0 = time_sec();
    if (!dx_run(c) && !vk_run(c) && !cl_run(c)) cpu_run(c);
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


/* NEOX rotation of the first n_rot values of a head: pairs (i, i + n_rot / 2) */
static void rope_neox(float *v, int n_rot, int pos, const float *inv_freq) {
    const int half = n_rot / 2;
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
    const int hd = m->head_dim, nh = m->n_head, kvd = m->n_kv_head * hd, qd = nh * hd;
    const int t = task / nh, h = task % nh;
    const int kvh = h / (nh / m->n_kv_head);
    const int qkvw = qd + 2 * kvd;
    const float *q = c->big + (size_t)t * qkvw + h * hd;
    const int npos = m->bidir ? a->pos0 + a->n : a->pos0 + t + 1;
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
    float *o = c->tmp + (size_t)t * qd + h * hd;
    float acc[512];
    memset(acc, 0, sizeof(float) * hd);
    for (int j = j0; j < npos; j++) KS.axpy_f16(s[j - j0] * inv, vc + (size_t)j * kvd, acc, hd);
    if (m->q_gate) {   /* Qwen3.5: elementwise sigmoid gate */
        const float *g = c->gate + (size_t)t * qd + h * hd;
        for (int i = 0; i < hd; i++) acc[i] *= g[i];
    }
    memcpy(o, acc, sizeof(float) * hd);
}

static void attention_block(Context *c, const Layer *L, int n) {
    const Model *m = c->m;
    const int D = m->dim, hd = m->head_dim, kvd = m->n_kv_head * hd, nbd = D / QK_K, qd = m->n_head * hd;
    const int qkvw = qd + 2 * kvd;
    quantize_rows(c->h, n, D, c->hq);
    job_clear(c);
    if (m->q_gate) job_add_q(c, &L->wq, NULL, c->hq, nbd, n, c->qg, 2 * qd);
    else job_add_q(c, &L->wq, NULL, c->hq, nbd, n, c->big, qkvw);
    job_add_q(c, &L->wk, NULL, c->hq, nbd, n, c->big + qd, qkvw);
    job_add_q(c, &L->wv, NULL, c->hq, nbd, n, c->big + qd + kvd, qkvw);
    if (L->attn_gate.data) job_add_q(c, &L->attn_gate, NULL, c->hq, nbd, n, c->gate, m->n_head);
    job_run(c, CAT_ATTN);

    double t0 = time_sec();
    if (L->attn_gate.data)
        for (int i = 0; i < n * m->n_head; i++) c->gate[i] = 1.f / (1.f + expf(-c->gate[i]));
    if (m->q_gate)   /* per head [q | gate]: q into its row, sigmoid(gate) aside */
        for (int t = 0; t < n; t++)
            for (int h = 0; h < m->n_head; h++) {
                const float *src = c->qg + (size_t)t * 2 * qd + (size_t)h * 2 * hd;
                float *g = c->gate + (size_t)t * qd + h * hd;
                memcpy(c->big + (size_t)t * qkvw + h * hd, src, sizeof(float) * hd);
                for (int i = 0; i < hd; i++) g[i] = 1.f / (1.f + expf(-src[hd + i]));
            }
    const float *inv_freq = L->is_swa ? c->inv_freq_swa : c->inv_freq;
    const size_t lbase = (size_t)L->attn_idx * c->n_ctx * kvd;
    for (int t = 0; t < n; t++) {
        float *q = c->big + (size_t)t * qkvw;
        float *k = q + qd, *v = q + qd + kvd;
        const int pos = c->pos + t;
        for (int h = 0; h < m->n_head; h++) {
            rmsnorm(q + h * hd, q + h * hd, L->q_norm, hd, m->eps);
            rope_neox(q + h * hd, m->n_rot, pos, inv_freq);
        }
        for (int h = 0; h < m->n_kv_head; h++) {
            rmsnorm(k + h * hd, k + h * hd, L->k_norm, hd, m->eps);
            rope_neox(k + h * hd, m->n_rot, pos, inv_freq);
        }
        KS.f32_to_f16_row(k, c->kc + lbase + (size_t)pos * kvd, kvd);
        KS.f32_to_f16_row(v, c->vc + lbase + (size_t)pos * kvd, kvd);
    }
    AttnArgs a = { c, L, n, c->pos };
    pool_run(c->pool, attn_task, &a, n * m->n_head);
    c->stats.t_attn += time_sec() - t0;

    quantize_rows(c->tmp, n, qd, c->tq);
    job_clear(c);
    job_add_q(c, &L->wo, NULL, c->tq, qd / QK_K, n, c->out, D);
    job_run(c, CAT_ATTN);
}

/* ------------------------------------------------------------------ Gated DeltaNet */

typedef struct {
    Context *c;
    const Layer *L;
    int n;
    float *st;          /* the layer's S matrices [vh][d][d], S[i][j]: key i, value j */
} DeltaArgs;

/* one V head over the n tokens of the chunk, in order (the recurrence is sequential in time) */
static void delta_task(void *ctx, int j, int tid) {
    (void)tid;
    DeltaArgs *a = (DeltaArgs *)ctx;
    Context *c = a->c;
    const Model *m = c->m;
    const int d = m->ssm_d, kd = m->ssm_kh * d, vd = m->ssm_vh * d, qkv = 2 * kd + vd;
    const int bw = qkv + vd + 2 * m->ssm_vh, kh = j % m->ssm_kh;   /* V heads tiled over the K heads */
    float *S = a->st + (size_t)j * d * d;
    float u[256], o[256];
    for (int t = 0; t < a->n; t++) {
        const float *row = c->big + (size_t)t * bw;
        const float *q = row + kh * d, *k = row + kd + kh * d, *v = row + 2 * kd + j * d;
        const float alpha = row[qkv + vd + j], beta = row[qkv + vd + m->ssm_vh + j];
        const float x = alpha + a->L->ssm_dt[j];
        const float sp = x > 20.f ? x : log1pf(expf(x));                 /* softplus */
        const float decay = expf(sp * a->L->ssm_a[j]);                   /* ssm_a = -exp(A_log) */
        const float b = 1.f / (1.f + expf(-beta));
        /* S *= decay; u = beta (v - S^T k) */
        for (int i = 0; i < d; i++) u[i] = 0.f;
        for (int i = 0; i < d; i++) {
            float *Si = S + (size_t)i * d;
            const float ki = k[i];
            for (int jj = 0; jj < d; jj++) { Si[jj] *= decay; u[jj] += Si[jj] * ki; }
        }
        for (int jj = 0; jj < d; jj++) u[jj] = b * (v[jj] - u[jj]);
        /* S += k u^T; o = S^T q */
        for (int jj = 0; jj < d; jj++) o[jj] = 0.f;
        for (int i = 0; i < d; i++) {
            float *Si = S + (size_t)i * d;
            const float ki = k[i], qi = q[i];
            for (int jj = 0; jj < d; jj++) { Si[jj] += ki * u[jj]; o[jj] += Si[jj] * qi; }
        }
        /* rmsnorm per head times silu(z) */
        const float *z = row + qkv + j * d;
        double ss = 0;
        for (int jj = 0; jj < d; jj++) ss += (double)o[jj] * o[jj];
        const float sc = (float)(1.0 / sqrt(ss / d + m->eps));
        float *dst = c->tmp + (size_t)t * vd + j * d;
        for (int jj = 0; jj < d; jj++) dst[jj] = o[jj] * sc * a->L->ssm_norm[jj] * (z[jj] / (1.f + expf(-z[jj])));
    }
}

static void deltanet_block(Context *c, const Layer *L, int n) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, d = m->ssm_d, kd = m->ssm_kh * d, vd = m->ssm_vh * d, vh = m->ssm_vh;
    const int qkv = 2 * kd + vd, bw = qkv + vd + 2 * vh, K = m->ssm_conv_k;
    quantize_rows(c->h, n, D, c->hq);
    job_clear(c);
    job_add_q(c, &L->ssm_qkv, NULL, c->hq, nbd, n, c->big, bw);
    job_add_q(c, &L->ssm_z, NULL, c->hq, nbd, n, c->big + qkv, bw);
    job_add_q(c, &L->ssm_alpha, NULL, c->hq, nbd, n, c->big + qkv + vd, bw);
    job_add_q(c, &L->ssm_beta, NULL, c->hq, nbd, n, c->big + qkv + vd + vh, bw);
    job_run(c, CAT_CONV);

    double t0 = time_sec();
    float *cs = c->conv + c->delta_off + (size_t)L->delta_idx * c->delta_size;   /* [K-1][qkv], oldest first */
    for (int t = 0; t < n; t++) {
        float *x = c->big + (size_t)t * bw;
        /* causal depthwise conv over the last K inputs, then silu (in place) */
        for (int ch = 0; ch < qkv; ch++) {
            const float *w = L->ssm_conv + (size_t)ch * K, xin = x[ch];
            float acc = w[K - 1] * xin;
            for (int k = 0; k < K - 1; k++) acc += w[k] * cs[(size_t)k * qkv + ch];
            for (int k = 0; k < K - 2; k++) cs[(size_t)k * qkv + ch] = cs[(size_t)(k + 1) * qkv + ch];
            cs[(size_t)(K - 2) * qkv + ch] = xin;
            x[ch] = acc / (1.f + expf(-acc));
        }
        /* q and k: L2 norm per head (x / max(|x|, eps)), q also times 1/sqrt(d) */
        for (int h = 0; h < 2 * m->ssm_kh; h++) {
            float *v = x + (size_t)h * d;
            double ss = 0;
            for (int i = 0; i < d; i++) ss += (double)v[i] * v[i];
            const double nrm = sqrt(ss);
            float s = (float)(1.0 / (nrm > m->eps ? nrm : m->eps));
            if (h < m->ssm_kh) s /= sqrtf((float)d);
            for (int i = 0; i < d; i++) v[i] *= s;
        }
    }
    DeltaArgs a = { c, L, n, cs + (size_t)(K - 1) * qkv };
    pool_run(c->pool, delta_task, &a, vh);
    c->stats.t_attn += time_sec() - t0;

    quantize_rows(c->tmp, n, vd, c->tq);
    job_clear(c);
    job_add_q(c, &L->ssm_out, NULL, c->tq, vd / QK_K, n, c->out, D);
    job_run(c, CAT_CONV);
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

/* out[n][dim] = down(silu(gate h) * up h) */
static void swiglu(Context *c, const Mat *gate, const Mat *up, const Mat *down, int n, float *out) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, F = gate->rows;
    quantize_rows(c->h, n, D, c->hq);
    job_clear(c);
    job_add_q(c, gate, NULL, c->hq, nbd, n, c->fg, F);
    job_add_q(c, up, NULL, c->hq, nbd, n, c->fu, F);
    job_run(c, CAT_DENSE);
    KS.silu_mul(c->fg, c->fu, n * F);
    quantize_rows(c->fg, n, F, c->fq);
    job_clear(c);
    job_add_q(c, down, NULL, c->fq, F / QK_K, n, out, D);
    job_run(c, CAT_DENSE);
}

static void dense_ffn(Context *c, const Layer *L, int n) { swiglu(c, &L->w_gate, &L->w_up, &L->w_down, n, c->out); }

/* the shared expert of Qwen3.5, weighted by sigmoid(sh_inp . h), added to c->out */
static void shared_expert(Context *c, const Layer *L, int n) {
    const int D = c->m->dim;
    swiglu(c, &L->sh_gate, &L->sh_up, &L->sh_down, n, c->tmp);
    for (int t = 0; t < n; t++) {
        const float g = 1.f / (1.f + expf(-KS.dot_f32(L->sh_inp, c->h + (size_t)t * D, D)));
        const float *s = c->tmp + (size_t)t * D;
        float *o = c->out + (size_t)t * D;
        for (int i = 0; i < D; i++) o[i] += g * s[i];
    }
}

/* the k best of sc[n], best first and the lower index first among equal scores: the same as k
 * passes of argmax over the experts not chosen yet, in one pass */
static void top_k(const float *sc, int n, int k, int *out) {
    int m = 0;
    for (int e = 0; e < n; e++) {
        if (m == k && !(sc[e] > sc[out[k - 1]])) continue;
        int j = m < k ? m++ : k - 1;
        for (; j > 0 && sc[e] > sc[out[j - 1]]; j--) out[j] = out[j - 1];
        out[j] = e;
    }
}

static void prefetch_expert(Context *c, const Layer *L, int e) {
    prefetcher_push(c->pfq, L->gate_exps.data + (size_t)e * L->gate_exps.expert_stride, L->gate_exps.expert_stride);
    if (L->up_exps.data)   /* Inkling: gate and up in one tensor */
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
    float hn[8192], sc[MAX_EXPERT] = {0};
    const float *rl = c->pred_lg;
    if (!rl) {
        if (D > 8192 || Ln->router.type != GGML_F32) return;
        rmsnorm(hn, c->x, Ln->ffn_norm, D, m->eps);
        /* the router of the next layer, on the threads (its logits into rlog, already consumed) */
        job_clear(c);
        job_add_f(c, &Ln->router, hn, D, 1, c->rlog, E);
        job_run(c, CAT_ROUTER);
        rl = c->rlog;
    }
    for (int e = 0; e < E; e++) {
        const float lg = rl[e];
        const float p = m->gating == 1 ? lg : 1.f / (1.f + expf(-lg));
        sc[e] = p + (Ln->exp_bias && m->gating != 1 ? Ln->exp_bias[e] : 0.f);
    }
    int np = m->n_expert_used + 2;
    if (np > E) np = E;
    prefetcher_clear(c->pfq);
    top_k(sc, E, np, c->pred);
    for (int k = 0; k < np; k++) {
        const int best = c->pred[k];
        /* experts used in the last few dozen tokens are almost surely resident: skip them */
        if (c->gpu && gpu_resident(c->gpu, ln, best)) continue; /* served from VRAM */
        if (c->vk_exp && vk_expert_resident(c->vk, ln, best)) continue;
        const uint32_t lu = c->last_used[(size_t)ln * E + best];
        if (c->tok_counter - lu > 48 || !lu) prefetch_expert(c, Ln, best);
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
    if (!c->rlog_ready) {
        job_clear(c);
        job_add_f(c, &L->router, c->h, D, n, c->rlog, E);
        job_run(c, CAT_ROUTER);
    }

    int cnt[MAX_EXPERT] = {0}, off[MAX_EXPERT + 1];
    for (int t = 0; t < n; t++) {
        float *lg = c->rlog + (size_t)t * E;
        float p[MAX_EXPERT], sc[MAX_EXPERT];
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
        top_k(sc, E, K, sel);
        for (int k = 0; k < K; k++) {
            const int best = sel[k];
            w[k] = p[best];
            wsum += p[best];
            if (c->gpu) gpu_note_usage(c->gpu, layer, best);
            if (c->vk_exp) vk_note_usage(c->vk, layer, best);
            if (layer < 64 && best < 64) c->stats.expert_hits[layer * 64 + best]++;
            c->last_used[(size_t)layer * E + best] = c->tok_counter;
        }
        if (m->norm_topk && wsum > 0)
            for (int k = 0; k < K; k++) w[k] /= wsum;
    }

    /* (token, expert) pairs whose expert is cached in VRAM run on the GPU, concurrently
     * with the CPU computing the others */
    int ng = 0;
    const int vk = !c->gpu && c->vk_exp && n * K <= VK_MOE_MAX && vk_moe_ready(c->vk, n * K);   /* the Vulkan pass's experts in VRAM */
    if (c->gpu || vk) {
        for (int i = 0; i < n * K; i++) {
            c->on_gpu[i] = vk ? vk_expert_resident(c->vk, layer, c->sel[i]) : gpu_resident(c->gpu, layer, c->sel[i]);
            if (c->on_gpu[i]) {
                c->ga_e[ng] = c->sel[i];
                c->ga_t[ng] = i / K;
                c->tok_a[i] = -1 - ng;
                ng++;
            }
        }
        if (ng && vk) {   /* the GPU adds them to the residual stream itself */
            float gw[VK_MOE_MAX];
            for (int i = 0; i < n * K; i++)
                if (c->on_gpu[i]) gw[-1 - c->tok_a[i]] = c->selw[i];
            vk_moe_begin(c->vk, layer, c->ga_e, c->ga_t, gw, ng);
        }
        else if (ng) gpu_moe_begin(c->gpu, layer, c->h, n, c->ga_e, c->ga_t, ng);
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
    int fill[MAX_EXPERT];
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
    if (ng && !vk) {
        double tw = time_sec();
        gpu_moe_end(c->gpu, c->gy);
        c->stats.t_gpu_wait += time_sec() - tw;
        if (gpu_failed(c->gpu)) c->gpu_lost = 1;   /* gy is not valid: ctx_forward drops the pass */
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
/* cos | sin of the RoPE angles at pos, for full and sliding-window attention layers */
static void rope_tables(const Context *c, int pos, float *rope, float *rope_swa) {
    const int hd = c->m->head_dim;
    for (int i = 0; i < hd / 2; i++) {
        const double a = (double)pos * c->inv_freq[i], as = (double)pos * c->inv_freq_swa[i];
        rope[i] = (float)cos(a);
        rope[hd / 2 + i] = (float)sin(a);
        rope_swa[i] = (float)cos(as);
        rope_swa[hd / 2 + i] = (float)sin(as);
    }
}

/* the input row of a token: its embedding, or row -id-1 of the ctx_set_embeds rows for id < 0 */
static void token_row(const Context *c, int id, float *dst) {
    const Model *m = c->m;
    if (id < 0 && -id - 1 < c->n_ext) {
        memcpy(dst, c->ext + (size_t)(-id - 1) * m->dim, sizeof(float) * (size_t)m->dim);
        return;
    }
    if (id < 0 || id >= m->n_vocab) id = 0;
    dequant_row(m->embed.type, m->embed.data + (size_t)id * m->embed.row_bytes, dst, m->dim);
}

void ctx_set_embeds(Context *c, const float *rows, int n) {
    c->ext = rows;
    c->n_ext = rows ? n : 0;
}

/* a token or a prompt chunk entirely on the Direct3D 12 GPU (models without experts) */
static void forward_dx(Context *c, const int *tokens, int n, float *logits) {
    const Model *m = c->m;
    const int D = m->dim, hd = m->head_dim, pos0 = c->pos;
    if (c->kv_gpu_n < pos0) {
        dx_dense_kv(c->dx, c->kc, c->vc, c->kv_gpu_n, pos0, 1);
        c->kv_gpu_n = pos0;
    }
    if (!c->conv_gpu_ok) {
        dx_dense_conv(c->dx, c->conv, 1);
        c->conv_gpu_ok = 1;
    }
    float *rope = c->rope, *rope_swa = c->rope + (size_t)n * hd;
    for (int t = 0; t < n; t++) {
        token_row(c, tokens[t], c->x + (size_t)t * D);
        rope_tables(c, pos0 + t, rope + (size_t)t * hd, rope_swa + (size_t)t * hd);
    }
    if (!dx_dense_forward(c->dx, c->x, n, rope, m->swa_window > 0 ? rope_swa : NULL, pos0, logits ? c->logits : NULL,
                          logits ? c->tmp : NULL)) {
        memset(c->logits, 0, sizeof(float) * (size_t)m->n_vocab);   /* the GPU failed: the message says so */
        c->dx_dense = 0;
        c->gpu_lost = 1;
    } else if (logits) {   /* the last residual row: ctx_hidden */
        rmsnorm(c->h, c->tmp, m->final_norm, D, m->eps);
        c->hidden_ok = 1;
    }
    if (logits && logits != c->logits) memcpy(logits, c->logits, sizeof(float) * (size_t)m->n_vocab);
    c->pos += n;
    c->tok_counter += (uint32_t)n;
    c->kv_gpu_n = c->pos;
    if (c->kv_cpu_n > pos0) c->kv_cpu_n = pos0;
    c->conv_cpu_ok = 0;
}

/* qwen35moe on the Vulkan GPU: the routed experts of one layer on the CPU (vk_dense_set_moe) */
static void vk_moe(void *arg, int layer, const float *h, const float *lg, const float *pred, int n, float *y, float *s) {
    Context *c = (Context *)arg;
    const Model *m = c->m;
    const Layer *L = &m->layers[layer];
    const int D = m->dim, E = m->n_expert;
    if (c->pfq && n >= 8 && layer + 1 < m->n_layer && m->layers[layer + 1].is_moe) {   /* as run_layers */
        prefetcher_clear(c->pfq);
        for (int e = 0; e < E; e++) prefetch_expert(c, &m->layers[layer + 1], e);
    }
    memcpy(c->h, h, sizeof(float) * (size_t)n * D);
    for (int t = 0; t < n; t++) {   /* the router's logits, then the shared expert's gate */
        memcpy(c->rlog + (size_t)t * E, lg + (size_t)t * (E + 1), sizeof(float) * (size_t)E);
        s[t] = 1.f / (1.f + expf(-lg[(size_t)t * (E + 1) + E]));
    }
    c->rlog_ready = 1;
    c->pred_lg = pred;
    moe_ffn(c, L, layer, n);
    c->rlog_ready = 0;
    c->pred_lg = NULL;
    memcpy(y, c->out, sizeof(float) * (size_t)n * D);
}

/* the same on the Vulkan GPU */
static void forward_vk(Context *c, const int *tokens, int n, float *logits) {
    const Model *m = c->m;
    const int D = m->dim, hd = m->head_dim, pos0 = c->pos;
    if (c->kv_gpu_n < pos0) {
        vk_dense_kv(c->vk, c->kc, c->vc, c->kv_gpu_n, pos0, 1);
        c->kv_gpu_n = pos0;
    }
    if (!c->conv_gpu_ok) {
        vk_dense_conv(c->vk, c->conv, 1);
        c->conv_gpu_ok = 1;
    }
    float *rope = c->rope, *rope_swa = c->rope + (size_t)n * hd;
    for (int t = 0; t < n; t++) {
        token_row(c, tokens[t], c->x + (size_t)t * D);
        rope_tables(c, pos0 + t, rope + (size_t)t * hd, rope_swa + (size_t)t * hd);
    }
    vk_dense_set_moe(c->vk, vk_moe, c);
    if (!vk_dense_forward(c->vk, c->x, n, rope, m->swa_window > 0 ? rope_swa : NULL, pos0, logits ? c->logits : NULL,
                          logits ? c->tmp : NULL)) {
        memset(c->logits, 0, sizeof(float) * (size_t)m->n_vocab);   /* the GPU failed: the message says so */
        c->vk_dense = 0;
        c->gpu_lost = 1;
    } else if (logits) {   /* the last residual row: ctx_hidden */
        rmsnorm(c->h, c->tmp, m->final_norm, D, m->eps);
        c->hidden_ok = 1;
    }
    if (logits && logits != c->logits) memcpy(logits, c->logits, sizeof(float) * (size_t)m->n_vocab);
    c->pos += n;
    c->tok_counter += (uint32_t)n;
    c->kv_gpu_n = c->pos;
    if (c->kv_cpu_n > pos0) c->kv_cpu_n = pos0;
    c->conv_cpu_ok = 0;
}

/* an OpenCL call failed: the device is not used again. lost: the results of this pass are not
 * valid (or the KV cache and state were on the GPU), so the context is lost like after a Vulkan
 * or D3D12 failure (lost_state); otherwise the CPU already redid the work and carries on. */
static void cl_lost(Context *c, int lost) {
    if (lost) {
        memset(c->logits, 0, sizeof(float) * (size_t)c->m->n_vocab);
        c->gpu_lost = 1;
    }
    if (c->gpu) fprintf(stderr, "OpenCL failed: the GPU is not used any more\n");
    c->gpu_dense = 0;
    c->gpu = NULL;
}

static void forward_token_gpu(Context *c, int token, float *all_logits) {
    const Model *m = c->m;
    const int D = m->dim, K = m->n_expert_used, pos = c->pos;
    Gpu *g = c->gpu;
    if (c->kv_gpu_n < pos) {
        gpu_dense_kv(g, c->kc, c->vc, c->kv_gpu_n, pos, 1);
        c->kv_gpu_n = pos;
    }
    if (!c->conv_gpu_ok) {
        gpu_dense_conv(g, c->conv, 1);
        c->conv_gpu_ok = 1;
    }
    token_row(c, token, c->x);
    float rope[512], rope_swa[512];
    rope_tables(c, pos, rope, rope_swa);
    gpu_dense_begin(g, c->x, rope, m->swa_window > 0 ? rope_swa : NULL, pos);
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        if (!L->is_moe) continue;
        const float *gx, *gh;
        const int *ret;
        const double tw = time_sec();
        const int ok = gpu_dense_wait(g, l, &gx, &gh, &ret);
        c->stats.t_gpu_wait += time_sec() - tw;
        if (!ok) {   /* garbage in ret: let the GPU run out its queue, the token is dropped below */
            gpu_dense_moe_finish(g, l, NULL);
            continue;
        }
        memcpy(c->x, gx, sizeof(float) * D);
        memcpy(c->h, gh, sizeof(float) * D);
        int sel[8], slot[8];
        float w[8];
        memcpy(sel, ret, sizeof(int) * K);
        memcpy(slot, ret + K, sizeof(int) * K);
        memcpy(w, ret + 2 * K, sizeof(float) * K);
        for (int k = 0; k < K; k++) {
            gpu_note_usage(g, l, sel[k]);
            if (l < 64 && sel[k] < 64) c->stats.expert_hits[l * 64 + sel[k]]++;
            c->last_used[(size_t)l * m->n_expert + sel[k]] = c->tok_counter;
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
    if (!gpu_dense_logits(g, c->logits)) {
        cl_lost(c, 1);
        return;
    }
    if (all_logits) memcpy(all_logits, c->logits, sizeof(float) * (size_t)m->n_vocab);
    c->pos++;
    c->tok_counter++;
    c->kv_gpu_n = pos + 1;
    if (c->kv_cpu_n > pos) c->kv_cpu_n = pos;
    c->conv_cpu_ok = 0;
}

/* ------------------------------------------------------------------ Inkling 2 */

/* Inkling 2 Mini (inkling2_loop), as llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp:
 *
 * Depth attention (mix): the outputs of the earlier blocks and the current state, each RMS-normed,
 * averaged with softmax(state . query / sqrt(dim)) weights; it gives the attention inputs (one
 * query on SWA layers, one each for q, k, v on full-attention ones), the FFN input and the output.
 * The first layer of each block of res_block layers saves its input as a block and starts a new
 * residual stream from its attention output.
 *
 * Attention: no RoPE. q and k L2-clipped (q after a per-channel scale), v RMS-normed, scores
 * q.k / sqrt(hd) plus a relative-position bias r . bias[distance] (r [n_rel] per head, clipped;
 * distances past the extent: no bias), a sink logit per query (sink_proj . input) in the softmax
 * denominator; on the full-attention layers q and r are scaled by tau = 1 + alpha log(max(1,
 * (pos + 1) / floor)). The output times sigmoid(gate . input). Short causal convolutions with a
 * residual, (x + conv(x)) / sqrt(l_cache + 1), on K, V, the attention output and the FFN output;
 * the full-attention layers' K and V also get input-dependent taps per head (then / sqrt(2 l_cache + 1)).
 *
 * MoE: sigmoid router plus a selection bias, top k, weights renormalized then times exp_scale;
 * fused [gate | up] experts, each output times its expert's scale; situ(g, u) = b tanh(g / b)
 * sigmoid(g) lb tanh(u / lb). Logits softcapped.
 *
 * Loop: the layers [loop_start, loop_end) run n_pass times, every pass from the prelude output.
 * Passes after the first get messages: depth attention over the previous pass's blocks after
 * its first core block and its output, four queries per layer; each one enters the q, k, v or
 * FFN input as x - tanh(forget) x + tanh(gain) message, gain and forget low-rank in [x, message]
 * (both RMS-normed). Every run of a layer has its own KV cache and conv state (slot). */

static void ik_rms(float *o, const float *x, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = (float)(1.0 / sqrt(ss / n + 1e-6));
    for (int i = 0; i < n; i++) o[i] = x[i] * s;
}

/* scale a vector down to an L2 norm of at most thr */
static void ik_clip(float *v, int n, float thr) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)v[i] * v[i];
    const float f = thr / (float)(sqrt(ss) + 1e-6);
    if (f < 1.f)
        for (int i = 0; i < n; i++) v[i] *= f;
}

static int ik_phys(const Model *m, int slot) {
    return slot < m->n_layer ? slot : m->loop_start + (slot - m->n_layer) % (m->loop_end - m->loop_start);
}

typedef struct {
    Context *c;
    float *const *blk;
    int nb, nq;
    const float *cur, *query;
    float *out;
    size_t ts, qs;          /* out of token t, query q: out + t * ts + q * qs */
} IkMixArgs;

static void ik_mix_task(void *ctx, int t, int tid) {
    (void)tid;
    const IkMixArgs *a = (const IkMixArgs *)ctx;
    const int D = a->c->m->dim;
    const float *st[17];
    float inv[17];
    int ns = 0;
    for (int b = 0; b < a->nb; b++) st[ns++] = a->blk[b] + (size_t)t * D;
    st[ns++] = a->cur + (size_t)t * D;
    for (int s = 0; s < ns; s++) {
        double ss = 0;
        for (int i = 0; i < D; i++) ss += (double)st[s][i] * st[s][i];
        inv[s] = (float)(1.0 / sqrt(ss / D + 1e-6));
    }
    for (int q = 0; q < a->nq; q++) {
        float p[17], mx = -1e30f, sum = 0;
        for (int s = 0; s < ns; s++) {
            p[s] = inv[s] * KS.dot_f32(st[s], a->query + (size_t)q * D, D) / sqrtf((float)D);
            if (p[s] > mx) mx = p[s];
        }
        for (int s = 0; s < ns; s++) { p[s] = expf(p[s] - mx); sum += p[s]; }
        float *o = a->out + (size_t)t * a->ts + (size_t)q * a->qs;
        for (int i = 0; i < D; i++) o[i] = 0;
        for (int s = 0; s < ns; s++) {
            const float w = p[s] / sum * inv[s];
            for (int i = 0; i < D; i++) o[i] += w * st[s][i];
        }
    }
}

static void ik_mix(Context *c, float *const *blk, int nb, const float *cur, const float *query, int nq, int n,
                   float *out, size_t ts, size_t qs) {
    IkMixArgs a = { c, blk, nb, nq, cur, query, out, ts, qs };
    if (n == 1) ik_mix_task(&a, 0, 0);
    else pool_run(c->pool, ik_mix_task, &a, n);
}

/* x[n][dim] = x - tanh(forget) x + tanh(gain) msg, controller k of the layer */
static void ik_control(Context *c, const IkLoop *lp, int k, float *x, const float *msg, int n) {
    const int D = c->m->dim, R = c->m->gate_rank;
    float *z = c->ik.z;
    for (int t = 0; t < n; t++) {
        ik_rms(z + (size_t)t * 2 * D, x + (size_t)t * D, D);
        ik_rms(z + (size_t)t * 2 * D + D, msg + (size_t)t * D, D);
    }
    job_clear(c);
    job_add_f(c, &lp->down[k], z, 2 * D, n, c->ik.cg, R);
    job_run(c, CAT_ATTN);
    job_clear(c);
    job_add_f(c, &lp->gain[k], c->ik.cg, R, n, c->ik.gain, D);
    job_add_f(c, &lp->forget[k], c->ik.cg, R, n, c->ik.forget, D);
    job_run(c, CAT_ATTN);
    for (size_t i = 0; i < (size_t)n * D; i++)
        x[i] = x[i] - tanhf(c->ik.forget[i]) * x[i] + tanhf(c->ik.gain[i]) * msg[i];
}

/* causal short conv with a residual over x[n][nch] (in place), the slot's state from channel choff;
 * dyn[n][nh][l_cache]: per-head taps added for the lags 0..l_cache-1, or NULL */
static void ik_conv(Context *c, int slot, float *x, int nch, int choff, const float *w, const float *dyn, int nh, int n) {
    const Model *m = c->m;
    const int lc = m->l_cache, hs = nch / nh;
    float *st = c->conv + (size_t)slot * (lc - 1) * m->ik_ch + choff;   /* [lc-1][ik_ch], oldest first */
    const float sc = 1.f / sqrtf((float)((dyn ? 2 : 1) * lc + 1));
    for (int t = 0; t < n; t++) {
        float *xt = x + (size_t)t * nch;
        const float *dt = dyn ? dyn + (size_t)t * nh * lc : NULL;
        for (int ch = 0; ch < nch; ch++) {
            float win[8];   /* oldest first, the current input last */
            for (int k = 0; k < lc - 1; k++) win[k] = st[(size_t)k * m->ik_ch + ch];
            win[lc - 1] = xt[ch];
            float acc = 0;
            for (int k = 0; k < lc; k++) acc += w[(size_t)ch * lc + k] * win[k];
            if (dt) {
                const float *tap = dt + (size_t)(ch / hs) * lc;
                for (int lag = 0; lag < lc; lag++) acc += tap[lag] * win[lc - 1 - lag];
            }
            for (int k = 0; k < lc - 2; k++) st[(size_t)k * m->ik_ch + ch] = st[(size_t)(k + 1) * m->ik_ch + ch];
            st[(size_t)(lc - 2) * m->ik_ch + ch] = win[lc - 1];
            xt[ch] = (win[lc - 1] + acc) * sc;
        }
    }
}

typedef struct {
    Context *c;
    const Layer *L;
    int slot, n, pos0;
} IkAttnArgs;

static void ik_attn_task(void *ctx, int task, int tid) {
    const IkAttnArgs *a = (const IkAttnArgs *)ctx;
    Context *c = a->c;
    const Model *m = c->m;
    const int hd = m->head_dim, nh = m->n_head, kvd = m->n_kv_head * hd, qd = nh * hd, R = m->n_rel;
    const int t = task / nh, h = task % nh, kvh = h / (nh / m->n_kv_head);
    const float *q = c->ik.q + (size_t)t * qd + (size_t)h * hd;
    const float *r = c->ik.r + (size_t)t * nh * R + (size_t)h * R;
    const float sink = c->ik.s[(size_t)t * nh + h];
    const int npos = a->pos0 + t + 1;
    const int j0 = a->L->is_swa && npos > m->swa_window ? npos - m->swa_window : 0;
    const int extent = a->L->is_swa ? m->swa_window : m->rel_extent;
    const size_t lbase = (size_t)a->slot * c->n_ctx * kvd;
    const f16_t *kc = c->kc + lbase + (size_t)kvh * hd, *vc = c->vc + lbase + (size_t)kvh * hd;
    float *s = c->scores + (size_t)tid * c->n_ctx;
    /* the bias by distance, up to the farthest position seen */
    float *tab = c->ik.tab + (size_t)tid * ((m->rel_extent > m->swa_window ? m->rel_extent : m->swa_window) + 1);
    const int nd = npos - j0 < extent ? npos - j0 : extent;
    for (int d = 0; d < nd; d++) {
        const float *b = a->L->ik_rel_bias + (size_t)d * R;
        float acc = 0;
        for (int i = 0; i < R; i++) acc += b[i] * r[i];
        tab[d] = acc;
    }
    const float scale = 1.f / sqrtf((float)hd);
    float mx = sink;
    for (int j = j0; j < npos; j++) {
        const int d = npos - 1 - j;
        const float v = KS.dot_f16_f32(kc + (size_t)j * kvd, q, hd) * scale + (d < extent ? tab[d] : 0.f);
        s[j - j0] = v;
        if (v > mx) mx = v;
    }
    const float sum = KS.exp_sum(s, npos - j0, mx) + expf(sink - mx);
    const float inv = 1.f / sum;
    float acc[256];
    memset(acc, 0, sizeof(float) * hd);
    for (int j = j0; j < npos; j++) KS.axpy_f16(s[j - j0] * inv, vc + (size_t)j * kvd, acc, hd);
    memcpy(c->tmp + (size_t)t * qd + (size_t)h * hd, acc, sizeof(float) * hd);
}

/* the MoE of physical layer ip over the n rows of c->h, into c->out */
static void ik_moe(Context *c, const Layer *L, int ip, int n) {
    const Model *m = c->m;
    const int D = m->dim, nbd = D / QK_K, E = m->n_expert, K = m->n_expert_used, F = m->ff_exp;
    const int FP = (F + QK_K - 1) / QK_K * QK_K, nbf = FP / QK_K;
    double t0 = time_sec();
    job_clear(c);
    job_add_f(c, &L->router, c->h, D, n, c->rlog, E);
    job_run(c, CAT_ROUTER);

    int cnt[MAX_EXPERT] = {0}, off[MAX_EXPERT + 1], fill[MAX_EXPERT];
    for (int t = 0; t < n; t++) {
        const float *lg = c->rlog + (size_t)t * E;
        float p[MAX_EXPERT], sc[MAX_EXPERT];
        for (int e = 0; e < E; e++) {
            p[e] = 1.f / (1.f + expf(-lg[e]));
            sc[e] = p[e] + (L->exp_bias ? L->exp_bias[e] : 0.f);
        }
        int *sel = c->sel + (size_t)t * K;
        float *w = c->selw + (size_t)t * K, wsum = 0;
        top_k(sc, E, K, sel);
        for (int k = 0; k < K; k++) {
            w[k] = p[sel[k]];
            wsum += w[k];
            cnt[sel[k]]++;
            if (ip < 64 && sel[k] < 64) c->stats.expert_hits[ip * 64 + sel[k]]++;
            c->last_used[(size_t)ip * E + sel[k]] = c->tok_counter;
        }
        for (int k = 0; k < K; k++) w[k] = w[k] / (wsum + 1e-20f) * m->exp_scale;
    }
    off[0] = 0;
    for (int e = 0; e < E; e++) off[e + 1] = off[e] + cnt[e];
    memcpy(fill, off, sizeof(int) * E);
    /* assignments grouped by expert: their token (a_tok), their expert (a_slot) */
    quantize_rows(c->h, n, D, c->hq);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < K; k++) {
            const int e = c->sel[(size_t)t * K + k], a = fill[e]++;
            c->a_tok[a] = t;
            c->a_slot[a] = e;
            c->tok_a[(size_t)t * K + k] = a;
            memcpy(c->xg + (size_t)a * nbd, c->hq + (size_t)t * nbd, sizeof(block_q8_K) * nbd);
        }
    c->stats.t_moe_route += time_sec() - t0;

    job_clear(c);
    for (int e = 0; e < E; e++)
        if (cnt[e])
            job_add_q(c, &L->gate_exps, L->gate_exps.data + (size_t)e * L->gate_exps.expert_stride, c->xg + (size_t)off[e] * nbd,
                      nbd, cnt[e], c->ik.gu + (size_t)off[e] * 2 * F, 2 * F);
    job_run(c, CAT_EXP_GU);
    const int na = off[E];
    for (int a = 0; a < na; a++) {
        const float s = L->ik_gu_scale ? L->ik_gu_scale[c->a_slot[a]] : 1.f;
        const float *g = c->ik.gu + (size_t)a * 2 * F, *u = g + F;
        float *o = c->ik.act + (size_t)a * FP;
        for (int i = 0; i < F; i++) {
            const float gi = g[i] * s, ui = u[i] * s;
            o[i] = m->situ_beta * tanhf(gi / m->situ_beta) / (1.f + expf(-gi)) * (m->situ_lb > 0 ? m->situ_lb * tanhf(ui / m->situ_lb) : ui);
        }
        for (int i = F; i < FP; i++) o[i] = 0;
    }
    quantize_rows(c->ik.act, na, FP, c->ik.eq);
    job_clear(c);
    for (int e = 0; e < E; e++)
        if (cnt[e])
            job_add_q(c, &L->down_exps, L->down_exps.data + (size_t)e * L->down_exps.expert_stride, c->ik.eq + (size_t)off[e] * nbf,
                      nbf, cnt[e], c->eo + (size_t)off[e] * D, D);
    job_run(c, CAT_EXP_DOWN);
    for (int t = 0; t < n; t++) {
        float *o = c->out + (size_t)t * D;
        memset(o, 0, sizeof(float) * D);
        for (int k = 0; k < K; k++) {
            const int a = c->tok_a[(size_t)t * K + k];
            const float w = c->selw[(size_t)t * K + k] * (L->ik_down_scale ? L->ik_down_scale[c->a_slot[a]] : 1.f);
            const float *eo = c->eo + (size_t)a * D;
            for (int i = 0; i < D; i++) o[i] += w * eo[i];
        }
    }
}

/* one run of physical layer ip as cache slot `slot` over the n rows of c->x; msg: the pass's
 * messages [4][n][dim] (NULL in the first pass and outside the core). The first layer of a
 * depth block saves its input into buf and appends it to blk. */
static void ik_layer(Context *c, int slot, int ip, int n, float **blk, int *nblk, float *buf, const float *msg,
                     const IkLoop *lp) {
    const Model *m = c->m;
    const Layer *L = &m->layers[ip];
    const int D = m->dim, nbd = D / QK_K, hd = m->head_dim, nh = m->n_head, nkv = m->n_kv_head;
    const int kvd = nkv * hd, qd = nh * hd, R = m->n_rel, lc = m->l_cache, swa = L->is_swa;
    const size_t BD = (size_t)n * D;
    float *x = c->x, *in = c->ik.in;
    if (c->pfq && n >= 8 && slot + 1 < m->n_slot) {
        /* prompt batches touch nearly every expert: stream those of the next run */
        const Layer *Ln = &m->layers[ik_phys(m, slot + 1)];
        prefetcher_clear(c->pfq);
        for (int e = 0; e < m->n_expert; e++) prefetch_expert(c, Ln, e);
    }

    /* the attention inputs in[3][n][dim] */
    const int ninp = swa && !msg ? 1 : 3;
    if (L->ik_attn_query) ik_mix(c, blk, *nblk, x, L->ik_attn_query, swa ? 1 : 3, n, in, D, BD);
    else memcpy(in, x, sizeof(float) * BD);
    for (int j = 1; j < ninp; j++)
        if (swa || !L->ik_attn_query) memcpy(in + j * BD, in, sizeof(float) * BD);
    if (msg)
        for (int j = 0; j < 3; j++) ik_control(c, lp, j, in + j * BD, msg + j * BD, n);
    for (int t = 0; t < n; t++)
        for (int j = 0; j < ninp; j++) {
            float *v = in + j * BD + (size_t)t * D;
            rmsnorm(v, v, swa ? L->attn_norm : j == 0 ? L->ik_q_norm : j == 1 ? L->ik_k_norm : L->ik_v_norm, D, m->eps);
        }
    const float *qin = in, *kin = ninp > 1 ? in + BD : in, *vin = ninp > 1 ? in + 2 * BD : in;
    block_q8_K *qq = c->ik.qin, *kq = ninp > 1 ? qq + (size_t)n * nbd : qq, *vq = ninp > 1 ? qq + (size_t)2 * n * nbd : qq;
    for (int j = 0; j < ninp; j++) quantize_rows(in + j * BD, n, D, qq + (size_t)j * n * nbd);

    job_clear(c);
    job_add_q(c, &L->wq, NULL, qq, nbd, n, c->ik.q, qd);
    job_add_q(c, &L->wk, NULL, kq, nbd, n, c->ik.k, kvd);
    job_add_q(c, &L->wv, NULL, vq, nbd, n, c->ik.v, kvd);
    job_add_q(c, &L->ik_rel, NULL, qq, nbd, n, c->ik.r, nh * R);
    job_add_q(c, &L->ik_gate, NULL, qq, nbd, n, c->ik.g, qd);
    job_add_f(c, &L->ik_sink, qin, D, n, c->ik.s, nh);
    if (!swa) {
        job_add_f(c, &L->ik_kdyn, kin, D, n, c->ik.kd, lc * nkv);
        job_add_f(c, &L->ik_vdyn, vin, D, n, c->ik.vd, lc * nkv);
    }
    job_run(c, CAT_ATTN);

    double t0 = time_sec();
    ik_conv(c, slot, c->ik.k, kvd, 0, L->ik_k_conv, swa ? NULL : c->ik.kd, nkv, n);
    ik_conv(c, slot, c->ik.v, kvd, kvd, L->ik_v_conv, swa ? NULL : c->ik.vd, nkv, n);
    const size_t lbase = (size_t)slot * c->n_ctx * kvd;
    for (int t = 0; t < n; t++) {
        const int pos = c->pos + t;
        const float tau = swa ? 1.f : 1.f + m->log_alpha * logf(fmaxf(1.f, (float)(pos + 1) / (float)m->log_floor));
        float *q = c->ik.q + (size_t)t * qd, *k = c->ik.k + (size_t)t * kvd, *v = c->ik.v + (size_t)t * kvd;
        float *r = c->ik.r + (size_t)t * nh * R;
        for (int i = 0; i < qd; i++) q[i] *= L->ik_q_clip[i];
        for (int i = 0; i < nh * R; i++) r[i] *= L->ik_rel_clip[i];
        for (int h = 0; h < nh; h++) {
            ik_clip(q + (size_t)h * hd, hd, m->qk_clip);
            ik_clip(r + (size_t)h * R, R, m->rel_clip);
            for (int i = 0; i < hd; i++) q[(size_t)h * hd + i] *= tau;
            for (int i = 0; i < R; i++) r[(size_t)h * R + i] *= tau;
        }
        for (int h = 0; h < nkv; h++) {
            ik_clip(k + (size_t)h * hd, hd, m->qk_clip);
            double ss = 0;
            float *vh = v + (size_t)h * hd;
            for (int i = 0; i < hd; i++) ss += (double)vh[i] * vh[i];
            const float s = (float)(1.0 / sqrt(ss / hd + m->eps));
            for (int i = 0; i < hd; i++) vh[i] *= s;
        }
        KS.f32_to_f16_row(k, c->kc + lbase + (size_t)pos * kvd, kvd);
        KS.f32_to_f16_row(v, c->vc + lbase + (size_t)pos * kvd, kvd);
    }
    IkAttnArgs aa = { c, L, slot, n, c->pos };
    pool_run(c->pool, ik_attn_task, &aa, n * nh);
    for (size_t i = 0; i < (size_t)n * qd; i++) c->tmp[i] /= 1.f + expf(-c->ik.g[i]);
    c->stats.t_attn += time_sec() - t0;
    quantize_rows(c->tmp, n, qd, c->tq);
    job_clear(c);
    job_add_q(c, &L->wo, NULL, c->tq, qd / QK_K, n, c->out, D);
    job_run(c, CAT_ATTN);
    ik_conv(c, slot, c->out, D, 2 * kvd, L->ik_o_conv, NULL, 1, n);

    /* a block's first layer: its input joins the blocks, the attention output starts the stream */
    if (ip % m->res_block == 0) {
        memcpy(buf, x, sizeof(float) * BD);
        blk[(*nblk)++] = buf;
        memcpy(x, c->out, sizeof(float) * BD);
    } else {
        for (size_t i = 0; i < BD; i++) x[i] += c->out[i];
    }

    ik_mix(c, blk, *nblk, x, L->ik_ffn_query, 1, n, c->h, D, 0);
    if (msg) ik_control(c, lp, 3, c->h, msg + 3 * BD, n);
    for (int t = 0; t < n; t++) rmsnorm(c->h + (size_t)t * D, c->h + (size_t)t * D, L->ffn_norm, D, m->eps);
    ik_moe(c, L, ip, n);
    ik_conv(c, slot, c->out, D, 2 * kvd + D, L->ik_f_conv, NULL, 1, n);
    for (size_t i = 0; i < BD; i++) x[i] += c->out[i];
}

/* every slot over the n token embeddings of c->x; c->x then holds the output's depth mix */
static void ik_layers(Context *c, int n) {
    const Model *m = c->m;
    const int D = m->dim, s0 = m->loop_start, s1 = m->loop_end, rb = m->res_block;
    const size_t BD = (size_t)n * D;
    float *x = c->x, *blk[16];
    int nblk = 0;
    for (int t = 0; t < n; t++) rmsnorm(x + (size_t)t * D, x + (size_t)t * D, m->embd_norm, D, m->eps);
    /* the buffer of the block starting at layer ip: two for the core blocks, by pass */
#define BUF(ip, pass) c->ik.blk[2 * ((ip) / rb) + ((ip) >= s0 && (ip) < s1 ? (pass) & 1 : 0)]
    for (int l = 0; l < s0; l++) ik_layer(c, l, l, n, blk, &nblk, BUF(l, 0), NULL, NULL);
    memcpy(c->ik.pre, x, sizeof(float) * BD);
    const int npre = nblk;
    float *bank[16];
    int nbank = 0;
    for (int pass = 0; pass < m->n_pass; pass++) {
        if (pass > 0) {
            /* the previous pass's blocks after its first core block, and its output */
            nbank = 0;
            for (int b = npre + 1; b < nblk; b++) bank[nbank++] = blk[b];
            memcpy(c->ik.bank_out, x, sizeof(float) * BD);
            memcpy(x, c->ik.pre, sizeof(float) * BD);
            nblk = npre;
        }
        for (int ip = s0; ip < s1; ip++) {
            const int slot = pass == 0 ? ip : m->n_layer + (pass - 1) * (s1 - s0) + (ip - s0);
            const IkLoop *lp = &m->ik_loop[ip - s0];
            if (pass > 0) ik_mix(c, bank, nbank, c->ik.bank_out, lp->msg_query, 4, n, c->ik.msg, D, BD);
            ik_layer(c, slot, ip, n, blk, &nblk, BUF(ip, pass), pass > 0 ? c->ik.msg : NULL, lp);
        }
    }
    for (int l = s1; l < m->n_layer; l++) ik_layer(c, l, l, n, blk, &nblk, BUF(l, 0), NULL, NULL);
#undef BUF
    ik_mix(c, blk, nblk, x, m->out_query, 1, n, c->ik.mix, D, 0);
    memcpy(x, c->ik.mix, sizeof(float) * BD);
}

/* ------------------------------------------------------------------ layers */

/* every layer on the CPU over the n rows of c->x, at positions c->pos.. */
static void run_layers(Context *c, int n) {
    const Model *m = c->m;
    const int D = m->dim;
    if (m->ik) { ik_layers(c, n); return; }
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
        else if (L->is_delta) deltanet_block(c, L, n);
        else shortconv_block(c, L, n);
        if (L->post_op_norm)
            for (int t = 0; t < n; t++) rmsnorm(c->out + (size_t)t * D, c->out + (size_t)t * D, L->post_op_norm, D, m->eps);
        for (size_t i = 0; i < (size_t)n * D; i++) c->x[i] += c->out[i];
        for (int t = 0; t < n; t++) rmsnorm(c->h + (size_t)t * D, c->x + (size_t)t * D, L->ffn_norm, D, m->eps);
        if (L->is_moe) moe_ffn(c, L, l, n);
        else dense_ffn(c, L, n);
        if (L->sh_up.data) shared_expert(c, L, n);
        if (L->post_ffn_norm)
            for (int t = 0; t < n; t++) rmsnorm(c->out + (size_t)t * D, c->out + (size_t)t * D, L->post_ffn_norm, D, m->eps);
        for (size_t i = 0; i < (size_t)n * D; i++) c->x[i] += c->out[i];
    }
}

/* ------------------------------------------------------------------ forward */

/* Inkling: logits = cap tanh(logits / cap) */
static void softcap(const Model *m, float *l, size_t n) {
    if (!(m->softcap > 0)) return;
    for (size_t i = 0; i < n; i++) l[i] = m->softcap * tanhf(l[i] / m->softcap);
}

static void forward_chunk(Context *c, const int *tokens, int n, int want_logits, float *all_logits) {
    const Model *m = c->m;
    const int D = m->dim;
    c->hidden_ok = 0;
    if (c->gpu_dense && n == 1 && !c->need_hidden) {
        forward_token_gpu(c, tokens[0], all_logits);
        return;
    }
    if (c->dx_dense && n == 1) {
        forward_dx(c, tokens, 1, all_logits ? all_logits : c->logits);
        return;
    }
    if (c->dx_dense && c->dx_chunks && !all_logits) {   /* logits of every token: the path below */
        forward_dx(c, tokens, n, want_logits ? c->logits : NULL);
        return;
    }
    if (c->vk_dense && n == 1) {
        forward_vk(c, tokens, 1, all_logits ? all_logits : c->logits);
        return;
    }
    if (c->vk_dense && c->vk_chunks && !all_logits) {
        forward_vk(c, tokens, n, want_logits ? c->logits : NULL);
        return;
    }
    const int pos0 = c->pos;
    sync_to_cpu(c, pos0);
    for (int t = 0; t < n; t++) token_row(c, tokens[t], c->x + (size_t)t * D);
    run_layers(c, n);
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
        softcap(m, all_logits, (size_t)n * m->n_vocab);
        memcpy(c->logits, all_logits + (size_t)(n - 1) * m->n_vocab, sizeof(float) * m->n_vocab);
    } else if (want_logits) {
        const float *xl = c->x + (size_t)(n - 1) * D;
        rmsnorm(c->h, xl, m->final_norm, D, m->eps);
        c->hidden_ok = 1;
        job_clear(c);
        if (m->output.type == GGML_F32 || m->output.type == GGML_F16) {
            job_add_f(c, &m->output, c->h, D, 1, c->logits, m->n_vocab);
        } else {
            quantize_rows(c->h, 1, D, c->hq);
            job_add_q(c, &m->output, NULL, c->hq, D / QK_K, 1, c->logits, m->n_vocab);
        }
        job_run(c, CAT_LOGITS);
        softcap(m, c->logits, (size_t)m->n_vocab);
    }
}

/* after a failure of the whole-pass GPU its KV cache and state are gone: the context starts
 * again from position 0 on the CPU, and the caller sees the failure (NULL / 0) */
static int lost_state(Context *c) {
    if (gpu_failed(c->gpu)) cl_lost(c, 0);
    if (!c->gpu_lost) return 0;
    c->gpu_lost = 0;
    c->n_lost++;
    ctx_reset(c);
    fprintf(stderr, "the context was lost with the GPU: continuing on the CPU from an empty context\n");
    return 1;
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
        if (c->vk_exp) vk_rebalance(c->vk, nb == 1 ? 2 : 16);
        if (lost_state(c)) return NULL;
    }
    return c->logits;
}

/* a bidirectional chunk (the depth decoder) entirely on the Direct3D 12 GPU: its rows after the
 * final norm in hidden; 0 if the GPU failed (nothing has moved: the CPU can take the chunk) */
static int forward_dx_rows(Context *c, const float *x, int n, float *hidden) {
    const Model *m = c->m;
    const int D = m->dim, hd = m->head_dim, pos0 = c->pos;
    if (c->kv_gpu_n < pos0) {
        dx_dense_kv(c->dx, c->kc, c->vc, c->kv_gpu_n, pos0, 1);
        c->kv_gpu_n = pos0;
    }
    float *rope = c->rope, *rope_swa = c->rope + (size_t)n * hd;
    for (int t = 0; t < n; t++) rope_tables(c, pos0 + t, rope + (size_t)t * hd, rope_swa + (size_t)t * hd);
    if (!dx_dense_forward(c->dx, x, n, rope, m->swa_window > 0 ? rope_swa : NULL, pos0, NULL, c->tmp)) {
        c->dx_dense = 0;
        return 0;
    }
    for (int t = 0; t < n; t++) rmsnorm(hidden + (size_t)t * D, c->tmp + (size_t)t * D, m->final_norm, D, m->eps);
    c->pos += n;
    c->tok_counter += (uint32_t)n;
    c->kv_gpu_n = c->pos;
    if (c->kv_cpu_n > pos0) c->kv_cpu_n = pos0;
    return 1;
}

int ctx_forward_embd(Context *c, const float *x, int n, float *hidden) {
    const Model *m = c->m;
    const int D = m->dim;
    if (c->pos + n > c->n_ctx) return 0;
    if (c->dx_dense && m->bidir && n <= c->n_batch && forward_dx_rows(c, x, n, hidden)) return 1;
    for (int i = 0; i < n; i += c->n_batch) {
        const int nb = n - i < c->n_batch ? n - i : c->n_batch, pos0 = c->pos;
        sync_to_cpu(c, pos0);
        memcpy(c->x, x + (size_t)i * D, sizeof(float) * (size_t)nb * D);
        run_layers(c, nb);
        c->pos += nb;
        c->tok_counter += (uint32_t)nb;
        c->kv_cpu_n = c->pos;
        if (c->kv_gpu_n > pos0) c->kv_gpu_n = pos0;
        c->conv_gpu_ok = 0;
        for (int t = 0; t < nb; t++)
            rmsnorm(hidden + (size_t)(i + t) * D, c->x + (size_t)t * D, m->final_norm, D, m->eps);
    }
    return 1;
}

const float *ctx_hidden(const Context *c) { return c->hidden_ok ? c->h : NULL; }
int ctx_token_on_gpu(const Context *c) { return c->dx_dense || (c->vk_dense && !c->m->n_expert); }
void ctx_need_hidden(Context *c, int on) { c->need_hidden = on; }

void ctx_matmul(Context *c, const Mat *w, const float *x, int n, float *out) {
    job_clear(c);
    if (w->type == GGML_F32 || w->type == GGML_F16) {
        job_add_f(c, w, x, w->cols, n, out, w->rows);
        job_run(c, CAT_DENSE);
        return;
    }
    block_q8_K *q = (block_q8_K *)aligned_malloc(sizeof(block_q8_K) * (size_t)n * (w->cols / QK_K) + 64, 64);
    quantize_rows(x, n, w->cols, q);
    job_add_q(c, w, NULL, q, w->cols / QK_K, n, out, w->rows);
    job_run(c, CAT_DENSE);
    aligned_free(q);
}

int ctx_forward_all(Context *c, const int *tokens, int n, float *out) {
    if (c->pos + n > c->n_ctx) return 0;
    for (int i = 0; i < n; i += c->n_batch) {
        int nb = n - i < c->n_batch ? n - i : c->n_batch;
        forward_chunk(c, tokens + i, nb, 0, out + (size_t)i * c->m->n_vocab);
        if (lost_state(c)) return 0;
    }
    return 1;
}
