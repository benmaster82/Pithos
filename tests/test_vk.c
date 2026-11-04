/* test_vk.c - checks the Vulkan matmuls (Q4_K and Q6_K) and the whole forward pass against the
 * CPU and times the matmuls:
 *   test_vk model.gguf [device]     (device: an index from the list it prints first) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gpu_vk.h"
#include "../src/model.h"
#include "../src/platform.h"
#include "../src/quant.h"
#include "../src/tokenizer.h"

static int check(Vulkan *v, const char *name, const Mat *w, int nt) {
    const int K = w->cols, nbk = K / QK_K;
    float *x = malloc(sizeof(float) * (size_t)K * nt);
    block_q8_K *xq = malloc(sizeof(block_q8_K) * (size_t)nbk * nt);
    float *ref = malloc(sizeof(float) * (size_t)w->rows * nt), *out = malloc(sizeof(float) * (size_t)w->rows * nt);
    unsigned s = 12345u + (unsigned)nt;
    for (size_t i = 0; i < (size_t)K * nt; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((s >> 8) / 16777216.f - 0.5f) * 2.f;
    }
    for (int t = 0; t < nt; t++) KS.quantize_q8_K(x + (size_t)t * K, xq + (size_t)t * nbk, K);
    double t0 = time_sec();
    for (int r = 0; r < w->rows; r++) KS.dot[w->type](K, w->data + (size_t)r * w->row_bytes, xq, nbk, nt, ref + r, w->rows);
    const double t_cpu = time_sec() - t0;

    VulkanJob j = {w->data, xq, nbk, nt, out, w->rows};
    memset(out, 0, sizeof(float) * (size_t)w->rows * nt);
    if (!vk_matmul(v, &j, 1)) { printf("  %-10s vk_matmul failed\n", name); return 0; }
    double err = 0, mag = 0;
    for (size_t i = 0; i < (size_t)w->rows * nt; i++) {
        err = fmax(err, fabs(out[i] - ref[i]));
        mag = fmax(mag, fabs(ref[i]));
    }
    const int reps = 10;
    t0 = time_sec();
    for (int i = 0; i < reps; i++) vk_matmul(v, &j, 1);
    const double t_gpu = (time_sec() - t0) / reps, flop = 2.0 * w->rows * K * nt;
    const int ok = err <= 1e-4 * mag + 1e-5;
    printf("  %-10s %5d x %4d  type %2u  %3d tok | max err %.2e of %.2e %s | GPU %7.2f ms %6.1f GFLOPS | CPU 1 thread %7.1f ms\n",
           name, w->rows, K, w->type, nt, err, mag, ok ? "ok  " : "FAIL", t_gpu * 1e3, flop / t_gpu * 1e-9, t_cpu * 1e3);
    free(x); free(xq); free(ref); free(out);
    return ok;
}

/* several matrices in one call, two of them sharing their activations (as the q/k/v of a layer) */
static int check_batch(Vulkan *v, const Mat *a, const Mat *b, const Mat *c, int nt) {
    const Mat *ws[3] = {a, b, c};
    const int K = a->cols;
    if (b->cols != K) return 1;
    block_q8_K *xq[2];
    float *ref[3], *out[3];
    unsigned s = 777u;
    for (int h = 0; h < 2; h++) {   /* h = 0: the input of a and b; 1: of c */
        const int Kh = h ? c->cols : K;
        xq[h] = malloc(sizeof(block_q8_K) * (size_t)(Kh / QK_K) * nt);
        float *xx = malloc(sizeof(float) * (size_t)Kh * nt);
        for (size_t i = 0; i < (size_t)Kh * nt; i++) {
            s = s * 1664525u + 1013904223u;
            xx[i] = ((s >> 8) / 16777216.f - 0.5f) * 2.f;
        }
        for (int t = 0; t < nt; t++) KS.quantize_q8_K(xx + (size_t)t * Kh, xq[h] + (size_t)t * (Kh / QK_K), Kh);
        free(xx);
    }
    VulkanJob jobs[3];
    for (int i = 0; i < 3; i++) {
        const Mat *w = ws[i];
        const block_q8_K *in = i < 2 ? xq[0] : xq[1];
        const int nb = w->cols / QK_K;
        ref[i] = malloc(sizeof(float) * (size_t)w->rows * nt);
        out[i] = calloc((size_t)w->rows * nt, sizeof(float));
        for (int r = 0; r < w->rows; r++) KS.dot[w->type](w->cols, w->data + (size_t)r * w->row_bytes, in, nb, nt, ref[i] + r, w->rows);
        VulkanJob j = {w->data, in, nb, nt, out[i], w->rows};
        jobs[i] = j;
    }
    int ok = vk_matmul(v, jobs, 3);
    double err = 0, mag = 0;
    for (int i = 0; i < 3 && ok; i++)
        for (size_t k = 0; k < (size_t)ws[i]->rows * nt; k++) {
            err = fmax(err, fabs(out[i][k] - ref[i][k]));
            mag = fmax(mag, fabs(ref[i][k]));
        }
    ok = ok && err <= 1e-4 * mag + 1e-5;
    printf("  3 jobs     one call, two share their input  %3d tok | max err %.2e of %.2e %s\n", nt, err, mag, ok ? "ok" : "FAIL");
    for (int i = 0; i < 3; i++) { free(ref[i]); free(out[i]); }
    free(xq[0]); free(xq[1]);
    return ok;
}

/* The whole forward pass against the CPU, as in test_dx: the same text read through a GPU
 * context and a CPU-only one, mostly token by token with prompt chunks of several sizes in
 * between (one longer than the batch, some below VK_MV_TOKENS), past the 256-token sliding
 * window. It compares the probability of the actual next token and the top choice. */
static const char *TEXT =
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting the steps as his father had "
    "taught him. There were one hundred and twelve of them, worn smooth in the middle by a century of boots. At "
    "the top he wiped the great lens with a soft cloth, trimmed the wick, and lit the lamp. Out on the water the "
    "fishing boats were already turning for home, their engines a low hum beneath the cries of the gulls. He "
    "liked this hour best: the sea grew dark before the sky did, and for a few minutes the horizon was a thin "
    "silver line that seemed to hold the whole world together. Il guardiano del faro scendeva poi in cucina, "
    "dove la moglie aveva preparato una zuppa di pesce con il pane del giorno prima. Parlavano del tempo, delle "
    "reti da riparare e del figlio che studiava ingegneria in città e scriveva lettere lunghe e piene di disegni. "
    "Nelle notti di tempesta il vento faceva vibrare i vetri e lui restava sveglio fino all'alba, controllando "
    "che la luce non si spegnesse mai. In spring the keeper planted tomatoes in the sheltered corner behind the "
    "tower, where the stone walls kept off the salt wind. In summer tourists came on the ferry and asked to see "
    "the lamp; he showed them gladly, explaining how the lens bent the light into a beam that could be seen "
    "twenty miles away. In autumn the storms returned, and in winter the island was quiet again, with only the "
    "two of them, the radio, and the long white beam sweeping across the black water every ten seconds.";

static int check_dense(Model *m, Vulkan *v) {
    char err[256];
    Tokenizer *tk = tok_load(&m->gguf, err, sizeof(err));
    ThreadPool *pool = pool_create(4);
    Context *g = ctx_create(m, pool, 1024, 64), *c = ctx_create(m, pool, 1024, 64);
    if (!tk || !ctx_set_vk(g, v, err, sizeof(err))) {
        printf("  whole forward pass not available: %s\n", tk ? err : "no tokenizer");
        return 1;
    }
    static int toks[1024];
    const int n = tok_encode(tk, TEXT, (int)strlen(TEXT), 0, toks, 1024);
    double nll_g = 0, nll_c = 0;
    int agree = 0, steps = 0;
    for (int pos = 0; pos + 1 < n;) {
        int k = pos == 0 ? 100 : pos % 80 == 0 ? 24 : pos % 80 == 40 ? 5 : 1;   /* prompt chunks */
        if (pos + k >= n) k = 1;
        const float *lg = ctx_forward(g, toks + pos, k), *lc = ctx_forward(c, toks + pos, k);
        pos += k;
        if (pos >= n) continue;
        double mg = -1e30, mc = -1e30, sg = 0, sc = 0;
        int ag = 0, ac = 0;
        for (int i = 0; i < m->n_vocab; i++) {
            if (lg[i] > mg) { mg = lg[i]; ag = i; }
            if (lc[i] > mc) { mc = lc[i]; ac = i; }
        }
        for (int i = 0; i < m->n_vocab; i++) {
            sg += exp(lg[i] - mg);
            sc += exp(lc[i] - mc);
        }
        nll_g += mg + log(sg) - lg[toks[pos]];
        nll_c += mc + log(sc) - lc[toks[pos]];
        agree += ag == ac;
        steps++;
    }
    nll_g /= steps;
    nll_c /= steps;
    /* Q8_K activations make any change of summation order flip some roundings (see test_dx) */
    const int ok = fabs(nll_g - nll_c) < 0.02 * nll_c && agree >= steps * 85 / 100;
    printf("  whole pass  %d tokens of text, %d predictions | top-1 agrees %d/%d | "
           "mean NLL GPU %.4f CPU %.4f %s\n", n, steps, agree, steps, nll_g, nll_c, ok ? "ok" : "FAIL");
    ctx_free(g);
    ctx_free(c);
    pool_destroy(pool);
    tok_free(tk);
    return ok;
}

int main(int argc, char **argv) {
    Model m;
    char err[512];
    if (argc < 2 || !model_load(&m, argv[1], err, sizeof(err))) { printf("usage: test_vk model.gguf [device]\n"); return 1; }
    quant_init(NULL);
    printf("Vulkan devices:\n");
    vk_list_devices();
    Vulkan *v = vk_create(&m, argc > 2 ? atoi(argv[2]) : -1, err, sizeof(err));
    if (!v) { printf("no Vulkan: %s\n", err); return 0; }
    printf("Vulkan: %s\n", err);
    /* the first layer with each shape and type */
    const Mat *pick[9] = {0};
    const char *names[9] = {"ffn_up", "ffn_down/4", "ffn_down/6", "attn_q", "attn_k", "attn_gate", "in_proj", "attn_v/6", "output"};
    const Mat *qkv[3] = {0};
    for (int l = 0; l < m.n_layer; l++) {
        const Layer *L = &m.layers[l];
        if (!pick[0] && L->w_up.data) pick[0] = &L->w_up;
        if (!pick[1] && L->w_down.data && L->w_down.type == GGML_Q4_K) pick[1] = &L->w_down;
        if (!pick[2] && L->w_down.data && L->w_down.type == GGML_Q6_K) pick[2] = &L->w_down;
        if (!pick[3] && L->wq.data) pick[3] = &L->wq;
        if (!pick[4] && L->wk.data) pick[4] = &L->wk;
        if (!pick[5] && L->attn_gate.data) pick[5] = &L->attn_gate;
        if (!pick[6] && L->in_proj.data) pick[6] = &L->in_proj;
        if (!pick[7] && L->wv.data && L->wv.type == GGML_Q6_K) pick[7] = &L->wv;
        if (!qkv[0] && L->wq.data && L->wk.data && L->w_down.data) { qkv[0] = &L->wq; qkv[1] = &L->wk; qkv[2] = &L->w_down; }
    }
    pick[8] = &m.output;
    int ok = 1;
    const int nts[] = {1, 3, 8, 37, 64, 256};
    for (int i = 0; i < 9; i++) {
        if (!pick[i]) continue;
        if (!vk_has(v, pick[i]->data)) {   /* only allowed for matrices over the device's buffer limit */
            printf("  %-10s not on the GPU (%llu MB)\n", names[i],
                   (unsigned long long)((size_t)pick[i]->rows * pick[i]->row_bytes >> 20));
            continue;
        }
        for (size_t k = 0; k < sizeof(nts) / sizeof(nts[0]); k++) ok &= check(v, names[i], pick[i], nts[k]);
    }
    if (qkv[0]) {
        ok &= check_batch(v, qkv[0], qkv[1], qkv[2], 1);
        ok &= check_batch(v, qkv[0], qkv[1], qkv[2], 64);
    }
    ok &= check_dense(&m, v);
    vk_destroy(v);
    printf("%s\n", ok ? "test_vk: OK" : "test_vk: FAILED");
    return ok ? 0 : 1;
}
