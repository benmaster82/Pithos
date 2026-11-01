/* test_dx.c - checks the Direct3D 12 prompt matmuls (Q4_K and Q6_K) against the CPU kernels
 * and times them:  test_dx model.gguf */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gpu_dx.h"
#include "../src/model.h"
#include "../src/platform.h"
#include "../src/quant.h"
#include "../src/tokenizer.h"

static int check(Dx *d, const char *name, const Mat *w, int nt) {
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

    DxJob j = {w->data, xq, nbk, nt, out, w->rows};
    memset(out, 0, sizeof(float) * (size_t)w->rows * nt);
    if (!dx_matmul(d, &j, 1)) { printf("  %-10s dx_matmul failed\n", name); return 0; }
    double err = 0, mag = 0;
    for (size_t i = 0; i < (size_t)w->rows * nt; i++) {
        err = fmax(err, fabs(out[i] - ref[i]));
        mag = fmax(mag, fabs(ref[i]));
    }
    const int reps = 10;
    t0 = time_sec();
    for (int i = 0; i < reps; i++) dx_matmul(d, &j, 1);
    const double t_gpu = (time_sec() - t0) / reps, flop = 2.0 * w->rows * K * nt;
    const int ok = err <= 1e-4 * mag + 1e-5;
    printf("  %-10s %5d x %4d  type %2u  %3d tok | max err %.2e of %.2e %s | GPU %7.2f ms %6.1f GFLOPS | CPU 1 thread %7.1f ms\n",
           name, w->rows, K, w->type, nt, err, mag, ok ? "ok  " : "FAIL", t_gpu * 1e3, flop / t_gpu * 1e-9, t_cpu * 1e3);
    free(x); free(xq); free(ref); free(out);
    return ok;
}

/* Whole-token decoding against the CPU: the same text read token by token through a GPU
 * context and a CPU-only one. Random token streams are chaotic (rounding differences grow), so
 * the check uses real text and compares what matters: the probability of the actual next token
 * and the top choice. Prompt chunks between single tokens move the KV cache and the
 * convolution state back and forth; over 300 positions pass the 256-token sliding window. */
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

static int check_dense(Model *m, Dx *d) {
    char err[256];
    Tokenizer *tk = tok_load(&m->gguf, err, sizeof(err));
    ThreadPool *pool = pool_create(4);
    Context *g = ctx_create(m, pool, 1024, 64), *c = ctx_create(m, pool, 1024, 64);
    if (!tk || (!getenv("DX_NONE") && !ctx_set_dx(g, d, err, sizeof(err)) && !getenv("PITHOS_DX_DENSE"))) {   /* =0: matmuls only */
        printf("  whole-token decoding not available: %s\n", tk ? err : "no tokenizer");
        return 1;
    }
    static int toks[1024];
    const int n = tok_encode(tk, TEXT, (int)strlen(TEXT), 0, toks, 1024);
    double nll_g = 0, nll_c = 0;
    int agree = 0, steps = 0;
    for (int pos = 0; pos + 1 < n;) {
        const int k = pos % 80 == 0 && pos + 24 < n ? 24 : 1;    /* a prompt chunk every 80 positions */
        const float *lg = ctx_forward(g, toks + pos, k), *lc = ctx_forward(c, toks + pos, k);
        pos += k;
        if (k > 1 || pos >= n) continue;
        /* log-probability of the actual next token under each */
        double mg = -1e30, mc = -1e30, sg = 0, sc = 0;
        int ag = 0, ac = 0;
        for (int v = 0; v < m->n_vocab; v++) {
            if (lg[v] > mg) { mg = lg[v]; ag = v; }
            if (lc[v] > mc) { mc = lc[v]; ac = v; }
        }
        for (int v = 0; v < m->n_vocab; v++) {
            sg += exp(lg[v] - mg);
            sc += exp(lc[v] - mc);
        }
        nll_g += mg + log(sg) - lg[toks[pos]];
        nll_c += mc + log(sc) - lc[toks[pos]];
        agree += ag == ac;
        if (getenv("DX_TRACE") && steps < 12) {
            double e = 0;
            for (int v = 0; v < m->n_vocab; v++) e = fmax(e, fabs(lg[v] - lc[v]));
            printf("    pos %3d  max |logit diff| %.5f (top %.3f)  top1 %s\n", pos - 1, e, mc, ag == ac ? "=" : "x");
        }
        steps++;
    }
    nll_g /= steps;
    nll_c /= steps;
    /* Q8_K activations make any change of summation order flip some roundings, which then
     * propagate: even the CPU kernels (scalar, AVX, AVX2) differ at this level */
    const int ok = fabs(nll_g - nll_c) < 0.02 * nll_c && agree >= steps * 85 / 100;
    printf("  whole tokens  %d tokens of text, %d read one at a time | top-1 agrees %d/%d | "
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
    if (argc < 2 || !model_load(&m, argv[1], err, sizeof(err))) { printf("usage: test_dx model.gguf\n"); return 1; }
    quant_init(NULL);
    Dx *d = dx_create(&m, err, sizeof(err));
    if (!d) { printf("no Direct3D 12: %s\n", err); return 0; }
    printf("Direct3D 12: %s\n", err);
    /* the first layer with each shape and type */
    const Mat *pick[9] = {0};
    const char *names[9] = {"ffn_up", "ffn_down/4", "ffn_down/6", "attn_q", "attn_k", "attn_gate", "in_proj", "attn_v/6", "output"};
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
    }
    pick[8] = &m.output;
    int ok = 1;
    const int nts[] = {1, 3, 8, 37, 64, 256};
    for (int i = 0; i < 9; i++) {
        if (!pick[i]) continue;
        if (!dx_has(d, pick[i]->data)) { printf("  %-10s not on the GPU\n", names[i]); ok = 0; continue; }
        for (size_t k = 0; k < sizeof(nts) / sizeof(nts[0]); k++) ok &= check(d, names[i], pick[i], nts[k]);
    }
    ok &= check_dense(&m, d);
    dx_destroy(d);
    printf("%s\n", ok ? "test_dx: OK" : "test_dx: FAILED");
    return ok ? 0 : 1;
}
