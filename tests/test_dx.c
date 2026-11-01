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

int main(int argc, char **argv) {
    Model m;
    char err[512];
    if (argc < 2 || !model_load(&m, argv[1], err, sizeof(err))) { printf("usage: test_dx model.gguf\n"); return 1; }
    quant_init(NULL);
    Dx *d = dx_create(&m, err, sizeof(err));
    if (!d) { printf("no Direct3D 12: %s\n", err); return 0; }
    printf("Direct3D 12: %s\n", err);
    /* the first layer with each shape and type */
    const Mat *pick[8] = {0};
    const char *names[8] = {"ffn_up", "ffn_down/4", "ffn_down/6", "attn_q", "attn_k", "attn_gate", "in_proj", "attn_v/6"};
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
    int ok = 1;
    const int nts[] = {1, 37, 64, 256};
    for (int i = 0; i < 8; i++) {
        if (!pick[i]) continue;
        if (!dx_has(d, pick[i]->data)) { printf("  %-10s not on the GPU\n", names[i]); ok = 0; continue; }
        for (size_t k = 0; k < sizeof(nts) / sizeof(nts[0]); k++) ok &= check(d, names[i], pick[i], nts[k]);
    }
    dx_destroy(d);
    printf("%s\n", ok ? "test_dx: OK" : "test_dx: FAILED");
    return ok ? 0 : 1;
}
