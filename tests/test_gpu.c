/* test_gpu.c - checks GPU expert outputs (Q4_K and Q6_K layers) against an f32 CPU reference. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gpu.h"
#include "../src/model.h"
#include "../src/platform.h"
#include "../src/quant.h"

static void ref_expert(const Layer *L, int e, const float *x, float *y, int D, int F) {
    float *w = malloc(sizeof(float) * (D > F ? D : F)), *g = malloc(sizeof(float) * F), *u = malloc(sizeof(float) * F);
    for (int r = 0; r < F; r++) {
        dequant_row(L->gate_exps.type, L->gate_exps.data + e * L->gate_exps.expert_stride + r * L->gate_exps.row_bytes, w, D);
        double s = 0; for (int i = 0; i < D; i++) s += (double)w[i] * x[i]; g[r] = (float)s;
        dequant_row(L->up_exps.type, L->up_exps.data + e * L->up_exps.expert_stride + r * L->up_exps.row_bytes, w, D);
        s = 0; for (int i = 0; i < D; i++) s += (double)w[i] * x[i]; u[r] = (float)s;
    }
    for (int r = 0; r < F; r++) g[r] = g[r] / (1.f + expf(-g[r])) * u[r];
    for (int r = 0; r < D; r++) {
        dequant_row(L->down_exps.type, L->down_exps.data + e * L->down_exps.expert_stride + r * L->down_exps.row_bytes, w, F);
        double s = 0; for (int i = 0; i < F; i++) s += (double)w[i] * g[i]; y[r] = (float)s;
    }
    free(w); free(g); free(u);
}

int main(int argc, char **argv) {
    Model m;
    char err[512];
    if (argc < 2 || !model_load(&m, argv[1], err, sizeof(err))) { printf("usage: test_gpu model.gguf\n"); return 1; }
    quant_init(NULL);
    Gpu *g = gpu_create(&m, -1, 200, 4, 0, NULL, err, sizeof(err));
    if (!g) { printf("no GPU: %s\n", err); return 0; }
    printf("GPU: %s\n", err);
    /* int8 activations (dp4a path) carry ~1% quantization error, like the CPU Q8_K path */
    const double tol = strstr(err, "int8") ? 3e-2 : 1e-4;
    const int D = m.dim, F = m.ff_exp;
    /* one layer with Q6_K down and one with Q4_K down */
    int layers[2] = { -1, -1 };
    for (int l = 0; l < m.n_layer; l++) {
        if (!m.layers[l].is_moe) continue;
        int q6 = m.layers[l].down_exps.type == 14;
        if (layers[q6] < 0) layers[q6] = l;
    }
    int ok = 1;
    for (int li = 0; li < 2; li++) {
        int l = layers[li];
        if (l < 0) continue;
        int experts[2] = { 3, 41 };
        for (int k = 0; k < 2; k++) for (int r = 0; r < 1000; r++) gpu_note_usage(g, l, experts[k]);
        gpu_rebalance(g, 4);
        double t0 = time_sec();
        while ((!gpu_resident(g, l, experts[0]) || !gpu_resident(g, l, experts[1])) && time_sec() - t0 < 30) ;
        float *x = malloc(sizeof(float) * D * 2), *y = malloc(sizeof(float) * D * 2), *ref = malloc(sizeof(float) * D);
        unsigned s = 99;
        for (int i = 0; i < D * 2; i++) { s = s * 1664525u + 1013904223u; x[i] = ((s >> 8) / 16777216.f - 0.5f) * 2.f; }
        int at[2] = { 0, 1 };
        gpu_moe_begin(g, l, x, 2, experts, at, 2);
        gpu_moe_end(g, y);
        for (int k = 0; k < 2; k++) {
            ref_expert(&m.layers[l], experts[k], x + k * D, ref, D, F);
            double num = 0, den = 0;
            for (int i = 0; i < D; i++) { num += (y[k * D + i] - ref[i]) * (y[k * D + i] - ref[i]); den += ref[i] * ref[i]; }
            double rel = sqrt(num / (den + 1e-30));
            printf("layer %2d (down %s) expert %2d: rel L2 err %.2e %s\n", l, m.layers[l].down_exps.type == 14 ? "Q6_K" : "Q4_K",
                   experts[k], rel, rel < tol ? "OK" : "FAIL");
            ok &= rel < tol;
        }
        /* latency / throughput of one decode-style launch */
        for (int na = 1; na <= 2; na++) {
            const int reps = 200;
            double tb = time_sec();
            for (int r = 0; r < reps; r++) {
                gpu_moe_begin(g, l, x, 2, experts, at, na);
                gpu_moe_end(g, y);
            }
            double dt = (time_sec() - tb) / reps;
            const Layer *L = &m.layers[l];
            double bytes = na * (double)(L->gate_exps.expert_stride + L->up_exps.expert_stride + L->down_exps.expert_stride);
            printf("   %d expert(s): %.1f us per launch (%.1f GB/s effective)\n", na, dt * 1e6, bytes / dt / 1e9);
            /* pipelined: many launches, one wait -> kernel throughput without per-call latency */
            tb = time_sec();
            for (int r = 0; r < 20; r++) gpu_moe_begin(g, l, x, 2, experts, at, na);
            gpu_moe_end(g, y);
            dt = (time_sec() - tb) / 20;
            printf("   %d expert(s) pipelined: %.1f us per launch (%.1f GB/s)\n", na, dt * 1e6, bytes / dt / 1e9);
        }
        free(x); free(y); free(ref);
    }
    if (getenv("PITHOS_GPU_SUSTAIN")) {
        /* sustained load for clock inspection (nvidia-smi) */
        int l = layers[0] >= 0 ? layers[0] : layers[1];
        int experts[2] = { 3, 41 }, at[2] = { 0, 1 };
        float *x = calloc((size_t)D * 2, sizeof(float)), *y = calloc((size_t)D * 2, sizeof(float));
        double tb = time_sec();
        int it = 0;
        while (time_sec() - tb < 8) { gpu_moe_begin(g, l, x, 2, experts, at, 2); gpu_moe_end(g, y); it++; }
        printf("sustained: %.1f us per 2-expert launch\n", (time_sec() - tb) / it * 1e6);
        free(x); free(y);
    }
    gpu_destroy(g);
    printf(ok ? "GPU OK\n" : "GPU FAILED\n");
    return ok ? 0 : 1;
}
