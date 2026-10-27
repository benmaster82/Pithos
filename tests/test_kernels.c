/* test_kernels.c - checks every dot kernel against f32 math on dequantized rows of a real GGUF. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gguf.h"
#include "../src/quant.h"

static float frand(unsigned *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.f) * 2.f - 1.f; }

static int check(const char *kname, const Kernels *k, const GGUFTensor *t) {
    const int K = (int)t->ne[0];
    const size_t rb = row_bytes(t->type, K);
    const int nt = 5;
    float *x = malloc(sizeof(float) * K * nt), *w = malloc(sizeof(float) * K), *xd = malloc(sizeof(float) * K * nt);
    block_q8_K *xq = malloc(sizeof(block_q8_K) * (K / QK_K) * nt);
    unsigned s = 1234;
    for (int i = 0; i < K * nt; i++) x[i] = frand(&s) * (i % 7 == 0 ? 3.f : 1.f);
    for (int j = 0; j < nt; j++) k->quantize_q8_K(x + j * K, xq + j * (K / QK_K), K);
    /* dequantize the activations so we measure the weight kernel only */
    for (int j = 0; j < nt; j++)
        for (int b = 0; b < K / QK_K; b++)
            for (int i = 0; i < QK_K; i++) xd[j * K + b * QK_K + i] = xq[j * (K / QK_K) + b].d * xq[j * (K / QK_K) + b].qs[i];
    double maxrel = 0;
    for (int r = 0; r < 8; r++) {
        int row = (int)((t->ne[1] * t->ne[2] - 1) * r / 7);
        const uint8_t *wr = t->data + (size_t)row * rb;
        dequant_row(t->type, wr, w, K);
        float out[5];
        for (int n = 1; n <= nt; n += 4) {
            k->dot[t->type](K, wr, xq, K / QK_K, n, out, 1);
            for (int j = 0; j < n; j++) {
                double ref = 0, mag = 0;
                for (int i = 0; i < K; i++) { ref += (double)w[i] * xd[j * K + i]; mag += fabs((double)w[i] * xd[j * K + i]); }
                double rel = fabs(out[j] - ref) / (mag + 1e-9);
                if (rel > maxrel) maxrel = rel;
            }
        }
    }
    /* bsums sanity */
    int bad = 0;
    for (int b = 0; b < K / QK_K; b++)
        for (int g = 0; g < 16; g++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += xq[b].qs[g * 16 + i];
            if (sum != xq[b].bsums[g]) bad++;
        }
    printf("%-7s %-28s %-5s K=%-6d max rel err %.2e %s%s\n", kname, t->name, ggml_type_name(t->type), K, maxrel,
           maxrel < 1e-5 ? "OK" : "FAIL", bad ? " BSUMS-BAD" : "");
    free(x); free(w); free(xd); free(xq);
    return maxrel < 1e-5 && !bad;
}

int main(int argc, char **argv) {
    GGUF g;
    char err[256];
    if (argc < 2 || !gguf_open(&g, argv[1], err, sizeof(err))) { printf("usage: test_kernels model.gguf (%s)\n", argc < 2 ? "" : err); return 1; }
    quant_init(NULL);
    Kernels ks[2] = { K_scalar, KS };
    const char *names[] = { "token_embd.weight", "blk.0.ffn_down.weight", "blk.0.ffn_gate.weight",
                            "blk.2.ffn_gate_exps.weight", "blk.2.ffn_down_exps.weight", "blk.2.attn_v.weight" };
    int ok = 1;
    for (int ki = 0; ki < 2; ki++)
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            const GGUFTensor *t = gguf_tensor(&g, names[i]);
            if (t && ks[ki].dot[t->type]) ok &= check(ks[ki].name, &ks[ki], t);
        }
    /* fp16 round trip */
    for (float f = -70000.f; f < 70000.f; f += 13.37f) {
        f16_t h = f32_to_f16(f);
        float back = f16_to_f32(h);
        if (fabsf(f) < 65000 && fabsf(back - f) > fabsf(f) * 1e-3f + 1e-3f) { printf("fp16 fail %f -> %f\n", f, back); ok = 0; break; }
    }
    printf(ok ? "ALL OK\n" : "SOME FAILED\n");
    /* single-core throughput over a whole tensor (second pass = resident in RAM) */
    const char *bn[] = { "blk.2.ffn_gate_exps.weight", "blk.2.ffn_down_exps.weight" };
    for (size_t i = 0; i < 2; i++) {
        const GGUFTensor *t = gguf_tensor(&g, bn[i]);
        const int K = (int)t->ne[0];
        const int rows = (int)(t->ne[1] * t->ne[2]);
        const size_t rb = row_bytes(t->type, K);
        block_q8_K *xq = malloc(sizeof(block_q8_K) * (K / QK_K) * 16);
        float *x = malloc(sizeof(float) * K * 16), *out = malloc(sizeof(float) * 16);
        unsigned s = 7;
        for (int j = 0; j < K * 16; j++) x[j] = frand(&s);
        for (int j = 0; j < 16; j++) KS.quantize_q8_K(x + j * K, xq + j * (K / QK_K), K);
        for (int nt = 1; nt <= 16; nt *= 4)
            for (int pass = 0; pass < 2; pass++) {
                double t0 = time_sec();
                float acc = 0;
                for (int r = 0; r < rows; r++) {
                    KS.dot[t->type](K, t->data + (size_t)r * rb, xq, K / QK_K, nt, out, 1);
                    acc += out[0];
                }
                double dt = time_sec() - t0;
                if (pass) printf("%-28s %s nt=%-2d 1 core: %.2f GB/s weights, %.1f Gop/s (%g)\n", bn[i],
                                 ggml_type_name(t->type), nt, rows * rb / dt / 1e9, 2.0 * rows * K * nt / dt / 1e9, acc);
            }
        /* compute-bound: 64 rows that stay in L2, repeated */
        {
            const int hot = 64, reps = 2000;
            double t0 = time_sec();
            float acc = 0;
            for (int rep = 0; rep < reps; rep++)
                for (int r = 0; r < hot; r++) {
                    KS.dot[t->type](K, t->data + (size_t)r * rb, xq, K / QK_K, 1, out, 1);
                    acc += out[0];
                }
            double dt = time_sec() - t0;
            printf("%-28s %s nt=1  L2-resident: %.2f GB/s (%g)\n", bn[i], ggml_type_name(t->type),
                   (double)hot * reps * rb / dt / 1e9, acc);
        }
        free(xq); free(x); free(out);
    }
    gguf_close(&g);
    return ok ? 0 : 1;
}
