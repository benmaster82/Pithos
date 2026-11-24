/* test_kernels.c - checks every dot kernel against f32 math on dequantized rows of a real GGUF. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gguf.h"
#include "../src/platform.h"
#include "../src/quant.h"

static float frand(unsigned *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.f) * 2.f - 1.f; }

static int check(const char *kname, const Kernels *k, const GGUFTensor *t) {
    const int K = (int)t->ne[0];
    /* activation rows of whole Q8_K blocks, zero past K (NVFP4 rows need not fill the last block) */
    const int nb = (K + QK_K - 1) / QK_K, KP = nb * QK_K;
    const size_t rb = row_bytes(t->type, K);
    const int nt = 5;
    float *x = calloc((size_t)KP * nt, sizeof(float)), *w = malloc(sizeof(float) * K), *xd = malloc(sizeof(float) * KP * nt);
    block_q8_K *xq = malloc(sizeof(block_q8_K) * nb * nt);
    unsigned s = 1234;
    for (int j = 0; j < nt; j++)
        for (int i = 0; i < K; i++) x[j * KP + i] = frand(&s) * (i % 7 == 0 ? 3.f : 1.f);
    for (int j = 0; j < nt; j++) k->quantize_q8_K(x + j * KP, xq + j * nb, KP);
    /* dequantize the activations so we measure the weight kernel only */
    for (int j = 0; j < nt; j++)
        for (int b = 0; b < nb; b++)
            for (int i = 0; i < QK_K; i++) xd[j * KP + b * QK_K + i] = xq[j * nb + b].d * xq[j * nb + b].qs[i];
    double maxrel = 0;
    for (int r = 0; r < 8; r++) {
        int row = (int)((t->ne[1] * t->ne[2] - 1) * r / 7);
        const uint8_t *wr = t->data + (size_t)row * rb;
        dequant_row(t->type, wr, w, K);
        float out[5];
        for (int n = 1; n <= nt; n += 4) {
            k->dot[t->type](K, wr, xq, nb, n, out, 1);
            for (int j = 0; j < n; j++) {
                double ref = 0, mag = 0;
                for (int i = 0; i < K; i++) { ref += (double)w[i] * xd[j * KP + i]; mag += fabs((double)w[i] * xd[j * KP + i]); }
                double rel = fabs(out[j] - ref) / (mag + 1e-9);
                if (rel > maxrel) maxrel = rel;
            }
        }
    }
    /* bsums sanity */
    int bad = 0;
    for (int b = 0; b < nb; b++)
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

/* Random rows for every weight format (covers types the test model does not use):
 * random bytes with sane f16 block scales, wrapped in a fake tensor. */
static int synth_check(const char *kname, const Kernels *k, uint32_t type, int K) {
    const int rows = 8;
    const size_t rb = row_bytes(type, K);
    uint8_t *data = malloc(rb * rows);
    unsigned s = 99 + type;
    for (size_t i = 0; i < rb * rows; i++) { s = s * 1664525u + 1013904223u; data[i] = (uint8_t)(s >> 24); }
    const size_t bsz = (size_t)ggml_type_size(type);
    for (size_t off = 0; off + bsz <= rb * rows; off += bsz) {
        if (type == GGML_NVFP4) {
            /* E4M3 scales around 2^-6, with some zero sub-blocks */
            for (int j = 0; j < 4; j++) data[off + j] = (uint8_t)(0x08 + (data[off + j] % 16) * (data[off + j] % 9 != 0));
            continue;
        }
        f16_t d = f32_to_f16(0.01f + 0.001f * (float)(off % 7)), dmin = f32_to_f16(0.005f);
        if (type == GGML_Q6_K) memcpy(data + off + bsz - 2, &d, 2);
        else memcpy(data + off, &d, 2);
        if (type == GGML_Q4_K || type == GGML_Q5_K) memcpy(data + off + 2, &dmin, 2);
    }
    GGUFTensor t;
    memset(&t, 0, sizeof(t));
    snprintf(t.name, sizeof(t.name), "synthetic.%s", ggml_type_name(type));
    t.type = type;
    t.ne[0] = K;
    t.ne[1] = rows;
    t.ne[2] = 1;
    t.data = data;
    int ok = check(kname, k, &t);
    free(data);
    return ok;
}

int main(int argc, char **argv) {
    GGUF g;
    char err[256];
    if (argc < 2 || !gguf_open(&g, argv[1], err, sizeof(err))) { printf("usage: test_kernels model.gguf (%s)\n", argc < 2 ? "" : err); return 1; }
    quant_init(NULL);
    /* scalar reference, then every SIMD set this CPU can run */
    Kernels ks[3];
    int nk = 0;
    ks[nk++] = K_scalar;
#if defined(__x86_64__) || defined(_M_X64)
    SysInfo si;
    sys_info(&si);
    if (si.has_avx && si.has_f16c) { ks[nk] = K_scalar; kernels_init_avx(&ks[nk]); nk++; }
#endif
    if (strcmp(KS.name, "scalar") && strcmp(KS.name, "avx")) ks[nk++] = KS;
    const char *names[] = { "token_embd.weight", "blk.0.ffn_down.weight", "blk.0.ffn_gate.weight",
                            "blk.2.ffn_gate_exps.weight", "blk.2.ffn_down_exps.weight", "blk.2.attn_v.weight",
                            "blk.2.ffn_gate_up_exps.weight", "blk.21.ffn_gate_up_exps.weight", "blk.21.ffn_down_exps.weight" };
    int ok = 1;
    for (int ki = 0; ki < nk; ki++)
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            const GGUFTensor *t = gguf_tensor(&g, names[i]);
            if (t && ks[ki].dot[t->type]) ok &= check(ks[ki].name, &ks[ki], t);
        }
    const uint32_t types[] = { GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_Q4_0, GGML_NVFP4, GGML_NVFP4,
                               GGML_Q5_0, GGML_Q5_0, GGML_IQ4_NL, GGML_IQ4_NL, GGML_IQ4_XS, GGML_IQ2_XXS,
                               GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S };
    const int sizes[] = { 2048, 2048, 2048, 2048, 2048, 2304, 1152, 2304, 1152, 2304, 1152, 2304, 2304, 2304, 2304, 2304, 2304 };
    for (int ki = 0; ki < nk; ki++)
        for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++)
            if (ks[ki].dot[types[i]]) ok &= synth_check(ks[ki].name, &ks[ki], types[i], sizes[i]);
    /* rows converted to Q8_0 (Vulkan's copy of the IQ experts): every SIMD set against the scalar one,
     * within one step of the Q8_0 grid */
    const uint32_t qt[] = { GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S, GGML_IQ4_XS, GGML_Q4_K, GGML_Q6_K };
    for (int ki = 1; ki < nk; ki++)
        for (size_t i = 0; i < sizeof(qt) / sizeof(qt[0]); i++) {
            const int K = 2304;
            const size_t rb = row_bytes(qt[i], K), bsz = (size_t)ggml_type_size(qt[i]);
            uint8_t *data = malloc(rb);
            unsigned s = 5 + qt[i];
            for (size_t j = 0; j < rb; j++) { s = s * 1664525u + 1013904223u; data[j] = (uint8_t)(s >> 24); }
            for (size_t off = 0; off < rb; off += bsz) {
                f16_t d = f32_to_f16(0.01f + 0.001f * (float)(off % 7));
                memcpy(data + (qt[i] == GGML_Q6_K ? off + bsz - 2 : off), &d, 2);
            }
            int8_t qa[2304], qb[2304];
            f16_t da[72], db[72];
            K_scalar.row_to_q8_0(qt[i], data, K, qa, da);
            ks[ki].row_to_q8_0(qt[i], data, K, qb, db);
            double worst = 0;
            for (int j = 0; j < K; j++) {
                const double a = f16_to_f32(da[j / 32]) * qa[j], b = f16_to_f32(db[j / 32]) * qb[j];
                const double step = f16_to_f32(da[j / 32]) + 1e-12;
                if (fabs(a - b) / step > worst) worst = fabs(a - b) / step;
            }
            printf("%-7s to Q8_0 %-7s max diff %.2f steps %s\n", ks[ki].name, ggml_type_name(qt[i]), worst, worst <= 1.01 ? "OK" : "FAIL");
            ok &= worst <= 1.01;
            free(data);
        }
    /* rows in Vulkan's nibble layout (mv_iq), decoded back here: the same values as dequant_row */
    const uint32_t nt4[] = { GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS };
    for (int ki = 1; ki < nk; ki++)
        for (size_t i = 0; ks[ki].row_to_iq4 && i < sizeof(nt4) / sizeof(nt4[0]); i++) {
            const int K = 2304;
            const size_t rb = row_bytes(nt4[i], K), bsz = (size_t)ggml_type_size(nt4[i]);
            uint8_t *data = malloc(rb), nib[1152], scb[144];
            float dd[9], ref[2304];
            unsigned s = 11 + nt4[i];
            for (size_t j = 0; j < rb; j++) { s = s * 1664525u + 1013904223u; data[j] = (uint8_t)(s >> 24); }
            for (size_t off = 0; off < rb; off += bsz) {
                f16_t d = f32_to_f16(0.01f + 0.001f * (float)(off % 7));
                memcpy(data + off, &d, 2);
            }
            int good = ks[ki].row_to_iq4(nt4[i], data, K, nib, scb, dd);
            dequant_row(nt4[i], data, ref, K);
            double worst = 0;
            for (int j = 0; good && j < K; j++) {
                const int b = j / QK_K, two = scb[b * 16] >> 7, c = (nib[j / 2] >> (4 * (j & 1))) & 15, m = c & 7;
                const int mag = two ? (m == 0 ? 8 : m == 1 ? 25 : 43) : 8 * m + (m == 7 ? 6 : 4);
                const double v = dd[b] * (scb[j / 16] & 31) * (c & 8 ? -mag : mag);
                if (fabs(v - ref[j]) > worst) worst = fabs(v - ref[j]);
            }
            printf("%-7s to the mv_iq layout %-7s max diff %.2e %s\n", ks[ki].name, ggml_type_name(nt4[i]), worst, good && worst < 1e-6 ? "OK" : "FAIL");
            ok &= good && worst < 1e-6;
            free(data);
        }
    /* fp16 round trip */
    for (float f = -70000.f; f < 70000.f; f += 13.37f) {
        f16_t h = f32_to_f16(f);
        float back = f16_to_f32(h);
        if (fabsf(f) < 65000 && fabsf(back - f) > fabsf(f) * 1e-3f + 1e-3f) { printf("fp16 fail %f -> %f\n", f, back); ok = 0; break; }
    }
    printf(ok ? "ALL OK\n" : "SOME FAILED\n");
    /* single-core throughput over a whole tensor (second pass = resident in RAM), per SIMD set;
     * MoE models use an expert tensor, dense ones the first FFN */
    const char *bn[] = { "blk.2.ffn_gate_exps.weight", "blk.2.ffn_down_exps.weight", "blk.2.ffn_gate_up_exps.weight",
                         "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight" };
    const int moe = gguf_tensor(&g, bn[0]) || gguf_tensor(&g, bn[2]);
    for (int ki = 1; ki < nk; ki++)
    for (size_t i = 0; i < 5; i++) {
        const GGUFTensor *t = gguf_tensor(&g, bn[i]);
        if (!t || (i >= 3 && moe) || !ks[ki].dot[t->type]) continue;
        const Kernels KS = ks[ki];
        printf("[%s] ", KS.name);
        const int K = (int)t->ne[0];
        const int rows = (int)(t->ne[1] * t->ne[2]);
        const size_t rb = row_bytes(t->type, K);
        const int nb = (K + QK_K - 1) / QK_K, KP = nb * QK_K;
        block_q8_K *xq = malloc(sizeof(block_q8_K) * nb * 16);
        float *x = calloc((size_t)KP * 16, sizeof(float)), *out = malloc(sizeof(float) * 16);
        unsigned s = 7;
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < K; i++) x[j * KP + i] = frand(&s);
        for (int j = 0; j < 16; j++) KS.quantize_q8_K(x + j * KP, xq + j * nb, KP);
        for (int nt = 1; nt <= 16; nt *= 4)
            for (int pass = 0; pass < 2; pass++) {
                double t0 = time_sec();
                float acc = 0;
                for (int r = 0; r < rows; r++) {
                    KS.dot[t->type](K, t->data + (size_t)r * rb, xq, nb, nt, out, 1);
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
                    KS.dot[t->type](K, t->data + (size_t)r * rb, xq, nb, 1, out, 1);
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
