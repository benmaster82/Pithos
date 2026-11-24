/* quant.c - portable scalar reference kernels + runtime dispatch. */
#include "quant.h"
#include "gguf.h"
#include "platform.h"
#include "iq_tables.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ fp16 */

static inline uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float f16_to_f32(f16_t h) {
    /* branch-free conversion (Maratyszcza/FP16) */
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const float exp_scale = bits_f32(0x07800000u);              /* 2^-112 */
    const float normalized = bits_f32((two_w >> 4) + (0xE0u << 23)) * exp_scale;
    const float denormalized = bits_f32((two_w >> 17) | (126u << 23)) - 0.5f;
    const uint32_t r = sign | (two_w < (1u << 27) ? f32_bits(denormalized) : f32_bits(normalized));
    return bits_f32(r);
}

f16_t f32_to_f16(float f) {
    float base = (fabsf(f) * bits_f32(0x77800000u)) * bits_f32(0x08800000u);
    const uint32_t w = f32_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = bits_f32((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = f32_bits(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (f16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

/* ------------------------------------------------------------------ helpers */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

size_t row_bytes(uint32_t type, int K) {
    return (size_t)K / ggml_block_size(type) * ggml_type_size(type);
}

int type_supported(uint32_t t) {
    return t == GGML_F32 || t == GGML_F16 || t == GGML_Q4_0 || t == GGML_Q8_0 ||
           t == GGML_Q4_K || t == GGML_Q5_K || t == GGML_Q6_K || t == GGML_NVFP4 || t == GGML_Q5_0 ||
           t == GGML_IQ2_XXS || t == GGML_IQ2_XS || t == GGML_IQ2_S || t == GGML_IQ3_XXS || t == GGML_IQ3_S ||
           t == GGML_IQ4_NL || t == GGML_IQ4_XS;
}

const int8_t nvfp4_values[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
float nvfp4_scale[256];

/* unsigned E4M3 (bias 7, subnormals; 0x7F, the NaN code, and the sign bit as in ggml), times 0.5 */
static void init_nvfp4_scale(void) {
    for (int x = 0; x < 256; x++) {
        const int e = (x >> 3) & 0xF, m = x & 7;
        const float v = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf((float)m, -9);
        nvfp4_scale[x] = x == 0 || x == 0x7F ? 0.f : v * 0.5f;
    }
}

/* ------------------------------------------------------------------ dequantization */

void dequant_row(uint32_t type, const void *src, float *y, int K) {
    switch (type) {
    case GGML_F32: memcpy(y, src, (size_t)K * 4); break;
    case GGML_F16: {
        const f16_t *s = (const f16_t *)src;
        for (int i = 0; i < K; i++) y[i] = f16_to_f32(s[i]);
    } break;
    case GGML_Q8_0: {
        const block_q8_0 *b = (const block_q8_0 *)src;
        for (int i = 0; i < K / QK8_0; i++) {
            float d = f16_to_f32(b[i].d);
            for (int j = 0; j < QK8_0; j++) *y++ = d * b[i].qs[j];
        }
    } break;
    case GGML_Q4_0: {
        const block_q4_0 *b = (const block_q4_0 *)src;
        for (int i = 0; i < K / QK8_0; i++) {
            float d = f16_to_f32(b[i].d);
            for (int j = 0; j < 16; j++) {
                y[j] = d * ((b[i].qs[j] & 0xF) - 8);
                y[j + 16] = d * ((b[i].qs[j] >> 4) - 8);
            }
            y += 32;
        }
    } break;
    case GGML_Q4_K: {
        const block_q4_K *b = (const block_q4_K *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d), dmin = f16_to_f32(b[i].dmin);
            const uint8_t *q = b[i].qs;
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                const float d1 = d * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                const float d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32;
                is += 2;
            }
        }
    } break;
    case GGML_Q5_K: {
        const block_q5_K *b = (const block_q5_K *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d), dmin = f16_to_f32(b[i].dmin);
            const uint8_t *ql = b[i].qs, *qh = b[i].qh;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                const float d1 = d * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                const float d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    } break;
    case GGML_Q6_K: {
        const block_q6_K *b = (const block_q6_K *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            const uint8_t *ql = b[i].ql, *qh = b[i].qh;
            const int8_t *sc = b[i].scales;
            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0] = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    } break;
    case GGML_NVFP4: {
        const block_nvfp4 *b = (const block_nvfp4 *)src;
        for (int i = 0; i < K / QK_NVFP4; i++)
            for (int s = 0; s < 4; s++) {
                const float d = nvfp4_scale[b[i].d[s]];
                for (int j = 0; j < 8; j++) {
                    y[j] = d * nvfp4_values[b[i].qs[8 * s + j] & 0xF];
                    y[j + 8] = d * nvfp4_values[b[i].qs[8 * s + j] >> 4];
                }
                y += 16;
            }
    } break;
    case GGML_Q5_0: {
        const block_q5_0 *b = (const block_q5_0 *)src;
        for (int i = 0; i < K / QK8_0; i++) {
            const float d = f16_to_f32(b[i].d);
            uint32_t qh;
            memcpy(&qh, b[i].qh, 4);
            for (int j = 0; j < 16; j++) {
                y[j] = d * ((int)((b[i].qs[j] & 0xF) | (((qh >> j) << 4) & 0x10)) - 16);
                y[j + 16] = d * ((int)((b[i].qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16);
            }
            y += 32;
        }
    } break;
    case GGML_IQ4_NL: {
        const block_iq4_nl *b = (const block_iq4_nl *)src;
        for (int i = 0; i < K / QK8_0; i++) {
            const float d = f16_to_f32(b[i].d);
            for (int j = 0; j < 16; j++) {
                y[j] = d * kvalues_iq4nl[b[i].qs[j] & 0xF];
                y[j + 16] = d * kvalues_iq4nl[b[i].qs[j] >> 4];
            }
            y += 32;
        }
    } break;
    case GGML_IQ4_XS: {
        const block_iq4_xs *b = (const block_iq4_xs *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            const uint8_t *qs = b[i].qs;
            for (int ib = 0; ib < 8; ib++) {
                const int ls = ((b[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((b[i].scales_h >> 2 * ib) & 3) << 4);
                const float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++) {
                    y[j] = dl * kvalues_iq4nl[qs[j] & 0xF];
                    y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
                }
                y += 32;
                qs += 16;
            }
        }
    } break;
    case GGML_IQ2_XXS: {
        const block_iq2_xxs *b = (const block_iq2_xxs *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            for (int ib = 0; ib < 8; ib++) {
                uint32_t a[2];
                memcpy(a, b[i].qs + 4 * ib, 8);
                const float db = d * (0.5f + (a[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++) {
                    const uint8_t *g = (const uint8_t *)(iq2xxs_grid + ((a[0] >> 8 * l) & 0xFF));
                    const uint8_t sg = ksigns_iq2xs[(a[1] >> 7 * l) & 127];
                    for (int j = 0; j < 8; j++) y[j] = db * g[j] * (sg >> j & 1 ? -1.f : 1.f);
                    y += 8;
                }
            }
        }
    } break;
    case GGML_IQ2_XS: {
        const block_iq2_xs *b = (const block_iq2_xs *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            for (int ib = 0; ib < 8; ib++)
                for (int l = 0; l < 4; l++) {
                    const float db = d * (0.5f + ((b[i].scales[ib] >> 4 * (l / 2)) & 0xF)) * 0.25f;
                    const uint16_t q = b[i].qs[4 * ib + l];
                    const uint8_t *g = (const uint8_t *)(iq2xs_grid + (q & 511));
                    const uint8_t sg = ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; j++) y[j] = db * g[j] * (sg >> j & 1 ? -1.f : 1.f);
                    y += 8;
                }
        }
    } break;
    case GGML_IQ2_S: {
        const block_iq2_s *b = (const block_iq2_s *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            const uint8_t *qs = b[i].qs, *sg = b[i].qs + QK_K / 8;
            for (int ib = 0; ib < 8; ib++) {
                for (int l = 0; l < 4; l++) {
                    const float db = d * (0.5f + ((b[i].scales[ib] >> 4 * (l / 2)) & 0xF)) * 0.25f;
                    const uint8_t *g = (const uint8_t *)(iq2s_grid + (qs[l] | ((b[i].qh[ib] << (8 - 2 * l)) & 0x300)));
                    for (int j = 0; j < 8; j++) y[j] = db * g[j] * (sg[l] >> j & 1 ? -1.f : 1.f);
                    y += 8;
                }
                qs += 4;
                sg += 4;
            }
        }
    } break;
    case GGML_IQ3_XXS: {
        const block_iq3_xxs *b = (const block_iq3_xxs *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            const uint8_t *qs = b[i].qs, *gas = b[i].qs + QK_K / 4;
            for (int ib = 0; ib < 8; ib++) {
                uint32_t a;
                memcpy(&a, gas + 4 * ib, 4);
                const float db = d * (0.5f + (a >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++) {
                    const uint8_t sg = ksigns_iq2xs[(a >> 7 * l) & 127];
                    const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l]);
                    const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
                    for (int j = 0; j < 4; j++) {
                        y[j] = db * g1[j] * (sg >> j & 1 ? -1.f : 1.f);
                        y[j + 4] = db * g2[j] * (sg >> (j + 4) & 1 ? -1.f : 1.f);
                    }
                    y += 8;
                }
                qs += 8;
            }
        }
    } break;
    case GGML_IQ3_S: {
        const block_iq3_s *b = (const block_iq3_s *)src;
        for (int i = 0; i < K / QK_K; i++) {
            const float d = f16_to_f32(b[i].d);
            const uint8_t *qs = b[i].qs, *sg = b[i].signs;
            for (int ib = 0; ib < 8; ib++) {
                const float db = d * (1 + 2 * ((b[i].scales[ib / 2] >> 4 * (ib % 2)) & 0xF));
                const int qh = b[i].qh[ib];
                for (int l = 0; l < 4; l++) {
                    const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
                    const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; j++) {
                        y[j] = db * g1[j] * (sg[l] >> j & 1 ? -1.f : 1.f);
                        y[j + 4] = db * g2[j] * (sg[l] >> (j + 4) & 1 ? -1.f : 1.f);
                    }
                    y += 8;
                }
                qs += 8;
                sg += 4;
            }
        }
    } break;
    default:
        memset(y, 0, (size_t)K * 4);
    }
}

/* ------------------------------------------------------------------ scalar kernels */

static void quantize_q8_K_scalar(const float *x, block_q8_K *y, int K) {
    for (int i = 0; i < K / QK_K; i++) {
        float amax = 0;
        for (int j = 0; j < QK_K; j++) {
            float a = fabsf(x[j]);
            if (a > amax) amax = a;
        }
        if (amax == 0) {
            memset(&y[i], 0, sizeof(block_q8_K));
            x += QK_K;
            continue;
        }
        const float iscale = 127.f / amax;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(x[j] * iscale);
            y[i].qs[j] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int s = 0;
            for (int l = 0; l < 16; l++) s += y[i].qs[j * 16 + l];
            y[i].bsums[j] = (int16_t)s;
        }
        y[i].d = 1.f / iscale;
        x += QK_K;
    }
}

static float dot_q4_K_1(int K, const block_q4_K *x, const block_q8_K *y) {
    float sumf = 0;
    for (int i = 0; i < K / QK_K; i++) {
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) get_scale_min_k4(j, x[i].scales, &sc[j], &mn[j]);
        const uint8_t *q = x[i].qs;
        const int8_t *a = y[i].qs;
        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            int s1 = 0, s2 = 0;
            for (int l = 0; l < 32; l++) {
                s1 += (q[l] & 0xF) * a[l];
                s2 += (q[l] >> 4) * a[l + 32];
            }
            sumi += sc[2 * j] * s1 + sc[2 * j + 1] * s2;
            q += 32;
            a += 64;
        }
        int summ = 0;
        for (int j = 0; j < 8; j++) summ += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
        sumf += y[i].d * (f16_to_f32(x[i].d) * sumi - f16_to_f32(x[i].dmin) * summ);
    }
    return sumf;
}

static float dot_q5_K_1(int K, const block_q5_K *x, const block_q8_K *y) {
    float sumf = 0;
    for (int i = 0; i < K / QK_K; i++) {
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) get_scale_min_k4(j, x[i].scales, &sc[j], &mn[j]);
        const uint8_t *q = x[i].qs, *qh = x[i].qh;
        const int8_t *a = y[i].qs;
        int sumi = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 4; j++) {
            int s1 = 0, s2 = 0;
            for (int l = 0; l < 32; l++) {
                s1 += ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) * a[l];
                s2 += ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) * a[l + 32];
            }
            sumi += sc[2 * j] * s1 + sc[2 * j + 1] * s2;
            q += 32;
            a += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
        int summ = 0;
        for (int j = 0; j < 8; j++) summ += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
        sumf += y[i].d * (f16_to_f32(x[i].d) * sumi - f16_to_f32(x[i].dmin) * summ);
    }
    return sumf;
}

static float dot_q6_K_1(int K, const block_q6_K *x, const block_q8_K *y) {
    float sumf = 0;
    for (int i = 0; i < K / QK_K; i++) {
        const uint8_t *ql = x[i].ql, *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        const int8_t *a = y[i].qs;
        int sumi = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int s[8] = {0};
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                const int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s[is + 0] += q1 * a[l + 0];
                s[is + 2] += q2 * a[l + 32];
                s[is + 4] += q3 * a[l + 64];
                s[is + 6] += q4 * a[l + 96];
            }
            for (int j = 0; j < 8; j++) sumi += sc[j] * s[j];
            ql += 64;
            qh += 32;
            sc += 8;
            a += 128;
        }
        sumf += y[i].d * f16_to_f32(x[i].d) * sumi;
    }
    return sumf;
}

/* K a multiple of 32: the last Q8_K block may be partial */
static float dot_q8_0_1(int K, const block_q8_0 *x, const block_q8_K *y) {
    float sumf = 0;
    for (int i = 0; i < (K + QK_K - 1) / QK_K; i++) {
        const block_q8_0 *b = x + i * 8;
        const int nj = K / QK8_0 - i * 8 < 8 ? K / QK8_0 - i * 8 : 8;
        float s = 0;
        for (int j = 0; j < nj; j++) {
            int si = 0;
            for (int l = 0; l < 32; l++) si += b[j].qs[l] * y[i].qs[j * 32 + l];
            s += f16_to_f32(b[j].d) * si;
        }
        sumf += s * y[i].d;
    }
    return sumf;
}

static float dot_q4_0_1(int K, const block_q4_0 *x, const block_q8_K *y) {
    float sumf = 0;
    for (int i = 0; i < K / QK_K; i++) {
        const block_q4_0 *b = x + i * 8;
        float s = 0;
        for (int j = 0; j < 8; j++) {
            const int8_t *a = y[i].qs + j * 32;
            int si = 0;
            for (int l = 0; l < 16; l++)
                si += ((b[j].qs[l] & 0xF) - 8) * a[l] + ((b[j].qs[l] >> 4) - 8) * a[l + 16];
            s += f16_to_f32(b[j].d) * si;
        }
        sumf += s * y[i].d;
    }
    return sumf;
}

/* 64-value blocks against Q8_K: block i reads the activations at i*64, four to a Q8_K block */
static float dot_nvfp4_1(int K, const block_nvfp4 *x, const block_q8_K *y) {
    float sumf = 0;
    for (int i = 0; i < K / QK_NVFP4; i++) {
        const int8_t *a = y[i / 4].qs + (i % 4) * QK_NVFP4;
        float s = 0;
        for (int k = 0; k < 4; k++) {
            int si = 0;
            for (int j = 0; j < 8; j++)
                si += nvfp4_values[x[i].qs[8 * k + j] & 0xF] * a[16 * k + j] +
                      nvfp4_values[x[i].qs[8 * k + j] >> 4] * a[16 * k + j + 8];
            s += nvfp4_scale[x[i].d[k]] * si;
        }
        sumf += s * y[i / 4].d;
    }
    return sumf;
}

#define DEF_ROWS(NAME, BT, FN)                                                              \
    static void NAME(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, \
                     int os) {                                                              \
        for (int t = 0; t < nt; t++) out[(size_t)t * os] = FN(K, (const BT *)w, x + (size_t)t * xs); \
    }
DEF_ROWS(rows_q4_K_scalar, block_q4_K, dot_q4_K_1)
DEF_ROWS(rows_q5_K_scalar, block_q5_K, dot_q5_K_1)
DEF_ROWS(rows_q6_K_scalar, block_q6_K, dot_q6_K_1)
DEF_ROWS(rows_q8_0_scalar, block_q8_0, dot_q8_0_1)
DEF_ROWS(rows_q4_0_scalar, block_q4_0, dot_q4_0_1)
DEF_ROWS(rows_nvfp4_scalar, block_nvfp4, dot_nvfp4_1)

/* The other formats through dequantization, 256 values at a time: K a multiple of the type's
 * block, the last Q8_K block may be partial */
static void rows_deq_scalar(uint32_t type, int K, const void *w, const block_q8_K *x, int xs, int nt, float *out,
                            int os) {
    float buf[QK_K];
    for (int t = 0; t < nt; t++) {
        const uint8_t *p = (const uint8_t *)w;
        const block_q8_K *y = x + (size_t)t * xs;
        float sumf = 0;
        for (int i = 0; i * QK_K < K; i++) {
            const int n = K - i * QK_K < QK_K ? K - i * QK_K : QK_K;
            dequant_row(type, p, buf, n);
            p += row_bytes(type, n);
            float s = 0;
            for (int j = 0; j < n; j++) s += buf[j] * y[i].qs[j];
            sumf += s * y[i].d;
        }
        out[(size_t)t * os] = sumf;
    }
}
#define DEF_DEQ(NAME, TYPE)                                                                          \
    static void NAME(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) { \
        rows_deq_scalar(TYPE, K, w, x, xs, nt, out, os);                                             \
    }
DEF_DEQ(rows_q5_0_scalar, GGML_Q5_0)
DEF_DEQ(rows_iq4_nl_scalar, GGML_IQ4_NL)
DEF_DEQ(rows_iq4_xs_scalar, GGML_IQ4_XS)
DEF_DEQ(rows_iq2_xxs_scalar, GGML_IQ2_XXS)
DEF_DEQ(rows_iq2_xs_scalar, GGML_IQ2_XS)
DEF_DEQ(rows_iq2_s_scalar, GGML_IQ2_S)
DEF_DEQ(rows_iq3_xxs_scalar, GGML_IQ3_XXS)
DEF_DEQ(rows_iq3_s_scalar, GGML_IQ3_S)

static void f16_to_f32_row_scalar(const f16_t *x, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = f16_to_f32(x[i]);
}
static void f32_to_f16_row_scalar(const float *x, f16_t *y, int n) {
    for (int i = 0; i < n; i++) y[i] = f32_to_f16(x[i]);
}
static float dot_f32_scalar(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}
static float dot_f16_f32_scalar(const f16_t *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += f16_to_f32(a[i]) * b[i];
    return s;
}
static void axpy_f16_scalar(float a, const f16_t *x, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] += a * f16_to_f32(x[i]);
}

static void silu_mul_scalar(float *g, const float *u, int n) {
    for (int i = 0; i < n; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}
static float exp_sum_scalar(float *s, int n, float mx) {
    float sum = 0;
    for (int i = 0; i < n; i++) { s[i] = expf(s[i] - mx); sum += s[i]; }
    return sum;
}

static void row_to_q8_0_scalar(uint32_t type, const void *row, int K, int8_t *qs, f16_t *d) {
    float y[QK_K];
    for (int i = 0; i < K / QK_K; i++) {
        dequant_row(type, (const uint8_t *)row + row_bytes(type, i * QK_K), y, QK_K);
        for (int j = 0; j < QK_K / QK8_0; j++, qs += QK8_0) {
            const float *b = y + j * QK8_0;
            float amax = 0;
            for (int l = 0; l < QK8_0; l++) amax = fabsf(b[l]) > amax ? fabsf(b[l]) : amax;
            const float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
            for (int l = 0; l < QK8_0; l++) qs[l] = (int8_t)lrintf(b[l] * id);
            *d++ = f32_to_f16(dd);
        }
    }
}

Kernels K_scalar;
Kernels KS;

static void init_scalar(Kernels *k) {
    memset(k, 0, sizeof(*k));
    k->name = "scalar";
    k->dot[GGML_Q4_K] = rows_q4_K_scalar;
    k->dot[GGML_Q5_K] = rows_q5_K_scalar;
    k->dot[GGML_Q6_K] = rows_q6_K_scalar;
    k->dot[GGML_Q8_0] = rows_q8_0_scalar;
    k->dot[GGML_Q4_0] = rows_q4_0_scalar;
    k->dot[GGML_NVFP4] = rows_nvfp4_scalar;
    k->dot[GGML_Q5_0] = rows_q5_0_scalar;
    k->dot[GGML_IQ4_NL] = rows_iq4_nl_scalar;
    k->dot[GGML_IQ4_XS] = rows_iq4_xs_scalar;
    k->dot[GGML_IQ2_XXS] = rows_iq2_xxs_scalar;
    k->dot[GGML_IQ2_XS] = rows_iq2_xs_scalar;
    k->dot[GGML_IQ2_S] = rows_iq2_s_scalar;
    k->dot[GGML_IQ3_XXS] = rows_iq3_xxs_scalar;
    k->dot[GGML_IQ3_S] = rows_iq3_s_scalar;
    k->quantize_q8_K = quantize_q8_K_scalar;
    k->f16_to_f32_row = f16_to_f32_row_scalar;
    k->f32_to_f16_row = f32_to_f16_row_scalar;
    k->dot_f32 = dot_f32_scalar;
    k->dot_f16_f32 = dot_f16_f32_scalar;
    k->axpy_f16 = axpy_f16_scalar;
    k->silu_mul = silu_mul_scalar;
    k->exp_sum = exp_sum_scalar;
    k->row_to_q8_0 = row_to_q8_0_scalar;
}

const char *quant_init(const char *force) {
    init_nvfp4_scale();
    init_scalar(&K_scalar);
    KS = K_scalar;
    if (force && !strcmp(force, "scalar")) return KS.name;
    SysInfo si;
    sys_info(&si);
#if defined(__x86_64__) || defined(_M_X64)
    const int avx2 = si.has_avx2 && si.has_fma && si.has_f16c, avx = si.has_avx && si.has_f16c;
    if (avx2 && !(force && !strcmp(force, "avx")))
        kernels_init_avx2(&KS);
    else if (avx)
        kernels_init_avx(&KS);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    kernels_init_neon(&KS);
#endif
    return KS.name;
}
