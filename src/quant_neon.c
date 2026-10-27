/* quant_neon.c - ARM64 NEON kernels (uses SDOT when compiled with +dotprod).
 * Formats without a NEON path fall back to the scalar kernels already in the table. */
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include "gguf.h"
#include "quant.h"

static inline int32x4_t dot16(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t p1 = vmull_high_s8(a, b);
    return vpadalq_s16(vpadalq_s16(acc, p0), p1);
#endif
}

static inline float h2f(f16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

static inline void scales_k4(const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    for (int j = 0; j < 8; j++) {
        if (j < 4) { sc[j] = q[j] & 63; mn[j] = q[j + 4] & 63; }
        else {
            sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }
}

static void rows_q4_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q4_K *b = (const block_q4_K *)w;
    const uint8x16_t m4 = vdupq_n_u8(0xF);
    for (int t = 0; t < nt; t++) out[(size_t)t * os] = 0;
    for (int i = 0; i < K / QK_K; i++) {
        uint8_t sc[8], mn[8];
        scales_k4(b[i].scales, sc, mn);
        const float d = h2f(b[i].d), dmin = h2f(b[i].dmin);
        int8x16_t q[16];
        for (int j = 0; j < 4; j++) {
            uint8x16_t b0 = vld1q_u8(b[i].qs + 32 * j), b1 = vld1q_u8(b[i].qs + 32 * j + 16);
            q[4 * j + 0] = vreinterpretq_s8_u8(vandq_u8(b0, m4));
            q[4 * j + 1] = vreinterpretq_s8_u8(vandq_u8(b1, m4));
            q[4 * j + 2] = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
            q[4 * j + 3] = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        }
        for (int t = 0; t < nt; t++) {
            const block_q8_K *y = x + (size_t)t * xs + i;
            int sumi = 0, summ = 0;
            for (int j = 0; j < 8; j++) {
                int32x4_t acc = vdupq_n_s32(0);
                acc = dot16(acc, q[2 * j], vld1q_s8(y->qs + 32 * j));
                acc = dot16(acc, q[2 * j + 1], vld1q_s8(y->qs + 32 * j + 16));
                sumi += sc[j] * vaddvq_s32(acc);
                summ += mn[j] * (y->bsums[2 * j] + y->bsums[2 * j + 1]);
            }
            out[(size_t)t * os] += y->d * (d * sumi - dmin * summ);
        }
    }
}

static void rows_q6_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q6_K *b = (const block_q6_K *)w;
    const uint8x16_t m4 = vdupq_n_u8(0xF), m3 = vdupq_n_u8(3);
    const int8x16_t m32 = vdupq_n_s8(32);
    for (int t = 0; t < nt; t++) out[(size_t)t * os] = 0;
    for (int i = 0; i < K / QK_K; i++) {
        int8x16_t q[16];
        for (int n = 0; n < 2; n++) {
            for (int hf = 0; hf < 2; hf++) {
                uint8x16_t l1 = vld1q_u8(b[i].ql + 64 * n + 16 * hf);
                uint8x16_t l2 = vld1q_u8(b[i].ql + 64 * n + 32 + 16 * hf);
                uint8x16_t h = vld1q_u8(b[i].qh + 32 * n + 16 * hf);
                uint8x16_t v0 = vorrq_u8(vandq_u8(l1, m4), vshlq_n_u8(vandq_u8(h, m3), 4));
                uint8x16_t v1 = vorrq_u8(vandq_u8(l2, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m3), 4));
                uint8x16_t v2 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m3), 4));
                uint8x16_t v3 = vorrq_u8(vshrq_n_u8(l2, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4));
                q[8 * n + 0 + hf] = vsubq_s8(vreinterpretq_s8_u8(v0), m32);
                q[8 * n + 2 + hf] = vsubq_s8(vreinterpretq_s8_u8(v1), m32);
                q[8 * n + 4 + hf] = vsubq_s8(vreinterpretq_s8_u8(v2), m32);
                q[8 * n + 6 + hf] = vsubq_s8(vreinterpretq_s8_u8(v3), m32);
            }
        }
        const float d = h2f(b[i].d);
        for (int t = 0; t < nt; t++) {
            const block_q8_K *y = x + (size_t)t * xs + i;
            int sumi = 0;
            for (int g = 0; g < 16; g++)
                sumi += b[i].scales[g] * vaddvq_s32(dot16(vdupq_n_s32(0), q[g], vld1q_s8(y->qs + 16 * g)));
            out[(size_t)t * os] += y->d * d * sumi;
        }
    }
}

static void quantize_q8_K(const float *x, block_q8_K *y, int K) {
    for (int i = 0; i < K / QK_K; i++, x += QK_K) {
        float32x4_t mx = vdupq_n_f32(0);
        for (int j = 0; j < QK_K; j += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(x + j)));
        const float amax = vmaxvq_f32(mx);
        if (amax == 0) { memset(&y[i], 0, sizeof(block_q8_K)); continue; }
        const float iscale = 127.f / amax;
        for (int j = 0; j < QK_K; j += 16) {
            int32x4_t a0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j), iscale));
            int32x4_t a1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 4), iscale));
            int32x4_t a2 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 8), iscale));
            int32x4_t a3 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 12), iscale));
            int16x8_t s0 = vcombine_s16(vqmovn_s32(a0), vqmovn_s32(a1));
            int16x8_t s1 = vcombine_s16(vqmovn_s32(a2), vqmovn_s32(a3));
            int8x16_t q = vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1));
            vst1q_s8(y[i].qs + j, q);
            y[i].bsums[j / 16] = (int16_t)vaddlvq_s8(q);
        }
        y[i].d = 1.f / iscale;
    }
}

static void f16_to_f32_row(const f16_t *x, float *y, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + i))));
    for (; i < n; i++) y[i] = f16_to_f32(x[i]);
}

static void f32_to_f16_row(const float *x, f16_t *y, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1_u16(y + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(x + i))));
    for (; i < n; i++) y[i] = f32_to_f16(x[i]);
}

static float dot_f32(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_f16_f32(const f16_t *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a + i))), vld1q_f32(b + i));
    float s = vaddvq_f32(s0);
    for (; i < n; i++) s += f16_to_f32(a[i]) * b[i];
    return s;
}

static void axpy_f16(float a, const f16_t *x, float *y, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + i))), a));
    for (; i < n; i++) y[i] += a * f16_to_f32(x[i]);
}

void kernels_init_neon(Kernels *k) {
#if defined(__ARM_FEATURE_DOTPROD)
    k->name = "neon+dotprod";
#else
    k->name = "neon";
#endif
    k->dot[GGML_Q4_K] = rows_q4_K;
    k->dot[GGML_Q6_K] = rows_q6_K;
    k->quantize_q8_K = quantize_q8_K;
    k->f16_to_f32_row = f16_to_f32_row;
    k->f32_to_f16_row = f32_to_f16_row;
    k->dot_f32 = dot_f32;
    k->dot_f16_f32 = dot_f16_f32;
    k->axpy_f16 = axpy_f16;
}
#endif
