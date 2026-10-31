/* quant_avx.c - 128-bit kernels for x86 CPUs with AVX and F16C but no AVX2/FMA
 * (AMD Jaguar in the Xbox One, Ivy Bridge). Integer math is SSSE3/SSE4.1, compiled
 * with -mavx -mf16c (or /arch:AVX) for the VEX encoding; selected at runtime. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "gguf.h"
#include "quant.h"

static inline float hsum_ps(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_movehdup_ps(v));
    return _mm_cvtss_f32(v);
}

static inline int hsum_epi32(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}

/* a*b + c without FMA */
static inline __m128 madd_ps(__m128 a, __m128 b, __m128 c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }

static inline float h2f(f16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
static inline f16_t f2h(float f) { return (f16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0); }

#define LD(p) _mm_loadu_si128((const __m128i *)(p))

/* max tokens per pass in the multi-token kernels */
#define MT_MAX 8

/* Decode the 12-byte packed 6-bit scales/mins of Q4_K/Q5_K into 8+8 bytes. */
static inline void unpack_scales_k4(const uint8_t *s12, uint8_t *sc, uint8_t *mn) {
    uint32_t u[4];
    const uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
    memcpy(u, s12, 12);
    u[3] = ((u[2] >> 4) & km2) | (((u[1] >> 6) & km3) << 4);
    const uint32_t t = u[1] & km1;
    u[1] = (u[2] & km2) | (((u[0] >> 6) & km3) << 4);
    u[2] = t;
    u[0] &= km1;
    memcpy(sc, u, 8);
    memcpy(mn, u + 2, 8);
}

/* per-lane mins[j] * (bsums[2j] + bsums[2j+1]), 4 int32 lanes */
static inline __m128i mins_prod(__m128i mins16, const int16_t *bsums) {
    return _mm_madd_epi16(mins16, _mm_hadd_epi16(LD(bsums), LD(bsums + 8)));
}

/* sum over 32 values: q (unsigned, 2x16 bytes) times a (signed), as 8 int16 lanes */
static inline __m128i dot32_u8i8(__m128i q0, __m128i q1, const int8_t *a) {
    return _mm_add_epi16(_mm_maddubs_epi16(q0, LD(a)), _mm_maddubs_epi16(q1, LD(a + 16)));
}

/* ------------------------------------------------------------------ Q4_K */

static inline void q4_unpack(const block_q4_K *b, __m128i qv[16]) {
    const __m128i m4 = _mm_set1_epi8(0xF);
    for (int j = 0; j < 4; j++) {
        const __m128i b0 = LD(b->qs + 32 * j), b1 = LD(b->qs + 32 * j + 16);
        qv[4 * j + 0] = _mm_and_si128(b0, m4);
        qv[4 * j + 1] = _mm_and_si128(b1, m4);
        qv[4 * j + 2] = _mm_and_si128(_mm_srli_epi16(b0, 4), m4);
        qv[4 * j + 3] = _mm_and_si128(_mm_srli_epi16(b1, 4), m4);
    }
}

static void rows_q4_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q4_K *b = (const block_q4_K *)w;
    const int nb = K / QK_K;
    __m128 acc[MT_MAX], accm[MT_MAX];
    for (int t0 = 0; t0 < nt; t0 += MT_MAX) {
        const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;
        for (int t = 0; t < tn; t++) { acc[t] = _mm_setzero_ps(); accm[t] = _mm_setzero_ps(); }
        for (int i = 0; i < nb; i++) {
            uint8_t sc[8], mn[8];
            unpack_scales_k4(b[i].scales, sc, mn);
            const float d = h2f(b[i].d), dmin = h2f(b[i].dmin);
            __m128i qv[16], sv[8];
            q4_unpack(&b[i], qv);
            for (int j = 0; j < 8; j++) sv[j] = _mm_set1_epi16(sc[j]);
            const __m128i mins16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mn));
            for (int t = 0; t < tn; t++) {
                const block_q8_K *y = x + (size_t)(t0 + t) * xs + i;
                const int8_t *a = y->qs;
                __m128i s0 = _mm_madd_epi16(sv[0], dot32_u8i8(qv[0], qv[1], a));
                __m128i s1 = _mm_madd_epi16(sv[1], dot32_u8i8(qv[2], qv[3], a + 32));
                for (int j = 2; j < 8; j += 2) {
                    s0 = _mm_add_epi32(s0, _mm_madd_epi16(sv[j], dot32_u8i8(qv[2 * j], qv[2 * j + 1], a + 32 * j)));
                    s1 = _mm_add_epi32(s1, _mm_madd_epi16(sv[j + 1], dot32_u8i8(qv[2 * j + 2], qv[2 * j + 3], a + 32 * j + 32)));
                }
                acc[t] = madd_ps(_mm_set1_ps(d * y->d), _mm_cvtepi32_ps(_mm_add_epi32(s0, s1)), acc[t]);
                accm[t] = madd_ps(_mm_set1_ps(dmin * y->d), _mm_cvtepi32_ps(mins_prod(mins16, y->bsums)), accm[t]);
            }
        }
        for (int t = 0; t < tn; t++) out[(size_t)(t0 + t) * os] = hsum_ps(_mm_sub_ps(acc[t], accm[t]));
    }
}

/* ------------------------------------------------------------------ Q5_K */

static void rows_q5_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q5_K *b = (const block_q5_K *)w;
    const int nb = K / QK_K;
    const __m128i m4 = _mm_set1_epi8(0xF), one = _mm_set1_epi8(1);
    __m128 acc[MT_MAX], accm[MT_MAX];
    for (int t0 = 0; t0 < nt; t0 += MT_MAX) {
        const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;
        for (int t = 0; t < tn; t++) { acc[t] = _mm_setzero_ps(); accm[t] = _mm_setzero_ps(); }
        for (int i = 0; i < nb; i++) {
            uint8_t sc[8], mn[8];
            unpack_scales_k4(b[i].scales, sc, mn);
            const float d = h2f(b[i].d), dmin = h2f(b[i].dmin);
            const __m128i hb0 = LD(b[i].qh), hb1 = LD(b[i].qh + 16);
            __m128i qv[16];
            for (int j = 0; j < 4; j++) {
                const __m128i b0 = LD(b[i].qs + 32 * j), b1 = LD(b[i].qs + 32 * j + 16);
                const __m128i l0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb0, 2 * j), one), 4);
                const __m128i l1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb1, 2 * j), one), 4);
                const __m128i u0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb0, 2 * j + 1), one), 4);
                const __m128i u1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb1, 2 * j + 1), one), 4);
                qv[4 * j + 0] = _mm_or_si128(_mm_and_si128(b0, m4), l0);
                qv[4 * j + 1] = _mm_or_si128(_mm_and_si128(b1, m4), l1);
                qv[4 * j + 2] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b0, 4), m4), u0);
                qv[4 * j + 3] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), u1);
            }
            const __m128i mins16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mn));
            for (int t = 0; t < tn; t++) {
                const block_q8_K *y = x + (size_t)(t0 + t) * xs + i;
                __m128i s = _mm_setzero_si128();
                for (int j = 0; j < 8; j++)
                    s = _mm_add_epi32(s, _mm_madd_epi16(_mm_set1_epi16(sc[j]), dot32_u8i8(qv[2 * j], qv[2 * j + 1], y->qs + 32 * j)));
                acc[t] = madd_ps(_mm_set1_ps(d * y->d), _mm_cvtepi32_ps(s), acc[t]);
                accm[t] = madd_ps(_mm_set1_ps(dmin * y->d), _mm_cvtepi32_ps(mins_prod(mins16, y->bsums)), accm[t]);
            }
        }
        for (int t = 0; t < tn; t++) out[(size_t)(t0 + t) * os] = hsum_ps(_mm_sub_ps(acc[t], accm[t]));
    }
}

/* ------------------------------------------------------------------ Q6_K */

/* 16 vectors of 16 values (0..63); vector g is scale group g */
static inline void q6_unpack(const block_q6_K *b, __m128i qv[16]) {
    const __m128i m4 = _mm_set1_epi8(0xF), m2 = _mm_set1_epi8(3);
    for (int n = 0; n < 2; n++)
        for (int k = 0; k < 2; k++) {
            const __m128i l1 = LD(b->ql + 64 * n + 16 * k);
            const __m128i l2 = LD(b->ql + 64 * n + 32 + 16 * k);
            const __m128i h = LD(b->qh + 32 * n + 16 * k);
            const __m128i h0 = _mm_slli_epi16(_mm_and_si128(h, m2), 4);
            const __m128i h1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), m2), 4);
            const __m128i h2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m2), 4);
            const __m128i h3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), m2), 4);
            qv[8 * n + 0 + k] = _mm_or_si128(_mm_and_si128(l1, m4), h0);
            qv[8 * n + 2 + k] = _mm_or_si128(_mm_and_si128(l2, m4), h1);
            qv[8 * n + 4 + k] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l1, 4), m4), h2);
            qv[8 * n + 6 + k] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l2, 4), m4), h3);
        }
}

static void rows_q6_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q6_K *b = (const block_q6_K *)w;
    const int nb = K / QK_K;
    __m128 acc[MT_MAX];
    for (int t0 = 0; t0 < nt; t0 += MT_MAX) {
        const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;
        for (int t = 0; t < tn; t++) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; i++) {
            __m128i qv[16];
            q6_unpack(&b[i], qv);
            const __m128i sc8 = LD(b[i].scales);
            const __m128i sclo = _mm_cvtepi8_epi16(sc8), schi = _mm_cvtepi8_epi16(_mm_srli_si128(sc8, 8));
            const float d = h2f(b[i].d);
            for (int t = 0; t < tn; t++) {
                const block_q8_K *y = x + (size_t)(t0 + t) * xs + i;
                /* -32 offset folded in: sum_g sc[g] * bsum[g] */
                const __m128i bias = _mm_add_epi32(_mm_madd_epi16(sclo, LD(y->bsums)), _mm_madd_epi16(schi, LD(y->bsums + 8)));
                __m128i s0 = _mm_sub_epi32(_mm_setzero_si128(), _mm_slli_epi32(bias, 5));
                __m128i s1 = _mm_setzero_si128();
                for (int g = 0; g < 16; g += 2) {
                    s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_set1_epi16(b[i].scales[g]), _mm_maddubs_epi16(qv[g], LD(y->qs + 16 * g))));
                    s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_set1_epi16(b[i].scales[g + 1]), _mm_maddubs_epi16(qv[g + 1], LD(y->qs + 16 * g + 16))));
                }
                acc[t] = madd_ps(_mm_set1_ps(d * y->d), _mm_cvtepi32_ps(_mm_add_epi32(s0, s1)), acc[t]);
            }
        }
        for (int t = 0; t < tn; t++) out[(size_t)(t0 + t) * os] = hsum_ps(acc[t]);
    }
}

/* ------------------------------------------------------------------ Q8_0 / Q4_0 (32-blocks vs Q8_K) */

/* signed x signed over 16 bytes, 4 int32 lanes */
static inline __m128i dot_i8_16(__m128i q, __m128i y) {
    return _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(q, q), _mm_sign_epi8(y, q)), _mm_set1_epi16(1));
}

static void rows_q8_0(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q8_0 *b = (const block_q8_0 *)w;
    for (int t = 0; t < nt; t++) {
        const block_q8_K *y = x + (size_t)t * xs;
        __m128 acc = _mm_setzero_ps();
        for (int i = 0; i < K / QK_K; i++) {
            __m128 acc_b = _mm_setzero_ps();
            for (int j = 0; j < 8; j++) {
                const block_q8_0 *bb = &b[i * 8 + j];
                const __m128i p = _mm_add_epi32(dot_i8_16(LD(bb->qs), LD(y[i].qs + 32 * j)),
                                                dot_i8_16(LD(bb->qs + 16), LD(y[i].qs + 32 * j + 16)));
                acc_b = madd_ps(_mm_set1_ps(h2f(bb->d)), _mm_cvtepi32_ps(p), acc_b);
            }
            acc = madd_ps(_mm_set1_ps(y[i].d), acc_b, acc);
        }
        out[(size_t)t * os] = hsum_ps(acc);
    }
}

static void rows_q4_0(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q4_0 *b = (const block_q4_0 *)w;
    const __m128i m4 = _mm_set1_epi8(0xF), off = _mm_set1_epi8(8);
    for (int t = 0; t < nt; t++) {
        const block_q8_K *y = x + (size_t)t * xs;
        __m128 acc = _mm_setzero_ps();
        for (int i = 0; i < K / QK_K; i++) {
            __m128 acc_b = _mm_setzero_ps();
            for (int j = 0; j < 8; j++) {
                const block_q4_0 *bb = &b[i * 8 + j];
                const __m128i raw = LD(bb->qs);
                const __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, m4), off);
                const __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), off);
                const __m128i p = _mm_add_epi32(dot_i8_16(lo, LD(y[i].qs + 32 * j)), dot_i8_16(hi, LD(y[i].qs + 32 * j + 16)));
                acc_b = madd_ps(_mm_set1_ps(h2f(bb->d)), _mm_cvtepi32_ps(p), acc_b);
            }
            acc = madd_ps(_mm_set1_ps(y[i].d), acc_b, acc);
        }
        out[(size_t)t * os] = hsum_ps(acc);
    }
}

/* ------------------------------------------------------------------ activation quantization */

static void quantize_q8_K(const float *x, block_q8_K *y, int K) {
    const __m128 sign = _mm_set1_ps(-0.0f);
    for (int i = 0; i < K / QK_K; i++, x += QK_K) {
        __m128 mx = _mm_setzero_ps();
        for (int j = 0; j < QK_K; j += 4) mx = _mm_max_ps(mx, _mm_andnot_ps(sign, _mm_loadu_ps(x + j)));
        mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
        mx = _mm_max_ss(mx, _mm_movehdup_ps(mx));
        const float amax = _mm_cvtss_f32(mx);
        if (amax == 0) {
            memset(&y[i], 0, sizeof(block_q8_K));
            continue;
        }
        const float iscale = 127.f / amax;
        const __m128 vs = _mm_set1_ps(iscale);
        for (int g = 0; g < 16; g++) {
            const float *xx = x + 16 * g;
            const __m128i i0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xx + 0), vs));
            const __m128i i1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xx + 4), vs));
            const __m128i i2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xx + 8), vs));
            const __m128i i3 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(xx + 12), vs));
            y[i].bsums[g] = (int16_t)hsum_epi32(_mm_add_epi32(_mm_add_epi32(i0, i1), _mm_add_epi32(i2, i3)));
            _mm_storeu_si128((__m128i *)(y[i].qs + 16 * g), _mm_packs_epi16(_mm_packs_epi32(i0, i1), _mm_packs_epi32(i2, i3)));
        }
        y[i].d = 1.f / iscale;
    }
}

/* ------------------------------------------------------------------ float helpers */

static inline __m128 load_h4(const f16_t *p) { return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)p)); }

static void f16_to_f32_row(const f16_t *x, float *y, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(y + i, load_h4(x + i));
    for (; i < n; i++) y[i] = h2f(x[i]);
}

static void f32_to_f16_row(const float *x, f16_t *y, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storel_epi64((__m128i *)(y + i), _mm_cvtps_ph(_mm_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; i++) y[i] = f2h(x[i]);
}

static float dot_f32(const float *a, const float *b, int n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(), s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = madd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), s0);
        s1 = madd_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4), s1);
        s2 = madd_ps(_mm_loadu_ps(a + i + 8), _mm_loadu_ps(b + i + 8), s2);
        s3 = madd_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12), s3);
    }
    for (; i + 4 <= n; i += 4) s0 = madd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), s0);
    float s = hsum_ps(_mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_f16_f32(const f16_t *a, const float *b, int n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = madd_ps(load_h4(a + i), _mm_loadu_ps(b + i), s0);
        s1 = madd_ps(load_h4(a + i + 4), _mm_loadu_ps(b + i + 4), s1);
    }
    for (; i + 4 <= n; i += 4) s0 = madd_ps(load_h4(a + i), _mm_loadu_ps(b + i), s0);
    float s = hsum_ps(_mm_add_ps(s0, s1));
    for (; i < n; i++) s += h2f(a[i]) * b[i];
    return s;
}

static void axpy_f16(float a, const f16_t *x, float *y, int n) {
    const __m128 va = _mm_set1_ps(a);
    int i = 0;
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(y + i, madd_ps(va, load_h4(x + i), _mm_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * h2f(x[i]);
}

/* ------------------------------------------------------------------ vector exp (ggml_v_expf, SSE2 form) */

#define PS(bits) _mm_castsi128_ps(_mm_set1_epi32((int)(bits)))

static inline __m128 v_expf(__m128 x) {
    const __m128 r = PS(0x4b400000);
    const __m128 z = madd_ps(x, PS(0x3fb8aa3b), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(n, PS(0x3f317200))), _mm_mul_ps(n, PS(0x35bfbe8e)));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j = madd_ps(madd_ps(madd_ps(PS(0x3c072010), b, PS(0x3d2b9f17)), u, madd_ps(PS(0x3e2aaf33), b, PS(0x3efffedb))),
                             u, _mm_mul_ps(PS(0x3f7ffff6), b));
    if (!_mm_movemask_epi8(c)) return madd_ps(j, k, k);
    const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())), _mm_set1_epi32((int)0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(madd_ps(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), madd_ps(k, j, k)))));
}

/* g[i] = silu(g[i]) * u[i] */
static void silu_mul(float *g, const float *u, int n) {
    const __m128 one = _mm_set1_ps(1.f), neg = _mm_set1_ps(-0.f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 x = _mm_loadu_ps(g + i);
        const __m128 s = _mm_div_ps(x, _mm_add_ps(one, v_expf(_mm_xor_ps(x, neg))));
        _mm_storeu_ps(g + i, _mm_mul_ps(s, _mm_loadu_ps(u + i)));
    }
    for (; i < n; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

/* s[i] = exp(s[i] - mx); returns the sum */
static float exp_sum(float *s, int n, float mx) {
    const __m128 vm = _mm_set1_ps(mx);
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 e = v_expf(_mm_sub_ps(_mm_loadu_ps(s + i), vm));
        _mm_storeu_ps(s + i, e);
        acc = _mm_add_ps(acc, e);
    }
    float sum = hsum_ps(acc);
    for (; i < n; i++) { s[i] = expf(s[i] - mx); sum += s[i]; }
    return sum;
}

void kernels_init_avx(Kernels *k) {
    k->silu_mul = silu_mul;
    k->exp_sum = exp_sum;
    k->name = "avx";
    k->dot[GGML_Q4_K] = rows_q4_K;
    k->dot[GGML_Q5_K] = rows_q5_K;
    k->dot[GGML_Q6_K] = rows_q6_K;
    k->dot[GGML_Q8_0] = rows_q8_0;
    k->dot[GGML_Q4_0] = rows_q4_0;
    k->quantize_q8_K = quantize_q8_K;
    k->f16_to_f32_row = f16_to_f32_row;
    k->f32_to_f16_row = f32_to_f16_row;
    k->dot_f32 = dot_f32;
    k->dot_f16_f32 = dot_f16_f32;
    k->axpy_f16 = axpy_f16;
}
#endif
