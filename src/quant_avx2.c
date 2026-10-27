/* quant_avx2.c - AVX2/FMA/F16C kernels. Compiled with -mavx2 -mfma -mf16c (or /arch:AVX2)
 * and only selected at runtime when the CPU supports them. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "gguf.h"
#include "quant.h"

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

static inline int hsum_epi32_128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}

static inline int hsum_epi32(__m256i v) {
    return hsum_epi32_128(_mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
}

/* scalar F16C conversions through the vector instructions (MSVC lacks _cvtsh_ss) */
static inline float h2f(f16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
static inline f16_t f2h(float f) { return (f16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0); }

/* max tokens per pass in the multi-token kernels (per-token accumulators live on the stack) */
#define MT_MAX 16

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

/* sum_j mins[j] * (bsums[2j] + bsums[2j+1]) */
static inline int mins_dot(const uint8_t *mn, const int16_t *bsums) {
    __m128i b0 = _mm_loadu_si128((const __m128i *)bsums);
    __m128i b1 = _mm_loadu_si128((const __m128i *)(bsums + 8));
    __m128i bs = _mm_hadd_epi16(b0, b1);
    __m128i m = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mn));
    return hsum_epi32_128(_mm_madd_epi16(m, bs));
}

/* ------------------------------------------------------------------ Q4_K */

static void rows_q4_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q4_K *b = (const block_q4_K *)w;
    const int nb = K / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);

    if (nt == 1) {
        /* everything stays in vector registers: scales broadcast with pshufb,
         * min corrections accumulated as a vector (no per-block horizontal sums) */
        __m256 acc = _mm256_setzero_ps();
        __m128 acc_m = _mm_setzero_ps();
        const uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
        for (int i = 0; i < nb; i++) {
            uint32_t u[4];
            memcpy(u, b[i].scales, 12);
            u[3] = ((u[2] >> 4) & km2) | (((u[1] >> 6) & km3) << 4);
            const uint32_t t = u[1] & km1;
            u[1] = (u[2] & km2) | (((u[0] >> 6) & km3) << 4);
            u[2] = t;
            u[0] &= km1;
            const float dy = x[i].d;
            const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int)u[3], (int)u[2], (int)u[1], (int)u[0]));
            const __m256i bs = _mm256_loadu_si256((const __m256i *)x[i].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));
            const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(ms, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dy * h2f(b[i].dmin)), _mm_cvtepi32_ps(prod), acc_m);
            const __m128i sc128 = _mm256_castsi256_si128(ms);
            const __m256i scales = _mm256_set_m128i(sc128, sc128);
            const uint8_t *q = b[i].qs;
            const int8_t *a = x[i].qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < 4; j++) {
                const __m256i sl = _mm256_shuffle_epi8(scales, _mm256_set1_epi16((short)(((4 * j + 1) << 8) | (4 * j))));
                const __m256i sh = _mm256_shuffle_epi8(scales, _mm256_set1_epi16((short)(((4 * j + 3) << 8) | (4 * j + 2))));
                const __m256i qb = _mm256_loadu_si256((const __m256i *)(q + 32 * j));
                const __m256i ql = _mm256_and_si256(qb, m4);
                const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4);
                __m256i pl = _mm256_maddubs_epi16(ql, _mm256_loadu_si256((const __m256i *)(a + 64 * j)));
                __m256i ph = _mm256_maddubs_epi16(qh, _mm256_loadu_si256((const __m256i *)(a + 64 * j + 32)));
                pl = _mm256_madd_epi16(sl, pl);
                ph = _mm256_madd_epi16(sh, ph);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dy * h2f(b[i].d)), _mm256_cvtepi32_ps(sumi), acc);
        }
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        out[0] = hsum_ps(acc) - _mm_cvtss_f32(acc_m);
        return;
    }

    /* multi-token: unpack each weight block (values, scales, mins) once and reuse it for
     * every token; per-token accumulators stay vectors until the end of the row */
    __m256 acc[MT_MAX];
    __m128 accm[MT_MAX];
    for (int t0 = 0; t0 < nt; t0 += MT_MAX) {
        const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;
        for (int t = 0; t < tn; t++) { acc[t] = _mm256_setzero_ps(); accm[t] = _mm_setzero_ps(); }
        for (int i = 0; i < nb; i++) {
            uint8_t sc[8], mn[8];
            unpack_scales_k4(b[i].scales, sc, mn);
            const float d = h2f(b[i].d), dmin = h2f(b[i].dmin);
            __m256i qv[8], sv[8];
            for (int j = 0; j < 4; j++) {
                const __m256i qb = _mm256_loadu_si256((const __m256i *)(b[i].qs + 32 * j));
                qv[2 * j] = _mm256_and_si256(qb, m4);
                qv[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4);
            }
            for (int j = 0; j < 8; j++) sv[j] = _mm256_set1_epi16(sc[j]);
            const __m128i mins16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mn));
            for (int t = 0; t < tn; t++) {
                const block_q8_K *y = x + (size_t)(t0 + t) * xs + i;
                const int8_t *a = y->qs;
                __m256i s0 = _mm256_madd_epi16(sv[0], _mm256_maddubs_epi16(qv[0], _mm256_loadu_si256((const __m256i *)(a + 0))));
                __m256i s1 = _mm256_madd_epi16(sv[1], _mm256_maddubs_epi16(qv[1], _mm256_loadu_si256((const __m256i *)(a + 32))));
                for (int j = 2; j < 8; j += 2) {
                    s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(sv[j], _mm256_maddubs_epi16(qv[j], _mm256_loadu_si256((const __m256i *)(a + 32 * j)))));
                    s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(sv[j + 1], _mm256_maddubs_epi16(qv[j + 1], _mm256_loadu_si256((const __m256i *)(a + 32 * j + 32)))));
                }
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y->d), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), acc[t]);
                const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128((const __m128i *)y->bsums), _mm_loadu_si128((const __m128i *)(y->bsums + 8)));
                accm[t] = _mm_fmadd_ps(_mm_set1_ps(dmin * y->d), _mm_cvtepi32_ps(_mm_madd_epi16(mins16, bs)), accm[t]);
            }
        }
        for (int t = 0; t < tn; t++) {
            __m128 m = _mm_add_ps(accm[t], _mm_movehl_ps(accm[t], accm[t]));
            m = _mm_add_ss(m, _mm_movehdup_ps(m));
            out[(size_t)(t0 + t) * os] = hsum_ps(acc[t]) - _mm_cvtss_f32(m);
        }
    }
}

/* ------------------------------------------------------------------ Q5_K */

static void rows_q5_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q5_K *b = (const block_q5_K *)w;
    const int nb = K / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i one = _mm256_set1_epi8(1);
    for (int t = 0; t < nt; t++) out[(size_t)t * os] = 0;
    for (int i = 0; i < nb; i++) {
        uint8_t sc[8], mn[8];
        unpack_scales_k4(b[i].scales, sc, mn);
        const float d = h2f(b[i].d), dmin = h2f(b[i].dmin);
        const __m256i hb = _mm256_loadu_si256((const __m256i *)b[i].qh);
        __m256i qv[8];
        for (int j = 0; j < 4; j++) {
            const __m256i qb = _mm256_loadu_si256((const __m256i *)(b[i].qs + 32 * j));
            __m256i h0 = _mm256_and_si256(_mm256_srli_epi16(hb, 2 * j), one);
            __m256i h1 = _mm256_and_si256(_mm256_srli_epi16(hb, 2 * j + 1), one);
            qv[2 * j] = _mm256_or_si256(_mm256_and_si256(qb, m4), _mm256_slli_epi16(h0, 4));
            qv[2 * j + 1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m4), _mm256_slli_epi16(h1, 4));
        }
        for (int t = 0; t < nt; t++) {
            const block_q8_K *y = x + (size_t)t * xs + i;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < 8; j++) {
                __m256i p = _mm256_maddubs_epi16(qv[j], _mm256_loadu_si256((const __m256i *)(y->qs + 32 * j)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_set1_epi16(sc[j]), p));
            }
            out[(size_t)t * os] += y->d * (d * (float)hsum_epi32(sumi) - dmin * (float)mins_dot(mn, y->bsums));
        }
    }
}

/* ------------------------------------------------------------------ Q6_K */

static inline void q6_unpack(const block_q6_K *b, __m256i qv[8]) {
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3);
    for (int n = 0; n < 2; n++) {
        const __m256i l1 = _mm256_loadu_si256((const __m256i *)(b->ql + 64 * n));
        const __m256i l2 = _mm256_loadu_si256((const __m256i *)(b->ql + 64 * n + 32));
        const __m256i h = _mm256_loadu_si256((const __m256i *)(b->qh + 32 * n));
        const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(h, m2), 4);
        const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m2), 4);
        const __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m2), 4);
        const __m256i h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m2), 4);
        qv[4 * n + 0] = _mm256_or_si256(_mm256_and_si256(l1, m4), h0);
        qv[4 * n + 1] = _mm256_or_si256(_mm256_and_si256(l2, m4), h1);
        qv[4 * n + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), h2);
        qv[4 * n + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l2, 4), m4), h3);
    }
}

static void rows_q6_K(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q6_K *b = (const block_q6_K *)w;
    const int nb = K / QK_K;
    if (nt == 1) {
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            __m256i qv[8];
            q6_unpack(&b[i], qv);
            const float dd = x[i].d * h2f(b[i].d);
            const __m128i sc8 = _mm_loadu_si128((const __m128i *)b[i].scales);
            /* -32 offset folded in: sum_g sc[g] * bsum[g] */
            const __m256i bias = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc8), _mm256_loadu_si256((const __m256i *)x[i].bsums));
            __m256i sumi = _mm256_slli_epi32(bias, 5);
            sumi = _mm256_sub_epi32(_mm256_setzero_si256(), sumi);
            for (int j = 0; j < 8; j++) {
                /* 8 x scales[2j] then 8 x scales[2j+1], sign-extended to int16 */
                const __m128i shuf = _mm_set_epi64x((long long)(0x0101010101010101ull * (uint64_t)(2 * j + 1)),
                                                    (long long)(0x0101010101010101ull * (uint64_t)(2 * j)));
                const __m256i scv = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(sc8, shuf));
                __m256i p = _mm256_maddubs_epi16(qv[j], _mm256_loadu_si256((const __m256i *)(x[i].qs + 32 * j)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scv, p));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(sumi), acc);
        }
        out[0] = hsum_ps(acc);
        return;
    }
    __m256 acc[MT_MAX];
    for (int t0 = 0; t0 < nt; t0 += MT_MAX) {
        const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;
        for (int t = 0; t < tn; t++) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            __m256i qv[8], scv[8];
            q6_unpack(&b[i], qv);
            const __m128i sc8 = _mm_loadu_si128((const __m128i *)b[i].scales);
            for (int j = 0; j < 8; j++) {
                const __m128i shuf = _mm_set_epi64x((long long)(0x0101010101010101ull * (uint64_t)(2 * j + 1)),
                                                    (long long)(0x0101010101010101ull * (uint64_t)(2 * j)));
                scv[j] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(sc8, shuf));
            }
            const __m256i sc16 = _mm256_cvtepi8_epi16(sc8);
            const float d = h2f(b[i].d);
            for (int t = 0; t < tn; t++) {
                const block_q8_K *y = x + (size_t)(t0 + t) * xs + i;
                const __m256i bias = _mm256_madd_epi16(sc16, _mm256_loadu_si256((const __m256i *)y->bsums));
                __m256i s0 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(bias, 5));
                __m256i s1 = _mm256_madd_epi16(scv[1], _mm256_maddubs_epi16(qv[1], _mm256_loadu_si256((const __m256i *)(y->qs + 32))));
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(scv[0], _mm256_maddubs_epi16(qv[0], _mm256_loadu_si256((const __m256i *)y->qs))));
                for (int j = 2; j < 8; j += 2) {
                    s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(scv[j], _mm256_maddubs_epi16(qv[j], _mm256_loadu_si256((const __m256i *)(y->qs + 32 * j)))));
                    s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(scv[j + 1], _mm256_maddubs_epi16(qv[j + 1], _mm256_loadu_si256((const __m256i *)(y->qs + 32 * j + 32)))));
                }
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y->d), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), acc[t]);
            }
        }
        for (int t = 0; t < tn; t++) out[(size_t)(t0 + t) * os] = hsum_ps(acc[t]);
    }
}

/* ------------------------------------------------------------------ Q8_0 / Q4_0 (32-blocks vs Q8_K) */

static inline __m256i dot_i8_32(__m256i q, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(q, q);
    const __m256i sy = _mm256_sign_epi8(y, q);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1));
}

static void rows_q8_0(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q8_0 *b = (const block_q8_0 *)w;
    for (int t = 0; t < nt; t++) {
        const block_q8_K *y = x + (size_t)t * xs;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < K / QK_K; i++) {
            __m256 acc_b = _mm256_setzero_ps();
            for (int j = 0; j < 8; j++) {
                const block_q8_0 *bb = &b[i * 8 + j];
                __m256i q = _mm256_loadu_si256((const __m256i *)bb->qs);
                __m256i p = dot_i8_32(q, _mm256_loadu_si256((const __m256i *)(y[i].qs + 32 * j)));
                acc_b = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bb->d)), _mm256_cvtepi32_ps(p), acc_b);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d), acc_b, acc);
        }
        out[(size_t)t * os] = hsum_ps(acc);
    }
}

static void rows_q4_0(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_q4_0 *b = (const block_q4_0 *)w;
    const __m256i m4 = _mm256_set1_epi8(0xF), off = _mm256_set1_epi8(8);
    for (int t = 0; t < nt; t++) {
        const block_q8_K *y = x + (size_t)t * xs;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < K / QK_K; i++) {
            __m256 acc_b = _mm256_setzero_ps();
            for (int j = 0; j < 8; j++) {
                const block_q4_0 *bb = &b[i * 8 + j];
                __m128i raw = _mm_loadu_si128((const __m128i *)bb->qs);
                __m256i q = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);
                q = _mm256_sub_epi8(_mm256_and_si256(q, m4), off);
                __m256i p = dot_i8_32(q, _mm256_loadu_si256((const __m256i *)(y[i].qs + 32 * j)));
                acc_b = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bb->d)), _mm256_cvtepi32_ps(p), acc_b);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d), acc_b, acc);
        }
        out[(size_t)t * os] = hsum_ps(acc);
    }
}

/* ------------------------------------------------------------------ activation quantization */

static void quantize_q8_K(const float *x, block_q8_K *y, int K) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i = 0; i < K / QK_K; i++, x += QK_K) {
        __m256 mx = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + j)));
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float amax = _mm_cvtss_f32(m);
        if (amax == 0) {
            memset(&y[i], 0, sizeof(block_q8_K));
            continue;
        }
        const float iscale = 127.f / amax;
        const __m256 vs = _mm256_set1_ps(iscale);
        for (int j = 0; j < 8; j++) {
            const float *xx = x + 32 * j;
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xx + 0), vs));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xx + 8), vs));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xx + 16), vs));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xx + 24), vs));
            y[i].bsums[2 * j] = (int16_t)hsum_epi32(_mm256_add_epi32(i0, i1));
            y[i].bsums[2 * j + 1] = (int16_t)hsum_epi32(_mm256_add_epi32(i2, i3));
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            i0 = _mm256_permutevar8x32_epi32(i0, perm);
            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
        }
        y[i].d = 1.f / iscale;
    }
}

/* ------------------------------------------------------------------ float helpers */

static void f16_to_f32_row(const f16_t *x, float *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))));
    for (; i < n; i++) y[i] = h2f(x[i]);
}

static void f32_to_f16_row(const float *x, f16_t *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(y + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; i++) y[i] = f2h(x[i]);
}

static float dot_f32(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float s = hsum_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_f16_f32(const f16_t *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i + 8))), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_loadu_ps(b + i), s0);
    float s = hsum_ps(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += h2f(a[i]) * b[i];
    return s;
}

static void axpy_f16(float a, const f16_t *x, float *y, int n) {
    const __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))),
                                                _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * h2f(x[i]);
}

/* ------------------------------------------------------------------ vector exp (ggml_v_expf, ~1.5 ulp) */

#define PS(bits) _mm256_castsi256_ps(_mm256_set1_epi32((int)(bits)))

static inline __m256 v_expf(__m256 x) {
    const __m256 r = PS(0x4b400000);
    const __m256 z = _mm256_fmadd_ps(x, PS(0x3fb8aa3b), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, PS(0x35bfbe8e), _mm256_fnmadd_ps(n, PS(0x3f317200), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(PS(0x3c072010), b, PS(0x3d2b9f17)), u, _mm256_fmadd_ps(PS(0x3e2aaf33), b, PS(0x3efffedb))),
        u, _mm256_mul_ps(PS(0x3f7ffff6), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                       _mm256_set1_epi32((int)0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d),
                         _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                                      _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

/* g[i] = silu(g[i]) * u[i] */
static void silu_mul(float *g, const float *u, int n) {
    const __m256 one = _mm256_set1_ps(1.f), neg = _mm256_set1_ps(-0.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(g + i);
        const __m256 s = _mm256_div_ps(x, _mm256_add_ps(one, v_expf(_mm256_xor_ps(x, neg))));
        _mm256_storeu_ps(g + i, _mm256_mul_ps(s, _mm256_loadu_ps(u + i)));
    }
    for (; i < n; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

/* s[i] = exp(s[i] - mx); returns the sum */
static float exp_sum(float *s, int n, float mx) {
    const __m256 vm = _mm256_set1_ps(mx);
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 e = v_expf(_mm256_sub_ps(_mm256_loadu_ps(s + i), vm));
        _mm256_storeu_ps(s + i, e);
        acc = _mm256_add_ps(acc, e);
    }
    float sum = hsum_ps(acc);
    for (; i < n; i++) { s[i] = expf(s[i] - mx); sum += s[i]; }
    return sum;
}

void kernels_init_avx2(Kernels *k) {
    k->silu_mul = silu_mul;
    k->exp_sum = exp_sum;
    k->name = "avx2";
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
