/* quant_avx2.c - AVX2/FMA/F16C kernels. Compiled with -mavx2 -mfma -mf16c (or /arch:AVX2)
 * and only selected at runtime when the CPU supports them. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "gguf.h"
#include "quant.h"
#include "iq_tables.h"

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

/* ------------------------------------------------------------------ NVFP4 */

/* One 64-value block as two vectors of signed int8 weights in activation order (wa: values
 * 0-31 = sub-blocks 0 and 1, wb: values 32-63), with the matching scales per 128-bit lane. */
static inline void nvfp4_unpack(const block_nvfp4 *b, __m256i *wa, __m256i *wb, __m256 *sa, __m256 *sb) {
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)nvfp4_values));
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i qb = _mm256_loadu_si256((const __m256i *)b->qs);
    const __m256i lo = _mm256_and_si256(qb, m4), hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4);
    /* per lane: sub-blocks 0|2 and 1|3, each its low then high nibbles */
    const __m256i x = _mm256_unpacklo_epi64(lo, hi), y = _mm256_unpackhi_epi64(lo, hi);
    *wa = _mm256_shuffle_epi8(table, _mm256_permute2x128_si256(x, y, 0x20));
    *wb = _mm256_shuffle_epi8(table, _mm256_permute2x128_si256(x, y, 0x31));
    *sa = _mm256_set_m128(_mm_set1_ps(nvfp4_scale[b->d[1]]), _mm_set1_ps(nvfp4_scale[b->d[0]]));
    *sb = _mm256_set_m128(_mm_set1_ps(nvfp4_scale[b->d[3]]), _mm_set1_ps(nvfp4_scale[b->d[2]]));
}

/* int32 sums of 4-value groups of w*a (each 128-bit lane stays within one sub-block) */
static inline __m256i nvfp4_dot(__m256i w, __m256i a) {
    const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

static void rows_nvfp4(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {
    const block_nvfp4 *b = (const block_nvfp4 *)w;
    const int nb = K / QK_NVFP4;

    if (nt == 1) {
        __m256 acc = _mm256_setzero_ps();
        for (int i0 = 0; i0 < nb; i0 += 4) {
            const int i1 = nb - i0 < 4 ? nb : i0 + 4;
            __m256 accb = _mm256_setzero_ps();
            for (int i = i0; i < i1; i++) {
                __m256i wa, wb;
                __m256 sa, sb;
                nvfp4_unpack(b + i, &wa, &wb, &sa, &sb);
                const int8_t *a = x[i / 4].qs + (i % 4) * QK_NVFP4;
                accb = _mm256_fmadd_ps(sa, _mm256_cvtepi32_ps(nvfp4_dot(wa, _mm256_loadu_si256((const __m256i *)a))), accb);
                accb = _mm256_fmadd_ps(sb, _mm256_cvtepi32_ps(nvfp4_dot(wb, _mm256_loadu_si256((const __m256i *)(a + 32)))), accb);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(x[i0 / 4].d), accb, acc);
        }
        out[0] = hsum_ps(acc);
        return;
    }

    /* multi-token: each weight block is unpacked once per group of MT_MAX tokens */
    __m256 acc[MT_MAX], accb[MT_MAX];
    for (int t0 = 0; t0 < nt; t0 += MT_MAX) {
        const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;
        for (int t = 0; t < tn; t++) acc[t] = _mm256_setzero_ps();
        for (int i0 = 0; i0 < nb; i0 += 4) {
            const int i1 = nb - i0 < 4 ? nb : i0 + 4;
            for (int t = 0; t < tn; t++) accb[t] = _mm256_setzero_ps();
            for (int i = i0; i < i1; i++) {
                __m256i wa, wb;
                __m256 sa, sb;
                nvfp4_unpack(b + i, &wa, &wb, &sa, &sb);
                for (int t = 0; t < tn; t++) {
                    const int8_t *a = x[(size_t)(t0 + t) * xs + i / 4].qs + (i % 4) * QK_NVFP4;
                    accb[t] = _mm256_fmadd_ps(sa, _mm256_cvtepi32_ps(nvfp4_dot(wa, _mm256_loadu_si256((const __m256i *)a))), accb[t]);
                    accb[t] = _mm256_fmadd_ps(sb, _mm256_cvtepi32_ps(nvfp4_dot(wb, _mm256_loadu_si256((const __m256i *)(a + 32)))), accb[t]);
                }
            }
            for (int t = 0; t < tn; t++)
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(x[(size_t)(t0 + t) * xs + i0 / 4].d), accb[t], acc[t]);
        }
        for (int t = 0; t < tn; t++) out[(size_t)(t0 + t) * os] = hsum_ps(acc[t]);
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
        for (int i = 0; i < (K + QK_K - 1) / QK_K; i++) {
            const int nj = K / QK8_0 - i * 8 < 8 ? K / QK8_0 - i * 8 : 8;   /* the last Q8_K block may be partial */
            __m256 acc_b = _mm256_setzero_ps();
            for (int j = 0; j < nj; j++) {
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

/* ------------------------------------------------------------------ IQ formats */

/* 32 sign bits (bit k for value k) as bytes, 0xFF where the value is negative */
static inline __m256i iq_sign_mask(uint32_t bits) {
    const __m256i sel = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    const __m256i v = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), sel), bit);
    return _mm256_cmpeq_epi8(v, bit);
}

static inline __m256i iq_signed(__m256i g, uint32_t bits) {
    const __m256i m = iq_sign_mask(bits);
    return _mm256_sub_epi8(_mm256_xor_si256(g, m), m);
}

/* four 7-bit sign fields with even parity -> 32 sign bits */
static inline uint32_t iq_signs7(uint32_t a) {
    return (uint32_t)ksigns_iq2xs[a & 127] | (uint32_t)ksigns_iq2xs[(a >> 7) & 127] << 8 |
           (uint32_t)ksigns_iq2xs[(a >> 14) & 127] << 16 | (uint32_t)ksigns_iq2xs[(a >> 21) & 127] << 24;
}

/* int16 scales for 32 values: lo for the first 16, hi for the other 16 */
static inline __m256i iq_scale2(int lo, int hi) {
    return _mm256_set_m128i(_mm_set1_epi16((short)hi), _mm_set1_epi16((short)lo));
}

/* Each unpacker turns a 256-value block into signed int8 weights q[8] (32 values each, in
 * activation order), int16 scales sc[8] and the block's float factor *d */
static inline void unpack_iq2_xxs(const block_iq2_xxs *b, __m256i q[8], __m256i sc[8], float *d) {
    *d = 0.125f * h2f(b->d);
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a[2];
        memcpy(a, b->qs + 4 * ib, 8);
        const __m256i g = _mm256_set_epi64x((long long)iq2xxs_grid[a[0] >> 24], (long long)iq2xxs_grid[(a[0] >> 16) & 255],
                                            (long long)iq2xxs_grid[(a[0] >> 8) & 255], (long long)iq2xxs_grid[a[0] & 255]);
        q[ib] = iq_signed(g, iq_signs7(a[1]));
        sc[ib] = _mm256_set1_epi16((short)(2 * (a[1] >> 28) + 1));
    }
}

static inline void unpack_iq2_xs(const block_iq2_xs *b, __m256i q[8], __m256i sc[8], float *d) {
    *d = 0.125f * h2f(b->d);
    for (int ib = 0; ib < 8; ib++) {
        const uint16_t *qs = b->qs + 4 * ib;
        const __m256i g = _mm256_set_epi64x((long long)iq2xs_grid[qs[3] & 511], (long long)iq2xs_grid[qs[2] & 511],
                                            (long long)iq2xs_grid[qs[1] & 511], (long long)iq2xs_grid[qs[0] & 511]);
        const uint32_t sg = (uint32_t)ksigns_iq2xs[qs[0] >> 9] | (uint32_t)ksigns_iq2xs[qs[1] >> 9] << 8 |
                            (uint32_t)ksigns_iq2xs[qs[2] >> 9] << 16 | (uint32_t)ksigns_iq2xs[qs[3] >> 9] << 24;
        q[ib] = iq_signed(g, sg);
        sc[ib] = iq_scale2(2 * (b->scales[ib] & 15) + 1, 2 * (b->scales[ib] >> 4) + 1);
    }
}

static inline void unpack_iq2_s(const block_iq2_s *b, __m256i q[8], __m256i sc[8], float *d) {
    *d = 0.125f * h2f(b->d);
    for (int ib = 0; ib < 8; ib++) {
        const uint8_t *qs = b->qs + 4 * ib;
        const int h = b->qh[ib];
        const __m256i g = _mm256_set_epi64x((long long)iq2s_grid[qs[3] | (h << 2 & 0x300)], (long long)iq2s_grid[qs[2] | (h << 4 & 0x300)],
                                            (long long)iq2s_grid[qs[1] | (h << 6 & 0x300)], (long long)iq2s_grid[qs[0] | (h << 8 & 0x300)]);
        uint32_t sg;
        memcpy(&sg, b->qs + QK_K / 8 + 4 * ib, 4);
        q[ib] = iq_signed(g, sg);
        sc[ib] = iq_scale2(2 * (b->scales[ib] & 15) + 1, 2 * (b->scales[ib] >> 4) + 1);
    }
}

static inline void unpack_iq3_xxs(const block_iq3_xxs *b, __m256i q[8], __m256i sc[8], float *d) {
    *d = 0.25f * h2f(b->d);
    for (int ib = 0; ib < 8; ib++) {
        const uint8_t *q3 = b->qs + 8 * ib;
        uint32_t a;
        memcpy(&a, b->qs + QK_K / 4 + 4 * ib, 4);
        const __m256i g = _mm256_set_epi32((int)iq3xxs_grid[q3[7]], (int)iq3xxs_grid[q3[6]], (int)iq3xxs_grid[q3[5]], (int)iq3xxs_grid[q3[4]],
                                           (int)iq3xxs_grid[q3[3]], (int)iq3xxs_grid[q3[2]], (int)iq3xxs_grid[q3[1]], (int)iq3xxs_grid[q3[0]]);
        q[ib] = iq_signed(g, iq_signs7(a));
        sc[ib] = _mm256_set1_epi16((short)(2 * (a >> 28) + 1));
    }
}

static inline void unpack_iq3_s(const block_iq3_s *b, __m256i q[8], __m256i sc[8], float *d) {
    *d = h2f(b->d);
    for (int ib = 0; ib < 8; ib++) {
        const uint8_t *qs = b->qs + 8 * ib;
        const int h = b->qh[ib];
#define G3(k) (int)iq3s_grid[qs[k] | ((h >> (k) & 1) << 8)]
        const __m256i g = _mm256_set_epi32(G3(7), G3(6), G3(5), G3(4), G3(3), G3(2), G3(1), G3(0));
#undef G3
        uint32_t sg;
        memcpy(&sg, b->signs + 4 * ib, 4);
        q[ib] = iq_signed(g, sg);
        sc[ib] = _mm256_set1_epi16((short)(1 + 2 * ((b->scales[ib / 2] >> 4 * (ib % 2)) & 15)));
    }
}

/* 16 bytes of nibbles -> 32 int8 codebook values: low nibbles first, then high */
static inline __m256i iq4_values(const uint8_t *qs) {
    const __m128i tab = _mm_loadu_si128((const __m128i *)kvalues_iq4nl), m4 = _mm_set1_epi8(0xF);
    const __m128i v = _mm_loadu_si128((const __m128i *)qs);
    return _mm256_set_m128i(_mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(v, 4), m4)), _mm_shuffle_epi8(tab, _mm_and_si128(v, m4)));
}

static inline void unpack_iq4_xs(const block_iq4_xs *b, __m256i q[8], __m256i sc[8], float *d) {
    *d = h2f(b->d);
    for (int ib = 0; ib < 8; ib++) {
        q[ib] = iq4_values(b->qs + 16 * ib);
        const int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 15) | ((b->scales_h >> 2 * ib) & 3) << 4;
        sc[ib] = _mm256_set1_epi16((short)(ls - 32));
    }
}

/* sum over the block of q * sc * a, as int32 lanes */
static inline __m256i iq_dot8(const __m256i q[8], const __m256i sc[8], const int8_t *a) {
    __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
    for (int j = 0; j < 8; j += 2) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + 32 * j));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + 32 * j + 32));
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(q[j]), _mm256_sign_epi8(a0, q[j])), sc[j]));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(q[j + 1]), _mm256_sign_epi8(a1, q[j + 1])), sc[j + 1]));
    }
    return _mm256_add_epi32(s0, s1);
}

/* each weight block is unpacked once per group of MT_MAX tokens */
#define DEF_IQ_ROWS(NAME, BT, UNPACK)                                                                     \
    static void NAME(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {    \
        const BT *b = (const BT *)w;                                                                      \
        const int nb = K / QK_K;                                                                          \
        __m256 acc[MT_MAX];                                                                               \
        for (int t0 = 0; t0 < nt; t0 += MT_MAX) {                                                         \
            const int tn = nt - t0 < MT_MAX ? nt - t0 : MT_MAX;                                           \
            for (int t = 0; t < tn; t++) acc[t] = _mm256_setzero_ps();                                    \
            for (int i = 0; i < nb; i++) {                                                                \
                __m256i q[8], sc[8];                                                                      \
                float d;                                                                                  \
                UNPACK(b + i, q, sc, &d);                                                                 \
                for (int t = 0; t < tn; t++) {                                                            \
                    const block_q8_K *y = x + (size_t)(t0 + t) * xs + i;                                  \
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y->d), _mm256_cvtepi32_ps(iq_dot8(q, sc, y->qs)), acc[t]); \
                }                                                                                         \
            }                                                                                             \
            for (int t = 0; t < tn; t++) out[(size_t)(t0 + t) * os] = hsum_ps(acc[t]);                    \
        }                                                                                                 \
    }
DEF_IQ_ROWS(rows_iq2_xxs, block_iq2_xxs, unpack_iq2_xxs)
DEF_IQ_ROWS(rows_iq2_xs, block_iq2_xs, unpack_iq2_xs)
DEF_IQ_ROWS(rows_iq2_s, block_iq2_s, unpack_iq2_s)
DEF_IQ_ROWS(rows_iq3_xxs, block_iq3_xxs, unpack_iq3_xxs)
DEF_IQ_ROWS(rows_iq3_s, block_iq3_s, unpack_iq3_s)
DEF_IQ_ROWS(rows_iq4_xs, block_iq4_xs, unpack_iq4_xs)

/* 8 int16 values times f, rounded, as int32 in two halves packed back to int16 */
static inline __m128i q8_round8(__m128i v, __m128 f) {
    const __m128i a = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(v)), f));
    const __m128i b = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v, 8))), f));
    return _mm_packs_epi32(a, b);
}

/* an unpacked block (q * sc * d) as 8 Q8_0 blocks: the products fit in int16 (|q| <= 127, |sc| <= 32 for IQ4_XS) */
static inline void iq_block_q8_0(const __m256i q[8], const __m256i sc[8], float d, int8_t *qs, f16_t *dq) {
    for (int j = 0; j < 8; j++, qs += 32) {
        const __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q[j]));
        const __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q[j], 1));
        const __m256i plo = _mm256_mullo_epi16(lo, _mm256_set1_epi16((short)_mm256_extract_epi16(sc[j], 0)));
        const __m256i phi = _mm256_mullo_epi16(hi, _mm256_set1_epi16((short)_mm256_extract_epi16(sc[j], 8)));
        __m256i mx = _mm256_max_epi16(_mm256_abs_epi16(plo), _mm256_abs_epi16(phi));
        __m128i m = _mm_max_epi16(_mm256_castsi256_si128(mx), _mm256_extracti128_si256(mx, 1));
        m = _mm_max_epi16(m, _mm_srli_si128(m, 8));
        m = _mm_max_epi16(m, _mm_srli_si128(m, 4));
        m = _mm_max_epi16(m, _mm_srli_si128(m, 2));
        const int amax = _mm_extract_epi16(m, 0);
        if (!amax) {
            memset(qs, 0, 32);
            dq[j] = 0;
            continue;
        }
        const __m128 f = _mm_set1_ps(127.f / (float)amax);
        _mm_storeu_si128((__m128i *)qs, _mm_packs_epi16(q8_round8(_mm256_castsi256_si128(plo), f), q8_round8(_mm256_extracti128_si256(plo, 1), f)));
        _mm_storeu_si128((__m128i *)(qs + 16), _mm_packs_epi16(q8_round8(_mm256_castsi256_si128(phi), f), q8_round8(_mm256_extracti128_si256(phi, 1), f)));
        dq[j] = f2h(d * (float)amax / 127.f);
    }
}

static void (*row_to_q8_0_other)(uint32_t type, const void *row, int K, int8_t *qs, f16_t *d);

#define IQ_Q8(BT, UNPACK)                                                     \
    for (int i = 0; i < K / QK_K; i++) {                                     \
        __m256i q[8], sc[8];                                                 \
        float dd;                                                            \
        UNPACK((const BT *)row + i, q, sc, &dd);                             \
        iq_block_q8_0(q, sc, dd, qs + i * QK_K, d + i * 8);                  \
    }                                                                        \
    break;
static void row_to_q8_0(uint32_t type, const void *row, int K, int8_t *qs, f16_t *d) {
    switch (type) {
    case GGML_IQ2_XXS: IQ_Q8(block_iq2_xxs, unpack_iq2_xxs)
    case GGML_IQ2_XS: IQ_Q8(block_iq2_xs, unpack_iq2_xs)
    case GGML_IQ2_S: IQ_Q8(block_iq2_s, unpack_iq2_s)
    case GGML_IQ3_XXS: IQ_Q8(block_iq3_xxs, unpack_iq3_xxs)
    case GGML_IQ3_S: IQ_Q8(block_iq3_s, unpack_iq3_s)
    case GGML_IQ4_XS: IQ_Q8(block_iq4_xs, unpack_iq4_xs)
    default: row_to_q8_0_other(type, row, K, qs, d);
    }
}
#undef IQ_Q8

/* an unpacked IQ2 / IQ3_XXS block as nibbles: the magnitude's index ((|q| - 8) >> 4 of 8 25 43, or
 * (|q| - 4) >> 3 of 4 12 .. 62) in bits 0-2, the sign in bit 3; value 2k in the low nibble of byte k */
static inline void iq_block_nib(const __m256i q[8], const __m256i sc[8], int two, uint8_t *nib, uint8_t *scb) {
    const __m256i base = _mm256_set1_epi8(two ? 8 : 4), m7 = _mm256_set1_epi8(7), m8 = _mm256_set1_epi8(8);
    for (int j = 0; j < 8; j++) {
        const __m256i a = _mm256_sub_epi8(_mm256_abs_epi8(q[j]), base);
        __m256i c = _mm256_and_si256(two ? _mm256_srli_epi16(a, 4) : _mm256_srli_epi16(a, 3), m7);
        c = _mm256_or_si256(c, _mm256_and_si256(_mm256_cmpgt_epi8(_mm256_setzero_si256(), q[j]), m8));
        const __m256i pr = _mm256_maddubs_epi16(c, _mm256_set1_epi16(0x1001));   /* c[2k] + 16 c[2k+1] */
        const __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(pr, pr), 0x08);
        _mm_storeu_si128((__m128i *)(nib + 16 * j), _mm256_castsi256_si128(pk));
        scb[2 * j] = (uint8_t)_mm256_extract_epi16(sc[j], 0);
        scb[2 * j + 1] = (uint8_t)_mm256_extract_epi16(sc[j], 8);
    }
    if (two) scb[0] |= 0x80;
}

#define IQ_NIB(BT, UNPACK, TWO)                                                \
    for (int i = 0; i < K / QK_K; i++) {                                      \
        __m256i q[8], s[8];                                                   \
        UNPACK((const BT *)row + i, q, s, d + i);                             \
        iq_block_nib(q, s, TWO, nib + i * (QK_K / 2), sc + i * (QK_K / 16));  \
    }                                                                         \
    return 1;
static int row_to_iq4(uint32_t type, const void *row, int K, uint8_t *nib, uint8_t *sc, float *d) {
    switch (type) {
    case GGML_IQ2_XXS: IQ_NIB(block_iq2_xxs, unpack_iq2_xxs, 1)
    case GGML_IQ2_XS: IQ_NIB(block_iq2_xs, unpack_iq2_xs, 1)
    case GGML_IQ2_S: IQ_NIB(block_iq2_s, unpack_iq2_s, 1)
    case GGML_IQ3_XXS: IQ_NIB(block_iq3_xxs, unpack_iq3_xxs, 0)
    default: return 0;
    }
}
#undef IQ_NIB

/* IQ4_NL and Q5_0: 32-blocks with an f16 scale, like Q8_0 (the last Q8_K block may be partial) */
static inline __m256i unpack_iq4_nl(const block_iq4_nl *b) { return iq4_values(b->qs); }

static inline __m256i unpack_q5_0(const block_q5_0 *b) {
    uint32_t qh;
    memcpy(&qh, b->qh, 4);
    const __m128i v = _mm_loadu_si128((const __m128i *)b->qs);
    const __m256i lo = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(v, 4), v), _mm256_set1_epi8(0xF));
    const __m256i hi = _mm256_and_si256(iq_sign_mask(qh), _mm256_set1_epi8(0x10));
    return _mm256_sub_epi8(_mm256_or_si256(lo, hi), _mm256_set1_epi8(16));
}

#define DEF_32_ROWS(NAME, BT, UNPACK)                                                                     \
    static void NAME(int K, const void *w, const block_q8_K *x, int xs, int nt, float *out, int os) {    \
        const BT *b = (const BT *)w;                                                                      \
        for (int t = 0; t < nt; t++) {                                                                    \
            const block_q8_K *y = x + (size_t)t * xs;                                                     \
            __m256 acc = _mm256_setzero_ps();                                                             \
            for (int i = 0; i < (K + QK_K - 1) / QK_K; i++) {                                             \
                const int nj = K / QK8_0 - i * 8 < 8 ? K / QK8_0 - i * 8 : 8;                             \
                __m256 acc_b = _mm256_setzero_ps();                                                       \
                for (int j = 0; j < nj; j++) {                                                            \
                    const BT *bb = &b[i * 8 + j];                                                         \
                    const __m256i p = dot_i8_32(UNPACK(bb), _mm256_loadu_si256((const __m256i *)(y[i].qs + 32 * j))); \
                    acc_b = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bb->d)), _mm256_cvtepi32_ps(p), acc_b);    \
                }                                                                                         \
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d), acc_b, acc);                                \
            }                                                                                             \
            out[(size_t)t * os] = hsum_ps(acc);                                                           \
        }                                                                                                 \
    }
DEF_32_ROWS(rows_iq4_nl, block_iq4_nl, unpack_iq4_nl)
DEF_32_ROWS(rows_q5_0, block_q5_0, unpack_q5_0)

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
    if (k->row_to_q8_0 != row_to_q8_0) row_to_q8_0_other = k->row_to_q8_0;
    k->row_to_q8_0 = row_to_q8_0;
    k->row_to_iq4 = row_to_iq4;
    k->name = "avx2";
    k->dot[GGML_Q4_K] = rows_q4_K;
    k->dot[GGML_Q5_K] = rows_q5_K;
    k->dot[GGML_Q6_K] = rows_q6_K;
    k->dot[GGML_Q8_0] = rows_q8_0;
    k->dot[GGML_Q4_0] = rows_q4_0;
    k->dot[GGML_NVFP4] = rows_nvfp4;
    k->dot[GGML_Q5_0] = rows_q5_0;
    k->dot[GGML_IQ4_NL] = rows_iq4_nl;
    k->dot[GGML_IQ4_XS] = rows_iq4_xs;
    k->dot[GGML_IQ2_XXS] = rows_iq2_xxs;
    k->dot[GGML_IQ2_XS] = rows_iq2_xs;
    k->dot[GGML_IQ2_S] = rows_iq2_s;
    k->dot[GGML_IQ3_XXS] = rows_iq3_xxs;
    k->dot[GGML_IQ3_S] = rows_iq3_s;
    k->quantize_q8_K = quantize_q8_K;
    k->f16_to_f32_row = f16_to_f32_row;
    k->f32_to_f16_row = f32_to_f16_row;
    k->dot_f32 = dot_f32;
    k->dot_f16_f32 = dot_f16_f32;
    k->axpy_f16 = axpy_f16;
}
#endif
