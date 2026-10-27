/* kernels.cl - OpenCL kernels: MoE experts cached in VRAM, and the dense decode path.
 * Embedded into the binary as src/kernels_cl.h (tools/embed.py); compiled by the driver.
 *
 * Two paths:
 *  - USE_DP4A (NVIDIA sm_61+, via inline PTX): activations quantized to int8 in 32-blocks,
 *    weight nibbles/6-bit values multiplied with dp4a (4 MACs per instruction).
 *  - default: f32 activations in local memory (any OpenCL 1.2 GPU).
 *
 * Work layout: a 32-lane group computes RPW rows; 8 lanes share a super-block and each lane
 * takes 16 consecutive weight bytes, so a group consumes 4 super-blocks per iteration. */

#define WG 256
#ifndef RPW
#define RPW 4
#endif

inline float h2f(__global const uchar *p) { return vload_half(0, (__global const half *)p); }

inline void scale_min_k4(int j, __global const uchar *q, float *sc, float *mn) {
    if (j < 4) { *sc = (float)(q[j] & 63); *mn = (float)(q[j + 4] & 63); }
    else { *sc = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *mn = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

inline ulong row_bytes(int type, int K) { return (ulong)(K / 256) * (type == 12 ? 144 : 210); }

inline __global const uchar *slot_ptr(__global const uchar *A0, __global const uchar *A1, __global const uchar *A2,
                                      __global const uchar *A3, int s, int spa, ulong slot_bytes) {
    const int a = s / spa;
    __global const uchar *base = a == 0 ? A0 : (a == 1 ? A1 : (a == 2 ? A2 : A3));
    return base + (ulong)(s - a * spa) * slot_bytes;
}

inline void reduce_rows(__local float *red, int lid, int lane, int grp, float *acc, __global float *out, int row0, int nrows,
                        int accumulate) {
    for (int r = 0; r < RPW; r++) red[lid * RPW + r] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < RPW) {
        float s = 0.f;
        for (int j = 0; j < 32; j++) s += red[(grp * 32 + j) * RPW + lane];
        const int row = row0 + lane * (WG / 32);
        if (row < nrows) out[row] = accumulate ? out[row] + s : s;
    }
}

/* orders this work-item's global writes for the other work-groups (OpenCL 1.2 has no device
 * scope; NVIDIA needs membar.gl, which mem_fence does not emit) */
#ifdef USE_DP4A
#define DEVICE_FENCE() asm volatile("membar.gl;" ::: "memory")
#else
#define DEVICE_FENCE() mem_fence(CLK_GLOBAL_MEM_FENCE)
#endif

inline float wg_sum(__local float *red, float v) {
    const int lid = get_local_id(0), n = get_local_size(0);
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

inline float wg_max(__local float *red, float v) {
    const int lid = get_local_id(0), n = get_local_size(0);
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] = fmax(red[lid], red[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

/* Dense matvec, up to three row segments (e.g. q|k|v) sharing the input x; every work-group
 * belongs to one segment. has_norm: x is RMS-normalized with the weights NA + nw_off first. */
#define MV_ARGS __global const uchar *W0, __global const uchar *W1, __global const uchar *W2, ulong w_off0, ulong w_off1, \
    ulong w_off2, int4 wtype, int4 rows, int4 y_off, int K, int k0, int klen, __global const float *x,                    \
    __global const uchar *NA, int has_norm, ulong nw_off, float eps, __global float *y, int accumulate
#define MV_PROLOGUE                                                                            \
    const int lid = get_local_id(0), lane = lid & 31, grp = lid >> 5;                          \
    const int4 nwg = (rows + 8 * RPW - 1) / (8 * RPW);                                         \
    int gid = get_group_id(0), seg = 0;                                                        \
    if (gid >= nwg.x) { gid -= nwg.x; seg = 1; if (gid >= nwg.y) { gid -= nwg.y; seg = 2; } }  \
    __global const uchar *W = seg == 0 ? W0 + w_off0 : (seg == 1 ? W1 + w_off1 : W2 + w_off2); \
    const int wt = seg == 0 ? wtype.x : (seg == 1 ? wtype.y : wtype.z);                        \
    const int nrows = seg == 0 ? rows.x : (seg == 1 ? rows.y : rows.z);                        \
    const int yo = seg == 0 ? y_off.x : (seg == 1 ? y_off.y : y_off.z);                        \
    __global const float *nw = (__global const float *)(NA + nw_off);                          \
    float ns = 1.f;                                                                            \
    if (has_norm) {                                                                            \
        float ss = 0.f;                                                                        \
        for (int i = lid; i < K; i += WG) ss += x[i] * x[i];                                   \
        ns = rsqrt(wg_sum(red, ss) / K + eps);                                                 \
    }                                                                                          \
    const ulong rb = row_bytes(wt, K), skip = row_bytes(wt, k0);                               \
    const int row0 = gid * (WG / 32) * RPW + grp;
#define XIN(i) (has_norm ? x[k0 + (i)] * ns * nw[k0 + (i)] : x[k0 + (i)])

#ifdef USE_DP4A
/* ======================================================================= int8 / dp4a path */

inline int dp4a(uint a, int b, int c) {
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

/* int8 activations in local memory: 16-byte units are permuted inside every 128-byte block so
 * the 8 lanes of a quarter-warp read distinct banks (both Q4_K and Q6_K layouts). */
inline int swz8(int b) { return ((((b >> 4) ^ (((b >> 7) & 1) * 6))) << 4) | (b & 15); }
#define XQ(off) vload4(0, (__local const int *)(xq + swz8(off)))

inline float dot_q4k(__global const uchar *row, __local const char *xq, __local const float *xd, int nb, int lane) {
    float acc = 0.f;
    const int sbl = lane >> 3, p8 = lane & 7, c = p8 >> 1, o = (p8 & 1) * 16;
    for (int i = sbl; i < nb; i += 4) {
        __global const uchar *b = row + i * 144;
        const float d = h2f(b), dmin = h2f(b + 2);
        float s0, m0, s1, m1;
        scale_min_k4(2 * c, b + 4, &s0, &m0);
        scale_min_k4(2 * c + 1, b + 4, &s1, &m1);
        const uint4 q = vload4(0, (__global const uint *)(b + 16 + 32 * c + o));
        const int f = i * 256 + 64 * c + o;
        const int4 xl = XQ(f), xh = XQ(f + 32);
        int sl = dp4a(q.x & 0x0F0F0F0Fu, xl.x, 0);
        sl = dp4a(q.y & 0x0F0F0F0Fu, xl.y, sl);
        sl = dp4a(q.z & 0x0F0F0F0Fu, xl.z, sl);
        sl = dp4a(q.w & 0x0F0F0F0Fu, xl.w, sl);
        int sh = dp4a((q.x >> 4) & 0x0F0F0F0Fu, xh.x, 0);
        sh = dp4a((q.y >> 4) & 0x0F0F0F0Fu, xh.y, sh);
        sh = dp4a((q.z >> 4) & 0x0F0F0F0Fu, xh.z, sh);
        sh = dp4a((q.w >> 4) & 0x0F0F0F0Fu, xh.w, sh);
        int ml = dp4a(0x01010101u, xl.x, 0);
        ml = dp4a(0x01010101u, xl.y, ml);
        ml = dp4a(0x01010101u, xl.z, ml);
        ml = dp4a(0x01010101u, xl.w, ml);
        int mh = dp4a(0x01010101u, xh.x, 0);
        mh = dp4a(0x01010101u, xh.y, mh);
        mh = dp4a(0x01010101u, xh.z, mh);
        mh = dp4a(0x01010101u, xh.w, mh);
        const float dl = xd[i * 8 + 2 * c], dh = xd[i * 8 + 2 * c + 1];
        acc += d * (s0 * dl * (float)sl + s1 * dh * (float)sh) - dmin * (m0 * dl * (float)ml + m1 * dh * (float)mh);
    }
    return acc;
}

inline uint u32_of(ushort8 v, int k) {
    switch (k) {
    case 0: return (uint)v.s0 | ((uint)v.s1 << 16);
    case 1: return (uint)v.s2 | ((uint)v.s3 << 16);
    case 2: return (uint)v.s4 | ((uint)v.s5 << 16);
    default: return (uint)v.s6 | ((uint)v.s7 << 16);
    }
}

inline float dot_q6k(__global const uchar *row, __local const char *xq, __local const float *xd, int nb, int lane) {
    float acc = 0.f;
    const int sbl = lane >> 3, p8 = lane & 7, n = p8 >> 2, j = (p8 & 3) * 16, hi = j >> 5, jj = j & 31;
    const uint sh = 2u * (uint)hi;
    const int sidx = 8 * n + (jj >> 4) + 2 * hi;
    for (int i = sbl; i < nb; i += 4) {
        __global const uchar *b = row + i * 210;
        const float d = h2f(b + 208);
        __global const char *sc = (__global const char *)(b + 192);
        const ushort8 qlv = vload8(0, (__global const ushort *)(b + 64 * n + j));
        const ushort8 qhv = vload8(0, (__global const ushort *)(b + 128 + 32 * n + jj));
        const int f = i * 256 + 128 * n + 32 * hi + jj;
        const int4 xa = XQ(f), xc = XQ(f + 64);
        int sl = 0, su = 0, ma = 0, mc = 0;
        for (int k = 0; k < 4; k++) {
            const uint l = u32_of(qlv, k), h = u32_of(qhv, k);
            const uint vlo = (l & 0x0F0F0F0Fu) | (((h >> sh) & 0x03030303u) << 4);
            const uint vhi = ((l >> 4) & 0x0F0F0F0Fu) | (((h >> (sh + 4u)) & 0x03030303u) << 4);
            const int xa_k = k == 0 ? xa.x : (k == 1 ? xa.y : (k == 2 ? xa.z : xa.w));
            const int xc_k = k == 0 ? xc.x : (k == 1 ? xc.y : (k == 2 ? xc.z : xc.w));
            sl = dp4a(vlo, xa_k, sl);
            su = dp4a(vhi, xc_k, su);
            ma = dp4a(0x01010101u, xa_k, ma);
            mc = dp4a(0x01010101u, xc_k, mc);
        }
        const float da = xd[f >> 5], dc = xd[(f + 64) >> 5];
        acc += d * ((float)sc[sidx] * da * (float)(sl - 32 * ma) + (float)sc[sidx + 4] * dc * (float)(su - 32 * mc));
    }
    return acc;
}

inline float dot_row(int type, __global const uchar *row, __local const char *xq, __local const float *xd, int K, int lane) {
    return type == 12 ? dot_q4k(row, xq, xd, K / 256, lane) : dot_q6k(row, xq, xd, K / 256, lane);
}

/* Inputs of one launch live in a single buffer ("meta"): slots[max_assign], xidx[max_assign],
 * then the f32 hidden states of the batch. One host->device copy per layer. */
#define META_SLOTS(meta) (meta)
#define META_XIDX(meta, ma) ((meta) + (ma))

/* quantize n floats (computed by f) into int8 32-blocks in local memory; one thread per block */
/* every thread takes 8 consecutive values; 4 neighbouring threads share a 32-block and combine
 * their maxima through `red` (free until reduce_rows) */
#define QUANT_LOCAL(N, EXPR)                                                        \
    for (int base = 0; base < (N); base += WG * 8) {                                \
        const int e0 = base + lid * 8;                                              \
        float v[8], am = 0.f;                                                       \
        if (e0 < (N))                                                               \
            for (int t = 0; t < 8; t++) { const int i = e0 + t; v[t] = (EXPR); am = fmax(am, fabs(v[t])); } \
        red[lid] = am;                                                              \
        barrier(CLK_LOCAL_MEM_FENCE);                                               \
        const int q0 = lid & ~3;                                                    \
        const float amax = fmax(fmax(red[q0], red[q0 + 1]), fmax(red[q0 + 2], red[q0 + 3])); \
        if (e0 < (N)) {                                                             \
            const float id = amax > 0.f ? 127.f / amax : 0.f;                       \
            for (int t = 0; t < 8; t++) xq[swz8(e0 + t)] = (char)convert_int_rte(v[t] * id); \
            if ((lid & 3) == 0) xd[e0 >> 5] = amax / 127.f;                        \
        }                                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                               \
    }

/* gate and up projections for every (expert, token) assignment: grid (ceil(F / (8*RPW)) * WG, 2 * n_assign) */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void moe_gate_up(__global const uchar *A0, __global const uchar *A1, __global const uchar *A2, __global const uchar *A3,
                 int spa, ulong slot_bytes, __global const int *meta, int max_assign, int K, int F, ulong up_off,
                 int wtype, __global float *gu, __local int *xqi, __local float *xd, __local float *red,
                 __global const float *X, ulong x_off) {
    __local char *xq = (__local char *)xqi;
    const int lid = get_local_id(0), lane = lid & 31, grp = lid >> 5;
    const int ek = get_group_id(1), a = ek >> 1, which = ek & 1;
    if (META_SLOTS(meta)[a] < 0) return; /* expert computed by the CPU */
    __global const float *x = X + x_off + (ulong)META_XIDX(meta, max_assign)[a] * K;
    QUANT_LOCAL(K, x[i])
    __global const uchar *base = slot_ptr(A0, A1, A2, A3, META_SLOTS(meta)[a], spa, slot_bytes) + (which ? up_off : 0);
    const ulong rb = row_bytes(wtype, K);
    const int row0 = get_group_id(0) * (WG / 32) * RPW + grp;
    float acc[RPW];
    for (int r = 0; r < RPW; r++) {
        const int row = row0 + r * (WG / 32);
        acc[r] = row < F ? dot_row(wtype, base + (ulong)row * rb, xq, xd, K, lane) : 0.f;
    }
    reduce_rows(red, lid, lane, grp, acc, gu + ((ulong)a * 2 + which) * F, row0, F, 0);
}

/* SiLU(gate)*up quantized in local memory, then the down projection: grid (ceil(D / (8*RPW)) * WG, n_assign) */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void moe_down(__global const uchar *A0, __global const uchar *A1, __global const uchar *A2, __global const uchar *A3,
              int spa, ulong slot_bytes, __global const int *meta, int F, int D, ulong down_off, int wtype,
              __global const float *gu, __global float *Y, __local int *xqi, __local float *xd, __local float *red) {
    __local char *xq = (__local char *)xqi;
    const int lid = get_local_id(0), lane = lid & 31, grp = lid >> 5;
    const int a = get_group_id(1);
    if (META_SLOTS(meta)[a] < 0) return;
    __global const float *g = gu + (ulong)a * 2 * F;
    QUANT_LOCAL(F, g[i] / (1.f + exp(-g[i])) * g[F + i])
    __global const uchar *base = slot_ptr(A0, A1, A2, A3, META_SLOTS(meta)[a], spa, slot_bytes) + down_off;
    const ulong rb = row_bytes(wtype, F);
    const int row0 = get_group_id(0) * (WG / 32) * RPW + grp;
    float acc[RPW];
    for (int r = 0; r < RPW; r++) {
        const int row = row0 + r * (WG / 32);
        acc[r] = row < D ? dot_row(wtype, base + (ulong)row * rb, xq, xd, F, lane) : 0.f;
    }
    reduce_rows(red, lid, lane, grp, acc, Y + (ulong)a * D, row0, D, 0);
}

/* y[row] (+)= W[row] . x over the K-range [k0, k0 + klen): dense layers kept in VRAM.
 * grid: sum over the segments of ceil(rows / (8*RPW)) * WG */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void matvec(MV_ARGS, __local int *xqi, __local float *xd, __local float *red) {
    __local char *xq = (__local char *)xqi;
    MV_PROLOGUE
    QUANT_LOCAL(klen, XIN(i))
    float acc[RPW];
    for (int r = 0; r < RPW; r++) {
        const int row = row0 + r * (WG / 32);
        acc[r] = row < nrows ? dot_row(wt, W + (ulong)row * rb + skip, xq, xd, klen, lane) : 0.f;
    }
    reduce_rows(red, lid, lane, grp, acc, y + yo, row0, nrows, accumulate);
}

#else
/* ======================================================================= f32 path (portable) */

/* Local-memory swizzle: permutes the eight float4 groups inside every 32-float block so the
 * 8 lanes of a quarter-warp hit distinct banks for both the Q4_K and the Q6_K lane layouts. */
inline int swz(int f) {
    const int b = f >> 5;
    const int key = (((b >> 2) & 1) << 1) | ((b ^ (b >> 1)) & 1);
    return (f & ~31) | ((((f >> 2) & 7) ^ key) << 2) | (f & 3);
}
#define NIB4(w, xv) dot((float4)((float)((w) & 15u), (float)(((w) >> 8) & 15u), (float)(((w) >> 16) & 15u), (float)(((w) >> 24) & 15u)), xv)
#define NIB4H(w, xv) dot((float4)((float)(((w) >> 4) & 15u), (float)(((w) >> 12) & 15u), (float)(((w) >> 20) & 15u), (float)((w) >> 28)), xv)
#define HSUM4(v) ((v).x + (v).y + (v).z + (v).w)
#define XLD(f) vload4(0, x + swz(f))

inline float dot_q4k(__global const uchar *row, __local const float *x, int nb, int lane) {
    float acc = 0.f;
    const int sbl = lane >> 3, p8 = lane & 7, c = p8 >> 1, o = (p8 & 1) * 16;
    for (int i = sbl; i < nb; i += 4) {
        __global const uchar *b = row + i * 144;
        const float d = h2f(b), dmin = h2f(b + 2);
        float s0, m0, s1, m1;
        scale_min_k4(2 * c, b + 4, &s0, &m0);
        scale_min_k4(2 * c + 1, b + 4, &s1, &m1);
        const uint4 q = vload4(0, (__global const uint *)(b + 16 + 32 * c + o));
        const int f = i * 256 + 64 * c + o;
        const float4 l0 = XLD(f), l1 = XLD(f + 4), l2 = XLD(f + 8), l3 = XLD(f + 12);
        const float4 h0 = XLD(f + 32), h1 = XLD(f + 36), h2 = XLD(f + 40), h3 = XLD(f + 44);
        const float sl = NIB4(q.x, l0) + NIB4(q.y, l1) + NIB4(q.z, l2) + NIB4(q.w, l3);
        const float sh = NIB4H(q.x, h0) + NIB4H(q.y, h1) + NIB4H(q.z, h2) + NIB4H(q.w, h3);
        const float ml = HSUM4(l0 + l1 + l2 + l3), mh = HSUM4(h0 + h1 + h2 + h3);
        acc += d * (s0 * sl + s1 * sh) - dmin * (m0 * ml + m1 * mh);
    }
    return acc;
}

#define Q6V(l, h, s) ((float)((int)(((l) & 15u) | ((((h) >> (s)) & 3u) << 4)) - 32))
#define Q6H(l, h, s) ((float)((int)(((l) >> 4) | ((((h) >> ((s) + 4u)) & 3u) << 4)) - 32))
#define Q6QUAD(k, m0, m1) { const uint la = qlv.s##m0, lb = qlv.s##m1, ha = qhv.s##m0, hb = qhv.s##m1; \
    const float4 xa = XLD(f + 4 * k), xc = XLD(f + 64 + 4 * k); \
    const float4 vl = (float4)(Q6V(la & 255u, ha & 255u, sh), Q6V(la >> 8, ha >> 8, sh), Q6V(lb & 255u, hb & 255u, sh), Q6V(lb >> 8, hb >> 8, sh)); \
    const float4 vh = (float4)(Q6H(la & 255u, ha & 255u, sh), Q6H(la >> 8, ha >> 8, sh), Q6H(lb & 255u, hb & 255u, sh), Q6H(lb >> 8, hb >> 8, sh)); \
    sl += dot(vl, xa); su += dot(vh, xc); }

inline float dot_q6k(__global const uchar *row, __local const float *x, int nb, int lane) {
    float acc = 0.f;
    const int sbl = lane >> 3, p8 = lane & 7, n = p8 >> 2, j = (p8 & 3) * 16, hi = j >> 5, jj = j & 31;
    const uint sh = 2u * (uint)hi;
    const int sidx = 8 * n + (jj >> 4) + 2 * hi;
    for (int i = sbl; i < nb; i += 4) {
        __global const uchar *b = row + i * 210;
        const float d = h2f(b + 208);
        __global const char *sc = (__global const char *)(b + 192);
        const float slo = d * (float)sc[sidx], shi = d * (float)sc[sidx + 4];
        const ushort8 qlv = vload8(0, (__global const ushort *)(b + 64 * n + j));
        const ushort8 qhv = vload8(0, (__global const ushort *)(b + 128 + 32 * n + jj));
        const int f = i * 256 + 128 * n + 32 * hi + jj;
        float sl = 0.f, su = 0.f;
        Q6QUAD(0, 0, 1) Q6QUAD(1, 2, 3) Q6QUAD(2, 4, 5) Q6QUAD(3, 6, 7)
        acc += slo * sl + shi * su;
    }
    return acc;
}

inline float dot_row(int type, __global const uchar *row, __local const float *x, int K, int lane) {
    return type == 12 ? dot_q4k(row, x, K / 256, lane) : dot_q6k(row, x, K / 256, lane);
}

#define META_SLOTS(meta) (meta)
#define META_XIDX(meta, ma) ((meta) + (ma))

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void moe_gate_up(__global const uchar *A0, __global const uchar *A1, __global const uchar *A2, __global const uchar *A3,
                 int spa, ulong slot_bytes, __global const int *meta, int max_assign, int K, int F, ulong up_off,
                 int wtype, __global float *gu, __local float *xl, __local float *red,
                 __global const float *X, ulong x_off) {
    const int lid = get_local_id(0), lane = lid & 31, grp = lid >> 5;
    const int ek = get_group_id(1), a = ek >> 1, which = ek & 1;
    if (META_SLOTS(meta)[a] < 0) return;
    __global const float *x = X + x_off + (ulong)META_XIDX(meta, max_assign)[a] * K;
    for (int i = lid; i < K; i += WG) xl[swz(i)] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const uchar *base = slot_ptr(A0, A1, A2, A3, META_SLOTS(meta)[a], spa, slot_bytes) + (which ? up_off : 0);
    const ulong rb = row_bytes(wtype, K);
    const int row0 = get_group_id(0) * (WG / 32) * RPW + grp;
    float acc[RPW];
    for (int r = 0; r < RPW; r++) {
        const int row = row0 + r * (WG / 32);
        acc[r] = row < F ? dot_row(wtype, base + (ulong)row * rb, xl, K, lane) : 0.f;
    }
    reduce_rows(red, lid, lane, grp, acc, gu + ((ulong)a * 2 + which) * F, row0, F, 0);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void moe_down(__global const uchar *A0, __global const uchar *A1, __global const uchar *A2, __global const uchar *A3,
              int spa, ulong slot_bytes, __global const int *meta, int F, int D, ulong down_off, int wtype,
              __global const float *gu, __global float *Y, __local float *al, __local float *red) {
    const int lid = get_local_id(0), lane = lid & 31, grp = lid >> 5;
    const int a = get_group_id(1);
    if (META_SLOTS(meta)[a] < 0) return;
    __global const float *g = gu + (ulong)a * 2 * F;
    for (int i = lid; i < F; i += WG) {
        const float gv = g[i];
        al[swz(i)] = gv / (1.f + exp(-gv)) * g[F + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const uchar *base = slot_ptr(A0, A1, A2, A3, META_SLOTS(meta)[a], spa, slot_bytes) + down_off;
    const ulong rb = row_bytes(wtype, F);
    const int row0 = get_group_id(0) * (WG / 32) * RPW + grp;
    float acc[RPW];
    for (int r = 0; r < RPW; r++) {
        const int row = row0 + r * (WG / 32);
        acc[r] = row < D ? dot_row(wtype, base + (ulong)row * rb, al, F, lane) : 0.f;
    }
    reduce_rows(red, lid, lane, grp, acc, Y + (ulong)a * D, row0, D, 0);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void matvec(MV_ARGS, __local float *xl, __local float *red) {
    MV_PROLOGUE
    for (int i = lid; i < klen; i += WG) xl[swz(i)] = XIN(i);
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc[RPW];
    for (int r = 0; r < RPW; r++) {
        const int row = row0 + r * (WG / 32);
        acc[r] = row < nrows ? dot_row(wt, W + (ulong)row * rb + skip, xl, klen, lane) : 0.f;
    }
    reduce_rows(red, lid, lane, grp, acc, y + yo, row0, nrows, accumulate);
}
#endif

/* ======================================================================= dense decode path (both builds)
 * One token per launch. Weights live in the dense arena (A + byte offset); the residual stream x,
 * the KV cache and the convolution state stay in VRAM between tokens. */

/* short convolution, one thread per channel: bcx = [B | C | x] from in_proj */
__kernel void shortconv(__global const float *bcx, __global float *state, ulong st_off, __global const uchar *A,
                        ulong cw_off, __global float *y, int D, int lc) {
    const int ch = get_global_id(0);
    if (ch >= D) return;
    __global float *st = state + st_off;
    __global const float *w = (__global const float *)(A + cw_off) + (ulong)ch * lc;
    const float bx = bcx[ch] * bcx[2 * D + ch];
    float acc = w[lc - 1] * bx;
    for (int k = 0; k < lc - 1; k++) acc += w[k] * st[(ulong)k * D + ch];
    for (int k = 0; k < lc - 2; k++) st[(ulong)k * D + ch] = st[(ulong)(k + 1) * D + ch];
    st[(ulong)(lc - 2) * D + ch] = bx;
    y[ch] = bcx[D + ch] * acc;
}

/* g = silu(g) * u, with u = g + u_off */
__kernel void silu_mul(__global float *g, int n, int u_off) {
    const int i = get_global_id(0);
    if (i < n) { const float v = g[i]; g[i] = v / (1.f + exp(-v)) * g[u_off + i]; }
}

/* Per-head RMSNorm + NEOX RoPE of q and k (cos/sin of this position from the host), then the
 * attention of one query head over positions [0, pos]. One work-group per query head: each one
 * prepares the k/v of the current position in local memory, and the first head of every kv
 * group appends them to the f16 KV cache (the other heads only read rows < pos from it). */
inline float rope_rot(__local const float *t, __global const float *rope, int i, int hh) {
    const int f = i < hh ? i : i - hh;
    const float cs = rope[f], sn = rope[hh + f];
    return i < hh ? t[i] * cs - t[i + hh] * sn : t[i - hh] * sn + t[i] * cs;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attn_decode(__global const float *qkv, __global const uchar *A, ulong qn_off, ulong kn_off,
                 __global const float *rope /* cos[hd/2] | sin[hd/2] */, __global half *kc, __global half *vc, ulong lbase,
                 int pos, int n_head, int n_kv, int hd, int D, float eps, __global float *scores, int n_ctx,
                 __global float *out, float scale) {
    __local float q[256], kn[256], vn[256], t[256], red[WG];
    const int h = get_group_id(0), lid = get_local_id(0), kvd = n_kv * hd, hh = hd / 2;
    const int grp = n_head / n_kv, kvh = h / grp, act = lid < hd;
    __global const float *qn = (__global const float *)(A + qn_off), *knw = (__global const float *)(A + kn_off);
    float v = act ? qkv[h * hd + lid] : 0.f;
    float s = rsqrt(wg_sum(red, v * v) / hd + eps);
    t[lid] = act ? v * s * qn[lid] : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (act) q[lid] = rope_rot(t, rope, lid, hh) * scale;
    v = act ? qkv[D + kvh * hd + lid] : 0.f;
    s = rsqrt(wg_sum(red, v * v) / hd + eps); /* its barriers also order the reads of t above */
    t[lid] = act ? v * s * knw[lid] : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (act) {
        kn[lid] = rope_rot(t, rope, lid, hh);
        vn[lid] = qkv[D + kvd + kvh * hd + lid];
        if (h % grp == 0) {
            vstore_half(kn[lid], 0, kc + lbase + (ulong)pos * kvd + kvh * hd + lid);
            vstore_half(vn[lid], 0, vc + lbase + (ulong)pos * kvd + kvh * hd + lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int npos = pos + 1;
    __global float *sc = scores + (ulong)h * n_ctx;
    float mx = -INFINITY;
    for (int j = lid; j < npos; j += WG) {
        float d = 0.f;
        if (j < pos) {
            __global const half *k = kc + lbase + (ulong)j * kvd + kvh * hd;
            for (int u = 0; u < hd; u += 4) d += dot(vload_half4(0, k + u), (float4)(q[u], q[u + 1], q[u + 2], q[u + 3]));
        } else {
            for (int u = 0; u < hd; u++) d += kn[u] * q[u];
        }
        sc[j] = d;
        mx = fmax(mx, d);
    }
    mx = wg_max(red, mx);
    float sum = 0.f;
    for (int j = lid; j < npos; j += WG) { const float e = exp(sc[j] - mx); sc[j] = e; sum += e; }
    sum = wg_sum(red, sum);
    barrier(CLK_GLOBAL_MEM_FENCE);
    /* weighted sum of V: WG/hd groups of hd threads take every (WG/hd)-th position */
    const int parts = WG / hd, d = lid % hd, part = lid / hd;
    float acc = 0.f;
    if (part < parts)
        for (int j = part; j < npos; j += parts)
            acc += sc[j] * (j < pos ? vload_half(0, vc + lbase + (ulong)j * kvd + kvh * hd + d) : vn[d]);
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (part == 0) {
        float o = 0.f;
        for (int k = 0; k < parts; k++) o += red[k * hd + d];
        out[h * hd + d] = o / sum;
    }
}

/* top-k of the router logits with normalized weights; resident experts get their VRAM slot */
inline void route_pick(__global int *logits, __global const uchar *A, ulong bias_off, int has_bias, int D, int E, int K,
                       int gating, __global const int *res, int res_off, __global int *meta, __global float *out) {
    float p[64], sc[64];
    __global const float *bias = (__global const float *)(A + bias_off);
    for (int j = 0; j < E; j++) p[j] = as_float(atomic_or(logits + j, 0));
    if (gating == 1) {
        float mx = -INFINITY, sum = 0.f;
        for (int j = 0; j < E; j++) mx = fmax(mx, p[j]);
        for (int j = 0; j < E; j++) { p[j] = exp(p[j] - mx); sum += p[j]; }
        for (int j = 0; j < E; j++) p[j] /= sum;
    } else {
        for (int j = 0; j < E; j++) p[j] = 1.f / (1.f + exp(-p[j]));
    }
    for (int j = 0; j < E; j++) sc[j] = p[j] + (has_bias ? bias[j] : 0.f);
    int sel[8];
    float w[8], wsum = 0.f;
    for (int k = 0; k < K; k++) {
        int best = -1;
        for (int j = 0; j < E; j++) {
            int used = 0;
            for (int u = 0; u < k; u++) used |= sel[u] == j;
            if (!used && (best < 0 || sc[j] > sc[best])) best = j;
        }
        sel[k] = best;
        w[k] = p[best];
        wsum += p[best];
    }
    __global int *ret = (__global int *)(out + 2 * D);
    for (int k = 0; k < K; k++) {
        const int slot = res[res_off + sel[k]];
        ret[k] = sel[k];
        ret[K + k] = slot;
        ret[2 * K + k] = as_int(wsum > 0.f ? w[k] / wsum : w[k]);
        meta[k] = slot;
        meta[K + k] = 0;
    }
}

/* MoE router of one token: h = rmsnorm(x), one work-group per expert logit; the last work-group
 * to finish (atomic counter) picks the experts.
 * out = x[D] | h[D] | ret with ret = sel[K] | slot[K] | weight bits[K], the single readback of
 * the host; meta = slots[K] | xidx[K] for the expert kernels. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void moe_route(__global const float *x, __global const uchar *A, ulong nw_off, ulong rw_off, ulong bias_off, int has_bias,
               int D, float eps, int E, int K, int gating, __global const int *res, int res_off,
               __global int *logits, __global int *count, __global int *meta, __global float *out) {
    __local float red[WG];
    __local int last;
    const int e = get_group_id(0), lid = get_local_id(0);
    __global const float *nw = (__global const float *)(A + nw_off);
    __global const float *rw = (__global const float *)(A + rw_off) + (ulong)e * D;
    float ss = 0.f;
    for (int i = lid; i < D; i += WG) ss += x[i] * x[i];
    const float s = rsqrt(wg_sum(red, ss) / D + eps);
    float acc = 0.f;
    for (int i = lid; i < D; i += WG) {
        const float h = x[i] * s * nw[i];
        acc += rw[i] * h;
        if (e == 0) { out[i] = x[i]; out[D + i] = h; }
    }
    acc = wg_sum(red, acc);
    if (lid == 0) {
        atomic_xchg(logits + e, as_int(acc));
        DEVICE_FENCE();
        last = atomic_inc(count) == E - 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!last || lid != 0) return;
    atomic_xchg(count, 0);
    route_pick(logits, A, bias_off, has_bias, D, E, K, gating, res, res_off, meta, out);
}

/* x += weighted GPU expert outputs + the CPU partial result */
__kernel void moe_combine(__global float *x, __global const float *Y, __global const int *rbuf, int ret_off,
                          __global const float *ycpu, int use_cpu, int D, int K) {
    const int i = get_global_id(0);
    __global const int *ret = rbuf + ret_off;
    if (i >= D) return;
    float v = use_cpu ? ycpu[i] : 0.f;
    for (int k = 0; k < K; k++)
        if (ret[K + k] >= 0) v += as_float(ret[2 * K + k]) * Y[(ulong)k * D + i];
    x[i] += v;
}

