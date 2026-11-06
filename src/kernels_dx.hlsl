// kernels_dx.hlsl - Direct3D 12 compute kernels for gpu_d3d12.c (matmuls).
//
// out[t][r] = sum_k W[r][k] * x[t][k]. Weights stay quantized in GPU memory (Q4_K as in the
// GGUF file, Q6_K repacked to 4-byte aligned 212-byte blocks, see gpu_d3d12.c); activations
// are the Q8_K rows the CPU path uses.
// mm_*: batches. A group of 256 threads computes a tile of 64 rows x 64 tokens: each step
//       dequantizes a 32-wide slice of K into group-shared memory, then every thread
//       accumulates 4 rows x 4 tokens.
// mv_*: a few tokens (decoding). A group computes 8 rows of one token; the 32 threads of a
//       row split it into 64-value pieces, take integer dot products like the CPU kernels,
//       and add up their sums in group-shared memory.
//
// Regenerate kernels_dx.h after editing:  tools\dx_shaders.cmd  (and kernels_vk.h: tools\vk_shaders.cmd)
//
// Vulkan (PITHOS_VK) compiles the same code with its own, 16-byte aligned GPU layouts of the Q8_K
// and Q6_K blocks (the defines below), and the matrix-vector kernels read them through uint4
// views of the same buffers: SPIR-V has no 16-byte load from a byte-address buffer, and the four
// 4-byte loads it gets instead are much slower on some GPUs (Intel).
//
// Intel GPUs get their own build of the kernels (tools\vk_shaders.cmd): mv_* with 8 lanes per row
// in groups of 128 threads (MV_LANES, MV_GROUP) and, for Q4_K, the products taken two pairs per
// 32-bit multiply (MV_SWAR): their integer units, not the memory, set the decoding speed there;
// mm_* with 8 x 4 outputs per thread (MM_INTEL, Vulkan only).

ByteAddressBuffer W : register(t0);     // weight rows, wrow bytes apart
RWByteAddressBuffer X : register(u1);   // Q8_K activation rows, xrow bytes apart (the decode pass's scratch)
RWByteAddressBuffer O : register(u0);   // out[t * orow + r], floats

cbuffer P : register(b0) {
    uint rows, nb, nt, wrow, xrow, orow;    // nb: 256-element blocks per row
};

#ifndef Q8_BYTES
#define Q8_BYTES 292                // Q8_K block: float d at Q8_D, 256 int8 at Q8_QS
#define Q8_D 0
#define Q8_QS 4
#endif
#define Q4K_BYTES 144
#ifndef Q6K_BYTES
#define Q6K_BYTES 212               // Q6_K block: half d at Q6_D, ql[128], qh[64], int8 scales[16]
#define Q6_D 0
#define Q6_QL 4
#define Q6_QH 132
#define Q6_SC 196
#endif

#ifdef PITHOS_VK
[[vk::binding(8, 0)]] StructuredBuffer<uint4> W4 : register(t4);     // W and X as 16-byte words
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> X4 : register(u4);
#endif

groupshared float4 Ws[32 * 16];     // [k][row / 4]
groupshared float4 Xs[32 * 16];     // [k][token / 4]

uint byte_of(uint3 v, uint n) {
    const uint w = n < 4 ? v.x : (n < 8 ? v.y : v.z);
    return (w >> ((n & 3) * 8)) & 255;
}

int sbyte(uint2 q, uint i) {    // signed byte i (0..7) of 8 packed bytes
    const uint w = i < 4 ? q.x : q.y;
    return int(w << (24 - 8 * (i & 3))) >> 24;
}

// x: 8 values of sub-block j for this thread's token
void load_x(uint xbase, uint b, uint j, uint part, uint slot) {
    const uint xb = xbase + b * Q8_BYTES;
    const float xd = asfloat(X.Load(xb + Q8_D));
    const uint2 q = X.Load2(xb + Q8_QS + j * 32 + part * 8);
    [unroll] for (uint i = 0; i < 8; i++) Xs[(part * 8 + i) * 16 + (slot >> 2)][slot & 3] = xd * sbyte(q, i);
}

// 4 rows x 4 tokens of products over the 32 values in group-shared memory
void accumulate(uint tr, uint tt, inout float4 a0, inout float4 a1, inout float4 a2, inout float4 a3) {
    [unroll] for (uint k = 0; k < 32; k++) {
        const float4 w = Ws[k * 16 + tr], x = Xs[k * 16 + tt];
        a0 += w * x.x;
        a1 += w * x.y;
        a2 += w * x.z;
        a3 += w * x.w;
    }
}

void store_token(uint r0, uint t, uint tr, float4 a) {
    if (t >= nt) return;
    const uint r = r0 + tr * 4, o = (t * orow + r) * 4;
    if (r + 0 < rows) O.Store(o + 0, asuint(a.x));
    if (r + 1 < rows) O.Store(o + 4, asuint(a.y));
    if (r + 2 < rows) O.Store(o + 8, asuint(a.z));
    if (r + 3 < rows) O.Store(o + 12, asuint(a.w));
}

void store(uint r0, uint t0, uint tr, uint tt, float4 a0, float4 a1, float4 a2, float4 a3) {
    const uint t = t0 + tt * 4;
    store_token(r0, t + 0, tr, a0);
    store_token(r0, t + 1, tr, a1);
    store_token(r0, t + 2, tr, a2);
    store_token(r0, t + 3, tr, a3);
}

#ifndef MM_INTEL
// q4_K block: half d, half dmin, 12 bytes of 6-bit scales and mins, 128 bytes of nibbles
[numthreads(256, 1, 1)]
void mm_q4_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r0 = gid.x * 64, t0 = gid.y * 64;
    const uint slot = li >> 2, part = li & 3;   // loading: row / token `slot`, values part*8 .. +8
    const uint tr = li & 15, tt = li >> 4;      // computing: rows tr*4 .. +4, tokens tt*4 .. +4
    const uint wbase = min(r0 + slot, rows - 1) * wrow, xbase = min(t0 + slot, nt - 1) * xrow;
    float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint b = 0; b < nb; b++) {
        const uint wb = wbase + b * Q4K_BYTES;
        const uint dd = W.Load(wb);
        const float d = f16tof32(dd), dmin = f16tof32(dd >> 16);
        const uint3 sc = W.Load3(wb + 4);
        for (uint j = 0; j < 8; j++) {
            uint s, m;
            if (j < 4) {
                s = byte_of(sc, j) & 63;
                m = byte_of(sc, j + 4) & 63;
            } else {
                s = (byte_of(sc, j + 4) & 15) | ((byte_of(sc, j - 4) >> 6) << 4);
                m = (byte_of(sc, j + 4) >> 4) | ((byte_of(sc, j) >> 6) << 4);
            }
            const float ds = d * s, dm = dmin * m;
            const uint2 q = W.Load2(wb + 16 + (j >> 1) * 32 + part * 8);
            const uint shift = (j & 1) * 4;
            [unroll] for (uint i = 0; i < 8; i++) {
                const uint w = i < 4 ? q.x : q.y;
                Ws[(part * 8 + i) * 16 + (slot >> 2)][slot & 3] = ds * ((w >> ((i & 3) * 8 + shift)) & 15) - dm;
            }
            load_x(xbase, b, j, part, slot);
            GroupMemoryBarrierWithGroupSync();
            accumulate(tr, tt, a0, a1, a2, a3);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    store(r0, t0, tr, tt, a0, a1, a2, a3);
}

// repacked q6_K block (Q6_* offsets): half d (+2 bytes padding), ql[128], qh[64], int8 scales[16]
[numthreads(256, 1, 1)]
void mm_q6_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r0 = gid.x * 64, t0 = gid.y * 64;
    const uint slot = li >> 2, part = li & 3;
    const uint tr = li & 15, tt = li >> 4;
    const uint wbase = min(r0 + slot, rows - 1) * wrow, xbase = min(t0 + slot, nt - 1) * xrow;
    float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint b = 0; b < nb; b++) {
        const uint wb = wbase + b * Q6K_BYTES;
        const float d = f16tof32(W.Load(wb + Q6_D));
        for (uint j = 0; j < 8; j++) {
            // sub-block j: half h = j / 4 of 128 values, quarter q = j % 4 of 32 values
            const uint h = j >> 2, q = j & 3;
            const uint2 ql = W.Load2(wb + Q6_QL + h * 64 + (q & 1) * 32 + part * 8);
            const uint2 qh = W.Load2(wb + Q6_QH + h * 32 + part * 8);
            const uint si = h * 8 + q * 2 + (part >> 1);
            const int s = int(W.Load(wb + Q6_SC + (si & ~3u)) << (24 - 8 * (si & 3))) >> 24;
            const float ds = d * s;
            const uint lshift = (q >> 1) * 4, hshift = q * 2;
            [unroll] for (uint i = 0; i < 8; i++) {
                const uint bl = ((i < 4 ? ql.x : ql.y) >> ((i & 3) * 8)) & 255;
                const uint bh = ((i < 4 ? qh.x : qh.y) >> ((i & 3) * 8)) & 255;
                const int v = int(((bl >> lshift) & 15) | (((bh >> hshift) & 3) << 4)) - 32;
                Ws[(part * 8 + i) * 16 + (slot >> 2)][slot & 3] = ds * v;
            }
            load_x(xbase, b, j, part, slot);
            GroupMemoryBarrierWithGroupSync();
            accumulate(tr, tt, a0, a1, a2, a3);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    store(r0, t0, tr, tt, a0, a1, a2, a3);
}

#else
// Intel (Vulkan): the same 64 x 64 tiles with 128 threads of 8 rows x 4 tokens each, so that every
// float4 of weights read from group-shared memory serves twice as many products (group-shared
// reads, not arithmetic, limit the 4 x 4 version there); each thread loads 16 values per step.

void store8(uint r0, uint t0, uint tr, uint tt, float4 a0, float4 a1, float4 a2, float4 a3,
            float4 c0, float4 c1, float4 c2, float4 c3) {
    store(r0, t0, tr, tt, a0, a1, a2, a3);
    store(r0, t0, tr + 8, tt, c0, c1, c2, c3);
}

// 8 rows (tr*4 .. +4 and 32 + tr*4 .. +4) x 4 tokens (tt*4 .. +4) over the 32 values in group-shared memory
void accumulate8(uint tr, uint tt, inout float4 a0, inout float4 a1, inout float4 a2, inout float4 a3,
                 inout float4 c0, inout float4 c1, inout float4 c2, inout float4 c3) {
    [unroll] for (uint k = 0; k < 32; k++) {
        const float4 w0 = Ws[k * 16 + tr], w1 = Ws[k * 16 + 8 + tr], x = Xs[k * 16 + tt];
        a0 += w0 * x.x; a1 += w0 * x.y; a2 += w0 * x.z; a3 += w0 * x.w;
        c0 += w1 * x.x; c1 += w1 * x.y; c2 += w1 * x.z; c3 += w1 * x.w;
    }
}

// x: values part*16 .. +16 of sub-block j for this thread's token
void load_x16(uint xb, float xd, uint j, uint part, uint slot) {
    const uint4 q = X4[(xb + Q8_QS) / 16 + j * 2 + part];
    [unroll] for (uint i = 0; i < 16; i++)
        Xs[(part * 16 + i) * 16 + (slot >> 2)][slot & 3] = xd * float(int(q[i >> 2] << (24 - 8 * (i & 3))) >> 24);
}

[numthreads(128, 1, 1)]
void mm_q4_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r0 = gid.x * 64, t0 = gid.y * 64;
    const uint slot = li >> 1, part = li & 1;   // loading: row / token `slot`, values part*16 .. +16
    const uint tr = li & 7, tt = li >> 3;       // computing: see accumulate8
    const uint wbase = min(r0 + slot, rows - 1) * wrow, xbase = min(t0 + slot, nt - 1) * xrow;
    float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (uint b = 0; b < nb; b++) {
        const uint wb = wbase + b * Q4K_BYTES, xb = xbase + b * Q8_BYTES;
        const uint4 hd = W4[wb >> 4];
        const float d = f16tof32(hd.x), dmin = f16tof32(hd.x >> 16), xd = asfloat(X.Load(xb + Q8_D));
        for (uint j = 0; j < 8; j++) {
            uint s, m;
            if (j < 4) {
                s = byte_of(hd.yzw, j) & 63;
                m = byte_of(hd.yzw, j + 4) & 63;
            } else {
                s = (byte_of(hd.yzw, j + 4) & 15) | ((byte_of(hd.yzw, j - 4) >> 6) << 4);
                m = (byte_of(hd.yzw, j + 4) >> 4) | ((byte_of(hd.yzw, j) >> 6) << 4);
            }
            const float ds = d * s, dm = dmin * m;
            const uint4 q = W4[(wb >> 4) + 1 + (j >> 1) * 2 + part];
            const uint shift = (j & 1) * 4;
            [unroll] for (uint i = 0; i < 16; i++)
                Ws[(part * 16 + i) * 16 + (slot >> 2)][slot & 3] = ds * ((q[i >> 2] >> ((i & 3) * 8 + shift)) & 15) - dm;
            load_x16(xb, xd, j, part, slot);
            GroupMemoryBarrierWithGroupSync();
            accumulate8(tr, tt, a0, a1, a2, a3, c0, c1, c2, c3);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    store8(r0, t0, tr, tt, a0, a1, a2, a3, c0, c1, c2, c3);
}

[numthreads(128, 1, 1)]
void mm_q6_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r0 = gid.x * 64, t0 = gid.y * 64;
    const uint slot = li >> 1, part = li & 1;
    const uint tr = li & 7, tt = li >> 3;
    const uint wbase = min(r0 + slot, rows - 1) * wrow, xbase = min(t0 + slot, nt - 1) * xrow;
    float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (uint b = 0; b < nb; b++) {
        const uint wb = wbase + b * Q6K_BYTES, xb = xbase + b * Q8_BYTES, w4 = wb / 16;
        const float d = f16tof32(W.Load(wb + Q6_D)), xd = asfloat(X.Load(xb + Q8_D));
        const uint4 scw = W4[w4 + Q6_SC / 16];
        for (uint j = 0; j < 8; j++) {
            // sub-block j: half h = j / 4 of 128 values, quarter q = j % 4 of 32; one scale per 16 values
            const uint h = j >> 2, q = j & 3, si = h * 8 + q * 2 + part;
            const uint4 ql = W4[w4 + (Q6_QL + h * 64 + (q & 1) * 32) / 16 + part];
            const uint4 qh = W4[w4 + (Q6_QH + h * 32) / 16 + part];
            const float ds = d * float(int(scw[si >> 2] << (24 - 8 * (si & 3))) >> 24);
            const uint lshift = (q >> 1) * 4, hshift = q * 2;
            [unroll] for (uint i = 0; i < 16; i++) {
                const uint bl = (ql[i >> 2] >> ((i & 3) * 8)) & 255, bh = (qh[i >> 2] >> ((i & 3) * 8)) & 255;
                Ws[(part * 16 + i) * 16 + (slot >> 2)][slot & 3] = ds * (int(((bl >> lshift) & 15) | (((bh >> hshift) & 3) << 4)) - 32);
            }
            load_x16(xb, xd, j, part, slot);
            GroupMemoryBarrierWithGroupSync();
            accumulate8(tr, tt, a0, a1, a2, a3, c0, c1, c2, c3);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    store8(r0, t0, tr, tt, a0, a1, a2, a3, c0, c1, c2, c3);
}
#endif

// ---------------------------------------------------------------- matrix x few vectors

#ifndef MV_LANES
#define MV_LANES 32                 // lanes per row (a power of two)
#define MV_LOG2 5
#define MV_GROUP 256                // threads per group: MV_GROUP / MV_LANES rows
#endif

groupshared float red[MV_GROUP];

int sx(uint w, uint k) { return int(w << (24 - 8 * k)) >> 24; }     // signed byte k of a word

// sum of the MV_LANES lanes of each row; lane 0 writes the result
void reduce_store(uint li, uint r, uint t, float sum) {
    red[li] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = MV_LANES / 2; s > 0; s >>= 1) {
        if ((li & (MV_LANES - 1)) < s) red[li] += red[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if ((li & (MV_LANES - 1)) == 0 && r < rows) O.Store((t * orow + r) * 4, asuint(red[li]));
}

// scales and mins (s1, s2, m1, m2) of sub-blocks 2p and 2p+1, read as 16-bit halves of the
// 12 scale bytes (fxc miscompiles the byte-index version with j = 2p + 1)
uint4 scales_pair(uint3 sc, uint p) {
    const uint sh = (p & 1) * 16, hx = sc.x >> sh, hy = sc.y >> sh, hz = sc.z >> sh;
    if (p < 2) return uint4(hx & 63, (hx >> 8) & 63, hy & 63, (hy >> 8) & 63);
    return uint4((hz & 15) | (((hx >> 6) & 3) << 4), ((hz >> 8) & 15) | (((hx >> 14) & 3) << 4),
                 ((hz >> 4) & 15) | (((hy >> 6) & 3) << 4), ((hz >> 12) & 15) | (((hy >> 14) & 3) << 4));
}

// piece p of a block: sub-blocks 2p (low nibbles of qs[32p..]) and 2p+1 (high nibbles)
[numthreads(MV_GROUP, 1, 1)]
void mv_q4_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r = gid.x * (MV_GROUP / MV_LANES) + (li >> MV_LOG2), t = gid.y;
    float sum = 0;
    if (r < rows) {
        const uint wbase = r * wrow, xbase = t * xrow;
        for (uint p = li & (MV_LANES - 1); p < nb * 4; p += MV_LANES) {
            const uint b = p >> 2, pp = p & 3;
            const uint wb = wbase + b * Q4K_BYTES, xb = xbase + b * Q8_BYTES;
#ifdef PITHOS_VK
            const uint4 hd = W4[wb >> 4];
            const uint dd = hd.x;
            const uint4 sm = scales_pair(hd.yzw, pp);
            const uint wq = (wb >> 4) + 1 + pp * 2, xq = (xb + Q8_QS) / 16 + pp * 4;
            const uint4 q[2] = { W4[wq], W4[wq + 1] };
            const uint4 xl[2] = { X4[xq], X4[xq + 1] };
            const uint4 xh[2] = { X4[xq + 2], X4[xq + 3] };
#else
            const uint dd = W.Load(wb);
            const uint3 sc = W.Load3(wb + 4);
            const uint4 sm = scales_pair(sc, pp);
            const uint4 q[2] = { W.Load4(wb + 16 + pp * 32), W.Load4(wb + 32 + pp * 32) };
            const uint4 xl[2] = { X.Load4(xb + 4 + pp * 64), X.Load4(xb + 20 + pp * 64) };
            const uint4 xh[2] = { X.Load4(xb + 36 + pp * 64), X.Load4(xb + 52 + pp * 64) };
#endif
#ifdef MV_SWAR
            // (w0 + w2 << 16) * (x2 + x0 << 16) holds w0 x0 + w2 x2 in its high half when no term
            // borrows or carries: x made unsigned (x + 128, corrected below with the nibble sums),
            // and 4 words per accumulator keep both halves below 2^16
            uint p1[2] = { 0, 0 }, p2[2] = { 0, 0 }, s1 = 0, s2 = 0, sa = 0, sc = 0;
            [unroll] for (uint i = 0; i < 8; i++) {
                const uint w = q[i >> 2][i & 3], a = xl[i >> 2][i & 3] ^ 0x80808080, c = xh[i >> 2][i & 3] ^ 0x80808080;
                const uint ra = (a << 16) | (a >> 16), rc = (c << 16) | (c >> 16);
                const uint l1 = w & 0x000F000F, l2 = (w >> 8) & 0x000F000F;
                const uint h1 = (w >> 4) & 0x000F000F, h2 = (w >> 12) & 0x000F000F;
                p1[i >> 2] += l1 * (ra & 0x00FF00FF) + l2 * ((ra >> 8) & 0x00FF00FF);
                p2[i >> 2] += h1 * (rc & 0x00FF00FF) + h2 * ((rc >> 8) & 0x00FF00FF);
                s1 += l1 + l2;
                s2 += h1 + h2;
                sa += (a & 0x00FF00FF) + ((a >> 8) & 0x00FF00FF);
                sc += (c & 0x00FF00FF) + ((c >> 8) & 0x00FF00FF);
            }
            const int d1 = int((p1[0] >> 16) + (p1[1] >> 16)) - 128 * int((s1 & 0xffff) + (s1 >> 16));
            const int d2 = int((p2[0] >> 16) + (p2[1] >> 16)) - 128 * int((s2 & 0xffff) + (s2 >> 16));
            const int n1 = int((sa & 0xffff) + (sa >> 16)) - 4096, n2 = int((sc & 0xffff) + (sc >> 16)) - 4096;
#else
            int d1 = 0, n1 = 0, d2 = 0, n2 = 0;
            [unroll] for (uint i = 0; i < 8; i++) {
                const uint w = q[i >> 2][i & 3], a = xl[i >> 2][i & 3], c = xh[i >> 2][i & 3];
                [unroll] for (uint k = 0; k < 4; k++) {
                    const int va = sx(a, k), vc = sx(c, k);
                    d1 += int((w >> (8 * k)) & 15) * va;
                    d2 += int((w >> (8 * k + 4)) & 15) * vc;
                    n1 += va;
                    n2 += vc;
                }
            }
#endif
            sum += asfloat(X.Load(xb + Q8_D)) * (f16tof32(dd) * float(int(sm.x) * d1 + int(sm.y) * d2) -
                                          f16tof32(dd >> 16) * float(int(sm.z) * n1 + int(sm.w) * n2));
        }
    }
    reduce_store(li, r, t, sum);
}

// piece p of a block: half h = p / 2, ql bytes [h*64 + a*32, +32) with a = p % 2: low nibbles
// make sub-block j1 = 4h + a, high nibbles j2 = j1 + 2; scales change every 16 values
[numthreads(MV_GROUP, 1, 1)]
void mv_q6_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r = gid.x * (MV_GROUP / MV_LANES) + (li >> MV_LOG2), t = gid.y;
    float sum = 0;
    if (r < rows) {
        const uint wbase = r * wrow, xbase = t * xrow;
        for (uint p = li & (MV_LANES - 1); p < nb * 4; p += MV_LANES) {
            const uint b = p >> 2, h = (p & 3) >> 1, a = p & 1;
            const uint wb = wbase + b * Q6K_BYTES, xb = xbase + b * Q8_BYTES;
            const uint j1 = h * 4 + a, j2 = j1 + 2;
#ifdef PITHOS_VK
            const uint wl = (wb + Q6_QL + h * 64 + a * 32) / 16, wh = (wb + Q6_QH + h * 32) / 16;
            const uint x1q = (xb + Q8_QS + j1 * 32) / 16, x2q = (xb + Q8_QS + j2 * 32) / 16;
            const uint4 ql[2] = { W4[wl], W4[wl + 1] };
            const uint4 qh[2] = { W4[wh], W4[wh + 1] };
            const uint4 x1[2] = { X4[x1q], X4[x1q + 1] };
            const uint4 x2[2] = { X4[x2q], X4[x2q + 1] };
            const uint4 sc = W4[(wb + Q6_SC) / 16];
            const uint sl = h ? sc.z : sc.x, sh = h ? sc.w : sc.y;
#else
            const uint4 ql[2] = { W.Load4(wb + 4 + h * 64 + a * 32), W.Load4(wb + 20 + h * 64 + a * 32) };
            const uint4 qh[2] = { W.Load4(wb + 132 + h * 32), W.Load4(wb + 148 + h * 32) };
            const uint4 x1[2] = { X.Load4(xb + 4 + j1 * 32), X.Load4(xb + 20 + j1 * 32) };
            const uint4 x2[2] = { X.Load4(xb + 4 + j2 * 32), X.Load4(xb + 20 + j2 * 32) };
            const uint sl = W.Load(wb + 196 + h * 8), sh = W.Load(wb + 200 + h * 8);
#endif
            const uint hs1 = a * 2, hs2 = a * 2 + 4;
            int d1[2] = { 0, 0 }, d2[2] = { 0, 0 };
            [unroll] for (uint i = 0; i < 8; i++) {
                const uint wl = ql[i >> 2][i & 3], wh = qh[i >> 2][i & 3];
                const uint u = x1[i >> 2][i & 3], v = x2[i >> 2][i & 3];
                [unroll] for (uint k = 0; k < 4; k++) {
                    const uint bl = (wl >> (8 * k)) & 255, bh = (wh >> (8 * k)) & 255;
                    d1[i >> 2] += (int((bl & 15) | (((bh >> hs1) & 3) << 4)) - 32) * sx(u, k);
                    d2[i >> 2] += (int((bl >> 4) | (((bh >> hs2) & 3) << 4)) - 32) * sx(v, k);
                }
            }
            const int dot = sx(sl, a * 2) * d1[0] + sx(sl, a * 2 + 1) * d1[1] + sx(sh, a * 2) * d2[0] +
                            sx(sh, a * 2 + 1) * d2[1];
            sum += asfloat(X.Load(xb + Q8_D)) * f16tof32(W.Load(wb + Q6_D)) * float(dot);
        }
    }
    reduce_store(li, r, t, sum);
}
