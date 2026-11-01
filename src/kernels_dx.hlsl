// kernels_dx.hlsl - Direct3D 12 compute kernels for gpu_d3d12.c (prompt matmuls).
//
// out[t][r] = sum_k W[r][k] * x[t][k] for a tile of 64 weight rows x 64 tokens per group of
// 256 threads. Weights stay quantized in GPU memory (Q4_K as in the GGUF file, Q6_K repacked
// to 4-byte aligned 212-byte blocks, see gpu_d3d12.c); activations are the Q8_K rows the CPU
// path uses. Each step dequantizes a 32-wide slice of K into group-shared memory, then every
// thread accumulates 4 rows x 4 tokens.
//
// Regenerate kernels_dx.h after editing:  tools\dx_shaders.cmd

ByteAddressBuffer W : register(t0);     // weight rows, wrow bytes apart
ByteAddressBuffer X : register(t1);     // Q8_K activation rows, xrow bytes apart
RWByteAddressBuffer O : register(u0);   // out[t * orow + r], floats

cbuffer P : register(b0) {
    uint rows, nb, nt, wrow, xrow, orow;    // nb: 256-element blocks per row
};

#define Q8_BYTES 292
#define Q4K_BYTES 144
#define Q6K_BYTES 212

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
    const float xd = asfloat(X.Load(xb));
    const uint2 q = X.Load2(xb + 4 + j * 32 + part * 8);
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

// repacked q6_K block: half d (+2 bytes padding), ql[128], qh[64], int8 scales[16]
[numthreads(256, 1, 1)]
void mm_q6_K(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint r0 = gid.x * 64, t0 = gid.y * 64;
    const uint slot = li >> 2, part = li & 3;
    const uint tr = li & 15, tt = li >> 4;
    const uint wbase = min(r0 + slot, rows - 1) * wrow, xbase = min(t0 + slot, nt - 1) * xrow;
    float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint b = 0; b < nb; b++) {
        const uint wb = wbase + b * Q6K_BYTES;
        const float d = f16tof32(W.Load(wb));
        for (uint j = 0; j < 8; j++) {
            // sub-block j: half h = j / 4 of 128 values, quarter q = j % 4 of 32 values
            const uint h = j >> 2, q = j & 3;
            const uint2 ql = W.Load2(wb + 4 + h * 64 + (q & 1) * 32 + part * 8);
            const uint2 qh = W.Load2(wb + 132 + h * 32 + part * 8);
            const uint si = h * 8 + q * 2 + (part >> 1);
            const int s = int(W.Load(wb + 196 + (si & ~3u)) << (24 - 8 * (si & 3))) >> 24;
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
