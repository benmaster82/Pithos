// kernels_dx_token.hlsl - the non-matmul steps of the whole-token decode pass (gpu_d3d12.c).
//
// One command list per token: norms, residual adds and Q8_K quantization of the matmul inputs,
// q/k norm + RoPE + KV cache writes, attention, short convolution and SwiGLU, with the matmuls
// in between (mv_* in kernels_dx.hlsl). All activations live in one scratch buffer; the
// constants give byte offsets into it. Regenerate kernels_dx.h after editing: tools\dx_shaders.cmd

globallycoherent RWByteAddressBuffer U : register(u0);  // scratch (other threads read what one wrote)
RWByteAddressBuffer KV : register(u1);   // key and value cache: f16 pairs, [n_attn][n_ctx][kvd] each
RWByteAddressBuffer CS : register(u2);   // short-conv state: [n_conv][l_cache - 1][dim] floats
ByteAddressBuffer SW : register(t0);     // small f32 weights: norms, conv kernels
ByteAddressBuffer IN : register(t1);     // this token: embedding[dim] | rope cos, sin [hd] | swa rope [hd]

cbuffer T : register(b0) { uint4 C0, C1, C2, C3; };

#define NONE 0xffffffff  // absent weight; never used in an address (root views have no bounds checks)
#define Q8_BYTES 292

groupshared float gs[256];
groupshared float2 g2[256];
groupshared float hs[512];
groupshared uint gq[256];

// f32 -> f16 rounded to nearest even, as the CPU (F16C) does; f32tof16 does not round the same way
uint f16(float f) {
    const uint x = asuint(f), sign = (x >> 16) & 0x8000, ax = x & 0x7fffffff;
    if (ax >= 0x47800000) return sign | (ax > 0x7f800000 ? 0x7e00 : 0x7c00);     // inf, nan, overflow
    if (ax < 0x38800000) return sign | uint(round(asfloat(ax) * 16777216.0));   // subnormal
    const uint r = ax - 0x38000000;
    return sign | ((r + 0xfff + ((r >> 13) & 1)) >> 13);
}

float ldu(uint o) { return asfloat(U.Load(o)); }
void stu(uint o, float v) { U.Store(o, asuint(v)); }
float ldw(uint o) { return asfloat(SW.Load(o)); }

float reduce_sum(uint li, float v) {
    gs[li] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (li < s) gs[li] += gs[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gs[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float reduce_max(uint li, float v) {
    gs[li] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (li < s) gs[li] = max(gs[li], gs[li + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gs[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Q8_K block from one value per thread (256 threads), as quantize_q8_K on the CPU (no bsums)
void quant_block(uint li, float v, uint qo) {
    const float amax = reduce_max(li, abs(v));
    const float iscale = amax > 0 ? 127.0 / amax : 0;
    gq[li] = uint(clamp(int(round(v * iscale)), -127, 127)) & 255;
    GroupMemoryBarrierWithGroupSync();
    if (li < 64) U.Store(qo + 4 + 4 * li, gq[4 * li] | (gq[4 * li + 1] << 8) | (gq[4 * li + 2] << 16) | (gq[4 * li + 3] << 24));
    if (li == 0) stu(qo, amax > 0 ? 1.0 / iscale : 0);
    GroupMemoryBarrierWithGroupSync();
}

// x = embedding (mode 0) or x += out, RMS-normalized by the post weight if any (mode 1);
// then the matmul input: Q8_K of rmsnorm(x) * norm weight. One group.
// C0: dim, x, out, q   C1: mode, post weight, norm weight, eps
[numthreads(256, 1, 1)]
void resid_norm(uint li : SV_GroupIndex) {
    const uint D = C0.x, xo = C0.y, oo = C0.z, qo = C0.w, mode = C1.x, nw = C1.z;
    const bool post = C1.y != NONE;
    const uint pw = post ? C1.y : nw;   // fxc flattens the branches: every address must stay valid
    const float eps = asfloat(C1.w);
    float ps = 1;
    if (mode == 1 && post) {
        float ss = 0;
        for (uint i = li; i < D; i += 256) {
            const float o = ldu(oo + 4 * i);
            ss += o * o;
        }
        ps = rsqrt(reduce_sum(li, ss) / D + eps);
    }
    float ss = 0;
    for (uint i = li; i < D; i += 256) {
        float v;
        if (mode == 0) {
            v = asfloat(IN.Load(4 * i));
        } else {
            float o = ldu(oo + 4 * i);
            if (post) o *= ps * ldw(pw + 4 * i);
            v = ldu(xo + 4 * i) + o;
        }
        stu(xo + 4 * i, v);
        ss += v * v;
    }
    const float s = rsqrt(reduce_sum(li, ss) / D + eps);
    for (uint b = 0; b < D / 256; b++) {
        const uint i = b * 256 + li;
        quant_block(li, ldu(xo + 4 * i) * s * ldw(nw + 4 * i), qo + b * Q8_BYTES);
    }
}

// Q8_K of n floats, one block per group.  C0: in, q
[numthreads(256, 1, 1)]
void quant(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    quant_block(li, ldu(C0.x + 4 * (gid.x * 256 + li)), C0.y + gid.x * Q8_BYTES);
}

// SwiGLU: Q8_K of silu(gate) * up, one block per group.  C0: gate, up, q
[numthreads(256, 1, 1)]
void silu_quant(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint i = gid.x * 256 + li;
    const float g = ldu(C0.x + 4 * i), u = ldu(C0.y + 4 * i);
    quant_block(li, g / (1 + exp(-g)) * u, C0.z + gid.x * Q8_BYTES);
}

// short convolution of 256 channels per group, then Q8_K of the result.
// C0: dim, bcx (in_proj output: B | C | x), q, l_cache   C1: conv weights, state
[numthreads(256, 1, 1)]
void shortconv(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, bo = C0.y, lc = C0.w, wo = C1.x, so = C1.y;
    const uint ch = gid.x * 256 + li;
    const float bx = ldu(bo + 4 * ch) * ldu(bo + 4 * (2 * D + ch));
    const uint w = wo + 4 * ch * lc;
    float acc = ldw(w + 4 * (lc - 1)) * bx;
    for (uint k = 0; k + 1 < lc; k++) acc += ldw(w + 4 * k) * asfloat(CS.Load(so + 4 * (k * D + ch)));
    for (uint m = 0; m + 2 < lc; m++) CS.Store(so + 4 * (m * D + ch), CS.Load(so + 4 * ((m + 1) * D + ch)));
    CS.Store(so + 4 * ((lc - 2) * D + ch), asuint(bx));
    quant_block(li, ldu(bo + 4 * (D + ch)) * acc, C0.z + gid.x * Q8_BYTES);
}

// per head: RMSNorm with the q or k norm, NEOX RoPE; q heads go back in place, k and v heads
// into the cache at this position. Groups: n_head q heads, then n_kv_head k/v heads.
// C0: hd, qkv (q | k | v), dim, kvd   C1: q norm, k norm, rope (offset in IN), eps
// C2: k cache offset of this layer and position, v cache offset, n_head
[numthreads(64, 1, 1)]
void attn_prep(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x, qkv = C0.y, D = C0.z, kvd = C0.w, nh = C2.z, half = hd / 2;
    const uint g = gid.x, kh = g - nh;
    const bool isq = g < nh;
    const uint off = isq ? qkv + 4 * g * hd : qkv + 4 * (D + kh * hd), nw = isq ? C1.x : C1.y;
    float ss = 0;
    for (uint i = li; i < hd; i += 64) {
        const float v = ldu(off + 4 * i);
        hs[i] = v;
        ss += v * v;
    }
    gs[li] = ss;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 32; s > 0; s >>= 1) {
        if (li < s) gs[li] += gs[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float sc = rsqrt(gs[0] / hd + asfloat(C1.w));
    for (uint r = li; r < half; r += 64) {
        const float x0 = hs[r] * sc * ldw(nw + 4 * r), x1 = hs[r + half] * sc * ldw(nw + 4 * (r + half));
        const float cs = asfloat(IN.Load(C1.z + 4 * r)), sn = asfloat(IN.Load(C1.z + 4 * (half + r)));
        hs[r] = x0 * cs - x1 * sn;
        hs[r + half] = x0 * sn + x1 * cs;
    }
    GroupMemoryBarrierWithGroupSync();
    if (isq) {
        for (uint o = li; o < hd; o += 64) stu(off + 4 * o, hs[o]);
    } else {
        const uint vo = qkv + 4 * (D + kvd + kh * hd);
        for (uint p = li; p < half; p += 64) {
            KV.Store(C2.x + 2 * kh * hd + 4 * p, f16(hs[2 * p]) | (f16(hs[2 * p + 1]) << 16));
            KV.Store(C2.y + 2 * kh * hd + 4 * p, f16(ldu(vo + 8 * p)) | (f16(ldu(vo + 8 * p + 4)) << 16));
        }
    }
}

// one q head per group over cache positions [j0, npos): scores, softmax, weighted sum of v,
// times the sigmoid output gate if any.
// C0: hd, q, gate (or NONE), out   C1: k cache of this layer, v cache, kvd, scores
// C2: j0, npos, q heads per kv head, scores row length   C3: 1/sqrt(hd)
[numthreads(256, 1, 1)]
void attention(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x, half = hd / 2, h = gid.x, kh = h / C2.z, kvd = C1.z, j0 = C2.x, npos = C2.y;
    const uint kb = C1.x + 2 * kh * hd, vb = C1.y + 2 * kh * hd, so = C1.w + 4 * h * C2.w;
    const float scale = asfloat(C3.x);
    for (uint i = li; i < hd; i += 256) hs[i] = ldu(C0.y + 4 * (h * hd + i));
    GroupMemoryBarrierWithGroupSync();
    float mx = -1e30;
    for (uint j = j0 + li; j < npos; j += 256) {
        const uint ko = kb + 2 * j * kvd;
        float d = 0;
        for (uint p = 0; p < half; p++) {
            const uint w = KV.Load(ko + 4 * p);
            d += f16tof32(w) * hs[2 * p] + f16tof32(w >> 16) * hs[2 * p + 1];
        }
        d *= scale;
        stu(so + 4 * (j - j0), d);
        mx = max(mx, d);
    }
    mx = reduce_max(li, mx);
    float sum = 0;
    for (uint je = j0 + li; je < npos; je += 256) {
        const float e = exp(ldu(so + 4 * (je - j0)) - mx);
        stu(so + 4 * (je - j0), e);
        sum += e;
    }
    sum = reduce_sum(li, sum);
    DeviceMemoryBarrierWithGroupSync();
    // 256 threads = half pairs of dimensions x 256 / half slices of positions
    const uint slices = 256 / half, p = li % half, sl = li / half;
    float2 acc = 0;
    for (uint jv = j0 + sl; jv < npos; jv += slices) {
        const uint w = KV.Load(vb + 2 * jv * kvd + 4 * p);
        acc += ldu(so + 4 * (jv - j0)) * float2(f16tof32(w), f16tof32(w >> 16));
    }
    g2[li] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (sl == 0) {
        for (uint s = 1; s < slices; s++) acc += g2[s * half + p];
        float inv = 1 / sum;
        const uint go = C0.z != NONE ? C0.z : C0.w;     // a valid address even without a gate
        if (C0.z != NONE) inv /= 1 + exp(-ldu(go + 4 * h));
        stu(C0.w + 4 * (h * hd + 2 * p), acc.x * inv);
        stu(C0.w + 4 * (h * hd + 2 * p + 1), acc.y * inv);
    }
}
