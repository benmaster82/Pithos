// kernels_dx_token.hlsl - the non-matmul steps of the whole forward pass on the GPU (gpu_d3d12.c).
//
// One command list per token or prompt chunk: norms, residual adds and Q8_K quantization of the
// matmul inputs, q/k norm + RoPE + KV cache writes, attention, short convolution and SwiGLU,
// with the matmuls in between (kernels_dx.hlsl). All activations live in one scratch buffer,
// one row per token; the constants give byte offsets into it and the second group index is the
// token. Regenerate kernels_dx.h after editing: tools\dx_shaders.cmd

globallycoherent RWByteAddressBuffer U : register(u0);  // scratch (other threads read what one wrote)
RWByteAddressBuffer KV : register(u1);   // key and value cache: f16 pairs, [n_attn][n_ctx][kvd] each
RWByteAddressBuffer CS : register(u2);   // short-conv state: [n_conv][l_cache - 1][dim] floats
ByteAddressBuffer SW : register(t0);     // small f32 weights: norms, conv kernels
ByteAddressBuffer IN : register(t1);     // per token: embeddings [n][dim] | rope cos, sin [n][hd] | swa rope [n][hd]

cbuffer T : register(b0) { uint4 C0, C1, C2, C3; };

#define NONE 0xffffffff  // absent weight; never used in an address (root views have no bounds checks)
#ifndef Q8_BYTES
#define Q8_BYTES 292                // Q8_K block: float d at Q8_D, 256 int8 at Q8_QS (see kernels_dx.hlsl)
#define Q8_D 0
#define Q8_QS 4
#endif

groupshared float gs[256];
groupshared float2 g2[256];
groupshared float hs[512];
groupshared uint gq[256];
groupshared float gp[256];

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
    if (li < 64) U.Store(qo + Q8_QS + 4 * li, gq[4 * li] | (gq[4 * li + 1] << 8) | (gq[4 * li + 2] << 16) | (gq[4 * li + 3] << 24));
    if (li == 0) stu(qo + Q8_D, amax > 0 ? 1.0 / iscale : 0);
    GroupMemoryBarrierWithGroupSync();
}

// x = embedding (mode 0) or x += out, RMS-normalized by the post weight if any (mode 1);
// then the matmul input: Q8_K of rmsnorm(x) * norm weight. One group per token.
// C0: dim, x, out, q   C1: mode, post weight, norm weight, eps
[numthreads(256, 1, 1)]
void resid_norm(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, t = gid.y, xo = C0.y + 4 * t * D, oo = C0.z + 4 * t * D, eo = 4 * t * D;
    const uint qo = C0.w + t * (D / 256) * Q8_BYTES, mode = C1.x, nw = C1.z;
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
            v = asfloat(IN.Load(eo + 4 * i));
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

// Q8_K of rows of floats, one block per group.  C0: in, q, in row bytes, q row bytes
[numthreads(256, 1, 1)]
void quant(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    quant_block(li, ldu(C0.x + gid.y * C0.z + 4 * (gid.x * 256 + li)), C0.y + gid.y * C0.w + gid.x * Q8_BYTES);
}

// SwiGLU: Q8_K of silu(gate) * up, one block per group.  C0: gate, up, q, row length
[numthreads(256, 1, 1)]
void silu_quant(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint i = gid.y * C0.w + gid.x * 256 + li;
    const float g = ldu(C0.x + 4 * i), u = ldu(C0.y + 4 * i);
    quant_block(li, g / (1 + exp(-g)) * u, C0.z + (gid.y * (C0.w / 256) + gid.x) * Q8_BYTES);
}

// short convolution of 256 channels per group for one token (decoding), then Q8_K of the result.
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
// into the cache at their positions. Groups: n_head q heads, then n_kv_head k/v heads; tokens.
// C0: hd, qkv (q | k | v), dim, kvd   C1: q norm, k norm, rope (offset in IN), eps
// C2: k cache offset of this layer and the first token's position, v cache offset, n_head
[numthreads(64, 1, 1)]
void attn_prep(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x, D = C0.z, kvd = C0.w, nh = C2.z, half = hd / 2, t = gid.y;
    const uint qkv = C0.y + 4 * t * (D + 2 * kvd), ro = C1.z + 4 * t * hd, ko = C2.x + 2 * t * kvd, vo2 = C2.y + 2 * t * kvd;
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
        const float cs = asfloat(IN.Load(ro + 4 * r)), sn = asfloat(IN.Load(ro + 4 * (half + r)));
        hs[r] = x0 * cs - x1 * sn;
        hs[r + half] = x0 * sn + x1 * cs;
    }
    GroupMemoryBarrierWithGroupSync();
    if (isq) {
        for (uint o = li; o < hd; o += 64) stu(off + 4 * o, hs[o]);
    } else {
        const uint vo = qkv + 4 * (D + kvd + kh * hd);
        for (uint p = li; p < half; p += 64) {
            KV.Store(ko + 2 * kh * hd + 4 * p, f16(hs[2 * p]) | (f16(hs[2 * p + 1]) << 16));
            KV.Store(vo2 + 2 * kh * hd + 4 * p, f16(ldu(vo + 8 * p)) | (f16(ldu(vo + 8 * p + 4)) << 16));
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

// ---------------------------------------------------------------- prompt chunks

// bx = B * x of channel ch for token i of the chunk (bcx rows of 3 * dim)
float conv_bx(uint bo, uint D, uint i, uint ch) { return ldu(bo + 4 * (i * 3 * D + ch)) * ldu(bo + 4 * (i * 3 * D + 2 * D + ch)); }

// short convolution of a chunk: token t sees the state, then the chunk's earlier tokens.
// Groups: 256 channels, tokens. The state moves on afterwards (conv_state).
// C0: dim, bcx, q, l_cache   C1: conv weights, state
[numthreads(256, 1, 1)]
void conv_chunk(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, bo = C0.y, lc = C0.w, wo = C1.x, so = C1.y, t = gid.y;
    const uint ch = gid.x * 256 + li, w = wo + 4 * ch * lc;
    float acc = 0;
    for (uint k = 0; k < lc; k++) {
        // input t - (lc - 1) + k: from the chunk, or from the state before it (both indices
        // clamped so that every address stays valid)
        const int i = int(t + k) - int(lc - 1);
        const float fromx = conv_bx(bo, D, uint(max(i, 0)), ch);
        const float froms = asfloat(CS.Load(so + 4 * (uint(min(int(lc - 1) + i, int(lc) - 2)) * D + ch)));
        acc += ldw(w + 4 * k) * (i >= 0 ? fromx : froms);
    }
    quant_block(li, ldu(bo + 4 * (t * 3 * D + D + ch)) * acc, C0.z + (t * (D / 256) + gid.x) * Q8_BYTES);
}

// the state after a chunk of n tokens: its last l_cache - 1 inputs.  C0: dim, bcx, n, l_cache  C1.y: state
[numthreads(256, 1, 1)]
void conv_state(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, bo = C0.y, n = C0.z, lc = C0.w, so = C1.y, ch = gid.x * 256 + li;
    for (uint m = 0; m + 1 < lc; m++) {     // reads index n + m > m: never one already rewritten
        const uint si = n + m;
        const float froms = asfloat(CS.Load(so + 4 * (min(si, lc - 2) * D + ch)));
        const float fromx = conv_bx(bo, D, si >= lc - 1 ? si - (lc - 1) : 0, ch);
        CS.Store(so + 4 * (m * D + ch), asuint(si < lc - 1 ? froms : fromx));
    }
}

// one q head of one token per group, over its cache positions in tiles of 256 with a running
// softmax (no scores buffer), times the sigmoid output gate if any.
// C0: hd, qkv, gate (or NONE), out   C1: k cache of this layer, v cache, kvd
// C2: position of the chunk's first token, sliding window (0: none), q heads per kv head, n_head
// C3: 1/sqrt(hd), dim
[numthreads(256, 1, 1)]
void attn_chunk(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x, half = hd / 2, h = gid.x, t = gid.y, kh = h / C2.z, kvd = C1.z, nh = C2.w, D = C3.y;
    // C1.w > 0: bidirectional, every token sees the C1.w positions of the chunk (the LFM3-Audio depth decoder)
    const uint npos = C1.w ? C1.w : C2.x + t + 1, j0 = C2.y && npos > C2.y ? npos - C2.y : 0;
    const uint kb = C1.x + 2 * kh * hd, vb = C1.y + 2 * kh * hd;
    const float scale = asfloat(C3.x);
    for (uint i = li; i < hd; i += 256) hs[i] = ldu(C0.y + 4 * (t * (D + 2 * kvd) + h * hd + i));
    GroupMemoryBarrierWithGroupSync();
    const uint slices = 256 / half, p = li % half, sl = li / half;
    float mx = -1e30, sum = 0;
    float2 acc = 0;
    for (uint base = j0; base < npos; base += 256) {
        const uint j = min(base + li, npos - 1);    // past the end: a valid row, ignored
        const uint ko = kb + 2 * j * kvd;
        float d = 0;
        for (uint q = 0; q < half; q++) {
            const uint w = KV.Load(ko + 4 * q);
            d += f16tof32(w) * hs[2 * q] + f16tof32(w >> 16) * hs[2 * q + 1];
        }
        const float s = base + li < npos ? d * scale : -1e30;
        const float m2 = max(mx, reduce_max(li, s)), corr = exp(mx - m2);
        const float e = base + li < npos ? exp(s - m2) : 0;
        gp[li] = e;
        sum = sum * corr + reduce_sum(li, e);   // its barriers also publish gp
        acc *= corr;
        const uint cnt = min(256u, npos - base);
        for (uint jv = sl; jv < cnt; jv += slices) {
            const uint w2 = KV.Load(vb + 2 * (base + jv) * kvd + 4 * p);
            acc += gp[jv] * float2(f16tof32(w2), f16tof32(w2 >> 16));
        }
        mx = m2;
        GroupMemoryBarrierWithGroupSync();      // gp is rewritten by the next tile
    }
    g2[li] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (sl == 0) {
        for (uint s2 = 1; s2 < slices; s2++) acc += g2[s2 * half + p];
        float inv = 1 / sum;
        const uint go = C0.z != NONE ? C0.z + 4 * (t * nh + h) : C0.w;   // a valid address even without a gate
        if (C0.z != NONE) inv /= 1 + exp(-ldu(go));
        const uint oo = C0.w + 4 * (t * D + h * hd + 2 * p);
        stu(oo, acc.x * inv);
        stu(oo + 4, acc.y * inv);
    }
}

#ifdef PITHOS_VK
// ---------------------------------------------------------------- Qwen3.5 / 3.6 (Vulkan only)
//
// The layers of qwen35moe run here, the routed experts on the CPU: after each layer's mixer the
// GPU stops with x and the normalized h read back; the CPU's expert sum y and the shared expert's
// gate s come back through IN for the next residual add. The CS buffer holds the CPU's DeltaNet
// state, layer by layer: the convolution's last K - 1 inputs [K-1][qkv], then S [vh][d][d].

// as reduce_sum, over 128 threads
float reduce_sum128(uint li, float v) {
    gs[li] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 64; s > 0; s >>= 1) {
        if (li < s) gs[li] += gs[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gs[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

groupshared float gm8[8][256];
groupshared uint gq8[8][256];

// quant_block for up to 8 blocks at once (value j of each thread in block j), with one reduction
void quant_blocks8(uint li, float v[8], uint nb, uint qo) {
    [unroll] for (uint j = 0; j < 8; j++) gm8[j][li] = abs(v[j]);
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (li < s) [unroll] for (uint j2 = 0; j2 < 8; j2++) gm8[j2][li] = max(gm8[j2][li], gm8[j2][li + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float amax[8];
    [unroll] for (uint j3 = 0; j3 < 8; j3++) {
        amax[j3] = gm8[j3][0];
        const float iscale = amax[j3] > 0 ? 127.0 / amax[j3] : 0;
        gq8[j3][li] = uint(clamp(int(round(v[j3] * iscale)), -127, 127)) & 255;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j4 = 0; j4 < nb; j4++) {
        const uint o = qo + j4 * Q8_BYTES;
        if (li < 64) U.Store(o + Q8_QS + 4 * li, gq8[j4][4 * li] | (gq8[j4][4 * li + 1] << 8) | (gq8[j4][4 * li + 2] << 16) | (gq8[j4][4 * li + 3] << 24));
        if (li == 0) stu(o + Q8_D, amax[j4] > 0 ? 1.0 / (127.0 / amax[j4]) : 0);
    }
    GroupMemoryBarrierWithGroupSync();
}

// x = embedding (mode 0), x += out (mode 1) or x += s * out + y + g (mode 2: out the shared
// expert, y and s from the CPU in IN, g the experts the GPU computed unless NONE); then Q8_K of
// rmsnorm(x) * norm weight, and the same values as floats at h unless NONE. One group per token.
// C0: dim, x, out, q   C1: mode, norm weight, eps, h   C2: y (offset in IN), s (offset in IN), g
[numthreads(256, 1, 1)]
void moe_resid(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, t = gid.y, xo = C0.y + 4 * t * D, oo = C0.z + 4 * t * D, eo = 4 * t * D;
    const uint qo = C0.w + t * (D / 256) * Q8_BYTES, mode = C1.x, nw = C1.y, ho = C1.w + 4 * t * D;
    const float eps = asfloat(C1.z);
    const float g = mode == 2 ? asfloat(IN.Load(C2.y + 4 * t)) : 1;
    float ss = 0;
    for (uint i = li; i < D; i += 256) {
        float v;
        if (mode == 0) v = asfloat(IN.Load(eo + 4 * i));
        else if (mode == 1) v = ldu(xo + 4 * i) + ldu(oo + 4 * i);
        else v = ldu(xo + 4 * i) + g * ldu(oo + 4 * i) + asfloat(IN.Load(C2.x + 4 * (t * D + i))) +
                 (C2.z != NONE ? ldu(C2.z + 4 * (t * D + i)) : 0);
        stu(xo + 4 * i, v);
        ss += v * v;
    }
    const float s = rsqrt(reduce_sum(li, ss) / D + eps);
    for (uint b0 = 0; b0 < D / 256; b0 += 8) {
        float v[8];
        [unroll] for (uint j = 0; j < 8; j++) {
            const uint i = (b0 + j) * 256 + li;
            v[j] = b0 + j < D / 256 ? ldu(xo + 4 * i) * s * ldw(nw + 4 * i) : 0;
            if (C1.w != NONE && b0 + j < D / 256) stu(ho + 4 * i, v[j]);
        }
        quant_blocks8(li, v, min(8u, D / 256 - b0), qo + b0 * Q8_BYTES);
    }
}

// DeltaNet's causal depthwise convolution over the last K inputs, then silu, in place, for the
// n tokens of the chunk in order; the state keeps the last K - 1 inputs (oldest first).
// Groups: 256 channels.  C0: channels, rows (qkv part of the in-projection), row bytes, n
// C1: conv weights [channels][K], state, K
[numthreads(256, 1, 1)]
void delta_conv(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint nc = C0.x, ch = gid.x * 256 + li, n = C0.w, K = C1.z, so = C1.y;
    if (ch >= nc) return;
    const uint w = C1.x + 4 * ch * K;
    float st[7];
    [unroll] for (uint k = 0; k < 7; k++) st[k] = k + 1 < K ? asfloat(CS.Load(so + 4 * (k * nc + ch))) : 0;
    for (uint t = 0; t < n; t++) {
        const uint xo = C0.y + t * C0.z + 4 * ch;
        const float xin = ldu(xo);
        float acc = ldw(w + 4 * (K - 1)) * xin;
        [unroll] for (uint k = 0; k < 7; k++) if (k + 1 < K) acc += ldw(w + 4 * k) * st[k];
        [unroll] for (uint m = 0; m < 6; m++) if (m + 2 < K) st[m] = st[m + 1];
        [unroll] for (uint e = 0; e < 7; e++) if (e + 2 == K) st[e] = xin;
        stu(xo, acc / (1 + exp(-acc)));
    }
    [unroll] for (uint k2 = 0; k2 < 7; k2++) if (k2 + 1 < K) CS.Store(so + 4 * (k2 * nc + ch), asuint(st[k2]));
}

groupshared float dq[128], dk[128];
groupshared float2 d2[512];

// the sums of v over the 512 threads
float2 reduce2_512(uint li, float2 v) {
    d2[li] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 256; s > 0; s >>= 1) {
        if (li < s) d2[li] += d2[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float2 r = d2[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// the gated delta rule of one V head over the n tokens in order: q and k L2-normalized (q also
// by 1/sqrt(d)); S = decay S + k u^T with u = beta (v - decay S^T k); o = S^T q; then RMSNorm of
// o times the norm weight and silu(z). 512 threads: value column c = li % 128 of S, rows
// [32 (li / 128), +32), kept in registers over the tokens.
// Groups: V heads (V head j reads K head j % kh).
// C0: rows (q | k | v), row bytes, z rows, z and out row bytes   C1: alpha rows, beta rows (vh floats), out, n
// C2: S of this layer (in CS), kh, vh, eps   C3: ssm_a, dt bias, norm weight (offsets in SW), 1/sqrt(d)
[numthreads(512, 1, 1)]
void delta_rec(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint d = 128, j = gid.x, kh = C2.y, vh = C2.z, kj = j % kh, n = C1.w, c = li % 128, i0 = (li / 128) * 32;
    const float eps = asfloat(C2.w), qs = asfloat(C3.w);
    const float A = ldw(C3.x + 4 * j), dt = ldw(C3.y + 4 * j), nw = ldw(C3.z + 4 * c);
    const uint so = C2.x + 4 * (j * d * d + i0 * d + c);    // S[i0 + k][c] at so + 4 * k * d
    float S[32];
    [unroll] for (uint k = 0; k < 32; k++) S[k] = asfloat(CS.Load(so + 4 * k * d));
    for (uint t = 0; t < n; t++) {
        const uint row = C0.x + t * C0.y;
        float q = 0, kk = 0;
        if (li < 128) {
            q = ldu(row + 4 * (kj * d + li));
            kk = ldu(row + 4 * (kh * d + kj * d + li));
        }
        const float2 nn = sqrt(reduce2_512(li, float2(q * q, kk * kk)));
        if (li < 128) {
            dq[li] = q / max(nn.x, eps) * qs;
            dk[li] = kk / max(nn.y, eps);
        }
        GroupMemoryBarrierWithGroupSync();
        const float v = ldu(row + 4 * (2 * kh * d + j * d + c));
        const float x = ldu(C1.x + 4 * (t * vh + j)) + dt, beta = ldu(C1.y + 4 * (t * vh + j));
        const float sp = x > 20 ? x : log(1 + exp(x));
        const float decay = exp(sp * A), b = 1 / (1 + exp(-beta));
        float u = 0;
        [unroll] for (uint k1 = 0; k1 < 32; k1++) u += S[k1] * dk[i0 + k1];
        d2[li].x = u;
        GroupMemoryBarrierWithGroupSync();
        u = d2[c].x + d2[c + 128].x + d2[c + 256].x + d2[c + 384].x;
        u = b * (v - decay * u);
        float o = 0;
        [unroll] for (uint k2 = 0; k2 < 32; k2++) {
            S[k2] = decay * S[k2] + dk[i0 + k2] * u;
            o += S[k2] * dq[i0 + k2];
        }
        GroupMemoryBarrierWithGroupSync();
        d2[li].y = o;
        GroupMemoryBarrierWithGroupSync();
        o = d2[c].y + d2[c + 128].y + d2[c + 256].y + d2[c + 384].y;
        GroupMemoryBarrierWithGroupSync();
        const float r = rsqrt(reduce2_512(li, float2(li < 128 ? o * o : 0, 0)).x / d + eps);
        if (li < 128) {
            const float z = ldu(C0.z + t * C0.w + 4 * (j * d + li));
            stu(C1.z + t * C0.w + 4 * (j * d + li), o * r * nw * (z / (1 + exp(-z))));
        }
    }
    [unroll] for (uint k3 = 0; k3 < 32; k3++) CS.Store(so + 4 * k3 * d, asuint(S[k3]));
}

// Qwen attention, per head: q heads [q | gate] -> RMSNorm and NEOX RoPE of the first n_rot
// values of q, sigmoid of the gate; k heads the same into the cache, v heads into the cache.
// Groups: n_head q heads, then n_kv_head k/v heads; tokens.
// C0: hd, q|gate rows (2 qd floats), k|v rows (2 kvd floats), n_head   C1: q norm, k norm, rope (in IN), eps
// C2: k cache offset of this layer and the first token's position, v cache offset, n_rot, q out (qd rows)
// C3: gate out (qd rows), qd, kvd
[numthreads(256, 1, 1)]
void qattn_prep(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x, nh = C0.w, t = gid.y, g = gid.x, qd = C3.y, kvd = C3.z, nr = C2.z / 2;
    const bool isq = g < nh;
    const uint kh = isq ? 0 : g - nh;
    const uint src = isq ? C0.y + 4 * (t * 2 * qd + g * 2 * hd) : C0.z + 4 * (t * 2 * kvd + kh * hd);
    const uint nw = isq ? C1.x : C1.y, ro = C1.z + 4 * t * hd;
    float ss = 0;
    for (uint i = li; i < hd; i += 256) {
        const float v = ldu(src + 4 * i);
        hs[i] = v;
        ss += v * v;
    }
    const float sc = rsqrt(reduce_sum(li, ss) / hd + asfloat(C1.w));
    for (uint i2 = li; i2 < hd; i2 += 256) hs[i2] *= sc * ldw(nw + 4 * i2);
    GroupMemoryBarrierWithGroupSync();
    for (uint r = li; r < nr; r += 256) {
        const float x0 = hs[r], x1 = hs[r + nr];
        const float cs = asfloat(IN.Load(ro + 4 * r)), sn = asfloat(IN.Load(ro + 4 * (hd / 2 + r)));
        hs[r] = x0 * cs - x1 * sn;
        hs[r + nr] = x0 * sn + x1 * cs;
    }
    GroupMemoryBarrierWithGroupSync();
    if (isq) {
        for (uint o = li; o < hd; o += 256) {
            stu(C2.w + 4 * (t * qd + g * hd + o), hs[o]);
            const float gv = ldu(src + 4 * (hd + o));
            stu(C3.x + 4 * (t * qd + g * hd + o), 1 / (1 + exp(-gv)));
        }
    } else {
        const uint ko = C2.x + 2 * t * kvd + 2 * kh * hd, vo = C2.y + 2 * t * kvd + 2 * kh * hd;
        const uint vs = C0.z + 4 * (t * 2 * kvd + kvd + kh * hd);
        for (uint p = li; p < hd / 2; p += 256) {
            KV.Store(ko + 4 * p, f16(hs[2 * p]) | (f16(hs[2 * p + 1]) << 16));
            KV.Store(vo + 4 * p, f16(ldu(vs + 8 * p)) | (f16(ldu(vs + 8 * p + 4)) << 16));
        }
    }
}

// attn_chunk for Qwen: q rows of qd floats, the output times the elementwise gate (qd rows).
// Groups: q heads, tokens.  C0: hd, q, gate, out   C1: k cache of this layer, v cache, kvd, qd
// C2: position of the chunk's first token, q heads per kv head   C3: 1/sqrt(hd)
[numthreads(256, 1, 1)]
void qattn(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x, half = hd / 2, h = gid.x, t = gid.y, kh = h / C2.y, kvd = C1.z, qd = C1.w;
    const uint npos = C2.x + t + 1, j0 = 0;
    const uint kb = C1.x + 2 * kh * hd, vb = C1.y + 2 * kh * hd;
    const float scale = asfloat(C3.x);
    for (uint i = li; i < hd; i += 256) hs[i] = ldu(C0.y + 4 * (t * qd + h * hd + i));
    GroupMemoryBarrierWithGroupSync();
    const uint slices = 256 / half, p = li % half, sl = li / half;
    float mx = -1e30, sum = 0;
    float2 acc = 0;
    for (uint base = j0; base < npos; base += 256) {
        const uint j = min(base + li, npos - 1);
        const uint ko = kb + 2 * j * kvd;
        float dd = 0;
        for (uint q = 0; q < half; q++) {
            const uint w = KV.Load(ko + 4 * q);
            dd += f16tof32(w) * hs[2 * q] + f16tof32(w >> 16) * hs[2 * q + 1];
        }
        const float s = base + li < npos ? dd * scale : -1e30;
        const float m2 = max(mx, reduce_max(li, s)), corr = exp(mx - m2);
        const float e = base + li < npos ? exp(s - m2) : 0;
        gp[li] = e;
        sum = sum * corr + reduce_sum(li, e);
        acc *= corr;
        const uint cnt = min(256u, npos - base);
        for (uint jv = sl; jv < cnt; jv += slices) {
            const uint w2 = KV.Load(vb + 2 * (base + jv) * kvd + 4 * p);
            acc += gp[jv] * float2(f16tof32(w2), f16tof32(w2 >> 16));
        }
        mx = m2;
        GroupMemoryBarrierWithGroupSync();
    }
    g2[li] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (sl == 0) {
        for (uint s2 = 1; s2 < slices; s2++) acc += g2[s2 * half + p];
        const float inv = 1 / sum;
        const uint oo = 4 * (t * qd + h * hd + 2 * p);
        stu(C0.w + oo, acc.x * inv * ldu(C0.z + oo));
        stu(C0.w + oo + 4, acc.y * inv * ldu(C0.z + oo + 4));
    }
}

[[vk::binding(8, 0)]] StructuredBuffer<uint4> SW4 : register(t4);   // SW as 16-byte words
[[vk::binding(0, 0)]] RWStructuredBuffer<uint4> U4 : register(u4);   // U as 16-byte words

// The router's logits (f32 weights in SW): out[t][r] = row r . x_t, or . rmsnorm(x_t) * norm
// weight unless NONE. Groups: 4 rows, 8 tokens; 64 threads per row.
// C0: dim, x, n, out   C1: weights, rows, out row (floats), norm weight   C2: eps
[numthreads(256, 1, 1)]
void router(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, t0 = gid.y * 8, nt = min(8u, C0.z - t0), nw = C1.w, lane = li % 64;
    const uint r = gid.x * 4 + li / 64, wo = C1.x + 4 * min(r, C1.y - 1) * D;
    if (nw != NONE) {
        for (uint tt = 0; tt < nt; tt++) {
            float ss = 0;
            for (uint i = li; i < D; i += 256) {
                const float v = ldu(C0.y + 4 * ((t0 + tt) * D + i));
                ss += v * v;
            }
            const float s = rsqrt(reduce_sum(li, ss) / D + asfloat(C2.x));
            if (li == 0) gp[tt] = s;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float acc[8];
    [unroll] for (uint k = 0; k < 8; k++) acc[k] = 0;
    for (uint i = 4 * lane; i < D; i += 256) {
        float4 w = asfloat(SW4[(wo + 4 * i) >> 4]);
        if (nw != NONE) w *= asfloat(SW4[(nw + 4 * i) >> 4]);
        [unroll] for (uint k = 0; k < 8; k++)
            if (k < nt) acc[k] += dot(w, asfloat(U4[(C0.y + 4 * ((t0 + k) * D + i)) >> 4]));
    }
    for (uint k = 0; k < nt; k++) {
        gs[li] = acc[k];
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint s = 32; s > 0; s >>= 1) {
            if (lane < s) gs[li] += gs[li + s];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0 && r < C1.y) stu(C0.w + 4 * ((t0 + k) * C1.z + r), gs[li] * (nw != NONE ? gp[k] : 1));
        GroupMemoryBarrierWithGroupSync();
    }
}

// the routed experts the GPU computed, weighted, per token: out[t] = sum over k of w * row,
// from a table in IN of K (row, weight) pairs per token (row NONE: none).
// Groups: 256 values, tokens.  C0: dim, rows (dim floats each), out, K   C1: table
[numthreads(256, 1, 1)]
void moe_combine(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, t = gid.y, i = gid.x * 256 + li, K = C0.w;
    float acc = 0;
    for (uint k = 0; k < K; k++) {
        const uint e = C1.x + 8 * (t * K + k), row = IN.Load(e);
        if (row != NONE) acc += asfloat(IN.Load(e + 4)) * ldu(C0.y + 4 * (row * D + i));
    }
    stu(C0.z + 4 * (t * D + i), acc);
}

// ---------------------------------------------------------------- Inkling 2 (Vulkan only)
//
// The slots of inkling2_loop run here (model.c's "Inkling 2" section says what each step is),
// the routed experts on the CPU: after each slot's FFN input the GPU stops with h and the router
// logits read back; the CPU's expert sum comes back through IN. CS is the CPU's conv state
// ([slot][l_cache - 1][channels]), KV its cache layout (window slots in rings).

float2 reduce2_256(uint li, float2 v) {
    g2[li] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (li < s) g2[li] += g2[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float2 r = g2[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// depth attention: out = sum over the states of softmax(rms(state) . query / sqrt(dim)) rms(state);
// the states are the block buffers set in mask (buffer b at blk + b * stride) and cur, row t of
// each at + 4 t dim. Groups: queries, tokens.
// C0: dim, cur, blk, stride   C1: mask, queries (SW, [nq][dim]), out, out bytes per query
[numthreads(256, 1, 1)]
void ik_mix(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, t = gid.y, q = gid.x, mask = C1.x, qo = C1.y + 4 * q * D;
    uint so[17];
    uint ns = 0;
    for (uint b = 0; b < 16; b++)
        if (mask & (1u << b)) so[ns++] = C0.z + b * C0.w + 4 * t * D;
    so[ns++] = C0.y + 4 * t * D;
    float sc[17], inv[17];
    float mx = -1e30;
    for (uint s = 0; s < ns; s++) {
        float ss = 0, dd = 0;
        for (uint i = li; i < D; i += 256) {
            const float v = ldu(so[s] + 4 * i);
            ss += v * v;
            dd += v * ldw(qo + 4 * i);
        }
        const float2 r = reduce2_256(li, float2(ss, dd));
        inv[s] = rsqrt(r.x / D + 1e-6);
        sc[s] = r.y * inv[s] * rsqrt(float(D));
        mx = max(mx, sc[s]);
    }
    float sum = 0;
    for (uint s2 = 0; s2 < ns; s2++) {
        sc[s2] = exp(sc[s2] - mx);
        sum += sc[s2];
    }
    for (uint i2 = li; i2 < D; i2 += 256) {
        float acc = 0;
        for (uint s3 = 0; s3 < ns; s3++) acc += sc[s3] / sum * inv[s3] * ldu(so[s3] + 4 * i2);
        stu(C1.z + q * C1.w + 4 * (t * D + i2), acc);
    }
}

// rmsnorm of rows, times the weight (NONE: none): floats at out (NONE: not), Q8_K at q (NONE: not).
// In place is fine. Groups: tokens.  C0: dim, in, in row bytes, out   C1: out row bytes, q, weight, eps
[numthreads(256, 1, 1)]
void ik_norm(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, t = gid.y, io = C0.y + t * C0.z, oo = C0.w + t * C1.x, nw = C1.z;
    const uint qo = C1.y + t * (D / 256) * Q8_BYTES;
    float ss = 0;
    for (uint i = li; i < D; i += 256) {
        const float v = ldu(io + 4 * i);
        ss += v * v;
    }
    const float s = rsqrt(reduce_sum(li, ss) / D + asfloat(C1.w));
    for (uint b0 = 0; b0 < D / 256; b0 += 8) {
        float v[8];
        [unroll] for (uint j = 0; j < 8; j++) {
            const uint i2 = (b0 + j) * 256 + li;
            v[j] = 0;
            if (b0 + j < D / 256) {
                v[j] = ldu(io + 4 * i2) * s * (nw != NONE ? ldw(nw + 4 * i2) : 1);
                if (C0.w != NONE) stu(oo + 4 * i2, v[j]);
            }
        }
        if (C1.y != NONE) quant_blocks8(li, v, min(8u, D / 256 - b0), qo + b0 * Q8_BYTES);
    }
}

// a loop controller's update: x = x - tanh(forget) x + tanh(gain) msg.  Groups: 256 values, tokens.
// C0: dim, x, gain, forget   C1: msg
[numthreads(256, 1, 1)]
void ik_ctl(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, i = gid.y * D + gid.x * 256 + li;
    if (gid.x * 256 + li >= D) return;
    const float x = ldu(C0.y + 4 * i);
    stu(C0.y + 4 * i, x - tanh(ldu(C0.w + 4 * i)) * x + tanh(ldu(C0.z + 4 * i)) * ldu(C1.x + 4 * i));
}

// dst = a + b over rows of dim floats (a from IN if C1.y, b NONE: none).  Groups: 256 values, tokens.
// C0: dim, dst, a, b   C1: -, a in IN
[numthreads(256, 1, 1)]
void ik_add(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint D = C0.x, c = gid.x * 256 + li, i = gid.y * D + c;
    if (c >= D) return;
    const float a = C1.y ? asfloat(IN.Load(C0.z + 4 * i)) : ldu(C0.z + 4 * i);
    stu(C0.y + 4 * i, a + (C0.w != NONE ? ldu(C0.w + 4 * i) : 0));
}

// causal short convolution with a residual, (x + conv(x)) * scale, in place, for the n tokens of
// the chunk in order; the state keeps the last l_cache - 1 inputs (oldest first, stride apart);
// dyn (NONE: none): per-head taps for the lags 0..l_cache-1 added.  Groups: 256 channels.
// C0: channels, x, x row bytes, n   C1: weights (SW, [channels][l_cache]), state (CS), l_cache, dyn
// C2: dyn row bytes, channels per head, scale, state stride (bytes)
[numthreads(256, 1, 1)]
void ik_conv(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint nc = C0.x, ch = gid.x * 256 + li, n = C0.w, lc = C1.z, so = C1.y + 4 * ch, ss = C2.w;
    if (ch >= nc) return;
    const uint w = C1.x + 4 * ch * lc;
    const float scale = asfloat(C2.z);
    float st[8];
    [unroll] for (uint k = 0; k < 8; k++) st[k] = k + 1 < lc ? asfloat(CS.Load(so + k * ss)) : 0;
    for (uint t = 0; t < n; t++) {
        const uint xo = C0.y + t * C0.z + 4 * ch;
        const float xin = ldu(xo);
        float win[8];
        [unroll] for (uint k1 = 0; k1 < 8; k1++) win[k1] = k1 + 1 < lc ? st[k1] : (k1 + 1 == lc ? xin : 0);
        float acc = 0;
        [unroll] for (uint k2 = 0; k2 < 8; k2++) if (k2 < lc) acc += ldw(w + 4 * k2) * win[k2];
        if (C1.w != NONE) {
            const uint tap = C1.w + t * C2.x + 4 * (ch / C2.y) * lc;
            [unroll] for (uint lag = 0; lag < 8; lag++) if (lag < lc) acc += ldu(tap + 4 * lag) * win[lc - 1 - lag];
        }
        [unroll] for (uint m = 0; m < 7; m++) if (m + 2 < lc) st[m] = st[m + 1];
        [unroll] for (uint e = 0; e < 8; e++) if (e + 2 == lc) st[e] = xin;
        stu(xo, (xin + acc) * scale);
    }
    [unroll] for (uint k3 = 0; k3 < 8; k3++) if (k3 + 1 < lc) CS.Store(so + k3 * ss, asuint(st[k3]));
}

// per head: q (times its clip scale) and r (times its own) L2-clipped and times tau, back in place;
// k L2-clipped and v RMS-normed (eps 1e-6) into the cache at the token's row (ring: pos % R).
// Groups: n_head q heads, n_head r heads, n_kv_head k heads, n_kv_head v heads; tokens.
// C0: q (rows n_head hd), r (rows n_head n_rel), k, v (rows kvd)   C1: k cache of the slot, v cache, ring R (0: none), pos0
// C2: n_head | n_kv_head << 16, hd | n_rel << 16, q clip (SW), r clip (SW)   C3: qk clip, r clip, log alpha (0: no tau), log floor
[numthreads(128, 1, 1)]
void ik_prep(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint nh = C2.x & 0xffff, nkv = C2.x >> 16, hd = C2.y & 0xffff, nr = C2.y >> 16, t = gid.y, g = gid.x;
    const uint kvd = nkv * hd, pos = C1.w + t;
    const float alpha = asfloat(C3.z);
    const float tau = alpha > 0 ? 1 + alpha * log(max(1.0, float(pos + 1) / float(C3.w))) : 1;
    const uint kind = g < nh ? 0 : g < 2 * nh ? 1 : g < 2 * nh + nkv ? 2 : 3;
    const uint h = kind == 0 ? g : kind == 1 ? g - nh : kind == 2 ? g - 2 * nh : g - 2 * nh - nkv;
    const uint len = kind == 1 ? nr : hd;
    const uint src = kind == 0 ? C0.x + 4 * (t * nh * hd + h * hd) : kind == 1 ? C0.y + 4 * (t * nh * nr + h * nr)
                   : kind == 2 ? C0.z + 4 * (t * kvd + h * hd) : C0.w + 4 * (t * kvd + h * hd);
    const uint cw = kind == 0 ? C2.z + 4 * h * hd : C2.w + 4 * h * nr;   // valid for every kind
    float v = 0;
    if (li < len) {
        v = ldu(src + 4 * li);
        if (kind < 2) v *= ldw(cw + 4 * li);
    }
    const float ss = reduce_sum128(li, v * v);
    if (kind == 3) {
        v *= rsqrt(ss / hd + 1e-6);
    } else {
        const float thr = asfloat(kind == 1 ? C3.y : C3.x);
        v *= min(1.0, thr / (sqrt(ss) + 1e-6));
        if (kind < 2) v *= tau;
    }
    if (kind < 2) {
        if (li < len) stu(src + 4 * li, v);
        return;
    }
    hs[li] = v;
    GroupMemoryBarrierWithGroupSync();
    const uint row = C1.z ? pos % C1.z : pos, base = (kind == 2 ? C1.x : C1.y) + 2 * (row * kvd + h * hd);
    if (li < hd / 2) KV.Store(base + 4 * li, f16(hs[2 * li]) | (f16(hs[2 * li + 1]) << 16));
}

groupshared float ikb[1024];   // the relative-position bias of a head, by distance
groupshared float ikr[64];

// one q head of one token per group over its cache positions (ring rows: j % R), in tiles of 256
// with a running softmax; scores q.k scale + bias[distance] (none past the extent), the sink logit
// in the denominator; the output times sigmoid(gate).  Groups: q heads, tokens.
// C0: hd | n_rel << 16, q (rows n_head hd), r (rows n_head n_rel), out (rows n_head hd)
// C1: k cache of the slot, v cache, n_head | n_kv_head << 16, gate (rows n_head hd)
// C2: pos0, window (0: none), ring R (0: none), sinks (rows n_head)   C3: scale, bias (SW, [extent][n_rel]), extent
[numthreads(256, 1, 1)]
void ik_attn(uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    const uint hd = C0.x & 0xffff, nr = C0.x >> 16, half = hd / 2, h = gid.x, t = gid.y;
    const uint nh = C1.z & 0xffff, nkv = C1.z >> 16, kh = h / (nh / nkv), kvd = nkv * hd, qd = nh * hd;
    const uint npos = C2.x + t + 1, j0 = C2.y && npos > C2.y ? npos - C2.y : 0, R = C2.z, ext = C3.z;
    const uint kb = C1.x + 2 * kh * hd, vb = C1.y + 2 * kh * hd;
    const float scale = asfloat(C3.x);
    for (uint i = li; i < hd; i += 256) hs[i] = ldu(C0.y + 4 * (t * qd + h * hd + i));
    if (li < nr) ikr[li] = ldu(C0.z + 4 * (t * nh * nr + h * nr + li));
    GroupMemoryBarrierWithGroupSync();
    const uint nd = min(npos - j0, ext);
    for (uint d = li; d < nd; d += 256) {
        float b = 0;
        for (uint r = 0; r < nr; r++) b += ldw(C3.y + 4 * (d * nr + r)) * ikr[r];
        ikb[d] = b;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint slices = 256 / half, p = li % half, sl = li / half;
    float mx = ldu(C2.w + 4 * (t * nh + h)), sum = 1;   // the sink: exp(sink - mx) with mx = sink
    float2 acc = 0;
    for (uint base = j0; base < npos; base += 256) {
        const uint j = min(base + li, npos - 1), dist = npos - 1 - j;
        const uint ko = kb + 2 * (R ? j % R : j) * kvd;
        float dd = 0;
        for (uint q = 0; q < half; q++) {
            const uint w = KV.Load(ko + 4 * q);
            dd += f16tof32(w) * hs[2 * q] + f16tof32(w >> 16) * hs[2 * q + 1];
        }
        const float s = base + li < npos ? dd * scale + (dist < ext ? ikb[min(dist, 1023u)] : 0) : -1e30;
        const float m2 = max(mx, reduce_max(li, s)), corr = exp(mx - m2);
        const float e = base + li < npos ? exp(s - m2) : 0;
        gp[li] = e;
        sum = sum * corr + reduce_sum(li, e);
        acc *= corr;
        const uint cnt = min(256u, npos - base);
        for (uint jv = sl; jv < cnt; jv += slices) {
            const uint jj = base + jv;
            const uint w2 = KV.Load(vb + 2 * (R ? jj % R : jj) * kvd + 4 * p);
            acc += gp[jv] * float2(f16tof32(w2), f16tof32(w2 >> 16));
        }
        mx = m2;
        GroupMemoryBarrierWithGroupSync();
    }
    g2[li] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (sl == 0) {
        for (uint s2 = 1; s2 < slices; s2++) acc += g2[s2 * half + p];
        const float inv = 1 / sum;
        const uint oo = 4 * (t * qd + h * hd + 2 * p);
        stu(C0.w + oo, acc.x * inv / (1 + exp(-ldu(C1.w + oo))));
        stu(C0.w + oo + 4, acc.y * inv / (1 + exp(-ldu(C1.w + oo + 4))));
    }
}
#endif
