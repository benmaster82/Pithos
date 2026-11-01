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
#define Q8_BYTES 292

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
    if (li < 64) U.Store(qo + 4 + 4 * li, gq[4 * li] | (gq[4 * li + 1] << 8) | (gq[4 * li + 2] << 16) | (gq[4 * li + 3] << 24));
    if (li == 0) stu(qo, amax > 0 ? 1.0 / iscale : 0);
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
    const uint npos = C2.x + t + 1, j0 = C2.y && npos > C2.y ? npos - C2.y : 0;
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
