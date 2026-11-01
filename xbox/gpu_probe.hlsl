// gpu_probe.hlsl - two micro-benchmarks for gpu_probe.cpp, built for cs_5_0 and cs_4_0 by build.cmd.

RWStructuredBuffer<float4> dst : register(u0);
StructuredBuffer<float4> src : register(t0);

// 4 independent chains of float4 multiply-add: 1024 iterations x 32 flops per thread
[numthreads(256, 1, 1)]
void flops(uint3 id : SV_DispatchThreadID) {
    float4 a = float4(id.x, 1, 2, 3) * 1e-6, d = a + 1, e = a + 2, f = a + 3;
    const float4 b = float4(0.999, 0.998, 0.997, 0.996), c = float4(1e-3, 2e-3, 3e-3, 4e-3);
    [loop]
    for (int i = 0; i < 1024; i++) {
        a = a * b + c;
        d = d * b + c;
        e = e * b + c;
        f = f * b + c;
    }
    dst[id.x] = a + d + e + f;
}

// memory bandwidth: read 16 bytes and write 16 bytes per thread
[numthreads(256, 1, 1)]
void copy(uint3 id : SV_DispatchThreadID) {
    dst[id.x] = src[id.x];
}
