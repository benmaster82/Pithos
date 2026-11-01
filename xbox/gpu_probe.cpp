// gpu_probe.cpp - measures what the Xbox GPU offers to this app: Direct3D feature level, memory,
// compute throughput and memory bandwidth. Runs when LocalState\gpu-probe exists and writes to
// the log; it decides whether a GPU path for prompt processing is worth building.
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi.h>
#include <stdio.h>
#include <winrt/base.h>
#include "gpu_probe_cs.h"   // fxc output of gpu_probe.hlsl: g_flops50, g_copy50, g_flops40, g_copy40

using winrt::com_ptr;

// GPU time of `runs` dispatches in seconds (timestamp queries), or -1.
static double time_dispatch(ID3D11Device *dev, ID3D11DeviceContext *ctx, UINT groups, int runs) {
    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    com_ptr<ID3D11Query> disjoint, t0, t1;
    if (FAILED(dev->CreateQuery(&qd, disjoint.put()))) return -1;
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, t0.put());
    dev->CreateQuery(&qd, t1.put());
    ctx->Dispatch(groups, 1, 1);   // warm-up
    ctx->Begin(disjoint.get());
    ctx->End(t0.get());
    for (int i = 0; i < runs; i++) ctx->Dispatch(groups, 1, 1);
    ctx->End(t1.get());
    ctx->End(disjoint.get());
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
    UINT64 a = 0, b = 0;
    while (ctx->GetData(disjoint.get(), &dj, sizeof(dj), 0) == S_FALSE) Sleep(1);
    while (ctx->GetData(t0.get(), &a, sizeof(a), 0) == S_FALSE) Sleep(1);
    while (ctx->GetData(t1.get(), &b, sizeof(b), 0) == S_FALSE) Sleep(1);
    if (dj.Disjoint || !dj.Frequency) return -1;
    return (double)(b - a) / (double)dj.Frequency;
}

// Times the two probe shaders on a Direct3D 12 queue with GPU timestamps.
static void bench_d3d12(ID3D12Device *dev, HMODULE lib) {
    auto serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)GetProcAddress(lib, "D3D12SerializeRootSignature");
    D3D12_ROOT_PARAMETER rp[2] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;   // u0
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;   // t0
    rp[0].ShaderVisibility = rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd = { 2, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
    com_ptr<ID3DBlob> blob, err;
    com_ptr<ID3D12RootSignature> rs;
    if (!serialize || FAILED(serialize(&rd, D3D_ROOT_SIGNATURE_VERSION_1, blob.put(), err.put())) ||
        FAILED(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(rs.put())))) {
        fprintf(stderr, "gpu-probe: d3d12 root signature failed\n");
        return;
    }
    auto pipeline = [&](const BYTE *cs, size_t n) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
        d.pRootSignature = rs.get();
        d.CS = { cs, n };
        com_ptr<ID3D12PipelineState> p;
        dev->CreateComputePipelineState(&d, IID_PPV_ARGS(p.put()));
        return p;
    };
    com_ptr<ID3D12PipelineState> flops = pipeline(g_flops50, sizeof(g_flops50)), copy = pipeline(g_copy50, sizeof(g_copy50));

    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    com_ptr<ID3D12CommandQueue> queue;
    com_ptr<ID3D12CommandAllocator> alloc;
    com_ptr<ID3D12GraphicsCommandList> cl;
    com_ptr<ID3D12Fence> fence;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()));
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.get(), nullptr, IID_PPV_ARGS(cl.put()));
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()));
    if (!flops || !copy || !cl || !fence) {
        fprintf(stderr, "gpu-probe: d3d12 pipeline or queue creation failed\n");
        return;
    }

    // two 64 MB buffers of float4, timestamps read back through a small buffer
    const UINT n = 1u << 22;
    D3D12_HEAP_PROPERTIES def = { D3D12_HEAP_TYPE_DEFAULT }, rb = { D3D12_HEAP_TYPE_READBACK };
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = (UINT64)n * 16;
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    com_ptr<ID3D12Resource> src, dst, times;
    dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(src.put()));
    dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(dst.put()));
    bd.Width = 64;
    bd.Flags = D3D12_RESOURCE_FLAG_NONE;
    dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(times.put()));
    D3D12_QUERY_HEAP_DESC qh = { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4 };
    com_ptr<ID3D12QueryHeap> heap;
    dev->CreateQueryHeap(&qh, IID_PPV_ARGS(heap.put()));
    if (!src || !dst || !times || !heap) {
        fprintf(stderr, "gpu-probe: d3d12 cannot allocate 2 x 64 MB of GPU buffers\n");
        return;
    }

    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // all UAV work before it has finished
    const UINT groups = 8192;
    cl->SetComputeRootSignature(rs.get());
    cl->SetComputeRootUnorderedAccessView(0, dst->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(1, src->GetGPUVirtualAddress());
    cl->SetPipelineState(flops.get());
    cl->Dispatch(groups, 1, 1);   // warm-up
    cl->ResourceBarrier(1, &uav);
    cl->EndQuery(heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    for (int i = 0; i < 5; i++) cl->Dispatch(groups, 1, 1);
    cl->ResourceBarrier(1, &uav);
    cl->EndQuery(heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    cl->SetPipelineState(copy.get());
    cl->Dispatch(n / 256, 1, 1);
    cl->ResourceBarrier(1, &uav);
    cl->EndQuery(heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
    for (int i = 0; i < 10; i++) cl->Dispatch(n / 256, 1, 1);
    cl->ResourceBarrier(1, &uav);
    cl->EndQuery(heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
    cl->ResolveQueryData(heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, times.get(), 0);
    cl->Close();
    ID3D12CommandList *lists[] = { cl.get() };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.get(), 1);
    HANDLE ev = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    fence->SetEventOnCompletion(1, ev);
    if (WaitForSingleObjectEx(ev, 60000, FALSE) != WAIT_OBJECT_0) {
        fprintf(stderr, "gpu-probe: d3d12 work did not finish\n");
        CloseHandle(ev);
        return;
    }
    CloseHandle(ev);
    UINT64 freq = 0, *t = nullptr;
    queue->GetTimestampFrequency(&freq);
    D3D12_RANGE r = { 0, 32 };
    if (FAILED(times->Map(0, &r, (void **)&t)) || !freq) return;
    double tf = (double)(t[1] - t[0]) / freq, tc = (double)(t[3] - t[2]) / freq;
    times->Unmap(0, nullptr);
    double gflop = 5.0 * groups * 256 * 1024 * 32 / 1e9;
    fprintf(stderr, "gpu-probe: d3d12 compute %.0f GFLOPS (%.1f GFLOP in %.1f ms)\n", gflop / tf, gflop, tf * 1e3);
    fprintf(stderr, "gpu-probe: d3d12 memory %.1f GB/s (read + write of 64 MB, 10 times in %.1f ms)\n",
            10.0 * 2 * n * 16 / tc / 1e9, tc * 1e3);
}

// Direct3D 12 (games get it on Xbox, apps may not): device creation and highest feature level.
static void probe_d3d12(void) {
    // the app API set only has LoadPackagedLibrary (DLLs inside the package); the system copy of
    // d3d12.dll is loaded through kernelbase's LoadLibraryExW, so a missing DLL is not fatal
    typedef HMODULE(WINAPI * LoadEx)(LPCWSTR, HANDLE, DWORD);
    HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
    LoadEx load = kb ? (LoadEx)GetProcAddress(kb, "LoadLibraryExW") : nullptr;
    HMODULE lib = load ? load(L"d3d12.dll", nullptr, 0x00000800 /* LOAD_LIBRARY_SEARCH_SYSTEM32 */) : nullptr;
    auto create = lib ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(lib, "D3D12CreateDevice") : nullptr;
    if (!create) {
        fprintf(stderr, "gpu-probe: Direct3D 12 not available to this app\n");
        return;
    }
    com_ptr<ID3D12Device> dev;
    HRESULT hr = create(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), dev.put_void());
    if (FAILED(hr)) {
        fprintf(stderr, "gpu-probe: D3D12CreateDevice failed (0x%08lx)\n", (unsigned long)hr);
        return;
    }
    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1 };
    D3D12_FEATURE_DATA_FEATURE_LEVELS fl = { 4, levels, D3D_FEATURE_LEVEL_11_0 };
    dev->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &fl, sizeof(fl));
    D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_0 };
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)))) sm.HighestShaderModel = D3D_SHADER_MODEL_5_1;
    fprintf(stderr, "gpu-probe: Direct3D 12 OK | feature level %x.%x | shader model %x.%x\n", (fl.MaxSupportedFeatureLevel >> 12) & 0xf,
            (fl.MaxSupportedFeatureLevel >> 8) & 0xf, (sm.HighestShaderModel >> 4) & 0xf, sm.HighestShaderModel & 0xf);
    bench_d3d12(dev.get(), lib);
}

void gpu_probe(void) {
    probe_d3d12();
    const D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1,
                                       D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    com_ptr<ID3D11Device> dev;
    com_ptr<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = {};
    HRESULT hr = E_FAIL;
    for (int skip = 0; skip < 3 && FAILED(hr); skip++)   // older runtimes reject the 12_x levels
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, want + skip, (UINT)(6 - skip),
                               D3D11_SDK_VERSION, dev.put(), &fl, ctx.put());
    if (FAILED(hr)) {
        fprintf(stderr, "gpu-probe: D3D11CreateDevice failed (0x%08lx)\n", (unsigned long)hr);
        return;
    }
    com_ptr<IDXGIDevice> dxgi = dev.as<IDXGIDevice>();
    com_ptr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC ad = {};
    if (SUCCEEDED(dxgi->GetAdapter(adapter.put()))) adapter->GetDesc(&ad);
    char name[128];
    WideCharToMultiByte(CP_UTF8, 0, ad.Description, -1, name, sizeof(name), nullptr, nullptr);
    fprintf(stderr, "gpu-probe: %s | feature level %x.%x | dedicated %.0f MB, shared %.0f MB\n", name,
            (fl >> 12) & 0xf, (fl >> 8) & 0xf, ad.DedicatedVideoMemory / 1048576.0, ad.SharedSystemMemory / 1048576.0);

    bool cs5 = fl >= D3D_FEATURE_LEVEL_11_0;
    if (!cs5) {
        D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS o = {};
        dev->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &o, sizeof(o));
        fprintf(stderr, "gpu-probe: compute shaders 4.x: %s\n", o.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x ? "yes" : "no");
        if (!o.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x) return;
    }
    com_ptr<ID3D11ComputeShader> flops, copy;
    if (FAILED(dev->CreateComputeShader(cs5 ? g_flops50 : g_flops40, cs5 ? sizeof(g_flops50) : sizeof(g_flops40), nullptr, flops.put())) ||
        FAILED(dev->CreateComputeShader(cs5 ? g_copy50 : g_copy40, cs5 ? sizeof(g_copy50) : sizeof(g_copy40), nullptr, copy.put()))) {
        fprintf(stderr, "gpu-probe: CreateComputeShader failed\n");
        return;
    }

    // two 64 MB structured buffers of float4
    const UINT n = 1u << 22;
    D3D11_BUFFER_DESC bd = { n * 16, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0,
                             D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
    com_ptr<ID3D11Buffer> bsrc, bdst;
    if (FAILED(dev->CreateBuffer(&bd, nullptr, bsrc.put())) || FAILED(dev->CreateBuffer(&bd, nullptr, bdst.put()))) {
        fprintf(stderr, "gpu-probe: cannot allocate 2 x 64 MB of GPU buffers\n");
        return;
    }
    com_ptr<ID3D11ShaderResourceView> srv;
    com_ptr<ID3D11UnorderedAccessView> uav;
    dev->CreateShaderResourceView(bsrc.get(), nullptr, srv.put());
    dev->CreateUnorderedAccessView(bdst.get(), nullptr, uav.put());
    ID3D11UnorderedAccessView *uavs[] = { uav.get() };
    ID3D11ShaderResourceView *srvs[] = { srv.get() };
    ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    ctx->CSSetShaderResources(0, 1, srvs);

    ctx->CSSetShader(flops.get(), nullptr, 0);
    const UINT groups = 8192;
    double t = time_dispatch(dev.get(), ctx.get(), groups, 5);
    double gflop = 5.0 * groups * 256 * 1024 * 32 / 1e9;
    fprintf(stderr, "gpu-probe: compute %.0f GFLOPS (%.1f GFLOP in %.1f ms, cs_%s)\n", t > 0 ? gflop / t : 0, gflop,
            t * 1e3, cs5 ? "5_0" : "4_0");

    ctx->CSSetShader(copy.get(), nullptr, 0);
    t = time_dispatch(dev.get(), ctx.get(), n / 256, 10);
    fprintf(stderr, "gpu-probe: memory %.1f GB/s (read + write of 64 MB, 10 times in %.1f ms)\n",
            t > 0 ? 10.0 * 2 * n * 16 / t / 1e9 : 0, t * 1e3);
}
