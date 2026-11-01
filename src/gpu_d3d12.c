/* gpu_d3d12.c - Direct3D 12 compute backend for prompt matmuls (see gpu_dx.h).
 *
 * Every weight matrix lives in its own GPU buffer (Q6_K blocks are repacked from 210 to 212
 * bytes so that the shaders read aligned words). dx_matmul copies the Q8_K activations into a
 * GPU buffer, dispatches one kernel per matrix (kernels_dx.hlsl: 64 rows x 64 tokens per
 * group) and reads the results back, all in one command list. Buffers rely on the implicit
 * state promotion from COMMON at the start of each command list and decay back after it. */
#include "gpu_dx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)

Dx *dx_create(const Model *m, char *info, int infolen) {
    (void)m;
    snprintf(info, (size_t)infolen, "Direct3D 12 needs Windows");
    return NULL;
}
void dx_destroy(Dx *d) { (void)d; }
int dx_has(const Dx *d, const uint8_t *w) { (void)d; (void)w; return 0; }
int dx_matmul(Dx *d, const DxJob *jobs, int n) { (void)d; (void)jobs; (void)n; return 0; }

#else

#define COBJMACROS
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include "kernels_dx.h"

/* local copies of the interface ids: no dxguid library needed */
static const GUID DX_IID_Device = {0x189819f1, 0x1db6, 0x4b57, {0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7}};
static const GUID DX_IID_CommandQueue = {0x0ec870a6, 0x5d7e, 0x4c22, {0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed}};
static const GUID DX_IID_CommandAllocator = {0x6102dee4, 0xaf59, 0x4b09, {0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24}};
static const GUID DX_IID_GraphicsCommandList = {0x5b160d0f, 0xac1b, 0x4185, {0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55}};
static const GUID DX_IID_Fence = {0x0a753dcf, 0xc4d8, 0x4b91, {0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76}};
static const GUID DX_IID_Resource = {0x696442be, 0xa72e, 0x4059, {0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad}};
static const GUID DX_IID_RootSignature = {0xc54a6b66, 0x72df, 0x4ee8, {0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14}};
static const GUID DX_IID_PipelineState = {0x765a30f3, 0xf624, 0x4c6f, {0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45}};
static const GUID DX_IID_Factory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};

#define Q8_BYTES 292        /* sizeof(block_q8_K) */
#define Q6K_GPU_BYTES 212   /* repacked q6_K block */
#define STAGE_BYTES (64u << 20)

enum { K_Q4_K, K_Q6_K, N_KERNELS };

typedef struct {
    const uint8_t *host;
    int kernel, rows, K;
    size_t row_bytes, gpu_row;      /* bytes per row in the file / on the GPU */
    ID3D12Resource *res;
    D3D12_GPU_VIRTUAL_ADDRESS va;
} DxMat;

typedef struct {
    ID3D12Resource *res;
    size_t size;
    uint8_t *map;                   /* upload / readback heaps stay mapped */
} DxBuf;

struct Dx {
    HMODULE lib;
    ID3D12Device *dev;
    ID3D12CommandQueue *queue;
    ID3D12CommandAllocator *alloc;
    ID3D12GraphicsCommandList *cl;
    ID3D12Fence *fence;
    HANDLE event;
    UINT64 fence_val;
    ID3D12RootSignature *rs;
    ID3D12PipelineState *pso[N_KERNELS];
    DxMat *mats;                    /* sorted by host address */
    int nmats, broken;
    DxBuf up, x, o, rb;             /* upload, activations, results, readback */
};

typedef HRESULT(WINAPI *CreateDevice_t)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
typedef HRESULT(WINAPI *CreateFactory_t)(REFIID, void **);

/* System DLLs: an app container (Xbox) has no LoadLibrary, but kernelbase exports LoadLibraryExW */
static HMODULE load_system_dll(const wchar_t *name) {
    typedef HMODULE(WINAPI * LoadEx_t)(LPCWSTR, HANDLE, DWORD);
    HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
    LoadEx_t load = kb ? (LoadEx_t)(void *)GetProcAddress(kb, "LoadLibraryExW") : NULL;
    return load ? load(name, NULL, 0x800 /* LOAD_LIBRARY_SEARCH_SYSTEM32 */) : NULL;
}

static ID3D12Resource *buffer(Dx *d, D3D12_HEAP_TYPE heap, size_t size, D3D12_RESOURCE_STATES state, int uav) {
    D3D12_HEAP_PROPERTIES hp;
    memset(&hp, 0, sizeof(hp));
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource *r = NULL;
    if (FAILED(ID3D12Device_CreateCommittedResource(d->dev, &hp, D3D12_HEAP_FLAG_NONE, &rd, state, NULL,
                                                    &DX_IID_Resource, (void **)&r)))
        return NULL;
    return r;
}

static void buf_free(DxBuf *b) {
    if (b->res) ID3D12Resource_Release(b->res);
    memset(b, 0, sizeof(*b));
}

/* (re)create b with at least `size` bytes; heap UPLOAD / READBACK buffers get mapped */
static int buf_reserve(Dx *d, DxBuf *b, D3D12_HEAP_TYPE heap, size_t size) {
    if (b->res && b->size >= size) return 1;
    buf_free(b);
    size = (size + 0xFFFFF) & ~(size_t)0xFFFFF;
    D3D12_RESOURCE_STATES st = heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                             : heap == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                : D3D12_RESOURCE_STATE_COMMON;
    b->res = buffer(d, heap, size, st, heap == D3D12_HEAP_TYPE_DEFAULT);
    if (!b->res) return 0;
    b->size = size;
    if (heap != D3D12_HEAP_TYPE_DEFAULT) {
        D3D12_RANGE none = {0, 0};
        void *p = NULL;
        if (FAILED(ID3D12Resource_Map(b->res, 0, heap == D3D12_HEAP_TYPE_READBACK ? NULL : &none, &p))) {
            buf_free(b);
            return 0;
        }
        b->map = (uint8_t *)p;
    }
    return 1;
}

static void barrier(Dx *d, ID3D12Resource *r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b;
    memset(&b, 0, sizeof(b));
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    ID3D12GraphicsCommandList_ResourceBarrier(d->cl, 1, &b);
}

/* close the command list, run it, wait, and open it again */
static int submit(Dx *d) {
    if (FAILED(ID3D12GraphicsCommandList_Close(d->cl))) return 0;
    ID3D12CommandList *lists[1] = {(ID3D12CommandList *)d->cl};
    ID3D12CommandQueue_ExecuteCommandLists(d->queue, 1, lists);
    const UINT64 v = ++d->fence_val;
    if (FAILED(ID3D12CommandQueue_Signal(d->queue, d->fence, v))) return 0;
    if (ID3D12Fence_GetCompletedValue(d->fence) < v) {
        if (FAILED(ID3D12Fence_SetEventOnCompletion(d->fence, v, d->event))) return 0;
        WaitForSingleObjectEx(d->event, INFINITE, FALSE);
    }
    if (ID3D12Fence_GetCompletedValue(d->fence) == UINT64_MAX) return 0;   /* device removed */
    if (FAILED(ID3D12CommandAllocator_Reset(d->alloc))) return 0;
    return SUCCEEDED(ID3D12GraphicsCommandList_Reset(d->cl, d->alloc, NULL));
}

static void add_mat(Dx *d, const Mat *w, int *cap) {
    if (!w->data || w->cols % QK_K) return;
    int kernel = w->type == GGML_Q4_K ? K_Q4_K : w->type == GGML_Q6_K ? K_Q6_K : -1;
    if (kernel < 0) return;
    if (d->nmats == *cap) {
        *cap = *cap ? *cap * 2 : 64;
        d->mats = (DxMat *)realloc(d->mats, sizeof(DxMat) * (size_t)*cap);
    }
    DxMat *x = &d->mats[d->nmats++];
    memset(x, 0, sizeof(*x));
    x->host = w->data;
    x->kernel = kernel;
    x->rows = w->rows;
    x->K = w->cols;
    x->row_bytes = w->row_bytes;
    x->gpu_row = kernel == K_Q6_K ? (size_t)(w->cols / QK_K) * Q6K_GPU_BYTES : w->row_bytes;
}

static int mat_cmp(const void *a, const void *b) {
    const uint8_t *x = ((const DxMat *)a)->host, *y = ((const DxMat *)b)->host;
    return x < y ? -1 : x > y;
}

static const DxMat *find_mat(const Dx *d, const uint8_t *w) {
    int lo = 0, hi = d->nmats - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (d->mats[mid].host == w) return &d->mats[mid];
        if (d->mats[mid].host < w) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

/* file layout -> GPU layout of one matrix */
static void pack_mat(const DxMat *x, uint8_t *dst) {
    if (x->kernel != K_Q6_K) {
        memcpy(dst, x->host, x->row_bytes * (size_t)x->rows);
        return;
    }
    const size_t nblk = (size_t)x->rows * (size_t)(x->K / QK_K);
    const block_q6_K *b = (const block_q6_K *)x->host;
    for (size_t i = 0; i < nblk; i++, dst += Q6K_GPU_BYTES) {
        memcpy(dst, &b[i].d, 2);
        dst[2] = dst[3] = 0;
        memcpy(dst + 4, b[i].ql, 128);
        memcpy(dst + 132, b[i].qh, 64);
        memcpy(dst + 196, b[i].scales, 16);
    }
}

static int upload_weights(Dx *d, char *err, int errlen) {
    if (!buf_reserve(d, &d->up, D3D12_HEAP_TYPE_UPLOAD, STAGE_BYTES)) {
        snprintf(err, (size_t)errlen, "cannot allocate the upload buffer");
        return 0;
    }
    size_t used = 0;
    for (int i = 0; i < d->nmats; i++) {
        DxMat *x = &d->mats[i];
        const size_t bytes = x->gpu_row * (size_t)x->rows;
        if (bytes > d->up.size) {
            snprintf(err, (size_t)errlen, "matrix of %zu MB is larger than the upload buffer", bytes >> 20);
            return 0;
        }
        x->res = buffer(d, D3D12_HEAP_TYPE_DEFAULT, bytes, D3D12_RESOURCE_STATE_COMMON, 0);
        if (!x->res) {
            snprintf(err, (size_t)errlen, "out of GPU memory after %d of %d matrices", i, d->nmats);
            return 0;
        }
        x->va = ID3D12Resource_GetGPUVirtualAddress(x->res);
        if (used + bytes > d->up.size) {
            if (!submit(d)) { snprintf(err, (size_t)errlen, "weight upload failed"); return 0; }
            used = 0;
        }
        pack_mat(x, d->up.map + used);
        ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, x->res, 0, d->up.res, used, bytes);
        used = (used + bytes + 255) & ~(size_t)255;
    }
    if (!submit(d)) { snprintf(err, (size_t)errlen, "weight upload failed"); return 0; }
    buf_free(&d->up);   /* dx_matmul allocates what it needs */
    return 1;
}

static int create_pipeline(Dx *d, HMODULE lib) {
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize =
        (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)GetProcAddress(lib, "D3D12SerializeRootSignature");
    if (!serialize) return 0;
    D3D12_ROOT_PARAMETER rp[4];
    memset(rp, 0, sizeof(rp));
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;   /* b0 */
    rp[0].Constants.Num32BitValues = 6;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;               /* t0: weights */
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;               /* t1: activations */
    rp[2].Descriptor.ShaderRegister = 1;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;               /* u0: results */
    D3D12_ROOT_SIGNATURE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    rd.NumParameters = 4;
    rd.pParameters = rp;
    ID3DBlob *blob = NULL, *msg = NULL;
    if (FAILED(serialize(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &msg))) {
        if (msg) ID3D10Blob_Release(msg);
        return 0;
    }
    HRESULT hr = ID3D12Device_CreateRootSignature(d->dev, 0, ID3D10Blob_GetBufferPointer(blob),
                                                  ID3D10Blob_GetBufferSize(blob), &DX_IID_RootSignature, (void **)&d->rs);
    ID3D10Blob_Release(blob);
    if (FAILED(hr)) return 0;
    const BYTE *code[N_KERNELS] = {g_mm_q4_K, g_mm_q6_K};
    const size_t size[N_KERNELS] = {sizeof(g_mm_q4_K), sizeof(g_mm_q6_K)};
    for (int k = 0; k < N_KERNELS; k++) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd;
        memset(&pd, 0, sizeof(pd));
        pd.pRootSignature = d->rs;
        pd.CS.pShaderBytecode = code[k];
        pd.CS.BytecodeLength = size[k];
        if (FAILED(ID3D12Device_CreateComputePipelineState(d->dev, &pd, &DX_IID_PipelineState, (void **)&d->pso[k])))
            return 0;
    }
    return 1;
}

/* the adapter with the most dedicated memory (a laptop's discrete GPU rather than the integrated one) */
static IDXGIAdapter1 *pick_adapter(char *name, int namelen) {
    HMODULE dxgi = load_system_dll(L"dxgi.dll");
    CreateFactory_t create = dxgi ? (CreateFactory_t)(void *)GetProcAddress(dxgi, "CreateDXGIFactory1") : NULL;
    IDXGIFactory1 *f = NULL;
    if (!create || FAILED(create(&DX_IID_Factory1, (void **)&f))) return NULL;
    IDXGIAdapter1 *best = NULL, *a = NULL;
    SIZE_T best_mem = 0;
    for (UINT i = 0; IDXGIFactory1_EnumAdapters1(f, i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!best || desc.DedicatedVideoMemory > best_mem)) {
            if (best) IDXGIAdapter1_Release(best);
            best = a;
            best_mem = desc.DedicatedVideoMemory;
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, namelen, NULL, NULL);
        } else {
            IDXGIAdapter1_Release(a);
        }
    }
    IDXGIFactory1_Release(f);
    return best;
}

Dx *dx_create(const Model *m, char *info, int infolen) {
    info[0] = 0;
    HMODULE lib = load_system_dll(L"d3d12.dll");
    CreateDevice_t create = lib ? (CreateDevice_t)(void *)GetProcAddress(lib, "D3D12CreateDevice") : NULL;
    if (!create) {
        snprintf(info, (size_t)infolen, "d3d12.dll not available");
        return NULL;
    }
    if (getenv("PITHOS_DX_DEBUG")) {
        typedef HRESULT(WINAPI * GetDebug_t)(REFIID, void **);
        static const GUID iid_debug = {0x344488b7, 0x6846, 0x474b, {0xb9, 0x89, 0xf0, 0x27, 0x44, 0x82, 0x45, 0xe0}};
        GetDebug_t get = (GetDebug_t)(void *)GetProcAddress(lib, "D3D12GetDebugInterface");
        ID3D12Debug *dbg = NULL;
        if (get && SUCCEEDED(get(&iid_debug, (void **)&dbg))) {
            ID3D12Debug_EnableDebugLayer(dbg);
            ID3D12Debug_Release(dbg);
        }
    }
    Dx *d = (Dx *)calloc(1, sizeof(Dx));
    d->lib = lib;
    char name[128] = "default adapter";
    IDXGIAdapter1 *adapter = pick_adapter(name, sizeof(name));
    HRESULT hr = create((IUnknown *)adapter, D3D_FEATURE_LEVEL_11_0, &DX_IID_Device, (void **)&d->dev);
    if (adapter) IDXGIAdapter1_Release(adapter);
    if (FAILED(hr)) {
        snprintf(info, (size_t)infolen, "no Direct3D 12 device (0x%08lx)%s", (unsigned long)hr,
                 hr == (HRESULT)0x887a0004 ? ": on Xbox, set the app type to Game" : "");
        free(d);
        return NULL;
    }
    D3D12_COMMAND_QUEUE_DESC qd;
    memset(&qd, 0, sizeof(qd));
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    d->event = CreateEventExW(NULL, NULL, 0, EVENT_ALL_ACCESS);
    if (FAILED(ID3D12Device_CreateCommandQueue(d->dev, &qd, &DX_IID_CommandQueue, (void **)&d->queue)) ||
        FAILED(ID3D12Device_CreateCommandAllocator(d->dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &DX_IID_CommandAllocator,
                                                   (void **)&d->alloc)) ||
        FAILED(ID3D12Device_CreateCommandList(d->dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, d->alloc, NULL,
                                              &DX_IID_GraphicsCommandList, (void **)&d->cl)) ||
        FAILED(ID3D12Device_CreateFence(d->dev, 0, D3D12_FENCE_FLAG_NONE, &DX_IID_Fence, (void **)&d->fence)) ||
        !d->event || !create_pipeline(d, lib)) {
        snprintf(info, (size_t)infolen, "Direct3D 12 setup failed on %s", name);
        dx_destroy(d);
        return NULL;
    }

    int cap = 0;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        const Mat *ws[] = {&L->in_proj, &L->out_proj, &L->wq, &L->wk, &L->wv, &L->wo, &L->attn_gate,
                           &L->w_gate, &L->w_up, &L->w_down};
        for (size_t i = 0; i < sizeof(ws) / sizeof(ws[0]); i++) add_mat(d, ws[i], &cap);
    }
    if (!d->nmats) {
        snprintf(info, (size_t)infolen, "no Q4_K / Q6_K dense weights in this model");
        dx_destroy(d);
        return NULL;
    }
    qsort(d->mats, (size_t)d->nmats, sizeof(DxMat), mat_cmp);
    char err[256];
    if (!upload_weights(d, err, sizeof(err))) {
        snprintf(info, (size_t)infolen, "%s: %s", name, err);
        dx_destroy(d);
        return NULL;
    }
    size_t total = 0;
    for (int i = 0; i < d->nmats; i++) total += d->mats[i].gpu_row * (size_t)d->mats[i].rows;
    snprintf(info, (size_t)infolen, "%s | %d matrices, %.0f MB of weights", name, d->nmats, total / 1048576.0);
    return d;
}

void dx_destroy(Dx *d) {
    if (!d) return;
    for (int i = 0; i < d->nmats; i++)
        if (d->mats[i].res) ID3D12Resource_Release(d->mats[i].res);
    free(d->mats);
    buf_free(&d->up);
    buf_free(&d->x);
    buf_free(&d->o);
    buf_free(&d->rb);
    for (int k = 0; k < N_KERNELS; k++)
        if (d->pso[k]) ID3D12PipelineState_Release(d->pso[k]);
    if (d->rs) ID3D12RootSignature_Release(d->rs);
    if (d->fence) ID3D12Fence_Release(d->fence);
    if (d->cl) ID3D12GraphicsCommandList_Release(d->cl);
    if (d->alloc) ID3D12CommandAllocator_Release(d->alloc);
    if (d->queue) ID3D12CommandQueue_Release(d->queue);
    if (d->dev) ID3D12Device_Release(d->dev);
    if (d->event) CloseHandle(d->event);
    free(d);
}

int dx_has(const Dx *d, const uint8_t *w) { return d && !d->broken && find_mat(d, w) != NULL; }

#define MAX_DX_JOBS 16

int dx_matmul(Dx *d, const DxJob *jobs, int n) {
    if (!d || d->broken || n < 1 || n > MAX_DX_JOBS) return 0;
    const DxMat *mat[MAX_DX_JOBS];
    size_t xoff[MAX_DX_JOBS], ooff[MAX_DX_JOBS];
    int fresh[MAX_DX_JOBS];
    size_t xbytes = 0, obytes = 0;
    for (int i = 0; i < n; i++) {
        mat[i] = find_mat(d, jobs[i].w);
        if (!mat[i]) return 0;
        /* jobs that read the same activations share one copy */
        fresh[i] = 1;
        for (int k = 0; k < i && fresh[i]; k++)
            if (jobs[k].xq == jobs[i].xq && jobs[k].nt == jobs[i].nt && mat[k]->K == mat[i]->K) {
                xoff[i] = xoff[k];
                fresh[i] = 0;
            }
        if (fresh[i]) {
            xoff[i] = xbytes;
            xbytes += ((size_t)jobs[i].nt * (size_t)(mat[i]->K / QK_K) * Q8_BYTES + 255) & ~(size_t)255;
        }
        ooff[i] = obytes;
        obytes += ((size_t)jobs[i].nt * (size_t)mat[i]->rows * 4 + 255) & ~(size_t)255;
    }
    if (!buf_reserve(d, &d->up, D3D12_HEAP_TYPE_UPLOAD, xbytes) || !buf_reserve(d, &d->x, D3D12_HEAP_TYPE_DEFAULT, xbytes) ||
        !buf_reserve(d, &d->o, D3D12_HEAP_TYPE_DEFAULT, obytes) || !buf_reserve(d, &d->rb, D3D12_HEAP_TYPE_READBACK, obytes))
        return 0;
    for (int i = 0; i < n; i++) {
        if (!fresh[i]) continue;
        const size_t nbk = (size_t)(mat[i]->K / QK_K), row = nbk * Q8_BYTES;
        for (int t = 0; t < jobs[i].nt; t++)
            memcpy(d->up.map + xoff[i] + (size_t)t * row, jobs[i].xq + (size_t)t * (size_t)jobs[i].xstride, row);
    }

    ID3D12GraphicsCommandList *cl = d->cl;
    ID3D12GraphicsCommandList_CopyBufferRegion(cl, d->x.res, 0, d->up.res, 0, xbytes);
    barrier(d, d->x.res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12GraphicsCommandList_SetComputeRootSignature(cl, d->rs);
    const D3D12_GPU_VIRTUAL_ADDRESS xva = ID3D12Resource_GetGPUVirtualAddress(d->x.res),
                                    ova = ID3D12Resource_GetGPUVirtualAddress(d->o.res);
    for (int i = 0; i < n; i++) {
        const DxMat *x = mat[i];
        const UINT nb = (UINT)(x->K / QK_K);
        const UINT p[6] = {(UINT)x->rows, nb, (UINT)jobs[i].nt, (UINT)x->gpu_row, nb * Q8_BYTES, (UINT)x->rows};
        ID3D12GraphicsCommandList_SetPipelineState(cl, d->pso[x->kernel]);
        ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(cl, 0, 6, p, 0);
        ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(cl, 1, x->va);
        ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(cl, 2, xva + xoff[i]);
        ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, 3, ova + ooff[i]);
        ID3D12GraphicsCommandList_Dispatch(cl, (UINT)(x->rows + 63) / 64, (UINT)(jobs[i].nt + 63) / 64, 1);
    }
    barrier(d, d->o.res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ID3D12GraphicsCommandList_CopyBufferRegion(cl, d->rb.res, 0, d->o.res, 0, obytes);
    if (!submit(d)) {
        d->broken = 1;
        fprintf(stderr, "Direct3D 12: the GPU stopped responding, continuing on the CPU\n");
        return 0;
    }
    for (int i = 0; i < n; i++) {
        const float *src = (const float *)(d->rb.map + ooff[i]);
        const int rows = mat[i]->rows;
        for (int t = 0; t < jobs[i].nt; t++)
            memcpy(jobs[i].out + (size_t)t * (size_t)jobs[i].ostride, src + (size_t)t * rows, sizeof(float) * (size_t)rows);
    }
    return 1;
}

#endif
