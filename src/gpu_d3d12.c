/* gpu_d3d12.c - Direct3D 12 compute backend for prompt matmuls (see gpu_dx.h).
 *
 * Every weight matrix lives in its own GPU buffer (Q6_K blocks are repacked from 210 to 212
 * bytes so that the shaders read aligned words). dx_matmul copies the Q8_K activations into a
 * GPU buffer, dispatches one kernel per matrix (kernels_dx.hlsl: tiles of 64 rows x 64 tokens
 * for batches, 8 rows of one token per group below DX_MV_TOKENS) and reads the results back,
 * all in one command list. Buffers rely on the implicit
 * state promotion from COMMON at the start of each command list and decay back after it. */
#include "gpu_dx.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#if !defined(_WIN32)

void dx_list_devices(void) { printf("  Direct3D 12 needs Windows\n"); }
Dx *dx_create(const Model *m, int device, char *info, int infolen) {
    (void)m; (void)device;
    snprintf(info, (size_t)infolen, "Direct3D 12 needs Windows");
    return NULL;
}
void dx_destroy(Dx *d) { (void)d; }
int dx_has(const Dx *d, const uint8_t *w) { (void)d; (void)w; return 0; }
int dx_matmul(Dx *d, const DxJob *jobs, int n) { (void)d; (void)jobs; (void)n; return 0; }
int dx_dense_init(Dx *d, const Model *m, int n_ctx, int n_batch, char *info, int infolen) {
    (void)d; (void)m; (void)n_ctx; (void)n_batch;
    snprintf(info, (size_t)infolen, "Direct3D 12 needs Windows");
    return 0;
}
int dx_dense_forward(Dx *d, const float *x, int n, const float *rope_cs, const float *rope_swa, int pos0, float *logits,
                     float *x_last) {
    (void)d; (void)x; (void)n; (void)rope_cs; (void)rope_swa; (void)pos0; (void)logits; (void)x_last;
    return 0;
}
void dx_dense_kv(Dx *d, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu) {
    (void)d; (void)kc; (void)vc; (void)r0; (void)r1; (void)to_gpu;
}
void dx_dense_conv(Dx *d, float *conv, int to_gpu) { (void)d; (void)conv; (void)to_gpu; }

#else

#define COBJMACROS
#include <windows.h>
#undef GetMessage   /* windows.h maps it to GetMessageA/W, which renames ID3D12InfoQueue's method */
#include <d3d12.h>
#include <dxgi.h>
#include <d3d12sdklayers.h>
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
#define DX_MV_TOKENS 12     /* fewer tokens: one matrix-vector product per token */

enum { K_Q4_K, K_Q6_K, N_KERNELS };
enum { T_RESID, T_QUANT, T_SILU, T_CONV, T_PREP, T_ATTN, T_CONV_CHUNK, T_CONV_STATE, T_ATTN_CHUNK,
       N_TOKEN_KERNELS };   /* kernels_dx_token.hlsl */
enum { ROOT_CONST, ROOT_T0, ROOT_T1, ROOT_U0, ROOT_U1, ROOT_U2 };              /* root parameters */

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
    ID3D12PipelineState *pso[N_KERNELS], *pso_mv[N_KERNELS], *pso_tok[N_TOKEN_KERNELS];
    DxMat *mats;                    /* sorted by host address */
    int nmats, broken;
    DxBuf up, x, o, rb;             /* upload, activations, results, readback */
    struct DxDense *dense;          /* whole-token decoding, or NULL */
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
/* with PITHOS_DX_DEBUG set (debug layer on): print what the layer reported */
static void debug_messages(Dx *d) {
    static const GUID iid_info = {0x0742a90b, 0xc387, 0x483f, {0xb9, 0x46, 0x30, 0xa7, 0xe4, 0xe6, 0x14, 0x58}};
    ID3D12InfoQueue *q = NULL;
    if (!getenv("PITHOS_DX_DEBUG") || FAILED(ID3D12Device_QueryInterface(d->dev, &iid_info, (void **)&q))) return;
    const UINT64 n = q->lpVtbl->GetNumStoredMessages(q);
    for (UINT64 i = 0; i < n && i < 20; i++) {
        SIZE_T len = 0;
        q->lpVtbl->GetMessage(q, i, NULL, &len);
        D3D12_MESSAGE *msg = (D3D12_MESSAGE *)malloc(len);
        if (msg && SUCCEEDED(q->lpVtbl->GetMessage(q, i, msg, &len)))
            fprintf(stderr, "  d3d12 debug: %.*s\n", (int)msg->DescriptionByteLength, msg->pDescription);
        free(msg);
    }
    q->lpVtbl->ClearStoredMessages(q);
    ID3D12InfoQueue_Release(q);
}

static int submit(Dx *d) {
    HRESULT hr = ID3D12GraphicsCommandList_Close(d->cl);
    debug_messages(d);
    if (FAILED(hr)) {
        fprintf(stderr, "Direct3D 12: invalid command list (0x%08lx)\n", (unsigned long)hr);
        return 0;
    }
    ID3D12CommandList *lists[1] = {(ID3D12CommandList *)d->cl};
    ID3D12CommandQueue_ExecuteCommandLists(d->queue, 1, lists);
    const UINT64 v = ++d->fence_val;
    if (FAILED(ID3D12CommandQueue_Signal(d->queue, d->fence, v))) return 0;
    /* decoding submits ~100 short command lists per token: poll for the first 2 ms, which
     * answers sooner than waking up from the event */
    const double t0 = time_sec();
    while (ID3D12Fence_GetCompletedValue(d->fence) < v && time_sec() - t0 < 2e-3) YieldProcessor();
    if (ID3D12Fence_GetCompletedValue(d->fence) < v) {
        if (FAILED(ID3D12Fence_SetEventOnCompletion(d->fence, v, d->event))) return 0;
        WaitForSingleObjectEx(d->event, INFINITE, FALSE);
    }
    if (ID3D12Fence_GetCompletedValue(d->fence) == UINT64_MAX) {
        debug_messages(d);
        fprintf(stderr, "Direct3D 12: device removed (0x%08lx)\n", (unsigned long)ID3D12Device_GetDeviceRemovedReason(d->dev));
        return 0;
    }
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

/* file layout -> GPU layout of rows [r0, r1) of a matrix */
static void pack_rows(const DxMat *x, int r0, int r1, uint8_t *dst) {
    const uint8_t *src = x->host + (size_t)r0 * x->row_bytes;
    if (x->kernel != K_Q6_K) {
        memcpy(dst, src, x->row_bytes * (size_t)(r1 - r0));
        return;
    }
    const size_t nblk = (size_t)(r1 - r0) * (size_t)(x->K / QK_K);
    const block_q6_K *b = (const block_q6_K *)src;
    for (size_t i = 0; i < nblk; i++, dst += Q6K_GPU_BYTES) {
        memcpy(dst, &b[i].d, 2);
        dst[2] = dst[3] = 0;
        memcpy(dst + 4, b[i].ql, 128);
        memcpy(dst + 132, b[i].qh, 64);
        memcpy(dst + 196, b[i].scales, 16);
    }
}

/* matrices go up through the staging buffer, in pieces of whole rows when they do not fit */
static int upload_weights(Dx *d, char *err, int errlen) {
    if (!buf_reserve(d, &d->up, D3D12_HEAP_TYPE_UPLOAD, STAGE_BYTES)) {
        snprintf(err, (size_t)errlen, "cannot allocate the upload buffer");
        return 0;
    }
    size_t used = 0;
    for (int i = 0; i < d->nmats; i++) {
        DxMat *x = &d->mats[i];
        x->res = buffer(d, D3D12_HEAP_TYPE_DEFAULT, x->gpu_row * (size_t)x->rows, D3D12_RESOURCE_STATE_COMMON, 0);
        if (!x->res) {
            snprintf(err, (size_t)errlen, "out of GPU memory after %d of %d matrices", i, d->nmats);
            return 0;
        }
        x->va = ID3D12Resource_GetGPUVirtualAddress(x->res);
        for (int r0 = 0; r0 < x->rows;) {
            int n = (int)((d->up.size - used) / x->gpu_row);
            if (n > x->rows - r0) n = x->rows - r0;
            if (n < 1 || (n < x->rows - r0 && used)) {   /* full: send what is staged */
                if (!submit(d)) { snprintf(err, (size_t)errlen, "weight upload failed"); return 0; }
                used = 0;
                continue;
            }
            const size_t bytes = x->gpu_row * (size_t)n;
            pack_rows(x, r0, r0 + n, d->up.map + used);
            ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, x->res, x->gpu_row * (size_t)r0, d->up.res, used, bytes);
            used = (used + bytes + 255) & ~(size_t)255;
            r0 += n;
        }
    }
    if (!submit(d)) { snprintf(err, (size_t)errlen, "weight upload failed"); return 0; }
    buf_free(&d->up);   /* dx_matmul allocates what it needs */
    return 1;
}

static int create_pipeline(Dx *d, HMODULE lib) {
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize =
        (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)GetProcAddress(lib, "D3D12SerializeRootSignature");
    if (!serialize) return 0;
    /* one root signature for every kernel: b0 constants, t0 t1, u0 u1 u2 (see ROOT_*) */
    D3D12_ROOT_PARAMETER rp[6];
    memset(rp, 0, sizeof(rp));
    rp[ROOT_CONST].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[ROOT_CONST].Constants.Num32BitValues = 16;
    for (int i = 0; i < 2; i++) {
        rp[ROOT_T0 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[ROOT_T0 + i].Descriptor.ShaderRegister = (UINT)i;
    }
    for (int i = 0; i < 3; i++) {
        rp[ROOT_U0 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[ROOT_U0 + i].Descriptor.ShaderRegister = (UINT)i;
    }
    D3D12_ROOT_SIGNATURE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    rd.NumParameters = 6;
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
    const BYTE *code[2 * N_KERNELS] = {g_mm_q4_K, g_mm_q6_K, g_mv_q4_K, g_mv_q6_K};
    const size_t size[2 * N_KERNELS] = {sizeof(g_mm_q4_K), sizeof(g_mm_q6_K), sizeof(g_mv_q4_K), sizeof(g_mv_q6_K)};
    for (int k = 0; k < 2 * N_KERNELS; k++) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd;
        memset(&pd, 0, sizeof(pd));
        pd.pRootSignature = d->rs;
        pd.CS.pShaderBytecode = code[k];
        pd.CS.BytecodeLength = size[k];
        ID3D12PipelineState **dst = k < N_KERNELS ? &d->pso[k] : &d->pso_mv[k - N_KERNELS];
        if (FAILED(ID3D12Device_CreateComputePipelineState(d->dev, &pd, &DX_IID_PipelineState, (void **)dst)))
            return 0;
    }
    const BYTE *tcode[N_TOKEN_KERNELS] = {g_resid_norm, g_quant, g_silu_quant, g_shortconv, g_attn_prep, g_attention,
                                          g_conv_chunk, g_conv_state, g_attn_chunk};
    const size_t tsize[N_TOKEN_KERNELS] = {sizeof(g_resid_norm), sizeof(g_quant), sizeof(g_silu_quant),
                                           sizeof(g_shortconv), sizeof(g_attn_prep), sizeof(g_attention),
                                           sizeof(g_conv_chunk), sizeof(g_conv_state), sizeof(g_attn_chunk)};
    for (int k = 0; k < N_TOKEN_KERNELS; k++) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd;
        memset(&pd, 0, sizeof(pd));
        pd.pRootSignature = d->rs;
        pd.CS.pShaderBytecode = tcode[k];
        pd.CS.BytecodeLength = tsize[k];
        if (FAILED(ID3D12Device_CreateComputePipelineState(d->dev, &pd, &DX_IID_PipelineState, (void **)&d->pso_tok[k])))
            return 0;
    }
    return 1;
}

static IDXGIFactory1 *dxgi_factory(void) {
    HMODULE dxgi = load_system_dll(L"dxgi.dll");
    CreateFactory_t create = dxgi ? (CreateFactory_t)(void *)GetProcAddress(dxgi, "CreateDXGIFactory1") : NULL;
    IDXGIFactory1 *f = NULL;
    if (!create || FAILED(create(&DX_IID_Factory1, (void **)&f))) return NULL;
    return f;
}

void dx_list_devices(void) {
    IDXGIFactory1 *f = dxgi_factory();
    if (!f) { printf("  DXGI not available\n"); return; }
    IDXGIAdapter1 *a = NULL;
    for (UINT i = 0; IDXGIFactory1_EnumAdapters1(f, i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        char name[128] = "?";
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &desc))) {
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), NULL, NULL);
            printf("  [%u] %s | %llu MB dedicated%s\n", i, name, (unsigned long long)(desc.DedicatedVideoMemory >> 20),
                   desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE ? " | software" : "");
        }
        IDXGIAdapter1_Release(a);
    }
    IDXGIFactory1_Release(f);
}

/* adapter number want, or (want < 0) the one with the most dedicated memory (a laptop's discrete
 * GPU rather than the integrated one) */
static IDXGIAdapter1 *pick_adapter(int want, char *name, int namelen) {
    IDXGIFactory1 *f = dxgi_factory();
    if (!f) return NULL;
    IDXGIAdapter1 *best = NULL, *a = NULL;
    SIZE_T best_mem = 0;
    for (UINT i = 0; IDXGIFactory1_EnumAdapters1(f, i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &desc)) &&
            (want >= 0 ? (int)i == want
                       : !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && (!best || desc.DedicatedVideoMemory > best_mem))) {
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

static void dense_free(Dx *d);

Dx *dx_create(const Model *m, int device, char *info, int infolen) {
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
    IDXGIAdapter1 *adapter = pick_adapter(device, name, sizeof(name));
    if (device >= 0 && !adapter) {
        snprintf(info, (size_t)infolen, "no Direct3D 12 adapter %d (see --list-gpus)", device);
        free(d);
        return NULL;
    }
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
    add_mat(d, &m->output, &cap);
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
    dense_free(d);
    for (int k = 0; k < N_KERNELS; k++) {
        if (d->pso[k]) ID3D12PipelineState_Release(d->pso[k]);
        if (d->pso_mv[k]) ID3D12PipelineState_Release(d->pso_mv[k]);
    }
    for (int k = 0; k < N_TOKEN_KERNELS; k++)
        if (d->pso_tok[k]) ID3D12PipelineState_Release(d->pso_tok[k]);
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
    barrier(d, d->x.res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12GraphicsCommandList_SetComputeRootSignature(cl, d->rs);
    const D3D12_GPU_VIRTUAL_ADDRESS xva = ID3D12Resource_GetGPUVirtualAddress(d->x.res),
                                    ova = ID3D12Resource_GetGPUVirtualAddress(d->o.res);
    for (int i = 0; i < n; i++) {
        const DxMat *x = mat[i];
        const UINT nb = (UINT)(x->K / QK_K);
        const UINT p[6] = {(UINT)x->rows, nb, (UINT)jobs[i].nt, (UINT)x->gpu_row, nb * Q8_BYTES, (UINT)x->rows};
        const int mv = jobs[i].nt < DX_MV_TOKENS;
        ID3D12GraphicsCommandList_SetPipelineState(cl, mv ? d->pso_mv[x->kernel] : d->pso[x->kernel]);
        ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(cl, ROOT_CONST, 6, p, 0);
        ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(cl, ROOT_T0, x->va);
        ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U1, xva + xoff[i]);
        ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U0, ova + ooff[i]);
        if (mv) ID3D12GraphicsCommandList_Dispatch(cl, (UINT)(x->rows + 7) / 8, (UINT)jobs[i].nt, 1);
        else ID3D12GraphicsCommandList_Dispatch(cl, (UINT)(x->rows + 63) / 64, (UINT)(jobs[i].nt + 63) / 64, 1);
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

/* ------------------------------------------------------------------ whole forward pass
 *
 * dx_dense_forward records the forward pass of one token or of a prompt chunk into one command
 * list (kernels in kernels_dx_token.hlsl between the matmuls, separated by UAV barriers) and
 * reads back only the logits of the last token. The GPU keeps its own copy of the KV cache and
 * of the convolution state, in the CPU layout; dx_dense_kv / dx_dense_conv move rows between
 * the two. Activations have one row per token, for up to n_batch tokens. */

#define NONE_OFF 0xffffffffu

typedef struct {
    UINT attn_norm, ffn_norm, post_op, post_ffn, q_norm, k_norm, conv_w;   /* byte offsets in sw */
    const DxMat *wq, *wk, *wv, *wo, *gate, *in_proj, *out_proj, *w_gate, *w_up, *w_down;
} DxLayer;

typedef struct DxDense {
    const Model *m;
    int n_ctx, B, D, hd, nh, nkv, kvd, F, lc, n_attn, n_conv;
    DxLayer *layers;
    const DxMat *output;
    UINT final_norm;
    ID3D12Resource *scratch, *kv, *cs, *sw;
    D3D12_GPU_VIRTUAL_ADDRESS sva, kva, cva, wva;
    DxBuf in, lrb;                  /* the tokens' input (upload), logits (readback) */
    size_t kv_half;                 /* bytes of the key half of kv */
    size_t o_x, o_out, o_tmp, o_hq, o_big, o_gate, o_fg, o_fu, o_fq, o_sc, o_logits;
} DxDense;

static void dense_free(Dx *d) {
    DxDense *t = d->dense;
    if (!t) return;
    ID3D12Resource *rs[4] = {t->scratch, t->kv, t->cs, t->sw};
    for (int i = 0; i < 4; i++)
        if (rs[i]) ID3D12Resource_Release(rs[i]);
    buf_free(&t->in);
    buf_free(&t->lrb);
    free(t->layers);
    free(t);
    d->dense = NULL;
}

/* append n floats to the small-weights array, returning their byte offset (NONE_OFF for NULL) */
static UINT add_vec(float **arr, size_t *len, size_t *cap, const float *v, size_t n) {
    if (!v) return NONE_OFF;
    const size_t need = *len + ((n + 63) & ~(size_t)63);   /* keep offsets 256-byte aligned */
    if (need > *cap) {
        *cap = need * 2;
        *arr = (float *)realloc(*arr, sizeof(float) * *cap);
    }
    memcpy(*arr + *len, v, sizeof(float) * n);
    const UINT off = (UINT)(*len * 4);
    *len = need;
    return off;
}

static size_t carve(size_t *top, size_t bytes) {
    const size_t o = *top;
    *top += (bytes + 255) & ~(size_t)255;
    return o;
}

int dx_dense_init(Dx *d, const Model *m, int n_ctx, int n_batch, char *info, int infolen) {
    info[0] = 0;
    if (!d || d->broken) return 0;
    const int D = m->dim, hd = m->head_dim;
    int F = 0, ok = D % QK_K == 0 && hd % 2 == 0 && hd <= 512 && 256 % (hd / 2) == 0 && m->l_cache >= 2;
    const DxMat *output = find_mat(d, m->output.data);
    ok = ok && output;
    for (int l = 0; l < m->n_layer && ok; l++) {
        const Layer *L = &m->layers[l];
        if (L->is_moe) ok = 0;
        else if (L->w_gate.rows > F) F = L->w_gate.rows;
    }
    if (!ok || F % QK_K) {
        snprintf(info, (size_t)infolen, "this model needs the CPU for part of each token");
        return 0;
    }
    DxDense *t = (DxDense *)calloc(1, sizeof(DxDense));
    t->m = m;
    t->n_ctx = n_ctx;
    t->B = n_batch < 1 ? 1 : n_batch;
    t->D = D;
    t->hd = hd;
    t->nh = m->n_head;
    t->nkv = m->n_kv_head;
    t->kvd = m->n_kv_head * hd;
    t->F = F;
    t->lc = m->l_cache;
    t->n_attn = m->n_attn;
    t->n_conv = m->n_conv;
    t->output = output;
    t->layers = (DxLayer *)calloc((size_t)m->n_layer, sizeof(DxLayer));
    d->dense = t;

    float *sw = NULL;
    size_t len = 0, cap = 0;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        DxLayer *X = &t->layers[l];
        X->attn_norm = add_vec(&sw, &len, &cap, L->attn_norm, (size_t)D);
        X->ffn_norm = add_vec(&sw, &len, &cap, L->ffn_norm, (size_t)D);
        X->post_op = add_vec(&sw, &len, &cap, L->post_op_norm, (size_t)D);
        X->post_ffn = add_vec(&sw, &len, &cap, L->post_ffn_norm, (size_t)D);
        if (L->is_attn) {
            X->q_norm = add_vec(&sw, &len, &cap, L->q_norm, (size_t)hd);
            X->k_norm = add_vec(&sw, &len, &cap, L->k_norm, (size_t)hd);
            X->wq = find_mat(d, L->wq.data);
            X->wk = find_mat(d, L->wk.data);
            X->wv = find_mat(d, L->wv.data);
            X->wo = find_mat(d, L->wo.data);
            X->gate = L->attn_gate.data ? find_mat(d, L->attn_gate.data) : NULL;
            ok = ok && X->wq && X->wk && X->wv && X->wo && (X->gate || !L->attn_gate.data) && L->q_norm && L->k_norm;
        } else {
            X->conv_w = add_vec(&sw, &len, &cap, L->conv_w, (size_t)D * (size_t)t->lc);
            X->in_proj = find_mat(d, L->in_proj.data);
            X->out_proj = find_mat(d, L->out_proj.data);
            ok = ok && X->in_proj && X->out_proj;
        }
        X->w_gate = find_mat(d, L->w_gate.data);
        X->w_up = find_mat(d, L->w_up.data);
        X->w_down = find_mat(d, L->w_down.data);
        ok = ok && X->w_gate && X->w_up && X->w_down && L->attn_norm && L->ffn_norm;
    }
    t->final_norm = add_vec(&sw, &len, &cap, m->final_norm, (size_t)D);
    if (!ok || !m->final_norm) {
        free(sw);
        dense_free(d);
        snprintf(info, (size_t)infolen, "some weights of this model are not on the GPU");
        return 0;
    }

    size_t top = 0;
    const size_t B = (size_t)t->B, big = 3 * D > D + 2 * t->kvd ? 3 * D : D + 2 * t->kvd;
    t->o_x = carve(&top, B * D * 4);
    t->o_out = carve(&top, B * D * 4);
    t->o_tmp = carve(&top, B * D * 4);
    t->o_hq = carve(&top, B * (D / QK_K) * Q8_BYTES);
    t->o_big = carve(&top, B * big * 4);
    t->o_gate = carve(&top, B * t->nh * 4);
    t->o_fg = carve(&top, B * F * 4);
    t->o_fu = carve(&top, B * F * 4);
    t->o_fq = carve(&top, B * (F / QK_K) * Q8_BYTES);
    t->o_sc = carve(&top, (size_t)t->nh * (size_t)n_ctx * 4);
    t->o_logits = carve(&top, (size_t)m->n_vocab * 4);
    t->kv_half = (size_t)t->n_attn * (size_t)n_ctx * (size_t)t->kvd * 2;
    const size_t cs_bytes = (size_t)t->n_conv * (size_t)(t->lc - 1) * (size_t)D * 4;
    t->scratch = buffer(d, D3D12_HEAP_TYPE_DEFAULT, top, D3D12_RESOURCE_STATE_COMMON, 1);
    t->kv = buffer(d, D3D12_HEAP_TYPE_DEFAULT, t->kv_half ? 2 * t->kv_half : 256, D3D12_RESOURCE_STATE_COMMON, 1);
    t->cs = buffer(d, D3D12_HEAP_TYPE_DEFAULT, cs_bytes ? cs_bytes : 256, D3D12_RESOURCE_STATE_COMMON, 1);
    t->sw = buffer(d, D3D12_HEAP_TYPE_DEFAULT, len * 4, D3D12_RESOURCE_STATE_COMMON, 0);
    ok = t->scratch && t->kv && t->cs && t->sw && buf_reserve(d, &t->in, D3D12_HEAP_TYPE_UPLOAD, B * (size_t)(D + 2 * hd) * 4) &&
         buf_reserve(d, &t->lrb, D3D12_HEAP_TYPE_READBACK, (size_t)(m->n_vocab + D) * 4) &&
         buf_reserve(d, &d->up, D3D12_HEAP_TYPE_UPLOAD, len * 4);
    if (ok) {
        memcpy(d->up.map, sw, len * 4);
        ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, t->sw, 0, d->up.res, 0, len * 4);
        ok = submit(d);
    }
    free(sw);
    if (!ok) {
        dense_free(d);
        snprintf(info, (size_t)infolen, "not enough GPU memory for a %d-token context and %d-token batches", n_ctx, t->B);
        return 0;
    }
    t->sva = ID3D12Resource_GetGPUVirtualAddress(t->scratch);
    t->kva = ID3D12Resource_GetGPUVirtualAddress(t->kv);
    t->cva = ID3D12Resource_GetGPUVirtualAddress(t->cs);
    t->wva = ID3D12Resource_GetGPUVirtualAddress(t->sw);
    snprintf(info, (size_t)infolen, "whole forward pass on the GPU, %.0f MB of KV cache, %.0f MB of activations",
             2.0 * t->kv_half / 1048576.0, top / 1048576.0);
    return 1;
}

static void uav_barrier(Dx *d) {
    D3D12_RESOURCE_BARRIER b;
    memset(&b, 0, sizeof(b));
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    ID3D12GraphicsCommandList_ResourceBarrier(d->cl, 1, &b);
}

/* out[t][r] (scratch offset oo, rows orow floats apart) = matrix x . the Q8_K rows of n tokens at xo */
static void tok_mm(Dx *d, const DxMat *x, size_t xo, size_t oo, int n, int orow) {
    const DxDense *t = d->dense;
    const UINT nb = (UINT)(x->K / QK_K);
    const UINT p[6] = {(UINT)x->rows, nb, (UINT)n, (UINT)x->gpu_row, nb * Q8_BYTES, (UINT)orow};
    const int mv = n < DX_MV_TOKENS;
    ID3D12GraphicsCommandList *cl = d->cl;
    ID3D12GraphicsCommandList_SetPipelineState(cl, mv ? d->pso_mv[x->kernel] : d->pso[x->kernel]);
    ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(cl, ROOT_CONST, 6, p, 0);
    ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(cl, ROOT_T0, x->va);
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U1, t->sva + xo);
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U0, t->sva + oo);
    if (mv) ID3D12GraphicsCommandList_Dispatch(cl, (UINT)(x->rows + 7) / 8, (UINT)n, 1);
    else ID3D12GraphicsCommandList_Dispatch(cl, (UINT)(x->rows + 63) / 64, (UINT)(n + 63) / 64, 1);
}

static void tok_run(Dx *d, int k, const UINT c[16], UINT groups, UINT tokens) {
    const DxDense *t = d->dense;
    ID3D12GraphicsCommandList *cl = d->cl;
    ID3D12GraphicsCommandList_SetPipelineState(cl, d->pso_tok[k]);
    ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(cl, ROOT_CONST, 16, c, 0);
    ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(cl, ROOT_T0, t->wva);
    ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(cl, ROOT_T1, ID3D12Resource_GetGPUVirtualAddress(t->in.res));
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U0, t->sva);
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U1, t->kva);
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(cl, ROOT_U2, t->cva);
    ID3D12GraphicsCommandList_Dispatch(cl, groups, tokens, 1);
    uav_barrier(d);
}

static UINT fbits(float f) {
    UINT u;
    memcpy(&u, &f, 4);
    return u;
}

/* x = norm (mode 0: embedding from the input; 1: x += out, RMS-normalized by post) -> Q8_K at hq */
static void tok_resid(Dx *d, UINT mode, UINT post, UINT norm, int n) {
    const DxDense *t = d->dense;
    const UINT c[16] = {(UINT)t->D, (UINT)t->o_x, (UINT)t->o_out, (UINT)t->o_hq, mode, post, norm, fbits(t->m->eps)};
    tok_run(d, T_RESID, c, 1, (UINT)n);
}

int dx_dense_forward(Dx *d, const float *x, int n, const float *rope_cs, const float *rope_swa, int pos0, float *logits,
                     float *x_last) {
    DxDense *t = d ? d->dense : NULL;
    if (!t || d->broken || n < 1 || n > t->B || pos0 + n > t->n_ctx) return 0;
    const Model *m = t->m;
    const int D = t->D, hd = t->hd, kvd = t->kvd, qkvw = D + 2 * kvd;
    const UINT nb = (UINT)(D / QK_K), un = (UINT)n;
    memcpy(t->in.map, x, sizeof(float) * (size_t)n * D);
    memcpy(t->in.map + 4 * (size_t)n * D, rope_cs, sizeof(float) * (size_t)n * hd);
    if (rope_swa) memcpy(t->in.map + 4 * (size_t)n * (D + hd), rope_swa, sizeof(float) * (size_t)n * hd);
    ID3D12GraphicsCommandList *cl = d->cl;
    ID3D12GraphicsCommandList_SetComputeRootSignature(cl, d->rs);

    tok_resid(d, 0, NONE_OFF, t->layers[0].attn_norm, n);
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        const DxLayer *X = &t->layers[l];
        if (L->is_attn) {
            tok_mm(d, X->wq, t->o_hq, t->o_big, n, qkvw);
            tok_mm(d, X->wk, t->o_hq, t->o_big + 4 * (size_t)D, n, qkvw);
            tok_mm(d, X->wv, t->o_hq, t->o_big + 4 * (size_t)(D + kvd), n, qkvw);
            if (X->gate) tok_mm(d, X->gate, t->o_hq, t->o_gate, n, t->nh);
            uav_barrier(d);
            const size_t kbase = (size_t)L->attn_idx * (size_t)t->n_ctx * (size_t)kvd * 2;
            const size_t krow = kbase + (size_t)pos0 * (size_t)kvd * 2;
            const UINT rope = (UINT)(4 * (size_t)n * (L->is_swa && rope_swa ? D + hd : D));
            const UINT cp[16] = {(UINT)hd, (UINT)t->o_big, (UINT)D, (UINT)kvd, X->q_norm, X->k_norm, rope, fbits(m->eps),
                                 (UINT)krow, (UINT)(t->kv_half + krow), (UINT)t->nh};
            tok_run(d, T_PREP, cp, (UINT)(t->nh + t->nkv), un);
            const UINT gate = X->gate ? (UINT)t->o_gate : NONE_OFF, scale = fbits(1.f / sqrtf((float)hd));
            if (n == 1) {
                const int npos = pos0 + 1, j0 = L->is_swa && npos > m->swa_window ? npos - m->swa_window : 0;
                const UINT ca[16] = {(UINT)hd, (UINT)t->o_big, gate, (UINT)t->o_tmp,
                                     (UINT)kbase, (UINT)(t->kv_half + kbase), (UINT)kvd, (UINT)t->o_sc,
                                     (UINT)j0, (UINT)npos, (UINT)(t->nh / t->nkv), (UINT)t->n_ctx, scale};
                tok_run(d, T_ATTN, ca, (UINT)t->nh, 1);
            } else {
                const UINT ca[16] = {(UINT)hd, (UINT)t->o_big, gate, (UINT)t->o_tmp,
                                     (UINT)kbase, (UINT)(t->kv_half + kbase), (UINT)kvd, 0,
                                     (UINT)pos0, (UINT)(L->is_swa && m->swa_window > 0 ? m->swa_window : 0),
                                     (UINT)(t->nh / t->nkv), (UINT)t->nh, scale, (UINT)D};
                tok_run(d, T_ATTN_CHUNK, ca, (UINT)t->nh, un);
            }
            const UINT cq[16] = {(UINT)t->o_tmp, (UINT)t->o_hq, (UINT)D * 4, nb * Q8_BYTES};
            tok_run(d, T_QUANT, cq, nb, un);
            tok_mm(d, X->wo, t->o_hq, t->o_out, n, D);
        } else {
            tok_mm(d, X->in_proj, t->o_hq, t->o_big, n, 3 * D);
            uav_barrier(d);
            const UINT state = (UINT)((size_t)L->conv_idx * (size_t)(t->lc - 1) * (size_t)D * 4);
            const UINT cc[16] = {(UINT)D, (UINT)t->o_big, (UINT)t->o_hq, (UINT)t->lc, X->conv_w, state};
            if (n == 1) {
                tok_run(d, T_CONV, cc, nb, 1);
            } else {
                tok_run(d, T_CONV_CHUNK, cc, nb, un);
                const UINT cs[16] = {(UINT)D, (UINT)t->o_big, un, (UINT)t->lc, 0, state};
                tok_run(d, T_CONV_STATE, cs, nb, 1);
            }
            tok_mm(d, X->out_proj, t->o_hq, t->o_out, n, D);
        }
        uav_barrier(d);
        tok_resid(d, 1, X->post_op, X->ffn_norm, n);
        const int F = X->w_gate->rows;
        tok_mm(d, X->w_gate, t->o_hq, t->o_fg, n, F);
        tok_mm(d, X->w_up, t->o_hq, t->o_fu, n, F);
        uav_barrier(d);
        const UINT cs[16] = {(UINT)t->o_fg, (UINT)t->o_fu, (UINT)t->o_fq, (UINT)F};
        tok_run(d, T_SILU, cs, (UINT)(F / QK_K), un);
        tok_mm(d, X->w_down, t->o_fq, t->o_out, n, D);
        uav_barrier(d);
        tok_resid(d, 1, X->post_ffn, l + 1 < m->n_layer ? t->layers[l + 1].attn_norm : t->final_norm, n);
    }
    if (logits) {
        tok_mm(d, t->output, t->o_hq + (size_t)(n - 1) * nb * Q8_BYTES, t->o_logits, 1, m->n_vocab);
        barrier(d, t->scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        ID3D12GraphicsCommandList_CopyBufferRegion(cl, t->lrb.res, 0, t->scratch, t->o_logits, (UINT64)m->n_vocab * 4);
        if (x_last)
            ID3D12GraphicsCommandList_CopyBufferRegion(cl, t->lrb.res, (UINT64)m->n_vocab * 4, t->scratch,
                                                       t->o_x + (size_t)(n - 1) * D * 4, (UINT64)D * 4);
    }
    if (!submit(d)) {
        d->broken = 1;
        fprintf(stderr, "Direct3D 12: the GPU stopped responding\n");
        return 0;
    }
    if (logits) memcpy(logits, t->lrb.map, sizeof(float) * (size_t)m->n_vocab);
    if (logits && x_last) memcpy(x_last, t->lrb.map + 4 * (size_t)m->n_vocab, sizeof(float) * (size_t)D);
    return 1;
}

void dx_dense_kv(Dx *d, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu) {
    DxDense *t = d ? d->dense : NULL;
    if (!t || d->broken || r1 <= r0 || !t->n_attn) return;
    const size_t row = (size_t)t->kvd * 2, bytes = (size_t)(r1 - r0) * row, total = 2 * (size_t)t->n_attn * bytes;
    DxBuf *st = to_gpu ? &d->up : &d->rb;
    if (!buf_reserve(d, st, to_gpu ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK, total)) return;
    for (int a = 0; a < t->n_attn; a++)
        for (int h = 0; h < 2; h++) {
            uint16_t *host = h ? vc : kc;
            const size_t off = ((size_t)a * (size_t)t->n_ctx + (size_t)r0) * row, gpu = (h ? t->kv_half : 0) + off;
            const size_t so = (2 * (size_t)a + (size_t)h) * bytes;
            if (to_gpu) {
                memcpy(st->map + so, (uint8_t *)host + off, bytes);
                ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, t->kv, gpu, st->res, so, bytes);
            } else {
                ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, st->res, so, t->kv, gpu, bytes);
            }
        }
    if (!submit(d)) { d->broken = 1; return; }
    if (!to_gpu)
        for (int a = 0; a < t->n_attn; a++)
            for (int h = 0; h < 2; h++) {
                const size_t off = ((size_t)a * (size_t)t->n_ctx + (size_t)r0) * row;
                memcpy((uint8_t *)(h ? vc : kc) + off, st->map + (2 * (size_t)a + (size_t)h) * bytes, bytes);
            }
}

void dx_dense_conv(Dx *d, float *conv, int to_gpu) {
    DxDense *t = d ? d->dense : NULL;
    if (!t || d->broken || !t->n_conv) return;
    const size_t bytes = (size_t)t->n_conv * (size_t)(t->lc - 1) * (size_t)t->D * 4;
    DxBuf *st = to_gpu ? &d->up : &d->rb;
    if (!buf_reserve(d, st, to_gpu ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK, bytes)) return;
    if (to_gpu) {
        memcpy(st->map, conv, bytes);
        ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, t->cs, 0, st->res, 0, bytes);
    } else {
        ID3D12GraphicsCommandList_CopyBufferRegion(d->cl, st->res, 0, t->cs, 0, bytes);
    }
    if (!submit(d)) { d->broken = 1; return; }
    if (!to_gpu) memcpy(conv, st->map, bytes);
}

#endif
