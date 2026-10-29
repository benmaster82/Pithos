/* gpu_opencl.c - expert co-processor on any OpenCL 1.2+ GPU.
 *
 * Nothing is linked at build time: the OpenCL ICD loader (OpenCL.dll / libOpenCL.so) is opened
 * at runtime and the kernels (src/kernels.cl, embedded) are compiled by the driver. On NVIDIA
 * GPUs with compute capability >= 6.1 an int8/dp4a path is used, elsewhere an f32 path.
 *
 * VRAM layout: fixed-size slots (gate | up | down of one expert) inside up to 4 arenas
 * (OpenCL caps single allocations, typically at 1/4 of VRAM). A background thread uploads
 * experts chosen by a decayed usage score; the compute path only uses slots marked READY. */
#include "gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"
#include "kernels_cl.h"
#include "platform.h"

/* ------------------------------------------------------------------ minimal OpenCL ABI */

#if defined(_WIN32)
#define CLAPI __stdcall
#else
#define CLAPI
#endif

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef void *cl_platform_id, *cl_device_id, *cl_context, *cl_command_queue, *cl_mem, *cl_program, *cl_kernel, *cl_event;

#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFu
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_VENDOR_ID 0x1001
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283

typedef struct {
    cl_int (CLAPI *GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (CLAPI *GetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id *, cl_uint *);
    cl_int (CLAPI *GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_context (CLAPI *CreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
    cl_command_queue (CLAPI *CreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int *);
    cl_mem (CLAPI *CreateBuffer)(cl_context, cl_ulong, size_t, void *, cl_int *);
    cl_program (CLAPI *CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    cl_int (CLAPI *BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
    cl_int (CLAPI *GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_kernel (CLAPI *CreateKernel)(cl_program, const char *, cl_int *);
    cl_int (CLAPI *SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (CLAPI *EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                                         const size_t *, cl_uint, const cl_event *, cl_event *);
    cl_int (CLAPI *EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint,
                                       const cl_event *, cl_event *);
    cl_int (CLAPI *EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint,
                                      const cl_event *, cl_event *);
    void *(CLAPI *EnqueueMapBuffer)(cl_command_queue, cl_mem, cl_uint, cl_ulong, size_t, size_t, cl_uint,
                                    const cl_event *, cl_event *, cl_int *);
    cl_int (CLAPI *Finish)(cl_command_queue);
    cl_int (CLAPI *Flush)(cl_command_queue);
    cl_int (CLAPI *GetEventProfilingInfo)(cl_event, cl_uint, size_t, void *, size_t *);
    cl_int (CLAPI *GetEventInfo)(cl_event, cl_uint, size_t, void *, size_t *);
    cl_event (CLAPI *CreateUserEvent)(cl_context, cl_int *);
    cl_int (CLAPI *SetUserEventStatus)(cl_event, cl_int);
    cl_int (CLAPI *ReleaseEvent)(cl_event);
    cl_int (CLAPI *ReleaseMemObject)(cl_mem);
    cl_int (CLAPI *ReleaseKernel)(cl_kernel);
    cl_int (CLAPI *ReleaseProgram)(cl_program);
    cl_int (CLAPI *ReleaseCommandQueue)(cl_command_queue);
    cl_int (CLAPI *ReleaseContext)(cl_context);
} CLApi;

static int cl_load(CLApi *cl) {
    static const char *names[] = {
#if defined(_WIN32)
        "OpenCL.dll",
#elif defined(__APPLE__)
        "/System/Library/Frameworks/OpenCL.framework/OpenCL",
#else
        "libOpenCL.so.1", "libOpenCL.so",
#endif
    };
    void *lib = NULL;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && !lib; i++) lib = lib_open(names[i]);
    if (!lib) return 0;
#define LOAD(f) if (!(*(void **)&cl->f = lib_sym(lib, "cl" #f))) return 0
    LOAD(GetPlatformIDs); LOAD(GetDeviceIDs); LOAD(GetDeviceInfo); LOAD(CreateContext); LOAD(CreateCommandQueue);
    LOAD(CreateBuffer); LOAD(CreateProgramWithSource); LOAD(BuildProgram); LOAD(GetProgramBuildInfo);
    LOAD(CreateKernel); LOAD(SetKernelArg); LOAD(EnqueueNDRangeKernel); LOAD(EnqueueWriteBuffer);
    LOAD(EnqueueReadBuffer); LOAD(EnqueueMapBuffer); LOAD(Finish); LOAD(Flush); LOAD(GetEventProfilingInfo);
    LOAD(GetEventInfo); LOAD(CreateUserEvent); LOAD(SetUserEventStatus); LOAD(ReleaseEvent); LOAD(ReleaseMemObject); LOAD(ReleaseKernel); LOAD(ReleaseProgram);
    LOAD(ReleaseCommandQueue); LOAD(ReleaseContext);
#undef LOAD
    return 1;
}

/* ------------------------------------------------------------------ state */

#define MAX_ARENAS 4
#define WG 256          /* must match kernels.cl */
#define N_EV 3

enum { SLOT_FREE = 0, SLOT_LOADING = 1, SLOT_READY = 2 };

typedef struct { int slot; int layer; int expert; } UploadReq;

/* a dense tensor in VRAM: buf = weight arena index, or -1 for the small f32 buffer */
typedef struct { int buf, type, rows, cols; cl_ulong off; const uint8_t *src; size_t bytes; } DT;
typedef struct {
    DT attn_norm, ffn_norm, wq, wk, wv, wo, qn, kn, in_proj, out_proj, conv_w, router, bias, wg, wu, wd;
    DT wgate, post_op, post_ffn;    /* LFM3: attention output gate, sandwich norms (src NULL if absent) */
} DL;

struct Gpu {
    CLApi cl;
    const Model *m;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q, qup;
    cl_program prog;
    cl_kernel k_gu, k_down;
    int dp4a, rpw;          /* rows per 32-lane group (passed to kernels.cl as RPW) */
    cl_ulong local_mem;     /* local memory per work-group (32 KB on AMD GCN) */
    cl_mem arena[MAX_ARENAS];
    int n_arenas, spa, n_slots;
    size_t slot_bytes;
    /* device buffers: meta = slots[max_assign] | xidx[max_assign] | x[max_tok][dim] */
    cl_mem d_meta, d_gu, d_y;
    int max_tok, max_assign, cur_assign;
    /* per MoE layer layout inside a slot */
    size_t up_off[64], down_off[64];
    int gate_type[64], down_type[64], layer_ok[64];
    /* residency */
    int *slot_owner;            /* layer*64+expert or -1 */
    volatile int *slot_state;
    int res_slot[64 * 64];
    float score[64 * 64];
    /* uploader thread */
    PithosThread *th;
    PithosLock *lk;
    UploadReq *req;
    int req_head, req_count, req_cap, stop;
    cl_event *fresh;            /* completed uploads the compute queue has not waited for yet */
    int n_fresh, cap_fresh;
    /* pinned (page-locked) host staging: transfers become asynchronous DMA */
    cl_mem p_meta, p_y;
    int *h_meta;                /* same layout as d_meta */
    int *h_slots, *h_xidx;      /* views into h_meta */
    float *h_x;
    float *h_y;                 /* max_assign x dim */
    int busy, reported;
    /* submission thread: driver calls (~200 us per layer on WDDM) overlap with CPU work */
    PithosThread *sub_th;
    PithosLock *sub_lk;
    volatile int job_seq, done_seq;
    int job_layer, job_ntok, sub_stop, spin_limit;
    /* optional device-side profiling (PITHOS_GPU_PROFILE=1) */
    int profiling, pcount;
    cl_event ev[N_EV];
    double ptime[N_EV];
    char profile[1024];
    GpuStats st;
    /* dense decode path (see below) */
    int dense, n_ctx, dff, n_dz, dz_fail;
    cl_ulong dense_bytes;
    DL *dl;
    DT d_out, d_fnorm;
    cl_mem dz[4], dsmall;
    cl_mem dx, dbig, dtmp, dfg, dscores, dkc, dvc, dconv, drope, dres, dlog, dcount, dmeta2, dout, dgu2, dy2, dycpu, dlogits;
    cl_mem dgate, dpost;        /* LFM3: per-head gate logits, a block output before its norm */
    cl_mem pin_in, pin_out, pin_y, pin_logits;
    float *h_in, *h_out, *h_yc, *h_logits;
    cl_ulong kv_bytes;
    size_t conv_bytes;
    cl_event ev_r[64], uev[64], ev_logits; /* per-layer route readback / CPU-part gate */
    int enq, enq_logits, cur_pos;          /* layers already in the queue for this token */
    int moe_open;                          /* MoE layer still waiting for its CPU part (-1: none) */
    int lookahead;                         /* 1: queue the next layer behind a user event */
    int eager;                             /* flush after every command */
    cl_kernel k_mv, k_conv, k_silu, k_attn, k_route, k_comb, k_addn;
};

static int state_load(volatile int *p) {
#if defined(_MSC_VER) && !defined(__clang__)
    int v = *p;
    _ReadWriteBarrier();
    return v;
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

static void state_store(volatile int *p, int v) {
#if defined(_MSC_VER) && !defined(__clang__)
    _ReadWriteBarrier();
    *p = v;
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

static void check(Gpu *g, cl_int e, const char *what) {
    if (e && !g->reported) {
        fprintf(stderr, "GPU: OpenCL error %d in %s\n", e, what);
        g->reported = 1;
    }
}

/* ------------------------------------------------------------------ device discovery */

typedef struct { cl_device_id id; char name[128]; cl_ulong mem, maxalloc; int unified, gpu; } DevInfo;

static int list_devices(CLApi *cl, DevInfo *out, int cap) {
    cl_platform_id plats[8];
    cl_uint np = 0;
    if (cl->GetPlatformIDs(8, plats, &np) != 0) return 0;
    int n = 0;
    for (cl_uint p = 0; p < np; p++) {
        cl_device_id devs[16];
        cl_uint nd = 0;
        if (cl->GetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, devs, &nd) != 0) continue;
        for (cl_uint d = 0; d < nd && n < cap; d++) {
            DevInfo *di = &out[n++];
            memset(di, 0, sizeof(*di));
            di->id = devs[d];
            cl->GetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(di->name) - 1, di->name, NULL);
            cl->GetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(di->mem), &di->mem, NULL);
            cl->GetDeviceInfo(devs[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(di->maxalloc), &di->maxalloc, NULL);
            cl_uint uni = 0;
            cl->GetDeviceInfo(devs[d], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(uni), &uni, NULL);
            di->unified = uni != 0;
            cl_ulong type = 0;
            cl->GetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            di->gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
        }
    }
    return n;
}

void gpu_list_devices(void) {
    CLApi cl;
    if (!cl_load(&cl)) { printf("OpenCL not available\n"); return; }
    DevInfo d[32];
    int n = list_devices(&cl, d, 32);
    for (int i = 0; i < n; i++)
        printf("  [%d] %s | %llu MB | %s%s\n", i, d[i].name, (unsigned long long)(d[i].mem >> 20),
               d[i].gpu ? "GPU" : "other", d[i].unified ? " integrated (shared memory)" : " dedicated");
}

/* NVIDIA compute capability >= 6.1 has dp4a; reachable from OpenCL through inline PTX */
static int device_has_dp4a(CLApi *cl, cl_device_id dev) {
    cl_uint vendor = 0, major = 0, minor = 0;
    cl->GetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);
    if (vendor != 0x10DE) return 0;
    if (cl->GetDeviceInfo(dev, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(major), &major, NULL) != 0) return 0;
    if (cl->GetDeviceInfo(dev, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(minor), &minor, NULL) != 0) return 0;
    return major > 6 || (major == 6 && minor >= 1);
}

/* ------------------------------------------------------------------ uploader */

static void upload_loop(void *arg) {
    Gpu *g = (Gpu *)arg;
    for (;;) {
        lock_acquire(g->lk);
        while (!g->req_count && !g->stop) lock_wait(g->lk);
        if (g->stop) { lock_release(g->lk); return; }
        UploadReq r = g->req[g->req_head];
        g->req_head = (g->req_head + 1) % g->req_cap;
        g->req_count--;
        lock_release(g->lk);

        const Layer *L = &g->m->layers[r.layer];
        const int a = r.slot / g->spa;
        const size_t base = (size_t)(r.slot - a * g->spa) * g->slot_bytes;
        const size_t gb = L->gate_exps.expert_stride, ub = L->up_exps.expert_stride, db = L->down_exps.expert_stride;
        const uint8_t *gs = L->gate_exps.data + (size_t)r.expert * gb;
        const uint8_t *us = L->up_exps.data + (size_t)r.expert * ub;
        const uint8_t *ds = L->down_exps.data + (size_t)r.expert * db;
        cl_event ev = NULL;
        cl_int e1 = g->cl.EnqueueWriteBuffer(g->qup, g->arena[a], 1, base, gb, gs, 0, NULL, NULL);
        cl_int e2 = g->cl.EnqueueWriteBuffer(g->qup, g->arena[a], 1, base + g->up_off[r.layer], ub, us, 0, NULL, NULL);
        cl_int e3 = g->cl.EnqueueWriteBuffer(g->qup, g->arena[a], 1, base + g->down_off[r.layer], db, ds, 0, NULL, &ev);
        /* the CPU will not touch this expert while it lives in VRAM: drop it from the working set */
        release_memory(gs, gb);
        release_memory(us, ub);
        release_memory(ds, db);
        const int ok = e1 == 0 && e2 == 0 && e3 == 0;
        /* the event is queued before the slot turns READY, so whoever sees READY also finds it
         * in take_uploads (the queue is in order: the last write covers the other two) */
        if (ev) {
            lock_acquire(g->lk);
            if (ok && g->n_fresh == g->cap_fresh) {
                const int cap = g->cap_fresh ? 2 * g->cap_fresh : 64;
                cl_event *p = (cl_event *)realloc(g->fresh, sizeof(cl_event) * (size_t)cap);
                if (p) { g->fresh = p; g->cap_fresh = cap; }
            }
            if (ok && g->n_fresh < g->cap_fresh) g->fresh[g->n_fresh++] = ev;
            else g->cl.ReleaseEvent(ev);
            lock_release(g->lk);
        }
        /* on failure the slot goes back to FREE; the main thread reclaims it in gpu_rebalance */
        state_store(&g->slot_state[r.slot], ok ? SLOT_READY : SLOT_FREE);
    }
}

static int request_upload(Gpu *g, int slot, int layer, int expert) {
    lock_acquire(g->lk);
    if (g->req_count >= g->req_cap) { lock_release(g->lk); return 0; }
    UploadReq *r = &g->req[(g->req_head + g->req_count) % g->req_cap];
    r->slot = slot;
    r->layer = layer;
    r->expert = expert;
    g->req_count++;
    lock_notify_all(g->lk);
    lock_release(g->lk);
    g->st.uploads++;
    return 1;
}

/* Uploads run on their own queue, so nothing orders them before the kernels of the compute
 * queue: a driver may then serve a reused slot from a GPU cache that still holds the previous
 * expert (garbled text on AMD). The first command that may read newly READY slots waits on
 * these events. Returns the count; the caller passes *evs as a wait list, then calls
 * drop_uploads. */
static cl_uint take_uploads(Gpu *g, cl_event **evs) {
    lock_acquire(g->lk);
    *evs = g->fresh;
    const cl_uint n = (cl_uint)g->n_fresh;
    g->fresh = NULL;
    g->n_fresh = g->cap_fresh = 0;
    lock_release(g->lk);
    return n;
}

static void drop_uploads(Gpu *g, cl_event *evs, cl_uint n) {
    for (cl_uint i = 0; i < n; i++) g->cl.ReleaseEvent(evs[i]);
    free(evs);
}

static int pending_uploads(Gpu *g) {
    lock_acquire(g->lk);
    int n = g->req_count;
    lock_release(g->lk);
    return n;
}

/* ------------------------------------------------------------------ create / destroy */

static void submit_loop(void *arg);

static size_t ceil256(size_t v) { return (v + 255) & ~(size_t)255; }

static cl_mem mkbuf(Gpu *g, cl_ulong flags, size_t bytes, int *ok) {
    cl_int err = 0;
    cl_mem b = g->cl.CreateBuffer(g->ctx, flags, bytes ? bytes : 4, NULL, &err);
    if (!b) *ok = 0;
    return b;
}

static cl_ulong dense_plan(Gpu *g, int n_ctx, cl_ulong arena_cap, cl_ulong *big_used, cl_ulong *small_used);
static int dense_alloc(Gpu *g, int n_ctx, const cl_ulong *big_used, cl_ulong small_used, char *err, int errlen);
static void dense_free(Gpu *g);

Gpu *gpu_create(const Model *m, int device, int vram_mb, int max_tokens, int dense_ctx, const char *profile_path,
                char *info, int infolen) {
    info[0] = 0;
    /* a model without experts (LFM3, dense LFM2) runs entirely on the dense decode path */
    const int moe = m->n_expert > 0;
    if (m->n_layer > 64 || m->n_expert > 64) { snprintf(info, infolen, "too many layers or experts for the GPU kernels"); return NULL; }
    if (!moe && dense_ctx <= 0) { snprintf(info, infolen, "dense model: the GPU needs its dense layers (no --no-gpu-dense)"); return NULL; }
    if (m->dim > 4096 || m->ff_exp > 2048) { snprintf(info, infolen, "sizes not supported by the GPU kernels"); return NULL; }
    Gpu *g = (Gpu *)calloc(1, sizeof(Gpu));
    g->m = m;
    for (int i = 0; i < 64 * 64; i++) g->res_slot[i] = -1; /* no expert in VRAM (dense models: never) */
    if (!cl_load(&g->cl)) { snprintf(info, infolen, "OpenCL not available"); free(g); return NULL; }
    CLApi *cl = &g->cl;
    DevInfo devs[32];
    int nd = list_devices(cl, devs, 32);
    int pick = -1;
    if (device >= 0 && device < nd) pick = device;
    else
        for (int i = 0; i < nd; i++)
            if (devs[i].gpu && !devs[i].unified && (pick < 0 || devs[i].mem > devs[pick].mem)) pick = i;
    if (pick < 0) { snprintf(info, infolen, "no dedicated OpenCL GPU (use --gpu-device for an integrated one)"); free(g); return NULL; }
    DevInfo *dv = &devs[pick];
    g->dev = dv->id;
    cl->GetDeviceInfo(g->dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(g->local_mem), &g->local_mem, NULL);
    if (g->local_mem < 16384) g->local_mem = 16384; /* the OpenCL 1.2 minimum, if the query fails */
    cl_int err = 0;
    g->ctx = cl->CreateContext(NULL, 1, &g->dev, NULL, NULL, &err);
    if (!g->ctx) { snprintf(info, infolen, "clCreateContext failed (%d)", err); free(g); return NULL; }
    g->profiling = getenv("PITHOS_GPU_PROFILE") != NULL;
    g->q = cl->CreateCommandQueue(g->ctx, g->dev, g->profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    g->qup = cl->CreateCommandQueue(g->ctx, g->dev, 0, &err);
    if (!g->q || !g->qup) { snprintf(info, infolen, "creating the OpenCL queues failed (%d)", err); gpu_destroy(g); return NULL; }

    /* kernels */
    g->dp4a = device_has_dp4a(cl, g->dev) && !getenv("PITHOS_GPU_NODP4A");
    const char *src = KERNELS_CL;
    size_t srclen = strlen(src);
    g->prog = cl->CreateProgramWithSource(g->ctx, 1, &src, &srclen, &err);
    char opts[256];
    g->rpw = getenv("PITHOS_GPU_RPW") ? atoi(getenv("PITHOS_GPU_RPW")) : 4;
    if (g->rpw < 1 || g->rpw > 32) g->rpw = 4;
    snprintf(opts, sizeof(opts), "-cl-mad-enable -DRPW=%d%s", g->rpw, g->dp4a ? " -DUSE_DP4A" : "");
    if (cl->BuildProgram(g->prog, 1, &g->dev, opts, NULL, NULL) != 0) {
        char log[2048] = { 0 };
        cl->GetProgramBuildInfo(g->prog, g->dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        snprintf(info, infolen, "OpenCL kernel build failed: %.300s", log);
        gpu_destroy(g);
        return NULL;
    }
    g->k_gu = cl->CreateKernel(g->prog, "moe_gate_up", &err);
    g->k_down = cl->CreateKernel(g->prog, "moe_down", &err);
    if (!g->k_gu || !g->k_down) {
        snprintf(info, infolen, "creating the kernels failed (%d)", err);
        gpu_destroy(g);
        return NULL;
    }

    /* slot layout */
    const int D = m->dim, F = m->ff_exp;
    size_t slot = 0;
    int ok_layers = 0;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        if (!L->is_moe) continue;
        const uint32_t gt = L->gate_exps.type, ut = L->up_exps.type, dt = L->down_exps.type;
        if ((gt != GGML_Q4_K && gt != GGML_Q6_K) || ut != gt || (dt != GGML_Q4_K && dt != GGML_Q6_K) ||
            L->gate_exps.cols != D || L->down_exps.cols != F)
            continue;
        g->layer_ok[l] = 1;
        ok_layers++;
        g->gate_type[l] = (int)gt;
        g->down_type[l] = (int)dt;
        g->up_off[l] = ceil256(L->gate_exps.expert_stride);
        g->down_off[l] = g->up_off[l] + ceil256(L->up_exps.expert_stride);
        size_t total = g->down_off[l] + ceil256(L->down_exps.expert_stride);
        if (total > slot) slot = total;
    }
    if (moe && !ok_layers) { snprintf(info, infolen, "expert formats not supported by the GPU"); gpu_destroy(g); return NULL; }
    g->slot_bytes = slot;

    /* VRAM budget: leave room for the display, the driver and the small buffers */
    cl_ulong budget = vram_mb > 0 ? (cl_ulong)vram_mb << 20
                                  : (dv->mem > (cl_ulong)(900u << 20) ? dv->mem - (cl_ulong)(900u << 20) : dv->mem / 2);
    cl_ulong per_arena = dv->maxalloc ? dv->maxalloc : budget;
    /* dense decode path first: every byte of it is used for every token */
    char dense_note[160] = "";
    if (dense_ctx > 0) {
        cl_ulong big_used[5] = { 0 }, small_used = 0;
        cl_ulong need = dense_plan(g, dense_ctx, per_arena, big_used, &small_used);
        if (!need || g->dz_fail) snprintf(dense_note, sizeof(dense_note), " | dense layers on CPU (unsupported format)");
        else if (need + (cl_ulong)(512u << 20) > budget) snprintf(dense_note, sizeof(dense_note), " | dense layers on CPU (not enough VRAM)");
        else {
            char derr[128];
            if (dense_alloc(g, dense_ctx, big_used, small_used, derr, sizeof(derr))) {
                g->dense = 1;
                g->dense_bytes = need;
                g->lookahead = getenv("PITHOS_GPU_LOOKAHEAD") && atoi(getenv("PITHOS_GPU_LOOKAHEAD")) > 0;
                g->eager = !getenv("PITHOS_GPU_EAGER") || atoi(getenv("PITHOS_GPU_EAGER")) > 0;
                budget -= need;
                snprintf(dense_note, sizeof(dense_note), " | dense layers in VRAM (%.2f GB)", need / 1073741824.0);
            } else {
                dense_free(g);
                memset(g->dz, 0, sizeof(g->dz));
                g->n_dz = 0;
                g->dl = NULL;
                snprintf(dense_note, sizeof(dense_note), " | dense layers on CPU (%s)", derr);
            }
        }
    }
    if (!moe) {
        if (!g->dense) {
            snprintf(info, infolen, "%s%s", dv->name, dense_note[0] ? dense_note : " | dense layers on CPU");
            gpu_destroy(g);
            return NULL;
        }
        g->lk = lock_create(); /* no expert cache: only take_uploads uses it */
        snprintf(info, infolen, "%s | %s | all layers in VRAM (%.2f GB)", dv->name, g->dp4a ? "int8/dp4a" : "f32",
                 g->dense_bytes / 1073741824.0);
        return g;
    }
    g->spa = (int)(per_arena / slot);
    if (g->spa < 1) { snprintf(info, infolen, "not enough VRAM"); gpu_destroy(g); return NULL; }
    int want = (int)(budget / slot);
    if (want > m->n_layer * m->n_expert) want = m->n_layer * m->n_expert;
    for (int a = 0; a < MAX_ARENAS && g->n_slots < want; a++) {
        int ns = want - g->n_slots < g->spa ? want - g->n_slots : g->spa;
        g->arena[a] = cl->CreateBuffer(g->ctx, CL_MEM_READ_ONLY, (size_t)ns * slot, NULL, &err);
        if (!g->arena[a]) break;
        /* force the allocation now (drivers allocate lazily) */
        unsigned char z = 0;
        if (cl->EnqueueWriteBuffer(g->q, g->arena[a], 1, (size_t)ns * slot - 1, 1, &z, 0, NULL, NULL) != 0) {
            cl->ReleaseMemObject(g->arena[a]);
            g->arena[a] = NULL;
            break;
        }
        g->n_arenas++;
        g->n_slots += ns;
        if (ns < g->spa) break;
    }
    if (!g->n_slots) { snprintf(info, infolen, "VRAM allocation failed"); gpu_destroy(g); return NULL; }
    for (int a = g->n_arenas; a < MAX_ARENAS; a++) g->arena[a] = g->arena[0]; /* unused kernel args */

    /* work buffers sized for the largest batch */
    g->max_tok = max_tokens < 1 ? 1 : max_tokens;
    g->max_assign = g->max_tok * m->n_expert_used;
    int ok = 1;
    const size_t meta_bytes = sizeof(int) * 2 * g->max_assign + sizeof(float) * D * g->max_tok;
    g->d_meta = mkbuf(g, CL_MEM_READ_ONLY, meta_bytes, &ok);
    g->d_gu = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * 2 * F * g->max_assign, &ok);
    g->d_y = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * D * g->max_assign, &ok);
    g->p_meta = mkbuf(g, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, meta_bytes, &ok);
    g->p_y = mkbuf(g, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * D * g->max_assign, &ok);
    if (ok) {
        g->h_meta = (int *)cl->EnqueueMapBuffer(g->q, g->p_meta, 1, CL_MAP_WRITE, 0, meta_bytes, 0, NULL, NULL, &err);
        g->h_y = (float *)cl->EnqueueMapBuffer(g->q, g->p_y, 1, CL_MAP_READ, 0, sizeof(float) * D * g->max_assign, 0, NULL, NULL, &err);
        if (g->h_meta) {
            g->h_slots = g->h_meta;
            g->h_xidx = g->h_meta + g->max_assign;
            g->h_x = (float *)(g->h_meta + 2 * g->max_assign);
        }
    }
    if (!ok || !g->h_meta || !g->h_y) {
        snprintf(info, infolen, "GPU buffer allocation failed");
        gpu_destroy(g);
        return NULL;
    }

    g->slot_owner = (int *)malloc(sizeof(int) * g->n_slots);
    g->slot_state = (volatile int *)calloc((size_t)g->n_slots, sizeof(int));
    for (int s = 0; s < g->n_slots; s++) g->slot_owner[s] = -1;
    for (int i = 0; i < 64 * 64; i++) g->res_slot[i] = -1;

    g->req_cap = g->n_slots + 64;
    g->req = (UploadReq *)calloc((size_t)g->req_cap, sizeof(UploadReq));
    g->lk = lock_create();
    g->th = thread_create(upload_loop, g);
    g->spin_limit = 20000; /* ~1 ms: covers the gap between two MoE layers */
    g->sub_lk = lock_create();
    g->sub_th = thread_create(submit_loop, g);

    /* warm start from a saved usage profile */
    if (profile_path) {
        snprintf(g->profile, sizeof(g->profile), "%s", profile_path);
        FILE *f = fopen(profile_path, "rb");
        if (f) {
            uint32_t hdr[3];
            if (fread(hdr, 4, 3, f) == 3 && hdr[0] == 0x4C464D50u && hdr[1] == (uint32_t)m->n_layer && hdr[2] == (uint32_t)m->n_expert) {
                for (int l = 0; l < m->n_layer; l++)
                    if (fread(g->score + l * 64, sizeof(float), (size_t)m->n_expert, f) != (size_t)m->n_expert) break;
            }
            fclose(f);
        }
    }
    /* with a profile, queue the hottest experts right away: VRAM fills in the background */
    gpu_rebalance(g, g->n_slots);
    double gb = (double)g->n_slots * slot / 1073741824.0;
    snprintf(info, infolen, "%s | %s | %d expert slots (%.2f GB VRAM, %.0f%% of the experts)%s", dv->name,
             g->dp4a ? "int8/dp4a" : "f32", g->n_slots, gb, 100.0 * g->n_slots / (ok_layers * m->n_expert), dense_note);
    g->st.slots = g->n_slots;
    return g;
}

void gpu_destroy(Gpu *g) {
    if (!g) return;
    if (g->profiling && g->pcount) {
        static const char *names[N_EV] = { "gate_up", "down", "read" };
        fprintf(stderr, "GPU device time (us/launch):");
        for (int i = 0; i < N_EV; i++) fprintf(stderr, " %s %.1f", names[i], 1e6 * g->ptime[i] / g->pcount);
        fprintf(stderr, "\n");
    }
    if (g->sub_th) {
        lock_acquire(g->sub_lk);
        g->sub_stop = 1;
        lock_notify_all(g->sub_lk);
        lock_release(g->sub_lk);
        thread_join(g->sub_th);
    }
    if (g->sub_lk) lock_destroy(g->sub_lk);
    if (g->th) {
        lock_acquire(g->lk);
        g->stop = 1;
        lock_notify_all(g->lk);
        lock_release(g->lk);
        thread_join(g->th);
    }
    if (g->profile[0] && g->m) {
        FILE *f = fopen(g->profile, "wb");
        if (f) {
            uint32_t hdr[3] = { 0x4C464D50u, (uint32_t)g->m->n_layer, (uint32_t)g->m->n_expert };
            fwrite(hdr, 4, 3, f);
            for (int l = 0; l < g->m->n_layer; l++) fwrite(g->score + l * 64, sizeof(float), (size_t)g->m->n_expert, f);
            fclose(f);
        }
    }
    CLApi *cl = &g->cl;
    if (g->q) cl->Finish(g->q);
    for (int i = 0; i < g->n_fresh; i++) cl->ReleaseEvent(g->fresh[i]);
    free(g->fresh);
    if (g->dl) dense_free(g);
    cl_mem bufs[] = { g->d_meta, g->d_gu, g->d_y, g->p_meta, g->p_y };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]) cl->ReleaseMemObject(bufs[i]);
    for (int a = 0; a < g->n_arenas; a++) if (g->arena[a]) cl->ReleaseMemObject(g->arena[a]);
    cl_kernel ks[] = { g->k_gu, g->k_down };
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++) if (ks[i]) cl->ReleaseKernel(ks[i]);
    if (g->prog) cl->ReleaseProgram(g->prog);
    if (g->qup) cl->ReleaseCommandQueue(g->qup);
    if (g->q) cl->ReleaseCommandQueue(g->q);
    if (g->ctx) cl->ReleaseContext(g->ctx);
    if (g->lk) lock_destroy(g->lk);
    free(g->req);
    free(g->slot_owner);
    free((void *)g->slot_state);
    free(g);
}

/* ------------------------------------------------------------------ compute */

int gpu_resident(const Gpu *g, int layer, int expert) {
    const int s = g->res_slot[layer * 64 + expert];
    return s >= 0 && state_load(&g->slot_state[s]) == SLOT_READY;
}

static void launch(Gpu *g, cl_kernel k, cl_uint dims, const size_t *gws, const size_t *lws, int ev, const char *what) {
    check(g, g->cl.EnqueueNDRangeKernel(g->q, k, dims, NULL, gws, lws, 0, NULL, g->profiling ? &g->ev[ev] : NULL), what);
    if (getenv("PITHOS_GPU_SYNC")) check(g, g->cl.Finish(g->q), what);
}

static void set_slot_args(Gpu *g, cl_kernel k, const cl_ulong *sb) {
    for (int a = 0; a < MAX_ARENAS; a++) g->cl.SetKernelArg(k, (cl_uint)a, sizeof(cl_mem), &g->arena[a]);
    g->cl.SetKernelArg(k, 4, sizeof(int), &g->spa);
    g->cl.SetKernelArg(k, 5, sizeof(cl_ulong), sb);
    g->cl.SetKernelArg(k, 6, sizeof(cl_mem), &g->d_meta);
}

/* Runs on the submission thread: one copy in, two kernels, one copy out, then wait. */
static void submit_job(Gpu *g) {
    CLApi *cl = &g->cl;
    const Model *m = g->m;
    const int D = m->dim, F = m->ff_exp, layer = g->job_layer, n_assign = g->cur_assign;
    /* one copy: slots, token indices and hidden states (prefix of the meta layout) */
    const size_t meta_used = sizeof(int) * 2 * g->max_assign + sizeof(float) * D * g->job_ntok;
    cl_event *up;
    const cl_uint n_up = take_uploads(g, &up);
    check(g, cl->EnqueueWriteBuffer(g->q, g->d_meta, 0, 0, meta_used, g->h_meta, n_up, up, NULL), "write meta");
    drop_uploads(g, up, n_up);

    const cl_ulong sb = g->slot_bytes, upo = g->up_off[layer], dno = g->down_off[layer];
    const int gt = g->gate_type[layer], dt = g->down_type[layer];
    const size_t lws[2] = { WG, 1 };
    const size_t gws_gu[2] = { (size_t)((F + 8 * g->rpw - 1) / (8 * g->rpw)) * WG, (size_t)(2 * n_assign) };
    const size_t gws_dn[2] = { (size_t)((D + 8 * g->rpw - 1) / (8 * g->rpw)) * WG, (size_t)n_assign };
    const size_t red_bytes = sizeof(float) * WG * g->rpw;

    cl_kernel k = g->k_gu;
    set_slot_args(g, k, &sb);
    cl->SetKernelArg(k, 7, sizeof(int), &g->max_assign);
    cl->SetKernelArg(k, 8, sizeof(int), &D);
    cl->SetKernelArg(k, 9, sizeof(int), &F);
    cl->SetKernelArg(k, 10, sizeof(cl_ulong), &upo);
    cl->SetKernelArg(k, 11, sizeof(int), &gt);
    cl->SetKernelArg(k, 12, sizeof(cl_mem), &g->d_gu);
    const cl_ulong x_off = 2 * (cl_ulong)g->max_assign; /* x follows slots and xidx in meta */
    if (g->dp4a) {
        cl->SetKernelArg(k, 13, (size_t)D, NULL);                      /* int8 x */
        cl->SetKernelArg(k, 14, sizeof(float) * (D / 32), NULL);       /* block scales */
        cl->SetKernelArg(k, 15, red_bytes, NULL);
        cl->SetKernelArg(k, 16, sizeof(cl_mem), &g->d_meta);
        cl->SetKernelArg(k, 17, sizeof(cl_ulong), &x_off);
    } else {
        cl->SetKernelArg(k, 13, sizeof(float) * D, NULL);
        cl->SetKernelArg(k, 14, red_bytes, NULL);
        cl->SetKernelArg(k, 15, sizeof(cl_mem), &g->d_meta);
        cl->SetKernelArg(k, 16, sizeof(cl_ulong), &x_off);
    }
    launch(g, k, 2, gws_gu, lws, 0, "moe_gate_up");

    k = g->k_down;
    set_slot_args(g, k, &sb);
    cl->SetKernelArg(k, 7, sizeof(int), &F);
    cl->SetKernelArg(k, 8, sizeof(int), &D);
    cl->SetKernelArg(k, 9, sizeof(cl_ulong), &dno);
    cl->SetKernelArg(k, 10, sizeof(int), &dt);
    cl->SetKernelArg(k, 11, sizeof(cl_mem), &g->d_gu);
    cl->SetKernelArg(k, 12, sizeof(cl_mem), &g->d_y);
    if (g->dp4a) {
        cl->SetKernelArg(k, 13, (size_t)F, NULL);
        cl->SetKernelArg(k, 14, sizeof(float) * (F / 32), NULL);
        cl->SetKernelArg(k, 15, red_bytes, NULL);
    } else {
        cl->SetKernelArg(k, 13, sizeof(float) * F, NULL);
        cl->SetKernelArg(k, 14, red_bytes, NULL);
    }
    launch(g, k, 2, gws_dn, lws, 1, "moe_down");
    check(g, cl->EnqueueReadBuffer(g->q, g->d_y, 0, 0, sizeof(float) * D * n_assign, g->h_y, 0, NULL,
                                   g->profiling ? &g->ev[2] : NULL), "read y");
    check(g, cl->Finish(g->q), "esecuzione");
    if (g->profiling) {
        for (int i = 0; i < N_EV; i++) {
            if (!g->ev[i]) continue;
            cl_ulong t0 = 0, t1 = 0;
            cl->GetEventProfilingInfo(g->ev[i], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
            cl->GetEventProfilingInfo(g->ev[i], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
            g->ptime[i] += (double)(t1 - t0) * 1e-9;
            cl->ReleaseEvent(g->ev[i]);
            g->ev[i] = NULL;
        }
        g->pcount++;
    }
}

static void submit_loop(void *arg) {
    Gpu *g = (Gpu *)arg;
    int seen = 0;
    for (;;) {
        int spins = 0;
        while (state_load(&g->job_seq) == seen && !g->sub_stop) {
            if (++spins < g->spin_limit) { cpu_relax(); continue; }
            lock_acquire(g->sub_lk);
            while (state_load(&g->job_seq) == seen && !g->sub_stop) lock_wait(g->sub_lk);
            lock_release(g->sub_lk);
        }
        if (g->sub_stop) return;
        seen = state_load(&g->job_seq);
        submit_job(g);
        state_store(&g->done_seq, seen);
    }
}

void gpu_moe_begin(Gpu *g, int layer, const float *x, int n_tok, const int *a_expert, const int *a_tok, int n_assign) {
    const int D = g->m->dim;
    if (n_assign <= 0) return;
    if (n_assign > g->max_assign) n_assign = g->max_assign;
    if (n_tok > g->max_tok) n_tok = g->max_tok;
    for (int i = 0; i < n_assign; i++) {
        g->h_slots[i] = g->res_slot[layer * 64 + a_expert[i]];
        g->h_xidx[i] = a_tok[i];
    }
    memcpy(g->h_x, x, sizeof(float) * D * n_tok);
    g->job_layer = layer;
    g->job_ntok = n_tok;
    g->cur_assign = n_assign;
    g->busy = 1;
    g->st.gpu_expert_calls += (uint64_t)n_assign;
    lock_acquire(g->sub_lk);
    state_store(&g->job_seq, g->job_seq + 1);
    lock_notify_all(g->sub_lk);
    lock_release(g->sub_lk);
}

void gpu_moe_end(Gpu *g, float *y) {
    if (!g->busy) return;
    double t0 = time_sec();
    while (state_load(&g->done_seq) != g->job_seq) cpu_relax();
    g->st.t_wait += time_sec() - t0;
    g->busy = 0;
    memcpy(y, g->h_y, sizeof(float) * g->m->dim * g->cur_assign);
}

/* ------------------------------------------------------------------ cache policy */

void gpu_note_usage(Gpu *g, int layer, int expert) {
    if (layer < 64 && expert < 64) g->score[layer * 64 + expert] += 1.f;
}

void gpu_rebalance(Gpu *g, int max_uploads) {
    const Model *m = g->m;
    /* exponential decay: half-life ~350 tokens */
    for (int l = 0; l < m->n_layer; l++)
        for (int e = 0; e < m->n_expert; e++) g->score[l * 64 + e] *= 0.998f;
    int budget = max_uploads - pending_uploads(g);
    while (budget-- > 0) {
        /* hottest expert not in VRAM */
        int best = -1;
        for (int l = 0; l < m->n_layer; l++) {
            if (!g->layer_ok[l]) continue;
            for (int e = 0; e < m->n_expert; e++) {
                const int i = l * 64 + e;
                if (g->res_slot[i] < 0 && (best < 0 || g->score[i] > g->score[best])) best = i;
            }
        }
        if (best < 0 || g->score[best] <= 0.f) return;
        /* a free slot, or the coldest ready one if clearly colder */
        int victim = -1;
        for (int s = 0; s < g->n_slots; s++) {
            const int st = state_load(&g->slot_state[s]);
            if (g->slot_owner[s] >= 0 && st == SLOT_FREE) { /* failed upload */
                g->res_slot[g->slot_owner[s]] = -1;
                g->slot_owner[s] = -1;
            }
            if (g->slot_owner[s] < 0) { victim = s; break; }
            if (st != SLOT_READY) continue;
            if (victim < 0 || g->score[g->slot_owner[s]] < g->score[g->slot_owner[victim]]) victim = s;
        }
        if (victim < 0) return;
        if (g->slot_owner[victim] >= 0) {
            if (g->score[best] < g->score[g->slot_owner[victim]] * 1.5f + 1.f) return;
            g->res_slot[g->slot_owner[victim]] = -1;
        }
        g->slot_owner[victim] = best;
        g->res_slot[best] = victim;
        state_store(&g->slot_state[victim], SLOT_LOADING);
        if (!request_upload(g, victim, best / 64, best % 64)) {
            g->res_slot[best] = -1;
            g->slot_owner[victim] = -1;
            state_store(&g->slot_state[victim], SLOT_FREE);
            return;
        }
    }
}

void gpu_stats(const Gpu *g, GpuStats *s) {
    *s = g->st;
    s->used = s->ready = 0;
    for (int i = 0; i < g->n_slots; i++) {
        if (g->slot_owner[i] >= 0) s->used++;
        if (state_load(&g->slot_state[i]) == SLOT_READY) s->ready++;
    }
}

int gpu_pending(Gpu *g) {
    int n = 0; /* queued and in-flight uploads are both in the LOADING state */
    for (int s = 0; s < g->n_slots; s++)
        if (state_load(&g->slot_state[s]) == SLOT_LOADING) n++;
    return n;
}

/* ------------------------------------------------------------------ dense decode path
 *
 * All non-expert weights (projections of the convolution and attention blocks, dense FFN,
 * router, output) are kept in VRAM together with the KV cache, the convolution state and the
 * residual stream. A decoded token then needs the CPU only for the experts that are not
 * resident: after each MoE router the GPU sends the hidden state and the top-k choice to the
 * host (8 KB), runs its resident experts while the CPU computes the others, and adds the CPU
 * part when it arrives. */

#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11D3

static size_t dense_rows_bytes(const Mat *w) { return w->expert_stride; }

/* place one tensor: big ones in the weight arenas, f32 vectors in the small buffer */
static void dense_place(Gpu *g, DT *t, const Mat *w, const void *vec, size_t vec_bytes, cl_ulong *big_used, cl_ulong *small_used,
                        cl_ulong arena_cap) {
    memset(t, 0, sizeof(*t));
    if (vec) {
        t->buf = -1;
        t->off = *small_used;
        t->src = (const uint8_t *)vec;
        t->bytes = vec_bytes;
        *small_used += ceil256(vec_bytes);
        return;
    }
    t->type = (int)w->type;
    t->rows = w->rows;
    t->cols = w->cols;
    t->src = w->data;
    t->bytes = dense_rows_bytes(w);
    int a = g->n_dz > 0 ? g->n_dz - 1 : 0;
    if (g->n_dz == 0) g->n_dz = 1;
    if (big_used[a] + t->bytes > arena_cap) a = g->n_dz++;
    if (a >= 4 || t->bytes > arena_cap) { g->dz_fail = 1; a = 0; }
    t->buf = a;
    t->off = big_used[a];
    big_used[a] += ceil256(t->bytes);
}

static int dense_type_ok(const Mat *w) { return w->type == GGML_Q4_K || w->type == GGML_Q6_K; }

/* Lays out and measures the dense weights. Returns the VRAM bytes needed (0 = unsupported). */
static cl_ulong dense_plan(Gpu *g, int n_ctx, cl_ulong arena_cap, cl_ulong *big_used, cl_ulong *small_used) {
    const Model *m = g->m;
    if (m->n_layer > 64 || m->head_dim > 256 || m->head_dim % 4 || (m->n_head % (m->n_kv_head ? m->n_kv_head : 1)) ||
        m->n_expert_used > 8 || !dense_type_ok(&m->output) || m->dim % 256)
        return 0;
    g->dl = (DL *)calloc((size_t)m->n_layer, sizeof(DL));
    const size_t D4 = sizeof(float) * (size_t)m->dim;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        DL *d = &g->dl[l];
        dense_place(g, &d->attn_norm, NULL, L->attn_norm, D4, big_used, small_used, arena_cap);
        dense_place(g, &d->ffn_norm, NULL, L->ffn_norm, D4, big_used, small_used, arena_cap);
        if (L->is_attn) {
            if (!dense_type_ok(&L->wq) || !dense_type_ok(&L->wk) || !dense_type_ok(&L->wv) || !dense_type_ok(&L->wo)) return 0;
            dense_place(g, &d->wq, &L->wq, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->wk, &L->wk, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->wv, &L->wv, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->wo, &L->wo, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->qn, NULL, L->q_norm, sizeof(float) * (size_t)m->head_dim, big_used, small_used, arena_cap);
            dense_place(g, &d->kn, NULL, L->k_norm, sizeof(float) * (size_t)m->head_dim, big_used, small_used, arena_cap);
            if (L->attn_gate.data) {
                if (!dense_type_ok(&L->attn_gate)) return 0;
                dense_place(g, &d->wgate, &L->attn_gate, NULL, 0, big_used, small_used, arena_cap);
            }
        } else {
            if (!dense_type_ok(&L->in_proj) || !dense_type_ok(&L->out_proj)) return 0;
            dense_place(g, &d->in_proj, &L->in_proj, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->out_proj, &L->out_proj, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->conv_w, NULL, L->conv_w, sizeof(float) * (size_t)m->dim * m->l_cache, big_used, small_used, arena_cap);
        }
        if (L->is_moe) {
            if (L->router.type != GGML_F32 || !g->layer_ok[l]) return 0;
            dense_place(g, &d->router, NULL, L->router.data, (size_t)L->router.rows * L->router.row_bytes, big_used, small_used,
                        arena_cap);
            if (L->exp_bias) dense_place(g, &d->bias, NULL, L->exp_bias, sizeof(float) * (size_t)m->n_expert, big_used, small_used, arena_cap);
        } else {
            if (!dense_type_ok(&L->w_gate) || !dense_type_ok(&L->w_up) || !dense_type_ok(&L->w_down)) return 0;
            dense_place(g, &d->wg, &L->w_gate, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->wu, &L->w_up, NULL, 0, big_used, small_used, arena_cap);
            dense_place(g, &d->wd, &L->w_down, NULL, 0, big_used, small_used, arena_cap);
            if (L->w_gate.rows > g->dff) g->dff = L->w_gate.rows;
        }
        if (L->post_op_norm) {
            if (L->is_moe) return 0; /* not wired into the MoE combine */
            dense_place(g, &d->post_op, NULL, L->post_op_norm, D4, big_used, small_used, arena_cap);
            dense_place(g, &d->post_ffn, NULL, L->post_ffn_norm, D4, big_used, small_used, arena_cap);
        }
    }
    dense_place(g, &g->d_out, &m->output, NULL, 0, big_used, small_used, arena_cap);
    dense_place(g, &g->d_fnorm, NULL, m->final_norm, D4, big_used, small_used, arena_cap);
    if (g->n_dz > 4) return 0;
    cl_ulong total = *small_used;
    for (int a = 0; a < g->n_dz; a++) total += big_used[a];
    const int kvd = m->n_kv_head * m->head_dim;
    g->kv_bytes = sizeof(uint16_t) * (cl_ulong)m->n_attn * n_ctx * kvd;
    total += 2 * g->kv_bytes;
    total += sizeof(float) * ((cl_ulong)m->n_head * n_ctx + (cl_ulong)m->n_vocab + 18 * (cl_ulong)m->dim + 2 * (cl_ulong)g->dff);
    total += sizeof(float) * (cl_ulong)(m->n_conv ? m->n_conv : 1) * (m->l_cache - 1) * m->dim;
    return total;
}

static int dense_alloc(Gpu *g, int n_ctx, const cl_ulong *big_used, cl_ulong small_used, char *err, int errlen) {
    const Model *m = g->m;
    CLApi *cl = &g->cl;
    cl_int e = 0;
    int ok = 1;
    const int D = m->dim, kvd = m->n_kv_head * m->head_dim;
    const int K = m->n_expert_used ? m->n_expert_used : 1, F = m->ff_exp ? m->ff_exp : 1; /* dense models: unused */
    g->n_ctx = n_ctx;
    for (int a = 0; a < g->n_dz; a++) g->dz[a] = mkbuf(g, CL_MEM_READ_ONLY, big_used[a], &ok);
    g->dsmall = mkbuf(g, CL_MEM_READ_ONLY, small_used, &ok);
    g->dx = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * D, &ok);
    size_t bigw = 3 * (size_t)D;
    if ((size_t)D + 2 * kvd > bigw) bigw = (size_t)D + 2 * kvd;
    g->dbig = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * bigw, &ok);
    g->dtmp = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * D, &ok);
    g->dfg = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * 2 * (g->dff ? g->dff : 1), &ok); /* gate | up */
    g->dscores = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * (size_t)m->n_head * n_ctx, &ok);
    g->dkc = mkbuf(g, CL_MEM_READ_WRITE, g->kv_bytes ? g->kv_bytes : 4, &ok);
    g->dvc = mkbuf(g, CL_MEM_READ_WRITE, g->kv_bytes ? g->kv_bytes : 4, &ok);
    g->conv_bytes = sizeof(float) * (size_t)(m->n_conv ? m->n_conv : 1) * (m->l_cache - 1) * D;
    g->dconv = mkbuf(g, CL_MEM_READ_WRITE, g->conv_bytes, &ok);
    g->drope = mkbuf(g, CL_MEM_READ_ONLY, sizeof(float) * 2 * (size_t)m->head_dim, &ok); /* full | sliding-window */
    g->dgate = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * (size_t)m->n_head, &ok);
    g->dpost = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * D, &ok);
    g->dres = mkbuf(g, CL_MEM_READ_ONLY, sizeof(int) * 64 * 64, &ok);
    g->dlog = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * 64, &ok);
    g->dcount = mkbuf(g, CL_MEM_READ_WRITE, sizeof(int), &ok);
    g->dmeta2 = mkbuf(g, CL_MEM_READ_WRITE, sizeof(int) * 2 * K, &ok);
    g->dout = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * 2 * D + sizeof(int) * 3 * K, &ok); /* x | h | ret */
    g->dgu2 = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * 2 * F * K, &ok);
    g->dy2 = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * D * K, &ok);
    g->dycpu = mkbuf(g, CL_MEM_READ_ONLY, sizeof(float) * D, &ok);
    g->dlogits = mkbuf(g, CL_MEM_READ_WRITE, sizeof(float) * (size_t)m->n_vocab, &ok);
    /* pinned staging */
    const size_t in_bytes = sizeof(float) * (D + 2 * m->head_dim) + sizeof(int) * 64 * 64;
    const size_t out_bytes = sizeof(float) * 2 * D + sizeof(int) * 3 * K;
    g->pin_in = mkbuf(g, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, in_bytes, &ok);
    g->pin_out = mkbuf(g, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, out_bytes, &ok);
    g->pin_y = mkbuf(g, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * D, &ok);
    g->pin_logits = mkbuf(g, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * (size_t)m->n_vocab, &ok);
    if (!ok) { snprintf(err, (size_t)errlen, "VRAM allocation failed"); return 0; }
    g->h_in = (float *)cl->EnqueueMapBuffer(g->q, g->pin_in, 1, CL_MAP_WRITE, 0, in_bytes, 0, NULL, NULL, &e);
    g->h_out = (float *)cl->EnqueueMapBuffer(g->q, g->pin_out, 1, CL_MAP_READ, 0, out_bytes, 0, NULL, NULL, &e);
    g->h_yc = (float *)cl->EnqueueMapBuffer(g->q, g->pin_y, 1, CL_MAP_WRITE, 0, sizeof(float) * D, 0, NULL, NULL, &e);
    g->h_logits = (float *)cl->EnqueueMapBuffer(g->q, g->pin_logits, 1, CL_MAP_READ, 0, sizeof(float) * (size_t)m->n_vocab, 0,
                                               NULL, NULL, &e);
    if (!g->h_in || !g->h_out || !g->h_yc || !g->h_logits) { snprintf(err, (size_t)errlen, "pinned buffers failed"); return 0; }
    /* upload the weights (read from the mapped model file) */
    for (int l = 0; l <= m->n_layer; l++) {
        DT *ts[24];
        int nt = 0;
        if (l < m->n_layer) {
            DL *d = &g->dl[l];
            DT *all[] = { &d->attn_norm, &d->ffn_norm, &d->wq, &d->wk, &d->wv, &d->wo, &d->qn, &d->kn, &d->in_proj,
                          &d->out_proj, &d->conv_w, &d->router, &d->bias, &d->wg, &d->wu, &d->wd,
                          &d->wgate, &d->post_op, &d->post_ffn };
            for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) if (all[i]->src) ts[nt++] = all[i];
        } else {
            ts[nt++] = &g->d_out;
            ts[nt++] = &g->d_fnorm;
        }
        for (int i = 0; i < nt; i++) {
            cl_mem b = ts[i]->buf < 0 ? g->dsmall : g->dz[ts[i]->buf];
            if (cl->EnqueueWriteBuffer(g->q, b, 1, ts[i]->off, ts[i]->bytes, ts[i]->src, 0, NULL, NULL) != 0) {
                snprintf(err, (size_t)errlen, "weight upload failed");
                return 0;
            }
        }
    }
    static const int zero = 0;
    if (cl->EnqueueWriteBuffer(g->q, g->dcount, 1, 0, sizeof(int), &zero, 0, NULL, NULL) != 0) {
        snprintf(err, (size_t)errlen, "counter init failed");
        return 0;
    }
    g->k_mv = cl->CreateKernel(g->prog, "matvec", &e);
    g->k_conv = cl->CreateKernel(g->prog, "shortconv", &e);
    g->k_silu = cl->CreateKernel(g->prog, "silu_mul", &e);
    g->k_attn = cl->CreateKernel(g->prog, "attn_decode", &e);
    g->k_route = cl->CreateKernel(g->prog, "moe_route", &e);
    g->k_comb = cl->CreateKernel(g->prog, "moe_combine", &e);
    g->k_addn = cl->CreateKernel(g->prog, "add_norm", &e);
    if (!g->k_mv || !g->k_conv || !g->k_silu || !g->k_attn || !g->k_route || !g->k_comb || !g->k_addn) {
        snprintf(err, (size_t)errlen, "dense kernels missing (%d)", e);
        return 0;
    }
    return 1;
}

static void dense_free(Gpu *g) {
    CLApi *cl = &g->cl;
    cl_mem bufs[] = { g->dsmall, g->dx, g->dbig, g->dtmp, g->dfg, g->dscores, g->dkc, g->dvc, g->dconv,
                      g->drope, g->dres, g->dlog, g->dcount, g->dmeta2, g->dout, g->dgu2, g->dy2, g->dycpu, g->dlogits, g->pin_in,
                      g->pin_out, g->pin_y, g->pin_logits, g->dgate, g->dpost };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]) cl->ReleaseMemObject(bufs[i]);
    for (int a = 0; a < g->n_dz; a++) if (g->dz[a]) cl->ReleaseMemObject(g->dz[a]);
    cl_kernel ks[] = { g->k_mv, g->k_conv, g->k_silu, g->k_attn, g->k_route, g->k_comb, g->k_addn };
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++) if (ks[i]) cl->ReleaseKernel(ks[i]);
    free(g->dl);
}

int gpu_dense_on(const Gpu *g) { return g && g->dense; }


static void arg(Gpu *g, cl_kernel k, cl_uint i, size_t sz, const void *v) { g->cl.SetKernelArg(k, i, sz, v); }
#define ARGM(k, i, m) arg(g, k, i, sizeof(cl_mem), &(m))
#define ARGI(k, i, v) do { const int _v = (v); arg(g, k, i, sizeof(int), &_v); } while (0)
#define ARGU(k, i, v) do { const cl_ulong _v = (cl_ulong)(v); arg(g, k, i, sizeof(cl_ulong), &_v); } while (0)
#define ARGF(k, i, v) do { const float _v = (v); arg(g, k, i, sizeof(float), &_v); } while (0)

static void run1(Gpu *g, cl_kernel k, size_t gws, size_t lws, const char *what) {
    const size_t gw[1] = { gws }, lw[1] = { lws };
    check(g, g->cl.EnqueueNDRangeKernel(g->q, k, 1, NULL, gw, lws ? lw : NULL, 0, NULL, NULL), what);
    if (g->eager) g->cl.Flush(g->q); /* the GPU starts while the host queues the next command */
}

/* y[y_off[i]..] (+)= W_i x for up to three weights with the same K (Q4_K / Q6_K in VRAM), in one
 * launch; norm != NULL: x is RMS-normalized with those weights first */
static void dense_mv(Gpu *g, int n, const DT *const *w, const int *y_off, cl_mem x, const DT *norm, cl_mem y, int accumulate) {
    cl_kernel k = g->k_mv;
    const int K = w[0]->cols, per = 8 * g->rpw;
    const size_t red = sizeof(float) * WG * g->rpw;
    /* the f32 build keeps x in local memory: split long rows (never normalized ones) into
     * chunks that fit next to the reduction buffer, in whole super-blocks; 1 KB stays free
     * for the driver (NVIDIA rejects a launch that takes all 48 KB) */
    int fit = (int)((g->local_mem - red - 1024) / sizeof(float)) & ~255;
    if (fit > 8192) fit = 8192;
    const int kmax = g->dp4a || norm || fit >= K ? K : fit;
    cl_int wt[4] = { 0 }, rows[4] = { 0 }, yo[4] = { 0 };
    size_t wgs = 0;
    for (int i = 0; i < 3; i++) {
        const DT *t = w[i < n ? i : 0];
        ARGM(k, (cl_uint)i, g->dz[t->buf]);
        ARGU(k, 3 + (cl_uint)i, t->off);
        if (i < n) {
            wt[i] = t->type;
            rows[i] = t->rows;
            yo[i] = y_off[i];
            wgs += (size_t)((t->rows + per - 1) / per);
        }
    }
    arg(g, k, 6, sizeof(wt), wt);
    arg(g, k, 7, sizeof(rows), rows);
    arg(g, k, 8, sizeof(yo), yo);
    ARGI(k, 9, K);
    ARGM(k, 12, x);
    ARGM(k, 13, g->dsmall);
    ARGI(k, 14, norm != NULL);
    ARGU(k, 15, norm ? norm->off : 0);
    ARGF(k, 16, g->m->eps);
    ARGM(k, 17, y);
    for (int k0 = 0; k0 < K; k0 += kmax) {
        const int klen = K - k0 < kmax ? K - k0 : kmax;
        ARGI(k, 10, k0);
        ARGI(k, 11, klen);
        ARGI(k, 18, accumulate || k0 > 0);
        if (g->dp4a) {
            arg(g, k, 19, (size_t)((klen + 15) & ~15), NULL);
            arg(g, k, 20, sizeof(float) * (size_t)(klen / 32 + 1), NULL);
            arg(g, k, 21, red, NULL);
        } else {
            arg(g, k, 19, sizeof(float) * (size_t)klen, NULL);
            arg(g, k, 20, red, NULL);
        }
        run1(g, k, wgs * WG, WG, "matvec");
    }
}

static void dense_mv1(Gpu *g, const DT *w, cl_mem x, const DT *norm, cl_mem y, int accumulate) {
    static const int y0 = 0;
    dense_mv(g, 1, &w, &y0, x, norm, y, accumulate);
}

static void wait_event(Gpu *g, cl_event ev) {
    /* spin on the status: a blocking wait sleeps in the driver and wakes up late */
    double t0 = time_sec();
    for (;;) {
        cl_int st = 1;
        if (g->cl.GetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL) != 0 || st <= 0) break;
        cpu_relax();
    }
    g->st.t_wait += time_sec() - t0;
}

static void dense_enqueue_upto(Gpu *g, int upto);
static void dense_combine(Gpu *g, int use_cpu);

static int next_moe(const Model *m, int l) {
    for (int i = l + 1; i < m->n_layer; i++) if (m->layers[i].is_moe) return i;
    return m->n_layer;
}

void gpu_dense_begin(Gpu *g, const float *x, const float *rope_cs, const float *rope_swa, int pos) {
    const Model *m = g->m;
    const int D = m->dim, hd = m->head_dim;
    memcpy(g->h_in, x, sizeof(float) * D);
    memcpy(g->h_in + D, rope_cs, sizeof(float) * hd);
    memcpy(g->h_in + D + hd, rope_swa ? rope_swa : rope_cs, sizeof(float) * hd);
    int *res = (int *)(g->h_in + D + 2 * hd);
    for (int i = 0; i < 64 * 64; i++) {
        const int s = g->res_slot[i];
        res[i] = s >= 0 && state_load(&g->slot_state[s]) == SLOT_READY ? s : -1;
    }
    CLApi *cl = &g->cl;
    cl_event *up;
    const cl_uint n_up = take_uploads(g, &up);
    check(g, cl->EnqueueWriteBuffer(g->q, g->dx, 0, 0, sizeof(float) * D, g->h_in, n_up, up, NULL), "write x");
    drop_uploads(g, up, n_up);
    check(g, cl->EnqueueWriteBuffer(g->q, g->drope, 0, 0, sizeof(float) * 2 * hd, g->h_in + D, 0, NULL, NULL), "write rope");
    check(g, cl->EnqueueWriteBuffer(g->q, g->dres, 0, 0, sizeof(int) * 64 * 64, res, 0, NULL, NULL), "write res");
    g->cur_pos = pos;
    g->enq = 0;
    g->enq_logits = 0;
    g->moe_open = -1;
    /* keep one layer of lookahead in the queue beyond the next point where the host waits */
    dense_enqueue_upto(g, next_moe(m, -1) + 1 + g->lookahead);
}

/* x (+)= W in: straight into the residual stream, or (LFM3) through dpost and its RMSNorm */
static void block_out(Gpu *g, const DT *w, cl_mem in, const DT *post) {
    if (!post->src) {
        dense_mv1(g, w, in, NULL, g->dx, 1);
        return;
    }
    dense_mv1(g, w, in, NULL, g->dpost, 0);
    cl_kernel k = g->k_addn;
    ARGM(k, 0, g->dx);
    ARGM(k, 1, g->dpost);
    ARGM(k, 2, g->dsmall);
    ARGU(k, 3, post->off);
    ARGI(k, 4, g->m->dim);
    ARGF(k, 5, g->m->eps);
    run1(g, k, WG, WG, "add_norm");
}

static void dense_enqueue_layer(Gpu *g, int l) {
    const Model *m = g->m;
    const int pos = g->cur_pos;
    const Layer *L = &m->layers[l];
    const DL *d = &g->dl[l];
    CLApi *cl = &g->cl;
    const int D = m->dim, hd = m->head_dim, kvd = m->n_kv_head * hd, K = m->n_expert_used, F = m->ff_exp;
    cl_kernel k;

    if (L->is_attn) {
        const DT *qkv[3] = { &d->wq, &d->wk, &d->wv };
        const int yo[3] = { 0, D, D + kvd };
        dense_mv(g, 3, qkv, yo, g->dx, &d->attn_norm, g->dbig, 0);
        if (d->wgate.src) dense_mv1(g, &d->wgate, g->dx, &d->attn_norm, g->dgate, 0);
        k = g->k_attn;
        ARGM(k, 0, g->dbig);
        ARGM(k, 1, g->dsmall);
        ARGU(k, 2, d->qn.off);
        ARGU(k, 3, d->kn.off);
        ARGM(k, 4, g->drope);
        ARGM(k, 5, g->dkc);
        ARGM(k, 6, g->dvc);
        ARGU(k, 7, (cl_ulong)L->attn_idx * g->n_ctx * kvd);
        ARGI(k, 8, pos);
        ARGI(k, 9, m->n_head);
        ARGI(k, 10, m->n_kv_head);
        ARGI(k, 11, hd);
        ARGI(k, 12, D);
        ARGF(k, 13, m->eps);
        ARGM(k, 14, g->dscores);
        ARGI(k, 15, g->n_ctx);
        ARGM(k, 16, g->dtmp);
        ARGF(k, 17, 1.f / sqrtf((float)hd));
        ARGI(k, 18, L->is_swa ? m->swa_window : 0);
        ARGI(k, 19, L->is_swa ? hd : 0);
        ARGM(k, 20, g->dgate);
        ARGI(k, 21, d->wgate.src != NULL);
        run1(g, k, (size_t)m->n_head * WG, WG, "attn_decode");
        block_out(g, &d->wo, g->dtmp, &d->post_op);
    } else {
        dense_mv1(g, &d->in_proj, g->dx, &d->attn_norm, g->dbig, 0);
        k = g->k_conv;
        ARGM(k, 0, g->dbig);
        ARGM(k, 1, g->dconv);
        ARGU(k, 2, (cl_ulong)L->conv_idx * (m->l_cache - 1) * D);
        ARGM(k, 3, g->dsmall);
        ARGU(k, 4, d->conv_w.off);
        ARGM(k, 5, g->dtmp);
        ARGI(k, 6, D);
        ARGI(k, 7, m->l_cache);
        run1(g, k, (size_t)((D + 255) & ~255), 256, "shortconv");
        block_out(g, &d->out_proj, g->dtmp, &d->post_op);
    }
    if (!L->is_moe) {
        const int Fd = d->wg.rows;
        const DT *gu[2] = { &d->wg, &d->wu };
        const int yo[2] = { 0, g->dff };
        dense_mv(g, 2, gu, yo, g->dx, &d->ffn_norm, g->dfg, 0);
        k = g->k_silu;
        ARGM(k, 0, g->dfg);
        ARGI(k, 1, Fd);
        ARGI(k, 2, g->dff);
        run1(g, k, (size_t)((Fd + 255) & ~255), 256, "silu_mul");
        block_out(g, &d->wd, g->dfg, &d->post_ffn);
        return;
    }
    /* norm + router + top-k on the GPU, then x | h | choice go to the host in one read */
    k = g->k_route;
    ARGM(k, 0, g->dx);
    ARGM(k, 1, g->dsmall);
    ARGU(k, 2, d->ffn_norm.off);
    ARGU(k, 3, d->router.off);
    ARGU(k, 4, d->bias.off);
    ARGI(k, 5, d->bias.src != NULL);
    ARGI(k, 6, D);
    ARGF(k, 7, m->eps);
    ARGI(k, 8, m->n_expert);
    ARGI(k, 9, K);
    ARGI(k, 10, m->gating);
    ARGM(k, 11, g->dres);
    ARGI(k, 12, l * 64);
    ARGM(k, 13, g->dlog);
    ARGM(k, 14, g->dcount);
    ARGM(k, 15, g->dmeta2);
    ARGM(k, 16, g->dout);
    run1(g, k, (size_t)m->n_expert * WG, WG, "moe_route");
    check(g, cl->EnqueueReadBuffer(g->q, g->dout, 0, 0, sizeof(float) * 2 * D + sizeof(int) * 3 * K, g->h_out, 0, NULL,
                                   &g->ev_r[l]), "read route");
    /* resident experts: the same kernels as the prompt path, on this token only */
    const cl_ulong sb = g->slot_bytes, upo = g->up_off[l], dno = g->down_off[l];
    const int gt = g->gate_type[l], dt = g->down_type[l];
    const size_t lws[2] = { WG, 1 };
    const size_t gws_gu[2] = { (size_t)((F + 8 * g->rpw - 1) / (8 * g->rpw)) * WG, (size_t)(2 * K) };
    const size_t gws_dn[2] = { (size_t)((D + 8 * g->rpw - 1) / (8 * g->rpw)) * WG, (size_t)K };
    const size_t red_bytes = sizeof(float) * WG * g->rpw;
    k = g->k_gu;
    for (int a = 0; a < MAX_ARENAS; a++) ARGM(k, (cl_uint)a, g->arena[a]);
    ARGI(k, 4, g->spa);
    ARGU(k, 5, sb);
    ARGM(k, 6, g->dmeta2);
    ARGI(k, 7, K);
    ARGI(k, 8, D);
    ARGI(k, 9, F);
    ARGU(k, 10, upo);
    ARGI(k, 11, gt);
    ARGM(k, 12, g->dgu2);
    if (g->dp4a) {
        arg(g, k, 13, (size_t)D, NULL);
        arg(g, k, 14, sizeof(float) * (D / 32), NULL);
        arg(g, k, 15, red_bytes, NULL);
        ARGM(k, 16, g->dout);
        ARGU(k, 17, D);
    } else {
        arg(g, k, 13, sizeof(float) * D, NULL);
        arg(g, k, 14, red_bytes, NULL);
        ARGM(k, 15, g->dout);
        ARGU(k, 16, D);
    }
    check(g, cl->EnqueueNDRangeKernel(g->q, k, 2, NULL, gws_gu, lws, 0, NULL, NULL), "moe_gate_up");
    k = g->k_down;
    for (int a = 0; a < MAX_ARENAS; a++) ARGM(k, (cl_uint)a, g->arena[a]);
    ARGI(k, 4, g->spa);
    ARGU(k, 5, sb);
    ARGM(k, 6, g->dmeta2);
    ARGI(k, 7, F);
    ARGI(k, 8, D);
    ARGU(k, 9, dno);
    ARGI(k, 10, dt);
    ARGM(k, 11, g->dgu2);
    ARGM(k, 12, g->dy2);
    if (g->dp4a) {
        arg(g, k, 13, (size_t)F, NULL);
        arg(g, k, 14, sizeof(float) * (F / 32), NULL);
        arg(g, k, 15, red_bytes, NULL);
    } else {
        arg(g, k, 13, sizeof(float) * F, NULL);
        arg(g, k, 14, red_bytes, NULL);
    }
    check(g, cl->EnqueueNDRangeKernel(g->q, k, 2, NULL, gws_dn, lws, 0, NULL, NULL), "moe_down");
    if (!g->lookahead) { /* gpu_dense_moe_finish uploads the CPU part and combines */
        g->moe_open = l;
        return;
    }
    /* lookahead: the CPU part's upload waits on a user event set by gpu_dense_moe_finish, so the
     * following layers already sit in the queue. Faster only where the driver releases such
     * commands promptly (NVIDIA on Windows adds ~0.3 ms per layer). */
    cl_int e = 0;
    g->uev[l] = cl->CreateUserEvent(g->ctx, &e);
    check(g, e, "user event");
    check(g, cl->EnqueueWriteBuffer(g->q, g->dycpu, 0, 0, sizeof(float) * D, g->h_yc, 1, &g->uev[l], NULL), "write ycpu");
    dense_combine(g, 1);
}

static void dense_combine(Gpu *g, int use_cpu) {
    const int D = g->m->dim;
    cl_kernel k = g->k_comb;
    ARGM(k, 0, g->dx);
    ARGM(k, 1, g->dy2);
    ARGM(k, 2, g->dout);
    ARGI(k, 3, 2 * D);
    ARGM(k, 4, g->dycpu);
    ARGI(k, 5, use_cpu);
    ARGI(k, 6, D);
    ARGI(k, 7, g->m->n_expert_used);
    run1(g, k, (size_t)((D + 255) & ~255), 256, "moe_combine");
}

static void dense_enqueue_upto(Gpu *g, int upto) {
    const Model *m = g->m;
    if (upto > m->n_layer) upto = m->n_layer;
    while (g->enq < upto) dense_enqueue_layer(g, g->enq++);
    if (g->enq == m->n_layer && g->moe_open < 0 && !g->enq_logits) {
        dense_mv1(g, &g->d_out, g->dx, &g->d_fnorm, g->dlogits, 0);
        check(g, g->cl.EnqueueReadBuffer(g->q, g->dlogits, 0, 0, sizeof(float) * (size_t)m->n_vocab, g->h_logits, 0, NULL,
                                         &g->ev_logits), "read logits");
        g->enq_logits = 1;
    }
    g->cl.Flush(g->q);
}

void gpu_dense_wait(Gpu *g, int l, const float **x, const float **h, const int **ret) {
    wait_event(g, g->ev_r[l]);
    g->cl.ReleaseEvent(g->ev_r[l]);
    g->ev_r[l] = NULL;
    *x = g->h_out;
    *h = g->h_out + g->m->dim;
    *ret = (const int *)(g->h_out + 2 * g->m->dim);
}

void gpu_dense_moe_finish(Gpu *g, int l, const float *ycpu) {
    const int D = g->m->dim;
    /* the previous upload of h_yc has completed: it precedes the readback we just waited for */
    if (ycpu) memcpy(g->h_yc, ycpu, sizeof(float) * D);
    else memset(g->h_yc, 0, sizeof(float) * D);
    if (g->lookahead) {
        check(g, g->cl.SetUserEventStatus(g->uev[l], 0), "set user event");
        g->cl.ReleaseEvent(g->uev[l]);
        g->uev[l] = NULL;
    } else {
        if (ycpu) check(g, g->cl.EnqueueWriteBuffer(g->q, g->dycpu, 0, 0, sizeof(float) * D, g->h_yc, 0, NULL, NULL), "write ycpu");
        dense_combine(g, ycpu != NULL);
        g->moe_open = -1;
    }
    dense_enqueue_upto(g, next_moe(g->m, l) + 1 + g->lookahead);
}

void gpu_dense_logits(Gpu *g, float *out) {
    dense_enqueue_upto(g, g->m->n_layer);
    wait_event(g, g->ev_logits);
    g->cl.ReleaseEvent(g->ev_logits);
    g->ev_logits = NULL;
    memcpy(out, g->h_logits, sizeof(float) * (size_t)g->m->n_vocab);
}

void gpu_dense_finish(Gpu *g) { check(g, g->cl.Finish(g->q), "finish"); }

/* state transfers between the CPU copy (model.c) and VRAM */
void gpu_dense_kv(Gpu *g, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu) {
    if (r1 <= r0) return;
    const Model *m = g->m;
    const size_t kvd = (size_t)m->n_kv_head * m->head_dim;
    for (int w = 0; w < 2; w++)
        for (int a = 0; a < m->n_attn; a++) {
            const size_t off = ((size_t)a * g->n_ctx + r0) * kvd * sizeof(uint16_t);
            const size_t n = (size_t)(r1 - r0) * kvd * sizeof(uint16_t);
            uint8_t *host = (uint8_t *)(w ? vc : kc) + off;
            cl_mem b = w ? g->dvc : g->dkc;
            if (to_gpu) check(g, g->cl.EnqueueWriteBuffer(g->q, b, 1, off, n, host, 0, NULL, NULL), "kv upload");
            else check(g, g->cl.EnqueueReadBuffer(g->q, b, 1, off, n, host, 0, NULL, NULL), "kv download");
        }
}

void gpu_dense_conv(Gpu *g, float *conv, int to_gpu) {
    if (to_gpu) check(g, g->cl.EnqueueWriteBuffer(g->q, g->dconv, 1, 0, g->conv_bytes, conv, 0, NULL, NULL), "conv upload");
    else check(g, g->cl.EnqueueReadBuffer(g->q, g->dconv, 1, 0, g->conv_bytes, conv, 0, NULL, NULL), "conv download");
}
