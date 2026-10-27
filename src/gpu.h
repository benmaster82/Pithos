/* gpu.h - optional GPU co-processor for MoE experts (OpenCL, loaded at runtime).
 *
 * The GPU keeps a dynamic cache of the most used experts in VRAM. During decoding, for each
 * MoE layer the experts that are resident run on the GPU while the CPU computes the others;
 * the partial results are summed. A background thread uploads newly-hot experts, so the
 * compute path never waits for PCIe or disk. Without a usable OpenCL device the engine runs
 * CPU-only. */
#ifndef PITHOS_GPU_H
#define PITHOS_GPU_H

#include <stddef.h>
#include "model.h"

typedef struct Gpu Gpu;

/* device: -1 = auto (largest discrete GPU), otherwise index in the flat device list.
 * vram_mb: VRAM budget (0 = auto). max_tokens: largest batch passed to gpu_moe_begin.
 * dense_ctx > 0: also keep every non-expert weight, a KV cache of dense_ctx tokens and the
 * recurrent state in VRAM for the single-token decode path (gpu_dense_*), if they fit.
 * info receives a one-line description. */
Gpu *gpu_create(const Model *m, int device, int vram_mb, int max_tokens, int dense_ctx, const char *profile_path,
                char *info, int infolen);
void gpu_destroy(Gpu *g);

/* List OpenCL devices to stdout. */
void gpu_list_devices(void);

/* Is expert e of layer l ready in VRAM? */
int gpu_resident(const Gpu *g, int layer, int expert);

/* Launch resident-expert work asynchronously. x: n_tok rows of dim floats (normalized hidden
 * states); assignment i evaluates expert a_expert[i] on token a_tok[i]. */
void gpu_moe_begin(Gpu *g, int layer, const float *x, int n_tok, const int *a_expert, const int *a_tok, int n_assign);
/* Wait for the launch above; y receives n_assign rows of dim floats (unweighted outputs). */
void gpu_moe_end(Gpu *g, float *y);

/* Usage feedback for the cache policy (call for every routed expert). */
void gpu_note_usage(Gpu *g, int layer, int expert);
/* Called once per generated token: decay scores and schedule up to max_uploads swaps. */
void gpu_rebalance(Gpu *g, int max_uploads);

/* Number of expert uploads still queued or in flight. */
int gpu_pending(Gpu *g);

/* ---- dense decode path: one token through all layers on the GPU, experts shared with the CPU.
 * Per token: gpu_dense_begin; then for every MoE layer, in order, gpu_dense_wait gives the
 * residual x, the normalized input h and ret = sel[K] | slot[K] | weight bits[K] (slot < 0:
 * the CPU computes that expert) and gpu_dense_moe_finish adds the CPU part (weighted sum, or
 * NULL); finally gpu_dense_logits. The pointers stay valid until gpu_dense_moe_finish. */
int  gpu_dense_on(const Gpu *g);
void gpu_dense_begin(Gpu *g, const float *x, const float *rope_cs /* cos[hd/2] | sin[hd/2] */, int pos);
void gpu_dense_wait(Gpu *g, int layer, const float **x, const float **h, const int **ret);
void gpu_dense_moe_finish(Gpu *g, int layer, const float *ycpu);
void gpu_dense_logits(Gpu *g, float *out);
/* copy KV rows [r0, r1) of every attention layer ([n_attn][n_ctx][kvd] f16 layout) and the
 * convolution state between the host copy and VRAM */
void gpu_dense_kv(Gpu *g, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu);
void gpu_dense_conv(Gpu *g, float *conv, int to_gpu);

typedef struct {
    int slots, used, ready;
    uint64_t gpu_expert_calls, cpu_expert_calls, uploads;
    double t_wait;
} GpuStats;
void gpu_stats(const Gpu *g, GpuStats *s);

#endif
