/* gpu_vk.h - optional Vulkan GPU (Windows, Linux).
 *
 * The same role as gpu_dx.h on any Vulkan 1.1 GPU, with the same kernels (kernels_dx*.hlsl
 * compiled to SPIR-V): the quantized weights of the dense layers (Q4_K, Q6_K) are copied once
 * into GPU memory and the matmuls run there. The Vulkan loader (vulkan-1.dll / libvulkan.so.1)
 * is opened at runtime: without a usable device the engine runs CPU-only. */
#ifndef PITHOS_GPU_VK_H
#define PITHOS_GPU_VK_H

#include "model.h"

typedef struct Vulkan Vulkan;

/* One matmul as the CPU path describes it: out[t * ostride + r] = row r of w . row t of xq. */
typedef struct {
    const uint8_t *w;               /* host address of the weight matrix (Mat.data) */
    const block_q8_K *xq; int xstride;  /* activation rows, xstride blocks apart */
    int nt;
    float *out; int ostride;
} VulkanJob;

/* Print the Vulkan devices (index, name, type, memory). */
void vk_list_devices(void);

/* device: an index from vk_list_devices, -1 for the discrete GPU with the most memory, or
 * VK_AUTO_DEVICE for a discrete NVIDIA GPU only (the vendor tested, what --gpu auto uses).
 * info receives a one-line description, or why the GPU is not used. */
#define VK_AUTO_DEVICE (-2)
Vulkan *vk_create(const Model *m, int device, char *info, int infolen);
void vk_destroy(Vulkan *v);

/* Is this weight matrix in GPU memory? */
int vk_has(const Vulkan *v, const uint8_t *w);

/* Run the jobs (all matrices must satisfy vk_has) and wait for the results.
 * Returns 0 if the GPU failed: nothing was written and the caller computes them on the CPU. */
int vk_matmul(Vulkan *v, const VulkanJob *jobs, int n);

/* ---- whole forward pass (models without experts), as dx_dense_* in gpu_dx.h: every step of
 * a token or of a prompt chunk on the GPU, which keeps its own KV cache ([n_attn][n_ctx][kvd]
 * f16, like the CPU) and convolution state. */
/* 0 if the model or the GPU does not allow it; info says why, or what was set up */
int  vk_dense_init(Vulkan *v, const Model *m, int n_ctx, int n_batch, char *info, int infolen);
/* n (<= n_batch) tokens at positions pos0.. : x their embeddings [n][dim]; rope_cs / rope_swa
 * cos[hd/2] | sin[hd/2] per position [n][hd] (rope_swa NULL: none). logits (NULL: not needed)
 * receives the n_vocab floats of the last token, x_last (NULL: not needed, only with logits) its
 * residual stream before the final norm [dim]. Returns 0 if the GPU failed. */
int  vk_dense_forward(Vulkan *v, const float *x, int n, const float *rope_cs, const float *rope_swa, int pos0,
                      float *logits, float *x_last);
/* qwen35moe runs the same way, except for the routed experts: at every layer the GPU stops and
 * fn gets the normalized rows h [n][dim] of the pass's tokens and their router logits lg
 * [n][n_expert + 1] (the last one: the shared expert's gate input . h); for a single token, pred
 * has the next layer's router logits of the residual stream, to read its experts ahead (else
 * NULL). fn writes the weighted sum of the experts to y [n][dim] and the shared expert's gate to
 * s [n]. Set it before vk_dense_forward. */
typedef void (*VkMoeFn)(void *arg, int layer, const float *h, const float *lg, const float *pred, int n, float *y, float *s);
void vk_dense_set_moe(Vulkan *v, VkMoeFn fn, void *arg);
/* ---- the most used routed experts in VRAM (a discrete GPU, with the qwen35moe pass): inside fn,
 * when vk_moe_ready(n) says that n assignments (at most VK_MOE_MAX) can run on the GPU now,
 * vk_moe_begin starts the resident ones (expert a_expert[i] on token a_tok[i] of fn's rows,
 * weight a_w[i]); the GPU adds their weighted sum to the residual stream itself, so y leaves
 * them out. 0: the GPU failed (the pass fails). vk_note_usage / vk_rebalance: the cache policy,
 * as gpu_note_usage / gpu_rebalance in gpu.h. */
#define VK_MOE_MAX 256
int  vk_experts_on(const Vulkan *v);
int  vk_expert_resident(const Vulkan *v, int layer, int expert);
int  vk_moe_ready(const Vulkan *v, int n);
int  vk_moe_begin(Vulkan *v, int layer, const int *a_expert, const int *a_tok, const float *a_w, int n);
/* Inkling (the experts of its NVFP4 layers): the same with each assignment's [gate | up] scale a_gs, a_w
 * already times the down scale; the next slot's FFN output takes their sum */
int  vk_ik_moe_begin(Vulkan *v, int layer, const int *a_expert, const int *a_tok, const float *a_w, const float *a_gs, int n);
void vk_note_usage(Vulkan *v, int layer, int expert);
void vk_rebalance(Vulkan *v, int max_uploads);
/* copy KV rows [r0, r1) / the convolution state (DeltaNet's included) between the host arrays and the GPU copy */
void vk_dense_kv(Vulkan *v, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu);
void vk_dense_conv(Vulkan *v, float *conv, int to_gpu);

#endif
