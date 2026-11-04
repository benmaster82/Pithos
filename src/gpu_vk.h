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

/* device: an index from vk_list_devices, or -1 for the discrete GPU with the most memory.
 * info receives a one-line description, or why the GPU is not used. */
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
 * receives the n_vocab floats of the last token. Returns 0 if the GPU failed. */
int  vk_dense_forward(Vulkan *v, const float *x, int n, const float *rope_cs, const float *rope_swa, int pos0,
                      float *logits);
/* copy KV rows [r0, r1) / the convolution state between the host arrays and the GPU copy */
void vk_dense_kv(Vulkan *v, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu);
void vk_dense_conv(Vulkan *v, float *conv, int to_gpu);

#endif
