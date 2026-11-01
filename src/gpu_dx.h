/* gpu_dx.h - optional Direct3D 12 GPU (Windows; Xbox One in Game mode).
 *
 * The quantized weights of the dense layers (Q4_K, Q6_K) are copied once into GPU memory.
 * Prompt chunks run their matmuls on the GPU (the rest on the CPU); single tokens go through
 * the GPU from embedding to logits when the model allows it (dx_dense_*). d3d12.dll is
 * loaded at runtime: without a usable device the engine runs CPU-only. */
#ifndef PITHOS_GPU_DX_H
#define PITHOS_GPU_DX_H

#include "model.h"

typedef struct Dx Dx;

/* One matmul as the CPU path describes it: out[t * ostride + r] = row r of w . row t of xq. */
typedef struct {
    const uint8_t *w;               /* host address of the weight matrix (Mat.data) */
    const block_q8_K *xq; int xstride;  /* activation rows, xstride blocks apart */
    int nt;
    float *out; int ostride;
} DxJob;

/* info receives a one-line description, or why the GPU is not used. */
Dx  *dx_create(const Model *m, char *info, int infolen);
void dx_destroy(Dx *d);

/* Is this weight matrix in GPU memory? */
int dx_has(const Dx *d, const uint8_t *w);

/* Run the jobs (all matrices must satisfy dx_has) and wait for the results.
 * Returns 0 if the GPU failed: nothing was written and the caller computes them on the CPU. */
int dx_matmul(Dx *d, const DxJob *jobs, int n);

/* ---- whole-token decoding (models without experts): every step of a single token on the GPU,
 * which keeps its own KV cache ([n_attn][n_ctx][kvd] f16, like the CPU) and convolution state. */
/* 0 if the model or the GPU memory does not allow it; info says why, or what was set up */
int  dx_dense_init(Dx *d, const Model *m, int n_ctx, char *info, int infolen);
/* x: the token's embedding; rope_cs / rope_swa: cos[hd/2] | sin[hd/2] at pos (rope_swa NULL: none).
 * logits receives n_vocab floats. Returns 0 if the GPU failed. */
int  dx_dense_token(Dx *d, const float *x, const float *rope_cs, const float *rope_swa, int pos, float *logits);
/* copy KV rows [r0, r1) / the convolution state between the host arrays and the GPU copy */
void dx_dense_kv(Dx *d, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu);
void dx_dense_conv(Dx *d, float *conv, int to_gpu);

#endif
