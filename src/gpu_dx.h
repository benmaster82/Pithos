/* gpu_dx.h - optional Direct3D 12 GPU for prompt processing (Windows; Xbox One in Game mode).
 *
 * The quantized weights of the dense layers (Q4_K, Q6_K) are copied once into GPU memory.
 * During the forward pass, the matmuls over a batch of several tokens (prompt chunks) run on
 * the GPU, synchronously; single tokens and everything else stay on the CPU. d3d12.dll is
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

#endif
