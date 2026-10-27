/* quant.h - ggml-compatible block formats and dot-product kernels.
 *
 * All quantized matmuls take activations quantized to Q8_K (256-element blocks,
 * one f32 scale, int8 values, 16-element partial sums). Weights stay in their
 * on-disk GGUF format and are read straight from the memory map. */
#ifndef PITHOS_QUANT_H
#define PITHOS_QUANT_H

#include <stddef.h>
#include <stdint.h>

#define QK_K 256
#define QK8_0 32

typedef uint16_t f16_t;

typedef struct { f16_t d; f16_t dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } block_q4_K;     /* 144 B */
typedef struct { f16_t d; f16_t dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } block_q5_K; /* 176 B */
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; f16_t d; } block_q6_K; /* 210 B */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } block_q8_K;                 /* 292 B */
typedef struct { f16_t d; int8_t qs[QK8_0]; } block_q8_0;                                          /* 34 B */
typedef struct { f16_t d; uint8_t qs[QK8_0 / 2]; } block_q4_0;                                     /* 18 B */

float f16_to_f32(f16_t h);
f16_t f32_to_f16(float f);

/* Kernel: dot one weight row (K elements, format-specific) against nt activation
 * rows (Q8_K, consecutive rows `xstride` blocks apart), writing out[t*ostride]. */
typedef void (*dot_rows_fn)(int K, const void *w, const block_q8_K *x, int xstride,
                            int nt, float *out, int ostride);

typedef struct {
    const char *name;
    dot_rows_fn dot[32];                 /* indexed by GGML type */
    void (*quantize_q8_K)(const float *x, block_q8_K *y, int K);
    void (*f16_to_f32_row)(const f16_t *x, float *y, int n);
    void (*f32_to_f16_row)(const float *x, f16_t *y, int n);
    float (*dot_f32)(const float *a, const float *b, int n);
    float (*dot_f16_f32)(const f16_t *a, const float *b, int n);
    void (*axpy_f16)(float a, const f16_t *x, float *y, int n);   /* y += a*x */
    void (*silu_mul)(float *g, const float *u, int n);            /* g = silu(g) * u */
    float (*exp_sum)(float *s, int n, float mx);                  /* s = exp(s - mx), returns sum */
} Kernels;

extern Kernels K_scalar;
#if defined(__x86_64__) || defined(_M_X64)
void kernels_init_avx2(Kernels *k);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
void kernels_init_neon(Kernels *k);
#endif

/* Active kernel table (set by quant_init). */
extern Kernels KS;
/* force: NULL/"auto", "scalar", "avx2", "neon". Returns name of selection. */
const char *quant_init(const char *force);

/* Dequantize one row of K elements of the given type into f32. */
void dequant_row(uint32_t type, const void *src, float *dst, int K);
/* Bytes of one row of K elements. */
size_t row_bytes(uint32_t type, int K);
int type_supported(uint32_t type);

#endif
