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

/* NVFP4: 64 E2M1 values in 4 sub-blocks of 16, each with an unsigned E4M3 scale; sub-block s
 * holds its values j and j+8 in the low and high nibbles of qs[8s+j] */
#define QK_NVFP4 64
typedef struct { uint8_t d[4]; uint8_t qs[QK_NVFP4 / 2]; } block_nvfp4;                            /* 36 B */

/* ggml's codebook formats (tables in iq_tables.h): 8 values per grid index (4 for the IQ3 ones),
 * signs as 7 bits with even parity (XXS/XS) or 8 explicit bits (S) */
typedef struct { f16_t d; uint16_t qs[QK_K / 8]; } block_iq2_xxs;                                  /* 66 B */
typedef struct { f16_t d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; } block_iq2_xs;         /* 74 B */
typedef struct { f16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; } block_iq2_s; /* 82 B */
typedef struct { f16_t d; uint8_t qs[3 * QK_K / 8]; } block_iq3_xxs;                                /* 98 B */
typedef struct { f16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[QK_K / 64]; } block_iq3_s; /* 110 B */
typedef struct { f16_t d; uint8_t qs[QK8_0 / 2]; } block_iq4_nl;                                   /* 18 B */
typedef struct { f16_t d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; } block_iq4_xs; /* 136 B */
typedef struct { f16_t d; uint8_t qh[4]; uint8_t qs[QK8_0 / 2]; } block_q5_0;                       /* 22 B */

/* E2M1 values times 2, and the E4M3 scales times 0.5 (the product is the weight) */
extern const int8_t nvfp4_values[16];
extern float nvfp4_scale[256];

float f16_to_f32(f16_t h);
f16_t f32_to_f16(float f);

/* Kernel: dot one weight row (K elements, format-specific) against nt activation
 * rows (Q8_K, consecutive rows `xstride` blocks apart), writing out[t*ostride].
 * K is a multiple of 256, except for NVFP4 (a multiple of 64) and Q8_0, Q5_0, IQ4_NL (a multiple of 32):
 * the activation rows then end in a partial Q8_K block, zero-padded. */
typedef void (*dot_rows_fn)(int K, const void *w, const block_q8_K *x, int xstride,
                            int nt, float *out, int ostride);

typedef struct {
    const char *name;
    dot_rows_fn dot[48];                 /* indexed by GGML type */
    void (*quantize_q8_K)(const float *x, block_q8_K *y, int K);
    void (*f16_to_f32_row)(const f16_t *x, float *y, int n);
    void (*f32_to_f16_row)(const float *x, f16_t *y, int n);
    float (*dot_f32)(const float *a, const float *b, int n);
    float (*dot_f16_f32)(const f16_t *a, const float *b, int n);
    void (*axpy_f16)(float a, const f16_t *x, float *y, int n);   /* y += a*x */
    void (*silu_mul)(float *g, const float *u, int n);            /* g = silu(g) * u */
    float (*exp_sum)(float *s, int n, float mx);                  /* s = exp(s - mx), returns sum */
    /* one row of a supported type (K a multiple of 256) as Q8_0: K int8 values and an f16 scale per 32 */
    void (*row_to_q8_0)(uint32_t type, const void *row, int K, int8_t *qs, f16_t *d);
    /* an IQ2_XXS/XS/S or IQ3_XXS row in Vulkan's nibble layout (mv_iq in kernels_dx.hlsl): K/2 nibble
     * bytes, K/16 scale bytes, K/256 floats; 0 for other types (NULL without SIMD) */
    int (*row_to_iq4)(uint32_t type, const void *row, int K, uint8_t *nib, uint8_t *sc, float *d);
} Kernels;

extern Kernels K_scalar;
#if defined(__x86_64__) || defined(_M_X64)
void kernels_init_avx2(Kernels *k);
void kernels_init_avx(Kernels *k);   /* AVX + F16C, no AVX2/FMA (Jaguar, Ivy Bridge) */
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
void kernels_init_neon(Kernels *k);
#endif

/* Active kernel table (set by quant_init). */
extern Kernels KS;
/* force: NULL/"auto", "scalar", "avx", "avx2", "neon". Returns name of selection. */
const char *quant_init(const char *force);

/* Dequantize one row of K elements of the given type into f32. */
void dequant_row(uint32_t type, const void *src, float *dst, int K);
/* Bytes of one row of K elements. */
size_t row_bytes(uint32_t type, int K);
int type_supported(uint32_t type);

#endif
