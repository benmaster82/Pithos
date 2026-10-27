/* gguf.h - zero-copy GGUF (v2/v3) reader over a memory-mapped file. */
#ifndef PITHOS_GGUF_H
#define PITHOS_GGUF_H

#include <stdint.h>
#include "platform.h"

enum {
    GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
    GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STRING, GGUF_T_ARRAY, GGUF_T_U64, GGUF_T_I64, GGUF_T_F64
};

/* ggml tensor types we know about */
enum {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
    GGML_Q8_0 = 8, GGML_Q8_1 = 9, GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12,
    GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_Q8_K = 15, GGML_BF16 = 30
};

typedef struct {
    const char *key; uint64_t key_len;
    uint32_t type;
    uint32_t arr_type;      /* for arrays */
    uint64_t arr_len;       /* for arrays */
    const uint8_t *val;     /* points into the file at the value (or array payload) */
} GGUFKV;

typedef struct {
    char name[96];
    uint32_t n_dims;
    int64_t ne[4];
    uint32_t type;
    uint64_t offset;        /* relative to data section */
    const uint8_t *data;
    uint64_t nbytes;
} GGUFTensor;

typedef struct {
    MappedFile mf;
    uint32_t version;
    uint64_t n_kv, n_tensors;
    GGUFKV *kv;
    GGUFTensor *tensors;
    uint64_t alignment;
    const uint8_t *data_start;
} GGUF;

int  gguf_open(GGUF *g, const char *path, char *err, int errlen);
void gguf_close(GGUF *g);

const GGUFKV *gguf_find(const GGUF *g, const char *key);
int64_t  gguf_get_int(const GGUF *g, const char *key, int64_t def);
double   gguf_get_float(const GGUF *g, const char *key, double def);
/* Returns pointer into file + length (not NUL terminated). */
const char *gguf_get_str(const GGUF *g, const char *key, uint64_t *len);
/* integer element i of an integer array */
int64_t  gguf_arr_int(const GGUFKV *kv, uint64_t i);

/* Iterate a string array: call with *cursor = kv->val initially. */
const char *gguf_next_str(const uint8_t **cursor, uint64_t *len);

const GGUFTensor *gguf_tensor(const GGUF *g, const char *name);

/* bytes per block and elements per block for a ggml type */
int ggml_block_size(uint32_t type);
int ggml_type_size(uint32_t type);
const char *ggml_type_name(uint32_t type);

#endif
