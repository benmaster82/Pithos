/* gguf.c - see gguf.h */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ggml_block_size(uint32_t t) {
    switch (t) {
    case GGML_F32: case GGML_F16: case GGML_BF16: return 1;
    case GGML_Q4_0: case GGML_Q4_1: case GGML_Q5_0: case GGML_Q5_1: case GGML_Q8_0: case GGML_Q8_1: return 32;
    default: return 256;
    }
}

int ggml_type_size(uint32_t t) {
    switch (t) {
    case GGML_F32: return 4;
    case GGML_F16: case GGML_BF16: return 2;
    case GGML_Q4_0: return 18;
    case GGML_Q4_1: return 20;
    case GGML_Q5_0: return 22;
    case GGML_Q5_1: return 24;
    case GGML_Q8_0: return 34;
    case GGML_Q8_1: return 36;
    case GGML_Q2_K: return 84;
    case GGML_Q3_K: return 110;
    case GGML_Q4_K: return 144;
    case GGML_Q5_K: return 176;
    case GGML_Q6_K: return 210;
    case GGML_Q8_K: return 292;
    default: return 0;
    }
}

const char *ggml_type_name(uint32_t t) {
    switch (t) {
    case GGML_F32: return "F32"; case GGML_F16: return "F16"; case GGML_BF16: return "BF16";
    case GGML_Q4_0: return "Q4_0"; case GGML_Q4_1: return "Q4_1"; case GGML_Q5_0: return "Q5_0";
    case GGML_Q5_1: return "Q5_1"; case GGML_Q8_0: return "Q8_0"; case GGML_Q2_K: return "Q2_K";
    case GGML_Q3_K: return "Q3_K"; case GGML_Q4_K: return "Q4_K"; case GGML_Q5_K: return "Q5_K";
    case GGML_Q6_K: return "Q6_K"; case GGML_Q8_K: return "Q8_K";
    default: return "?";
    }
}

typedef struct { const uint8_t *p, *end; int ok; } Rd;

static uint64_t rd_u64(Rd *r) {
    if (r->p + 8 > r->end) { r->ok = 0; return 0; }
    uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v;
}
static uint32_t rd_u32(Rd *r) {
    if (r->p + 4 > r->end) { r->ok = 0; return 0; }
    uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v;
}

static int scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
    case GGUF_T_U16: case GGUF_T_I16: return 2;
    case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
    case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
    default: return 0;
    }
}

static void skip_value(Rd *r, uint32_t type) {
    if (type == GGUF_T_STRING) {
        uint64_t n = rd_u64(r);
        if (r->p + n > r->end) { r->ok = 0; return; }
        r->p += n;
    } else if (type == GGUF_T_ARRAY) {
        uint32_t at = rd_u32(r);
        uint64_t n = rd_u64(r);
        if (at == GGUF_T_STRING || at == GGUF_T_ARRAY) {
            for (uint64_t i = 0; i < n && r->ok; i++) skip_value(r, at);
        } else {
            uint64_t sz = n * (uint64_t)scalar_size(at);
            if (!scalar_size(at) || r->p + sz > r->end) { r->ok = 0; return; }
            r->p += sz;
        }
    } else {
        int s = scalar_size(type);
        if (!s || r->p + s > r->end) { r->ok = 0; return; }
        r->p += s;
    }
}

int gguf_open(GGUF *g, const char *path, char *err, int errlen) {
    memset(g, 0, sizeof(*g));
    if (!map_file(&g->mf, path)) { snprintf(err, errlen, "cannot open/map '%s'", path); return 0; }
    Rd r = { g->mf.data, g->mf.data + g->mf.size, 1 };
    if (g->mf.size < 24 || memcmp(r.p, "GGUF", 4) != 0) { snprintf(err, errlen, "not a GGUF file"); goto fail; }
    r.p += 4;
    g->version = rd_u32(&r);
    if (g->version < 2) { snprintf(err, errlen, "unsupported GGUF version %u", g->version); goto fail; }
    g->n_tensors = rd_u64(&r);
    g->n_kv = rd_u64(&r);
    g->kv = (GGUFKV *)calloc(g->n_kv, sizeof(GGUFKV));
    g->tensors = (GGUFTensor *)calloc(g->n_tensors, sizeof(GGUFTensor));
    g->alignment = 32;
    for (uint64_t i = 0; i < g->n_kv && r.ok; i++) {
        GGUFKV *kv = &g->kv[i];
        kv->key_len = rd_u64(&r);
        kv->key = (const char *)r.p;
        r.p += kv->key_len;
        kv->type = rd_u32(&r);
        if (kv->type == GGUF_T_ARRAY) {
            const uint8_t *save = r.p;
            kv->arr_type = rd_u32(&r);
            kv->arr_len = rd_u64(&r);
            kv->val = r.p;
            r.p = save;
        } else {
            kv->val = r.p;
        }
        skip_value(&r, kv->type);
    }
    if (!r.ok) { snprintf(err, errlen, "corrupt GGUF metadata"); goto fail; }
    g->alignment = (uint64_t)gguf_get_int(g, "general.alignment", 32);
    for (uint64_t i = 0; i < g->n_tensors && r.ok; i++) {
        GGUFTensor *t = &g->tensors[i];
        uint64_t nl = rd_u64(&r);
        size_t cp = nl < sizeof(t->name) - 1 ? (size_t)nl : sizeof(t->name) - 1;
        memcpy(t->name, r.p, cp);
        t->name[cp] = 0;
        r.p += nl;
        t->n_dims = rd_u32(&r);
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims && d < 4; d++) t->ne[d] = (int64_t)rd_u64(&r);
        t->type = rd_u32(&r);
        t->offset = rd_u64(&r);
    }
    if (!r.ok) { snprintf(err, errlen, "corrupt GGUF tensor info"); goto fail; }
    uint64_t pos = (uint64_t)(r.p - g->mf.data);
    pos = (pos + g->alignment - 1) / g->alignment * g->alignment;
    g->data_start = g->mf.data + pos;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        GGUFTensor *t = &g->tensors[i];
        int bs = ggml_block_size(t->type), ts = ggml_type_size(t->type);
        if (!ts) { snprintf(err, errlen, "tensor %s has unsupported type %u", t->name, t->type); goto fail; }
        uint64_t n = 1;
        for (int d = 0; d < 4; d++) n *= (uint64_t)t->ne[d];
        t->nbytes = n / bs * ts;
        t->data = g->data_start + t->offset;
        if (t->data + t->nbytes > g->mf.data + g->mf.size) {
            snprintf(err, errlen, "tensor %s out of file bounds (truncated download?)", t->name);
            goto fail;
        }
    }
    return 1;
fail:
    gguf_close(g);
    return 0;
}

void gguf_close(GGUF *g) {
    free(g->kv);
    free(g->tensors);
    unmap_file(&g->mf);
    memset(g, 0, sizeof(*g));
}

const GGUFKV *gguf_find(const GGUF *g, const char *key) {
    size_t kl = strlen(key);
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key_len == kl && memcmp(g->kv[i].key, key, kl) == 0) return &g->kv[i];
    return NULL;
}

static int64_t read_int(uint32_t type, const uint8_t *p) {
    switch (type) {
    case GGUF_T_U8: return *p;
    case GGUF_T_I8: return *(const int8_t *)p;
    case GGUF_T_BOOL: return *p != 0;
    case GGUF_T_U16: { uint16_t v; memcpy(&v, p, 2); return v; }
    case GGUF_T_I16: { int16_t v; memcpy(&v, p, 2); return v; }
    case GGUF_T_U32: { uint32_t v; memcpy(&v, p, 4); return v; }
    case GGUF_T_I32: { int32_t v; memcpy(&v, p, 4); return v; }
    case GGUF_T_U64: { uint64_t v; memcpy(&v, p, 8); return (int64_t)v; }
    case GGUF_T_I64: { int64_t v; memcpy(&v, p, 8); return v; }
    case GGUF_T_F32: { float v; memcpy(&v, p, 4); return (int64_t)v; }
    case GGUF_T_F64: { double v; memcpy(&v, p, 8); return (int64_t)v; }
    default: return 0;
    }
}

int64_t gguf_get_int(const GGUF *g, const char *key, int64_t def) {
    const GGUFKV *kv = gguf_find(g, key);
    if (!kv || kv->type == GGUF_T_STRING || kv->type == GGUF_T_ARRAY) return def;
    return read_int(kv->type, kv->val);
}

double gguf_get_float(const GGUF *g, const char *key, double def) {
    const GGUFKV *kv = gguf_find(g, key);
    if (!kv) return def;
    if (kv->type == GGUF_T_F32) { float v; memcpy(&v, kv->val, 4); return v; }
    if (kv->type == GGUF_T_F64) { double v; memcpy(&v, kv->val, 8); return v; }
    if (kv->type == GGUF_T_STRING || kv->type == GGUF_T_ARRAY) return def;
    return (double)read_int(kv->type, kv->val);
}

const char *gguf_get_str(const GGUF *g, const char *key, uint64_t *len) {
    const GGUFKV *kv = gguf_find(g, key);
    if (!kv || kv->type != GGUF_T_STRING) return NULL;
    uint64_t n; memcpy(&n, kv->val, 8);
    if (len) *len = n;
    return (const char *)kv->val + 8;
}

int64_t gguf_arr_int(const GGUFKV *kv, uint64_t i) {
    if (kv->type != GGUF_T_ARRAY || i >= kv->arr_len) return 0;
    int s = scalar_size(kv->arr_type);
    return read_int(kv->arr_type, kv->val + i * (uint64_t)s);
}

const char *gguf_next_str(const uint8_t **cursor, uint64_t *len) {
    uint64_t n; memcpy(&n, *cursor, 8);
    const char *s = (const char *)*cursor + 8;
    *cursor += 8 + n;
    *len = n;
    return s;
}

const GGUFTensor *gguf_tensor(const GGUF *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}
