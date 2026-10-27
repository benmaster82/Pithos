/* tokenizer.c - see tokenizer.h */
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unicode_data.h"

#define TT_NORMAL 1
#define TT_CONTROL 3
#define TT_USER 4

typedef struct { uint64_t key; int32_t rank; int32_t merged; } MergeSlot;

struct Tokenizer {
    int n_vocab;
    const char **text;     /* token text as stored in GGUF (byte-level encoded) */
    int *text_len;
    char **piece;          /* decoded raw bytes */
    int *piece_len;
    int *type;
    int32_t *vmap; uint32_t vmask;          /* text -> id */
    MergeSlot *mmap; uint32_t mmask;        /* (a,b) -> rank, merged */
    int byte_tok[256];
    int *special; int n_special;            /* control/user tokens, longest first */
    int bos, eos;
};

/* ------------------------------------------------------------------ unicode helpers */

static int uni_flags(uint32_t c) {
    if (c < 128) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return UNI_LETTER;
        if (c >= '0' && c <= '9') return UNI_NUMBER;
        if (c == ' ' || (c >= 9 && c <= 13)) return UNI_SPACE;
        return 0;
    }
    int lo = 0, hi = (int)(sizeof(uni_ranges) / sizeof(uni_ranges[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < uni_ranges[mid][0]) hi = mid - 1;
        else if (c > uni_ranges[mid][1]) lo = mid + 1;
        else return (int)uni_ranges[mid][2];
    }
    return 0;
}

/* decode one UTF-8 codepoint; invalid bytes decode as themselves with length 1 */
static uint32_t utf8_next(const unsigned char *s, int n, int *len) {
    unsigned c = s[0];
    if (c < 0x80) { *len = 1; return c; }
    int need = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!need || need > n) { *len = 1; return 0xFFFD0000u | c; }
    uint32_t cp = c & (0x7F >> need);
    for (int i = 1; i < need; i++) {
        if ((s[i] & 0xC0) != 0x80) { *len = 1; return 0xFFFD0000u | c; }
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    *len = need;
    return cp;
}

static int utf8_put(uint32_t cp, char *o) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* GPT-2 bytes_to_unicode */
static void byte_unicode_map(uint32_t b2u[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) b2u[b] = (uint32_t)b;
        else b2u[b] = 256 + n++;
    }
}

/* ------------------------------------------------------------------ hashing */

static uint32_t fnv1a(const char *s, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static uint32_t hash64(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull; k ^= k >> 33;
    return (uint32_t)k;
}

static int vocab_lookup(const Tokenizer *t, const char *s, int n) {
    uint32_t i = fnv1a(s, n) & t->vmask;
    for (;;) {
        int32_t id = t->vmap[i];
        if (id < 0) return -1;
        if (t->text_len[id] == n && memcmp(t->text[id], s, (size_t)n) == 0) return id;
        i = (i + 1) & t->vmask;
    }
}

static const MergeSlot *merge_lookup(const Tokenizer *t, int a, int b) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint32_t i = hash64(key) & t->mmask;
    for (;;) {
        const MergeSlot *s = &t->mmap[i];
        if (s->rank < 0) return NULL;
        if (s->key == key) return s;
        i = (i + 1) & t->mmask;
    }
}

/* ------------------------------------------------------------------ load */

static const Tokenizer *g_sort_tok;
static int cmp_special_len(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return g_sort_tok->text_len[ib] - g_sort_tok->text_len[ia];
}

Tokenizer *tok_load(const GGUF *g, char *err, int errlen) {
    const GGUFKV *kt = gguf_find(g, "tokenizer.ggml.tokens");
    const GGUFKV *ky = gguf_find(g, "tokenizer.ggml.token_type");
    const GGUFKV *km = gguf_find(g, "tokenizer.ggml.merges");
    uint64_t ml = 0;
    const char *model = gguf_get_str(g, "tokenizer.ggml.model", &ml);
    if (!kt || !km || !model || ml != 4 || memcmp(model, "gpt2", 4) != 0) {
        snprintf(err, errlen, "unsupported tokenizer (need gpt2 byte-level BPE)");
        return NULL;
    }
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    int n = t->n_vocab = (int)kt->arr_len;
    t->text = (const char **)malloc(sizeof(char *) * n);
    t->text_len = (int *)malloc(sizeof(int) * n);
    t->piece = (char **)malloc(sizeof(char *) * n);
    t->piece_len = (int *)malloc(sizeof(int) * n);
    t->type = (int *)malloc(sizeof(int) * n);

    const uint8_t *cur = kt->val;
    for (int i = 0; i < n; i++) {
        uint64_t l;
        t->text[i] = gguf_next_str(&cur, &l);
        t->text_len[i] = (int)l;
        t->type[i] = ky ? (int)gguf_arr_int(ky, (uint64_t)i) : TT_NORMAL;
    }

    /* vocab hash */
    uint32_t cap = 1;
    while (cap < (uint32_t)n * 2) cap <<= 1;
    t->vmask = cap - 1;
    t->vmap = (int32_t *)malloc(sizeof(int32_t) * cap);
    memset(t->vmap, 0xff, sizeof(int32_t) * cap);
    for (int i = 0; i < n; i++) {
        uint32_t h = fnv1a(t->text[i], t->text_len[i]) & t->vmask;
        while (t->vmap[h] >= 0) h = (h + 1) & t->vmask;
        t->vmap[h] = i;
    }

    /* byte-level maps */
    uint32_t b2u[256];
    byte_unicode_map(b2u);
    int u2b[512];
    for (int i = 0; i < 512; i++) u2b[i] = -1;
    for (int b = 0; b < 256; b++) {
        u2b[b2u[b]] = b;
        char buf[4];
        int l = utf8_put(b2u[b], buf);
        t->byte_tok[b] = vocab_lookup(t, buf, l);
    }

    /* decoded pieces */
    for (int i = 0; i < n; i++) {
        const char *s = t->text[i];
        int l = t->text_len[i];
        char *p = (char *)malloc((size_t)l + 1);
        int pl = 0;
        if (t->type[i] == TT_CONTROL || t->type[i] == TT_USER) {
            memcpy(p, s, (size_t)l);
            pl = l;
        } else {
            for (int j = 0; j < l;) {
                int cl;
                uint32_t cp = utf8_next((const unsigned char *)s + j, l - j, &cl);
                if (cp < 512 && u2b[cp] >= 0) p[pl++] = (char)u2b[cp];
                else { memcpy(p + pl, s + j, (size_t)cl); pl += cl; }
                j += cl;
            }
        }
        p[pl] = 0;
        t->piece[i] = p;
        t->piece_len[i] = pl;
    }

    /* merges */
    int nm = (int)km->arr_len;
    cap = 1;
    while (cap < (uint32_t)nm * 2) cap <<= 1;
    t->mmask = cap - 1;
    t->mmap = (MergeSlot *)malloc(sizeof(MergeSlot) * cap);
    for (uint32_t i = 0; i < cap; i++) t->mmap[i].rank = -1;
    cur = km->val;
    char tmp[1024];
    int bad = 0;
    for (int r = 0; r < nm; r++) {
        uint64_t l;
        const char *s = gguf_next_str(&cur, &l);
        const char *sp = NULL;
        for (uint64_t j = 1; j < l; j++) if (s[j] == ' ') { sp = s + j; break; }
        if (!sp || l >= sizeof(tmp)) { bad++; continue; }
        int la = (int)(sp - s), lb = (int)l - la - 1;
        int a = vocab_lookup(t, s, la), b = vocab_lookup(t, sp + 1, lb);
        memcpy(tmp, s, (size_t)la);
        memcpy(tmp + la, sp + 1, (size_t)lb);
        int m = vocab_lookup(t, tmp, la + lb);
        if (a < 0 || b < 0 || m < 0) { bad++; continue; }
        uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
        uint32_t h = hash64(key) & t->mmask;
        while (t->mmap[h].rank >= 0 && t->mmap[h].key != key) h = (h + 1) & t->mmask;
        if (t->mmap[h].rank >= 0) continue; /* keep first (lowest rank) */
        t->mmap[h].key = key;
        t->mmap[h].rank = r;
        t->mmap[h].merged = m;
    }
    (void)bad;

    /* special tokens */
    t->special = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++)
        if ((t->type[i] == TT_CONTROL || t->type[i] == TT_USER) && t->text_len[i] > 0) t->special[t->n_special++] = i;
    g_sort_tok = t;
    qsort(t->special, (size_t)t->n_special, sizeof(int), cmp_special_len);

    t->bos = (int)gguf_get_int(g, "tokenizer.ggml.bos_token_id", -1);
    t->eos = (int)gguf_get_int(g, "tokenizer.ggml.eos_token_id", -1);
    return t;
}

void tok_free(Tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->n_vocab; i++) free(t->piece[i]);
    free(t->piece); free(t->piece_len); free(t->text); free(t->text_len); free(t->type);
    free(t->vmap); free(t->mmap); free(t->special);
    free(t);
}

/* ------------------------------------------------------------------ BPE */

static int bpe_word(const Tokenizer *t, const unsigned char *s, int n, int *out, int max_out, int count) {
    /* no whole-word shortcut: always apply merges in rank order, like the reference BPE */
    int stack_sym[256];
    int *sym = n <= 256 ? stack_sym : (int *)malloc(sizeof(int) * n);
    int ns = n;
    for (int i = 0; i < n; i++) sym[i] = t->byte_tok[s[i]];
    for (;;) {
        int best = -1, best_rank = 0x7fffffff, best_m = -1;
        for (int i = 0; i + 1 < ns; i++) {
            const MergeSlot *m = merge_lookup(t, sym[i], sym[i + 1]);
            if (m && m->rank < best_rank) { best_rank = m->rank; best = i; best_m = m->merged; }
        }
        if (best < 0) break;
        sym[best] = best_m;
        memmove(sym + best + 1, sym + best + 2, sizeof(int) * (size_t)(ns - best - 2));
        ns--;
    }
    for (int i = 0; i < ns; i++) {
        if (count < max_out) out[count] = sym[i];
        count++;
    }
    if (sym != stack_sym) free(sym);
    return count;
}

/* LFM2 / llama3 regex:
 * (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
 *  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ */
static int encode_plain(const Tokenizer *t, const unsigned char *s, int n, int *out, int max_out, int count) {
    if (n <= 0) return count;
    /* decode to codepoints with byte offsets */
    int cap = n + 1;
    uint32_t *cp = (uint32_t *)malloc(sizeof(uint32_t) * cap);
    int *off = (int *)malloc(sizeof(int) * cap);
    unsigned char *fl = (unsigned char *)malloc((size_t)cap);
    int nc = 0;
    for (int i = 0; i < n;) {
        int l;
        uint32_t c = utf8_next(s + i, n - i, &l);
        off[nc] = i;
        cp[nc] = c;
        fl[nc] = (unsigned char)((c >> 16) == 0xFFFD ? 0 : uni_flags(c));
        nc++;
        i += l;
    }
    off[nc] = n;
#define CP(k) ((k) < nc ? cp[k] : 0xFFFFFFFFu)
#define FL(k) ((k) < nc ? fl[k] : 0)
    int pos = 0;
    while (pos < nc) {
        int start = pos;
        uint32_t c = cp[pos];
        int f = fl[pos];
        /* contractions */
        if (c == '\'' && pos + 1 < nc) {
            uint32_t c1 = CP(pos + 1);
            if (c1 < 128 && c1 >= 'A' && c1 <= 'Z') c1 += 32;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { pos += 2; goto emit; }
            if (pos + 2 < nc) {
                uint32_t c2 = CP(pos + 2);
                if (c2 < 128 && c2 >= 'A' && c2 <= 'Z') c2 += 32;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { pos += 3; goto emit; }
            }
        }
        /* [^\r\n\p{L}\p{N}]?\p{L}+ */
        if (!(c == '\r' || c == '\n' || (f & UNI_NUMBER))) {
            if ((f & UNI_LETTER) || (FL(pos + 1) & UNI_LETTER)) {
                pos++;
                while (FL(pos) & UNI_LETTER) pos++;
                goto emit;
            }
        }
        /* \p{N}{1,3} */
        if (f & UNI_NUMBER) {
            int k = 0;
            while (k < 3 && (FL(pos) & UNI_NUMBER)) { pos++; k++; }
            goto emit;
        }
        /*  ?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            int p2 = (c == ' ') ? pos + 1 : pos;
            if (p2 < nc && !(FL(p2) & (UNI_SPACE | UNI_LETTER | UNI_NUMBER))) {
                pos = p2;
                while (pos < nc && !(FL(pos) & (UNI_SPACE | UNI_LETTER | UNI_NUMBER))) pos++;
                while (CP(pos) == '\r' || CP(pos) == '\n') pos++;
                goto emit;
            }
        }
        /* \s*[\r\n]+ | \s+(?!\S) | \s+ */
        {
            int nw = 0, last_rn = 0;
            while (FL(pos + nw) & UNI_SPACE) {
                uint32_t c2 = CP(pos + nw);
                if (c2 == '\r' || c2 == '\n') last_rn = pos + nw + 1;
                nw++;
            }
            if (last_rn > 0) { pos = last_rn; goto emit; }
            if (nw > 1 && pos + nw < nc) { pos += nw - 1; goto emit; }
            if (nw > 0) { pos += nw; goto emit; }
        }
        pos++;
    emit:
        count = bpe_word(t, s + off[start], off[pos] - off[start], out, max_out, count);
    }
#undef CP
#undef FL
    free(cp);
    free(off);
    free(fl);
    return count;
}

int tok_encode(const Tokenizer *t, const char *text, int len, int parse_special, int *out, int max_out) {
    const unsigned char *s = (const unsigned char *)text;
    int count = 0, seg = 0;
    if (parse_special) {
        for (int i = 0; i < len;) {
            int matched = -1;
            if (s[i] == '<') {
                for (int k = 0; k < t->n_special; k++) {
                    int id = t->special[k], l = t->text_len[id];
                    if (l <= len - i && memcmp(s + i, t->text[id], (size_t)l) == 0) { matched = id; break; }
                }
            }
            if (matched >= 0) {
                count = encode_plain(t, s + seg, i - seg, out, max_out, count);
                if (count < max_out) out[count] = matched;
                count++;
                i += t->text_len[matched];
                seg = i;
            } else {
                i++;
            }
        }
    }
    count = encode_plain(t, s + seg, len - seg, out, max_out, count);
    return count;
}

const char *tok_piece(const Tokenizer *t, int id, int *len) {
    if (id < 0 || id >= t->n_vocab) { *len = 0; return ""; }
    *len = t->piece_len[id];
    return t->piece[id];
}

int tok_n_vocab(const Tokenizer *t) { return t->n_vocab; }
int tok_bos(const Tokenizer *t) { return t->bos; }
int tok_eos(const Tokenizer *t) { return t->eos; }
int tok_is_control(const Tokenizer *t, int id) { return id >= 0 && id < t->n_vocab && t->type[id] == TT_CONTROL; }
int tok_find(const Tokenizer *t, const char *s) { return vocab_lookup(t, s, (int)strlen(s)); }
