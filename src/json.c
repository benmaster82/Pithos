/* json.c - recursive-descent JSON parser and writer. */
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end;
    char *err;
    int errlen, depth;
} Parser;

static void fail(Parser *ps, const char *msg) {
    if (ps->err && !ps->err[0]) snprintf(ps->err, (size_t)ps->errlen, "%s", msg);
}

static void skip_ws(Parser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static Json *node(JType t) {
    Json *j = (Json *)calloc(1, sizeof(Json));
    j->t = t;
    return j;
}

static int hex4(const char *p, unsigned *v) {
    *v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        *v <<= 4;
        if (c >= '0' && c <= '9') *v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') *v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') *v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    return 1;
}

static int put_utf8(char *o, unsigned cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | cp >> 6); o[1] = (char)(0x80 | (cp & 63)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | cp >> 12); o[1] = (char)(0x80 | ((cp >> 6) & 63)); o[2] = (char)(0x80 | (cp & 63));
        return 3;
    }
    o[0] = (char)(0xF0 | cp >> 18); o[1] = (char)(0x80 | ((cp >> 12) & 63));
    o[2] = (char)(0x80 | ((cp >> 6) & 63)); o[3] = (char)(0x80 | (cp & 63));
    return 4;
}

/* Parses a string literal at ps->p (on the opening quote); returns malloc'ed UTF-8. */
static char *parse_string(Parser *ps, int *out_len) {
    ps->p++;
    const char *q = ps->p;
    /* output is never longer than the escaped input */
    while (q < ps->end && *q != '"') q += (*q == '\\' && q + 1 < ps->end) ? 2 : 1;
    if (q >= ps->end) { fail(ps, "unterminated string"); return NULL; }
    char *o = (char *)malloc((size_t)(q - ps->p) + 1);
    int n = 0;
    while (ps->p < q) {
        char c = *ps->p++;
        if ((unsigned char)c < 0x20) { free(o); fail(ps, "control character in string"); return NULL; }
        if (c != '\\') { o[n++] = c; continue; }
        c = *ps->p++;
        switch (c) {
        case '"': o[n++] = '"'; break;
        case '\\': o[n++] = '\\'; break;
        case '/': o[n++] = '/'; break;
        case 'b': o[n++] = '\b'; break;
        case 'f': o[n++] = '\f'; break;
        case 'n': o[n++] = '\n'; break;
        case 'r': o[n++] = '\r'; break;
        case 't': o[n++] = '\t'; break;
        case 'u': {
            unsigned cp, lo;
            if (q - ps->p < 4 || !hex4(ps->p, &cp)) { free(o); fail(ps, "bad \\u escape"); return NULL; }
            ps->p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && q - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u' &&
                hex4(ps->p + 2, &lo) && lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                ps->p += 6;
            } else if (cp >= 0xD800 && cp < 0xE000) {
                cp = 0xFFFD; /* lone surrogate */
            }
            n += put_utf8(o + n, cp);
            break;
        }
        default: free(o); fail(ps, "bad escape"); return NULL;
        }
    }
    ps->p = q + 1;
    o[n] = 0;
    *out_len = n;
    return o;
}

static Json *parse_value(Parser *ps);

static Json *parse_container(Parser *ps, int is_obj) {
    Json *j = node(is_obj ? J_OBJ : J_ARR);
    Json **tail = &j->child;
    ps->p++;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == (is_obj ? '}' : ']')) { ps->p++; return j; }
    for (;;) {
        char *key = NULL;
        if (is_obj) {
            skip_ws(ps);
            int kl;
            if (ps->p >= ps->end || *ps->p != '"' || !(key = parse_string(ps, &kl))) {
                fail(ps, "expected member name");
                json_free(j);
                return NULL;
            }
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') { free(key); fail(ps, "expected ':'"); json_free(j); return NULL; }
            ps->p++;
        }
        Json *v = parse_value(ps);
        if (!v) { free(key); json_free(j); return NULL; }
        v->key = key;
        *tail = v;
        tail = &v->next;
        j->n++;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == (is_obj ? '}' : ']')) { ps->p++; return j; }
        fail(ps, is_obj ? "expected ',' or '}'" : "expected ',' or ']'");
        json_free(j);
        return NULL;
    }
}

static Json *parse_value(Parser *ps) {
    skip_ws(ps);
    if (ps->p >= ps->end) { fail(ps, "unexpected end of input"); return NULL; }
    if (++ps->depth > 200) { fail(ps, "nesting too deep"); return NULL; }
    Json *j = NULL;
    char c = *ps->p;
    if (c == '{' || c == '[') {
        j = parse_container(ps, c == '{');
    } else if (c == '"') {
        int n;
        char *s = parse_string(ps, &n);
        if (s) { j = node(J_STR); j->s = s; j->len = n; }
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        const char *st = ps->p;
        while (ps->p < ps->end && strchr("+-0123456789.eE", *ps->p)) ps->p++;
        j = node(J_NUM);
        j->len = (int)(ps->p - st);
        j->s = (char *)malloc((size_t)j->len + 1);
        memcpy(j->s, st, (size_t)j->len);
        j->s[j->len] = 0;
        char *e;
        j->num = strtod(j->s, &e);
        if (*e) { fail(ps, "bad number"); json_free(j); j = NULL; }
    } else if (ps->end - ps->p >= 4 && !memcmp(ps->p, "true", 4)) {
        j = node(J_TRUE); ps->p += 4;
    } else if (ps->end - ps->p >= 5 && !memcmp(ps->p, "false", 5)) {
        j = node(J_FALSE); ps->p += 5;
    } else if (ps->end - ps->p >= 4 && !memcmp(ps->p, "null", 4)) {
        j = node(J_NULL); ps->p += 4;
    } else {
        fail(ps, "unexpected character");
    }
    ps->depth--;
    return j;
}

Json *json_parse(const char *text, int len, char *err, int errlen) {
    Parser ps = { text, text + len, err, errlen, 0 };
    if (err && errlen) err[0] = 0;
    Json *j = parse_value(&ps);
    if (!j) return NULL;
    skip_ws(&ps);
    if (ps.p != ps.end) { fail(&ps, "trailing characters"); json_free(j); return NULL; }
    return j;
}

void json_free(Json *j) {
    while (j) {
        Json *nx = j->next;
        json_free(j->child);
        free(j->key);
        free(j->s);
        free(j);
        j = nx;
    }
}

Json *json_get(const Json *obj, const char *key) {
    if (!obj || obj->t != J_OBJ) return NULL;
    Json *found = NULL;
    for (Json *c = obj->child; c; c = c->next)
        if (!strcmp(c->key, key)) found = c; /* last duplicate wins, like most parsers */
    return found;
}

const char *json_str(const Json *j, const char *def) { return j && j->t == J_STR ? j->s : def; }
double json_num(const Json *j, double def) { return j && j->t == J_NUM ? j->num : def; }
int json_bool(const Json *j, int def) {
    if (!j) return def;
    if (j->t == J_TRUE) return 1;
    if (j->t == J_FALSE || j->t == J_NULL) return 0;
    if (j->t == J_NUM) return j->num != 0;
    return def;
}

/* ------------------------------------------------------------------ writer */

void buf_add(Buf *b, const char *s, int n) {
    if (n < 0) n = (int)strlen(s);
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2 + 64;
        b->s = (char *)realloc(b->s, (size_t)b->cap);
    }
    memcpy(b->s + b->n, s, (size_t)n);
    b->n += n;
    b->s[b->n] = 0;
}

void buf_printf(Buf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < (int)sizeof(tmp)) { buf_add(b, tmp, n); return; }
    char *big = (char *)malloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_add(b, big, n);
    free(big);
}

void buf_free(Buf *b) {
    free(b->s);
    b->s = NULL;
    b->n = b->cap = 0;
}

void buf_json_str(Buf *b, const char *s, int n) {
    if (n < 0) n = (int)strlen(s);
    buf_add(b, "\"", 1);
    int st = 0;
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        const char *esc = NULL;
        char u[8];
        switch (c) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        default:
            if (c < 0x20) { snprintf(u, sizeof(u), "\\u%04x", c); esc = u; }
        }
        if (esc) {
            buf_add(b, s + st, i - st);
            buf_add(b, esc, -1);
            st = i + 1;
        }
    }
    buf_add(b, s + st, n - st);
    buf_add(b, "\"", 1);
}

void json_write(Buf *b, const Json *j) {
    switch (j->t) {
    case J_NULL: buf_add(b, "null", 4); break;
    case J_TRUE: buf_add(b, "true", 4); break;
    case J_FALSE: buf_add(b, "false", 5); break;
    case J_NUM: buf_add(b, j->s, j->len); break;
    case J_STR: buf_json_str(b, j->s, j->len); break;
    case J_ARR:
    case J_OBJ:
        buf_add(b, j->t == J_ARR ? "[" : "{", 1);
        for (const Json *c = j->child; c; c = c->next) {
            if (c != j->child) buf_add(b, ", ", 2);
            if (j->t == J_OBJ) { buf_json_str(b, c->key, -1); buf_add(b, ": ", 2); }
            json_write(b, c);
        }
        buf_add(b, j->t == J_ARR ? "]" : "}", 1);
        break;
    }
}
