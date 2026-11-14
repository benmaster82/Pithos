/* server.c - minimal HTTP/1.1 server with an OpenAI-compatible API and the web chat UI.
 *
 * One request at a time runs on the model (an engine lock); every connection has its own
 * thread so the UI, /health and queued requests are served while a generation is running.
 * Prompts reuse the longest cached prefix (pcache): in a multi-turn chat only the new
 * message is processed. After each answer the context is rewritten in the form the chat
 * template gives to past turns (reasoning removed), while the client is reading. */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET sock_t;
#define BAD_SOCK INVALID_SOCKET
#define sock_close closesocket
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
typedef int sock_t;
#define BAD_SOCK (-1)
#define sock_close close
#endif

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "json.h"
#include "platform.h"
#include "server.h"
#include "audio.h"
#include "speech.h"
#include "webui_html.h"
#include "talk_html.h"

#define MAX_HEADER (64 * 1024)
#define MAX_BODY (64 * 1024 * 1024)
#define PREFILL_PIECE 128
#define MAX_STOPS 8

typedef struct {
    ServerConfig cfg;
    Model *m;
    Context *ctx;
    Tokenizer *tk;
    PCache *pc;
    PithosLock *lock;
    int t_im_end, t_eot, t_eos, t_think_end, t_tc_start, t_tc_end;
    int lfm3;                   /* LFM3 template: XML tool calls, <tools> block, <think></think> on past turns */
    int reasons;                /* template_reasons: 0 for LFM3-Audio (no reasoning blocks) */
    int t_audio, t_think_start, t_audio_out, t_text_end, t_mixed_start, t_audio_start;
    /* Audio of the request being rendered, under the lock (audio_begin / audio_end): render_content
     * turns input_audio parts into clips and toks_text each <|audio|> into the clip's token ids
     * -1, -2... (ctx_set_embeds); prompts with them are not reused (the ids repeat). Only the thread
     * serving that request (t_collect) does so: /tokenize and /apply-template run without the lock. */
    struct { float *pcm; int n, ntok; } clip[16];
    int n_clips, clip_used, clip_ids;
    char clip_err[160];
    float *emb;
    uint64_t rng;
    volatile int busy;
    char *ui;
    int ui_len;
} Server;

static Server S;
static PITHOS_THREAD_LOCAL int t_collect;   /* this thread renders an audio request (audio_begin) */
static PITHOS_THREAD_LOCAL int t_voice;     /* this thread renders a request answered in speech (handle_chat) */

/* ------------------------------------------------------------------ connection */

typedef struct {
    sock_t s;
    int dead, closed;
    char peer[64];
    /* in-process client (server_local_chat): the response goes to sink instead of a socket */
    void (*sink)(void *u, const char *s, int n);
    void *sink_u;
    volatile int *stop;
} Conn;

static int conn_send(Conn *c, const char *p, int n) {
    if (n < 0) n = (int)strlen(p);
    if (c->sink) {
        c->sink(c->sink_u, p, n);
        return 1;
    }
    while (n > 0 && !c->dead) {
        int k = (int)send(c->s, p, n, 0);
        if (k <= 0) { c->dead = 1; break; }
        p += k;
        n -= k;
    }
    return !c->dead;
}

/* Non-blocking check: has the client closed the connection? */
static int conn_gone(Conn *c) {
    if (c->dead) return 1;
    if (c->sink) return c->stop && *c->stop;
    fd_set rs;
    FD_ZERO(&rs);
    FD_SET(c->s, &rs);
    struct timeval tv = { 0, 0 };
    if (select((int)c->s + 1, &rs, NULL, NULL, &tv) > 0) {
        char b;
        if (recv(c->s, &b, 1, MSG_PEEK) <= 0) c->dead = 1;
    }
    return c->dead;
}

static void set_recv_timeout(sock_t s, int ms) {
#if defined(_WIN32)
    DWORD t = (DWORD)ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&t, sizeof(t));
#else
    struct timeval tv = { ms / 1000, (ms % 1000) * 1000 };
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

/* Graceful close: the client sees the whole response even if it sent more than we read. */
static void conn_close(Conn *c) {
    if (c->closed) return;
    c->closed = 1;
    if (c->sink) return;
#if defined(_WIN32)
    shutdown(c->s, SD_SEND);
#else
    shutdown(c->s, SHUT_WR);
#endif
    set_recv_timeout(c->s, 500);
    char b[512];
    for (int i = 0; i < 64 && recv(c->s, b, sizeof(b), 0) > 0; i++) {}
    sock_close(c->s);
}

/* ------------------------------------------------------------------ HTTP */

typedef struct {
    char method[16], path[512];
    char *body;
    int body_len;
    char auth[512];
    char ctype[256];
} Req;

static const char *status_text(int code) {
    switch (code) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    default: return "Error";
    }
}

static int header_value(const char *hdr, const char *name, char *out, int cap) {
    size_t nl = strlen(name);
    for (const char *p = hdr; p && *p; ) {
        const char *eol = strstr(p, "\r\n");
        if (!eol) break;
        if ((size_t)(eol - p) > nl && p[nl] == ':') {
            int match = 1;
            for (size_t i = 0; i < nl; i++)
                if (tolower((unsigned char)p[i]) != tolower((unsigned char)name[i])) { match = 0; break; }
            if (match) {
                const char *v = p + nl + 1;
                while (v < eol && (*v == ' ' || *v == '\t')) v++;
                int n = (int)(eol - v);
                if (n >= cap) n = cap - 1;
                memcpy(out, v, (size_t)n);
                out[n] = 0;
                return 1;
            }
        }
        p = eol + 2;
    }
    return 0;
}

static int read_request(Conn *c, Req *r) {
    memset(r, 0, sizeof(*r));
    char *buf = (char *)malloc(MAX_HEADER + 1);
    int n = 0;
    char *hend = NULL;
    set_recv_timeout(c->s, 30000);
    while (!hend) {
        if (n >= MAX_HEADER) { free(buf); return 413; }
        int k = (int)recv(c->s, buf + n, MAX_HEADER - n, 0);
        if (k <= 0) { free(buf); return 0; }
        n += k;
        buf[n] = 0;
        hend = strstr(buf, "\r\n\r\n");
    }
    int hlen = (int)(hend - buf) + 4;
    if (sscanf(buf, "%15s %511s", r->method, r->path) != 2) { free(buf); return 400; }
    char *q = strchr(r->path, '?');
    if (q) *q = 0;
    hend[2] = 0; /* headers end with the last \r\n */
    const char *hdr = strstr(buf, "\r\n") + 2;
    char v[64];
    long long clen = 0;
    if (header_value(hdr, "Content-Length", v, sizeof(v))) clen = atoll(v);
    if (clen < 0 || clen > MAX_BODY) { free(buf); return 413; }
    header_value(hdr, "Authorization", r->auth, sizeof(r->auth));
    header_value(hdr, "Content-Type", r->ctype, sizeof(r->ctype));
    if (clen > n - hlen && header_value(hdr, "Expect", v, sizeof(v)) && strstr(v, "100"))
        conn_send(c, "HTTP/1.1 100 Continue\r\n\r\n", -1);
    r->body = (char *)malloc((size_t)clen + 1);
    int have = n - hlen;
    if (have > clen) have = (int)clen;
    memcpy(r->body, buf + hlen, (size_t)have);
    free(buf);
    while (have < clen) {
        int k = (int)recv(c->s, r->body + have, (int)(clen - have), 0);
        if (k <= 0) { free(r->body); r->body = NULL; return 0; }
        have += k;
    }
    r->body[clen] = 0;
    r->body_len = (int)clen;
    return 200;
}

static void send_head(Conn *c, int code, const char *ctype, long long len) {
    char h[512];
    int n = snprintf(h, sizeof(h),
                     "HTTP/1.1 %d %s\r\nServer: pithos\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n",
                     code, status_text(code));
    if (ctype) n += snprintf(h + n, sizeof(h) - (size_t)n, "Content-Type: %s\r\n", ctype);
    if (len >= 0) n += snprintf(h + n, sizeof(h) - (size_t)n, "Content-Length: %lld\r\n", len);
    else n += snprintf(h + n, sizeof(h) - (size_t)n, "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n");
    n += snprintf(h + n, sizeof(h) - (size_t)n, "\r\n");
    conn_send(c, h, n);
}

static void send_body(Conn *c, int code, const char *ctype, const char *body, int len) {
    send_head(c, code, ctype, len);
    conn_send(c, body, len);
}

static void send_json(Conn *c, int code, const Buf *b) { send_body(c, code, "application/json; charset=utf-8", b->s, b->n); }

static void send_error(Conn *c, int code, const char *type, const char *msg) {
    Buf b = { 0 };
    buf_add(&b, "{\"error\":{\"code\":", -1);
    buf_printf(&b, "%d,\"type\":", code);
    buf_json_str(&b, type, -1);
    buf_add(&b, ",\"message\":", -1);
    buf_json_str(&b, msg, -1);
    buf_add(&b, "}}", 2);
    send_json(c, code, &b);
    buf_free(&b);
}

/* ------------------------------------------------------------------ tokens */

typedef struct { int *v; int n, cap; } Toks;

static void toks_push(Toks *t, int id) {
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 1024;
        t->v = (int *)realloc(t->v, sizeof(int) * (size_t)t->cap);
    }
    t->v[t->n++] = id;
}

static void toks_text(Toks *t, const char *s, int len) {
    if (len < 0) len = (int)strlen(s);
    int need = len + 8;
    if (t->n + need > t->cap) {
        t->cap = (t->n + need) * 2;
        t->v = (int *)realloc(t->v, sizeof(int) * (size_t)t->cap);
    }
    int k = tok_encode(S.tk, s, len, 1, t->v + t->n, t->cap - t->n);
    if (k > t->cap - t->n) {
        t->cap = t->n + k + 1024;
        t->v = (int *)realloc(t->v, sizeof(int) * (size_t)t->cap);
        k = tok_encode(S.tk, s, len, 1, t->v + t->n, t->cap - t->n);
    }
    t->n += k;
    for (int i = t->n - k; i < t->n && t_collect; i++) {
        if (t->v[i] != S.t_audio || S.clip_used >= S.n_clips) continue;
        const int nt = S.clip[S.clip_used].ntok;
        if (t->n + nt > t->cap) {
            t->cap = t->n + nt + 1024;
            t->v = (int *)realloc(t->v, sizeof(int) * (size_t)t->cap);
        }
        memmove(t->v + i + nt, t->v + i + 1, sizeof(int) * (size_t)(t->n - i - 1));
        for (int j = 0; j < nt; j++) t->v[i + j] = -1 - (S.clip_ids + j);
        S.clip_ids += nt;
        S.clip_used++;
        t->n += nt - 1;
        i += nt - 1;
    }
}

/* ------------------------------------------------------------------ audio input */

static int b64_val(int ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+' || ch == '-') return 62;
    if (ch == '/' || ch == '_') return 63;
    return -1;
}

/* base64 (standard or URL alphabet, padding and whitespace ignored); the caller frees */
static unsigned char *b64_decode(const char *s, int len, size_t *out_len) {
    unsigned char *o = (unsigned char *)malloc((size_t)len / 4 * 3 + 3);
    size_t n = 0;
    unsigned acc = 0;
    int bits = 0;
    for (int i = 0; i < len; i++) {
        const int v = b64_val((unsigned char)s[i]);
        if (v < 0) continue;
        acc = (acc << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            o[n++] = (unsigned char)(acc >> bits);
        }
    }
    *out_len = n;
    return o;
}

static void add_clip(const unsigned char *wav, size_t len) {
    if (S.clip_err[0]) return;
    if (S.n_clips >= (int)(sizeof(S.clip) / sizeof(S.clip[0]))) { snprintf(S.clip_err, sizeof(S.clip_err), "too many audio parts"); return; }
    char e[120];
    int n = 0;
    float *pcm = audio_decode_wav(wav, len, &n, e, sizeof(e));
    if (!pcm || n <= 0) {
        snprintf(S.clip_err, sizeof(S.clip_err), "audio part %d: %s", S.n_clips + 1, pcm ? "no samples" : e);
        free(pcm);
        return;
    }
    S.clip[S.n_clips].pcm = pcm;
    S.clip[S.n_clips].n = n;
    S.clip[S.n_clips].ntok = audio_n_tokens(n);
    S.n_clips++;
}

/* an OpenAI input_audio part: {"data": base64 WAV, "format": "wav"} */
static void audio_part(Buf *b, const Json *in) {
    buf_add(b, "<|audio|>", -1);
    if (!t_collect) return;
    const char *fmt = json_str(json_get(in, "format"), "wav");
    const Json *d = json_get(in, "data");
    if (strcmp(fmt, "wav") || !d || d->t != J_STR) {
        if (!S.clip_err[0]) snprintf(S.clip_err, sizeof(S.clip_err), "input_audio needs \"data\": base64 WAV, \"format\": \"wav\"");
        return;
    }
    size_t len = 0;
    unsigned char *wav = b64_decode(d->s, d->len, &len);
    add_clip(wav, len);
    free(wav);
}

static int has_audio_parts(const Json *messages) {
    for (const Json *m = messages ? messages->child : NULL; m; m = m->next) {
        const Json *c = json_get(m, "content");
        for (const Json *it = c && c->t == J_ARR ? c->child : NULL; it; it = it->next)
            if (it->t == J_OBJ && !strcmp(json_str(json_get(it, "type"), ""), "input_audio")) return 1;
    }
    return 0;
}

/* under the lock: from here render_content and toks_text collect the request's audio */
static void audio_begin(void) {
    t_collect = 1;
    S.n_clips = S.clip_used = S.clip_ids = 0;
    S.clip_err[0] = 0;
}

/* the embeddings of the collected clips, in token order; 0 with err on a mismatch */
static int audio_encode(char *err, int errlen) {
    if (S.clip_used != S.n_clips) { snprintf(err, (size_t)errlen, "%d audio parts but %d <|audio|> placeholders", S.n_clips, S.clip_used); return 0; }
    const int D = S.m->dim;
    S.emb = (float *)malloc(sizeof(float) * (size_t)(S.clip_ids ? S.clip_ids : 1) * D);
    int off = 0;
    for (int i = 0; i < S.n_clips; i++) {
        int n = 0;
        float *e = audio_embed(S.cfg.audio, S.clip[i].pcm, S.clip[i].n, &n);
        if (n != S.clip[i].ntok) { free(e); snprintf(err, (size_t)errlen, "audio part %d: encoding failed", i + 1); return 0; }
        memcpy(S.emb + (size_t)off * D, e, sizeof(float) * (size_t)n * D);
        off += n;
        free(e);
    }
    ctx_set_embeds(S.ctx, S.emb, S.clip_ids);
    return 1;
}

static void audio_end(void) {
    for (int i = 0; i < S.n_clips; i++) free(S.clip[i].pcm);
    S.n_clips = S.clip_used = S.clip_ids = 0;
    t_collect = 0;
    S.clip_err[0] = 0;
    ctx_set_embeds(S.ctx, NULL, 0);
    free(S.emb);
    S.emb = NULL;
}

/* ------------------------------------------------------------------ chat template (LFM2) */

static int truthy(const Json *j) {
    if (!j) return 0;
    switch (j->t) {
    case J_NULL: case J_FALSE: return 0;
    case J_STR: return j->len > 0;
    case J_ARR: case J_OBJ: return j->n > 0;
    case J_NUM: return j->num != 0;
    default: return 1;
    }
}

/* parse_content macro: string, list of parts (text / image), or JSON */
static void render_content(Buf *b, const Json *c) {
    if (!c) return;
    if (c->t == J_STR) { buf_add(b, c->s, c->len); return; }
    if (c->t == J_ARR) {
        for (const Json *it = c->child; it; it = it->next) {
            if (it->t == J_STR) { buf_add(b, it->s, it->len); continue; }
            const char *type = json_str(json_get(it, "type"), NULL);
            if (type && !strcmp(type, "image")) buf_add(b, S.lfm3 ? "<|vision_start|><|image|><|vision_end|>" : "<image>", -1);
            else if (type && !strcmp(type, "text")) buf_add(b, json_str(json_get(it, "text"), ""), -1);
            else if (type && !strcmp(type, "input_audio")) audio_part(b, json_get(it, "input_audio"));
            else json_write(b, it);
        }
        return;
    }
    json_write(b, c);
}

static void render_arg_value(Buf *b, const Json *v) {
    switch (v->t) {
    case J_STR:
        buf_add(b, "'", 1);
        for (int i = 0; i < v->len; i++) {
            char ch = v->s[i];
            if (ch == '\\') buf_add(b, "\\\\", 2);
            else if (ch == '\'') buf_add(b, "\\'", 2);
            else if (ch == '\n') buf_add(b, "\\n", 2);
            else if (ch == '\r') buf_add(b, "\\r", 2);
            else buf_add(b, &ch, 1);
        }
        buf_add(b, "'", 1);
        break;
    case J_TRUE: buf_add(b, "True", 4); break;
    case J_FALSE: buf_add(b, "False", 5); break;
    case J_NULL: buf_add(b, "None", 4); break;
    case J_NUM: buf_add(b, v->s, v->len); break;
    default: json_write(b, v);
    }
}

/* render_tool_calls macro: <|tool_call_start|>[name(arg='v', n=1)]<|tool_call_end|> */
static int render_tool_calls(Buf *b, const Json *calls, char *err, int errlen) {
    buf_add(b, "<|tool_call_start|>[", -1);
    for (const Json *tc = calls->child; tc; tc = tc->next) {
        const Json *fn = json_get(tc, "function");
        if (!fn) fn = tc;
        const char *name = json_str(json_get(fn, "name"), NULL);
        if (!name) { snprintf(err, (size_t)errlen, "tool call without a function name"); return 0; }
        if (tc != calls->child) buf_add(b, ", ", 2);
        buf_add(b, name, -1);
        buf_add(b, "(", 1);
        const Json *args = json_get(fn, "arguments");
        Json *parsed = NULL;
        if (args && args->t == J_STR) {
            /* OpenAI clients send arguments as a JSON-encoded string */
            parsed = json_parse(args->s, args->len, err, errlen);
            if (!parsed && args->len > 0) {
                const char *t = args->s;
                while (*t == ' ') t++;
                if (*t) { snprintf(err, (size_t)errlen, "tool call arguments are not valid JSON"); return 0; }
            }
            args = parsed;
        }
        if (args && args->t == J_OBJ)
            for (const Json *a = args->child; a; a = a->next) {
                if (a != args->child) buf_add(b, ", ", 2);
                buf_add(b, a->key, -1);
                buf_add(b, "=", 1);
                render_arg_value(b, a);
            }
        json_free(parsed);
        buf_add(b, ")", 1);
    }
    buf_add(b, "]<|tool_call_end|>", -1);
    if (err) err[0] = 0;
    return 1;
}

static void trim_view(const char **s, int *n) {
    while (*n > 0 && isspace((unsigned char)**s)) { (*s)++; (*n)--; }
    while (*n > 0 && isspace((unsigned char)(*s)[*n - 1])) (*n)--;
}

/* One assistant turn as the template renders it. reasoning == NULL: past turn without it. */
static int render_assistant(Buf *b, const char *reasoning, int rlen, const char *content, int clen, int keep_thinking,
                            const Json *calls, char *err, int errlen) {
    buf_add(b, "<|im_start|>assistant\n", -1);
    if (reasoning && rlen > 0 && keep_thinking) {
        buf_add(b, "<think>", -1);
        buf_add(b, reasoning, rlen);
        buf_add(b, "</think>", -1);
    }
    if (!keep_thinking && content) {
        /* content.split("</think>")[-1] | trim */
        const char *last = NULL;
        for (const char *p = content; (p = strstr(p, "</think>")) != NULL && p < content + clen; p += 8) last = p;
        if (last) {
            clen -= (int)(last + 8 - content);
            content = last + 8;
            trim_view(&content, &clen);
        }
    }
    if (content) buf_add(b, content, clen);
    if (calls && truthy(calls) && !render_tool_calls(b, calls, err, errlen)) return 0;
    buf_add(b, "<|im_end|>\n", -1);
    return 1;
}

/* ------------------------------------------------------------------ chat template (LFM3) */

static const char *find_in(const char *s, int n, const char *needle) {
    const int k = (int)strlen(needle);
    for (int i = 0; i + k <= n; i++)
        if (s[i] == needle[0] && !memcmp(s + i, needle, (size_t)k)) return s + i;
    return NULL;
}

/* render_value macro: strings verbatim, the rest as JSON (true / false / null in lower case) */
static void render_value_lfm3(Buf *b, const Json *v) {
    if (v->t == J_STR) buf_add(b, v->s, v->len);
    else json_write(b, v);
}

/* <tool_call>\n<function=name>\n<parameter=k>\nvalue\n</parameter>\n</function>\n</tool_call>, "\n\n" between */
static int render_tool_calls_lfm3(Buf *b, const Json *calls, char *err, int errlen) {
    for (const Json *tc = calls->child; tc; tc = tc->next) {
        const Json *fn = json_get(tc, "function");
        if (!fn) fn = tc;
        const char *name = json_str(json_get(fn, "name"), NULL);
        if (!name) { snprintf(err, (size_t)errlen, "tool call without a function name"); return 0; }
        if (tc != calls->child) buf_add(b, "\n\n", 2);
        buf_add(b, "<tool_call>\n<function=", -1);
        buf_add(b, name, -1);
        buf_add(b, ">\n", 2);
        const Json *args = json_get(fn, "arguments");
        Json *parsed = NULL;
        if (args && args->t == J_STR) {
            /* OpenAI clients send arguments as a JSON-encoded string */
            parsed = json_parse(args->s, args->len, err, errlen);
            if (!parsed && args->len > 0) {
                const char *t = args->s;
                while (*t == ' ') t++;
                if (*t) { snprintf(err, (size_t)errlen, "tool call arguments are not valid JSON"); return 0; }
            }
            args = parsed;
        }
        if (args && args->t == J_OBJ)
            for (const Json *a = args->child; a; a = a->next) {
                buf_add(b, "<parameter=", -1);
                buf_add(b, a->key, -1);
                buf_add(b, ">\n", 2);
                render_value_lfm3(b, a);
                buf_add(b, "\n</parameter>\n", -1);
            }
        json_free(parsed);
        buf_add(b, "</function>\n</tool_call>", -1);
    }
    if (err) err[0] = 0;
    return 1;
}

/* tool_to_json macro: the function object without its defer_loading key */
static void tool_to_json_lfm3(Buf *b, const Json *fn) {
    int first = 1;
    buf_add(b, "{", 1);
    for (const Json *c = fn->child; c; c = c->next) {
        if (!strcmp(c->key, "defer_loading")) continue;
        if (!first) buf_add(b, ", ", 2);
        first = 0;
        buf_json_str(b, c->key, -1);
        buf_add(b, ": ", 2);
        json_write(b, c);
    }
    buf_add(b, "}", 1);
}

/* the <tools> block appended to the system prompt; tools with defer_loading are left out */
static void render_tools_lfm3(Buf *sys, const Json *tools) {
    Buf lines = { 0 };
    for (const Json *t = tools->child; t; t = t->next) {
        const Json *fn = json_get(t, "function");
        if (!fn) fn = t;
        if (fn->t != J_OBJ) continue;
        if (truthy(json_get(fn, "defer_loading")) || truthy(json_get(t, "defer_loading"))) continue;
        if (lines.n) buf_add(&lines, "\n", 1);
        tool_to_json_lfm3(&lines, fn);
    }
    if (lines.n) {
        if (sys->n) buf_add(sys, "\n\n", 2);
        buf_add(sys, "<tools>\n", -1);
        buf_add(sys, lines.s, lines.n);
        buf_add(sys, "\n</tools>", -1);
    }
    buf_free(&lines);
}

/* render_tool_response macro, for plain results (additional_tools is not supported) */
static void render_tool_response_lfm3(Buf *b, const Json *c) {
    if (c && c->t == J_STR && find_in(c->s, c->len, "<tool_response>")) { buf_add(b, c->s, c->len); return; }
    buf_add(b, "<tool_response>\n", -1);
    if (c && c->t == J_ARR) {
        for (const Json *it = c->child; it; it = it->next) {
            if (it != c->child) buf_add(b, "\n", 1);
            const char *type = it->t == J_OBJ ? json_str(json_get(it, "type"), NULL) : NULL;
            if (it->t == J_STR) buf_add(b, it->s, it->len);
            else if (type && !strcmp(type, "text")) buf_add(b, json_str(json_get(it, "text"), ""), -1);
            else json_write(b, it);
        }
    } else if (c && c->t == J_OBJ) {
        const char *type = json_str(json_get(c, "type"), NULL);
        if (type && !strcmp(type, "text")) buf_add(b, json_str(json_get(c, "text"), ""), -1);
        else json_write(b, c);
    } else if (c && c->t == J_STR) {
        buf_add(b, c->s, c->len);
    } else if (c && c->t != J_NULL) {
        json_write(b, c);
    }
    buf_add(b, "\n</tool_response>", -1);
}

/* One assistant message. keep: its thinking is kept (turns after the last user message, or
 * preserve_thinking); a turn without thinking gets an empty <think></think>. */
static int render_assistant_lfm3(Buf *b, const Json *m, int keep, char *err, int errlen) {
    static const char cfm[] = "CONTINUE_FINAL_MESSAGE_TAG ";
    const int ncfm = (int)sizeof(cfm) - 1;
    const Json *content = json_get(m, "content");
    const Json *th = json_get(m, "thinking");
    if (!th) th = json_get(m, "reasoning");
    if (!th) th = json_get(m, "reasoning_content");
    Buf cb = { 0 };
    if (truthy(content)) render_content(&cb, content);
    if (!cb.s) buf_add(&cb, "", 0);
    buf_add(b, "<|im_start|>assistant\n", -1);
    if (th && keep) {
        buf_add(b, "<think>", -1);
        if (th->t == J_STR) buf_add(b, th->s, th->len);
        else if (truthy(th)) json_write(b, th);
        buf_add(b, "</think>", -1);
    } else if (keep && S.reasons && !(content && content->t == J_STR && find_in(content->s, content->len, "</think>"))) {
        buf_add(b, "<think></think>", -1);
    }
    const char *cs = cb.s;
    int cn = cb.n;
    if (!keep) {
        /* content.split("</think>")[-1] | trim, after an empty <think></think> */
        const char *last = NULL;
        for (const char *p = cs; (p = find_in(p, (int)(cs + cn - p), "</think>")) != NULL; p += 8) last = p;
        if (last) {
            cn -= (int)(last + 8 - cs);
            cs = last + 8;
            trim_view(&cs, &cn);
        }
        if (S.reasons) buf_add(b, "<think></think>", -1);
    }
    const int has_cfm = cn >= ncfm && !memcmp(cs + cn - ncfm, cfm, (size_t)ncfm);
    if (has_cfm) cn -= ncfm;
    buf_add(b, cs, cn);
    const Json *calls = json_get(m, "tool_calls");
    int ok = 1;
    if (calls && truthy(calls)) ok = render_tool_calls_lfm3(b, calls, err, errlen);
    if (has_cfm) buf_add(b, cfm, ncfm);
    buf_add(b, "<|im_end|>\n", -1);
    buf_free(&cb);
    return ok;
}

/* Renders the conversation into tokens; bounds[] receives the token position where each
 * message starts (the last one is the start of the generation prompt). */
static int render_chat(const Json *messages, const Json *tools, int preserve_thinking, int think, Toks *out,
                       int **bounds, int *nbounds, Buf *text, char *err, int errlen) {
    const Json *msg = messages->child;
    Buf sys = { 0 }, seg = { 0 };
    int nb = 0, cap = messages->n + 2;
    *bounds = (int *)malloc(sizeof(int) * (size_t)cap);
    toks_push(out, tok_bos(S.tk));
    if (text) buf_add(text, "<|startoftext|>", -1);
    if (msg && !strcmp(json_str(json_get(msg, "role"), ""), "system")) {
        const Json *c = json_get(msg, "content");
        if (truthy(c)) render_content(&sys, c);
        msg = msg->next;
    }
    if (S.lfm3 && tools && tools->t == J_ARR && tools->n > 0) {
        render_tools_lfm3(&sys, tools);
    } else if (tools && tools->t == J_ARR && tools->n > 0) {
        if (sys.n) buf_add(&sys, "\n", 1);
        buf_add(&sys, "List of tools: [", -1);
        for (const Json *t = tools->child; t; t = t->next) {
            if (t != tools->child) buf_add(&sys, ", ", 2);
            if (t->t == J_STR) buf_add(&sys, t->s, t->len);
            else json_write(&sys, t);
        }
        buf_add(&sys, "]", 1);
    }
    if (sys.n) {
        (*bounds)[nb++] = out->n;
        buf_add(&seg, "<|im_start|>system\n", -1);
        buf_add(&seg, sys.s, sys.n);
        buf_add(&seg, "<|im_end|>\n", -1);
        toks_text(out, seg.s, seg.n);
        if (text) buf_add(text, seg.s, seg.n);
    }
    int idx = 0, last_user = -1;
    for (const Json *m = msg; m; m = m->next, idx++)
        if (!strcmp(json_str(json_get(m, "role"), ""), "user")) last_user = idx;
    idx = 0;
    int ok = 1;
    for (const Json *m = msg; m && ok; m = m->next, idx++) {
        const char *role = json_str(json_get(m, "role"), NULL);
        if (!role) { snprintf(err, (size_t)errlen, "message %d has no role", idx); ok = 0; break; }
        seg.n = 0;
        const Json *content = json_get(m, "content");
        if (S.lfm3 && !strcmp(role, "assistant")) {
            ok = render_assistant_lfm3(&seg, m, preserve_thinking || idx > last_user, err, errlen);
        } else if (S.lfm3 && !strcmp(role, "tool")) {
            buf_add(&seg, "<|im_start|>tool\n", -1);
            render_tool_response_lfm3(&seg, content);
            buf_add(&seg, "<|im_end|>\n", -1);
        } else if (!strcmp(role, "assistant")) {
            int keep = preserve_thinking || idx > last_user;
            const Json *th = json_get(m, "thinking");
            if (!th || th->t != J_STR) th = json_get(m, "reasoning");
            if (!th || th->t != J_STR) th = json_get(m, "reasoning_content");
            Buf cb = { 0 };
            if (truthy(content)) render_content(&cb, content);
            ok = render_assistant(&seg, th && th->t == J_STR ? th->s : NULL, th && th->t == J_STR ? th->len : 0,
                                  cb.s ? cb.s : "", cb.n, keep, json_get(m, "tool_calls"), err, errlen);
            buf_free(&cb);
        } else {
            buf_add(&seg, "<|im_start|>", -1);
            buf_add(&seg, role, -1);
            buf_add(&seg, "\n", 1);
            if (truthy(content)) render_content(&seg, content);
            buf_add(&seg, "<|im_end|>\n", -1);
        }
        if (nb == cap) { cap *= 2; *bounds = (int *)realloc(*bounds, sizeof(int) * (size_t)cap); }
        (*bounds)[nb++] = out->n;
        if (ok) toks_text(out, seg.s, seg.n);
        if (ok && text) buf_add(text, seg.s, seg.n);
    }
    if (nb == cap) { cap++; *bounds = (int *)realloc(*bounds, sizeof(int) * (size_t)cap); }
    (*bounds)[nb++] = out->n;
    /* after audio, or with the reasoning off for a template without reasoning blocks, the plain
     * generation prompt of LFM3-Audio's template (after audio the model may open <think> itself) */
    const char *gp = t_voice ? "<|im_start|>assistant\n<|mixed_start|>"
                     : t_collect && S.n_clips ? "<|im_start|>assistant\n"
                     : think ? "<|im_start|>assistant\n<think>"
                     : S.reasons ? think_off_prompt(S.lfm3) : "<|im_start|>assistant\n";
    toks_text(out, gp, -1);
    if (text) buf_add(text, gp, -1);
    *nbounds = nb;
    buf_free(&sys);
    buf_free(&seg);
    return ok;
}

/* ------------------------------------------------------------------ tool calls emitted by the model */

typedef struct { const char *p, *e; } Cur;

static void cur_ws(Cur *c) { while (c->p < c->e && isspace((unsigned char)*c->p)) c->p++; }

static int cur_ident(Cur *c, const char **s, int *n) {
    cur_ws(c);
    *s = c->p;
    while (c->p < c->e && (isalnum((unsigned char)*c->p) || *c->p == '_' || *c->p == '.' || *c->p == '-' ||
                           (unsigned char)*c->p >= 0x80))
        c->p++;
    *n = (int)(c->p - *s);
    return *n > 0;
}

/* Python string literal -> raw bytes */
static int py_string(Cur *c, Buf *o) {
    char q = *c->p++;
    while (c->p < c->e && *c->p != q) {
        char ch = *c->p++;
        if (ch != '\\' || c->p >= c->e) { buf_add(o, &ch, 1); continue; }
        ch = *c->p++;
        switch (ch) {
        case 'n': buf_add(o, "\n", 1); break;
        case 't': buf_add(o, "\t", 1); break;
        case 'r': buf_add(o, "\r", 1); break;
        case 'u': case 'x': {
            int nd = ch == 'u' ? 4 : 2;
            unsigned cp = 0;
            for (int i = 0; i < nd && c->p < c->e && isxdigit((unsigned char)*c->p); i++, c->p++)
                cp = cp * 16 + (unsigned)(isdigit((unsigned char)*c->p) ? *c->p - '0' : (tolower((unsigned char)*c->p) - 'a' + 10));
            char u[4];
            int k;
            if (cp < 0x80) { u[0] = (char)cp; k = 1; }
            else if (cp < 0x800) { u[0] = (char)(0xC0 | cp >> 6); u[1] = (char)(0x80 | (cp & 63)); k = 2; }
            else { u[0] = (char)(0xE0 | cp >> 12); u[1] = (char)(0x80 | ((cp >> 6) & 63)); u[2] = (char)(0x80 | (cp & 63)); k = 3; }
            buf_add(o, u, k);
            break;
        }
        default: buf_add(o, &ch, 1); /* \\ \' \" and unknown escapes */
        }
    }
    if (c->p >= c->e) return 0;
    c->p++;
    return 1;
}

/* Python literal -> compact JSON */
static int py_value(Cur *c, Buf *o, int depth) {
    cur_ws(c);
    if (c->p >= c->e || depth > 32) return 0;
    char ch = *c->p;
    if (ch == '\'' || ch == '"') {
        Buf s = { 0 };
        int ok = py_string(c, &s);
        if (ok) buf_json_str(o, s.s ? s.s : "", s.n);
        buf_free(&s);
        return ok;
    }
    if (ch == '[' || ch == '(') {
        char close = ch == '[' ? ']' : ')';
        c->p++;
        buf_add(o, "[", 1);
        for (int first = 1;; first = 0) {
            cur_ws(c);
            if (c->p < c->e && *c->p == close) { c->p++; break; }
            if (!first) {
                if (c->p >= c->e || *c->p != ',') return 0;
                c->p++;
                cur_ws(c);
                if (c->p < c->e && *c->p == close) { c->p++; break; }
                buf_add(o, ",", 1);
            }
            if (!py_value(c, o, depth + 1)) return 0;
        }
        buf_add(o, "]", 1);
        return 1;
    }
    if (ch == '{') {
        c->p++;
        buf_add(o, "{", 1);
        for (int first = 1;; first = 0) {
            cur_ws(c);
            if (c->p < c->e && *c->p == '}') { c->p++; break; }
            if (!first) {
                if (c->p >= c->e || *c->p != ',') return 0;
                c->p++;
                cur_ws(c);
                if (c->p < c->e && *c->p == '}') { c->p++; break; }
                buf_add(o, ",", 1);
            }
            Buf k = { 0 };
            if (!py_value(c, &k, depth + 1)) { buf_free(&k); return 0; }
            if (k.s[0] == '"') buf_add(o, k.s, k.n);
            else buf_json_str(o, k.s, k.n);
            buf_free(&k);
            cur_ws(c);
            if (c->p >= c->e || *c->p != ':') return 0;
            c->p++;
            buf_add(o, ":", 1);
            if (!py_value(c, o, depth + 1)) return 0;
        }
        buf_add(o, "}", 1);
        return 1;
    }
    const char *s;
    int n;
    if (ch == '-' || ch == '+' || ch == '.' || isdigit((unsigned char)ch)) {
        s = c->p;
        while (c->p < c->e && strchr("+-.0123456789eE", *c->p)) c->p++;
        char num[64];
        n = (int)(c->p - s);
        if (n <= 0 || n >= 64) return 0;
        memcpy(num, s, (size_t)n);
        num[n] = 0;
        char *end;
        strtod(num, &end);
        if (*end) return 0;
        buf_add(o, num[0] == '+' ? num + 1 : num, -1);
        return 1;
    }
    if (!cur_ident(c, &s, &n)) return 0;
    if ((n == 4 && !memcmp(s, "True", 4)) || (n == 4 && !memcmp(s, "true", 4))) buf_add(o, "true", 4);
    else if ((n == 5 && !memcmp(s, "False", 5)) || (n == 5 && !memcmp(s, "false", 5))) buf_add(o, "false", 5);
    else if ((n == 4 && !memcmp(s, "None", 4)) || (n == 4 && !memcmp(s, "null", 4))) buf_add(o, "null", 4);
    else return 0;
    return 1;
}

static void random_id(char *out, int n) {
    static const char al[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i = 0; i < n; i++) {
        S.rng ^= S.rng << 13; S.rng ^= S.rng >> 7; S.rng ^= S.rng << 17;
        out[i] = al[S.rng % 62];
    }
    out[n] = 0;
}

/* "[f(a='x', b=1), g()]" -> OpenAI tool_calls array elements appended to `out`. */
static int parse_tool_calls(const char *s, int n, Buf *out, int *count) {
    Cur c = { s, s + n };
    Buf tmp = { 0 };
    int k = *count;
    cur_ws(&c);
    int bracket = c.p < c.e && *c.p == '[';
    if (bracket) c.p++;
    for (;;) {
        const char *name;
        int nl;
        if (!cur_ident(&c, &name, &nl)) goto fail;
        cur_ws(&c);
        if (c.p >= c.e || *c.p != '(') goto fail;
        c.p++;
        Buf args = { 0 };
        buf_add(&args, "{", 1);
        for (int first = 1;; first = 0) {
            cur_ws(&c);
            if (c.p < c.e && *c.p == ')') { c.p++; break; }
            if (!first) {
                if (c.p >= c.e || *c.p != ',') { buf_free(&args); goto fail; }
                c.p++;
                buf_add(&args, ",", 1);
            }
            const char *an;
            int al;
            if (!cur_ident(&c, &an, &al)) { buf_free(&args); goto fail; }
            cur_ws(&c);
            if (c.p >= c.e || *c.p != '=') { buf_free(&args); goto fail; }
            c.p++;
            buf_json_str(&args, an, al);
            buf_add(&args, ":", 1);
            if (!py_value(&c, &args, 0)) { buf_free(&args); goto fail; }
        }
        buf_add(&args, "}", 1);
        char id[32];
        memcpy(id, "call_", 5);
        random_id(id + 5, 24);
        if (tmp.n || k) buf_add(&tmp, ",", 1);
        buf_printf(&tmp, "{\"index\":%d,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":", k++, id);
        buf_json_str(&tmp, name, nl);
        buf_add(&tmp, ",\"arguments\":", -1);
        buf_json_str(&tmp, args.s, args.n);
        buf_add(&tmp, "}}", 2);
        buf_free(&args);
        cur_ws(&c);
        if (c.p < c.e && *c.p == ',') { c.p++; continue; }
        break;
    }
    cur_ws(&c);
    if (bracket) {
        if (c.p >= c.e || *c.p != ']') goto fail;
        c.p++;
    }
    cur_ws(&c);
    if (c.p != c.e) goto fail;
    if (tmp.n) buf_add(out, tmp.s, tmp.n);
    buf_free(&tmp);
    *count = k;
    return 1;
fail:
    buf_free(&tmp);
    return 0;
}

/* LFM3: "\n<function=name>\n<parameter=k>\nvalue\n</parameter>\n</function>\n" (the text between
 * <tool_call> and </tool_call>) -> one OpenAI tool_calls element appended to `out`. A value that
 * is valid JSON other than a string (number, true, object...) keeps its type; anything else is
 * a string. */
static int parse_tool_call_xml(const char *s, int n, Buf *out, int *count) {
    const char *e = s + n, *p = find_in(s, n, "<function=");
    if (!p) return 0;
    p += 10;
    const char *ne = memchr(p, '>', (size_t)(e - p));
    if (!ne || ne == p) return 0;
    const char *name = p;
    const int nl = (int)(ne - p);
    p = ne + 1;
    Buf args = { 0 };
    buf_add(&args, "{", 1);
    int nargs = 0;
    for (;;) {
        while (p < e && isspace((unsigned char)*p)) p++;
        if (e - p >= 11 && !memcmp(p, "</function>", 11)) { p += 11; break; }
        if (e - p < 11 || memcmp(p, "<parameter=", 11)) { buf_free(&args); return 0; }
        p += 11;
        const char *ke = memchr(p, '>', (size_t)(e - p));
        if (!ke || ke == p) { buf_free(&args); return 0; }
        const char *key = p;
        const int kl = (int)(ke - p);
        p = ke + 1;
        const char *ve = find_in(p, (int)(e - p), "</parameter>");
        if (!ve) { buf_free(&args); return 0; }
        const char *v = p;
        int vl = (int)(ve - p);
        p = ve + 12;
        /* the template writes "\n" + value + "\n" */
        if (vl > 0 && v[0] == '\n') { v++; vl--; }
        if (vl > 0 && v[vl - 1] == '\n') vl--;
        if (nargs++) buf_add(&args, ",", 1);
        buf_json_str(&args, key, kl);
        buf_add(&args, ":", 1);
        char err[8];
        Json *j = vl > 0 ? json_parse(v, vl, err, sizeof(err)) : NULL;
        if (j && j->t != J_STR) json_write(&args, j);
        else buf_json_str(&args, v, vl);
        json_free(j);
    }
    while (p < e && isspace((unsigned char)*p)) p++;
    if (p != e) { buf_free(&args); return 0; }
    buf_add(&args, "}", 1);
    char id[32];
    memcpy(id, "call_", 5);
    random_id(id + 5, 24);
    const int k = *count;
    if (k) buf_add(out, ",", 1);
    buf_printf(out, "{\"index\":%d,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":", k, id);
    buf_json_str(out, name, nl);
    buf_add(out, ",\"arguments\":", -1);
    buf_json_str(out, args.s, args.n);
    buf_add(out, "}}", 2);
    buf_free(&args);
    *count = k + 1;
    return 1;
}

/* ------------------------------------------------------------------ generation */

int think_parse(const char *s, int *budget) {
    if (!strcmp(s, "off") || !strcmp(s, "none")) *budget = THINK_OFF;
    else if (!strcmp(s, "low") || !strcmp(s, "minimal")) *budget = 512;
    else if (!strcmp(s, "medium")) *budget = 2048;
    else if (!strcmp(s, "high")) *budget = THINK_FULL;
    else if (isdigit((unsigned char)s[0])) *budget = atoi(s) > 0 ? atoi(s) : THINK_OFF;
    else return 0;
    return 1;
}

/* The generation prompt with the reasoning off. LFM3's template writes "<think></think>"
 * (enable_thinking=false), but with it LFM3 often reasons anyway inside the answer; one newline
 * inside the block avoids most of that (measured: 13 of 30 GSM8K answers reasoned, then 3).
 * LFM2.5 has no switch in its template; the plain empty block works best for it. */
const char *think_off_prompt(int lfm3) {
    return lfm3 ? "<|im_start|>assistant\n<think>\n</think>\n" : "<|im_start|>assistant\n<think></think>";
}

int template_reasons(const Model *m) {
    if (strcmp(m->arch, "lfm3")) return 1;   /* LFM2.5: no switch in the template, but it reasons */
    uint64_t tl;
    const char *tpl = gguf_get_str(&m->gguf, "tokenizer.chat_template", &tl);
    return !tpl || find_in(tpl, (int)tl, "enable_thinking") != NULL;
}

/* Closing with a sentence (the budget-forcing one from Qwen3) or with a bare </think> measured the
 * same on MATH; the sentence gives the answer a cleaner start. */
const char *think_close(void) {
    return "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n"
           "</think>\n\n";
}

enum { ST_THINK, ST_CONTENT, ST_TOOL };
enum { FIN_NONE, FIN_STOP, FIN_LENGTH, FIN_TOOLS, FIN_ABORT };

typedef struct {
    Conn *c;
    int chat, stream, include_usage, progress;
    int max_tokens;
    SamplerParams sp;
    char *stops[MAX_STOPS];
    int nstops;
    char id[48];
    long long created;
    /* output */
    int state, skip_ws, finish;
    Buf reasoning, content, toolraw, calls;
    int rsent, csent, ncalls, calls_sent, calls_off;
    int n_prompt, n_reused, n_gen;
    int think_budget, n_think;  /* THINK_FULL / THINK_OFF / tokens; reasoning tokens so far */
    double t_prompt, t_gen;
} Gen;

static const char *finish_name(int f) {
    switch (f) {
    case FIN_LENGTH: return "length";
    case FIN_TOOLS: return "tool_calls";
    default: return "stop";
    }
}

/* Bytes of s[0..n) that end on a complete UTF-8 character. */
static int utf8_complete(const char *s, int n) {
    for (int back = 1; back <= 4 && back <= n; back++) {
        unsigned char ch = (unsigned char)s[n - back];
        if ((ch & 0xC0) == 0x80) continue;
        int need = ch < 0x80 ? 1 : (ch >> 5) == 6 ? 2 : (ch >> 4) == 14 ? 3 : (ch >> 3) == 30 ? 4 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

static void chunk_open(Gen *g, Buf *b) {
    buf_add(b, "data: {\"id\":", -1);
    buf_json_str(b, g->id, -1);
    buf_printf(b, ",\"object\":\"%s\",\"created\":%lld,\"model\":", g->chat ? "chat.completion.chunk" : "text_completion",
               g->created);
    buf_json_str(b, S.cfg.alias, -1);
    buf_add(b, ",\"system_fingerprint\":\"pithos\",\"choices\":[", -1);
}

static void send_delta(Gen *g, const char *field, const char *s, int n) {
    Buf b = { 0 };
    chunk_open(g, &b);
    if (g->chat) {
        buf_printf(&b, "{\"index\":0,\"delta\":{\"%s\":", field);
        buf_json_str(&b, s, n);
        buf_add(&b, "},\"logprobs\":null,\"finish_reason\":null}]}\n\n", -1);
    } else {
        buf_add(&b, "{\"index\":0,\"text\":", -1);
        buf_json_str(&b, s, n);
        buf_add(&b, ",\"logprobs\":null,\"finish_reason\":null}]}\n\n", -1);
    }
    conn_send(g->c, b.s, b.n);
    buf_free(&b);
}

static void timings_json(Gen *g, Buf *b) {
    int pn = g->n_prompt - g->n_reused;
    buf_printf(b,
               "{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.3f,\"prompt_per_second\":%.3f,"
               "\"predicted_n\":%d,\"predicted_ms\":%.3f,\"predicted_per_second\":%.3f}",
               g->n_reused, pn, g->t_prompt * 1e3, pn / (g->t_prompt > 0 ? g->t_prompt : 1e-9), g->n_gen, g->t_gen * 1e3,
               g->n_gen / (g->t_gen > 0 ? g->t_gen : 1e-9));
}

static void usage_json(Gen *g, Buf *b) {
    buf_printf(b, "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
                  "\"prompt_tokens_details\":{\"cached_tokens\":%d}}",
               g->n_prompt, g->n_gen, g->n_prompt + g->n_gen, g->n_reused);
}

/* Longest suffix of s[0..n) that is a proper prefix of a stop string (must be held back). */
static int stop_holdback(Gen *g, const char *s, int n) {
    int hold = 0;
    for (int i = 0; i < g->nstops; i++) {
        int sl = (int)strlen(g->stops[i]);
        for (int k = sl - 1; k > hold; k--)
            if (k <= n && !memcmp(s + n - k, g->stops[i], (size_t)k)) { hold = k; break; }
    }
    return hold;
}

/* Send what is ready. final: no more text will come. */
static void flush_text(Gen *g, int final) {
    int r = final ? g->reasoning.n : utf8_complete(g->reasoning.s, g->reasoning.n);
    if (r > g->rsent) {
        if (g->stream) send_delta(g, "reasoning_content", g->reasoning.s + g->rsent, r - g->rsent);
        g->rsent = r;
    }
    int cn = g->content.n;
    if (!final) cn = utf8_complete(g->content.s, cn - stop_holdback(g, g->content.s, cn));
    if (cn > g->csent) {
        if (g->stream) send_delta(g, g->chat ? "content" : "text", g->content.s + g->csent, cn - g->csent);
        g->csent = cn;
    }
    if (g->ncalls > g->calls_sent) {
        if (g->stream) {
            Buf b = { 0 };
            chunk_open(g, &b);
            buf_add(&b, "{\"index\":0,\"delta\":{\"tool_calls\":[", -1);
            /* calls completed since the last flush (the buffer is a comma-separated list) */
            int off = g->calls_off;
            if (off < g->calls.n && g->calls.s[off] == ',') off++;
            buf_add(&b, g->calls.s + off, g->calls.n - off);
            buf_add(&b, "]},\"logprobs\":null,\"finish_reason\":null}]}\n\n", -1);
            conn_send(g->c, b.s, b.n);
            buf_free(&b);
        }
        g->calls_sent = g->ncalls;
        g->calls_off = g->calls.n;
    }
}

/* Appends a token's text to the right channel. Returns 1 when generation must stop. */
static int on_token(Gen *g, int tok) {
    if (g->chat && tok == S.t_think_start && g->state == ST_CONTENT && !g->content.n && !g->ncalls) {
        g->state = ST_THINK;   /* the model opens its reasoning itself (after audio) */
        return 0;
    }
    if (g->chat && tok == S.t_think_end && g->state == ST_THINK) {
        g->state = ST_CONTENT;
        g->skip_ws = 1;
        return 0;
    }
    if (g->chat && tok == S.t_think_end && g->state == ST_CONTENT) {
        /* The model reasoned inside the answer (reasoning off, or past its budget) and closed it
         * now: what it wrote so far was reasoning, as the template's split("</think>")[-1] reads
         * it. Streaming clients already have that text as content: content_to_reasoning tells them. */
        flush_text(g, 0);
        const int unsent = g->content.n - g->csent; /* held back: it goes out as reasoning */
        buf_add(&g->reasoning, g->content.s ? g->content.s : "", g->content.n);
        g->rsent = g->reasoning.n - unsent;
        g->content.n = 0;
        g->csent = 0;
        g->skip_ws = 1;
        if (g->stream) send_delta(g, "content_to_reasoning", "", 0);
        return 0;
    }
    if (g->chat && tok == S.t_tc_start && g->state != ST_TOOL) {
        g->state = ST_TOOL;
        g->toolraw.n = 0;
        return 0;
    }
    if (g->chat && tok == S.t_tc_end && g->state == ST_TOOL) {
        int calls_before = g->calls.n;
        const char *raw = g->toolraw.s ? g->toolraw.s : "";
        if (!(S.lfm3 ? parse_tool_call_xml(raw, g->toolraw.n, &g->calls, &g->ncalls)
                     : parse_tool_calls(raw, g->toolraw.n, &g->calls, &g->ncalls))) {
            g->calls.n = calls_before;
            buf_add(&g->content, S.lfm3 ? "<tool_call>" : "<|tool_call_start|>", -1); /* not parseable: keep it visible */
            buf_add(&g->content, raw, g->toolraw.n);
            buf_add(&g->content, S.lfm3 ? "</tool_call>" : "<|tool_call_end|>", -1);
        }
        g->state = ST_CONTENT;
        return 0;
    }
    if (tok_is_control(S.tk, tok)) return 0;
    int pl;
    const char *piece = tok_piece(S.tk, tok, &pl);
    if (g->state == ST_THINK) { buf_add(&g->reasoning, piece, pl); return 0; }
    if (g->state == ST_TOOL) { buf_add(&g->toolraw, piece, pl); return 0; }
    if (g->skip_ws) {
        while (pl > 0 && (*piece == '\n' || *piece == ' ' || *piece == '\r' || *piece == '\t')) { piece++; pl--; }
        if (pl == 0) return 0;
        g->skip_ws = 0;
    }
    int before = g->content.n;
    buf_add(&g->content, piece, pl);
    for (int i = 0; i < g->nstops; i++) {
        int sl = (int)strlen(g->stops[i]);
        int from = before - sl + 1;
        if (from < 0) from = 0;
        for (int p = from; p + sl <= g->content.n; p++)
            if (!memcmp(g->content.s + p, g->stops[i], (size_t)sl)) {
                g->content.n = p;
                g->content.s[p] = 0;
                return 1;
            }
    }
    return 0;
}

static void send_progress(Gen *g, int done, int total) {
    Buf b = { 0 };
    chunk_open(g, &b);
    buf_printf(&b, "],\"prompt_progress\":{\"total\":%d,\"cache\":%d,\"processed\":%d,\"time_ms\":%.0f}}\n\n", total,
               g->n_reused, done, g->t_prompt * 1e3);
    conn_send(g->c, b.s, b.n);
    buf_free(&b);
}

/* Process the prompt after the reused prefix, with checkpoints at message boundaries. */
static float *prefill(Gen *g, const Toks *t, const int *bounds, int nb) {
    double t0 = time_sec();
    int cur = pcache_prepare(S.pc, t->v, t->n, 0);
    g->n_reused = cur;
    float *lg = NULL;
    int bi = 0;
    if (g->stream && g->progress) send_progress(g, cur, t->n);
    while (cur < t->n) {
        while (bi < nb && bounds[bi] <= cur) bi++;
        int end = t->n;
        if (bi < nb && bounds[bi] < end) end = bounds[bi];
        if (end - cur > PREFILL_PIECE) end = cur + PREFILL_PIECE;
        lg = pcache_feed(S.pc, t->v + cur, end - cur);
        if (!lg) return NULL;
        cur = end;
        if (bi < nb && cur == bounds[bi]) pcache_checkpoint(S.pc);
        g->t_prompt = time_sec() - t0;
        if (g->stream && g->progress) send_progress(g, cur, t->n);
        if (conn_gone(g->c)) { g->finish = FIN_ABORT; return NULL; }
    }
    g->t_prompt = time_sec() - t0;
    return lg;
}

static void generate(Gen *g, float *lg, const Toks *prompt) {
    Sampler smp;
    sampler_init(&smp, &g->sp, S.m->n_vocab);
    int sp[4];
    sampler_ban(&smp, sp, tok_speech_ids(S.tk, sp));
    if (g->chat && !S.reasons && g->think_budget == THINK_OFF) {   /* no reasoning of its own */
        const int th[2] = { S.t_think_start, S.t_think_end };
        sampler_ban(&smp, th, 2);
    }
    int from = prompt->n > g->sp.repeat_last_n ? prompt->n - g->sp.repeat_last_n : 0;
    for (int i = from; i < prompt->n; i++) sampler_accept(&smp, prompt->v[i]);
    double t0 = time_sec();
    int limit = g->max_tokens > 0 ? g->max_tokens : ctx_n_ctx(S.ctx);
    g->finish = FIN_LENGTH;
    for (int i = 0; i < limit; i++) {
        int tok = sampler_sample(&smp, lg);
        g->n_gen++;
        if (tok == S.t_im_end || tok == S.t_eot || tok == S.t_eos) { g->finish = FIN_STOP; break; }
        if (on_token(g, tok)) { g->finish = FIN_STOP; break; }
        flush_text(g, 0);
        if (g->c->dead || ((i & 7) == 7 && conn_gone(g->c))) { g->finish = FIN_ABORT; break; }
        sampler_accept(&smp, tok);
        if (ctx_pos(S.ctx) + 1 >= ctx_n_ctx(S.ctx)) break;
        lg = pcache_feed(S.pc, &tok, 1);
        if (!lg) break;
        if (g->state == ST_THINK && g->think_budget > 0 && ++g->n_think >= g->think_budget) {
            /* reasoning budget spent: close the reasoning ourselves, the model continues with the answer */
            Toks close = { 0 };
            toks_text(&close, think_close(), -1);
            if (ctx_pos(S.ctx) + close.n + 1 >= ctx_n_ctx(S.ctx)) { free(close.v); break; }
            for (int k = 0; k < close.n; k++) {
                on_token(g, close.v[k]);
                sampler_accept(&smp, close.v[k]);
            }
            flush_text(g, 0);
            g->n_gen += close.n;
            i += close.n;
            lg = pcache_feed(S.pc, close.v, close.n);
            free(close.v);
            if (!lg) break;
        }
    }
    if (g->state == ST_TOOL) {
        /* unterminated tool call block: show it as text */
        buf_add(&g->content, S.lfm3 ? "<tool_call>" : "<|tool_call_start|>", -1);
        buf_add(&g->content, g->toolraw.s ? g->toolraw.s : "", g->toolraw.n);
    }
    if (g->finish == FIN_STOP && g->ncalls) g->finish = FIN_TOOLS;
    if (!g->content.s) buf_add(&g->content, "", 0);
    if (!g->reasoning.s) buf_add(&g->reasoning, "", 0);
    g->t_gen = time_sec() - t0;
    sampler_free(&smp);
}

/* ------------------------------------------------------------------ spoken answers */

static void b64_encode(Buf *b, const unsigned char *d, size_t n) {
    static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char q[4];
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (unsigned)d[i] << 16 | (i + 1 < n ? (unsigned)d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        q[0] = tab[v >> 18 & 63];
        q[1] = tab[v >> 12 & 63];
        q[2] = i + 1 < n ? tab[v >> 6 & 63] : '=';
        q[3] = i + 2 < n ? tab[v & 63] : '=';
        buf_add(b, q, 4);
    }
}

/* n samples of speech as an SSE event: delta.audio.data, base64 of 16-bit 24 kHz mono PCM */
static void send_audio(Gen *g, const float *pcm, int n) {
    if (n <= 0 || !g->stream) return;
    unsigned char *raw = (unsigned char *)malloc(2 * (size_t)n);
    for (int i = 0; i < n; i++) {
        float x = pcm[i] * 32767.f;
        x = x > 32767.f ? 32767.f : x < -32768.f ? -32768.f : x;
        const short v = (short)lrintf(x);
        raw[2 * i] = (unsigned char)v;
        raw[2 * i + 1] = (unsigned char)((unsigned short)v >> 8);
    }
    Buf b = { 0 };
    chunk_open(g, &b);
    buf_add(&b, "{\"index\":0,\"delta\":{\"audio\":{\"data\":\"", -1);
    b64_encode(&b, raw, 2 * (size_t)n);
    buf_add(&b, "\"}},\"logprobs\":null,\"finish_reason\":null}]}\n\n", -1);
    conn_send(g->c, b.s, b.n);
    buf_free(&b);
    free(raw);
}

/* The model's interleaved answer (prompt ending with <|mixed_start|>): 6 text tokens, then 12
 * frames of speech, and so on; <|text_end|> ends the text, the speech goes on to its end code,
 * after which the model writes text again (usually <|im_end|>). As generate() in the reference.
 * The text streams as content (the captions), the speech as audio. */
static void generate_voice(Gen *g, float *lg, const Toks *prompt) {
    enum { TEXT, AUDIO, MIXED };
    Sampler smp;
    sampler_init(&smp, &g->sp, S.m->n_vocab);
    const int ban[2] = { S.t_audio, S.t_audio_out };   /* placeholders, never sampled */
    sampler_ban(&smp, ban, 2);
    int from = prompt->n > g->sp.repeat_last_n ? prompt->n - g->sp.repeat_last_n : 0;
    for (int i = from; i < prompt->n; i++)
        if (prompt->v[i] >= 0) sampler_accept(&smp, prompt->v[i]);
    Speech *sp = S.cfg.speech;
    SpeechParams spp = speech_defaults();
    uint64_t rng = S.rng = S.rng * 6364136223846793005ull + 1442695040888963407ull;
    float *pcm = (float *)malloc(sizeof(float) * 2 * SPEECH_FRAME);
    float *row = (float *)malloc(sizeof(float) * (size_t)S.m->dim);
    ctx_set_embeds(S.ctx, row, 1);
    speech_reset(sp);
    int mode = MIXED, slots = 6, speaking = 0, text_done = 0, frames = 0, in_speech = 0;
    const float *h = ctx_hidden(S.ctx);
    const int limit = g->max_tokens > 0 ? g->max_tokens : ctx_n_ctx(S.ctx);
    const double t0 = time_sec();
    double tt[5] = { 0 };   /* PITHOS_SPEECH_PROFILE: text steps, depth decoder, codec, sending, audio steps of the model */
    g->finish = FIN_LENGTH;
    for (int i = 0; i < limit && lg && h; i++) {
        const int interleaving = mode == MIXED;
        double ta = time_sec();
        if (mode == TEXT || (interleaving && !speaking)) {
            const int tok = sampler_sample(&smp, lg);
            g->n_gen++;
            if (tok == S.t_im_end || tok == S.t_eot || tok == S.t_eos) { g->finish = FIN_STOP; break; }
            if (interleaving && tok == S.t_text_end) {
                text_done = 1;
                speaking = 1;
            } else if (!interleaving && tok == S.t_text_end) {
                g->finish = FIN_STOP;
                break;
            } else if (!interleaving && tok == S.t_audio_start) {
                mode = AUDIO;
            } else if (!interleaving && tok == S.t_mixed_start) {
                mode = MIXED;
                slots = 6;
                speaking = text_done = 0;
            } else {
                on_token(g, tok);
                flush_text(g, 0);
            }
            sampler_accept(&smp, tok);
            if (ctx_pos(S.ctx) + 1 >= ctx_n_ctx(S.ctx)) break;
            lg = pcache_feed(S.pc, &tok, 1);
            tt[0] += time_sec() - ta;
        } else {
            int codes[SPEECH_CODEBOOKS];
            const int eos = speech_frame(sp, h, &spp, &rng, codes);
            double tb = time_sec();
            tt[1] += tb - ta;
            frames++;
            int np;
            if (eos) {   /* the end of this stretch of speech: its tail, and back to text */
                np = speech_flush(sp, pcm);
                in_speech = 0;
                mode = TEXT;
                speaking = 0;
            } else {
                np = speech_decode(sp, codes, pcm);
                in_speech = 1;
            }
            double tc = time_sec();
            tt[2] += tc - tb;
            send_audio(g, pcm, np);
            double td = time_sec();
            tt[3] += td - tc;
            speech_embed(sp, codes, row);
            if (ctx_pos(S.ctx) + 1 >= ctx_n_ctx(S.ctx)) break;
            const int id = -1;
            lg = pcache_feed(S.pc, &id, 1);
            tt[4] += time_sec() - td;
        }
        h = ctx_hidden(S.ctx);
        if (interleaving && !text_done && --slots == 0) {
            speaking = !speaking;
            slots = speaking ? 12 : 6;
        }
        if (g->c->dead || ((i & 7) == 7 && conn_gone(g->c))) { g->finish = FIN_ABORT; break; }
    }
    if (in_speech) send_audio(g, pcm, speech_flush(sp, pcm));   /* stopped mid-speech */
    else speech_reset(sp);
    ctx_set_embeds(S.ctx, NULL, 0);
    if (!g->content.s) buf_add(&g->content, "", 0);
    if (!g->reasoning.s) buf_add(&g->reasoning, "", 0);
    g->t_gen = time_sec() - t0;
    if (S.cfg.verbose) fprintf(stderr, "  spoken answer: %d frames (%.1f s)\n", frames, frames * 0.08);
    if (getenv("PITHOS_SPEECH_PROFILE"))
        fprintf(stderr, "  text steps %.0f ms | per frame: depth %.1f ms, codec %.1f ms, sending %.1f ms, model %.1f ms\n",
                1e3 * tt[0], 1e3 * tt[1] / (frames ? frames : 1), 1e3 * tt[2] / (frames ? frames : 1),
                1e3 * tt[3] / (frames ? frames : 1), 1e3 * tt[4] / (frames ? frames : 1));
    sampler_free(&smp);
    free(pcm);
    free(row);
}

static void send_final(Gen *g) {
    Buf b = { 0 };
    if (g->stream) {
        flush_text(g, 1);
        chunk_open(g, &b);
        if (g->chat) buf_printf(&b, "{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"%s\"}]", finish_name(g->finish));
        else buf_printf(&b, "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":\"%s\"}]", finish_name(g->finish));
        buf_add(&b, ",\"timings\":", -1);
        timings_json(g, &b);
        buf_add(&b, "}\n\n", -1);
        if (g->include_usage) {
            chunk_open(g, &b);
            buf_add(&b, "],\"usage\":", -1);
            usage_json(g, &b);
            buf_add(&b, ",\"timings\":", -1);
            timings_json(g, &b);
            buf_add(&b, "}\n\n", -1);
        }
        buf_add(&b, "data: [DONE]\n\n", -1);
        conn_send(g->c, b.s, b.n);
        buf_free(&b);
        return;
    }
    buf_add(&b, "{\"id\":", -1);
    buf_json_str(&b, g->id, -1);
    buf_printf(&b, ",\"object\":\"%s\",\"created\":%lld,\"model\":", g->chat ? "chat.completion" : "text_completion", g->created);
    buf_json_str(&b, S.cfg.alias, -1);
    buf_add(&b, ",\"system_fingerprint\":\"pithos\",\"choices\":[{\"index\":0,", -1);
    if (g->chat) {
        buf_add(&b, "\"message\":{\"role\":\"assistant\",\"content\":", -1);
        if (g->content.n || !g->ncalls) buf_json_str(&b, g->content.s, g->content.n);
        else buf_add(&b, "null", 4);
        if (g->reasoning.n) {
            buf_add(&b, ",\"reasoning_content\":", -1);
            buf_json_str(&b, g->reasoning.s, g->reasoning.n);
        }
        if (g->ncalls) {
            buf_add(&b, ",\"tool_calls\":[", -1);
            buf_add(&b, g->calls.s, g->calls.n);
            buf_add(&b, "]", 1);
        }
        buf_add(&b, "}", 1);
    } else {
        buf_add(&b, "\"text\":", -1);
        buf_json_str(&b, g->content.s, g->content.n);
    }
    buf_printf(&b, ",\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":", finish_name(g->finish));
    usage_json(g, &b);
    buf_add(&b, ",\"timings\":", -1);
    timings_json(g, &b);
    buf_add(&b, "}", 1);
    send_json(g->c, 200, &b);
    buf_free(&b);
}

static int parse_params(Gen *g, const Json *body, char *err, int errlen) {
    g->sp = S.cfg.sp;
    const Json *v;
    if ((v = json_get(body, "temperature")) && v->t == J_NUM) g->sp.temp = (float)v->num;
    if ((v = json_get(body, "top_p")) && v->t == J_NUM) g->sp.top_p = (float)v->num;
    if ((v = json_get(body, "top_k")) && v->t == J_NUM) g->sp.top_k = (int)v->num;
    if ((v = json_get(body, "min_p")) && v->t == J_NUM) g->sp.min_p = (float)v->num;
    if ((v = json_get(body, "repeat_penalty")) && v->t == J_NUM) g->sp.repeat_penalty = (float)v->num;
    if ((v = json_get(body, "repeat_last_n")) && v->t == J_NUM) g->sp.repeat_last_n = (int)v->num;
    if ((v = json_get(body, "seed")) && v->t == J_NUM) g->sp.seed = (uint64_t)v->num;
    if (g->sp.repeat_last_n < 0) g->sp.repeat_last_n = 64;
    g->max_tokens = S.cfg.n_predict;
    if ((v = json_get(body, "max_completion_tokens")) && v->t == J_NUM) g->max_tokens = (int)v->num;
    if ((v = json_get(body, "max_tokens")) && v->t == J_NUM) g->max_tokens = (int)v->num;
    if ((v = json_get(body, "n_predict")) && v->t == J_NUM) g->max_tokens = (int)v->num;
    g->stream = json_bool(json_get(body, "stream"), 0);
    g->include_usage = json_bool(json_get(json_get(body, "stream_options"), "include_usage"), 0);
    g->progress = json_bool(json_get(body, "return_progress"), 0);
    v = json_get(body, "stop");
    if (v && v->t == J_STR && v->len) g->stops[g->nstops++] = v->s;
    else if (v && v->t == J_ARR)
        for (const Json *s = v->child; s && g->nstops < MAX_STOPS; s = s->next)
            if (s->t == J_STR && s->len) g->stops[g->nstops++] = s->s;
    (void)err; (void)errlen;
    return 1;
}

/* Reasoning for this request: "thinking_budget" (tokens, 0 = off), then
 * chat_template_kwargs.enable_thinking=false, then "reasoning_effort" or "reasoning":{"effort"},
 * else the server default. */
static int request_think(const Json *body) {
    const Json *v = json_get(body, "thinking_budget");
    if (v && v->t == J_NUM) return v->num > 0 ? (int)v->num : THINK_OFF;
    const Json *kw = json_get(json_get(body, "chat_template_kwargs"), "enable_thinking");
    if (kw && !json_bool(kw, 1)) return THINK_OFF;
    const char *effort = json_str(json_get(body, "reasoning_effort"), NULL);
    if (!effort) effort = json_str(json_get(json_get(body, "reasoning"), "effort"), NULL);
    int budget;
    if (effort && think_parse(effort, &budget)) return budget;
    return S.cfg.think_budget;
}

static void gen_free(Gen *g) {
    buf_free(&g->reasoning);
    buf_free(&g->content);
    buf_free(&g->toolraw);
    buf_free(&g->calls);
}

static void log_request(const char *what, Gen *g) {
    if (g->c->sink) return; /* the in-process client shows its own statistics */
    int pn = g->n_prompt - g->n_reused;
    fprintf(stderr, "[%s] %s | prompt %d tok (%d cached, %d processed, %.1f tok/s) | answer %d tok, %.2f tok/s | %s\n",
            g->c->peer, what, g->n_prompt, g->n_reused, pn, pn / (g->t_prompt > 0 ? g->t_prompt : 1e-9), g->n_gen,
            g->n_gen / (g->t_gen > 0 ? g->t_gen : 1e-9), g->finish == FIN_ABORT ? "aborted" : finish_name(g->finish));
}

static void begin_gen(Gen *g, Conn *c, int chat) {
    memset(g, 0, sizeof(*g));
    g->c = c;
    g->chat = chat;
    g->created = (long long)time(NULL);
    char rid[32];
    random_id(rid, 24);
    snprintf(g->id, sizeof(g->id), "%s%s", chat ? "chatcmpl-" : "cmpl-", rid);
}

static void handle_chat(Conn *c, const Json *body) {
    char err[256] = "";
    const Json *messages = json_get(body, "messages");
    if (!messages || messages->t != J_ARR || messages->n == 0) {
        send_error(c, 400, "invalid_request_error", "'messages' must be a non-empty array");
        return;
    }
    Gen g;
    begin_gen(&g, c, 1);
    parse_params(&g, body, err, sizeof(err));
    const Json *kw = json_get(body, "chat_template_kwargs");
    int preserve = json_bool(json_get(kw, "preserve_thinking"), 0);
    g.think_budget = request_think(body);
    if (g.think_budget == THINK_OFF) {
        g.state = ST_CONTENT;
        g.skip_ws = 1; /* the answer usually starts with a newline after the empty block */
    }
    const Json *mods = json_get(body, "modalities");
    int voice = 0;
    for (const Json *mo = mods && mods->t == J_ARR ? mods->child : NULL; mo; mo = mo->next)
        voice |= mo->t == J_STR && !strcmp(mo->s, "audio");
    if (voice && (!S.cfg.speech || !g.stream)) {
        send_error(c, 400, "invalid_request_error", !S.cfg.speech ? "this model does not speak (it needs LFM3-Audio and its mmproj file)"
                                                                  : "spoken answers (modalities: audio) need \"stream\": true");
        gen_free(&g);
        return;
    }
    t_voice = voice;
    const int audio = has_audio_parts(messages);
    if (audio && !S.cfg.audio) {
        send_error(c, 400, "invalid_request_error", "this model takes no audio (it needs LFM3-Audio and its mmproj file)");
        gen_free(&g);
        return;
    }
    if (audio) { /* the clips are decoded while rendering and encoded with the model's threads: one request at a time */
        lock_acquire(S.lock);
        audio_begin();
    }
    Toks t = { 0 };
    int *bounds = NULL, nb = 0;
    int rendered = render_chat(messages, json_get(body, "tools"), preserve, g.think_budget != THINK_OFF, &t, &bounds, &nb,
                               NULL, err, sizeof(err));
    t_voice = 0;
    if (rendered && S.clip_err[0]) { snprintf(err, sizeof(err), "%s", S.clip_err); rendered = 0; }
    if (!rendered || t.n >= ctx_n_ctx(S.ctx)) {
        if (rendered) {
            snprintf(err, sizeof(err), "the conversation (%d tokens) exceeds the context size (%d tokens)", t.n, ctx_n_ctx(S.ctx));
            send_error(c, 400, "exceed_context_size_error", err);
        } else send_error(c, 400, "invalid_request_error", err[0] ? err : "cannot render the conversation");
        if (audio) { audio_end(); lock_release(S.lock); }
        free(t.v); free(bounds); gen_free(&g);
        return;
    }
    g.n_prompt = t.n;
    if (audio || voice) {
        g.state = ST_CONTENT;
        g.skip_ws = 1;
    }
    if (!audio) lock_acquire(S.lock);
    S.busy = 1;
    if (audio && !audio_encode(err, sizeof(err))) {
        send_error(c, 400, "invalid_request_error", err);
        audio_end();
        S.busy = 0;
        lock_release(S.lock);
        free(t.v); free(bounds); gen_free(&g);
        return;
    }
    if (audio || voice || !json_bool(json_get(body, "cache_prompt"), 1)) pcache_clear(S.pc);
    if (voice) ctx_need_hidden(S.ctx, 1);
    if (g.stream) send_head(c, 200, "text/event-stream; charset=utf-8", -1);
    float *lg = prefill(&g, &t, bounds, nb);
    if (lg) {
        if (g.stream) {
            Buf b = { 0 };
            chunk_open(&g, &b);
            buf_add(&b, "{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null},\"logprobs\":null,\"finish_reason\":null}]}\n\n", -1);
            conn_send(c, b.s, b.n);
            buf_free(&b);
        }
        if (voice) generate_voice(&g, lg, &t);
        else generate(&g, lg, &t);
        send_final(&g);
    } else if (g.finish != FIN_ABORT) {
        if (g.stream) conn_send(c, "data: {\"error\":{\"code\":500,\"message\":\"context full\"}}\n\n", -1);
        else send_error(c, 500, "server_error", "context full");
    }
    conn_close(c);
    log_request("chat", &g);

    if (voice) ctx_need_hidden(S.ctx, 0);
    if (audio || voice) {   /* the state holds this request's audio: nothing to keep */
        pcache_clear(S.pc);
        if (audio) audio_end();
    } else if (lg) {
        /* rewrite the answer as the template renders past turns, then persist the session */
        double t0 = time_sec();
        Buf a = { 0 }, mj = { 0 };
        buf_add(&mj, "{\"role\":\"assistant\",\"content\":", -1);
        buf_json_str(&mj, g.content.s ? g.content.s : "", g.content.n);
        buf_add(&mj, ",\"reasoning_content\":", -1);
        buf_json_str(&mj, g.reasoning.s ? g.reasoning.s : "", g.reasoning.n);
        if (g.ncalls) {
            buf_add(&mj, ",\"tool_calls\":[", -1);
            buf_add(&mj, g.calls.s, g.calls.n);
            buf_add(&mj, "]", 1);
        }
        buf_add(&mj, "}", 1);
        /* the answer as the next request will send it back */
        Json *msg = json_parse(mj.s, mj.n, NULL, 0);
        int ok = msg != NULL;
        if (ok && S.lfm3) ok = render_assistant_lfm3(&a, msg, preserve, err, sizeof(err));
        else if (ok) ok = render_assistant(&a, g.reasoning.s, g.reasoning.n, g.content.s, g.content.n, preserve,
                                           json_get(msg, "tool_calls"), err, sizeof(err));
        if (ok) {
            int gp = bounds[nb - 1];
            t.n = gp;
            toks_text(&t, a.s, a.n);
            int r = pcache_prepare(S.pc, t.v, t.n, 1);
            if (r < t.n) pcache_feed(S.pc, t.v + r, t.n - r);
            pcache_checkpoint(S.pc);
            pcache_save(S.pc);
            if (S.cfg.verbose)
                fprintf(stderr, "  context realigned: %d tokens reprocessed in %.2f s, %d sessions on disk\n", t.n - r,
                        time_sec() - t0, pcache_n_files(S.pc));
        }
        json_free(msg);
        buf_free(&mj);
        buf_free(&a);
    }
    S.busy = 0;
    lock_release(S.lock);
    gen_free(&g);
    free(t.v);
    free(bounds);
}

static void handle_completion(Conn *c, const Json *body) {
    char err[256] = "";
    const char *prompt = json_str(json_get(body, "prompt"), NULL);
    if (!prompt) {
        const Json *p = json_get(body, "prompt");
        if (p && p->t == J_ARR && p->child && p->child->t == J_STR) prompt = p->child->s;
    }
    if (!prompt) { send_error(c, 400, "invalid_request_error", "'prompt' must be a string"); return; }
    Gen g;
    begin_gen(&g, c, 0);
    g.state = ST_CONTENT;
    parse_params(&g, body, err, sizeof(err));
    Toks t = { 0 };
    toks_text(&t, prompt, -1);
    if (t.n == 0) toks_push(&t, tok_bos(S.tk));
    g.n_prompt = t.n;
    if (t.n >= ctx_n_ctx(S.ctx)) {
        send_error(c, 400, "exceed_context_size_error", "the prompt exceeds the context size");
        free(t.v);
        return;
    }
    lock_acquire(S.lock);
    S.busy = 1;
    if (!json_bool(json_get(body, "cache_prompt"), 1)) pcache_clear(S.pc);
    if (g.stream) send_head(c, 200, "text/event-stream; charset=utf-8", -1);
    int end = t.n; /* checkpoint after the prompt */
    float *lg = prefill(&g, &t, &end, 1);
    if (lg) {
        generate(&g, lg, &t);
        send_final(&g);
    } else if (g.finish != FIN_ABORT) {
        send_error(c, 500, "server_error", "context full");
    }
    conn_close(c);
    log_request("completion", &g);
    S.busy = 0;
    lock_release(S.lock);
    gen_free(&g);
    free(t.v);
}

/* ------------------------------------------------------------------ other endpoints */

static void handle_models(Conn *c) {
    Buf b = { 0 };
    buf_add(&b, "{\"object\":\"list\",\"data\":[{\"id\":", -1);
    buf_json_str(&b, S.cfg.alias, -1);
    buf_printf(&b, ",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"pithos\",\"meta\":{\"n_ctx\":%d,\"n_vocab\":%d,"
                   "\"n_ctx_train\":%d}}]}",
               (long long)time(NULL), ctx_n_ctx(S.ctx), S.m->n_vocab, S.m->n_ctx_train);
    send_json(c, 200, &b);
    buf_free(&b);
}

static void handle_props(Conn *c) {
    Buf b = { 0 };
    const SamplerParams *sp = &S.cfg.sp;
    buf_add(&b, "{\"model_alias\":", -1);
    buf_json_str(&b, S.cfg.alias, -1);
    buf_printf(&b, ",\"build_info\":\"pithos\",\"total_slots\":1,\"is_busy\":%s,\"n_ctx\":%d,\"cached_sessions\":%d,"
                   "\"agent_tools\":%d,",
               S.busy ? "true" : "false", ctx_n_ctx(S.ctx), pcache_n_files(S.pc), S.cfg.tools ? tools_count(S.cfg.tools) : 0);
    buf_printf(&b, "\"modalities\":{\"vision\":false,\"audio\":%s,\"speech\":%s},", S.cfg.audio ? "true" : "false",
               S.cfg.speech ? "true" : "false");
    buf_add(&b, "\"system_info\":", -1);
    buf_json_str(&b, S.cfg.info ? S.cfg.info : "", -1);
    buf_printf(&b, ",\"default_generation_settings\":{\"n_ctx\":%d,\"params\":{\"temperature\":%g,\"top_k\":%d,"
                   "\"top_p\":%g,\"min_p\":%g,\"repeat_penalty\":%g,\"repeat_last_n\":%d,\"seed\":%llu,\"n_predict\":%d}}",
               ctx_n_ctx(S.ctx), sp->temp, sp->top_k, sp->top_p, sp->min_p, sp->repeat_penalty, sp->repeat_last_n,
               (unsigned long long)sp->seed, S.cfg.n_predict);
    /* the reasoning levels this model offers, and the one requests get without reasoning_effort */
    const char *def = "custom";
    int budget;
    static const char *const levels[] = { "high", "medium", "low", "off" };
    for (int i = 0; i < 4; i++)
        if (think_parse(levels[i], &budget) && budget == S.cfg.think_budget) { def = levels[i]; break; }
    buf_printf(&b, ",\"reasoning\":{\"default\":\"%s\",\"levels\":[%s\"high\",\"medium\",\"low\",\"off\"]}", def,
               strcmp(def, "custom") ? "" : "\"custom\",");
    uint64_t tl;
    const char *tpl = gguf_get_str(&S.m->gguf, "tokenizer.chat_template", &tl);
    if (tpl) {
        buf_add(&b, ",\"chat_template\":", -1);
        buf_json_str(&b, tpl, (int)tl);
    }
    buf_add(&b, "}", 1);
    send_json(c, 200, &b);
    buf_free(&b);
}

static void handle_tokenize(Conn *c, const Json *body) {
    const char *text = json_str(json_get(body, "content"), NULL);
    if (!text) { send_error(c, 400, "invalid_request_error", "'content' must be a string"); return; }
    Toks t = { 0 };
    toks_text(&t, text, -1);
    Buf b = { 0 };
    buf_add(&b, "{\"tokens\":[", -1);
    for (int i = 0; i < t.n; i++) buf_printf(&b, i ? ",%d" : "%d", t.v[i]);
    buf_add(&b, "]}", 2);
    send_json(c, 200, &b);
    buf_free(&b);
    free(t.v);
}

/* The rendered prompt text and its tokens (for checking the template). */
static void handle_apply_template(Conn *c, const Json *body) {
    char err[256] = "";
    const Json *messages = json_get(body, "messages");
    if (!messages || messages->t != J_ARR) { send_error(c, 400, "invalid_request_error", "'messages' must be an array"); return; }
    Toks t = { 0 };
    Buf text = { 0 };
    int *bounds = NULL, nb = 0;
    int preserve = json_bool(json_get(json_get(body, "chat_template_kwargs"), "preserve_thinking"), 0);
    if (!render_chat(messages, json_get(body, "tools"), preserve, request_think(body) != THINK_OFF, &t, &bounds, &nb,
                     &text, err, sizeof(err))) {
        send_error(c, 400, "invalid_request_error", err);
    } else {
        Buf b = { 0 };
        buf_add(&b, "{\"prompt\":", -1);
        buf_json_str(&b, text.s, text.n);
        buf_add(&b, ",\"tokens\":[", -1);
        for (int i = 0; i < t.n; i++) buf_printf(&b, i ? ",%d" : "%d", t.v[i]);
        buf_add(&b, "]}", 2);
        send_json(c, 200, &b);
        buf_free(&b);
    }
    buf_free(&text);
    free(t.v);
    free(bounds);
}

static void handle_ui(Conn *c) {
    if (S.cfg.ui_path) {
        FILE *f = file_open(S.cfg.ui_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *s = (char *)malloc((size_t)n + 1);
            n = (long)fread(s, 1, (size_t)n, f);
            fclose(f);
            send_body(c, 200, "text/html; charset=utf-8", s, (int)n);
            free(s);
            return;
        }
    }
    send_body(c, 200, "text/html; charset=utf-8", (const char *)WEBUI_HTML, (int)sizeof(WEBUI_HTML) - 1);
}

/* ------------------------------------------------------------------ agent tools */

static void handle_tools_list(Conn *c) {
    Buf b = { 0 };
    if (S.cfg.tools) tools_list_json(S.cfg.tools, &b);
    else buf_add(&b, "{\"tools\":[],\"servers\":[],\"workspace\":\"\",\"web\":false}", -1);
    send_json(c, 200, &b);
    buf_free(&b);
}

static void handle_tools_call(Conn *c, const Json *body) {
    const char *name = json_str(json_get(body, "name"), NULL);
    if (!name) { send_error(c, 400, "invalid_request_error", "'name' is required"); return; }
    if (!S.cfg.tools) { send_error(c, 404, "not_found_error", "tools are disabled on this server"); return; }
    const Json *args = json_get(body, "arguments");
    Json *parsed = NULL;
    if (args && args->t == J_STR) {
        char err[160];
        parsed = json_parse(args->s, args->len, err, sizeof(err));
        if (!parsed && args->len) { send_error(c, 400, "invalid_request_error", "'arguments' is not valid JSON"); return; }
        args = parsed;
    }
    Buf out = { 0 };
    double t0 = time_sec();
    int ok = tools_call(S.cfg.tools, name, args, &out);
    double dt = time_sec() - t0;
    fprintf(stderr, "[%s] tool %s: %s, %d bytes, %.2f s\n", c->peer, name, ok ? "ok" : "error", out.n, dt);
    Buf b = { 0 };
    buf_printf(&b, "{\"ok\":%s,\"ms\":%.0f,\"content\":", ok ? "true" : "false", dt * 1e3);
    buf_json_str(&b, out.s, out.n);
    buf_add(&b, "}", 1);
    send_json(c, 200, &b);
    buf_free(&b);
    buf_free(&out);
    json_free(parsed);
}

static int authorized(const Req *r) {
    if (!S.cfg.api_key || !*S.cfg.api_key) return 1;
    const char *a = r->auth;
    if (!strncmp(a, "Bearer ", 7)) a += 7;
    return !strcmp(a, S.cfg.api_key);
}

static int is_path(const char *p, const char *a, const char *b) { return !strcmp(p, a) || (b && !strcmp(p, b)); }

/* ------------------------------------------------------------------ /v1/audio/transcriptions */

static const char *mem_find(const char *h, size_t hn, const char *n, size_t nn) {
    if (nn == 0 || hn < nn) return NULL;
    for (size_t i = 0; i + nn <= hn; i++)
        if (h[i] == n[0] && !memcmp(h + i, n, nn)) return h + i;
    return NULL;
}

typedef struct { char name[64]; const char *data; size_t len; } Part;

/* the parts of a multipart/form-data body, by their form field name */
static int multipart(const Req *r, Part *parts, int cap) {
    const char *bp = strstr(r->ctype, "boundary=");
    if (!bp || strncmp(r->ctype, "multipart/form-data", 19)) return -1;
    char delim[200];
    bp += 9;
    int bl = 0;
    if (*bp == '"') bp++;
    while (bp[bl] && bp[bl] != '"' && bp[bl] != ';' && bl < 150) bl++;
    const int dl = snprintf(delim, sizeof(delim), "\r\n--%.*s", bl, bp);
    /* the first delimiter may start the body without the CRLF */
    const char *body = r->body;
    const size_t blen = (size_t)r->body_len;
    const char *p = mem_find(body, blen, delim + 2, (size_t)dl - 2);
    int n = 0;
    while (p && n < cap) {
        p += p == body || p[0] == '-' ? (size_t)dl - 2 : (size_t)dl;
        if ((size_t)(p - body) + 2 > blen || !memcmp(p, "--", 2)) break;   /* the closing delimiter */
        const char *h = p + 2, *he = mem_find(h, blen - (size_t)(h - body), "\r\n\r\n", 4);
        if (!he) break;
        const char *d = he + 4, *e = mem_find(d, blen - (size_t)(d - body), delim, (size_t)dl);
        if (!e) break;
        Part *pt = &parts[n++];
        pt->name[0] = 0;
        const char *nm = mem_find(h, (size_t)(he - h), "name=\"", 6);
        if (nm) {
            nm += 6;
            int k = 0;
            while (nm + k < he && nm[k] != '"' && k < (int)sizeof(pt->name) - 1) k++;
            memcpy(pt->name, nm, (size_t)k);
            pt->name[k] = 0;
        }
        pt->data = d;
        pt->len = (size_t)(e - d);
        p = e;
    }
    return n;
}

static const Part *part_get(const Part *parts, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (!strcmp(parts[i].name, name)) return &parts[i];
    return NULL;
}

/* "it" -> "Italian": the language names of the model's prompts (ISO 639-1 codes) */
static const char *language_name(const char *code) {
    static const char *const map[][2] = {
        { "ar", "Arabic" }, { "da", "Danish" }, { "de", "German" }, { "el", "Greek" }, { "en", "English" },
        { "es", "Spanish" }, { "fi", "Finnish" }, { "fr", "French" }, { "he", "Hebrew" }, { "it", "Italian" },
        { "ja", "Japanese" }, { "ko", "Korean" }, { "nl", "Dutch" }, { "pl", "Polish" }, { "pt", "Portuguese" },
        { "sv", "Swedish" }, { "tr", "Turkish" }, { "zh", "Chinese" },
    };
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
        if (!strcmp(code, map[i][0])) return map[i][1];
    return code;
}

/* ------------------------------------------------------------------ /v1/audio/speech */

#define SPEECH_MAX_FRAMES 1500   /* 2 minutes per request (speech_load's max_frames) */

/* OpenAI's text-to-speech: {"input", "voice": "female" / "male" (other names: female),
 * "language": ISO code such as "it" (an extension; default en), "response_format": "wav" or
 * "pcm" (16-bit 24 kHz mono), "temperature"}. The whole audio comes back at once: callers that want
 * it sooner send a sentence at a time. */
static void handle_speech(Conn *c, const Json *body) {
    if (!S.cfg.speech) { send_error(c, 400, "invalid_request_error", "this model does not speak (it needs LFM3-Audio and its mmproj file)"); return; }
    const char *input = json_str(json_get(body, "input"), NULL);
    if (!input || !*input) { send_error(c, 400, "invalid_request_error", "'input' must be a non-empty string"); return; }
    const char *fmt = json_str(json_get(body, "response_format"), "wav");
    if (strcmp(fmt, "wav") && strcmp(fmt, "pcm")) { send_error(c, 400, "invalid_request_error", "response_format must be wav or pcm"); return; }
    const char *voice = json_str(json_get(body, "voice"), "female");
    SpeechParams sp = speech_defaults();
    const Json *v = json_get(body, "temperature");
    if (v && v->t == J_NUM) sp.temp = (float)v->num;
    v = json_get(body, "seed");
    uint64_t rng = v && v->t == J_NUM ? (uint64_t)v->num * 0x9E3779B97F4A7C15ull | 1 : 0;
    char *prompt = speech_prompt(input, json_str(json_get(body, "language"), "en"), strcmp(voice, "male") ? "female" : "male");
    Gen g;
    begin_gen(&g, c, 0);
    lock_acquire(S.lock);
    S.busy = 1;
    if (!rng) rng = S.rng = S.rng * 6364136223846793005ull + 1442695040888963407ull;
    pcache_clear(S.pc);   /* the frames' rows are not tokens: nothing of this state is reused */
    Toks t = { 0 };
    toks_text(&t, prompt, -1);
    free(prompt);
    const int max_frames = ctx_n_ctx(S.ctx) - t.n - 1 < SPEECH_MAX_FRAMES ? ctx_n_ctx(S.ctx) - t.n - 1 : SPEECH_MAX_FRAMES;
    float *pcm = (float *)malloc(sizeof(float) * ((size_t)(max_frames > 0 ? max_frames : 0) + 1) * SPEECH_FRAME);
    float *row = (float *)malloc(sizeof(float) * (size_t)S.m->dim);
    int n = 0, frames = 0;
    ctx_need_hidden(S.ctx, 1);
    const float *h = max_frames > 0 && prefill(&g, &t, NULL, 0) ? ctx_hidden(S.ctx) : NULL;
    ctx_set_embeds(S.ctx, row, 1);
    const double t0 = time_sec();
    const int id = -1;
    while (h && frames < max_frames) {
        int codes[SPEECH_CODEBOOKS];
        if (speech_frame(S.cfg.speech, h, &sp, &rng, codes)) break;
        n += speech_decode(S.cfg.speech, codes, pcm + n);
        frames++;
        if ((frames & 15) == 0 && conn_gone(c)) { g.finish = FIN_ABORT; break; }
        speech_embed(S.cfg.speech, codes, row);
        h = pcache_feed(S.pc, &id, 1) ? ctx_hidden(S.ctx) : NULL;
    }
    n += speech_flush(S.cfg.speech, pcm + n);
    g.t_gen = time_sec() - t0;
    g.n_prompt = t.n;
    g.n_gen = frames;
    ctx_set_embeds(S.ctx, NULL, 0);
    ctx_need_hidden(S.ctx, 0);
    pcache_clear(S.pc);
    if (g.finish != FIN_ABORT) {
        unsigned char *wav = (unsigned char *)malloc(44 + 2 * (size_t)n);
        const size_t len = speech_wav_bytes(pcm, n, SPEECH_RATE, wav);
        if (!strcmp(fmt, "pcm")) send_body(c, 200, "audio/pcm", (const char *)wav + 44, (int)(len - 44));
        else send_body(c, 200, "audio/wav", (const char *)wav, (int)len);
        free(wav);
        fprintf(stderr, "[speech] %.2f s of audio (%d frames) in %.2f s, %.2fx real time\n", n / (double)SPEECH_RATE,
                frames, g.t_gen, n / (double)SPEECH_RATE / (g.t_gen > 0 ? g.t_gen : 1e-9));
    }
    conn_close(c);
    S.busy = 0;
    lock_release(S.lock);
    free(pcm);
    free(row);
    free(t.v);
    gen_free(&g);
}

/* OpenAI's speech-to-text: form fields file (WAV), language, response_format (json, text,
 * verbose_json), temperature; model and prompt are accepted and ignored */
static void handle_transcription(Conn *c, const Req *r) {
    if (!S.cfg.audio) { send_error(c, 400, "invalid_request_error", "this model takes no audio (it needs LFM3-Audio and its mmproj file)"); return; }
    Part parts[16];
    const int np = multipart(r, parts, 16);
    if (np < 0) { send_error(c, 400, "invalid_request_error", "expected a multipart/form-data body"); return; }
    const Part *file = part_get(parts, np, "file");
    if (!file) { send_error(c, 400, "invalid_request_error", "missing the 'file' field (a WAV file)"); return; }
    char lang[64] = "", fmt[32] = "json", temp[32] = "";
    const Part *v;
    if ((v = part_get(parts, np, "language")) && v->len < sizeof(lang)) { memcpy(lang, v->data, v->len); lang[v->len] = 0; }
    if ((v = part_get(parts, np, "response_format")) && v->len < sizeof(fmt)) { memcpy(fmt, v->data, v->len); fmt[v->len] = 0; }
    if ((v = part_get(parts, np, "temperature")) && v->len < sizeof(temp)) { memcpy(temp, v->data, v->len); temp[v->len] = 0; }
    if (strcmp(fmt, "json") && strcmp(fmt, "text") && strcmp(fmt, "verbose_json")) {
        send_error(c, 400, "invalid_request_error", "response_format must be json, text or verbose_json");
        return;
    }
    Gen g;
    begin_gen(&g, c, 1);
    g.sp = S.cfg.sp;
    g.sp.temp = temp[0] ? (float)atof(temp) : 0.f;   /* OpenAI's default: 0 */
    g.max_tokens = S.cfg.n_predict;
    g.state = ST_CONTENT;
    g.skip_ws = 1;
    lock_acquire(S.lock);
    S.busy = 1;
    audio_begin();
    add_clip((const unsigned char *)file->data, file->len);
    char err[200] = "";
    Toks t = { 0 };
    float *lg = NULL;
    double secs = S.n_clips ? S.clip[0].n / (double)AUDIO_RATE : 0;
    const int heard = S.n_clips && audio_has_speech(S.clip[0].pcm, S.clip[0].n);
    if (S.clip_err[0]) snprintf(err, sizeof(err), "%s", S.clip_err + (strncmp(S.clip_err, "audio part 1: ", 14) ? 0 : 14));
    else if (!heard) buf_add(&g.content, "", 0);   /* no speech: an empty transcription, not made-up words */
    else {
        char sys[200];
        if (lang[0]) snprintf(sys, sizeof(sys), "Perform ASR in %s.", language_name(lang));
        else snprintf(sys, sizeof(sys), "Perform ASR.");
        Buf p = { 0 };
        buf_add(&p, "<|startoftext|><|im_start|>system\n", -1);
        buf_add(&p, sys, -1);
        buf_add(&p, "<|im_end|>\n<|im_start|>user\n<|audio|><|im_end|>\n<|im_start|>assistant\n", -1);
        toks_text(&t, p.s, p.n);
        buf_free(&p);
        if (t.n >= ctx_n_ctx(S.ctx)) snprintf(err, sizeof(err), "the audio (%d tokens) exceeds the context size (%d tokens)", t.n, ctx_n_ctx(S.ctx));
        else if (audio_encode(err, sizeof(err))) {
            pcache_clear(S.pc);
            g.n_prompt = t.n;
            lg = prefill(&g, &t, NULL, 0);
            if (lg) generate(&g, lg, &t);
            else snprintf(err, sizeof(err), "context full");
        }
    }
    if (err[0]) send_error(c, 400, "invalid_request_error", err);
    else if (!strcmp(fmt, "text")) send_body(c, 200, "text/plain; charset=utf-8", g.content.s, g.content.n);
    else {
        Buf b = { 0 };
        if (!strcmp(fmt, "verbose_json")) {
            buf_printf(&b, "{\"task\":\"transcribe\",\"duration\":%.2f,", secs);
            if (lang[0]) {
                buf_add(&b, "\"language\":", -1);
                buf_json_str(&b, language_name(lang), -1);
                buf_add(&b, ",", 1);
            }
        } else buf_add(&b, "{", 1);
        buf_add(&b, "\"text\":", -1);
        buf_json_str(&b, g.content.s, g.content.n);
        buf_printf(&b, ",\"usage\":{\"type\":\"tokens\",\"input_tokens\":%d,\"output_tokens\":%d,\"total_tokens\":%d}}",
                   g.n_prompt, g.n_gen, g.n_prompt + g.n_gen);
        send_json(c, 200, &b);
        buf_free(&b);
    }
    conn_close(c);
    if (!err[0]) log_request("transcription", &g);
    pcache_clear(S.pc);
    audio_end();
    S.busy = 0;
    lock_release(S.lock);
    gen_free(&g);
    free(t.v);
}

static void route(Conn *c, Req *r) {
    if (!strcmp(r->method, "OPTIONS")) { send_head(c, 204, NULL, 0); return; }
    int get = !strcmp(r->method, "GET"), post = !strcmp(r->method, "POST");
    if (get && is_path(r->path, "/", "/index.html")) { handle_ui(c); return; }
    if (get && is_path(r->path, "/talk", "/talk.html")) {   /* the voice conversation page */
        send_body(c, 200, "text/html; charset=utf-8", (const char *)TALK_HTML, (int)sizeof(TALK_HTML) - 1);
        return;
    }
    if (get && is_path(r->path, "/health", "/v1/health")) {
        const char *ok = "{\"status\":\"ok\"}";
        send_body(c, 200, "application/json", ok, (int)strlen(ok));
        return;
    }
    if (get && !strcmp(r->path, "/favicon.ico")) { send_head(c, 204, NULL, 0); return; }
    if (!authorized(r)) { send_error(c, 401, "authentication_error", "invalid API key"); return; }
    if (get && is_path(r->path, "/v1/models", "/models")) { handle_models(c); return; }
    if (get && is_path(r->path, "/props", NULL)) { handle_props(c); return; }
    if (get && is_path(r->path, "/v1/tools", "/tools")) { handle_tools_list(c); return; }
    int chat = is_path(r->path, "/v1/chat/completions", "/chat/completions");
    int compl = is_path(r->path, "/v1/completions", "/completions");
    int tokn = is_path(r->path, "/tokenize", NULL);
    int tmpl = is_path(r->path, "/apply-template", NULL);
    int tcall = is_path(r->path, "/v1/tools/call", "/tools/call");
    int trans = is_path(r->path, "/v1/audio/transcriptions", "/audio/transcriptions");
    int speech = is_path(r->path, "/v1/audio/speech", "/audio/speech");
    if (!chat && !compl && !tokn && !tmpl && !tcall && !trans && !speech) { send_error(c, 404, "not_found_error", "unknown endpoint"); return; }
    if (!post) { send_error(c, 405, "invalid_request_error", "use POST"); return; }
    if (trans) { handle_transcription(c, r); return; }
    char err[160];
    Json *body = json_parse(r->body, r->body_len, err, sizeof(err));
    if (!body || body->t != J_OBJ) {
        char msg[220];
        snprintf(msg, sizeof(msg), "invalid JSON body: %s", body ? "expected an object" : err);
        send_error(c, 400, "invalid_request_error", msg);
        json_free(body);
        return;
    }
    if (chat) handle_chat(c, body);
    else if (compl) handle_completion(c, body);
    else if (tmpl) handle_apply_template(c, body);
    else if (tcall) handle_tools_call(c, body);
    else if (speech) handle_speech(c, body);
    else handle_tokenize(c, body);
    json_free(body);
}

static void conn_main(void *arg) {
    Conn *c = (Conn *)arg;
    Req r;
    int rc = read_request(c, &r);
    if (rc == 200) route(c, &r);
    else if (rc) send_error(c, rc, "invalid_request_error", status_text(rc));
    free(r.body);
    conn_close(c);
    free(c);
}

/* ------------------------------------------------------------------ main loop */

void server_attach(const ServerConfig *cfg, Model *m, Context *ctx, Tokenizer *tk, PCache *pc) {
    memset(&S, 0, sizeof(S));
    S.cfg = *cfg;
    S.m = m;
    S.ctx = ctx;
    S.tk = tk;
    S.pc = pc;
    S.lock = lock_create();
    S.t_im_end = tok_find(tk, "<|im_end|>");
    S.t_eot = tok_find(tk, "<|endoftext|>");
    S.t_eos = tok_eos(tk);
    S.t_think_end = tok_find(tk, "</think>");
    S.t_think_start = tok_find(tk, "<think>");
    S.t_audio = tok_find(tk, "<|audio|>");
    S.t_audio_out = tok_find(tk, "<|audio_out|>");
    S.t_text_end = tok_find(tk, "<|text_end|>");
    S.t_mixed_start = tok_find(tk, "<|mixed_start|>");
    S.t_audio_start = tok_find(tk, "<|audio_start|>");
    S.lfm3 = !strcmp(m->arch, "lfm3");
    S.reasons = template_reasons(m);
    S.t_tc_start = tok_find(tk, S.lfm3 ? "<tool_call>" : "<|tool_call_start|>");
    S.t_tc_end = tok_find(tk, S.lfm3 ? "</tool_call>" : "<|tool_call_end|>");
    S.rng = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ull | 1;
}

void server_local_chat(const char *body, int len, void (*sink)(void *u, const char *s, int n), void *u,
                       volatile int *stop) {
    Conn c;
    memset(&c, 0, sizeof(c));
    c.sink = sink;
    c.sink_u = u;
    c.stop = stop;
    snprintf(c.peer, sizeof(c.peer), "local");
    char err[160];
    Json *j = json_parse(body, len, err, sizeof(err));
    if (!j || j->t != J_OBJ) {
        char msg[220];
        snprintf(msg, sizeof(msg), "invalid JSON body: %s", j ? "expected an object" : err);
        send_error(&c, 400, "invalid_request_error", msg);
    } else {
        handle_chat(&c, j);
    }
    json_free(j);
}

int server_run(const ServerConfig *cfg, Model *m, Context *ctx, Tokenizer *tk, PCache *pc) {
    server_attach(cfg, m, ctx, tk, pc);

#if defined(_WIN32)
    WSADATA wd;
    if (WSAStartup(MAKEWORD(2, 2), &wd)) { fprintf(stderr, "error: WSAStartup failed\n"); return 1; }
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    struct addrinfo hints, *ai = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    char port[16];
    snprintf(port, sizeof(port), "%d", cfg->port);
    if (getaddrinfo(cfg->host, port, &hints, &ai) || !ai) {
        fprintf(stderr, "error: invalid address %s:%d\n", cfg->host, cfg->port);
        return 1;
    }
    sock_t ls = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    int one = 1;
#if !defined(_WIN32)
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
#endif
    if (ls == BAD_SOCK || bind(ls, ai->ai_addr, (int)ai->ai_addrlen) || listen(ls, 64)) {
        fprintf(stderr, "error: cannot listen on %s:%d (port in use?)\n", cfg->host, cfg->port);
        freeaddrinfo(ai);
        return 1;
    }
    freeaddrinfo(ai);
    const char *shown = strcmp(cfg->host, "0.0.0.0") ? cfg->host : "127.0.0.1";
    fprintf(stderr, "\nPandora web UI: http://%s:%d\n", shown, cfg->port);
    fprintf(stderr, "OpenAI-compatible API: http://%s:%d/v1  (model \"%s\")\n", shown, cfg->port, cfg->alias);
    fprintf(stderr, "Press Ctrl+C to stop.\n\n");
    if (cfg->open_browser) {
        char cmd[400];
#if defined(_WIN32)
        snprintf(cmd, sizeof(cmd), "start \"\" \"http://%s:%d/\"", shown, cfg->port);
#elif defined(__APPLE__)
        snprintf(cmd, sizeof(cmd), "open 'http://%s:%d/'", shown, cfg->port);
#else
        snprintf(cmd, sizeof(cmd), "xdg-open 'http://%s:%d/' >/dev/null 2>&1 &", shown, cfg->port);
#endif
        if (system(cmd) != 0) fprintf(stderr, "(could not open the browser: visit the address above)\n");
    }
    for (;;) {
        struct sockaddr_storage sa;
        socklen_t sl = sizeof(sa);
        sock_t s = accept(ls, (struct sockaddr *)&sa, &sl);
        if (s == BAD_SOCK) continue;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
        Conn *c = (Conn *)calloc(1, sizeof(Conn));
        c->s = s;
        if (sa.ss_family == AF_INET)
            inet_ntop(AF_INET, &((struct sockaddr_in *)&sa)->sin_addr, c->peer, sizeof(c->peer));
        else if (sa.ss_family == AF_INET6)
            inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&sa)->sin6_addr, c->peer, sizeof(c->peer));
        if (!thread_spawn(conn_main, c)) { sock_close(s); free(c); }
    }
}
