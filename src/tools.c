/* tools.c - see tools.h. */
#include "tools.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "platform.h"
#include "proc.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

#define MAX_RESULT 24000            /* hard cap of a tool result, in bytes */
#define MAX_FILE (4 << 20)          /* largest file read by the file tools */
#define MAX_JOBS 32                 /* background processes kept in the list */
#define JOB_KEEP (1 << 20)          /* output kept per stream of a process (the oldest part is dropped) */
#define MAX_COMMAND 6000            /* longest shell command, in bytes */
#define USER_AGENT "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

/* ------------------------------------------------------------------ registry */

typedef struct Mcp Mcp;
typedef struct Job Job;

typedef struct {
    char name[136];
    char *desc;
    Json *params;
    int confirm;                    /* ask the user before running it (by default) */
    int builtin;                    /* builtin id, or -1 for an MCP tool */
    Mcp *srv;
    char *remote;                   /* tool name on the MCP server */
} Tool;

struct Mcp {
    Tools *owner;
    char name[64];
    char **argv, **env;
    Proc *p;
    PithosLock *mx, *req, *wr;        /* response hand-off, one request at a time, writes */
    PithosThread *reader, *starter;
    volatile int alive, await_id;
    Json *resp;
    int next_id;
    char status[32], error[256];
    int ntools;
};

struct Tools {
    ToolsConfig cfg;
    char workspace[1024];
    PithosLock *lock;
    Tool *t;
    int n, cap;
    Mcp **srv;
    int nsrv;
    int has_curl;
    /* shell */
    Job *jobs[MAX_JOBS];
    int njobs, next_job;
    volatile int interrupt;
    char *env_info;
};

enum { B_SEARCH, B_FETCH, B_CALC, B_TIME, B_LS, B_READ, B_FIND, B_GREP, B_WRITE, B_EDIT,
       B_RUN, B_START, B_READP, B_STOPP, B_LISTP };

#define SHELL_PARAM "\"shell\": {\"type\": \"string\", \"description\": \"powershell or cmd on Windows, bash or sh elsewhere (default: the first)\"}"
#define CWD_PARAM "\"cwd\": {\"type\": \"string\", \"description\": \"folder to run in, relative to the working folder or absolute (default: the working folder)\"}"

static const struct { int id; const char *name, *desc, *params; int confirm; } BUILTINS[] = {
    { B_SEARCH, "web_search",
      "Search the web (DuckDuckGo) and return titles, URLs and snippets of the results. Use it for facts, "
      "recent events, documentation or anything you are not sure about, then open the best pages with fetch_url.",
      "{\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"search terms\"}, "
      "\"max_results\": {\"type\": \"integer\", \"description\": \"number of results (1-10, default 5)\"}}, \"required\": [\"query\"]}", 0 },
    { B_FETCH, "fetch_url",
      "Download a web page (or text/JSON file) and return its readable text. Long pages come in parts: "
      "call again with start set to the offset given at the end of the previous part.",
      "{\"type\": \"object\", \"properties\": {\"url\": {\"type\": \"string\", \"description\": \"http or https URL\"}, "
      "\"start\": {\"type\": \"integer\", \"description\": \"character offset to start from (default 0)\"}, "
      "\"max_chars\": {\"type\": \"integer\", \"description\": \"characters to return (default 6000, max 20000)\"}}, \"required\": [\"url\"]}", 0 },
    { B_CALC, "calculator",
      "Evaluate a math expression exactly, e.g. '2^10 / 3' or 'sqrt(2) * pi'. Supports + - * / ^ %, parentheses, "
      "sqrt, abs, exp, ln, log10, log2, sin, cos, tan, asin, acos, atan, floor, ceil, round, min, max, pi, e.",
      "{\"type\": \"object\", \"properties\": {\"expression\": {\"type\": \"string\"}}, \"required\": [\"expression\"]}", 0 },
    { B_TIME, "get_current_datetime", "Current local date, time, weekday and time zone.",
      "{\"type\": \"object\", \"properties\": {}}", 0 },
    { B_LS, "list_directory",
      "List files and folders in the workspace folder. Paths are relative to the workspace ('.' is its root).",
      "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"folder (default '.')\"}}}", 0 },
    { B_READ, "read_file",
      "Read a text file from the workspace folder, with line numbers. Long files come in parts (start_line).",
      "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"start_line\": {\"type\": \"integer\", "
      "\"description\": \"first line (default 1)\"}, \"max_lines\": {\"type\": \"integer\", \"description\": \"default 200\"}}, "
      "\"required\": [\"path\"]}", 0 },
    { B_FIND, "find_files",
      "Find files in the workspace folder (recursively) whose name matches a pattern with * and ?, e.g. '*.yaml'.",
      "{\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\"}, \"path\": {\"type\": \"string\", "
      "\"description\": \"folder to search (default '.')\"}}, \"required\": [\"pattern\"]}", 0 },
    { B_GREP, "search_text",
      "Search text inside the files of the workspace folder (case-insensitive). Returns file:line: matching line.",
      "{\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}, \"path\": {\"type\": \"string\", "
      "\"description\": \"folder to search (default '.')\"}, \"pattern\": {\"type\": \"string\", \"description\": "
      "\"only files whose name matches, e.g. '*.py' (default all)\"}}, \"required\": [\"query\"]}", 0 },
    { B_WRITE, "write_file",
      "Create or overwrite a text file in the workspace folder (parent folders are created).",
      "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"content\": {\"type\": \"string\"}}, "
      "\"required\": [\"path\", \"content\"]}", 1 },
    { B_EDIT, "edit_file",
      "Replace an exact piece of text in a file of the workspace folder. old_text must match the file exactly "
      "(spaces and indentation included) and appear only once, unless replace_all is true. Read the file first.",
      "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"old_text\": {\"type\": \"string\"}, "
      "\"new_text\": {\"type\": \"string\"}, \"replace_all\": {\"type\": \"boolean\", \"description\": \"replace every "
      "occurrence (default false)\"}}, \"required\": [\"path\", \"old_text\", \"new_text\"]}", 1 },
    { B_RUN, "run_command", NULL, /* description written at start-up: it names the OS and the shell */
      "{\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, " CWD_PARAM ", "
      "\"timeout\": {\"type\": \"integer\", \"description\": \"seconds before the command is stopped (default 120, max 3600)\"}, "
      SHELL_PARAM "}, \"required\": [\"command\"]}", 1 },
    { B_START, "start_process",
      "Start a command in the background and return at once with a process id: for servers, watchers and anything "
      "that runs for a long time or forever. Check it later with read_process, stop it with stop_process.",
      "{\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, " CWD_PARAM ", "
      "\"wait\": {\"type\": \"integer\", \"description\": \"seconds to wait for the first output (default 3)\"}, "
      SHELL_PARAM "}, \"required\": [\"command\"]}", 1 },
    { B_READP, "read_process",
      "Status (running or exit code), elapsed time and the new output of a background process since the last read. "
      "Can wait for it to finish or for a text to appear in its output.",
      "{\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"integer\"}, \"wait\": {\"type\": \"integer\", "
      "\"description\": \"seconds to wait for the process to end (default 0, max 600)\"}, \"until\": {\"type\": \"string\", "
      "\"description\": \"stop waiting as soon as this text appears in the output\"}}, \"required\": [\"id\"]}", 0 },
    { B_STOPP, "stop_process", "Stop a background process (and everything it started) and return its last output.",
      "{\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"integer\"}}, \"required\": [\"id\"]}", 0 },
    { B_LISTP, "list_processes", "List the background processes started with start_process, with status and run time.",
      "{\"type\": \"object\", \"properties\": {}}", 0 },
};
#define N_BUILTINS ((int)(sizeof(BUILTINS) / sizeof(BUILTINS[0])))

static Tool *add_tool(Tools *T) {
    if (T->n == T->cap) {
        T->cap = T->cap ? T->cap * 2 : 32;
        T->t = (Tool *)realloc(T->t, sizeof(Tool) * (size_t)T->cap);
    }
    Tool *t = &T->t[T->n++];
    memset(t, 0, sizeof(*t));
    t->builtin = -1;
    return t;
}

static char *dupstr(const char *s) {
    size_t n = strlen(s);
    char *r = (char *)malloc(n + 1);
    memcpy(r, s, n + 1);
    return r;
}

static long long arg_int(const Json *args, const char *k, long long def) {
    const Json *v = json_get(args, k);
    if (v && v->t == J_NUM) return (long long)v->num;
    if (v && v->t == J_STR && v->len) return atoll(v->s);
    return def;
}

static void cap_result(Buf *out) {
    if (out->n <= MAX_RESULT) return;
    int n = MAX_RESULT;
    while (n > 0 && ((unsigned char)out->s[n] & 0xC0) == 0x80) n--;
    out->n = n;
    out->s[n] = 0;
    buf_add(out, "\n[result truncated]", -1);
}

/* ------------------------------------------------------------------ text helpers */

static int utf8_put(char *o, unsigned cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | cp >> 6); o[1] = (char)(0x80 | (cp & 63)); return 2; }
    if (cp < 0x10000) { o[0] = (char)(0xE0 | cp >> 12); o[1] = (char)(0x80 | ((cp >> 6) & 63)); o[2] = (char)(0x80 | (cp & 63)); return 3; }
    o[0] = (char)(0xF0 | cp >> 18); o[1] = (char)(0x80 | ((cp >> 12) & 63)); o[2] = (char)(0x80 | ((cp >> 6) & 63)); o[3] = (char)(0x80 | (cp & 63));
    return 4;
}

/* Decodes an HTML entity at s (on '&'); returns bytes consumed (0 if not an entity). */
static int html_entity(const char *s, const char *end, char *o, int *olen) {
    static const struct { const char *n; unsigned cp; } E[] = {
        { "amp", '&' }, { "lt", '<' }, { "gt", '>' }, { "quot", '"' }, { "apos", '\'' }, { "nbsp", ' ' },
        { "mdash", 0x2014 }, { "ndash", 0x2013 }, { "hellip", 0x2026 }, { "rsquo", 0x2019 }, { "lsquo", 0x2018 },
        { "ldquo", 0x201C }, { "rdquo", 0x201D }, { "copy", 0xA9 }, { "reg", 0xAE }, { "euro", 0x20AC },
        { "middot", 0xB7 }, { "laquo", 0xAB }, { "raquo", 0xBB }, { "times", 0xD7 }, { "deg", 0xB0 },
        { "agrave", 0xE0 }, { "egrave", 0xE8 }, { "eacute", 0xE9 }, { "igrave", 0xEC }, { "ograve", 0xF2 }, { "ugrave", 0xF9 },
    };
    const char *p = s + 1;
    const char *semi = memchr(p, ';', (size_t)(end - p > 12 ? 12 : end - p));
    if (!semi) return 0;
    unsigned cp = 0;
    if (*p == '#') {
        p++;
        int hex = (*p == 'x' || *p == 'X');
        if (hex) p++;
        if (p == semi) return 0;
        for (; p < semi; p++) {
            int d = isdigit((unsigned char)*p) ? *p - '0' : hex && isxdigit((unsigned char)*p) ? tolower((unsigned char)*p) - 'a' + 10 : -1;
            if (d < 0) return 0;
            cp = cp * (hex ? 16u : 10u) + (unsigned)d;
            if (cp > 0x10FFFF) return 0;
        }
        if (cp == 0xA0) cp = ' ';
    } else {
        size_t n = (size_t)(semi - p);
        int found = 0;
        for (size_t i = 0; i < sizeof(E) / sizeof(E[0]); i++)
            if (strlen(E[i].n) == n && !memcmp(E[i].n, p, n)) { cp = E[i].cp; found = 1; break; }
        if (!found) return 0;
    }
    *olen = utf8_put(o, cp ? cp : '?');
    return (int)(semi + 1 - s);
}

static void add_decoded(Buf *b, const char *s, int n) {
    const char *end = s + n;
    while (s < end) {
        char o[4];
        int ol, k;
        if (*s == '&' && (k = html_entity(s, end, o, &ol)) > 0) { buf_add(b, o, ol); s += k; continue; }
        buf_add(b, s, 1);
        s++;
    }
}

static int ci_prefix(const char *s, const char *end, const char *w) {
    size_t n = strlen(w);
    if ((size_t)(end - s) < n) return 0;
    for (size_t i = 0; i < n; i++)
        if (tolower((unsigned char)s[i]) != w[i]) return 0;
    return 1;
}

/* Readable text from HTML: drops scripts, styles and navigation, keeps block structure. */
static void html_to_text(const char *h, int n, Buf *out, Buf *title) {
    static const char *skip[] = { "script", "style", "noscript", "svg", "template", "iframe", "nav", "footer",
                                  "aside", "form", "select", "button", "head" };
    static const char *blocks[] = { "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
                                    "article", "header", "ul", "ol", "table", "blockquote", "pre", "hr", "dt", "dd",
                                    "main", "figure", "figcaption" };
    const char *s = h, *end = h + n;
    /* title */
    for (const char *p = h; p < end; p++)
        if (*p == '<' && ci_prefix(p + 1, end, "title")) {
            const char *a = memchr(p, '>', (size_t)(end - p));
            if (!a) break;
            const char *b = a + 1;
            while (b < end && !(*b == '<' && ci_prefix(b + 1, end, "/title"))) b++;
            add_decoded(title, a + 1, (int)(b - a - 1));
            break;
        }
    int nl = 2, sp = 0;
    while (s < end) {
        if (*s == '<') {
            if (s + 3 < end && !memcmp(s, "<!--", 4)) {
                const char *e = strstr(s, "-->");
                s = e ? e + 3 : end;
                continue;
            }
            const char *p = s + 1;
            int closing = *p == '/';
            if (closing) p++;
            char tag[16];
            int tl = 0;
            while (p < end && (isalnum((unsigned char)*p)) && tl < 15) tag[tl++] = (char)tolower((unsigned char)*p++);
            tag[tl] = 0;
            /* end of the tag, respecting quoted attribute values */
            const char *q = p;
            char quote = 0;
            while (q < end && (quote || *q != '>')) {
                if (quote) { if (*q == quote) quote = 0; }
                else if (*q == '"' || *q == '\'') quote = *q;
                q++;
            }
            s = q < end ? q + 1 : end;
            if (!tl) continue;
            int skipped = 0;
            for (size_t i = 0; i < sizeof(skip) / sizeof(skip[0]) && !closing; i++)
                if (!strcmp(tag, skip[i]) && !(q > p && q[-1] == '/')) { /* not self-closing */
                    /* jump to the matching close tag */
                    char close[24];
                    snprintf(close, sizeof(close), "</%s", tag);
                    const char *e = s;
                    while (e < end && !(e[0] == '<' && ci_prefix(e, end, close))) e++;
                    const char *gt = e < end ? memchr(e, '>', (size_t)(end - e)) : NULL;
                    s = gt ? gt + 1 : end;
                    skipped = 1;
                    break;
                }
            if (skipped) continue;
            if (!strcmp(tag, "td") || !strcmp(tag, "th")) { /* cells of a row stay on one line */
                if (closing) { buf_add(out, " | ", 3); nl = 0; sp = 0; }
                continue;
            }
            for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++)
                if (!strcmp(tag, blocks[i])) {
                    int want = (tag[0] == 'h' && isdigit((unsigned char)tag[1])) || !strcmp(tag, "p") ? 2 : 1;
                    while (nl < want) { buf_add(out, "\n", 1); nl++; }
                    sp = 0;
                    if (!closing && !strcmp(tag, "li")) { buf_add(out, "- ", 2); nl = 0; }
                    if (!closing && tag[0] == 'h' && isdigit((unsigned char)tag[1]) && tag[2] == 0) {
                        buf_add(out, "## ", 3);
                        nl = 0;
                    }
                    break;
                }
            continue;
        }
        if (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
            sp = 1;
            s++;
            continue;
        }
        if (sp && nl == 0) buf_add(out, " ", 1);
        sp = 0;
        nl = 0;
        char o[4];
        int ol, k;
        if (*s == '&' && (k = html_entity(s, end, o, &ol)) > 0) {
            if (!(ol == 1 && o[0] == ' ')) buf_add(out, o, ol);
            else sp = 1;
            s += k;
            continue;
        }
        buf_add(out, s, 1);
        s++;
    }
}

static void url_encode(Buf *b, const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') buf_add(b, (const char *)&c, 1);
        else if (c == ' ') buf_add(b, "+", 1);
        else buf_printf(b, "%%%02X", c);
    }
}

static void url_decode(Buf *b, const char *s, int n) {
    for (int i = 0; i < n; i++) {
        if (s[i] == '%' && i + 2 < n && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) {
            char hx[3] = { s[i + 1], s[i + 2], 0 };
            char c = (char)strtol(hx, NULL, 16);
            buf_add(b, &c, 1);
            i += 2;
        } else if (s[i] == '+') {
            buf_add(b, " ", 1);
        } else {
            buf_add(b, s + i, 1);
        }
    }
}

/* ------------------------------------------------------------------ web tools (curl) */

static int curl_get(const char *url, char **body, int *len, int max_bytes, char *err, int errlen) {
    const char *argv[] = { "curl", "-s", "-S", "-L", "--max-redirs", "5", "--max-time", "25", "--compressed",
                           "--proto", "=http,https", "--proto-redir", "=http,https", "-A", USER_AGENT,
                           "-H", "Accept-Language: en,it;q=0.8", "--fail-with-body", url, NULL };
    int rc = proc_run(argv, body, len, max_bytes, 30000);
    if (rc == -1) { snprintf(err, (size_t)errlen, "curl is not available on this system"); return 0; }
    if (rc == -2) { snprintf(err, (size_t)errlen, "timeout downloading %s", url); return 0; }
    if (rc != 0) {
        snprintf(err, (size_t)errlen, "download failed (%s, curl exit code %d)",
                 rc == 22 ? "HTTP error" : rc == 6 ? "host not found" : rc == 7 ? "connection refused" : rc == 28 ? "timeout" : "error", rc);
        return 0;
    }
    return 1;
}

static int tool_search(const Json *args, Buf *out) {
    const char *q = json_str(json_get(args, "query"), NULL);
    if (!q || !*q) { buf_add(out, "error: 'query' is required", -1); return 0; }
    int maxr = (int)arg_int(args, "max_results", 5);
    if (maxr < 1) maxr = 1;
    if (maxr > 10) maxr = 10;
    Buf url = { 0 };
    buf_add(&url, "https://html.duckduckgo.com/html/?q=", -1);
    url_encode(&url, q);
    char *html = NULL, err[256];
    int n = 0;
    if (!curl_get(url.s, &html, &n, 2 << 20, err, sizeof(err))) {
        buf_add(out, err, -1);
        buf_free(&url);
        free(html);
        return 0;
    }
    buf_free(&url);
    const char *s = html, *end = html + n;
    int found = 0;
    while (found < maxr && (s = strstr(s, "class=\"result__a\"")) != NULL) {
        const char *tag_end = strchr(s, '>');
        const char *href = strstr(s, "href=\"");
        if (!tag_end || !href || href > tag_end) { s++; continue; }
        href += 6;
        const char *href_end = strchr(href, '"');
        const char *close = strstr(tag_end, "</a>");
        if (!href_end || !close) break;
        Buf link = { 0 }, title = { 0 }, snip = { 0 }, raw = { 0 };
        add_decoded(&raw, href, (int)(href_end - href));
        const char *u = raw.s ? strstr(raw.s, "uddg=") : NULL;
        if (u) {
            u += 5;
            const char *amp = strchr(u, '&');
            url_decode(&link, u, amp ? (int)(amp - u) : (int)strlen(u));
        } else if (raw.s) {
            buf_add(&link, raw.s, raw.n);
        }
        /* title text without inner tags */
        Buf t = { 0 };
        html_to_text(tag_end + 1, (int)(close - tag_end - 1), &t, &raw);
        if (t.s) buf_add(&title, t.s, t.n);
        buf_free(&t);
        const char *next = strstr(close, "class=\"result__a\"");
        const char *sn = strstr(close, "class=\"result__snippet\"");
        if (sn && (!next || sn < next)) {
            const char *a = strchr(sn, '>');
            const char *b = a ? strstr(a, "</a>") : NULL;
            Buf dummy = { 0 };
            if (a && b) html_to_text(a + 1, (int)(b - a - 1), &snip, &dummy);
            buf_free(&dummy);
        }
        int ad = link.s && (strstr(link.s, "duckduckgo.com/y.js") || strstr(link.s, "ad_domain"));
        if (link.s && !ad && strncmp(link.s, "http", 4) == 0) {
            found++;
            buf_printf(out, "%d. %s\n   %s\n", found, title.s ? title.s : "(no title)", link.s);
            if (snip.s) buf_printf(out, "   %s\n", snip.s);
        }
        buf_free(&link); buf_free(&title); buf_free(&snip); buf_free(&raw);
        s = close;
    }
    free(html);
    (void)end;
    if (!found) { buf_add(out, "no results (the search service may have refused the request; try other words)", -1); return 0; }
    return 1;
}

static int tool_fetch(const Json *args, Buf *out) {
    const char *url = json_str(json_get(args, "url"), NULL);
    if (!url || (strncmp(url, "http://", 7) && strncmp(url, "https://", 8))) {
        buf_add(out, "error: 'url' must start with http:// or https://", -1);
        return 0;
    }
    long long start = arg_int(args, "start", 0), maxc = arg_int(args, "max_chars", 6000);
    if (start < 0) start = 0;
    if (maxc < 200) maxc = 200;
    if (maxc > 20000) maxc = 20000;
    char *body = NULL, err[256];
    int n = 0;
    if (!curl_get(url, &body, &n, 8 << 20, err, sizeof(err))) {
        buf_add(out, err, -1);
        if (body && n) { buf_add(out, "\n", 1); buf_add(out, body, n > 500 ? 500 : n); }
        free(body);
        return 0;
    }
    int is_html = 0;
    for (int i = 0; i < n && i < 2048; i++)
        if (body[i] == '<' && (ci_prefix(body + i + 1, body + n, "html") || ci_prefix(body + i + 1, body + n, "!doctype html") ||
                               ci_prefix(body + i + 1, body + n, "head") || ci_prefix(body + i + 1, body + n, "body"))) { is_html = 1; break; }
    Buf text = { 0 }, title = { 0 };
    if (is_html) html_to_text(body, n, &text, &title);
    else {
        for (int i = 0; i < n && i < 4096; i++)
            if (body[i] == 0) { buf_add(out, "error: binary content (not text)", -1); free(body); return 0; }
        buf_add(&text, body, n);
    }
    free(body);
    if (!text.s) buf_add(&text, "", 0);
    int total = text.n;
    if (start > total) start = total;
    int a = (int)start, b = (int)(start + maxc < total ? start + maxc : total);
    while (a > 0 && a < total && ((unsigned char)text.s[a] & 0xC0) == 0x80) a++;
    while (b < total && ((unsigned char)text.s[b] & 0xC0) == 0x80) b++;
    if (title.s) buf_printf(out, "Title: %s\n", title.s);
    buf_printf(out, "URL: %s\n", url);
    if (a > 0 || b < total) buf_printf(out, "Characters %d-%d of %d\n", a, b, total);
    buf_add(out, "\n", 1);
    buf_add(out, text.s + a, b - a);
    if (b < total) buf_printf(out, "\n\n[continues: call fetch_url with start=%d]", b);
    buf_free(&text);
    buf_free(&title);
    return 1;
}

/* ------------------------------------------------------------------ calculator */

typedef struct { const char *p; int err; } Calc;

static double c_expr(Calc *c);

static void c_ws(Calc *c) { while (isspace((unsigned char)*c->p)) c->p++; }

static double c_primary(Calc *c) {
    c_ws(c);
    if (*c->p == '(') {
        c->p++;
        double v = c_expr(c);
        c_ws(c);
        if (*c->p == ')') c->p++; else c->err = 1;
        return v;
    }
    if (*c->p == '-') { c->p++; return -c_primary(c); }
    if (*c->p == '+') { c->p++; return c_primary(c); }
    if (isdigit((unsigned char)*c->p) || *c->p == '.') {
        char *e;
        double v = strtod(c->p, &e);
        if (e == c->p) c->err = 1;
        c->p = e;
        return v;
    }
    if (isalpha((unsigned char)*c->p)) {
        char id[16];
        int n = 0;
        while (isalnum((unsigned char)*c->p) && n < 15) id[n++] = (char)tolower((unsigned char)*c->p++);
        id[n] = 0;
        if (!strcmp(id, "pi")) return 3.14159265358979323846;
        if (!strcmp(id, "e")) return 2.71828182845904523536;
        c_ws(c);
        if (*c->p != '(') { c->err = 1; return 0; }
        c->p++;
        double a = c_expr(c), b = 0;
        int two = 0;
        c_ws(c);
        if (*c->p == ',') { c->p++; b = c_expr(c); two = 1; c_ws(c); }
        if (*c->p == ')') c->p++; else c->err = 1;
        if (two && !strcmp(id, "min")) return a < b ? a : b;
        if (two && !strcmp(id, "max")) return a > b ? a : b;
        if (two && !strcmp(id, "pow")) return pow(a, b);
        if (two && !strcmp(id, "round")) { double m = pow(10, b); return round(a * m) / m; }
        if (two) { c->err = 1; return 0; }
        if (!strcmp(id, "sqrt")) return sqrt(a);
        if (!strcmp(id, "abs")) return fabs(a);
        if (!strcmp(id, "exp")) return exp(a);
        if (!strcmp(id, "ln") || !strcmp(id, "log")) return log(a);
        if (!strcmp(id, "log10")) return log10(a);
        if (!strcmp(id, "log2")) return log2(a);
        if (!strcmp(id, "sin")) return sin(a);
        if (!strcmp(id, "cos")) return cos(a);
        if (!strcmp(id, "tan")) return tan(a);
        if (!strcmp(id, "asin")) return asin(a);
        if (!strcmp(id, "acos")) return acos(a);
        if (!strcmp(id, "atan")) return atan(a);
        if (!strcmp(id, "floor")) return floor(a);
        if (!strcmp(id, "ceil")) return ceil(a);
        if (!strcmp(id, "round")) return round(a);
        c->err = 1;
        return 0;
    }
    c->err = 1;
    return 0;
}

static double c_power(Calc *c) {
    double b = c_primary(c);
    c_ws(c);
    if (*c->p == '^' || (c->p[0] == '*' && c->p[1] == '*')) {
        c->p += *c->p == '^' ? 1 : 2;
        return pow(b, c_power(c)); /* right associative */
    }
    return b;
}

static double c_term(Calc *c) {
    double v = c_power(c);
    for (;;) {
        c_ws(c);
        char op = *c->p;
        if ((op != '*' && op != '/' && op != '%') || (op == '*' && c->p[1] == '*')) return v;
        c->p++;
        double r = c_power(c);
        v = op == '*' ? v * r : op == '/' ? v / r : fmod(v, r);
    }
}

static double c_expr(Calc *c) {
    double v = c_term(c);
    for (;;) {
        c_ws(c);
        char op = *c->p;
        if (op != '+' && op != '-') return v;
        c->p++;
        double r = c_term(c);
        v = op == '+' ? v + r : v - r;
    }
}

static int tool_calc(const Json *args, Buf *out) {
    const char *e = json_str(json_get(args, "expression"), NULL);
    if (!e) { buf_add(out, "error: 'expression' is required", -1); return 0; }
    Calc c = { e, 0 };
    double v = c_expr(&c);
    c_ws(&c);
    if (c.err || *c.p) {
        buf_printf(out, "error: cannot parse the expression near '%.20s'", *c.p ? c.p : "(end)");
        return 0;
    }
    if (v == floor(v) && fabs(v) < 1e15) buf_printf(out, "%.0f", v);
    else buf_printf(out, "%.15g", v);
    return 1;
}

static int tool_time(Buf *out) {
    time_t now = time(NULL);
    struct tm lt, gt;
#if defined(_WIN32)
    localtime_s(&lt, &now);
    gmtime_s(&gt, &now);
#else
    localtime_r(&now, &lt);
    gmtime_r(&now, &gt);
#endif
    char s[64];
    strftime(s, sizeof(s), "%Y-%m-%d %H:%M:%S, %A", &lt);
    gt.tm_isdst = lt.tm_isdst;
    long off = (long)difftime(mktime(&lt), mktime(&gt));
    buf_printf(out, "%s (UTC%c%02ld:%02ld)", s, off < 0 ? '-' : '+', labs(off) / 3600, (labs(off) % 3600) / 60);
    return 1;
}

/* ------------------------------------------------------------------ workspace file tools */

static void absolute_path(const char *p, char *out, int cap);

/* Workspace-relative path -> full path. Rejects '..' and absolute paths outside the workspace
 * (an absolute path inside it is taken as the relative one). */
static int ws_path(Tools *T, const char *rel, char *out, int cap, Buf *err) {
    if (!T->workspace[0]) { buf_add(err, "error: no workspace folder is configured", -1); return 0; }
    char r[1024];
    snprintf(r, sizeof(r), "%s", rel && *rel ? rel : ".");
    for (char *p = r; *p; p++) if (*p == '\\') *p = '/';
    if (r[0] == '/' || strchr(r, ':')) {
        char ws[1100];
        absolute_path(T->workspace, ws, sizeof(ws));
        for (char *p = ws; *p; p++) if (*p == '\\') *p = '/';
        size_t l = strlen(ws);
        while (l > 1 && ws[l - 1] == '/') ws[--l] = 0;
#if defined(_WIN32)
        int inside = !_strnicmp(r, ws, l);
#else
        int inside = !strncmp(r, ws, l);
#endif
        if (!inside || (r[l] && r[l] != '/')) {
            buf_printf(err, "error: %s is outside the working folder %s: use a path relative to it (or run_command)", rel, ws);
            return 0;
        }
        memmove(r, r + l + (r[l] == '/'), strlen(r + l + (r[l] == '/')) + 1);
        if (!r[0]) snprintf(r, sizeof(r), ".");
    }
    for (const char *p = r; *p; ) {
        const char *e = strchr(p, '/');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        if (n == 2 && !strncmp(p, "..", 2)) { buf_add(err, "error: '..' is not allowed: stay inside the workspace", -1); return 0; }
        p += n + (e ? 1 : 0);
        if (!e) break;
    }
    const char *s = r;
    while (s[0] == '.' && s[1] == '/') s += 2;
    int need = (!strcmp(s, ".") || !*s) ? snprintf(out, (size_t)cap, "%s", T->workspace)
                                         : snprintf(out, (size_t)cap, "%s/%s", T->workspace, s);
    if (need >= cap) { buf_add(err, "error: path too long", -1); return 0; }
    return 1;
}

static int glob_match(const char *p, const char *s) {
    for (; *p; p++, s++) {
        if (*p == '*') {
            while (p[1] == '*') p++;
            if (!p[1]) return 1;
            for (; *s; s++)
                if (glob_match(p + 1, s)) return 1;
            return 0;
        }
        if (!*s) return 0;
        if (*p != '?' && tolower((unsigned char)*p) != tolower((unsigned char)*s)) return 0;
    }
    return !*s;
}

typedef struct { char name[256]; int is_dir; uint64_t size; } Ent;
typedef struct { Ent *e; int n, cap; } Ents;

static void ent_cb(const char *name, int is_dir, uint64_t size, void *u) {
    Ents *E = (Ents *)u;
    if (name[0] == '.' || !strcmp(name, "node_modules") || !strcmp(name, "__pycache__")) return;
    if (E->n == E->cap) {
        E->cap = E->cap ? E->cap * 2 : 64;
        E->e = (Ent *)realloc(E->e, sizeof(Ent) * (size_t)E->cap);
    }
    snprintf(E->e[E->n].name, sizeof(E->e[0].name), "%s", name);
    E->e[E->n].is_dir = is_dir;
    E->e[E->n].size = size;
    E->n++;
}

static int ent_cmp(const void *a, const void *b) {
    const Ent *x = (const Ent *)a, *y = (const Ent *)b;
    if (x->is_dir != y->is_dir) return y->is_dir - x->is_dir;
    return strcmp(x->name, y->name);
}

static int tool_ls(Tools *T, const Json *args, Buf *out) {
    char full[1100];
    const char *rel = json_str(json_get(args, "path"), ".");
    if (!ws_path(T, rel, full, sizeof(full), out)) return 0;
    Ents E = { 0 };
    if (!dir_entries(full, ent_cb, &E)) { buf_printf(out, "error: folder not found: %s", rel); return 0; }
    qsort(E.e, (size_t)E.n, sizeof(Ent), ent_cmp);
    buf_printf(out, "%s (%d entries)\n", rel, E.n);
    for (int i = 0; i < E.n && i < 500; i++) {
        if (E.e[i].is_dir) buf_printf(out, "  %s/\n", E.e[i].name);
        else buf_printf(out, "  %s  (%llu bytes)\n", E.e[i].name, (unsigned long long)E.e[i].size);
    }
    if (E.n > 500) buf_add(out, "  ...\n", -1);
    free(E.e);
    return 1;
}

static char *read_all(const char *path, int *n, Buf *err, const char *rel) {
    FILE *f = file_open(path, "rb");
    if (!f) { buf_printf(err, "error: file not found: %s", rel); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz > MAX_FILE) { fclose(f); buf_printf(err, "error: file too large (%ld bytes)", sz); return NULL; }
    char *s = (char *)malloc((size_t)sz + 1);
    *n = (int)fread(s, 1, (size_t)sz, f);
    s[*n] = 0;
    fclose(f);
    return s;
}

static int is_binary(const char *s, int n) {
    for (int i = 0; i < n && i < 8192; i++)
        if (s[i] == 0) return 1;
    return 0;
}

static int tool_read(Tools *T, const Json *args, Buf *out) {
    char full[1100];
    const char *rel = json_str(json_get(args, "path"), NULL);
    if (!rel) { buf_add(out, "error: 'path' is required", -1); return 0; }
    if (!ws_path(T, rel, full, sizeof(full), out)) return 0;
    int n;
    char *s = read_all(full, &n, out, rel);
    if (!s) return 0;
    if (is_binary(s, n)) { free(s); buf_printf(out, "error: %s is a binary file", rel); return 0; }
    long long first = arg_int(args, "start_line", 1), maxl = arg_int(args, "max_lines", 200);
    if (first < 1) first = 1;
    if (maxl < 1) maxl = 1;
    if (maxl > 1000) maxl = 1000;
    int total = 0;
    for (int i = 0; i < n; i++) if (s[i] == '\n') total++;
    if (n && s[n - 1] != '\n') total++;
    Buf body = { 0 };
    int line = 1, shown = 0;
    for (const char *p = s; p < s + n && shown < maxl; line++) {
        const char *e = memchr(p, '\n', (size_t)(s + n - p));
        int len = e ? (int)(e - p) : (int)(s + n - p);
        if (line >= first) {
            if (len && p[len - 1] == '\r') len--;
            buf_printf(&body, "%5d| ", line);
            buf_add(&body, p, len);
            buf_add(&body, "\n", 1);
            shown++;
        }
        p = e ? e + 1 : s + n;
    }
    long long last = first + shown - 1;
    buf_printf(out, "%s (lines %lld-%lld of %d)\n", rel, shown ? first : 0, shown ? last : 0, total);
    if (body.s) buf_add(out, body.s, body.n);
    if (last < total) buf_printf(out, "[more lines: call read_file with start_line=%lld]", last + 1);
    buf_free(&body);
    free(s);
    return 1;
}

typedef struct { Tools *T; const char *pattern, *query; Buf *out; int hits, limit, files; } Walk;

static void walk(Walk *w, const char *full, const char *rel, int depth) {
    if (depth > 12 || w->hits >= w->limit) return;
    Ents E = { 0 };
    if (!dir_entries(full, ent_cb, &E)) return;
    qsort(E.e, (size_t)E.n, sizeof(Ent), ent_cmp);
    for (int i = 0; i < E.n && w->hits < w->limit; i++) {
        char f2[1400], r2[1400];
        snprintf(f2, sizeof(f2), "%s/%s", full, E.e[i].name);
        if (*rel) snprintf(r2, sizeof(r2), "%s/%s", rel, E.e[i].name);
        else snprintf(r2, sizeof(r2), "%s", E.e[i].name);
        if (E.e[i].is_dir) { walk(w, f2, r2, depth + 1); continue; }
        if (w->pattern && !glob_match(w->pattern, E.e[i].name)) continue;
        if (!w->query) {
            buf_printf(w->out, "%s  (%llu bytes)\n", r2, (unsigned long long)E.e[i].size);
            w->hits++;
            continue;
        }
        if (E.e[i].size > MAX_FILE) continue;
        int n;
        Buf dummy = { 0 };
        char *s = read_all(f2, &n, &dummy, r2);
        buf_free(&dummy);
        if (!s) continue;
        if (is_binary(s, n)) { free(s); continue; }
        w->files++;
        size_t ql = strlen(w->query);
        int line = 1;
        for (const char *p = s; p < s + n && w->hits < w->limit; line++) {
            const char *e = memchr(p, '\n', (size_t)(s + n - p));
            int len = e ? (int)(e - p) : (int)(s + n - p);
            for (int k = 0; k + (int)ql <= len; k++) {
                size_t j = 0;
                while (j < ql && tolower((unsigned char)p[k + j]) == tolower((unsigned char)w->query[j])) j++;
                if (j == ql) {
                    int show = len > 300 ? 300 : len;
                    if (show && p[show - 1] == '\r') show--;
                    buf_printf(w->out, "%s:%d: ", r2, line);
                    buf_add(w->out, p, show);
                    buf_add(w->out, "\n", 1);
                    w->hits++;
                    break;
                }
            }
            p = e ? e + 1 : s + n;
        }
        free(s);
    }
    free(E.e);
}

static int tool_find(Tools *T, const Json *args, Buf *out, int grep) {
    char full[1100];
    const char *rel = json_str(json_get(args, "path"), ".");
    if (!ws_path(T, rel, full, sizeof(full), out)) return 0;
    Walk w = { T, NULL, NULL, out, 0, grep ? 100 : 200, 0 };
    if (grep) {
        w.query = json_str(json_get(args, "query"), NULL);
        w.pattern = json_str(json_get(args, "pattern"), NULL);
        if (!w.query || !*w.query) { buf_add(out, "error: 'query' is required", -1); return 0; }
    } else {
        w.pattern = json_str(json_get(args, "pattern"), NULL);
        if (!w.pattern || !*w.pattern) { buf_add(out, "error: 'pattern' is required", -1); return 0; }
    }
    const char *r = rel;
    while (r[0] == '.' && (r[1] == '/' || r[1] == '\\')) r += 2;
    walk(&w, full, strcmp(r, ".") ? r : "", 0);
    if (!w.hits) buf_add(out, grep ? "no matches" : "no files found", -1);
    else if (w.hits >= w.limit) buf_add(out, "[more results not shown: narrow the search]", -1);
    return 1;
}

static int tool_write(Tools *T, const Json *args, Buf *out) {
    char full[1100];
    const char *rel = json_str(json_get(args, "path"), NULL);
    const Json *content = json_get(args, "content");
    if (!rel || !content || content->t != J_STR) { buf_add(out, "error: 'path' and 'content' are required", -1); return 0; }
    if (!ws_path(T, rel, full, sizeof(full), out)) return 0;
    /* create parent folders */
    for (char *p = full + strlen(T->workspace) + 1; *p; p++)
        if (*p == '/') { *p = 0; dir_create(full); *p = '/'; }
    FILE *f = file_open(full, "wb");
    if (!f) { buf_printf(out, "error: cannot write %s", rel); return 0; }
    size_t w = fwrite(content->s, 1, (size_t)content->len, f);
    fclose(f);
    if (w != (size_t)content->len) { buf_printf(out, "error: write failed for %s", rel); return 0; }
    buf_printf(out, "wrote %d bytes to %s", content->len, rel);
    return 1;
}

/* Exact text replacement. Files with CRLF line ends also match text written with \n. */
static char *to_crlf(const char *s, int n, int *out_n) {
    char *r = (char *)malloc((size_t)n * 2 + 1);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == '\n' && (i == 0 || s[i - 1] != '\r')) r[k++] = '\r';
        r[k++] = s[i];
    }
    r[k] = 0;
    *out_n = k;
    return r;
}

static int count_text(const char *s, int n, const char *q, int ql, int *first) {
    int c = 0;
    *first = -1;
    for (int i = 0; i + ql <= n;) {
        const char *p = (const char *)memchr(s + i, q[0], (size_t)(n - ql - i + 1));
        if (!p) break;
        i = (int)(p - s);
        if (!memcmp(p, q, (size_t)ql)) {
            if (*first < 0) *first = i;
            c++;
            i += ql;
        } else {
            i++;
        }
    }
    return c;
}

static int tool_edit(Tools *T, const Json *args, Buf *out) {
    char full[1100];
    const char *rel = json_str(json_get(args, "path"), NULL);
    const Json *o = json_get(args, "old_text"), *nw = json_get(args, "new_text");
    if (!rel || !o || o->t != J_STR || !nw || nw->t != J_STR) {
        buf_add(out, "error: 'path', 'old_text' and 'new_text' are required", -1);
        return 0;
    }
    if (!o->len) { buf_add(out, "error: old_text is empty: use write_file to create a file", -1); return 0; }
    if (!ws_path(T, rel, full, sizeof(full), out)) return 0;
    int n;
    char *s = read_all(full, &n, out, rel);
    if (!s) return 0;
    if (is_binary(s, n)) { free(s); buf_printf(out, "error: %s is a binary file", rel); return 0; }
    const char *os = o->s, *ns = nw->s;
    int ol = o->len, nl = nw->len, first;
    char *oc = NULL, *nc = NULL;
    int cnt = count_text(s, n, os, ol, &first);
    if (!cnt && memchr(os, '\n', (size_t)ol) && strstr(s, "\r\n")) {
        int ocl, ncl;
        oc = to_crlf(os, ol, &ocl);
        nc = to_crlf(ns, nl, &ncl);
        cnt = count_text(s, n, oc, ocl, &first);
        if (cnt) { os = oc; ol = ocl; ns = nc; nl = ncl; }
    }
    int all = json_bool(json_get(args, "replace_all"), 0), ok = 0;
    if (!cnt) {
        buf_printf(out, "error: old_text was not found in %s. Read the file again and copy the text exactly, "
                        "including spaces and indentation.", rel);
    } else if (cnt > 1 && !all) {
        buf_printf(out, "error: old_text appears %d times in %s: include more of the surrounding lines to make it "
                        "unique, or set replace_all to true.", cnt, rel);
    } else {
        Buf r = { 0 };
        int i = 0, done = 0;
        while (i < n) {
            if ((all || !done) && i + ol <= n && !memcmp(s + i, os, (size_t)ol)) {
                buf_add(&r, ns, nl);
                i += ol;
                done++;
            } else {
                const char *p = (all || !done) ? (const char *)memchr(s + i + 1, os[0], (size_t)(n - i - 1)) : NULL;
                int stop = p ? (int)(p - s) : n;
                buf_add(&r, s + i, stop - i);
                i = stop;
            }
        }
        FILE *f = file_open(full, "wb");
        if (!f) buf_printf(out, "error: cannot write %s", rel);
        else {
            size_t w = r.n ? fwrite(r.s, 1, (size_t)r.n, f) : 0;
            fclose(f);
            if (w != (size_t)r.n) buf_printf(out, "error: write failed for %s", rel);
            else {
                int line = 1;
                for (int k = 0; k < first; k++) line += s[k] == '\n';
                buf_printf(out, "edited %s: %d replacement%s (first at line %d)", rel, done, done > 1 ? "s" : "", line);
                ok = 1;
            }
        }
        buf_free(&r);
    }
    free(oc);
    free(nc);
    free(s);
    return ok;
}

/* ------------------------------------------------------------------ shell commands and background processes */

struct Job {
    int id, pid;
    char *cmd;
    char shell[16];
    Proc *p;
    PithosLock *lk;
    double t0, t1;
    int code, done, refs;           /* refs: the owner, two pipe readers and the exit waiter */
    int out_eof, err_eof;
    Buf out, err;                   /* output kept so far (the oldest part goes beyond JOB_KEEP) */
    int out_seen, err_seen;         /* bytes of out / err already returned by read_process */
    long long out_drop, err_drop;   /* bytes dropped from the start */
    int ps, clixml;                 /* PowerShell; its stderr turned out to be CLIXML */
    Buf pend;                       /* CLIXML not decoded yet */
    char *script;                   /* temporary script file, removed at the end */
    int chains;                     /* '&&' / '||' rewritten for Windows PowerShell 5.1 */
};

static const char *const SHELL_ENV[] = { "NO_COLOR=1", "TERM=dumb", "GIT_TERMINAL_PROMPT=0", "GIT_PAGER=cat", "PAGER=cat",
                                         "PYTHONIOENCODING=utf-8", "PYTHONUNBUFFERED=1", "DEBIAN_FRONTEND=noninteractive",
                                         NULL };

static void job_release(Job *j) {
    lock_acquire(j->lk);
    int left = --j->refs;
    lock_release(j->lk);
    if (left) return;
    proc_free(j->p);
    buf_free(&j->out);
    buf_free(&j->err);
    buf_free(&j->pend);
    if (j->script) { file_remove(j->script); free(j->script); }
    free(j->cmd);
    lock_destroy(j->lk);
    free(j);
}

/* Text of a CLIXML string: XML entities and _xHHHH_ (UTF-16 code units) decoded. */
static void clixml_decode(const char *s, const char *e, Buf *o) {
    unsigned hi = 0;
    while (s < e) {
        if (*s == '_' && e - s >= 7 && s[1] == 'x' && s[6] == '_' && isxdigit((unsigned char)s[2]) &&
            isxdigit((unsigned char)s[3]) && isxdigit((unsigned char)s[4]) && isxdigit((unsigned char)s[5])) {
            char hex[5] = { s[2], s[3], s[4], s[5], 0 };
            unsigned u = (unsigned)strtoul(hex, NULL, 16);
            s += 7;
            if (u >= 0xD800 && u < 0xDC00) { hi = u; continue; }
            if (u >= 0xDC00 && u < 0xE000 && hi) u = 0x10000 + ((hi - 0xD800) << 10) + (u - 0xDC00);
            hi = 0;
            char c[4];
            buf_add(o, c, utf8_put(c, u));
            continue;
        }
        if (*s == '&') {
            char c[4];
            int cl, k = html_entity(s, e, c, &cl);
            if (k > 0) { buf_add(o, c, cl); s += k; continue; }
        }
        buf_add(o, s, 1);
        s++;
    }
}

/* PowerShell writes its error stream as CLIXML when stderr is a pipe: keep the text of the
 * <S S="Error">...</S> records (progress objects and the like are dropped). */
static void clixml_take(Job *j) {
    Buf *p = &j->pend;
    int used = 0;
    for (;;) {
        char *s = strstr(p->s + used, "<S S=\"");
        if (!s) { used = p->n > 8 ? p->n - 8 : used; break; } /* keep a possible partial tag */
        char *gt = strchr(s, '>');
        char *e = gt ? strstr(gt, "</S>") : NULL;
        if (!e) { used = (int)(s - p->s); break; }
        clixml_decode(gt + 1, e, &j->err);
        used = (int)(e + 4 - p->s);
    }
    memmove(p->s, p->s + used, (size_t)(p->n - used + 1));
    p->n -= used;
}

static void job_reader(Job *j, int is_err) {
    char tmp[16384];
    int k;
    Buf *b = is_err ? &j->err : &j->out;
    int *seen = is_err ? &j->err_seen : &j->out_seen;
    long long *drop = is_err ? &j->err_drop : &j->out_drop;
    while ((k = is_err ? proc_read_err(j->p, tmp, sizeof(tmp)) : proc_read(j->p, tmp, sizeof(tmp))) > 0) {
        lock_acquire(j->lk);
        if (is_err && j->ps && !j->clixml && b->n == 0 && j->pend.n + k >= 9) {
            buf_add(&j->pend, tmp, k);
            j->clixml = !strncmp(j->pend.s, "#< CLIXML", 9);
            if (!j->clixml) { buf_add(b, j->pend.s, j->pend.n); j->pend.n = 0; }
            else clixml_take(j);
        } else if (is_err && j->ps && (j->clixml || b->n == 0)) {
            buf_add(&j->pend, tmp, k);
            if (j->clixml) clixml_take(j);
        } else {
            buf_add(b, tmp, k);
        }
        if (b->n > JOB_KEEP) {
            int cut = b->n - JOB_KEEP / 2;
            memmove(b->s, b->s + cut, (size_t)(b->n - cut + 1));
            b->n -= cut;
            *drop += cut;
            *seen = *seen > cut ? *seen - cut : 0;
        }
        lock_notify_all(j->lk);
        lock_release(j->lk);
    }
    lock_acquire(j->lk);
    if (is_err && !j->clixml && j->pend.n) buf_add(b, j->pend.s, j->pend.n); /* short plain stderr */
    if (is_err) j->err_eof = 1;
    else j->out_eof = 1;
    lock_notify_all(j->lk);
    lock_release(j->lk);
    job_release(j);
}

static void job_read_out(void *arg) { job_reader((Job *)arg, 0); }
static void job_read_err(void *arg) { job_reader((Job *)arg, 1); }

static void job_waiter(void *arg) {
    Job *j = (Job *)arg;
    int code = proc_wait(j->p, -1);
    lock_acquire(j->lk);
    j->code = code;
    j->t1 = time_sec();
    j->done = 1;
    lock_notify_all(j->lk);
    lock_release(j->lk);
    job_release(j);
}

#if defined(_WIN32)
/* PowerShell runs the command from a script file (-File): no quoting rule can break it, errors
 * point at the failing line only and "exit N" sets the exit code. UTF-8 with BOM, as Windows
 * PowerShell 5.1 expects. */
static int ps_script(const char *script, char *path, int cap) {
    static volatile long seq;
    wchar_t tmp[MAX_PATH];
    char dir[MAX_PATH * 3];
    if (!GetTempPathW(MAX_PATH, tmp) || !WideCharToMultiByte(CP_UTF8, 0, tmp, -1, dir, sizeof(dir), NULL, NULL)) return 0;
    snprintf(path, (size_t)cap, "%spithos-cmd-%lu-%ld.ps1", dir, (unsigned long)GetCurrentProcessId(),
             InterlockedIncrement(&seq));
    FILE *f = file_open(path, "wb");
    if (!f) return 0;
    fwrite("\xEF\xBB\xBF", 1, 3, f);
    size_t n = strlen(script), w = fwrite(script, 1, n, f);
    fclose(f);
    return w == n;
}
#endif

/* Windows PowerShell 5.1 has no '&&' / '||', and models write them all the time: outside
 * quotes, braces and parentheses "a && b" becomes "a; if (-not $?) { exit <code> }; b" (the
 * command fails with a's exit code, as in a shell) and "a || b" "a; if (-not $?) { b }".
 * Returns the number of rewrites. */
static int ps_rewrite_chains(const char *cmd, Buf *out) {
    int n = (int)strlen(cmd), depth = 0, rewrites = 0, seg = 0;
    char q = 0;
    Buf b = { 0 };
    int closes = 0; /* if-blocks to close at the end */
    for (int i = 0; i < n; i++) {
        char c = cmd[i];
        if (q) {
            if (c == '`' && q == '"' && i + 1 < n) { i++; continue; }
            if (c == q) q = 0;
            continue;
        }
        if (c == '\'' || c == '"') { q = c; continue; }
        if (c == '{' || c == '(') depth++;
        else if ((c == '}' || c == ')') && depth > 0) depth--;
        else if (!depth && i + 1 < n && ((c == '&' && cmd[i + 1] == '&') || (c == '|' && cmd[i + 1] == '|'))) {
            buf_add(&b, cmd + seg, i - seg);
            while (b.n && b.s[b.n - 1] == ' ') b.n--;
            if (c == '&') {
                buf_add(&b, "; if (-not $?) { exit $(if ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }) }; ", -1);
            } else {
                buf_add(&b, "; if (-not $?) { ", -1);
                closes++;
            }
            rewrites++;
            i++;
            while (i + 1 < n && cmd[i + 1] == ' ') i++;
            seg = i + 1;
        }
    }
    buf_add(&b, cmd + seg, n - seg);
    for (int k = 0; k < closes; k++) buf_add(&b, " }", 2);
    if (rewrites) buf_add(out, b.s, b.n);
    buf_free(&b);
    return rewrites;
}

static int is_abs_path(const char *p) {
    return p[0] == '/' || p[0] == '\\' || (isalpha((unsigned char)p[0]) && p[1] == ':');
}

/* Starts the command of a run_command / start_process call. */
static Job *job_start(Tools *T, const Json *args, Buf *err) {
    const char *cmd = json_str(json_get(args, "command"), NULL);
    if (!cmd || !*cmd) { buf_add(err, "error: 'command' is required", -1); return NULL; }
    if (strlen(cmd) > MAX_COMMAND) {
        buf_add(err, "error: the command is too long: write a script file with write_file and run that", -1);
        return NULL;
    }
    const char *shell = json_str(json_get(args, "shell"), "");
    const char *rel = json_str(json_get(args, "cwd"), "");
    char cwd[1100] = "";
    if (*rel && is_abs_path(rel)) snprintf(cwd, sizeof(cwd), "%s", rel);
    else if (*rel) snprintf(cwd, sizeof(cwd), "%s/%s", T->workspace[0] ? T->workspace : ".", rel);
    else if (T->workspace[0]) snprintf(cwd, sizeof(cwd), "%s", T->workspace);
    const char *argv[12];
    int na = 0;
    char *script = NULL, *cmdline = NULL;
    char used[16], script_path[1100] = "";
    int chains = 0;
#if defined(_WIN32)
    if (!strcmp(shell, "cmd")) {
        snprintf(used, sizeof(used), "cmd");
        size_t l = strlen(cmd) + 64;
        cmdline = (char *)malloc(l);
        snprintf(cmdline, l, "cmd.exe /d /s /c \"chcp 65001>nul & %s\"", cmd);
        argv[na++] = "cmd.exe";
    } else if (!*shell || !strcmp(shell, "powershell") || !strcmp(shell, "pwsh")) {
        int pwsh = !strcmp(shell, "pwsh");
        snprintf(used, sizeof(used), "%s", pwsh ? "pwsh" : "powershell");
        /* one line, so that error positions are off by one line only */
        static const char head[] = "$ProgressPreference='SilentlyContinue'; try { [Console]::OutputEncoding="
                                   "[Text.Encoding]::UTF8 } catch {}; $OutputEncoding=[Text.Encoding]::UTF8\n";
        static const char tail[] = "\nexit $(if ($?) { 0 } elseif ($LASTEXITCODE) { $LASTEXITCODE } else { 1 })\n";
        Buf fixed = { 0 };
        if (!pwsh) chains = ps_rewrite_chains(cmd, &fixed);
        const char *body = chains ? fixed.s : cmd;
        size_t l = sizeof(head) + strlen(body) + sizeof(tail);
        script = (char *)malloc(l);
        snprintf(script, l, "%s%s%s", head, body, tail);
        buf_free(&fixed);
        if (!ps_script(script, script_path, sizeof(script_path))) {
            free(script);
            buf_add(err, "error: cannot write the command to the temporary folder", -1);
            return NULL;
        }
        argv[na++] = pwsh ? "pwsh" : "powershell";
        argv[na++] = "-NoLogo";
        argv[na++] = "-NoProfile";
        argv[na++] = "-NonInteractive";
        argv[na++] = "-ExecutionPolicy";
        argv[na++] = "Bypass";
        argv[na++] = "-File";
        argv[na++] = script_path;
    } else {
        buf_add(err, "error: on Windows the shell is powershell (default), pwsh or cmd", -1);
        return NULL;
    }
#else
    FILE *bf = fopen("/bin/bash", "rb");
    int has_bash = bf != NULL;
    if (bf) fclose(bf);
    if (!*shell) shell = has_bash ? "bash" : "sh";
    if (strcmp(shell, "bash") && strcmp(shell, "sh")) { buf_add(err, "error: the shell is bash or sh", -1); return NULL; }
    snprintf(used, sizeof(used), "%s", shell);
    argv[na++] = !strcmp(shell, "bash") ? "/bin/bash" : "/bin/sh";
    argv[na++] = "-c";
    argv[na++] = cmd;
#endif
    argv[na] = NULL;
    ProcOptions po;
    memset(&po, 0, sizeof(po));
    po.argv = argv;
    po.env = SHELL_ENV;
    po.cwd = cwd[0] ? cwd : NULL;
    po.err_mode = PROC_ERR_PIPE;
    po.win_cmdline = cmdline;
    char e[300];
    Proc *p = proc_start(&po, e, sizeof(e));
    free(script);
    free(cmdline);
    if (!p) {
        if (script_path[0]) file_remove(script_path);
        buf_printf(err, "error: %s", e);
        return NULL;
    }
    proc_close_stdin(p); /* no input: a command waiting for it sees the end at once */
    Job *j = (Job *)calloc(1, sizeof(Job));
    j->p = p;
    j->pid = proc_pid(p);
    j->cmd = dupstr(cmd);
    snprintf(j->shell, sizeof(j->shell), "%s", used);
    j->lk = lock_create();
    j->t0 = time_sec();
    j->refs = 4;
    j->ps = !strcmp(used, "powershell") || !strcmp(used, "pwsh");
    j->chains = chains;
    if (script_path[0]) j->script = dupstr(script_path);
    if (!thread_spawn(job_read_out, j)) { j->out_eof = 1; j->refs--; }
    if (!thread_spawn(job_read_err, j)) { j->err_eof = 1; j->refs--; }
    if (!thread_spawn(job_waiter, j)) { proc_kill(p); j->done = 1; j->code = -1; j->refs--; }
    return j;
}

/* Waits for the pipes to close (the readers have everything), at most ms. Holds j->lk. */
static void job_drain(Job *j, int ms) {
    double t0 = time_sec();
    while (!(j->out_eof && j->err_eof) && (time_sec() - t0) * 1e3 < ms) lock_wait_ms(j->lk, 20);
}

/* Command output for the model: no terminal escapes, \r\n as \n, a lone \r overwrites the line
 * (progress bars), invalid UTF-8 as '?'. */
static void clean_output(const char *s, int n, Buf *o) {
    for (int i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c == 0x1b) { /* CSI ... final byte, OSC ... BEL / ST */
            i++;
            if (i < n && s[i] == '[') {
                for (i++; i < n && !((unsigned char)s[i] >= 0x40 && (unsigned char)s[i] <= 0x7e); i++) {}
                i++;
            } else if (i < n && s[i] == ']') {
                for (i++; i < n && s[i] != 7 && !(s[i] == 0x1b && i + 1 < n && s[i + 1] == '\\'); i++) {}
                i += i < n && s[i] == 7 ? 1 : 2;
            } else {
                i++;
            }
            continue;
        }
        if (c == '\r') {
            if (i + 1 < n && s[i + 1] == '\n') { i++; continue; }
            if (i + 1 < n) { /* back to the start of the line */
                int k = o->n;
                while (k > 0 && o->s[k - 1] != '\n') k--;
                o->n = k;
                if (o->s) o->s[k] = 0;
            }
            i++;
            continue;
        }
        if (c < 0x20 && c != '\n' && c != '\t') { i++; continue; }
        if (c < 0x80) { buf_add(o, s + i, 1); i++; continue; }
        int need = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        int ok = need && i + need <= n;
        for (int k = 1; ok && k < need; k++) ok = ((unsigned char)s[i + k] & 0xC0) == 0x80;
        if (ok) { buf_add(o, s + i, need); i += need; continue; }
#if defined(_WIN32)
        /* not UTF-8: a console program writing in the OEM code page (e.g. PowerShell's own
         * parse errors, printed before the script can switch to UTF-8) */
        wchar_t w;
        char u[4];
        int k;
        if (MultiByteToWideChar(CP_OEMCP, 0, s + i, 1, &w, 1) == 1 &&
            (k = WideCharToMultiByte(CP_UTF8, 0, &w, 1, u, sizeof(u), NULL, NULL)) > 0) {
            buf_add(o, u, k);
            i++;
            continue;
        }
#endif
        buf_add(o, "?", 1);
        i++;
    }
}

/* One stream for the model: head and tail when it is long (the end matters most: errors, results). */
static void show_stream(Buf *out, const char *label, const char *s, int n, long long dropped, int max) {
    Buf c = { 0 };
    clean_output(s, n, &c);
    while (c.n > 0 && (c.s[c.n - 1] == '\n' || c.s[c.n - 1] == ' ')) c.n--;
    if (c.n == 0) { buf_free(&c); return; }
    buf_printf(out, "\n--- %s ---\n", label);
    if (dropped > 0) buf_printf(out, "[%lld earlier bytes not kept]\n", dropped);
    if (c.n <= max) {
        buf_add(out, c.s, c.n);
    } else {
        int head = max / 4, tail = max - head;
        while (head > 0 && ((unsigned char)c.s[head] & 0xC0) == 0x80) head--;
        int from = c.n - tail;
        while (from < c.n && ((unsigned char)c.s[from] & 0xC0) == 0x80) from++;
        int lines = 0;
        for (int k = head; k < from; k++) lines += c.s[k] == '\n';
        buf_add(out, c.s, head);
        buf_printf(out, "\n[... %d lines, %d bytes omitted ...]\n", lines, from - head);
        buf_add(out, c.s + from, c.n - from);
    }
    buf_free(&c);
}

static void show_cmd(Buf *out, const char *cmd) {
    int l = (int)strlen(cmd);
    const char *nl = strchr(cmd, '\n');
    int show = nl ? (int)(nl - cmd) : l;
    if (show > 120) show = 120;
    buf_add(out, cmd, show);
    if (show < l) buf_add(out, " ...", -1);
}

/* "process 3 (pid 1234) is running for 2.1 s" / "exited with code 1 after 0.4 s". Holds j->lk. */
static void job_status(Job *j, Buf *out) {
    if (!j->done) buf_printf(out, "process %d (pid %d) is running for %.1f s: ", j->id, j->pid, time_sec() - j->t0);
    else buf_printf(out, "process %d (pid %d) exited with code %d after %.1f s: ", j->id, j->pid, j->code, j->t1 - j->t0);
    show_cmd(out, j->cmd);
}

/* New output since the last read. Holds j->lk. */
static void job_new_output(Job *j, Buf *out, int max) {
    int before = out->n;
    show_stream(out, "stdout", j->out.s ? j->out.s + j->out_seen : "", j->out.n - j->out_seen,
                j->out_seen == 0 ? j->out_drop : 0, max * 2 / 3);
    show_stream(out, "stderr", j->err.s ? j->err.s + j->err_seen : "", j->err.n - j->err_seen,
                j->err_seen == 0 ? j->err_drop : 0, max / 3);
    j->out_seen = j->out.n;
    j->err_seen = j->err.n;
    if (out->n == before) buf_add(out, "\n(no new output)", -1);
}

static int tool_run(Tools *T, const Json *args, Buf *out) {
    long long timeout = arg_int(args, "timeout", 120);
    if (timeout < 1) timeout = 1;
    if (timeout > 3600) timeout = 3600;
    Job *j = job_start(T, args, out);
    if (!j) return 0;
    lock_acquire(j->lk);
    while (!j->done && !T->interrupt && time_sec() - j->t0 < (double)timeout) lock_wait_ms(j->lk, 100);
    int finished = j->done;
    double elapsed = time_sec() - j->t0;
    if (finished) {
        elapsed = j->t1 - j->t0;
        job_drain(j, 300); /* the rest of the output, unless something the command started holds the pipe */
    }
    lock_release(j->lk);
    proc_kill(j->p); /* the command, or what it left running: programs that must stay use start_process */
    lock_acquire(j->lk);
    double t0 = time_sec();
    while (!j->done && time_sec() - t0 < 3) lock_wait_ms(j->lk, 20);
    job_drain(j, 2000);
    int code = j->code;
    if (finished)
        buf_printf(out, "exit code %d (%s) | %.2f s | %s", code, code ? "failed" : "success", elapsed, j->shell);
    else if (T->interrupt)
        buf_printf(out, "INTERRUPTED by the user after %.1f s: the command was stopped | %s", elapsed, j->shell);
    else
        buf_printf(out, "TIMED OUT after %lld s: the command was stopped. If it is a program that keeps running "
                        "(server, watcher), use start_process; otherwise give it a larger timeout | %s", timeout, j->shell);
    const char *cwd = json_str(json_get(args, "cwd"), "");
    if (*cwd) buf_printf(out, " | cwd: %s", cwd);
    if (j->chains)
        buf_add(out, "\n(note: Windows PowerShell 5.1 has no '&&' / '||': they were run with 'if ($?)' checks)", -1);
    int before = out->n;
    show_stream(out, "stdout", j->out.s ? j->out.s : "", j->out.n, j->out_drop, 12000);
    show_stream(out, "stderr", j->err.s ? j->err.s : "", j->err.n, j->err_drop, 6000);
    if (out->n == before) buf_add(out, "\n(no output)", -1);
    lock_release(j->lk);
    job_release(j);
    return finished && code == 0;
}

static Job *job_get(Tools *T, const Json *args, Buf *out) {
    long long id = arg_int(args, "id", -1);
    Job *j = NULL;
    lock_acquire(T->lock);
    for (int i = 0; i < T->njobs; i++)
        if (T->jobs[i]->id == id) { j = T->jobs[i]; lock_acquire(j->lk); j->refs++; lock_release(j->lk); }
    lock_release(T->lock);
    if (!j) buf_printf(out, "error: no background process with id %lld (see list_processes)", id);
    return j;
}

static int tool_start(Tools *T, const Json *args, Buf *out) {
    lock_acquire(T->lock);
    int room = T->njobs < MAX_JOBS;
    for (int i = 0; i < T->njobs && !room; i++) room = T->jobs[i]->done;
    lock_release(T->lock);
    if (!room) { buf_add(out, "error: too many background processes: stop some with stop_process", -1); return 0; }
    Job *j = job_start(T, args, out);
    if (!j) return 0;
    lock_acquire(T->lock);
    if (T->njobs == MAX_JOBS) { /* forget the oldest finished one */
        for (int i = 0; i < T->njobs; i++)
            if (T->jobs[i]->done) {
                Job *old = T->jobs[i];
                memmove(T->jobs + i, T->jobs + i + 1, sizeof(Job *) * (size_t)(T->njobs - i - 1));
                T->njobs--;
                job_release(old);
                break;
            }
    }
    j->id = ++T->next_job;
    T->jobs[T->njobs++] = j;
    lock_acquire(j->lk); /* the list holds the owner reference, this call one more */
    j->refs++;
    lock_release(j->lk);
    lock_release(T->lock);
    long long wait = arg_int(args, "wait", 3);
    if (wait < 0) wait = 0;
    if (wait > 60) wait = 60;
    lock_acquire(j->lk);
    while (!j->done && !T->interrupt && time_sec() - j->t0 < (double)wait) lock_wait_ms(j->lk, 100);
    if (j->done) job_drain(j, 300);
    buf_add(out, "started: ", -1);
    job_status(j, out);
    job_new_output(j, out, 8000);
    if (!j->done) buf_printf(out, "\nUse read_process with id=%d to follow it and stop_process to end it.", j->id);
    int ok = !j->done || j->code == 0;
    lock_release(j->lk);
    job_release(j);
    return ok;
}

static int tool_readp(Tools *T, const Json *args, Buf *out) {
    Job *j = job_get(T, args, out);
    if (!j) return 0;
    long long wait = arg_int(args, "wait", 0);
    if (wait < 0) wait = 0;
    if (wait > 600) wait = 600;
    const char *until = json_str(json_get(args, "until"), NULL);
    if (until && !*until) until = NULL;
    double t0 = time_sec();
    lock_acquire(j->lk);
    while (!j->done && !T->interrupt && time_sec() - t0 < (double)wait) {
        /* anywhere in the kept output: it may have been shown by an earlier call already */
        if (until && ((j->out.s && strstr(j->out.s, until)) || (j->err.s && strstr(j->err.s, until)))) break;
        lock_wait_ms(j->lk, 100);
    }
    if (j->done) job_drain(j, 300);
    job_status(j, out);
    if (until && !j->done && time_sec() - t0 >= (double)wait && wait > 0)
        buf_printf(out, "\n('%s' did not appear within %lld s)", until, wait);
    job_new_output(j, out, 12000);
    int ok = !j->done || j->code == 0;
    lock_release(j->lk);
    job_release(j);
    return ok;
}

static int tool_stopp(Tools *T, const Json *args, Buf *out) {
    Job *j = job_get(T, args, out);
    if (!j) return 0;
    lock_acquire(j->lk);
    int was_running = !j->done;
    lock_release(j->lk);
    proc_kill(j->p);
    lock_acquire(j->lk);
    double t0 = time_sec();
    while (!j->done && time_sec() - t0 < 3) lock_wait_ms(j->lk, 20);
    job_drain(j, 1000);
    if (was_running) buf_add(out, "stopped. ", -1);
    else buf_add(out, "it had already ended. ", -1);
    job_status(j, out);
    job_new_output(j, out, 8000);
    lock_release(j->lk);
    job_release(j);
    return 1;
}

static int tool_listp(Tools *T, Buf *out) {
    lock_acquire(T->lock);
    if (!T->njobs) buf_add(out, "no background processes", -1);
    for (int i = 0; i < T->njobs; i++) {
        Job *j = T->jobs[i];
        lock_acquire(j->lk);
        if (i) buf_add(out, "\n", 1);
        job_status(j, out);
        lock_release(j->lk);
    }
    lock_release(T->lock);
    return 1;
}

/* Commands that need the user's word even in autonomous mode. */
const char *tools_danger(const char *name, const Json *args) {
    static const struct { const char *pat, *why; } D[] = {
        { "rm -rf", "deletes files recursively" }, { "rm -fr", "deletes files recursively" },
        { "rm -r ", "deletes files recursively" }, { "rm --recursive", "deletes files recursively" },
        { "rmdir /s", "deletes folders recursively" }, { "rd /s", "deletes folders recursively" },
        { "del /s", "deletes files recursively" }, { "del /q", "deletes files without asking" },
        { "erase /s", "deletes files recursively" }, { "format-volume", "formats a disk" },
        { "clear-disk", "erases a disk" }, { "initialize-disk", "erases a disk" }, { "remove-partition", "deletes a partition" },
        { "diskpart", "changes disks and partitions" }, { "mkfs", "formats a disk" }, { "dd if=", "writes raw data" },
        { "wipefs", "erases a disk" }, { "of=/dev/", "writes to a device" }, { "> /dev/sd", "writes to a disk" },
        { "format c:", "formats a disk" }, { "shutdown", "shuts down the computer" }, { "reboot", "restarts the computer" },
        { "restart-computer", "restarts the computer" }, { "stop-computer", "shuts down the computer" },
        { "poweroff", "shuts down the computer" }, { "git push", "publishes to a remote repository" },
        { "git reset --hard", "discards uncommitted changes" }, { "git clean -f", "deletes untracked files" },
        { "git checkout -- .", "discards uncommitted changes" }, { "git restore .", "discards uncommitted changes" },
        { "npm publish", "publishes a package" }, { "cargo publish", "publishes a package" },
        { "twine upload", "publishes a package" }, { "docker push", "publishes an image" },
        { "gh repo delete", "deletes a repository" }, { "gh release delete", "deletes a release" },
        { "chmod -r", "changes permissions recursively" }, { "chown -r", "changes owners recursively" },
        { "takeown", "takes ownership of files" }, { "icacls", "changes permissions" }, { "reg delete", "edits the registry" },
        { "reg add", "edits the registry" }, { "bcdedit", "changes the boot configuration" }, { "cipher /w", "wipes free space" },
        { "sudo ", "runs as administrator" }, { "runas ", "runs as another user" },
        { "set-executionpolicy", "changes the script policy" }, { "netsh ", "changes network settings" },
        { "| sh", "runs a downloaded script" }, { "| bash", "runs a downloaded script" }, { "|sh", "runs a downloaded script" },
        { "|bash", "runs a downloaded script" }, { "iex ", "runs code from a string" }, { "iex(", "runs code from a string" },
        { "invoke-expression", "runs code from a string" }, { "taskkill", "ends processes" }, { "stop-process", "ends processes" },
        { "killall", "ends processes" }, { "pkill", "ends processes" }, { "drop database", "deletes a database" },
        { "drop table", "deletes a table" }, { "truncate table", "empties a table" }, { ":(){", "fork bomb" },
    };
    if (strcmp(name, "run_command") && strcmp(name, "start_process")) return NULL;
    const char *cmd = json_str(json_get(args, "command"), "");
    size_t n = strlen(cmd);
    char *low = (char *)malloc(n + 2);
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)cmd[i]);
    low[n] = ' '; /* "rm -r" at the end matches "rm -r " */
    low[n + 1] = 0;
    const char *why = NULL;
    for (size_t i = 0; i < sizeof(D) / sizeof(D[0]) && !why; i++)
        if (strstr(low, D[i].pat)) why = D[i].why;
    if (!why && strstr(low, "remove-item") && (strstr(low, "-recurse") || strstr(low, " -r "))) why = "deletes files recursively";
    free(low);
    return why;
}

/* ------------------------------------------------------------------ environment for the agent */

static int on_path(const char *name) {
#if defined(_WIN32)
    wchar_t w[64], found[MAX_PATH];
    if (!MultiByteToWideChar(CP_UTF8, 0, name, -1, w, 64)) return 0;
    const wchar_t *exts[] = { L".exe", L".cmd", L".bat", L".com" };
    for (int i = 0; i < 4; i++) {
        DWORD r = SearchPathW(NULL, w, exts[i], MAX_PATH, found, NULL);
        /* the python / python3 "App Execution Aliases" only open the Microsoft Store */
        if (r && r < MAX_PATH && !(strstr(name, "python") && wcsstr(found, L"\\WindowsApps\\"))) return 1;
    }
    return 0;
#else
    const char *path = getenv("PATH");
    if (!path) path = "/usr/local/bin:/usr/bin:/bin";
    char f[1100];
    for (const char *p = path; *p;) {
        const char *e = strchr(p, ':');
        int l = e ? (int)(e - p) : (int)strlen(p);
        if (l > 0 && l < 1000) {
            snprintf(f, sizeof(f), "%.*s/%s", l, p, name);
            if (access(f, X_OK) == 0) return 1;
        }
        p += l + (e ? 1 : 0);
        if (!e) break;
    }
    return 0;
#endif
}

static void absolute_path(const char *p, char *out, int cap) {
#if defined(_WIN32)
    wchar_t w[1100], full[1100];
    if (MultiByteToWideChar(CP_UTF8, 0, p, -1, w, 1100) && GetFullPathNameW(w, 1100, full, NULL) &&
        WideCharToMultiByte(CP_UTF8, 0, full, -1, out, cap, NULL, NULL))
        return;
#else
    char r[PATH_MAX];
    if (realpath(p, r) && (int)strlen(r) < cap) { memcpy(out, r, strlen(r) + 1); return; }
#endif
    snprintf(out, (size_t)cap, "%s", p);
}

static void build_env_info(Tools *T) {
    static const char *progs[] = { "git", "gh", "python", "python3", "py", "pip", "uv", "node", "npm", "npx", "deno", "bun",
                                   "gcc", "g++", "clang", "cl", "make", "mingw32-make", "cmake", "ninja", "cargo", "rustc",
                                   "go", "java", "javac", "mvn", "gradle", "dotnet", "docker", "podman", "kubectl", "curl",
                                   "wget", "ssh", "tar", "zip", "unzip", "7z", "pwsh", "bash", "wsl", "winget", "choco",
                                   "scoop", "apt", "dnf", "yum", "pacman", "zypper", "apk", "brew", "sudo", "systemctl",
                                   "sqlite3", "psql", "mysql", "ffmpeg", "jq", "rg", "code" };
    char os[256], wd[1100];
    os_describe(os, sizeof(os));
    absolute_path(T->workspace[0] ? T->workspace : ".", wd, sizeof(wd));
    Buf b = { 0 }, found = { 0 };
    int has_pwsh = 0;
    for (size_t i = 0; i < sizeof(progs) / sizeof(progs[0]); i++)
        if (on_path(progs[i])) {
            if (found.n) buf_add(&found, ", ", 2);
            buf_add(&found, progs[i], -1);
            if (!strcmp(progs[i], "pwsh")) has_pwsh = 1;
        }
    (void)has_pwsh; /* only Windows offers it */
    buf_printf(&b, "Operating system: %s\n", os);
#if defined(_WIN32)
    buf_printf(&b, "Shell for run_command / start_process: Windows PowerShell 5.1 (the default), or cmd.exe with "
                   "shell=\"cmd\"%s.\n", has_pwsh ? ", or PowerShell 7 with shell=\"pwsh\"" : "");
#else
    FILE *bf = fopen("/bin/bash", "rb");
    buf_printf(&b, "Shell for run_command / start_process: %s.\n", bf ? "bash (the default), or sh with shell=\"sh\"" : "sh");
    if (bf) fclose(bf);
#endif
    buf_add(&b, "Working folder: ", -1);
    buf_add(&b, wd, -1);
    buf_add(&b, " (file tool paths are relative to it; commands run in it unless cwd is given)\n", -1);
    buf_add(&b, "Programs found on the PATH: ", -1);
    buf_add(&b, found.n ? found.s : "none of the common ones", -1);
    buf_add(&b, " (others are probably not installed: check before relying on them)\n\nCommand rules for this system:\n", -1);
#if defined(_WIN32)
    buf_add(&b,
            "- This is Windows: do not use Linux commands or paths (no ls -la, grep, cat | head, rm -rf, touch, which, export, /dev/null, ~/).\n"
            "- PowerShell syntax: separate commands with ';'. Windows PowerShell 5.1 has no '&&' and '||': run the next command "
            "only on success with 'if ($?) { ... }' (a top-level '&&' is rewritten that way). The exit code of the last "
            "program is $LASTEXITCODE. To run in another folder, pass cwd instead of 'cd'.\n"
            "- PowerShell equivalents: Get-ChildItem (ls), Get-ChildItem -Recurse -Filter *.c (find), Select-String -Pattern x -Path *.c (grep), "
            "Get-Content f -TotalCount 20 / -Tail 20 (head / tail), Remove-Item -Recurse -Force (rm -rf), "
            "New-Item -ItemType Directory -Force (mkdir -p), Get-Command (which), $env:NAME = 'v' (export), $null (/dev/null), "
            "Copy-Item / Move-Item (cp / mv).\n"
            "- Quote paths with spaces; run a program given by a quoted path as: & \"C:\\path\\app.exe\" args.\n"
            "- Windows PowerShell 5.1 loses double quotes inside arguments passed to programs (python -c \"...\" with "
            "quotes inside breaks): for code longer than a line, write a script file with write_file and run it.\n"
            "- With shell=\"cmd\": cmd.exe syntax ('&&' and '||' work, variables are %NAME%, dir, type, del, copy).\n",
            -1);
#else
    buf_add(&b,
            "- POSIX shell syntax: '&&' runs the next command only on success; quote paths with spaces.\n"
            "- sudo cannot ask for a password here: use 'sudo -n' and, if that fails, tell the user the command to run.\n",
            -1);
#endif
    buf_add(&b,
            "- Commands run without a terminal and without input: never start editors, pagers or interactive prompts; "
            "pass flags such as -y, --yes, --no-input or -Force.\n"
            "- run_command waits for the end (default timeout 120 s, then the command is stopped). Programs that keep "
            "running (servers, watchers, 'npm run dev') must be started with start_process and followed with read_process.\n",
            -1);
    T->env_info = b.s;
    buf_free(&found);
}

const char *tools_env_info(Tools *T) {
    if (!T->env_info) build_env_info(T);
    return T->env_info;
}

/* Only a flag (callable from a signal handler): the waits look at it every 100 ms. */
void tools_set_interrupt(Tools *T, int on) { T->interrupt = on; }

/* ------------------------------------------------------------------ MCP client (stdio, JSON-RPC 2.0) */

static void mcp_send(Mcp *s, const Buf *b) {
    lock_acquire(s->wr);
    if (s->p && !proc_write(s->p, b->s, b->n)) s->alive = 0;
    lock_release(s->wr);
}

static void mcp_handle(Mcp *s, Json *j) {
    const Json *id = json_get(j, "id");
    const char *method = json_str(json_get(j, "method"), NULL);
    if (method && id) {
        /* request from the server: answer ping, refuse the rest */
        Buf b = { 0 };
        buf_add(&b, "{\"jsonrpc\":\"2.0\",\"id\":", -1);
        json_write(&b, id);
        if (!strcmp(method, "ping")) buf_add(&b, ",\"result\":{}}\n", -1);
        else buf_add(&b, ",\"error\":{\"code\":-32601,\"message\":\"method not supported by this client\"}}\n", -1);
        mcp_send(s, &b);
        buf_free(&b);
        json_free(j);
        return;
    }
    if (id && id->t == J_NUM && (json_get(j, "result") || json_get(j, "error"))) {
        lock_acquire(s->mx);
        if ((int)id->num == s->await_id && !s->resp) {
            s->resp = j;
            j = NULL;
            lock_notify_all(s->mx);
        }
        lock_release(s->mx);
    }
    json_free(j); /* notifications and stale responses */
}

static void mcp_reader(void *arg) {
    Mcp *s = (Mcp *)arg;
    Buf line = { 0 };
    char buf[16384];
    int k;
    while ((k = proc_read(s->p, buf, sizeof(buf))) > 0) {
        for (int i = 0; i < k; i++) {
            if (buf[i] != '\n') { buf_add(&line, buf + i, 1); continue; }
            int n = line.n;
            while (n > 0 && (line.s[n - 1] == '\r' || line.s[n - 1] == ' ')) n--;
            if (n > 0) {
                Json *j = json_parse(line.s, n, NULL, 0);
                if (j) mcp_handle(s, j);
            }
            line.n = 0;
        }
    }
    buf_free(&line);
    lock_acquire(s->mx);
    s->alive = 0;
    lock_notify_all(s->mx);
    lock_release(s->mx);
}

/* Send a request and wait for its response (the whole message, caller frees). */
static Json *mcp_rpc(Mcp *s, const char *method, const char *params, int timeout_ms, char *err, int errlen) {
    lock_acquire(s->req);
    Buf b = { 0 };
    lock_acquire(s->mx);
    int id = ++s->next_id;
    s->await_id = id;
    s->resp = NULL;
    lock_release(s->mx);
    buf_printf(&b, "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"%s\",\"params\":%s}\n", id, method, params ? params : "{}");
    mcp_send(s, &b);
    buf_free(&b);
    double t_end = time_sec() + timeout_ms / 1000.0;
    lock_acquire(s->mx);
    while (!s->resp && s->alive) {
        double left = t_end - time_sec();
        if (left <= 0) break;
        lock_wait_ms(s->mx, (int)(left * 1000) + 1);
    }
    Json *r = s->resp;
    s->resp = NULL;
    s->await_id = 0;
    int alive = s->alive;
    lock_release(s->mx);
    lock_release(s->req);
    if (!r) {
        snprintf(err, (size_t)errlen, alive ? "MCP server '%s' did not answer %s in time" : "MCP server '%s' has stopped",
                 s->name, method);
        if (!alive) snprintf(s->status, sizeof(s->status), "stopped");
        return NULL;
    }
    const Json *e = json_get(r, "error");
    if (e) {
        snprintf(err, (size_t)errlen, "MCP error: %s", json_str(json_get(e, "message"), "unknown"));
        json_free(r);
        return NULL;
    }
    return r;
}

static void sanitize(char *o, int cap, const char *s) {
    int n = 0;
    for (; *s && n < cap - 1; s++) o[n++] = (isalnum((unsigned char)*s) || *s == '_' || *s == '-') ? *s : '_';
    o[n] = 0;
}

static void mcp_start(void *arg) {
    Mcp *s = (Mcp *)arg;
    Tools *T = s->owner;
    char err[256];
    s->p = proc_spawn((const char *const *)s->argv, (const char *const *)s->env, 0, err, sizeof(err));
    if (!s->p) {
        snprintf(s->status, sizeof(s->status), "error");
        snprintf(s->error, sizeof(s->error), "%s", err);
        fprintf(stderr, "MCP %s: %s\n", s->name, err);
        return;
    }
    s->alive = 1;
    s->reader = thread_create(mcp_reader, s);
    Json *r = mcp_rpc(s, "initialize",
                      "{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},"
                      "\"clientInfo\":{\"name\":\"pandora\",\"title\":\"Pandora (Pithos)\",\"version\":\"0.4.0\"}}",
                      180000, err, sizeof(err));
    if (!r) goto fail;
    json_free(r);
    Buf note = { 0 };
    buf_add(&note, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n", -1);
    mcp_send(s, &note);
    buf_free(&note);
    char cursor[512] = "";
    for (int page = 0; page < 20; page++) {
        char params[600];
        if (cursor[0]) snprintf(params, sizeof(params), "{\"cursor\":\"%s\"}", cursor);
        else snprintf(params, sizeof(params), "{}");
        r = mcp_rpc(s, "tools/list", params, 60000, err, sizeof(err));
        if (!r) goto fail;
        const Json *res = json_get(r, "result");
        const Json *list = json_get(res, "tools");
        lock_acquire(T->lock);
        for (const Json *t = list ? list->child : NULL; t; t = t->next) {
            const char *name = json_str(json_get(t, "name"), NULL);
            if (!name) continue;
            Tool *tool = add_tool(T);
            char a[64], b[64];
            sanitize(a, sizeof(a), s->name);
            sanitize(b, sizeof(b), name);
            snprintf(tool->name, sizeof(tool->name), "%s__%s", a, b);
            tool->desc = dupstr(json_str(json_get(t, "description"), ""));
            const Json *schema = json_get(t, "inputSchema");
            Buf sb = { 0 };
            if (schema) json_write(&sb, schema);
            else buf_add(&sb, "{\"type\": \"object\", \"properties\": {}}", -1);
            tool->params = json_parse(sb.s, sb.n, NULL, 0);
            buf_free(&sb);
            tool->confirm = !json_bool(json_get(json_get(t, "annotations"), "readOnlyHint"), 0);
            tool->srv = s;
            tool->remote = dupstr(name);
            s->ntools++;
        }
        lock_release(T->lock);
        const char *next = json_str(json_get(res, "nextCursor"), NULL);
        snprintf(cursor, sizeof(cursor), "%s", next ? next : "");
        json_free(r);
        if (!cursor[0]) break;
    }
    snprintf(s->status, sizeof(s->status), "ready");
    fprintf(stderr, "MCP %s: %d tools\n", s->name, s->ntools);
    return;
fail:
    snprintf(s->status, sizeof(s->status), "error");
    snprintf(s->error, sizeof(s->error), "%s", err);
    fprintf(stderr, "MCP %s: %s\n", s->name, err);
}

static int mcp_call(Mcp *s, const char *remote, const Json *args, Buf *out) {
    char err[300];
    if (!s->alive) { buf_printf(out, "error: MCP server '%s' is not running", s->name); return 0; }
    Buf p = { 0 };
    buf_add(&p, "{\"name\":", -1);
    buf_json_str(&p, remote, -1);
    buf_add(&p, ",\"arguments\":", -1);
    if (args && args->t == J_OBJ) json_write(&p, args);
    else buf_add(&p, "{}", 2);
    buf_add(&p, "}", 1);
    Json *r = mcp_rpc(s, "tools/call", p.s, 300000, err, sizeof(err));
    buf_free(&p);
    if (!r) { buf_add(out, err, -1); return 0; }
    const Json *res = json_get(r, "result");
    int is_err = json_bool(json_get(res, "isError"), 0);
    const Json *content = json_get(res, "content");
    for (const Json *c = content ? content->child : NULL; c; c = c->next) {
        const char *type = json_str(json_get(c, "type"), "");
        if (out->n) buf_add(out, "\n", 1);
        if (!strcmp(type, "text")) buf_add(out, json_str(json_get(c, "text"), ""), -1);
        else if (!strcmp(type, "resource")) {
            const Json *rs = json_get(c, "resource");
            const char *txt = json_str(json_get(rs, "text"), NULL);
            if (txt) buf_add(out, txt, -1);
            else buf_printf(out, "[resource %s]", json_str(json_get(rs, "uri"), ""));
        } else if (!strcmp(type, "resource_link")) {
            buf_printf(out, "[link %s]", json_str(json_get(c, "uri"), ""));
        } else {
            buf_printf(out, "[%s content]", type);
        }
    }
    if (!content) {
        const Json *sc = json_get(res, "structuredContent");
        if (sc) json_write(out, sc);
    }
    if (!out->s) buf_add(out, "(no output)", -1);
    json_free(r);
    return !is_err;
}

static void load_mcp_config(Tools *T, const char *path) {
    FILE *f = file_open(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    n = (long)fread(s, 1, (size_t)n, f);
    fclose(f);
    char err[160];
    Json *cfg = json_parse(s, (int)n, err, sizeof(err));
    free(s);
    if (!cfg) { fprintf(stderr, "MCP: %s is not valid JSON (%s)\n", path, err); return; }
    const Json *servers = json_get(cfg, "mcpServers");
    if (!servers) servers = json_get(cfg, "servers");
    for (const Json *e = servers ? servers->child : NULL; e; e = e->next) {
        const char *cmd = json_str(json_get(e, "command"), NULL);
        if (json_bool(json_get(e, "disabled"), 0)) continue;
        if (!cmd) {
            fprintf(stderr, "MCP %s: only stdio servers (\"command\") are supported\n", e->key);
            continue;
        }
        Mcp *m = (Mcp *)calloc(1, sizeof(Mcp));
        m->owner = T;
        snprintf(m->name, sizeof(m->name), "%s", e->key);
        snprintf(m->status, sizeof(m->status), "starting");
        const Json *args = json_get(e, "args");
        int na = args && args->t == J_ARR ? args->n : 0;
        m->argv = (char **)calloc((size_t)na + 2, sizeof(char *));
        m->argv[0] = dupstr(cmd);
        int k = 1;
        for (const Json *a = na ? args->child : NULL; a; a = a->next)
            if (a->t == J_STR) m->argv[k++] = dupstr(a->s);
        const Json *env = json_get(e, "env");
        int ne = env && env->t == J_OBJ ? env->n : 0;
        m->env = (char **)calloc((size_t)ne + 1, sizeof(char *));
        k = 0;
        for (const Json *v = ne ? env->child : NULL; v; v = v->next) {
            if (v->t != J_STR) continue;
            size_t l = strlen(v->key) + (size_t)v->len + 2;
            m->env[k] = (char *)malloc(l);
            snprintf(m->env[k++], l, "%s=%s", v->key, v->s);
        }
        m->mx = lock_create();
        m->req = lock_create();
        m->wr = lock_create();
        T->srv = (Mcp **)realloc(T->srv, sizeof(Mcp *) * (size_t)(T->nsrv + 1));
        T->srv[T->nsrv++] = m;
        m->starter = thread_create(mcp_start, m);
    }
    json_free(cfg);
}

/* ------------------------------------------------------------------ public API */

Tools *tools_create(const ToolsConfig *cfg) {
    Tools *T = (Tools *)calloc(1, sizeof(Tools));
    T->cfg = *cfg;
    T->lock = lock_create();
    if (cfg->web) {
        const char *argv[] = { "curl", "--version", NULL };
        char *o = NULL;
        T->has_curl = proc_run(argv, &o, NULL, 4096, 10000) == 0;
        free(o);
        if (!T->has_curl) fprintf(stderr, "web tools disabled: curl not found on the PATH\n");
    }
    if (cfg->workspace && *cfg->workspace) {
        if (dir_create(cfg->workspace)) {
            snprintf(T->workspace, sizeof(T->workspace), "%s", cfg->workspace);
            for (char *p = T->workspace; *p; p++) if (*p == '\\') *p = '/';
            size_t l = strlen(T->workspace);
            while (l > 1 && T->workspace[l - 1] == '/') T->workspace[--l] = 0;
        } else {
            fprintf(stderr, "workspace %s cannot be created: file tools disabled\n", cfg->workspace);
        }
    }
    for (int i = 0; i < N_BUILTINS; i++) {
        int id = BUILTINS[i].id;
        if ((id == B_SEARCH || id == B_FETCH) && !T->has_curl) continue;
        if (id >= B_LS && id <= B_EDIT && !T->workspace[0]) continue;
        if ((id == B_WRITE || id == B_EDIT) && !cfg->allow_write) continue;
        if (id >= B_RUN && !cfg->shell) continue;
        Tool *t = add_tool(T);
        snprintf(t->name, sizeof(t->name), "%s", BUILTINS[i].name);
        if (id == B_RUN) {
            Buf d = { 0 };
            buf_add(&d, "Run a shell command and wait for it to end (or for the timeout). Returns the exit code, the run "
                        "time, and stdout and stderr separately. ", -1);
#if defined(_WIN32)
            buf_add(&d, "This computer runs Windows: commands use Windows PowerShell 5.1 syntax by default "
                        "(shell=\"cmd\" for cmd.exe).", -1);
#else
            buf_add(&d, "Commands use bash syntax (sh if bash is missing).", -1);
#endif
            buf_add(&d, " Programs that keep running (servers, watchers) need start_process instead.", -1);
            t->desc = d.s;
        } else {
            t->desc = dupstr(BUILTINS[i].desc);
        }
        t->params = json_parse(BUILTINS[i].params, (int)strlen(BUILTINS[i].params), NULL, 0);
        t->confirm = BUILTINS[i].confirm;
        t->builtin = id;
    }
    if (cfg->mcp_config && *cfg->mcp_config) load_mcp_config(T, cfg->mcp_config);
    if (cfg->shell) build_env_info(T); /* now, not lazily: server threads read it */
    return T;
}

void tools_free(Tools *T) {
    if (!T) return;
    for (int i = 0; i < T->nsrv; i++) {
        Mcp *m = T->srv[i];
        thread_join(m->starter);
        if (m->p) proc_stop(m->p, 1500); /* the reader then sees end of output and exits */
        thread_join(m->reader);
        if (m->p) proc_free(m->p);
        for (char **a = m->argv; *a; a++) free(*a);
        for (char **e = m->env; *e; e++) free(*e);
        free(m->argv);
        free(m->env);
        json_free(m->resp);
        lock_destroy(m->mx); lock_destroy(m->req); lock_destroy(m->wr);
        free(m);
    }
    free(T->srv);
    for (int i = 0; i < T->njobs; i++) { /* background processes end with us */
        proc_kill(T->jobs[i]->p);
        job_release(T->jobs[i]);
    }
    for (int i = 0; i < T->n; i++) { free(T->t[i].desc); free(T->t[i].remote); json_free(T->t[i].params); }
    free(T->t);
    free(T->env_info);
    lock_destroy(T->lock);
    free(T);
}

int tools_confirm(Tools *T, const char *name) {
    int c = 1; /* unknown tools ask */
    lock_acquire(T->lock);
    for (int i = 0; i < T->n; i++)
        if (!strcmp(T->t[i].name, name)) { c = T->t[i].confirm; break; }
    lock_release(T->lock);
    return c;
}

int tools_count(Tools *T) {
    lock_acquire(T->lock);
    int n = T->n;
    lock_release(T->lock);
    return n;
}

void tools_list_json(Tools *T, Buf *b) {
    lock_acquire(T->lock);
    buf_add(b, "{\"tools\":[", -1);
    for (int i = 0; i < T->n; i++) {
        Tool *t = &T->t[i];
        if (i) buf_add(b, ",", 1);
        buf_add(b, "{\"type\":\"function\",\"function\":{\"name\":", -1);
        buf_json_str(b, t->name, -1);
        buf_add(b, ",\"description\":", -1);
        buf_json_str(b, t->desc, -1);
        buf_add(b, ",\"parameters\":", -1);
        if (t->params) json_write(b, t->params);
        else buf_add(b, "{}", 2);
        buf_add(b, "},\"source\":", -1);
        if (t->srv) {
            char src[96];
            snprintf(src, sizeof(src), "mcp:%s", t->srv->name);
            buf_json_str(b, src, -1);
        } else {
            buf_add(b, "\"builtin\"", -1);
        }
        buf_printf(b, ",\"confirm\":%s}", t->confirm ? "true" : "false");
    }
    buf_add(b, "],\"servers\":[", -1);
    for (int i = 0; i < T->nsrv; i++) {
        Mcp *m = T->srv[i];
        if (i) buf_add(b, ",", 1);
        buf_add(b, "{\"name\":", -1);
        buf_json_str(b, m->name, -1);
        buf_add(b, ",\"status\":", -1);
        buf_json_str(b, m->status, -1);
        buf_printf(b, ",\"tools\":%d,\"error\":", m->ntools);
        buf_json_str(b, m->error, -1);
        buf_add(b, "}", 1);
    }
    buf_add(b, "],\"workspace\":", -1);
    buf_json_str(b, T->workspace, -1);
    buf_printf(b, ",\"web\":%s", T->has_curl ? "true" : "false");
    lock_release(T->lock);
    if (T->cfg.shell) { /* the system and shell rules the model needs to write commands */
        buf_add(b, ",\"env\":", -1);
        buf_json_str(b, tools_env_info(T), -1);
    }
    buf_add(b, "}", 1);
}

int tools_call(Tools *T, const char *name, const Json *args, Buf *out) {
    lock_acquire(T->lock);
    int builtin = -2;
    Mcp *srv = NULL;
    char remote[256] = "";
    for (int i = 0; i < T->n; i++)
        if (!strcmp(T->t[i].name, name)) {
            builtin = T->t[i].builtin;
            srv = T->t[i].srv;
            if (T->t[i].remote) snprintf(remote, sizeof(remote), "%s", T->t[i].remote);
            break;
        }
    lock_release(T->lock);
    int ok;
    switch (builtin) {
    case -2: buf_printf(out, "error: unknown tool '%s'", name); ok = 0; break;
    case -1: ok = mcp_call(srv, remote, args, out); break;
    case B_SEARCH: ok = tool_search(args, out); break;
    case B_FETCH: ok = tool_fetch(args, out); break;
    case B_CALC: ok = tool_calc(args, out); break;
    case B_TIME: ok = tool_time(out); break;
    case B_LS: ok = tool_ls(T, args, out); break;
    case B_READ: ok = tool_read(T, args, out); break;
    case B_FIND: ok = tool_find(T, args, out, 0); break;
    case B_GREP: ok = tool_find(T, args, out, 1); break;
    case B_WRITE: ok = tool_write(T, args, out); break;
    case B_EDIT: ok = tool_edit(T, args, out); break;
    case B_RUN: ok = tool_run(T, args, out); break;
    case B_START: ok = tool_start(T, args, out); break;
    case B_READP: ok = tool_readp(T, args, out); break;
    case B_STOPP: ok = tool_stopp(T, args, out); break;
    case B_LISTP: ok = tool_listp(T, out); break;
    default: buf_add(out, "error: tool not available", -1); ok = 0;
    }
    if (!out->s) buf_add(out, "", 0);
    cap_result(out);
    return ok;
}
