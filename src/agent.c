/* agent.c - see agent.h. Each step sends the whole conversation to the in-process server
 * (chat template, tool-call parsing and prompt reuse are the server's), streams the answer to
 * the terminal, runs the tool calls (asking the user first when the mode requires it) and
 * feeds the results back, until the model answers without calling tools. */
#include "agent.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "json.h"
#include "platform.h"
#include "server.h"

#define RESULT_MAX 7000     /* characters of a tool result kept in the conversation */
#define FAIL_REPEAT 3       /* the same failing call this many times: tell the model to change approach */

#define DIM "\x1b[2m"
#define BOLD "\x1b[1m"
#define RESET "\x1b[0m"
#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define YELLOW "\x1b[33m"
#define CYAN "\x1b[36m"

typedef struct {
    char role[12];
    char *content;
    char *calls;            /* assistant: JSON array of tool calls, or NULL */
    char id[64];            /* tool: the call it answers */
} Msg;

static const AgentConfig *C;
static volatile int g_stop;  /* Ctrl+C: stop the generation and the running tool */
static int g_auto;
static Msg *M;
static int nm, cap_m;
static char *g_system, *g_tools;
static char g_always[32][64];
static int n_always;
static char g_fail_sig[512];
static int g_fail_count;

static void on_ctrl_c(void) {
    g_stop = 1;
    if (C && C->tools) tools_set_interrupt(C->tools, 1);
}

/* ------------------------------------------------------------------ terminal */

static void out(const char *s) { fputs(s, stdout); fflush(stdout); }
static void outn(const char *s, int n) { fwrite(s, 1, (size_t)n, stdout); fflush(stdout); }
static void col(const char *code) { if (C->color) out(code); }
static void status_clear(void) { if (C->color) out("\r\x1b[2K"); }

/* Up to max_lines lines of text, indented and dimmed. */
static void preview(const char *s, int max_lines, int max_cols) {
    col(DIM);
    int lines = 0;
    while (*s && lines < max_lines) {
        const char *e = strchr(s, '\n');
        int l = e ? (int)(e - s) : (int)strlen(s);
        int show = l > max_cols ? max_cols : l;
        while (show > 0 && ((unsigned char)s[show] & 0xC0) == 0x80 && show < l) show--;
        out("      ");
        outn(s, show);
        if (show < l) out("...");
        out("\n");
        lines++;
        if (!e) break;
        s = e + 1;
    }
    if (*s && lines == max_lines) {
        int more = 0;
        for (const char *p = s; *p; p++) more += *p == '\n';
        if (more > 0) {
            char m[64];
            snprintf(m, sizeof(m), "      ... (%d more lines)\n", more + 1);
            out(m);
        }
    }
    col(RESET);
}

/* ------------------------------------------------------------------ conversation */

static Msg *msg_new(const char *role) {
    if (nm == cap_m) {
        cap_m = cap_m ? cap_m * 2 : 32;
        M = (Msg *)realloc(M, sizeof(Msg) * (size_t)cap_m);
    }
    Msg *m = &M[nm++];
    memset(m, 0, sizeof(*m));
    snprintf(m->role, sizeof(m->role), "%s", role);
    return m;
}

static char *dup_n(const char *s, int n) {
    if (n < 0) n = (int)strlen(s);
    char *r = (char *)malloc((size_t)n + 1);
    memcpy(r, s, (size_t)n);
    r[n] = 0;
    return r;
}

static void msgs_clear(void) {
    for (int i = 0; i < nm; i++) { free(M[i].content); free(M[i].calls); }
    nm = 0;
}

/* Long tool results keep their start and, mostly, their end (errors and results come last). */
static char *clip_result(const char *s) {
    int n = (int)strlen(s);
    if (n <= RESULT_MAX) return dup_n(s, n);
    int head = RESULT_MAX / 4, from = n - (RESULT_MAX - head);
    while (head > 0 && ((unsigned char)s[head] & 0xC0) == 0x80) head--;
    while (from < n && ((unsigned char)s[from] & 0xC0) == 0x80) from++;
    Buf b = { 0 };
    buf_add(&b, s, head);
    buf_printf(&b, "\n[... %d characters omitted ...]\n", from - head);
    buf_add(&b, s + from, n - from);
    return b.s;
}

/* Frees context: old tool results shrink to a stub. Returns 1 if something was removed. */
static int compact(void) {
    int changed = 0, last_tools = 0;
    for (int i = nm - 1; i >= 0; i--) {
        if (strcmp(M[i].role, "tool")) continue;
        if (++last_tools <= 2) continue; /* keep the newest results whole */
        if (strlen(M[i].content) <= 300) continue;
        char *s = (char *)malloc(400);
        int k = 200;
        while (k > 0 && ((unsigned char)M[i].content[k] & 0xC0) == 0x80) k--;
        snprintf(s, 400, "%.*s\n[... rest of the output removed to save context: run the tool again if needed]", k,
                 M[i].content);
        free(M[i].content);
        M[i].content = s;
        changed = 1;
    }
    return changed;
}

static void build_request(Buf *b) {
    buf_add(b, "{\"messages\":[{\"role\":\"system\",\"content\":", -1);
    buf_json_str(b, g_system, -1);
    buf_add(b, "}", 1);
    for (int i = 0; i < nm; i++) {
        Msg *m = &M[i];
        buf_add(b, ",{\"role\":", -1);
        buf_json_str(b, m->role, -1);
        buf_add(b, ",\"content\":", -1);
        buf_json_str(b, m->content ? m->content : "", -1);
        if (m->calls) {
            buf_add(b, ",\"tool_calls\":", -1);
            buf_add(b, m->calls, -1);
        }
        if (m->id[0]) {
            buf_add(b, ",\"tool_call_id\":", -1);
            buf_json_str(b, m->id, -1);
        }
        buf_add(b, "}", 1);
    }
    buf_add(b, "],\"tools\":", -1);
    buf_add(b, g_tools, -1);
    buf_add(b, ",\"stream\":true,\"return_progress\":true", -1);
    if (C->max_tokens > 0) buf_printf(b, ",\"max_tokens\":%d", C->max_tokens);
    buf_add(b, "}", 1);
}

/* ------------------------------------------------------------------ one model call */

typedef struct {
    Buf raw, body, content, reasoning, calls;
    int head_done, status, ncalls;
    int think_n, content_started, think_started, progress_shown;
    char finish[24], error[400];
    int prompt_n, cache_n, predicted_n;
    double prompt_ms, predicted_ms;
} Stream;

static void on_event(Stream *s, const char *data, int n) {
    if (n == 6 && !memcmp(data, "[DONE]", 6)) return;
    Json *j = json_parse(data, n, NULL, 0);
    if (!j) return;
    const Json *e = json_get(j, "error");
    if (e) snprintf(s->error, sizeof(s->error), "%s", json_str(json_get(e, "message"), "error"));
    const Json *pp = json_get(j, "prompt_progress");
    if (pp && C->color) {
        int total = (int)json_num(json_get(pp, "total"), 0), cache = (int)json_num(json_get(pp, "cache"), 0);
        int done = (int)json_num(json_get(pp, "processed"), 0);
        if (total - cache > 64 && done < total) {
            char m[160];
            snprintf(m, sizeof(m), "\r\x1b[2K" DIM "  reading %d new tokens of context... %d%%" RESET, total - cache,
                     (int)(100.0 * (done - cache) / (total - cache)));
            out(m);
            s->progress_shown = 1;
        }
    }
    const Json *tm = json_get(j, "timings");
    if (tm) {
        s->prompt_n = (int)json_num(json_get(tm, "prompt_n"), 0);
        s->cache_n = (int)json_num(json_get(tm, "cache_n"), 0);
        s->predicted_n = (int)json_num(json_get(tm, "predicted_n"), 0);
        s->prompt_ms = json_num(json_get(tm, "prompt_ms"), 0);
        s->predicted_ms = json_num(json_get(tm, "predicted_ms"), 0);
    }
    const Json *ch = json_get(j, "choices");
    ch = ch && ch->t == J_ARR ? ch->child : NULL;
    if (ch) {
        const char *fr = json_str(json_get(ch, "finish_reason"), NULL);
        if (fr) snprintf(s->finish, sizeof(s->finish), "%s", fr);
        const Json *d = json_get(ch, "delta");
        const Json *r = json_get(d, "reasoning_content");
        if (r && r->t == J_STR && r->len) {
            buf_add(&s->reasoning, r->s, r->len);
            s->think_n++;
            if (C->show_think) {
                if (!s->think_started) {
                    status_clear();
                    col(DIM);
                    s->think_started = 1;
                }
                outn(r->s, r->len);
            } else if (C->color && (s->think_n % 4) == 1) {
                char m[96];
                snprintf(m, sizeof(m), "\r\x1b[2K" DIM "  thinking... %d" RESET, s->think_n);
                out(m);
            }
        }
        const Json *c = json_get(d, "content");
        if (c && c->t == J_STR && c->len) {
            if (!s->content_started) {
                if (s->think_started) { col(RESET); out("\n\n"); }
                else status_clear();
                s->content_started = 1;
            }
            buf_add(&s->content, c->s, c->len);
            outn(c->s, c->len);
        }
        const Json *tc = json_get(d, "tool_calls");
        for (const Json *t = tc && tc->t == J_ARR ? tc->child : NULL; t; t = t->next) {
            if (s->ncalls++) buf_add(&s->calls, ",", 1);
            json_write(&s->calls, t);
        }
    }
    json_free(j);
}

static void sink(void *u, const char *p, int n) {
    Stream *s = (Stream *)u;
    buf_add(&s->raw, p, n);
    if (!s->head_done) {
        char *e = strstr(s->raw.s, "\r\n\r\n");
        if (!e) return;
        s->status = s->raw.n > 12 ? atoi(s->raw.s + 9) : 0;
        int used = (int)(e + 4 - s->raw.s);
        memmove(s->raw.s, s->raw.s + used, (size_t)(s->raw.n - used + 1));
        s->raw.n -= used;
        s->head_done = 1;
    }
    if (s->status != 200) {
        buf_add(&s->body, s->raw.s, s->raw.n);
        s->raw.n = 0;
        return;
    }
    for (;;) {
        char *e = strstr(s->raw.s, "\n\n");
        if (!e) break;
        *e = 0;
        for (char *line = s->raw.s; line && *line;) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = 0;
            if (!strncmp(line, "data: ", 6)) on_event(s, line + 6, (int)strlen(line + 6));
            line = nl ? nl + 1 : NULL;
        }
        int used = (int)(e + 2 - s->raw.s);
        memmove(s->raw.s, s->raw.s + used, (size_t)(s->raw.n - used + 1));
        s->raw.n -= used;
    }
}

static void stream_free(Stream *s) {
    buf_free(&s->raw);
    buf_free(&s->body);
    buf_free(&s->content);
    buf_free(&s->reasoning);
    buf_free(&s->calls);
}

/* Returns 1 on an answer, 0 on an error (message in s->error), -1 if interrupted. */
static int model_call(Stream *s) {
    memset(s, 0, sizeof(*s));
    Buf req = { 0 };
    build_request(&req);
    if (C->color) out(DIM "  ..." RESET);
    server_local_chat(req.s, req.n, sink, s, &g_stop);
    buf_free(&req);
    if (C->show_think && s->think_started && !s->content_started) col(RESET);
    if (!s->content_started) status_clear();
    else out("\n");
    if (s->status != 200 && !s->error[0]) {
        Json *j = s->body.s ? json_parse(s->body.s, s->body.n, NULL, 0) : NULL;
        snprintf(s->error, sizeof(s->error), "%s",
                 json_str(json_get(json_get(j, "error"), "message"), s->status ? "request failed" : "no answer"));
        json_free(j);
    }
    if (g_stop) return -1;
    if (s->error[0]) return 0;
    char m[200];
    snprintf(m, sizeof(m), DIM "  [%.1f s | context: %d tokens read, %d reused | answer: %d tokens, %.1f tok/s]\n" RESET,
             (s->prompt_ms + s->predicted_ms) / 1e3, s->prompt_n, s->cache_n, s->predicted_n,
             s->predicted_ms > 0 ? s->predicted_n * 1e3 / s->predicted_ms : 0.0);
    if (C->color) out(m);
    else {
        /* the same without escapes */
        snprintf(m, sizeof(m), "  [%.1f s | context: %d tokens read, %d reused | answer: %d tokens]\n",
                 (s->prompt_ms + s->predicted_ms) / 1e3, s->prompt_n, s->cache_n, s->predicted_n);
        out(m);
    }
    return 1;
}

/* ------------------------------------------------------------------ tool calls */

static void show_call(const char *name, const Json *args) {
    col(CYAN);
    out("  * ");
    out(name);
    col(RESET);
    const char *cmd = json_str(json_get(args, "command"), NULL);
    const char *path = json_str(json_get(args, "path"), NULL);
    if (cmd) {
        const char *shell = json_str(json_get(args, "shell"), NULL), *cwd = json_str(json_get(args, "cwd"), NULL);
        if (shell || cwd) {
            col(DIM);
            out("  (");
            if (shell) out(shell);
            if (shell && cwd) out(", ");
            if (cwd) { out("in "); out(cwd); }
            out(")");
            col(RESET);
        }
        out("\n");
        col(BOLD);
        const char *p = cmd;
        while (*p) {
            const char *e = strchr(p, '\n');
            out("    $ ");
            outn(p, e ? (int)(e - p) : (int)strlen(p));
            out("\n");
            if (!e) break;
            p = e + 1;
        }
        col(RESET);
    } else if (path && !strcmp(name, "write_file")) {
        const char *content = json_str(json_get(args, "content"), "");
        char m[1200];
        snprintf(m, sizeof(m), "  %s (%d bytes)\n", path, (int)strlen(content));
        out(m);
        preview(content, 8, 160);
    } else if (path && !strcmp(name, "edit_file")) {
        out("  ");
        out(path);
        out("\n");
        col(RED);
        out("    - replace:\n");
        col(RESET);
        preview(json_str(json_get(args, "old_text"), ""), 6, 160);
        col(GREEN);
        out("    + with:\n");
        col(RESET);
        preview(json_str(json_get(args, "new_text"), ""), 6, 160);
    } else {
        Buf b = { 0 };
        if (args) json_write(&b, args);
        if (b.n > 2) {
            out("  ");
            col(DIM);
            outn(b.s, b.n > 300 ? 300 : b.n);
            if (b.n > 300) out("...");
            col(RESET);
        }
        out("\n");
        buf_free(&b);
    }
}

static int is_always(const char *name) {
    for (int i = 0; i < n_always; i++)
        if (!strcmp(g_always[i], name)) return 1;
    return 0;
}

/* Asks the user. Returns 1 to run the call; 0 to skip it, with the user's words (if any) in why. */
static int approve(const char *name, const char *danger, Buf *why) {
    if (danger) {
        col(YELLOW);
        out("    ! careful: this command ");
        out(danger);
        out("\n");
        col(RESET);
    }
    char line[8192];
    for (;;) {
        col(YELLOW);
        out("    run it? [y]es / [n]o");
        if (!danger) { out(" / [a]lways "); out(name); }
        out(" / or type what to do instead: ");
        col(RESET);
        if (!console_read_line(line, sizeof(line))) {
            out("\n");
            if (g_stop) return 0;
            buf_add(why, "no answer from the user (end of input)", -1);
            return 0;
        }
        char *s = line;
        while (isspace((unsigned char)*s)) s++;
        size_t l = strlen(s);
        while (l > 0 && isspace((unsigned char)s[l - 1])) s[--l] = 0;
        if (!l) continue;
        char low[16];
        size_t k;
        for (k = 0; k + 1 < sizeof(low) && s[k]; k++) low[k] = (char)tolower((unsigned char)s[k]);
        low[k] = 0;
        if (!strcmp(low, "y") || !strcmp(low, "yes") || !strcmp(low, "s") || !strcmp(low, "si") || !strcmp(low, "s\xc3\xac") ||
            !strcmp(low, "ok"))
            return 1;
        if (!strcmp(low, "n") || !strcmp(low, "no")) return 0;
        if (!danger && (!strcmp(low, "a") || !strcmp(low, "always") || !strcmp(low, "sempre"))) {
            if (n_always < 32) snprintf(g_always[n_always++], sizeof(g_always[0]), "%s", name);
            return 1;
        }
        buf_add(why, s, (int)l);
        return 0;
    }
}

/* Runs (or skips) one call and appends its result. Returns 0 if the user interrupted. */
static int do_call(const Json *call) {
    const char *id = json_str(json_get(call, "id"), "");
    const Json *fn = json_get(call, "function");
    const char *name = json_str(json_get(fn, "name"), "?");
    const Json *a = json_get(fn, "arguments");
    Json *args = a && a->t == J_STR ? json_parse(a->s, a->len, NULL, 0) : NULL;
    show_call(name, args);

    const char *danger = tools_danger(name, args);
    int ask = g_auto ? (danger && !C->yolo) : (danger || !is_always(name));
    Buf why = { 0 }, res = { 0 };
    int run = 1, ok = 0;
    if (ask) run = approve(name, danger, &why);
    double t0 = time_sec();
    if (g_stop) {
        buf_add(&res, "not run: the user interrupted the task", -1);
    } else if (!run) {
        if (why.n) {
            buf_add(&res, "The user did not allow this call and said: ", -1);
            buf_add(&res, why.s, why.n);
        } else {
            buf_add(&res, "The user did not allow this call. Do not try to get around it: ask what to do instead.", -1);
        }
        col(DIM);
        out("    skipped\n");
        col(RESET);
    } else {
        tools_set_interrupt(C->tools, 0);
        ok = tools_call(C->tools, name, args, &res);
        double dt = time_sec() - t0;
        char m[96];
        snprintf(m, sizeof(m), "    %s | %.2f s\n", ok ? "ok" : (g_stop ? "interrupted" : "error"), dt);
        col(ok ? GREEN : RED);
        out(m);
        col(RESET);
        preview(res.s ? res.s : "", ok ? 4 : 10, 200);
    }
    /* the same call failing again and again */
    Buf sig = { 0 };
    buf_add(&sig, name, -1);
    if (args) json_write(&sig, args);
    if (run && !ok && !g_stop) {
        if (sig.n < (int)sizeof(g_fail_sig) && !strcmp(sig.s, g_fail_sig)) g_fail_count++;
        else {
            snprintf(g_fail_sig, sizeof(g_fail_sig), "%s", sig.n < (int)sizeof(g_fail_sig) ? sig.s : "");
            g_fail_count = 1;
        }
        if (g_fail_count >= FAIL_REPEAT)
            buf_printf(&res, "\n[note: this exact call has failed %d times. Do not repeat it: find the cause, change "
                             "approach, or ask the user]", g_fail_count);
    } else if (ok) {
        g_fail_sig[0] = 0;
        g_fail_count = 0;
    }
    buf_free(&sig);
    Msg *m = msg_new("tool");
    m->content = clip_result(res.s ? res.s : "");
    snprintf(m->id, sizeof(m->id), "%s", id);
    buf_free(&why);
    buf_free(&res);
    json_free(args);
    return !g_stop;
}

/* ------------------------------------------------------------------ tasks */

static const char *mode_note(void) {
    return g_auto ? "\n\n[Autonomous mode: work on your own until the task is done and verified; do not ask for "
                    "confirmation of each step. Ask only if the goal is unclear or you are stuck.]"
                  : "\n\n[Safe mode: the user sees and approves each tool call.]";
}

/* Returns 0 when the model finished, 1 on interruption or error, 2 at the step limit. */
static int run_task(const char *task) {
    Msg *u = msg_new("user");
    Buf t = { 0 };
    buf_add(&t, task, -1);
    buf_add(&t, mode_note(), -1);
    u->content = t.s;
    g_fail_sig[0] = 0;
    g_fail_count = 0;
    for (int step = 0, retried = 0;; step++) {
        if (step >= C->max_steps) {
            char m[160];
            snprintf(m, sizeof(m), "\n[stopped after %d steps: write 'continue' to go on, or give new instructions]\n",
                     C->max_steps);
            col(YELLOW);
            out(m);
            col(RESET);
            return 2;
        }
        g_stop = 0;
        tools_set_interrupt(C->tools, 0);
        Stream s;
        int r = model_call(&s);
        if (r < 0) {
            stream_free(&s);
            col(YELLOW);
            out("\n[interrupted]\n");
            col(RESET);
            return 1;
        }
        if (r == 0) {
            int too_long = strstr(s.error, "context size") != NULL;
            if (too_long && !retried && compact()) {
                col(DIM);
                out("  [context full: old tool outputs shortened, retrying]\n");
                col(RESET);
                stream_free(&s);
                retried = 1;
                step--;
                continue;
            }
            col(RED);
            out("\n[error] ");
            out(s.error);
            if (too_long) out(" (start over with /reset, or a larger context with -c)");
            out("\n");
            col(RESET);
            stream_free(&s);
            return 1;
        }
        retried = 0;
        Msg *a = msg_new("assistant");
        a->content = dup_n(s.content.s ? s.content.s : "", s.content.n);
        if (s.ncalls) {
            Buf arr = { 0 };
            buf_add(&arr, "[", 1);
            buf_add(&arr, s.calls.s, s.calls.n);
            buf_add(&arr, "]", 1);
            a->calls = arr.s;
        }
        int ncalls = s.ncalls;
        if (!strcmp(s.finish, "length") && !ncalls) {
            col(YELLOW);
            out("[the answer reached the token limit]\n");
            col(RESET);
        }
        stream_free(&s);
        if (!ncalls) return 0;
        Json *calls = json_parse(a->calls, (int)strlen(a->calls), NULL, 0);
        int go = 1;
        for (const Json *c = calls ? calls->child : NULL; c; c = c->next) {
            if (!go) { /* every call needs a result: the rest are skipped */
                Msg *m = msg_new("tool");
                m->content = dup_n("not run: the user interrupted the task", -1);
                snprintf(m->id, sizeof(m->id), "%s", json_str(json_get(c, "id"), ""));
                continue;
            }
            go = do_call(c);
        }
        json_free(calls);
        if (!go) {
            col(YELLOW);
            out("\n[interrupted: tell me how to continue]\n");
            col(RESET);
            return 1;
        }
    }
}

static void build_system(void) {
    char date[64];
    time_t now = time(NULL);
    strftime(date, sizeof(date), "%A, %B %d, %Y", localtime(&now));
    Buf b = { 0 };
    buf_printf(&b,
               "You are the Pithos agent: you work in a terminal on the user's computer and act through tools. You run "
               "commands, read and change files, follow background processes and search the web. Today is %s.\n\n",
               date);
    buf_add(&b, tools_env_info(C->tools), -1);
    buf_add(&b,
            "\nHow to work:\n"
            "- Check the facts first (files, versions, what is installed), then act step by step.\n"
            "- After every command read the exit code, the output and the time it took: never assume that a command worked.\n"
            "- When something fails, read the error, find the cause, fix it and try again with a corrected command. "
            "Do not repeat a failing command unchanged; after three failed attempts at the same problem, change "
            "approach or ask the user.\n"
            "- Use edit_file for changes to existing files (read them first) and write_file for new files.\n"
            "- Before saying that the task is done, verify the result: run the program or the tests, read the file back.\n"
            "- Several independent tool calls can go in the same answer.\n"
            "- Be brief. At the end, summarize what you did and how you verified it.\n"
            "- Always write your answers in the language of the user's request (Italian if the user writes in Italian).",
            -1);
    if (C->system && *C->system) {
        buf_add(&b, "\n\n", 2);
        buf_add(&b, C->system, -1);
    }
    g_system = b.s;
}

static void build_tools(void) {
    /* MCP servers start in the background: give them time to list their tools */
    for (int i = 0; i < 200; i++) {
        Buf b = { 0 };
        tools_list_json(C->tools, &b);
        int starting = strstr(b.s, "\"status\":\"starting\"") != NULL;
        buf_free(&b);
        if (!starting) break;
        PithosLock *pause = lock_create();
        lock_acquire(pause);
        lock_wait_ms(pause, 100);
        lock_release(pause);
        lock_destroy(pause);
    }
    Buf b = { 0 }, o = { 0 };
    tools_list_json(C->tools, &b);
    Json *j = json_parse(b.s, b.n, NULL, 0);
    buf_add(&o, "[", 1);
    int n = 0;
    for (const Json *t = json_get(j, "tools") ? json_get(j, "tools")->child : NULL; t; t = t->next) {
        if (n++) buf_add(&o, ",", 1);
        buf_add(&o, "{\"type\":\"function\",\"function\":", -1);
        json_write(&o, json_get(t, "function"));
        buf_add(&o, "}", 1);
    }
    buf_add(&o, "]", 1);
    json_free(j);
    buf_free(&b);
    g_tools = o.s;
}

/* ------------------------------------------------------------------ start-up banner */

/* The Pandora mark (web/brand) at 32 x 28 pixels: two pixel rows per text row with half blocks. */
static const char *const LOGO[] = {
    "...............#######..........", ".............############.......",
    "...........###############......", "..........#################.....",
    ".........###################....", "........#####################...",
    "........######################..", ".......#######################..",
    "......#########################.", "......#######################.#.",
    "......#######################.#.", ".....########################..#",
    ".....########################..#", ".....###############....#####..#",
    "....###############......####..#", "....###############.......####.#",
    "....###############.......######", "...################.......######",
    "...################......######.", "..##################.....######.",
    ".##############################.", ".#############################..",
    "##############################..", "#############################...",
    "..##########################....", "..........#################.....",
    ".............############.......", "................#######.........",
};
#define LOGO_W 32
#define LOGO_H 28

/* The wordmark as readable source: '#' full block, other characters are box drawing (the shadow). */
static const char *const WORD_SRC[] = {
    "######b ##b########b##b  ##b ######b #######b",
    "##rqq##b##|aqq##rqqd##|  ##|##rqqq##b##rqqqqd",
    "######rd##|   ##|   #######|##|   ##|#######b",
    "##rqqqd ##|   ##|   ##rqq##|##|   ##|aqqqq##|",
    "##|     ##|   ##|   ##|  ##|a######rd#######|",
    "aqd     aqd   aqd   aqd  aqd aqqqqqd aqqqqqqd",
};
#define WORD_W 45

/* b ╗  r ╔  d ╝  a ╚  q ═  | ║ */
static const char *word_glyph(char c, int *shadow) {
    *shadow = 1;
    switch (c) {
    case '#': *shadow = 0; return "\xe2\x96\x88"; /* █ */
    case 'b': return "\xe2\x95\x97";              /* ╗ */
    case 'r': return "\xe2\x95\x94";              /* ╔ */
    case 'd': return "\xe2\x95\x9d";              /* ╝ */
    case 'a': return "\xe2\x95\x9a";              /* ╚ */
    case 'q': return "\xe2\x95\x90";              /* ═ */
    case '|': return "\xe2\x95\x91";              /* ║ */
    default: *shadow = 0; return " ";
    }
}

/* violet -> rose -> amber */
static void gradient(double t, int rgb[3]) {
    static const double st[3][4] = { { 0.0, 124, 92, 246 }, { 0.45, 236, 92, 140 }, { 1.0, 255, 196, 96 } };
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    int k = t <= st[1][0] ? 0 : 1;
    double u = (t - st[k][0]) / (st[k + 1][0] - st[k][0]);
    for (int i = 0; i < 3; i++) rgb[i] = (int)(st[k][i + 1] + (st[k + 1][i + 1] - st[k][i + 1]) * u);
}

static void put_fg(const int c[3], double scale) {
    char e[40];
    snprintf(e, sizeof(e), "\x1b[38;2;%d;%d;%dm", (int)(c[0] * scale), (int)(c[1] * scale), (int)(c[2] * scale));
    out(e);
}

/* One text column of the logo (two pixel rows). */
static void logo_cell(int row, int x, int total) {
    int top = LOGO[2 * row][x] == '#', bot = LOGO[2 * row + 1][x] == '#';
    if (!top && !bot) { out(" "); return; }
    int ct[3], cb[3];
    gradient((double)x / total * 0.85 + (2.0 * row) / LOGO_H * 0.15, ct);
    gradient((double)x / total * 0.85 + (2.0 * row + 1) / LOGO_H * 0.15, cb);
    if (top && bot) {
        char e[40];
        put_fg(ct, 1.0);
        snprintf(e, sizeof(e), "\x1b[48;2;%d;%d;%dm", cb[0], cb[1], cb[2]);
        out(e);
        out("\xe2\x96\x80\x1b[49m"); /* ▀ */
    } else {
        put_fg(top ? ct : cb, 1.0);
        out(top ? "\xe2\x96\x80" : "\xe2\x96\x84"); /* ▀ ▄ */
    }
}

static void word_row(int i, int x0, int total) {
    for (int j = 0; j < WORD_W; j++) {
        int shadow;
        const char *g = word_glyph(WORD_SRC[i][j], &shadow);
        if (*g != ' ') {
            int c[3];
            gradient((double)(x0 + j) / total * 0.85 + i / 6.0 * 0.15, c);
            put_fg(c, shadow ? 0.55 : 1.0);
        }
        out(g);
    }
}

/* Text cut to fit max columns (counting UTF-8 characters); keep_end: drop the start instead
 * (the informative part of a path is its end). */
static void out_fit(const char *s, int max, int keep_end) {
    int cols = 0;
    const char *p = s;
    while (*p && cols < max) {
        p++;
        while ((*p & 0xC0) == 0x80) p++;
        cols++;
    }
    if (*p && max > 3 && !keep_end) {
        const char *q = s;
        for (int k = 0; k < max - 3; k++) {
            q++;
            while ((*q & 0xC0) == 0x80) q++;
        }
        outn(s, (int)(q - s));
        out("...");
        return;
    }
    if (*p && max > 3) {
        int n = 0;
        for (const char *q = s; *q; q++) n += (*q & 0xC0) != 0x80;
        const char *q = s;
        for (int skip = n - (max - 3); skip > 0 && *q; skip--) {
            q++;
            while ((*q & 0xC0) == 0x80) q++;
        }
        out("...");
        out(q);
        return;
    }
    outn(s, (int)(p - s));
}

static void banner(const char *os, const char *folder) {
    const char *mode = g_auto ? (C->yolo ? "autonomous mode, without the safety checks"
                                         : "autonomous mode: destructive commands still ask")
                              : "safe mode: every action needs your approval";
    char title[64];
    snprintf(title, sizeof(title), "local AI agent  v%s", C->version ? C->version : "");
    if (!C->color) {
        printf("\nPithos agent v%s\n%s\n%s\n%s\n", C->version ? C->version : "", os, folder, mode);
        fflush(stdout);
        return;
    }
    const char *info[4] = { title, os, folder, mode };
    int cols = console_columns(), with_logo = cols >= LOGO_W + 2 + WORD_W + 1; /* 80 columns are enough */
    if (cols < WORD_W + 2) { /* too narrow for any art */
        printf("\n" BOLD "Pithos agent" RESET " v%s\n" DIM "%s\n%s\n" RESET "%s\n", C->version, os, folder, mode);
        fflush(stdout);
        return;
    }
    int x0 = with_logo ? LOGO_W + 2 : 0, total = x0 + WORD_W, text_rows = 6 + 1 + 4;
    int rows = with_logo ? LOGO_H / 2 : text_rows, top = with_logo ? (rows - text_rows) / 2 : 0;
    out("\n");
    for (int r = 0; r < rows; r++) {
        if (with_logo) {
            for (int x = 0; x < LOGO_W; x++) logo_cell(r, x, total);
            out(RESET "  ");
        }
        int t = r - top;
        if (t >= 0 && t < 6) word_row(t, x0, total);
        else if (t >= 7 && t < text_rows) {
            int k = t - 7;
            out(k == 0 ? BOLD : k == 3 ? (g_auto ? YELLOW : GREEN) : DIM);
            out_fit(info[k], cols - x0 - 1, k == 2);
        }
        out(RESET "\n");
    }
}

void agent_banner(const AgentConfig *cfg) {
    const AgentConfig *keep = C;
    int keep_auto = g_auto;
    C = cfg;
    g_auto = cfg->autonomous;
    char os[256], folder[1100];
    os_describe(os, sizeof(os));
    char *v = strstr(os, "(version "); /* shorter for the banner: "(version 10.0, build N)" -> "(build N)" */
    char *b = v ? strstr(v, "build ") : NULL;
    if (b) memmove(v + 1, b, strlen(b) + 1);
    if (!dir_current(folder, sizeof(folder))) snprintf(folder, sizeof(folder), ".");
    banner(os, folder);
    C = keep;
    g_auto = keep_auto;
}

static void show_mode(void) {
    col(BOLD);
    out(g_auto ? "mode: autonomous" : "mode: safe");
    col(RESET);
    out(g_auto ? (C->yolo ? " - works on its own, without asking even for destructive commands\n"
                          : " - works on its own; destructive commands still need your approval\n")
               : " - every action needs your approval\n");
}

static void show_help(void) {
    out("  /auto   autonomous mode       /safe   safe mode\n"
        "  /reset  new conversation      /tools  list the tools\n"
        "  /exit   quit                  Ctrl+C  stop the current step\n");
}

static void show_tools(void) {
    Json *j = json_parse(g_tools, (int)strlen(g_tools), NULL, 0);
    for (const Json *t = j ? j->child : NULL; t; t = t->next) {
        const char *name = json_str(json_get(json_get(t, "function"), "name"), "?");
        char m[200];
        snprintf(m, sizeof(m), "  %-24s%s\n", name, tools_confirm(C->tools, name) ? " (changes things)" : "");
        out(m);
    }
    json_free(j);
}

int agent_run(const AgentConfig *cfg) {
    C = cfg;
    g_auto = cfg->autonomous;
    console_on_interrupt(on_ctrl_c);
    build_tools();
    build_system();

    if (!cfg->banner_shown) agent_banner(cfg);
    C = cfg;

    int rc = 0;
    if (cfg->task) {
        out("\n> ");
        out(cfg->task);
        out("\n");
        rc = run_task(cfg->task);
    } else {
        out("Type a task, or /help.\n");
        static char line[65536];
        for (;;) {
            col(BOLD GREEN);
            out("\n> ");
            col(RESET);
            g_stop = 0;
            if (!console_read_line(line, sizeof(line))) {
                if (g_stop) { /* Ctrl+C at the prompt */
                    out("\n(type /exit to quit)\n");
                    continue;
                }
                break;
            }
            char *s = line;
            while (isspace((unsigned char)*s)) s++;
            if (!*s) continue;
            if (!strcmp(s, "/exit") || !strcmp(s, "/quit")) break;
            if (!strcmp(s, "/auto")) { g_auto = 1; show_mode(); continue; }
            if (!strcmp(s, "/safe")) { g_auto = 0; show_mode(); continue; }
            if (!strcmp(s, "/reset")) { msgs_clear(); n_always = 0; out("new conversation\n"); continue; }
            if (!strcmp(s, "/tools")) { show_tools(); continue; }
            if (!strcmp(s, "/help") || !strcmp(s, "/?")) { show_help(); continue; }
            rc = run_task(s);
        }
    }
    console_on_interrupt(NULL);
    msgs_clear();
    free(M);
    free(g_system);
    free(g_tools);
    return rc;
}
