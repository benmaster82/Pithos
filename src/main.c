/* main.c - pithos command line: chat, single prompt, raw completion, benchmark, tokenizer test. */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agent.h"
#include "gpu.h"
#include "model.h"
#include "pcache.h"
#include "platform.h"
#include "quant.h"
#include "sampler.h"
#include "server.h"
#include "tools.h"
#include "json.h"
#include "tokenizer.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#define VERSION "0.4.0"

typedef struct {
    const char *model, *prompt, *system, *kernels, *tokenize, *ppl;
    const char *host, *alias, *api_key, *cache_dir, *ui_path, *workspace, *mcp, *tool_name, *tool_args;
    int tools_list;
    int port, server, cache_mb, open_browser, tools, web, read_only, ctx_set, shell, workspace_set;
    int agent, agent_auto, yolo, max_steps, agent_think;
    int threads, n_ctx, n_batch, n_predict;
    int interactive, raw, bench, show_think, keep_think, prefetch, mlock, stats, verbose, ids;
    int gpu, gpu_device, gpu_mem, gpu_dense;
    int bench_pp, bench_tg;
    SamplerParams sp;
} Args;

static int g_color = 1;

/* ------------------------------------------------------------------ console */

static void console_init(void) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(h, &mode)) {
        if (!SetConsoleMode(h, mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */)) g_color = 0;
    } else {
        g_color = 0; /* redirected */
    }
#else
    g_color = isatty(1);
#endif
    const char *force = getenv("PITHOS_COLOR"); /* 1: colors even through a pipe, 0: never */
    if (force && *force) g_color = *force != '0';
}

/* Streams token bytes to stdout, holding back incomplete UTF-8 sequences. */
typedef struct { char pend[8]; int np; } Utf8Out;

static void out_bytes(Utf8Out *u, const char *s, int n) {
    char buf[1024];
    int bl = 0;
    for (int i = 0; i < n; i++) {
        u->pend[u->np++] = s[i];
        unsigned char c0 = (unsigned char)u->pend[0];
        int need = c0 < 0x80 ? 1 : (c0 >> 5) == 6 ? 2 : (c0 >> 4) == 14 ? 3 : (c0 >> 3) == 30 ? 4 : 1;
        if (u->np >= need || u->np >= 4) {
            if (bl + u->np > (int)sizeof(buf)) { fwrite(buf, 1, (size_t)bl, stdout); bl = 0; }
            memcpy(buf + bl, u->pend, (size_t)u->np);
            bl += u->np;
            u->np = 0;
        }
    }
    if (bl) fwrite(buf, 1, (size_t)bl, stdout);
    fflush(stdout);
}

static void out_str(const char *s) { fputs(s, stdout); fflush(stdout); }
static void color(const char *code) { if (g_color) out_str(code); }

/* ------------------------------------------------------------------ helpers */

typedef struct { char *s; int n, cap; } Str;

static void str_add(Str *b, const char *s, int n) {
    if (n < 0) n = (int)strlen(s);
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->s = (char *)realloc(b->s, (size_t)b->cap);
    }
    memcpy(b->s + b->n, s, (size_t)n);
    b->n += n;
    b->s[b->n] = 0;
}

static int *encode(const Tokenizer *tk, const char *text, int *n_out) {
    int len = (int)strlen(text);
    int cap = len + 16;
    int *ids = (int *)malloc(sizeof(int) * cap);
    int n = tok_encode(tk, text, len, 1, ids, cap);
    if (n > cap) {
        ids = (int *)realloc(ids, sizeof(int) * n);
        n = tok_encode(tk, text, len, 1, ids, n);
    }
    *n_out = n;
    return ids;
}

typedef struct { char path[16][512]; int n; const char *dir; } FoundModels;

static void collect_gguf(const char *name, void *u) {
    FoundModels *f = (FoundModels *)u;
    const size_t l = strlen(name);
    if (l <= 5 || strcmp(name + l - 5, ".gguf")) return; /* also skips unfinished *.gguf.part */
    if (f->n < 16) snprintf(f->path[f->n], sizeof(f->path[0]), "%s/%s", f->dir, name);
    f->n++;
}

/* Without -m: the reference model if it is there, otherwise the only GGUF in the model folders.
 * With several, or none, it says what it found and returns NULL. */
static const char *default_model_path(void) {
    static const char *dirs[] = { "models", "../models", "../../models" };
    static const char *ref = "LFM2.5-24B-A2B-Q4_K_M.gguf";
    static char path[512];
    const int nd = (int)(sizeof(dirs) / sizeof(dirs[0]));
    for (int i = 0; i < nd; i++) {
        snprintf(path, sizeof(path), "%s/%s", dirs[i], ref);
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); return path; }
    }
    static FoundModels found;
    found.n = 0;
    for (int i = 0; i < nd; i++) {
        found.dir = dirs[i];
        dir_list(dirs[i], collect_gguf, &found);
    }
    if (found.n == 1) {
        snprintf(path, sizeof(path), "%s", found.path[0]);
        fprintf(stderr, "no -m given: using the only model found, %s\n", path);
        return path;
    }
    if (found.n > 1) {
        qsort(found.path, (size_t)(found.n < 16 ? found.n : 16), sizeof(found.path[0]),
              (int (*)(const void *, const void *))strcmp); /* the same order on every OS */
        fprintf(stderr, "no -m given and %d models found; choose one:\n", found.n);
        for (int i = 0; i < found.n && i < 16; i++) fprintf(stderr, "  -m %s\n", found.path[i]);
        if (found.n > 16) fprintf(stderr, "  ...\n");
    } else {
        fprintf(stderr, "no model found in models/ (also looked in ../models and ../../models).\n"
                        "Download one (README: \"Downloading the model\"; on Linux: bash build.sh --list-models)\n"
                        "or pass its path with -m PATH.\n");
    }
    return NULL;
}

static void usage(void) {
    printf(
        "Pithos " VERSION " - dependency-free C inference engine for LFM2 / LFM2-MoE GGUF models\n\n"
        "usage: pithos [options]\n"
        "  -m, --model PATH        GGUF model (default: models/LFM2.5-24B-A2B-Q4_K_M.gguf, else the only\n"
        "                          .gguf in models/; also looked up in ../models and ../../models)\n"
        "  -p, --prompt TEXT       single prompt (chat template applied), then exit\n"
        "  -f, --file PATH         read the prompt from a file\n"
        "  -s, --system TEXT       system prompt\n"
        "  -i, --interactive       interactive chat (default when no prompt is given)\n"
        "      --raw               no chat template: plain text completion of the prompt\n"
        "  -n, --n-predict N       max new tokens per answer (default 4096; server: until the context is full)\n"
        "  -c, --ctx N             context size in tokens (default 8192, max 131072)\n"
        "  -t, --threads N         worker threads (default: physical cores)\n"
        "  -b, --batch N           prompt-processing batch (default 64)\n"
        "      --temp F            temperature (default 0.1, model card)\n"
        "      --top-k N           top-k (default 50)\n"
        "      --top-p F           top-p (default 1.0 = off)\n"
        "      --min-p F           min-p (default 0 = off)\n"
        "      --repeat-penalty F  repetition penalty (default 1.05)\n"
        "      --repeat-last-n N   penalty window (default 64)\n"
        "      --seed N            RNG seed (default 42)\n"
        "      --hide-think        do not print the reasoning section\n"
        "      --keep-think        keep past reasoning in context (template preserve_thinking)\n"
        "      --prefetch, --no-prefetch\n"
        "                          background read-ahead of experts (default: on if the model exceeds free RAM)\n"
        "      --mlock             try to lock the whole model in RAM\n"
        "      --no-gpu            CPU only (default: hot experts cached on a dedicated OpenCL GPU)\n"
        "      --gpu-device N      use OpenCL device N (see --list-gpus; integrated GPUs too)\n"
        "      --gpu-mem MB        VRAM budget (default: VRAM - 900 MB)\n"
        "      --no-gpu-dense      keep attention/convolution/output on the CPU (GPU only for experts)\n"
        "      --list-gpus         list OpenCL devices and exit\n"
        "      --kernels K         force kernels: auto | scalar\n"
        "      --bench [PP,TG]     benchmark prompt processing and generation (default 128,64)\n"
        "      --tokenize TEXT     print token ids for TEXT and exit\n"
        "      --ids               print generated token ids (for testing)\n"
        "      --ppl FILE          perplexity of FILE in chunks of -c tokens (llama-perplexity method)\n"
        "      --stats             print timing breakdown after each answer\n"
        "\nserver (OpenAI-compatible API + web chat):\n"
        "      --server            start the HTTP server instead of the terminal chat\n"
        "      --host ADDR         listen address (default 127.0.0.1; 0.0.0.0 = whole network)\n"
        "      --port N            port (default 8080)\n"
        "      --alias NAME        model id reported by the API (default: file name)\n"
        "      --api-key KEY       require Authorization: Bearer KEY on /v1\n"
        "      --cache-dir DIR     where conversations are kept for prompt reuse (default MODEL.pithos-cache)\n"
        "      --cache-mb N        disk budget for stored conversations (default 4096)\n"
        "      --no-cache          keep reused prompts in memory only\n"
        "      --ui FILE           serve this HTML file instead of the embedded interface\n"
        "      --open              open the web interface in the browser when ready\n"
        "\nterminal agent (works on tasks with shell commands, files, web and MCP tools):\n"
        "      --agent             interactive agent session in safe mode: every action needs your approval\n"
        "      --auto              autonomous mode: works on its own until the task is done (also /auto, /safe)\n"
        "      --yolo              with --auto, do not ask even for destructive commands (rm -rf, git push...)\n"
        "      --max-steps N       model calls per task before stopping to ask (default 40)\n"
        "      --show-think        stream the reasoning instead of a counter\n"
        "      -p / -f TASK        run one task and exit\n"
        "\nagent tools (web UI agent mode, /v1/tools):\n"
        "      --workspace DIR     folder the file tools can read and write (default ./workspace; agent: .)\n"
        "      --shell             also offer run_command and the process tools in the web UI\n"
        "      --mcp FILE          MCP servers to start, mcp.json format (default ./mcp.json if present)\n"
        "      --read-only         no write_file tool\n"
        "      --no-web            no web_search / fetch_url tools\n"
        "      --no-tools          disable all agent tools\n"
        "      --tools-list        print the available tools (built-in + MCP) and exit, without the model\n"
        "      --tool NAME [JSON]  run one tool with JSON arguments and print the result, without the model\n"
        "  -v, --verbose           more information while loading\n"
        "  -h, --help              this help\n");
}

static int parse_args(int argc, char **argv, Args *a) {
    memset(a, 0, sizeof(*a));
    a->n_ctx = 8192;
    a->n_batch = 64;
    a->n_predict = -1;
    a->host = "127.0.0.1";
    a->port = 8080;
    a->cache_mb = 4096;
    a->tools = 1;
    a->web = 1;
    a->workspace = "workspace";
    a->show_think = 1;
    a->prefetch = -1;
    a->gpu = 1;
    a->gpu_dense = 1;
    a->gpu_device = -1;
    a->show_think = 1;
    a->bench_pp = 128;
    a->bench_tg = 64;
    a->sp.temp = 0.1f;
    a->sp.top_k = 50;
    a->sp.top_p = 1.0f;
    a->sp.min_p = 0.0f;
    a->sp.repeat_penalty = 1.05f;
    a->sp.repeat_last_n = 64;
    a->sp.seed = 42;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing value for %s\n", s), exit(1), ""))
        if (!strcmp(s, "-m") || !strcmp(s, "--model")) a->model = NEXT();
        else if (!strcmp(s, "-p") || !strcmp(s, "--prompt")) a->prompt = NEXT();
        else if (!strcmp(s, "-f") || !strcmp(s, "--file")) {
            const char *path = NEXT();
            FILE *f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "cannot read %s\n", path); return 0; }
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *b = (char *)malloc((size_t)n + 1);
            n = (long)fread(b, 1, (size_t)n, f);
            b[n] = 0;
            fclose(f);
            a->prompt = b;
        }
        else if (!strcmp(s, "-s") || !strcmp(s, "--system")) a->system = NEXT();
        else if (!strcmp(s, "-i") || !strcmp(s, "--interactive")) a->interactive = 1;
        else if (!strcmp(s, "--raw")) a->raw = 1;
        else if (!strcmp(s, "-n") || !strcmp(s, "--n-predict")) a->n_predict = atoi(NEXT());
        else if (!strcmp(s, "-c") || !strcmp(s, "--ctx")) { a->n_ctx = atoi(NEXT()); a->ctx_set = 1; }
        else if (!strcmp(s, "-t") || !strcmp(s, "--threads")) a->threads = atoi(NEXT());
        else if (!strcmp(s, "-b") || !strcmp(s, "--batch")) a->n_batch = atoi(NEXT());
        else if (!strcmp(s, "--temp")) a->sp.temp = (float)atof(NEXT());
        else if (!strcmp(s, "--top-k")) a->sp.top_k = atoi(NEXT());
        else if (!strcmp(s, "--top-p")) a->sp.top_p = (float)atof(NEXT());
        else if (!strcmp(s, "--min-p")) a->sp.min_p = (float)atof(NEXT());
        else if (!strcmp(s, "--repeat-penalty")) a->sp.repeat_penalty = (float)atof(NEXT());
        else if (!strcmp(s, "--repeat-last-n")) a->sp.repeat_last_n = atoi(NEXT());
        else if (!strcmp(s, "--seed")) a->sp.seed = (uint64_t)strtoull(NEXT(), NULL, 10);
        else if (!strcmp(s, "--hide-think")) a->show_think = 0;
        else if (!strcmp(s, "--keep-think")) a->keep_think = 1;
        else if (!strcmp(s, "--prefetch")) a->prefetch = 1;
        else if (!strcmp(s, "--no-prefetch")) a->prefetch = 0;
        else if (!strcmp(s, "--no-gpu")) a->gpu = 0;
        else if (!strcmp(s, "--no-gpu-dense")) a->gpu_dense = 0;
        else if (!strcmp(s, "--gpu-device")) a->gpu_device = atoi(NEXT());
        else if (!strcmp(s, "--gpu-mem")) a->gpu_mem = atoi(NEXT());
        else if (!strcmp(s, "--list-gpus")) { gpu_list_devices(); exit(0); }
        else if (!strcmp(s, "--mlock")) a->mlock = 1;
        else if (!strcmp(s, "--kernels")) a->kernels = NEXT();
        else if (!strcmp(s, "--tokenize")) a->tokenize = NEXT();
        else if (!strcmp(s, "--ids")) a->ids = 1;
        else if (!strcmp(s, "--ppl")) a->ppl = NEXT();
        else if (!strcmp(s, "--stats")) a->stats = 1;
        else if (!strcmp(s, "-v") || !strcmp(s, "--verbose")) a->verbose = 1;
        else if (!strcmp(s, "--server")) a->server = 1;
        else if (!strcmp(s, "--open")) a->open_browser = 1;
        else if (!strcmp(s, "--workspace")) { a->workspace = NEXT(); a->workspace_set = 1; }
        else if (!strcmp(s, "--agent")) a->agent = 1;
        else if (!strcmp(s, "--auto")) { a->agent = 1; a->agent_auto = 1; }
        else if (!strcmp(s, "--yolo")) a->yolo = 1;
        else if (!strcmp(s, "--max-steps")) a->max_steps = atoi(NEXT());
        else if (!strcmp(s, "--show-think")) a->agent_think = 1;
        else if (!strcmp(s, "--shell")) a->shell = 1;
        else if (!strcmp(s, "--mcp")) a->mcp = NEXT();
        else if (!strcmp(s, "--read-only")) a->read_only = 1;
        else if (!strcmp(s, "--no-web")) a->web = 0;
        else if (!strcmp(s, "--no-tools")) a->tools = 0;
        else if (!strcmp(s, "--tools-list")) a->tools_list = 1;
        else if (!strcmp(s, "--tool")) {
            a->tool_name = NEXT();
            if (i + 1 < argc && argv[i + 1][0] != '-') a->tool_args = argv[++i];
        }
        else if (!strcmp(s, "--host")) a->host = NEXT();
        else if (!strcmp(s, "--port")) a->port = atoi(NEXT());
        else if (!strcmp(s, "--alias")) a->alias = NEXT();
        else if (!strcmp(s, "--api-key")) a->api_key = NEXT();
        else if (!strcmp(s, "--cache-dir")) a->cache_dir = NEXT();
        else if (!strcmp(s, "--cache-mb")) a->cache_mb = atoi(NEXT());
        else if (!strcmp(s, "--no-cache")) a->cache_mb = 0;
        else if (!strcmp(s, "--ui")) a->ui_path = NEXT();
        else if (!strcmp(s, "--bench")) {
            a->bench = 1;
            if (i + 1 < argc && isdigit((unsigned char)argv[i + 1][0])) {
                const char *v = argv[++i];
                a->bench_pp = atoi(v);
                const char *cm = strchr(v, ',');
                if (cm) a->bench_tg = atoi(cm + 1);
            }
        }
        else if (!strcmp(s, "-h") || !strcmp(s, "--help")) { usage(); exit(0); }
        else { fprintf(stderr, "unknown option %s (see --help)\n", s); return 0; }
#undef NEXT
    }
    if (a->n_ctx < 64) a->n_ctx = 64;
    if (a->n_batch < 1) a->n_batch = 1;
    if (a->n_batch > 512) a->n_batch = 512;
    if (a->n_predict < 0) a->n_predict = a->server ? 0 : 4096;
    if ((a->server || a->agent) && !a->ctx_set) a->n_ctx = 16384; /* room for tool results in agent mode */
    if (a->agent) {
        a->server = 0;
        a->shell = 1;
        if (!a->workspace_set) a->workspace = "."; /* the agent works in the folder it is started from */
        if (a->max_steps <= 0) a->max_steps = 40;
    }
    if (!a->prompt && !a->bench && !a->tokenize && !a->ppl && !a->server && !a->tools_list && !a->tool_name)
        a->interactive = 1;
    return 1;
}

/* ------------------------------------------------------------------ generation */

typedef struct {
    Model *m;
    Context *ctx;
    Tokenizer *tk;
    Sampler smp;
    Args *a;
    int tok_im_end, tok_eot, tok_think_end;
    double t_prompt, t_gen;
    int n_prompt, n_gen;
} Engine;

static int feed(Engine *e, const int *ids, int n) {
    double t0 = time_sec();
    if (!ctx_forward(e->ctx, ids, n)) return 0;
    e->t_prompt += time_sec() - t0;
    e->n_prompt += n;
    for (int i = 0; i < n; i++) sampler_accept(&e->smp, ids[i]);
    return 1;
}

static int feed_text(Engine *e, const char *text) {
    int n;
    int *ids = encode(e->tk, text, &n);
    int ok = feed(e, ids, n);
    free(ids);
    return ok;
}

/* Generate until end-of-turn. `in_think`: generation starts inside a <think> block.
 * Returns the generated text (caller frees); *last receives the stop token (or -1). */
static char *generate(Engine *e, int in_think, int *last) {
    Str out = { 0 };
    Utf8Out uo = { { 0 }, 0 };
    float *logits = ctx_forward(e->ctx, NULL, 0);
    int thinking = in_think;
    if (thinking && e->a->show_think) color("\x1b[90m");
    if (thinking && !e->a->show_think) { color("\x1b[90m"); out_str("[thinking..."); color("\x1b[0m"); }
    *last = -1;
    double t0 = time_sec();
    int n = 0, skip_ws = 0;
    for (; n < e->a->n_predict; n++) {
        int tok = sampler_sample(&e->smp, logits);
        if (tok == e->tok_im_end || tok == e->tok_eot || tok == tok_eos(e->tk)) { *last = tok; break; }
        int pl;
        const char *piece = tok_piece(e->tk, tok, &pl);
        str_add(&out, piece, pl);
        if (e->a->ids) printf("[%d]", tok);
        if (tok == e->tok_think_end && thinking) {
            thinking = 0;
            skip_ws = 1;
            if (e->a->show_think) { out_bytes(&uo, piece, pl); color("\x1b[0m"); out_str("\n\n"); }
            else { color("\x1b[90m"); out_str(" done]\n\n"); color("\x1b[0m"); }
        } else if (!thinking || e->a->show_think) {
            if (skip_ws) {
                /* the answer usually starts with blank lines after </think>: drop them */
                while (pl > 0 && (*piece == '\n' || *piece == ' ' || *piece == '\r')) { piece++; pl--; }
                if (pl > 0) skip_ws = 0;
            }
            out_bytes(&uo, piece, pl);
        }
        sampler_accept(&e->smp, tok);
        if (ctx_pos(e->ctx) + 1 >= ctx_n_ctx(e->ctx)) {
            fprintf(stderr, "\n[context full: %d tokens]\n", ctx_n_ctx(e->ctx));
            break;
        }
        logits = ctx_forward(e->ctx, &tok, 1);
        if (!logits) break;
    }
    if (thinking) color("\x1b[0m");
    e->t_gen += time_sec() - t0;
    e->n_gen += n;
    if (!out.s) str_add(&out, "", 0);
    return out.s;
}

static void print_expert_coverage(const Model *m, const Stats *st);

static void print_stats(Engine *e, double t_prompt, int n_prompt, double t_gen, int n_gen) {
    color("\x1b[90m");
    printf("\n[prompt: %d tok in %.2f s = %.1f tok/s | generation: %d tok in %.2f s = %.2f tok/s | context %d/%d]\n",
           n_prompt, t_prompt, n_prompt / (t_prompt > 0 ? t_prompt : 1e-9), n_gen, t_gen,
           n_gen / (t_gen > 0 ? t_gen : 1e-9), ctx_pos(e->ctx), ctx_n_ctx(e->ctx));
    if (e->a->stats) {
        Stats *st = ctx_stats(e->ctx);
        print_expert_coverage(e->m, st);
        printf("[matmul %.2f s | attention %.2f s | routing %.2f s | experts on GPU %llu, GPU wait %.2f s]\n", st->t_matmul,
               st->t_attn, st->t_moe_route, (unsigned long long)st->gpu_experts, st->t_gpu_wait);
    }
    color("\x1b[0m");
    fflush(stdout);
}

static void print_profile(const Stats *st, int ntok, const char *title) {
    static const char *cn[N_CAT] = { "attn proj", "conv proj", "dense ffn", "router", "experts g/u", "experts down", "logits" };
    if (ntok < 1) ntok = 1;
    printf("profile %s (per token):\n", title);
    for (int k = 0; k < N_CAT; k++)
        printf("  %-13s %7.2f ms  %6.1f MB  %5.1f GB/s  %5.1f calls\n", cn[k], 1e3 * st->t_cat[k] / ntok,
               st->bytes_cat[k] / ntok / 1048576.0, st->bytes_cat[k] / (st->t_cat[k] + 1e-12) / 1e9,
               (double)st->calls_cat[k] / ntok);
    printf("  %-13s %7.2f ms\n  %-13s %7.2f ms\n  %-13s %7.2f ms\n", "attention", 1e3 * st->t_attn / ntok, "routing",
           1e3 * st->t_moe_route / ntok, "resto", 1e3 * (st->t_forward - st->t_matmul - st->t_attn - st->t_moe_route) / ntok);
    if (st->pred_total)
        printf("  next-layer expert prediction: %.1f%% correct\n", 100.0 * st->pred_hits / st->pred_total);
    if (st->gpu_experts)
        printf("  experts computed on GPU: %.1f per token | GPU wait %.2f ms/token\n", (double)st->gpu_experts / ntok,
               1e3 * st->t_gpu_wait / ntok);
}

static void print_expert_coverage(const Model *m, const Stats *st) {
    /* expert usage skew: share of activations covered by the hottest experts of each layer */
    int fr[] = { 8, 16, 24, 32 };
    for (int fi = 0; fi < 4; fi++) {
        double cov = 0, tot = 0;
        for (int l = 0; l < m->n_layer && l < 64; l++) {
            if (!m->layers[l].is_moe) continue;
            uint64_t h[64];
            memcpy(h, st->expert_hits + l * 64, sizeof(h));
            for (int i = 0; i < m->n_expert; i++)
                for (int j = i + 1; j < m->n_expert; j++)
                    if (h[j] > h[i]) { uint64_t t = h[i]; h[i] = h[j]; h[j] = t; }
            for (int i = 0; i < m->n_expert; i++) { tot += h[i]; if (i < fr[fi]) cov += h[i]; }
        }
        printf("  top %2d/%d experts per layer cover %.1f%% of the activations\n", fr[fi], m->n_expert, 100 * cov / (tot + 1e-9));
    }
}

static char *trim_dup(const char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) n--;
    char *r = (char *)malloc(n + 1);
    memcpy(r, s, n);
    r[n] = 0;
    return r;
}

/* One chat turn. Returns 0 if the context overflowed. */
static int chat_turn(Engine *e, const char *user, int first) {
    Str p = { 0 };
    if (first) {
        str_add(&p, "<|startoftext|>", -1);
        if (e->a->system && *e->a->system) {
            str_add(&p, "<|im_start|>system\n", -1);
            str_add(&p, e->a->system, -1);
            str_add(&p, "<|im_end|>\n", -1);
        }
    }
    str_add(&p, "<|im_start|>user\n", -1);
    str_add(&p, user, -1);
    str_add(&p, "<|im_end|>\n", -1);
    double tp0 = e->t_prompt, tg0 = e->t_gen;
    int np0 = e->n_prompt, ng0 = e->n_gen;
    if (!feed_text(e, p.s)) { free(p.s); return 0; }
    free(p.s);
    CtxSnapshot snap;
    ctx_snapshot(e->ctx, &snap);
    if (!feed_text(e, "<|im_start|>assistant\n<think>")) { ctx_snapshot_free(&snap); return 0; }
    int last;
    char *gen = generate(e, 1, &last);
    double tp = e->t_prompt - tp0;
    int npr = e->n_prompt - np0;
    print_stats(e, tp, npr, e->t_gen - tg0, e->n_gen - ng0);
    int ok = 1;
    if (e->a->keep_think) {
        /* KV already holds the turn verbatim: just close it */
        int close[2] = { e->tok_im_end, -1 };
        int nl;
        int *ids = encode(e->tk, "\n", &nl);
        close[1] = ids[0];
        free(ids);
        ok = feed(e, close, 2);
    } else {
        /* template default: past turns keep only the final answer */
        const char *c = strstr(gen, "</think>");
        char *content = trim_dup(c ? c + 8 : gen);
        ctx_restore(e->ctx, &snap);
        Str a = { 0 };
        str_add(&a, "<|im_start|>assistant\n", -1);
        if (!strcmp(e->m->arch, "lfm3")) str_add(&a, "<think></think>", -1); /* its template marks the dropped thinking */
        str_add(&a, content, -1);
        str_add(&a, "<|im_end|>\n", -1);
        double t0 = e->t_prompt;
        int n0 = e->n_prompt;
        ok = feed_text(e, a.s);
        e->t_prompt = t0; /* bookkeeping only, not user-visible prompt time */
        e->n_prompt = n0;
        free(a.s);
        free(content);
    }
    ctx_snapshot_free(&snap);
    free(gen);
    return ok;
}

/* ------------------------------------------------------------------ tools without the model */

static ToolsConfig tools_config(const Args *a) {
    ToolsConfig tc;
    memset(&tc, 0, sizeof(tc));
    tc.workspace = a->workspace;
    tc.web = a->web;
    tc.allow_write = !a->read_only;
    tc.shell = a->shell;
    tc.mcp_config = a->mcp;
    if (!tc.mcp_config) {
        FILE *f = file_open("mcp.json", "rb");
        if (f) { fclose(f); tc.mcp_config = "mcp.json"; }
    }
    return tc;
}

/* --tools-list / --tool: check the tools and an mcp.json without loading the model. */
static int tools_cli(const Args *a) {
    ToolsConfig tc = tools_config(a);
    Tools *t = tools_create(&tc);
    /* wait for the MCP servers to finish starting */
    for (int i = 0; i < 1200; i++) {
        Buf b = { 0 };
        tools_list_json(t, &b);
        int starting = strstr(b.s, "\"status\":\"starting\"") != NULL;
        buf_free(&b);
        if (!starting) break;
        PithosLock *pause = lock_create();
        lock_acquire(pause);
        lock_wait_ms(pause, 100);
        lock_release(pause);
        lock_destroy(pause);
    }
    int rc = 0;
    if (a->tools_list) {
        Buf b = { 0 };
        tools_list_json(t, &b);
        Json *j = json_parse(b.s, b.n, NULL, 0);
        for (const Json *x = json_get(j, "tools") ? json_get(j, "tools")->child : NULL; x; x = x->next) {
            const Json *fn = json_get(x, "function");
            printf("%-34s %-14s %s\n", json_str(json_get(fn, "name"), "?"), json_str(json_get(x, "source"), ""),
                   json_bool(json_get(x, "confirm"), 0) ? "(asks first)" : "");
        }
        for (const Json *x = json_get(j, "servers") ? json_get(j, "servers")->child : NULL; x; x = x->next)
            printf("MCP %-20s %s %s\n", json_str(json_get(x, "name"), "?"), json_str(json_get(x, "status"), ""),
                   json_str(json_get(x, "error"), ""));
        json_free(j);
        buf_free(&b);
    } else {
        char err[160];
        const char *src = a->tool_args ? a->tool_args : "{}";
        Json *args = json_parse(src, (int)strlen(src), err, sizeof(err));
        if (!args) { fprintf(stderr, "invalid JSON arguments: %s\n", err); rc = 2; }
        else {
            Buf out = { 0 };
            int ok = tools_call(t, a->tool_name, args, &out);
            printf("%s\n", out.s ? out.s : "");
            rc = ok ? 0 : 1;
            buf_free(&out);
            json_free(args);
        }
    }
    tools_free(t);
    return rc;
}

/* ------------------------------------------------------------------ main */

static int real_main(int argc, char **argv) {
    Args a;
    if (!parse_args(argc, argv, &a)) return 1;
    console_init();
    if (a.agent) { /* the banner first: something to look at while the model loads */
        AgentConfig bc;
        memset(&bc, 0, sizeof(bc));
        bc.color = g_color;
        bc.version = VERSION;
        bc.autonomous = a.agent_auto;
        bc.yolo = a.yolo;
        agent_banner(&bc);
    }

    SysInfo si;
    sys_info(&si);
    const char *kn = quant_init(a.kernels);
    if (!a.threads) a.threads = si.n_physical;
    if (a.tools_list || a.tool_name) return tools_cli(&a);
    if (!a.model) a.model = default_model_path();
    if (!a.model) return 1; /* default_model_path said why */

    fprintf(stderr, "Pithos %s | CPU: %s | %d core / %d thread | RAM %.1f GB (%.1f GB free)\n", VERSION,
            si.brand[0] ? si.brand : "?", si.n_physical, si.n_logical, si.total_ram / 1073741824.0,
            si.avail_ram / 1073741824.0);
    fprintf(stderr, "kernel: %s%s%s | thread: %d\n", kn, si.has_avx512f ? " (AVX-512 present)" : "",
            si.has_avxvnni ? " (AVX-VNNI present)" : "", a.threads);

    double t0 = time_sec();
    char err[256];
    Model m;
    if (!model_load(&m, a.model, err, sizeof(err))) { fprintf(stderr, "error: %s\n", err); return 1; }
    Tokenizer *tk = tok_load(&m.gguf, err, sizeof(err));
    if (!tk) { fprintf(stderr, "error: %s\n", err); return 1; }
    char moe[48] = "", swa[48] = "";
    int n_swa = 0;
    for (int l = 0; l < m.n_layer; l++) n_swa += m.layers[l].is_swa;
    if (m.n_expert) snprintf(moe, sizeof(moe), " | %d experts top-%d", m.n_expert, m.n_expert_used);
    if (n_swa) snprintf(swa, sizeof(swa), ", %d of them sliding-window %d", n_swa, m.swa_window);
    fprintf(stderr, "model: %s | %d layers (%d attn%s, %d conv) | dim %d%s | vocab %d | %.2f GB | ~%.0f MB/token | %.2f s\n",
            m.arch, m.n_layer, m.n_attn, swa, m.n_conv, m.dim, moe, m.n_vocab,
            m.gguf.mf.size / 1073741824.0, m.bytes_per_token / 1048576.0, time_sec() - t0);

    if (a.tokenize) {
        int n;
        int *ids = encode(tk, a.tokenize, &n);
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
        free(ids);
        return 0;
    }

    if (a.n_ctx > m.n_ctx_train) a.n_ctx = m.n_ctx_train;
    if (si.avail_ram && m.gguf.mf.size > si.avail_ram)
        fprintf(stderr, "note: the model (%.1f GB) exceeds the free RAM (%.1f GB): weights will be read from disk on demand.\n",
                m.gguf.mf.size / 1073741824.0, si.avail_ram / 1073741824.0);
    if (a.mlock) {
        int ok = lock_memory(m.gguf.mf.data, (size_t)m.gguf.mf.size);
        fprintf(stderr, "mlock: %s\n", ok ? "ok" : "failed (continuing without it)");
    }

    ThreadPool *pool = pool_create(a.threads);
    Context *ctx = ctx_create(&m, pool, a.n_ctx, a.n_batch);
    if (!ctx) { fprintf(stderr, "error: not enough memory for the context\n"); return 1; }
    if (a.prefetch < 0) a.prefetch = m.n_expert && si.avail_ram && m.gguf.mf.size > si.avail_ram * 9 / 10;
    ctx_set_prefetch(ctx, a.prefetch);
    if (a.prefetch) fprintf(stderr, "background expert read-ahead: on\n");
    Gpu *gpu = NULL;
    if (a.gpu) {
        char ginfo[512], prof[1100];
        snprintf(prof, sizeof(prof), "%s.pithos-experts", a.model);
        gpu = gpu_create(&m, a.gpu_device, a.gpu_mem, a.n_batch, a.gpu_dense ? a.n_ctx : 0, prof, ginfo, sizeof(ginfo));
        fprintf(stderr, "GPU: %s\n", gpu ? ginfo : "not used");
        if (!gpu && ginfo[0]) fprintf(stderr, "     (%s)\n", ginfo);
        ctx_set_gpu(ctx, gpu);
    }

    Engine e;
    memset(&e, 0, sizeof(e));
    e.m = &m;
    e.ctx = ctx;
    e.tk = tk;
    e.a = &a;
    e.tok_im_end = tok_find(tk, "<|im_end|>");
    e.tok_eot = tok_find(tk, "<|endoftext|>");
    e.tok_think_end = tok_find(tk, "</think>");
    sampler_init(&e.smp, &a.sp, m.n_vocab);

    if (a.server || a.agent) {
        char alias[256], cache_dir[1100], info[512];
        if (a.alias) snprintf(alias, sizeof(alias), "%s", a.alias);
        else {
            const char *b = a.model + strlen(a.model);
            while (b > a.model && b[-1] != '/' && b[-1] != '\\') b--;
            snprintf(alias, sizeof(alias), "%s", b);
            char *dot = strstr(alias, ".gguf");
            if (dot) *dot = 0;
        }
        snprintf(cache_dir, sizeof(cache_dir), "%s", a.cache_dir ? a.cache_dir : "");
        if (!a.cache_dir) snprintf(cache_dir, sizeof(cache_dir), "%s.pithos-cache", a.model);
        /* stored sessions are tied to this exact model file: size + start of the metadata */
        uint64_t mid = 1469598103934665603ull ^ m.gguf.mf.size;
        for (size_t i = 0; i < 65536 && i < m.gguf.mf.size; i++) mid = (mid ^ m.gguf.mf.data[i]) * 1099511628211ull;
        mid ^= (uint64_t)a.n_ctx << 1;
        PCache *pc = pcache_create(ctx, a.cache_mb > 0 ? cache_dir : NULL, mid, 64, (uint64_t)a.cache_mb << 20);
        if (a.cache_mb > 0)
            fprintf(stderr, "prompt reuse: %d stored conversations in %s\n", pcache_n_files(pc), cache_dir);
        snprintf(info, sizeof(info), "%s | %d thread | kernel %s | GPU: %s", si.brand[0] ? si.brand : "CPU", a.threads, kn,
                 gpu ? "yes" : "no");
        ServerConfig sc;
        memset(&sc, 0, sizeof(sc));
        sc.host = a.host;
        sc.port = a.port;
        sc.alias = alias;
        sc.ui_path = a.ui_path;
        sc.api_key = a.api_key;
        sc.n_predict = a.n_predict;
        sc.sp = a.sp;
        sc.info = info;
        sc.verbose = a.verbose || a.stats;
        sc.open_browser = a.open_browser;
        Tools *tools = NULL;
        if (a.tools || a.agent) {
            ToolsConfig tc = tools_config(&a);
            tools = tools_create(&tc);
            fprintf(stderr, "agent tools: %d built-in%s%s | workspace: %s\n", tools_count(tools),
                    tc.mcp_config ? " + MCP servers from " : "", tc.mcp_config ? tc.mcp_config : "",
                    a.workspace);
        }
        sc.tools = tools;
        int rc;
        if (a.agent) {
            server_attach(&sc, &m, ctx, tk, pc);
            AgentConfig ac;
            memset(&ac, 0, sizeof(ac));
            ac.tools = tools;
            ac.autonomous = a.agent_auto;
            ac.yolo = a.yolo;
            ac.task = a.prompt;
            ac.system = a.system;
            ac.max_steps = a.max_steps;
            ac.max_tokens = a.n_predict;
            ac.show_think = a.agent_think;
            ac.color = g_color;
            ac.version = VERSION;
            ac.banner_shown = 1;
            rc = agent_run(&ac);
        } else {
            rc = server_run(&sc, &m, ctx, tk, pc);
        }
        tools_free(tools);
        pcache_free(pc);
        return rc;
    }

    if (a.ppl) {
        FILE *f = fopen(a.ppl, "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", a.ppl); return 1; }
        fseek(f, 0, SEEK_END);
        long fl = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *txt = (char *)malloc((size_t)fl + 1);
        fl = (long)fread(txt, 1, (size_t)fl, f);
        txt[fl] = 0;
        fclose(f);
        int *ids = (int *)malloc(sizeof(int) * ((size_t)fl + 2));
        /* like llama-perplexity: a BOS only where the model asks for one (LFM3 does not) */
        const int bos = m.add_bos;
        ids[0] = tok_bos(tk);
        int nt = bos + tok_encode(tk, txt, (int)fl, 0, ids + bos, (int)fl + 1);
        const int nc = a.n_ctx, nchunks = nt / nc, first = nc / 2;
        float *lg = (float *)malloc(sizeof(float) * (size_t)nc * m.n_vocab);
        double nll = 0;
        int count = 0;
        double tp = time_sec();
        fprintf(stderr, "%d tokens, %d chunks of %d\n", nt, nchunks, nc);
        for (int ch = 0; ch < nchunks; ch++) {
            int *tk0 = ids + (size_t)ch * nc;
            int save = tk0[0];
            if (bos) tk0[0] = tok_bos(tk);
            ctx_reset(ctx);
            ctx_forward_all(ctx, tk0, nc, lg);
            tk0[0] = save;
            for (int j = first; j < nc - 1; j++) {
                const float *l = lg + (size_t)j * m.n_vocab;
                float mx = l[0];
                for (int v = 1; v < m.n_vocab; v++) if (l[v] > mx) mx = l[v];
                double se = 0;
                for (int v = 0; v < m.n_vocab; v++) se += exp((double)l[v] - mx);
                nll += -(l[ids[(size_t)ch * nc + j + 1]] - mx - log(se));
                count++;
            }
            printf("[%d]%.4f ", ch + 1, exp(nll / count));
            fflush(stdout);
        }
        printf("\nPPL = %.4f (%d tokens evaluated, %.1f s)\n", exp(nll / (count ? count : 1)), count, time_sec() - tp);
        free(lg); free(ids); free(txt);
    } else if (a.bench) {
        int npp = a.bench_pp, ntg = a.bench_tg;
        int *ids = (int *)malloc(sizeof(int) * (npp > 0 ? npp : 1));
        for (int i = 0; i < npp; i++) ids[i] = 1000 + (i * 7919) % 50000;
        if (gpu) {
            /* measure with the VRAM expert cache filled (as after the first minutes of use) */
            double tw = time_sec();
            while (gpu_pending(gpu) > 0) {
#if defined(_WIN32)
                Sleep(50);
#endif
            }
            printf("VRAM cache fill: %.1f s\n", time_sec() - tw);
        }
        /* warm-up: touch weights once */
        ctx_forward(ctx, ids, 1);
        ctx_reset(ctx);
        double tb = 0;
        for (int pass = 0; pass < 2 && npp > 0; pass++) {
            ctx_reset(ctx);
            memset(ctx_stats(ctx), 0, sizeof(Stats));
            uint64_t pf0, rss0, pf1, rss1;
            proc_mem_stats(&pf0, &rss0);
            tb = time_sec();
            ctx_forward(ctx, ids, npp);
            double tpp = time_sec() - tb;
            proc_mem_stats(&pf1, &rss1);
            printf("pp%d %s: %.2f tok/s (%.2f s) | page fault %.0f (%.2f GB) | working set %.2f GB\n", npp,
                   pass ? "(warm)" : "(first)", npp / tpp, tpp, (double)(pf1 - pf0), (pf1 - pf0) * 4096.0 / 1073741824.0,
                   rss1 / 1073741824.0);
            if (pass) print_profile(ctx_stats(ctx), npp, "warm prompt");
        }
        /* two identical generation passes: the second one runs with the touched experts resident */
        for (int pass = 0; pass < 2; pass++) {
            ctx_reset(ctx);
            if (npp > 0) ctx_forward(ctx, ids, npp < 16 ? npp : 16);
            int tok = 1000;
            if (pass) memset(ctx_stats(ctx), 0, sizeof(Stats));
            uint64_t pf0, rss0, pf1, rss1;
            proc_mem_stats(&pf0, &rss0);
            tb = time_sec();
            for (int i = 0; i < ntg; i++) {
                float *lg = ctx_forward(ctx, &tok, 1);
                int best = 0;
                for (int v = 1; v < m.n_vocab; v++) if (lg[v] > lg[best]) best = v;
                tok = best;
            }
            double ttg = time_sec() - tb;
            printf("tg%d %s: %.2f tok/s (%.2f s) | effective weight bandwidth ~%.1f GB/s\n", ntg, pass ? "(warm)" : "(first)",
                   ntg / ttg, ttg, m.bytes_per_token * (ntg / ttg) / 1e9);
            proc_mem_stats(&pf1, &rss1);
            printf("   page fault/token: %.0f | working set %.2f GB\n", (double)(pf1 - pf0) / (ntg ? ntg : 1),
                   rss1 / 1073741824.0);
        }
        print_profile(ctx_stats(ctx), ntg, "warm generation");
        if (a.verbose) print_expert_coverage(&m, ctx_stats(ctx));
        free(ids);
    } else if (a.raw) {
        if (!feed_text(&e, a.prompt)) { fprintf(stderr, "prompt too long for the context\n"); return 1; }
        out_str(a.prompt);
        int last;
        char *g = generate(&e, 0, &last);
        print_stats(&e, e.t_prompt, e.n_prompt, e.t_gen, e.n_gen);
        free(g);
    } else if (!a.interactive) {
        if (!chat_turn(&e, a.prompt, 1)) fprintf(stderr, "context full\n");
    } else {
        fprintf(stderr, "chat ready. Commands: /reset new conversation, /exit quit.\n");
        int first = 1;
        static char line[65536];
        for (;;) {
            color("\x1b[1;32m");
            out_str("\n> ");
            color("\x1b[0m");
            if (!console_read_line(line, sizeof(line))) break;
            if (!line[0]) continue;
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            if (!strcmp(line, "/reset")) {
                ctx_reset(ctx);
                sampler_reset_history(&e.smp);
                first = 1;
                out_str("[conversation reset]\n");
                continue;
            }
            if (!chat_turn(&e, line, first)) {
                out_str("\n[context full: conversation reset]\n");
                ctx_reset(ctx);
                sampler_reset_history(&e.smp);
                first = 1;
                continue;
            }
            first = 0;
        }
    }

    sampler_free(&e.smp);
    ctx_free(ctx);
    gpu_destroy(gpu);
    pool_destroy(pool);
    tok_free(tk);
    model_free(&m);
    return 0;
}

#if defined(_WIN32)
/* Windows passes argv in the ANSI code page: take the UTF-16 command line and convert to UTF-8. */
int wmain(int argc, wchar_t **wargv) {
    char **argv = (char **)malloc(sizeof(char *) * (size_t)(argc + 1));
    for (int i = 0; i < argc; i++) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        argv[i] = (char *)malloc((size_t)n);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], n, NULL, NULL);
    }
    argv[argc] = NULL;
    return real_main(argc, argv);
}
#else
int main(int argc, char **argv) { return real_main(argc, argv); }
#endif
