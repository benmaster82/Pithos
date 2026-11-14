/* server.h - HTTP server: OpenAI-compatible API (chat/completions, completions, models)
 * plus the embedded web chat interface. */
#ifndef PITHOS_SERVER_H
#define PITHOS_SERVER_H

#include "model.h"
#include "pcache.h"
#include "sampler.h"
#include "tokenizer.h"
#include "tools.h"

typedef struct {
    const char *host;
    int port;
    const char *alias;          /* model id reported by the API */
    const char *ui_path;        /* serve the web UI from this file instead of the embedded copy */
    const char *api_key;        /* require "Authorization: Bearer KEY" on /v1 (NULL = open) */
    int n_predict;              /* default max new tokens (<= 0: until the context is full) */
    SamplerParams sp;           /* default sampling */
    const char *info;           /* hardware summary for /props */
    int verbose;
    int open_browser;           /* open the web UI once the server is listening */
    Tools *tools;               /* agent tools served on /v1/tools (NULL = none) */
    int think_budget;           /* default reasoning: THINK_FULL, THINK_OFF or a token budget */
    struct Audio *audio;        /* speech input (LFM3-Audio): input_audio parts, /v1/audio/transcriptions */
    struct Speech *speech;      /* speech output (LFM3-Audio): /v1/audio/speech */
} ServerConfig;

/* Reasoning control. THINK_FULL: the model reasons as long as it wants (the template default);
 * THINK_OFF: the answer starts after an empty <think></think>; N > 0: after N reasoning tokens
 * the reasoning is closed and the model moves on to the answer. */
enum { THINK_FULL = -1, THINK_OFF = 0 };

/* "off" / "none", "low" / "minimal", "medium", "high", or a number of tokens. Returns 0 if the
 * text is none of these. */
int think_parse(const char *s, int *budget);
/* The text that closes the reasoning when the budget runs out. */
const char *think_close(void);
/* The assistant turn opening with the reasoning off (lfm3: the LFM3 architecture). */
const char *think_off_prompt(int lfm3);
/* Whether the model's chat template opens and closes reasoning blocks. LFM3-Audio's has no
 * enable_thinking switch and writes no blocks: past turns have no empty <think></think>, and with
 * the reasoning off its turn starts plainly (an answer written after "<think>\n</think>\n" often
 * ends with a second </think> and the answer again). Asked to reason, it reasons after "<think>". */
int template_reasons(const Model *m);

/* Serves until the process is terminated. Returns non-zero if the socket cannot be opened. */
int server_run(const ServerConfig *cfg, Model *m, Context *ctx, Tokenizer *tk, PCache *pc);

/* In-process use without HTTP (the terminal agent): server_attach() once, then each
 * server_local_chat() runs a /v1/chat/completions request (JSON body) and hands the raw response
 * to sink: the HTTP head, then the SSE events when "stream" is true. Generation stops early
 * when *stop becomes non-zero. */
void server_attach(const ServerConfig *cfg, Model *m, Context *ctx, Tokenizer *tk, PCache *pc);
void server_local_chat(const char *body, int len, void (*sink)(void *u, const char *s, int n), void *u,
                       volatile int *stop);

#endif
