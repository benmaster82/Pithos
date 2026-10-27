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
} ServerConfig;

/* Serves until the process is terminated. Returns non-zero if the socket cannot be opened. */
int server_run(const ServerConfig *cfg, Model *m, Context *ctx, Tokenizer *tk, PCache *pc);

#endif
