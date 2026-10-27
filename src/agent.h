/* agent.h - terminal agent: the model works on a task through the tools (shell commands,
 * background processes, files, web, MCP), in safe mode (every action is approved by the user)
 * or autonomous mode (it works on its own; destructive commands still need approval). */
#ifndef PITHOS_AGENT_H
#define PITHOS_AGENT_H

#include "tools.h"

typedef struct {
    Tools *tools;
    int autonomous;             /* start in autonomous mode (switch with /auto and /safe) */
    int yolo;                   /* autonomous mode does not ask even for destructive commands */
    const char *task;           /* run this task, then exit (NULL: interactive session) */
    const char *system;         /* extra instructions for the system prompt */
    int max_steps;              /* model calls per task before stopping to ask */
    int max_tokens;             /* per answer (<= 0: until the context is full) */
    int show_think;             /* stream the reasoning instead of a one-line counter */
    int color;                  /* ANSI colors on the terminal */
} AgentConfig;

/* Needs server_attach() first: the agent talks to the model through server_local_chat().
 * Returns 0 when the (last) task ended normally. */
int agent_run(const AgentConfig *cfg);

#endif
