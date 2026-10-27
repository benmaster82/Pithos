/* tools.h - tools the model can call in agent mode: built-in ones (web search and pages via
 * curl, files in a sandboxed workspace folder, calculator, date and time) plus the tools of
 * MCP servers (Model Context Protocol, stdio transport) listed in an mcp.json file. */
#ifndef PITHOS_TOOLS_H
#define PITHOS_TOOLS_H

#include "json.h"

typedef struct {
    const char *workspace;      /* folder for the file tools (created if missing), NULL = no file tools */
    const char *mcp_config;     /* mcp.json path, NULL = no MCP servers */
    int web;                    /* enable web_search / fetch_url (needs curl on the PATH) */
    int allow_write;            /* enable write_file */
} ToolsConfig;

typedef struct Tools Tools;

/* MCP servers start in the background: their tools appear in the list when ready. */
Tools *tools_create(const ToolsConfig *cfg);
void   tools_free(Tools *t);

/* {"tools":[{"type":"function","function":{name,description,parameters},"source":..,"confirm":..}],
 *  "servers":[{"name","status","tools","error"}],"workspace":".."} */
void tools_list_json(Tools *t, Buf *b);
int  tools_count(Tools *t);

/* Run a tool. args: JSON object (or NULL). The text result goes to out.
 * Returns 1 on success, 0 if the tool reported an error (message in out). */
int tools_call(Tools *t, const char *name, const Json *args, Buf *out);

#endif
