"""Minimal MCP server (stdio, newline-delimited JSON-RPC) used by tests/test_tools.py.
Tools: echo(text), add(a, b), fail(). Standard library only."""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Return the given text.", "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "add", "description": "Add two numbers.",
     "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
]


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, id_ = msg.get("method"), msg.get("id")
    if id_ is None:
        continue  # notification
    if method == "initialize":
        # a log notification and a ping first: the client must cope with both
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "hi"}}) + "\n")
        sys.stdout.flush()
        reply(id_, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "test", "version": "1"}})
    elif method == "tools/list":
        cursor = (msg.get("params") or {}).get("cursor")
        if cursor is None:
            reply(id_, {"tools": TOOLS[:2], "nextCursor": "page2"})
        else:
            reply(id_, {"tools": TOOLS[2:]})
    elif method == "tools/call":
        p = msg["params"]
        name, args = p["name"], p.get("arguments") or {}
        if name == "echo":
            reply(id_, {"content": [{"type": "text", "text": args.get("text", "")}]})
        elif name == "add":
            reply(id_, {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]})
        elif name == "fail":
            reply(id_, {"content": [{"type": "text", "text": "something went wrong"}], "isError": True})
        else:
            reply(id_, error={"code": -32602, "message": "unknown tool " + name})
    else:
        reply(id_, error={"code": -32601, "message": "method not found"})
