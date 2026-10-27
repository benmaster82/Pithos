"""Checks for the agent tools of `pithos --server` (no model calls needed).

Start the server with the test MCP server and a scratch workspace, e.g.:
  build\\pithos.exe --server --workspace %TEMP%\\pithos-ws --mcp tests\\mcp_test.json
then:  python tests/test_tools.py [http://127.0.0.1:8080] [--no-net]"""
import json
import sys
import time
import urllib.request

BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8080").rstrip("/")
NET = "--no-net" not in sys.argv
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FAILS = []


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=120) as r:
        return json.load(r)


def tool(name, **args):
    return call("/v1/tools/call", {"name": name, "arguments": args})


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + str(detail)[:300]) if not cond else ""))
    if not cond:
        FAILS.append(name)


# MCP servers start in the background
for _ in range(60):
    lst = call("/v1/tools")
    if all(s["status"] != "starting" for s in lst["servers"]):
        break
    time.sleep(0.5)
names = {t["function"]["name"]: t for t in lst["tools"]}
print("tools:", ", ".join(names))
print("servers:", lst["servers"])

print("built-in:")
r = tool("calculator", expression="2^10 / 4 + sqrt(16) * (3 - 1)")
check("calculator", r["ok"] and r["content"] == "264", r)
r = tool("calculator", expression="2 +* 3")
check("calculator error", not r["ok"] and "error" in r["content"], r)
r = tool("get_current_datetime")
check("datetime", r["ok"] and r["content"][:2] == "20" and "UTC" in r["content"], r)

if lst["workspace"]:
    r = tool("write_file", path="notes/config.yaml", content="server:\n  port: 9443\nname: àèì ✓\n")
    check("write_file (creates folders)", r["ok"], r)
    r = tool("read_file", path="notes/config.yaml")
    check("read_file", r["ok"] and "9443" in r["content"] and "àèì ✓" in r["content"], r)
    r = tool("list_directory", path=".")
    check("list_directory", r["ok"] and "notes/" in r["content"], r)
    r = tool("find_files", pattern="*.yaml")
    check("find_files", r["ok"] and "notes/config.yaml" in r["content"], r)
    r = tool("search_text", query="PORT")
    check("search_text", r["ok"] and "notes/config.yaml:2:" in r["content"], r)
    for bad in ["../secret.txt", "/etc/passwd", "C:/Windows/win.ini", "notes/../../x"]:
        r = tool("read_file", path=bad)
        check("sandbox rejects " + bad, not r["ok"], r)

if NET and "web_search" in names:
    print("web (network):")
    r = tool("web_search", query="Liquid AI LFM2.5", max_results=3)
    check("web_search", r["ok"] and "http" in r["content"], r)
    r = tool("fetch_url", url="https://example.com/")
    check("fetch_url html->text", r["ok"] and "Example Domain" in r["content"] and "<" not in r["content"], r)
    r = tool("fetch_url", url="file:///etc/passwd")
    check("fetch_url rejects file://", not r["ok"], r)

if any(s["name"] == "test" for s in lst["servers"]):
    print("MCP:")
    check("MCP tools listed across pages", {"test__echo", "test__add", "test__fail"} <= set(names), list(names))
    check("readOnlyHint -> no confirmation", names.get("test__echo", {}).get("confirm") is False and names.get("test__add", {}).get("confirm") is True)
    r = tool("test__echo", text="ciao ✓")
    check("MCP call", r["ok"] and r["content"] == "ciao ✓", r)
    r = call("/v1/tools/call", {"name": "test__add", "arguments": json.dumps({"a": 2, "b": 3.5})})
    check("MCP call with string arguments", r["ok"] and r["content"] == "5.5", r)
    r = tool("test__fail")
    check("MCP isError", not r["ok"] and "wrong" in r["content"], r)

r = tool("no_such_tool")
check("unknown tool", not r["ok"], r)
print("\n%s" % ("ALL OK" if not FAILS else "FAILED: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
