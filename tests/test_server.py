"""End-to-end checks for `pithos --server` (Python standard library; jinja2 optional).

  1. the C chat template renderer against the model's own Jinja template (text and tokens)
  2. chat completions, streaming, multi-turn prompt reuse, tool calls, completions, stop strings

usage: python tests/test_server.py [http://127.0.0.1:8080] [--template-only]"""
import json
import sys
import time
import urllib.request

BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8080").rstrip("/")
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FAILS = []


def call(path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=timeout) as r:
        return json.load(r)


def stream(body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"content": "", "reasoning": "", "tool_calls": [], "finish": None, "timings": None, "usage": None, "progress": 0}
    with HTTP.open(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                raise RuntimeError(ev["error"])
            if ev.get("prompt_progress"):
                out["progress"] += 1
            out["timings"] = ev.get("timings") or out["timings"]
            out["usage"] = ev.get("usage") or out["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
                out["tool_calls"] += d.get("tool_calls") or []
                out["finish"] = ch.get("finish_reason") or out["finish"]
    return out


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ 1. template

def jinja_render(template, messages, tools=None, **kw):
    import jinja2
    from jinja2 import nodes
    from jinja2.ext import Extension, loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class Generation(Extension):  # transformers' {% generation %} tag, rendered as its body
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, loopcontrols])
    # transformers' tojson: no key sorting, non-ASCII kept
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, tools=tools, bos_token="<|startoftext|>",
                                            add_generation_prompt=True, **kw)


TOOL = {"type": "function", "function": {"name": "get_order_status", "description": "Stato di un ordine",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_order_status",
        "arguments": {"order_id": "RE-42", "note": "l'utente \\ \"urgente\"\nriga 2", "n": 3, "flag": True, "x": None,
                      "tags": ["a", "b"], "opt": {"k": 1.5}}}}

TEMPLATE_CASES = {
    "single user": dict(messages=[{"role": "user", "content": "Ciao, come stai?"}]),
    "system + multi-turn": dict(messages=[
        {"role": "system", "content": "Sei un assistente conciso."},
        {"role": "user", "content": "Capitale della Francia?"},
        {"role": "assistant", "content": "Parigi.", "reasoning_content": "Domanda facile."},
        {"role": "user", "content": "E della Germania?"}]),
    "content with </think>": dict(messages=[
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "<think>bozza</think>\n\n  Risposta finale  "},
        {"role": "user", "content": "b"}]),
    "content parts + unicode": dict(messages=[
        {"role": "user", "content": [{"type": "text", "text": "Perché è così? "}, {"type": "text", "text": "日本語 ✓ 🙂"}]}]),
    "tools + tool result": dict(tools=[TOOL], messages=[
        {"role": "user", "content": "Dov'è l'ordine RE-42?"},
        {"role": "assistant", "content": "", "reasoning_content": "Serve lo strumento.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"status\": \"spedito\"}"}]),
    "system + tools": dict(tools=[TOOL], messages=[
        {"role": "system", "content": "Usa gli strumenti."}, {"role": "user", "content": "Stato di RE-42?"}]),
    "preserve_thinking": dict(kw={"preserve_thinking": True}, messages=[
        {"role": "user", "content": "x"}, {"role": "assistant", "content": "y", "reasoning_content": "r"},
        {"role": "user", "content": "z"}]),
}


def test_template(template):
    print("template (C renderer vs Jinja):")
    try:
        import jinja2  # noqa: F401
    except ImportError:
        print("  skip: jinja2 not installed")
        return
    for name, case in TEMPLATE_CASES.items():
        kw = case.get("kw", {})
        want = jinja_render(template, case["messages"], case.get("tools"), **kw)
        # the API receives tool-call arguments as a JSON string, as OpenAI clients send them
        api_msgs = json.loads(json.dumps(case["messages"]))
        for m in api_msgs:
            for tc in m.get("tool_calls", []):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
        body = {"messages": api_msgs}
        if case.get("tools"):
            body["tools"] = case["tools"]
        if kw:
            body["chat_template_kwargs"] = kw
        got = call("/apply-template", body)
        check(name + " (text)", got["prompt"] == want, "\n    C:     %r\n    jinja: %r" % (got["prompt"], want))
        toks = call("/tokenize", {"content": want})["tokens"]
        check(name + " (tokens)", toks == got["tokens"])


# ------------------------------------------------------------------ 2. API and prompt reuse

def test_api():
    print("chat completions:")
    msgs = [{"role": "system", "content": "Rispondi in italiano, in una frase."},
            {"role": "user", "content": "Quanto fa 17 per 23? Dai solo il risultato."}]
    t0 = time.time()
    r = call("/v1/chat/completions", {"messages": msgs, "max_tokens": 1024})
    m = r["choices"][0]["message"]
    print("    turno 1: %.1f s, timings %s" % (time.time() - t0, r["timings"]))
    check("answer is correct", "391" in m["content"], m["content"])
    check("reasoning separated", bool(m.get("reasoning_content")) and "<think>" not in m["content"])
    check("usage reported", r["usage"]["prompt_tokens"] > 10 and r["usage"]["completion_tokens"] > 0)

    msgs += [{"role": "assistant", "content": m["content"]}, {"role": "user", "content": "E 391 diviso 17?"}]
    t0 = time.time()
    s = stream({"messages": msgs, "max_tokens": 1024, "return_progress": True})
    t = s["timings"]
    print("    turno 2: %.1f s, timings %s" % (time.time() - t0, t))
    check("streamed answer", "23" in s["content"], s["content"])
    check("prompt reuse: only the new message is processed", t["cache_n"] > 0 and t["prompt_n"] < 30,
          "cache_n=%d prompt_n=%d" % (t["cache_n"], t["prompt_n"]))
    check("progress events", s["progress"] >= 1)
    check("usage chunk", s["usage"] and s["usage"]["prompt_tokens"] == t["cache_n"] + t["prompt_n"])

    # same request again (regenerate): everything up to the last message is reused
    s2 = stream({"messages": msgs, "max_tokens": 1024})
    t2 = s2["timings"]
    check("regenerate reuses the prompt", t2["prompt_n"] < 30, "prompt_n=%d" % t2["prompt_n"])

    print("tool calls:")
    tmsgs = [{"role": "user", "content": "Qual è lo stato dell'ordine RE-42? Usa lo strumento."}]
    s = stream({"messages": tmsgs, "tools": [TOOL], "max_tokens": 1024})
    calls = s["tool_calls"]
    ok = len(calls) == 1 and calls[0]["function"]["name"] == "get_order_status" and \
        json.loads(calls[0]["function"]["arguments"]).get("order_id") == "RE-42"
    check("tool call parsed", ok, json.dumps(calls))
    check("finish_reason tool_calls", s["finish"] == "tool_calls", str(s["finish"]))
    if ok:
        tmsgs += [{"role": "assistant", "content": s["content"] or None, "tool_calls": calls},
                  {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"order_id": "RE-42", "status": "spedito"})}]
        s = stream({"messages": tmsgs, "tools": [TOOL], "max_tokens": 1024})
        check("answer uses the tool result", "sped" in s["content"].lower(), s["content"])
        check("tool turn reused from cache", s["timings"]["prompt_n"] < 40, "prompt_n=%d" % s["timings"]["prompt_n"])

    print("completions:")
    r = call("/v1/completions", {"prompt": "<|startoftext|>La capitale d'Italia è", "max_tokens": 8, "temperature": 0})
    check("raw completion", "Rom" in r["choices"][0]["text"], r["choices"][0]["text"])
    r = call("/v1/completions", {"prompt": "<|startoftext|>Conta: 1, 2, 3, 4,", "max_tokens": 20, "temperature": 0, "stop": [" 7"]})
    txt = r["choices"][0]["text"]
    check("stop string", " 7" not in txt and "6" in txt, repr(txt))

    print("errors:")
    try:
        call("/v1/chat/completions", {"messages": []})
        check("empty messages rejected", False)
    except urllib.error.HTTPError as e:
        check("empty messages rejected", e.code == 400)


if __name__ == "__main__":
    props = call("/props")
    test_template(props["chat_template"])
    if "--template-only" not in sys.argv:
        test_api()
    print("\n%s" % ("ALL OK" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
