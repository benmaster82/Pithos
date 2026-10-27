"""Head-to-head llama.cpp vs pithos on the same model, prompts and sampling.
Each engine runs as an OpenAI-compatible server; engines alternate; the GPU cools below COOL C
(nvidia-smi) before every measured request. Results: JSON lines in versus.jsonl in the current
directory, server logs next to it.

usage: python tests/versus.py [ROUNDS] [llama.cpp|llama.cpp-cuda]
The llama.cpp builds are expected in ../llama (Vulkan) and ../llama-cuda next to pithos/."""
import json, os, subprocess, sys, threading, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = ROOT + r"\models\LFM2.5-24B-A2B-Q4_K_M.gguf"
COOL = 62
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = os.path.abspath("versus.jsonl")

ENGINES = {
    "llama.cpp": ([ROOT + r"\llama\llama-server.exe", "-m", MODEL, "-c", "8192", "-np", "1", "-t", "6", "-tb", "6",
                   "-dev", "Vulkan0", "-fit", "on", "--jinja", "--cache-ram", "0", "--host", "127.0.0.1",
                   "--port", "8095", "--no-webui"], 8095),
    "llama.cpp-cuda": ([ROOT + r"\llama-cuda\llama-server.exe", "-m", MODEL, "-c", "8192", "-np", "1", "-t", "6", "-tb", "6",
                        "-dev", "CUDA0", "-fit", "on", "--jinja", "--cache-ram", "0", "--host", "127.0.0.1",
                        "--port", "8097", "--no-webui"], 8097),
    "pithos": ([ROOT + r"\pithos\build\pithos.exe", "--server", "-m", MODEL, "-c", "8192", "--no-cache",
              "--host", "127.0.0.1", "--port", "8096"], 8096),
}

ESSAY = ("Write a detailed essay on the history of the Roman Empire, from the founding of Rome "
         "to the fall of Constantinople.")
LICENSE = open(ROOT + r"\models\LICENSE", encoding="utf-8", errors="replace").read()[:6500]
LONG = "Read the following license text.\n\n" + LICENSE + "\n\nIn one sentence: what is this document?"
TESTS = [("gen600", ESSAY, 600), ("prompt1500", LONG, 48)]


def gpu_temp():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip())


def cool():
    while gpu_temp() >= COOL:
        time.sleep(5)


def post(port, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(port, prompt, n):
    return post(port, {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0,
                       "seed": 42, "cache_prompt": False, "stream": False})


def working_set_mb(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                         text=True).stdout
    try:
        return int(out.strip().split('","')[-1].strip('"').replace(".", "").replace(",", "").split()[0]) / 1024
    except Exception:
        return 0


def run_engine(name, rnd):
    cmd, port = ENGINES[name]
    log = open(os.path.join(os.path.dirname(OUT), f"versus_{name}.log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    peak = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], working_set_mb(p.pid))
            stop.wait(2)
    threading.Thread(target=sampler, daemon=True).start()
    t0 = time.time()
    try:
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited")
                time.sleep(1)
        load_s = time.time() - t0
        ask(port, "Say hello.", 16)  # warm-up
        for tname, prompt, n in TESTS:
            cool()
            temp0 = gpu_temp()
            t = time.time()
            r = ask(port, prompt, n)
            wall = time.time() - t
            tm = r.get("timings", {})
            msg = r["choices"][0]["message"]
            rec = {"engine": name, "round": rnd, "test": tname, "gpu_start_c": temp0, "gpu_end_c": gpu_temp(),
                   "wall_s": round(wall, 2), "load_s": round(load_s, 1),
                   "prompt_n": tm.get("prompt_n"), "prompt_tps": round(tm.get("prompt_per_second", 0), 2),
                   "gen_n": tm.get("predicted_n"), "gen_tps": round(tm.get("predicted_per_second", 0), 2),
                   "text": ((msg.get("reasoning_content") or "") + "|" + (msg.get("content") or ""))[:160]}
            print(json.dumps(rec), flush=True)
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    finally:
        stop.set()
        p.terminate()
        try:
            p.wait(30)
        except Exception:
            p.kill()
        print(json.dumps({"engine": name, "round": rnd, "peak_ws_mb": round(peak[0])}), flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"engine": name, "round": rnd, "peak_ws_mb": round(peak[0])}) + "\n")
        time.sleep(10)  # let the OS reclaim memory before the next engine


for rnd in range(1, ROUNDS + 1):
    for name in (sys.argv[2] if len(sys.argv) > 2 else "llama.cpp", "pithos"):
        run_engine(name, rnd)
