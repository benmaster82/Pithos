"""Deploy Pithos to an Xbox in Developer Mode through the Device Portal REST API (stdlib only).

    python xbox/deploy.py --xbox CONSOLE-IP install           # Pithos.appx + VCLibs dependency
    python xbox/deploy.py --xbox CONSOLE-IP use MODEL.gguf    # switch model: see below
    python xbox/deploy.py --xbox CONSOLE-IP upload MODEL.gguf # into the app's LocalState
    python xbox/deploy.py --xbox CONSOLE-IP launch
    python xbox/deploy.py --xbox CONSOLE-IP log               # LocalState\\pithos.log
    python xbox/deploy.py --xbox CONSOLE-IP files             # list LocalState
    python xbox/deploy.py --xbox CONSOLE-IP get mic-raw.wav   # download a file from LocalState
    python xbox/deploy.py --xbox CONSOLE-IP touch mic-probe   # an empty file there (a probe switch)
    python xbox/deploy.py --xbox CONSOLE-IP rm mic-probe      # delete a file from LocalState
    python xbox/deploy.py --xbox CONSOLE-IP stop

More engine options go in a pithos.args file in LocalState (for example "-t 6" or "-c 8192"),
read when the app starts: upload it, then stop and launch the app.

"use" puts one model on the console in a single step: it stops the app, deletes the other .gguf
files from LocalState (the app's storage holds one model at a time), uploads the model's audio
projector (mmproj-..., found next to the model) before the model itself, since the app starts
as soon as a model is complete, writes the recommended pithos.args (and for a speech model the
speech lines of pithos.env, keeping its other lines), and starts the app. Files already there
with the same size are not uploaded again. The saved conversations of the other model
(MODEL.gguf.pithos-cache) would not leave room for the new one: they move to the PC, into
build-xbox\\conversations, and come back when that model is chosen again (--drop-cache deletes
them instead). The recommended pithos.args caps them at --cache-mb 256 on the console.
--dry-run only prints the steps, --keep-args leaves pithos.args and pithos.env alone, --args "..."
chooses pithos.args, --no-launch does not start the app.

The portal uses a self-signed certificate, so TLS verification is off. --user/--password are
only needed when Device Portal authentication is enabled in Dev Home.
"""
import argparse, base64, http.client, http.cookiejar, json, os, ssl, sys, time, urllib.parse, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APPX = os.path.join(ROOT, "build-xbox", "Pithos.appx")
VCLIBS = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft SDKs", "Windows Kits",
                      "10", "ExtensionSDKs", "Microsoft.VCLibs", "14.0", "Appx", "Retail", "x64", "Microsoft.VCLibs.x64.14.00.appx")
PACKAGE_NAME = "Pithos.Server"


class Portal:
    def __init__(self, host, port, user, password):
        self.host, self.port = host, port
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.auth = None
        if user:
            self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=self.ctx),
                                                  urllib.request.HTTPCookieProcessor(self.jar))
        self.request("GET", "/api/os/info")   # sets the CSRF-Token cookie

    def headers(self):
        h = {}
        if self.auth: h["Authorization"] = self.auth
        for c in self.jar:
            if c.name == "CSRF-Token": h["X-CSRF-Token"] = c.value
        return h

    def request(self, method, path, body=None, headers=None, timeout=60):
        h = self.headers()
        h.update(headers or {})
        req = urllib.request.Request(f"https://{self.host}:{self.port}{path}", data=body, method=method, headers=h)
        with self.opener.open(req, timeout=timeout) as r:
            return r.status, r.read()

    def json(self, method, path):
        status, data = self.request(method, path)
        return json.loads(data.decode("utf-8-sig")) if data else {}

    def upload(self, path, files, progress=True):
        """multipart/form-data POST streamed from disk (files: list of local paths)."""
        boundary = uuid.uuid4().hex
        parts = []
        for f in files:
            name = os.path.basename(f)
            head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}"\r\n'
                    f'Content-Type: application/octet-stream\r\n\r\n').encode()
            parts.append((head, f))
        tail = f"--{boundary}--\r\n".encode()
        total = sum(len(h) + os.path.getsize(f) + 2 for h, f in parts) + len(tail)
        conn = http.client.HTTPSConnection(self.host, self.port, context=self.ctx, timeout=600)
        conn.putrequest("POST", path)
        h = self.headers()
        h["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        h["Content-Length"] = str(total)
        cookies = "; ".join(f"{c.name}={c.value}" for c in self.jar)
        if cookies: h["Cookie"] = cookies
        for k, v in h.items(): conn.putheader(k, v)
        conn.endheaders()
        sent, t0, last = 0, time.time(), 0
        for head, f in parts:
            conn.send(head); sent += len(head)
            with open(f, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk: break
                    conn.send(chunk); sent += len(chunk)
                    if progress and time.time() - last > 2:
                        last = time.time()
                        rate = sent / max(last - t0, 1e-3) / 1e6
                        print(f"\r  {sent / 1e6:8.1f} / {total / 1e6:.1f} MB  ({rate:.1f} MB/s)", end="", flush=True)
            conn.send(b"\r\n"); sent += 2
        conn.send(tail)
        r = conn.getresponse()
        body = r.read()
        if progress: print(f"\r  {total / 1e6:8.1f} / {total / 1e6:.1f} MB  done in {time.time() - t0:.0f} s        ")
        if r.status >= 300: raise RuntimeError(f"HTTP {r.status}: {body[:300]!r}")
        return body

    def package(self):
        """the newest installed version (an older one can stay listed until the console lets go of it)"""
        found = [p for p in self.json("GET", "/api/app/packagemanager/packages").get("InstalledPackages", [])
                 if p.get("Name") == PACKAGE_NAME or p.get("PackageFullName", "").startswith(PACKAGE_NAME + "_")]
        ver = lambda p: tuple(p.get("Version", {}).get(k, 0) for k in ("Major", "Minor", "Build", "Revision"))
        return max(found, key=ver) if found else None


def b64(s):
    return base64.b64encode(s.encode()).decode()


def cmd_install(px, a):
    if not os.path.exists(APPX): sys.exit(f"{APPX} is missing: run xbox\\build.cmd first")
    files = [APPX] + ([VCLIBS] if os.path.exists(VCLIBS) else [])
    print("installing", ", ".join(os.path.basename(f) for f in files))
    px.upload("/api/app/packagemanager/package?package=" + urllib.parse.quote(os.path.basename(APPX)), files, progress=False)
    for _ in range(300):
        time.sleep(2)
        try:
            status, data = px.request("GET", "/api/app/packagemanager/state")
        except urllib.error.HTTPError as e:
            sys.exit(f"install failed: {e.read().decode('utf-8', 'replace')}")
        if status == 200:
            info = json.loads(data.decode("utf-8-sig")) if data else {}
            if info.get("Success", True) is False: sys.exit(f"install failed: {info}")
            break
    p = px.package()
    print("installed:", p["PackageFullName"] if p else "(not found in the package list)")


def need_package(px):
    p = px.package()
    if not p: sys.exit("Pithos is not installed on the console: run install first")
    return p


def local_state(p, extra="", sub=""):
    q = {"knownfolderid": "LocalAppData", "packagefullname": p["PackageFullName"], "path": "\\LocalState" + (sub and "\\" + sub)}
    q.update(extra)
    return urllib.parse.urlencode(q)


def cmd_upload(px, a):
    p = need_package(px)
    print(f"uploading {a.file} ({os.path.getsize(a.file) / 1e9:.2f} GB) to LocalState")
    px.upload("/api/filesystem/apps/file?" + local_state(p), [a.file])


def cmd_files(px, a):
    p = need_package(px)
    for it in px.json("GET", "/api/filesystem/apps/files?" + local_state(p)).get("Items", []):
        print(f"{it.get('FileSize', 0):>14,}  {it['Name']}")


def cmd_log(px, a):
    p = need_package(px)
    try:
        status, data = px.request("GET", "/api/filesystem/apps/file?" + local_state(p, {"filename": a.name}))
        sys.stdout.write(data.decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        sys.exit(f"{a.name} not found ({e.code}): has the app been started?")


def cmd_get(px, a):
    p = need_package(px)
    try:
        status, data = px.request("GET", "/api/filesystem/apps/file?" + local_state(p, {"filename": a.name}))
    except urllib.error.HTTPError as e:
        sys.exit(f"{a.name} not found ({e.code})")
    out = a.out or a.name
    with open(out, "wb") as f:
        f.write(data)
    print(f"{a.name} -> {out} ({len(data):,} bytes)")


def cmd_touch(px, a):
    """an empty file in LocalState, e.g. the mic-probe or gpu-probe switch"""
    p = need_package(px)
    tmp = os.path.join(os.environ.get("TEMP", "."), a.name)
    open(tmp, "wb").close()
    px.upload("/api/filesystem/apps/file?" + local_state(p), [tmp], progress=False)
    os.remove(tmp)
    print("created", a.name)


def cmd_rm(px, a):
    p = need_package(px)
    px.request("DELETE", "/api/filesystem/apps/file?" + local_state(p, {"filename": a.name}))
    print("removed", a.name)


def cmd_launch(px, a):
    p = need_package(px)
    px.request("POST", f"/api/taskmanager/app?appid={b64(p['PackageRelativeId'])}&package={b64(p['PackageFullName'])}")
    print("started", p["PackageRelativeId"])


def cmd_stop(px, a):
    p = need_package(px)
    px.request("DELETE", f"/api/taskmanager/app?package={b64(p['PackageFullName'])}")
    print("stopped")


# pithos.args for Game mode: the GPU does the work; with a speech model one more core stays free
# for the audio codec. --cache-mb: the saved conversations' budget (the engine's 4 GB default is for
# a PC; the app's storage is a few GB, and a model and its conversations must fit in it together).
# pithos.env: the speech depth decoder on the GPU, the codec on the engine's threads.
CACHE_MB = 256
ARGS_TEXT, ARGS_SPEECH = f"-t 6 --gpu d3d12 --cache-mb {CACHE_MB}", f"-t 5 --gpu d3d12 --cache-mb {CACHE_MB}"
ENV_SPEECH = {"PITHOS_SPEECH_DX": "1", "PITHOS_SPEECH_CODEC_THREADS": "0"}
# a model's saved conversations are kept on the PC while another model is on the console
CACHE_HOME = os.path.join(ROOT, "build-xbox", "conversations")
CACHE_EXT = ".pithos-cache"


def ls(px, p, sub=""):
    return px.json("GET", "/api/filesystem/apps/files?" + local_state(p, sub=sub)).get("Items", [])


def save_cache(px, p, folder, keep):
    """moves a model's conversations folder from the console to the PC (keep=False: just deletes it)"""
    items = [it for it in ls(px, p, folder) if it.get("Type") != 16]
    dest = os.path.join(CACHE_HOME, folder)
    if keep:
        os.makedirs(dest, exist_ok=True)
        for it in items:
            data = px.request("GET", "/api/filesystem/apps/file?" + local_state(p, {"filename": it["Name"]}, folder))[1]
            if len(data) != it.get("FileSize", len(data)): sys.exit(f"{folder}\\{it['Name']}: short download, nothing deleted")
            with open(os.path.join(dest, it["Name"]), "wb") as f:
                f.write(data)
    for it in items:
        px.request("DELETE", "/api/filesystem/apps/file?" + local_state(p, {"filename": it["Name"]}, folder))
    px.request("DELETE", "/api/filesystem/apps/file?" + local_state(p, {"filename": folder}))
    print(f"conversations {folder}: {len(items)} files " + (f"saved in {dest}, removed from the console" if keep else "deleted"))


def restore_cache(px, p, folder):
    """puts a model's conversations saved by save_cache back on the console; the PC copy is removed
    once every file is there"""
    src = os.path.join(CACHE_HOME, folder)
    files = sorted(os.path.join(src, f) for f in os.listdir(src))
    px.request("POST", "/api/filesystem/apps/folder?" + local_state(p, {"newfoldername": folder}))
    try:
        for f in files:
            px.upload("/api/filesystem/apps/file?" + local_state(p, sub=folder), [f], progress=False)
    except RuntimeError as e:
        print(f"conversations {folder}: could not be put back ({e}); they stay in {src}")
        return
    for f in files: os.remove(f)
    os.rmdir(src)
    print(f"conversations {folder}: {len(files)} files back on the console")


def find_projector(model):
    """the model's audio projector next to it: mmproj-<model file name>, or mmproj-<name without
    the quantization>-*.gguf, or None"""
    folder, name = os.path.split(os.path.abspath(model))
    exact = os.path.join(folder, "mmproj-" + name)
    if os.path.exists(exact): return exact
    stem = name[:-5].rsplit("-", 1)[0] if "-" in name else name[:-5]
    found = sorted(f for f in os.listdir(folder) if f.startswith("mmproj-" + stem + "-") and f.endswith(".gguf"))
    return os.path.join(folder, found[0]) if found else None


def put_text(px, p, name, text):
    tmp = os.path.join(os.environ.get("TEMP", "."), name)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    px.upload("/api/filesystem/apps/file?" + local_state(p), [tmp], progress=False)
    os.remove(tmp)


def get_text(px, p, name):
    try:
        return px.request("GET", "/api/filesystem/apps/file?" + local_state(p, {"filename": name}))[1].decode("utf-8", "replace")
    except urllib.error.HTTPError:
        return ""


def cmd_use(px, a):
    if not os.path.isfile(a.model) or not a.model.lower().endswith(".gguf"): sys.exit(f"{a.model}: not a .gguf file")
    if os.path.basename(a.model).lower().startswith("mmproj"): sys.exit(f"{a.model} is a projector: give the model")
    proj = find_projector(a.model)
    wanted = ([proj] if proj else []) + [a.model]   # the projector first: the app starts once the model is complete
    names = {os.path.basename(f): os.path.getsize(f) for f in wanted}
    p = need_package(px)
    items = ls(px, p)
    there = {it["Name"]: it.get("FileSize", 0) for it in items if it.get("Type") != 16}
    drop = [n for n in there if n.lower().endswith(".gguf") and names.get(n) != there[n]]
    send = [f for f in wanted if there.get(os.path.basename(f)) != names[os.path.basename(f)]]
    # the conversations of any other model leave the console (saved on the PC unless --drop-cache),
    # and the chosen model's come back if they were saved
    mine = os.path.basename(a.model) + CACHE_EXT
    away = [it["Name"] for it in items if it.get("Type") == 16 and it["Name"].endswith(CACHE_EXT) and it["Name"] != mine]
    back = mine if (os.path.isdir(os.path.join(CACHE_HOME, mine)) and not any(it["Name"] == mine for it in items)) else None
    args = a.args or (ARGS_SPEECH if proj else ARGS_TEXT)
    env = None
    if proj and not a.keep_args:
        keep = [l for l in get_text(px, p, "pithos.env").splitlines() if l.strip() and l.split("=", 1)[0].strip() not in ENV_SPEECH]
        env = "\n".join(keep + [f"{k}={v}" for k, v in ENV_SPEECH.items()]) + "\n"

    print(f"model: {os.path.basename(a.model)}" + (f" with its projector {os.path.basename(proj)} (text and voice)" if proj
                                                    else " (text; no projector next to it, so no voice)"))
    print("steps:")
    print("  stop the app")
    for n in away:
        print(f"  conversations {n}: " + ("delete" if a.drop_cache else f"save in {os.path.join(CACHE_HOME, n)}, then delete"))
    for n in drop: print(f"  delete {n} ({there[n] / 1e9:.2f} GB)")
    for f in wanted:
        n = os.path.basename(f)
        print(f"  upload {n} ({names[n] / 1e9:.2f} GB)" if f in send else f"  keep {n} (already there)")
    if back: print(f"  conversations {back}: put back from {os.path.join(CACHE_HOME, back)}")
    if not a.keep_args:
        print(f"  pithos.args: {args}")
        if env: print("  pithos.env: " + " ".join(env.split()))
    if not a.no_launch: print("  start the app")
    if a.dry_run:
        print("dry run: nothing changed")
        return

    try:
        cmd_stop(px, a)
    except urllib.error.HTTPError:
        print("(the app was not running)")
    for n in away: save_cache(px, p, n, keep=not a.drop_cache)
    for n in drop:
        px.request("DELETE", "/api/filesystem/apps/file?" + local_state(p, {"filename": n}))
        print("removed", n)
    for f in send:
        print(f"uploading {os.path.basename(f)}")
        try:
            px.upload("/api/filesystem/apps/file?" + local_state(p), [f])
        except RuntimeError as e:
            free = "not enough space" in str(e).lower()
            sys.exit(f"upload of {os.path.basename(f)} failed: {e}\n" +
                     ("The console's storage is full. A failed upload leaves its copy in the portal's temporary folder until the "
                      "console restarts: restart it (Settings, or the Device Portal), then run 'use' again; the files already "
                      "there are kept. Crash dumps also take space (Device Portal > App crash dumps)."
                      if free else "Run 'use' again: the files already there are kept."))
    if back: restore_cache(px, p, back)
    if not a.keep_args:
        put_text(px, p, "pithos.args", args + "\n")
        if env: put_text(px, p, "pithos.env", env)
        print("wrote pithos.args" + (" and pithos.env" if env else ""))
    if not a.no_launch:
        try:
            cmd_launch(px, a)
        except urllib.error.HTTPError as e:
            print(f"could not start the app ({e.code}): start it from Dev Home (a user must be signed in)")
    print("Game mode (Dev Home > Pithos.X Server > View details > App type: Game) is needed for --gpu d3d12.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xbox", required=True, help="console address (Device Portal)")
    ap.add_argument("--port", type=int, default=11443)
    ap.add_argument("--user", default=os.environ.get("XBOX_USER"))
    ap.add_argument("--password", default=os.environ.get("XBOX_PASSWORD"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install")
    m = sub.add_parser("use", help="switch the console to this model")
    m.add_argument("model")
    m.add_argument("--args", help='pithos.args to write (default: "%s", or "%s" with a speech model)' % (ARGS_TEXT, ARGS_SPEECH))
    m.add_argument("--keep-args", action="store_true", help="leave pithos.args and pithos.env as they are")
    m.add_argument("--no-launch", action="store_true", help="do not start the app afterwards")
    m.add_argument("--drop-cache", action="store_true", help="delete other models' saved conversations instead of saving them on the PC")
    m.add_argument("--dry-run", action="store_true", help="print the steps without changing anything")
    u = sub.add_parser("upload"); u.add_argument("file")
    sub.add_parser("files")
    l = sub.add_parser("log"); l.add_argument("name", nargs="?", default="pithos.log")
    g = sub.add_parser("get"); g.add_argument("name"); g.add_argument("-o", "--out")
    t = sub.add_parser("touch"); t.add_argument("name")
    r = sub.add_parser("rm"); r.add_argument("name")
    sub.add_parser("launch")
    sub.add_parser("stop")
    a = ap.parse_args()
    px = Portal(a.xbox, a.port, a.user, a.password)
    {"install": cmd_install, "use": cmd_use, "upload": cmd_upload, "files": cmd_files, "log": cmd_log, "get": cmd_get,
     "touch": cmd_touch, "rm": cmd_rm, "launch": cmd_launch, "stop": cmd_stop}[a.cmd](px, a)


if __name__ == "__main__":
    main()
