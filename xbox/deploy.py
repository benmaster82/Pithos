"""Deploy Pithos to an Xbox in Developer Mode through the Device Portal REST API (stdlib only).

    python xbox/deploy.py --xbox 192.168.1.23 install           # Pithos.appx + VCLibs dependency
    python xbox/deploy.py --xbox 192.168.1.23 upload MODEL.gguf # into the app's LocalState
    python xbox/deploy.py --xbox 192.168.1.23 launch
    python xbox/deploy.py --xbox 192.168.1.23 log               # LocalState\\pithos.log
    python xbox/deploy.py --xbox 192.168.1.23 files             # list LocalState
    python xbox/deploy.py --xbox 192.168.1.23 stop

More engine options go in a pithos.args file in LocalState (for example "-t 6" or "-c 8192"),
read when the app starts: upload it, then stop and launch the app.

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
        for p in self.json("GET", "/api/app/packagemanager/packages").get("InstalledPackages", []):
            if p.get("Name") == PACKAGE_NAME or p.get("PackageFullName", "").startswith(PACKAGE_NAME + "_"):
                return p
        return None


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


def local_state(p, extra=""):
    q = {"knownfolderid": "LocalAppData", "packagefullname": p["PackageFullName"], "path": "\\LocalState"}
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


def cmd_launch(px, a):
    p = need_package(px)
    px.request("POST", f"/api/taskmanager/app?appid={b64(p['PackageRelativeId'])}&package={b64(p['PackageFullName'])}")
    print("started", p["PackageRelativeId"])


def cmd_stop(px, a):
    p = need_package(px)
    px.request("DELETE", f"/api/taskmanager/app?package={b64(p['PackageFullName'])}")
    print("stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xbox", required=True, help="console address (Device Portal)")
    ap.add_argument("--port", type=int, default=11443)
    ap.add_argument("--user", default=os.environ.get("XBOX_USER"))
    ap.add_argument("--password", default=os.environ.get("XBOX_PASSWORD"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install")
    u = sub.add_parser("upload"); u.add_argument("file")
    sub.add_parser("files")
    l = sub.add_parser("log"); l.add_argument("name", nargs="?", default="pithos.log")
    sub.add_parser("launch")
    sub.add_parser("stop")
    a = ap.parse_args()
    px = Portal(a.xbox, a.port, a.user, a.password)
    {"install": cmd_install, "upload": cmd_upload, "files": cmd_files, "log": cmd_log,
     "launch": cmd_launch, "stop": cmd_stop}[a.cmd](px, a)


if __name__ == "__main__":
    main()
