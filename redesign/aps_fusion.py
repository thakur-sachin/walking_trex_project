"""Read-only access to Sachin's Fusion 360 data through Autodesk Platform Services (APS).

    python3 redesign/aps_fusion.py login     # opens the browser once; stores a token locally
    python3 redesign/aps_fusion.py hubs      # hubs, their projects, and the files in Dev_designs
    python3 redesign/aps_fusion.py download "T-Rex Final Version" --out cad/incoming/final_version

Credentials: ~/.config/aps/trex.env (APS_CLIENT_ID, APS_CLIENT_SECRET, APS_CALLBACK_URL), chmod 600.
Token: ~/.config/aps/trex_token.json (chmod 600), refreshed automatically. Neither is ever printed.
Scope is data:read only: this script cannot change anything in Fusion.

3-legged OAuth: the app acts as Sachin, so it sees every hub his account belongs to (the app is not
tied to a hub). The callback URL must be registered on the app in the APS developer portal, and the
Data Management API enabled. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

CONF = Path.home() / ".config" / "aps"
ENV_FILE = CONF / "trex.env"
TOKEN_FILE = CONF / "trex_token.json"
API = "https://developer.api.autodesk.com"
SCOPE = "data:read"
TARGET_HUB = "Dev_designs"


def load_env() -> dict:
    if not ENV_FILE.exists():
        sys.exit(f"missing {ENV_FILE}")
    if ENV_FILE.stat().st_mode & 0o077:
        sys.exit(f"{ENV_FILE} is readable by others: chmod 600 {ENV_FILE}")
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("APS_CLIENT_ID", "APS_CLIENT_SECRET", "APS_CALLBACK_URL") if not env.get(k)]
    if missing:
        sys.exit(f"{ENV_FILE}: fill in {', '.join(missing)}")
    return env


def save_token(tok: dict) -> None:
    tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
    CONF.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(tok, f)


def token_request(env: dict, form: dict) -> dict:
    basic = base64.b64encode(f"{env['APS_CLIENT_ID']}:{env['APS_CLIENT_SECRET']}".encode()).decode()
    req = urllib.request.Request(f"{API}/authentication/v2/token", data=urllib.parse.urlencode(form).encode(),
                                 headers={"Authorization": f"Basic {basic}",
                                          "Content-Type": "application/x-www-form-urlencoded",
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"token request failed: HTTP {e.code} {e.read().decode(errors='replace')[:300]}")


def login() -> None:
    env = load_env()
    cb = urllib.parse.urlparse(env["APS_CALLBACK_URL"])
    if cb.hostname not in ("localhost", "127.0.0.1"):
        sys.exit("APS_CALLBACK_URL must point at localhost for this script")
    state = secrets.token_urlsafe(16)
    url = f"{API}/authentication/v2/authorize?" + urllib.parse.urlencode(
        {"response_type": "code", "client_id": env["APS_CLIENT_ID"], "redirect_uri": env["APS_CALLBACK_URL"],
         "scope": SCOPE, "state": state})
    result = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path != cb.path:
                self.send_response(404)
                self.end_headers()
                return
            q = urllib.parse.parse_qs(u.query)
            result.update({k: v[0] for k, v in q.items()})
            ok = result.get("state") == state and "code" in result
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(("Signed in. You can close this tab." if ok else
                              f"Sign-in failed: {result.get('error', 'state mismatch')}").encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer((cb.hostname, cb.port or 80), Handler)
    print("Opening the Autodesk sign-in page. If no browser opens, visit:\n" + url, flush=True)
    webbrowser.open(url)
    srv.timeout = 300
    deadline = time.time() + 300
    while "code" not in result and "error" not in result and time.time() < deadline:
        srv.handle_request()
    srv.server_close()
    if result.get("state") != state or "code" not in result:
        sys.exit(f"sign-in failed: {result.get('error_description') or result.get('error') or 'timed out or state mismatch'}")
    tok = token_request(env, {"grant_type": "authorization_code", "code": result["code"],
                              "redirect_uri": env["APS_CALLBACK_URL"]})
    save_token(tok)
    print(f"signed in; token stored in {TOKEN_FILE} (refresh token: {'yes' if tok.get('refresh_token') else 'no'})")


def access_token() -> str:
    if not TOKEN_FILE.exists():
        sys.exit("not signed in: run `python3 redesign/aps_fusion.py login` first")
    tok = json.loads(TOKEN_FILE.read_text())
    if time.time() < tok.get("expires_at", 0):
        return tok["access_token"]
    if not tok.get("refresh_token"):
        sys.exit("token expired and no refresh token: run login again")
    new = token_request(load_env(), {"grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "scope": SCOPE})
    new.setdefault("refresh_token", tok["refresh_token"])
    save_token(new)
    return new["access_token"]


def get(path: str) -> dict:
    req = urllib.request.Request(path if path.startswith("http") else API + path,
                                 headers={"Authorization": f"Bearer {access_token()}", "Accept": "application/vnd.api+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GET {path}: HTTP {e.code} {e.read().decode(errors='replace')[:300]}") from None


def get_all(path: str) -> list:
    out, nxt = [], path
    while nxt:
        page = get(nxt)
        out += page.get("data", [])
        nxt = (page.get("links", {}).get("next") or {}).get("href")
    return out


def walk(project_id: str, folder_id: str, depth: int, indent: str, max_depth: int) -> None:
    for e in get_all(f"/data/v1/projects/{project_id}/folders/{folder_id}/contents"):
        a = e["attributes"]
        name = a.get("displayName") or a.get("name")
        if e["type"] == "folders":
            print(f"{indent}[folder] {name}")
            if depth < max_depth:
                walk(project_id, e["id"], depth + 1, indent + "  ", max_depth)
        else:
            ext = (a.get("extension") or {}).get("type", "").split(":")[-1]
            print(f"{indent}{name}  ({ext})")


def hubs(max_depth: int) -> None:
    hs = get_all("/project/v1/hubs")
    if not hs:
        print("no hubs visible to this account")
    for h in hs:
        a = h["attributes"]
        kind = (a.get("extension") or {}).get("type", "").split(":")[-1]
        print(f"hub: {a['name']}  ({kind}, region {a.get('region', '?')})")
        try:
            projects = get_all(f"/project/v1/hubs/{h['id']}/projects")
        except RuntimeError as e:
            print(f"  cannot list projects: {e}")
            continue
        for p in projects:
            print(f"  project: {p['attributes']['name']}")
            if a["name"] != TARGET_HUB:
                continue
            try:
                for f in get_all(f"/project/v1/hubs/{h['id']}/projects/{p['id']}/topFolders"):
                    print(f"    [folder] {f['attributes'].get('displayName') or f['attributes']['name']}")
                    walk(p["id"], f["id"], 1, "      ", max_depth)
            except RuntimeError as e:
                print(f"    cannot list folders: {e}")
    if TARGET_HUB not in [h["attributes"]["name"] for h in hs]:
        print(f"\n{TARGET_HUB} is not among the hubs this account can see")


def find_project(name: str) -> tuple[str, str]:
    hub = next(h for h in get_all("/project/v1/hubs") if h["attributes"]["name"] == TARGET_HUB)
    p = next(p for p in get_all(f"/project/v1/hubs/{hub['id']}/projects") if p["attributes"]["name"] == name)
    return hub["id"], p["id"]


def items_in(project_id: str, folder_id: str, path: str = ""):
    """Yield (folder path, item) for every item under a folder, recursively."""
    for e in get_all(f"/data/v1/projects/{project_id}/folders/{folder_id}/contents"):
        name = e["attributes"].get("displayName") or e["attributes"].get("name")
        if e["type"] == "folders":
            yield from items_in(project_id, e["id"], f"{path}{name}/")
        else:
            yield path, e


def download_object(storage_id: str, dest: Path) -> int:
    """Download an OSS object (urn:adsk.objects:os.object:<bucket>/<key>) via a signed S3 URL."""
    bucket, key = storage_id.split(":")[-1].split("/", 1)
    signed = get(f"/oss/v2/buckets/{bucket}/objects/{urllib.parse.quote(key, safe='')}/signeds3download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(signed["url"], timeout=300) as r, open(dest, "wb") as f:
        n = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            n += len(chunk)
    return n


def download(project: str, out: Path, version: int | None, only: str | None) -> None:
    """Latest version (or --version) of every item in a project: plain files as they are, Fusion
    designs as their native .f3d. Writes manifest.json (item, version, date, size) next to them."""
    _, pid = find_project(project)
    manifest = []
    for top in get_all(f"/project/v1/hubs/{find_project(project)[0]}/projects/{pid}/topFolders"):
        for path, item in items_in(pid, top["id"]):
            name = item["attributes"]["displayName"]
            if only and name != only:
                continue
            vs = get_all(f"/data/v1/projects/{pid}/items/{item['id']}/versions")
            v = (next((x for x in vs if x["attributes"]["versionNumber"] == version), None) if version
                 else max(vs, key=lambda x: x["attributes"]["versionNumber"]))
            if v is None:
                continue
            a = v["attributes"]
            storage = v.get("relationships", {}).get("storage", {}).get("data", {}).get("id")
            if not storage:
                print(f"  skip {path}{name} v{a['versionNumber']}: no stored file")
                continue
            fname = name if "." in name else f"{name}_v{a['versionNumber']}.f3d"
            dest = out / path / fname
            n = download_object(storage, dest)
            manifest.append({"path": str(dest.relative_to(out)), "item": name, "version": a["versionNumber"],
                             "created": a["createTime"], "bytes": n, "version_urn": v["id"]})
            print(f"  {dest.relative_to(out)}  v{a['versionNumber']}  {n / 1e6:.1f} MB")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("login")
    h = sub.add_parser("hubs")
    h.add_argument("--depth", type=int, default=3, help="folder levels to list inside Dev_designs projects")
    d = sub.add_parser("download", help="latest version of every item in a Dev_designs project")
    d.add_argument("project")
    d.add_argument("--out", required=True)
    d.add_argument("--version", type=int, default=None, help="this version number instead of the latest")
    d.add_argument("--only", default=None, help="only the item with this name")
    a = ap.parse_args()
    if a.command == "login":
        login()
    elif a.command == "hubs":
        hubs(a.depth)
    else:
        download(a.project, Path(a.out).expanduser(), a.version, a.only)


if __name__ == "__main__":
    main()
