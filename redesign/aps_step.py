"""STEP export of a Fusion design's latest version through the Manufacturing Data Model API
(ComponentVersion.derivatives). Generation runs on Autodesk's servers; the design is not changed.

    python3 redesign/aps_step.py "T-Rex Final Version" Main --modified 2024-04-01 --out cad/incoming/final_version/Main_v24.step
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aps_fusion as A  # noqa: E402

GQL = "https://developer.api.autodesk.com/mfg/graphql"


def gql(q, v=None):
    req = urllib.request.Request(GQL, data=json.dumps({"query": q, "variables": v or {}}).encode(),
                                 headers={"Authorization": "Bearer " + A.access_token(), "Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        sys.exit(f"GraphQL HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
    if r.get("errors"):
        sys.exit("GraphQL errors: " + json.dumps(r["errors"])[:600])
    return r["data"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("item")
    ap.add_argument("--modified", default=None, help="pick the item whose last modification date starts with this")
    ap.add_argument("--format", default="STEP", choices=["STEP", "STL", "OBJ"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hubs = gql("{ hubs { results { id name } } }")["hubs"]["results"]
    hub = next(h["id"] for h in hubs if h["name"] == A.TARGET_HUB)
    proj = next(p["id"] for p in gql("query($h:ID!){ projects(hubId:$h, pagination:{limit:50}) { results { id name } } }",
                                     {"h": hub})["projects"]["results"] if p["name"] == a.project)
    items = gql("query($p:ID!){ itemsByProject(projectId:$p, pagination:{limit:50}) { results { __typename id name lastModifiedOn } } }",
                {"p": proj})["itemsByProject"]["results"]
    cands = [i for i in items if i["name"] == a.item and i["__typename"] == "DesignItem"
             and (not a.modified or i["lastModifiedOn"].startswith(a.modified))]
    if len(cands) != 1:
        sys.exit(f"{len(cands)} matching design items: {[(i['name'], i['lastModifiedOn']) for i in cands]}")
    q = """query($h:ID!,$i:ID!,$f:[OutputFormatEnum]){ item(hubId:$h,itemId:$i){ ... on DesignItem {
             tipVersion { versionNumber } tipRootComponentVersion { name
               derivatives(derivativeInput:{outputFormat:$f, generate:true}) { outputFormat status signedUrl } } } } }"""
    for attempt in range(60):
        d = gql(q, {"h": hub, "i": cands[0]["id"], "f": [a.format]})["item"]
        der = d["tipRootComponentVersion"]["derivatives"][0]
        print(f"v{d['tipVersion']['versionNumber']} {d['tipRootComponentVersion']['name']}: {der['status']}", flush=True)
        if der["status"] == "SUCCESS" and der["signedUrl"]:
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(der["signedUrl"], timeout=600) as r, open(out, "wb") as f:
                f.write(r.read())
            print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return
        if der["status"] == "FAILED":
            sys.exit("export failed on the server")
        time.sleep(20)
    sys.exit("timed out waiting for the export")


if __name__ == "__main__":
    main()
