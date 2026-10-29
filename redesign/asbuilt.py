"""As-built CAD (Fusion "T-Rex Final Version", Main v24 STEP) vs the simulation URDF.

Needs `python redesign/step_parts.py cad/incoming/final_version/Main_v24.step --tag main_v24 --cylinders`
first. Everything here is in the URDF assembly frame (x forward, y left, z up, mm), after the fixed
STEP -> URDF map fitted in `frame_map()`.

  python redesign/asbuilt.py   -> outputs/redesign/asbuilt.json and a printed report

1. frame map STEP -> URDF from the servo output axes, residual of every URDF joint origin/axis
2. part -> URDF link assignment (surface proximity to the URDF link meshes at the zero pose)
3. the 4-bar: every pivot from the cylindrical faces of the couplers, brackets and feet
"""

from __future__ import annotations

import collections
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
STEP_DIR = ROOT / "outputs" / "redesign" / "step" / "main_v24"
URDF = ROOT / "assets" / "robots" / "trex" / "Trex.urdf"
MESH_DIR = ROOT / "assets" / "robots" / "trex" / "meshes"

# STEP frame: +y forward (head), +z up, left = -x. URDF frame: x forward, y left, z up.
R_STEP_TO_URDF = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def load_parts():
    recs = json.loads((STEP_DIR / "parts.json").read_text())
    z = np.load(STEP_DIR / "meshes.npz")
    return recs, z


def group_of(path: list[str]) -> str:
    """Physical part name (occurrence path collapsed to the part that matters)."""
    p = " / ".join(path)
    for key in ("Bracket_hip", "Bracket_Femur", "Bracket_Tibia", "Coupler_Femur", "Coupler_Tibia_Feet",
                "Femur_Tibia_Coupler", "Main_Trex_Feet"):
        if key in p:
            side = "left" if "Left" in path[0] else "right"
            return f"{side}:{key}"
    if "LX-16A Servo" in p:
        tag = "Shaft Adapter" if "Shaft Adapter" in p else "Motor Adapter" if "Motor Adapter" in p else "body"
        si = [i for i, s in enumerate(path) if "LX-16A Servo" in s][0]
        return f"servo[{' / '.join(path[:si + 1])}]:{tag}"
    if p.startswith("Screws"):
        return "screws"
    return path[0].split(":")[0]


def urdf_joints():
    root = ET.parse(URDF).getroot()
    J = {}
    parent = {}
    for j in root.findall("joint"):
        parent[j.find("child").get("link")] = (j.find("parent").get("link"), j.get("name"))
        J[j.get("name")] = {"xyz": np.array(list(map(float, j.find("origin").get("xyz").split()))) * 1000,
                            "axis": np.array(list(map(float, j.find("axis").get("xyz").split()))),
                            "parent": j.find("parent").get("link"), "child": j.find("child").get("link")}
    # zero-pose world position of each joint origin (all rpy are 0)
    def world(link):
        if link == "base_link":
            return np.zeros(3)
        par, jn = parent[link]
        return world(par) + J[jn]["xyz"]
    for n, j in J.items():
        j["world"] = world(j["child"])
    return J


def servo_axes(recs):
    S = collections.defaultdict(dict)
    for r in recs:
        g = group_of(r["path"])
        if g.startswith("servo["):
            name, tag = g.rsplit(":", 1)
            if tag != "body":
                S[name][tag] = np.array(r["com_mm"])
            else:
                S[name].setdefault("body", []).append(r)
    out = {}
    for n, s in S.items():
        d = s["Shaft Adapter"] - s["Motor Adapter"]
        out[n] = {"point": (s["Shaft Adapter"] + s["Motor Adapter"]) / 2, "dir": d / np.linalg.norm(d)}
    return out


SERVO_TO_JOINT = {  # by position in the STEP (left leg at x = -147, right at x = 0; head +y, tail -y)
    "head_joint": lambda p, d: abs(d[0]) > 0.99 and p[1] > 50 and p[2] > 20,
    "tail_joint": lambda p, d: abs(d[0]) > 0.99 and p[1] < -50 and p[2] > 20,
    "left_hip_joint": lambda p, d: abs(d[1]) > 0.99 and p[0] < -70,
    "right_hip_joint": lambda p, d: abs(d[1]) > 0.99 and p[0] > -70,
    "left_femur_joint": lambda p, d: abs(d[0]) > 0.99 and p[0] < -70 and -80 < p[2] < -30,
    "right_femur_joint": lambda p, d: abs(d[0]) > 0.99 and p[0] > -70 and -80 < p[2] < -30,
    "left_tibia_joint": lambda p, d: abs(d[0]) > 0.99 and p[0] < -70 and p[2] < -100,
    "right_tibia_joint": lambda p, d: abs(d[0]) > 0.99 and p[0] > -70 and p[2] < -100,
}


def frame_map(recs):
    """Translation t such that urdf = R @ step + t, fitted on the joint axes (coordinates normal to
    each axis only; the position along an axis is arbitrary). Returns t and per-joint residuals."""
    J = urdf_joints()
    ax = servo_axes(recs)
    match = {}
    for jn, test in SERVO_TO_JOINT.items():
        c = [n for n, a in ax.items() if test(a["point"], a["dir"])]
        assert len(c) == 1, (jn, c)
        match[jn] = ax[c[0]]
    # ankle pivots: the foot's cylinder coaxial with the shank's bottom hole
    for side in ("left", "right"):
        cyl = [c for r in recs if group_of(r["path"]) == f"{side}:Bracket_Tibia" for c in r["cylinders"] if abs(c["d"][0]) > 0.99]
        low = min(cyl, key=lambda c: c["p"][2])
        match[f"{side}_ankle_joint"] = {"point": np.array([low["p"][0], low["p"][1], low["p"][2]]), "dir": np.array(low["d"])}
    rows, A, b = {}, [], []
    for jn, m in match.items():
        pu = J[jn]["world"]
        du = R_STEP_TO_URDF @ m["dir"]
        P = np.eye(3) - np.outer(du, du)  # project out the axis direction
        A.append(P)
        b.append(P @ (pu - R_STEP_TO_URDF @ m["point"]))
    t = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    for jn, m in match.items():
        du = R_STEP_TO_URDF @ m["dir"]
        p = R_STEP_TO_URDF @ m["point"] + t
        diff = J[jn]["world"] - p
        diff_perp = diff - du * (diff @ du)
        ang = np.degrees(np.arccos(min(1.0, abs(du @ J[jn]["axis"] / np.linalg.norm(J[jn]["axis"])))))
        rows[jn] = {"urdf_mm": J[jn]["world"].round(2).tolist(), "asbuilt_axis_point_mm": p.round(2).tolist(),
                    "offset_normal_to_axis_mm": diff_perp.round(2).tolist(), "offset_norm_mm": float(np.linalg.norm(diff_perp)),
                    "axis_angle_deg": float(ang)}
    return t, rows


def to_urdf(v, t):
    return v @ R_STEP_TO_URDF.T + t


def assign_links(recs, z, t, tol=1.0):
    """Each STEP solid -> URDF link whose zero-pose mesh surface is closest to most of its points."""
    links = {}
    for f in MESH_DIR.glob("*.stl"):
        links[f.stem] = trimesh.load(str(f))  # assembly frame, mm
    from scipy.spatial import cKDTree
    trees = {n: cKDTree(trimesh.sample.sample_surface_even(m, 60000, seed=1)[0]) for n, m in links.items()}
    out = []
    for r in recs:
        v = z[f"v_{r['i']}"]
        if len(v) == 0:
            continue
        pts = to_urdf(v.astype(float), t)
        if len(pts) > 2000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 2000, replace=False)]
        best = {n: float(np.mean(tr.query(pts)[0] < tol)) for n, tr in trees.items()}
        n = max(best, key=best.get)
        out.append({"i": r["i"], "group": group_of(r["path"]), "link": n if best[n] > 0.05 else None,
                    "match_frac": best[n], "volume_cm3": r["volume_mm3"] / 1000})
    return out


def pivots(recs, t):
    """Lateral-axis holes/pins of the leg parts, in the URDF frame (x, z), radius."""
    res = collections.defaultdict(list)
    for r in recs:
        g = group_of(r["path"])
        if not (g.startswith("left:") or g.startswith("right:")):
            continue
        seen = {}
        for c in r["cylinders"]:
            if abs(c["d"][0]) < 0.99 or c["r"] > 9.0:
                continue
            p = R_STEP_TO_URDF @ np.array(c["p"]) + t
            key = (round(p[0], 1), round(p[2], 1))
            seen[key] = max(seen.get(key, 0.0), c["r"])
        res[g] += [{"x_mm": k[0], "z_mm": k[1], "r_mm": v} for k, v in sorted(seen.items())]
    return dict(res)


def main():
    recs, z = load_parts()
    t, rows = frame_map(recs)
    print("1. frame map: urdf = R step + t, t =", t.round(2))
    for jn, r in rows.items():
        print(f"   {jn:18s} offset normal to axis {np.round(r['offset_normal_to_axis_mm'], 2)} mm (|.| {r['offset_norm_mm']:.2f}), "
              f"axis angle {r['axis_angle_deg']:.3f} deg")
    asg = assign_links(recs, z, t)
    by = collections.defaultdict(lambda: collections.defaultdict(float))
    for a in asg:
        by[a["group"]][a["link"]] += a["volume_cm3"]
    print("2. STEP part -> URDF link (volume cm3 per link)")
    for g, d in sorted(by.items()):
        print(f"   {g[:70]:70s} " + ", ".join(f"{k}:{v:.1f}" for k, v in d.items()))
    pv = pivots(recs, t)
    print("3. pivots (URDF frame x, z; lateral axes)")
    for g, ps in sorted(pv.items()):
        print(f"   {g:28s} " + "; ".join(f"({p['x_mm']:.1f},{p['z_mm']:.1f}) r{p['r_mm']:.2f}" for p in ps))
    out = {"t": t.tolist(), "R": R_STEP_TO_URDF.tolist(), "joints": rows, "assignment": asg,
           "by_group": {g: dict(d) for g, d in by.items()}, "pivots": pv}
    (ROOT / "outputs" / "redesign" / "asbuilt.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
