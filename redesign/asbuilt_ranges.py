"""Joint ranges of the AS-BUILT robot (Main v24 STEP, real couplers/idlers, servos, screws), by
collision sweep. Run `step_parts.py ... --cylinders` and `asbuilt.py` first.

Rigid groups of one leg and how they move (URDF frame, pivots from the CAD, see asbuilt.py):
  H   hip bracket + femur servo body + hip servo horns      rotates with the hip only
  Th  thigh bracket + tibia servo body + femur servo horns  R_y(-femur) about the femur axis F
  UI  upper idler (Femur_Tibia_Coupler)                     R_y(-femur) about F' (on H)
  KC  knee coupler (Coupler_Femur)                          translates with the tibia axis T
  Sh  shank bracket + tibia servo horns                     R_y(-femur + tibia) about T
  LI  lower idler (Coupler_Tibia_Feet)                      R_y(-femur + tibia) about T' (on KC)
  Ft  foot                                                  translates with the ankle axis A
  torso: chassis (with its arch), battery, torso brackets, velcro, hip/head/tail servo bodies.
The whole leg then rolls about the hip axis. Screws join the group of the nearest part.
Screws farther than 3 mm from every part are dropped. Excluded: Head v1 / Tail v1 (not the printed versions), the electronics (not placed in the CAD),
a duplicate foot solid.

Method as range_check.py: surface samples of one group inside another's filled 1 mm voxel grid,
counted beyond the zero-pose overlap; contact = more than `MIN_PTS` new points.

    python redesign/asbuilt_ranges.py [--step 2]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.asbuilt import group_of, load_parts, to_urdf  # noqa: E402

VOX = 1.0  # mm
MIN_PTS = 20
SAMPLE_DENSITY = 0.5  # points per mm^2

LEG_PARTS = {"H": ["Bracket_hip"], "Th": ["Bracket_Femur"], "UI": ["Femur_Tibia_Coupler"], "KC": ["Coupler_Femur"],
             "Sh": ["Bracket_Tibia"], "LI": ["Coupler_Tibia_Feet"], "Ft": ["Main_Trex_Feet"]}
EXCLUDE = ("Head v1", "Tail v1", "Assembly Raspberry", "BusLinker", "DC-DC", "Acrylic Plate", "controller_case", "=>[")


def classify(rec, side_of_servo):
    g = group_of(rec["path"])
    if g.startswith(EXCLUDE) or rec["path"][0].startswith(EXCLUDE):
        return None
    for side in ("left", "right"):
        for grp, keys in LEG_PARTS.items():
            if any(g == f"{side}:{k}" for k in keys):
                return f"{side}:{grp}"
    if g.startswith("servo["):
        name, tag = g.rsplit(":", 1)
        role = side_of_servo.get(name)  # (side, joint) or ("torso", ...)
        if role is None:
            return "torso"
        side, joint = role
        if joint == "hip":
            return "torso" if tag == "body" else f"{side}:H"
        if joint == "femur":
            return f"{side}:H" if tag == "body" else f"{side}:Th"
        if joint == "tibia":
            return f"{side}:Th" if tag == "body" else f"{side}:Sh"
        return "torso"
    if g == "screws":
        return "screws"
    return "torso"


def servo_roles(recs, t):
    from redesign.asbuilt import servo_axes, SERVO_TO_JOINT
    ax = servo_axes(recs)
    roles = {}
    for jn, test in SERVO_TO_JOINT.items():
        for n, a in ax.items():
            if test(a["point"], a["dir"]):
                side = "left" if jn.startswith("left") else "right" if jn.startswith("right") else "torso"
                roles[n] = (side, jn.split("_")[1] if side != "torso" else jn)
    return {n: r for n, r in roles.items() if r[0] != "torso"}


class Body:
    def __init__(self, meshes):
        self.mesh = trimesh.util.concatenate(meshes)
        n = min(20000, max(200, int(self.mesh.area * SAMPLE_DENSITY)))
        self.samples = trimesh.sample.sample_surface(self.mesh, n, seed=0)[0]
        # one occupancy grid for the whole group: union of each solid's filled voxels
        pts = np.vstack([m.voxelized(pitch=VOX).fill().points for m in meshes])
        self.lo = pts.min(0) - VOX
        ijk = np.round((pts - self.lo) / VOX).astype(int)
        self.grid = np.zeros(ijk.max(0) + 2, bool)
        self.grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    def inside(self, pts_local):
        ijk = np.round((pts_local - self.lo) / VOX).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.grid.shape)), axis=1)
        return int(self.grid[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]].sum())


def rot_y(a_deg, about):
    R = Rot.from_euler("y", a_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = about - R @ about
    return T


def trans(d):
    T = np.eye(4)
    T[:3, 3] = d
    return T


class Robot:
    def __init__(self, asb):
        recs, z = load_parts()
        t = np.array(asb["t"])
        roles = servo_roles(recs, t)
        meshes = {}
        screws = []
        for r in recs:
            v = z[f"v_{r['i']}"]
            if len(v) == 0:
                continue
            g = classify(r, roles)
            if g is None:
                continue
            m = trimesh.Trimesh(to_urdf(v.astype(float), t), z[f"f_{r['i']}"], process=True)
            m.merge_vertices()
            if g == "screws":
                screws.append(m)
            else:
                meshes.setdefault(g, []).append(m)
        # screws -> group of the nearest part
        cents = {g: cKDTree(np.vstack([mm.vertices for mm in ms])) for g, ms in meshes.items()}
        self.dropped_screws = 0
        for s in screws:  # screws not seated in a part (e.g. by the unplaced electronics) are dropped
            d = {g_: cents[g_].query(s.vertices)[0].min() for g_ in cents}
            g = min(d, key=d.get)
            if d[g] < 3.0:
                meshes[g].append(s)
            else:
                self.dropped_screws += 1
        self.bodies = {g: Body(ms) for g, ms in meshes.items()}
        pv = asb["pivots"]
        self.piv = {}
        for side in ("left", "right"):
            # main pivots from the shank (T top hole, A bottom hole); idler line 50 mm behind
            sh = pv[f"{side}:Bracket_Tibia"]
            T_ = max(sh, key=lambda q: q["r_mm"] if q["z_mm"] > 60 else -1)
            A_ = min(sh, key=lambda q: q["z_mm"])
            ui = pv[f"{side}:Femur_Tibia_Coupler"]
            F_ = [q for q in ui if q["z_mm"] > 120][0]
            Fp = np.array([F_["x_mm"], 0, F_["z_mm"]])
            self.piv[side] = {"F": np.array([T_["x_mm"], 0, F_["z_mm"]]), "Fp": Fp,
                              "T": np.array([T_["x_mm"], 0, T_["z_mm"]]), "A": np.array([A_["x_mm"], 0, A_["z_mm"]]),
                              "Tp": np.array([Fp[0], 0, [q for q in ui if q["z_mm"] < 120][0]["z_mm"]])}
        self.hip = {}
        for side in ("left", "right"):
            pt = np.array(asb["joints"][f"{side}_hip_joint"]["asbuilt_axis_point_mm"])
            self.hip[side] = (pt, np.array([-1.0, 0, 0]) if side == "left" else np.array([1.0, 0, 0]))

    def poses(self, q):
        """q: dict femur/tibia/hip per side (deg, URDF signs) -> 4x4 per group."""
        out = {"torso": np.eye(4)}
        for side in ("left", "right"):
            f, tb, h = q.get(f"{side}_femur", 0.0), q.get(f"{side}_tibia", 0.0), q.get(f"{side}_hip", 0.0)
            P = self.piv[side]
            Rf = rot_y(-f, P["F"])
            T_new = (Rf @ np.r_[P["T"], 1])[:3]
            Rsh = rot_y(-f + tb, P["T"])
            A_new = T_new + (rot_y(-f + tb, np.zeros(3))[:3, :3] @ (P["A"] - P["T"]))
            KCm = trans(T_new - P["T"])
            G = {"H": np.eye(4), "Th": Rf, "UI": rot_y(-f, P["Fp"]), "KC": KCm, "Sh": KCm @ Rsh,
                 "LI": KCm @ rot_y(-f + tb, P["Tp"]), "Ft": trans(A_new - P["A"])}
            pt, ax = self.hip[side]
            Rh = np.eye(4)
            Rh[:3, :3] = Rot.from_rotvec(np.radians(h) * ax).as_matrix()
            Rh[:3, 3] = pt - Rh[:3, :3] @ pt
            for k, M in G.items():
                out[f"{side}:{k}"] = Rh @ M
        return out

    def overlap(self, a, b, P):
        A, B = self.bodies[a], self.bodies[b]
        Ma, Mb = P[a], P[b]
        Mab = np.linalg.inv(Mb) @ Ma  # a-local -> b-local
        Mba = np.linalg.inv(Ma) @ Mb
        pa = A.samples @ Mab[:3, :3].T + Mab[:3, 3]
        pb = B.samples @ Mba[:3, :3].T + Mba[:3, 3]
        return B.inside(pa) + A.inside(pb)


def first_contact(rb, pairs, q_of, angles):
    P0 = rb.poses(q_of(0.0))
    base = {p: rb.overlap(*p, P0) for p in pairs}
    first = {}
    for a in angles:
        P = rb.poses(q_of(a))
        for p in pairs:
            if p in first:
                continue
            if rb.overlap(*p, P) - base[p] > MIN_PTS:
                first[p] = float(a)
        if len(first) == len(pairs):
            break
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--sides", nargs="+", default=["left", "right"], help="legs for the sagittal sweep")
    ap.add_argument("--skip-hip", action="store_true")
    a = ap.parse_args()
    asb = json.loads((ROOT / "outputs" / "redesign" / "asbuilt.json").read_text())
    t0 = time.time()
    rb = Robot(asb)
    print(f"[ranges] groups {sorted(rb.bodies)}; pivots left " +
          ", ".join(f"{k} ({v[0]:.1f},{v[2]:.1f})" for k, v in rb.piv['left'].items()) + f" [{time.time() - t0:.0f}s]")
    res = {"pivots": {s: {k: v.tolist() for k, v in d.items()} for s, d in rb.piv.items()}}
    legs = ["H", "Th", "UI", "KC", "Sh", "LI", "Ft"]
    # --- hip roll, each side alone, torso vs leg groups, and the other leg
    for side in ([] if a.skip_hip else ["left", "right"]):
        other = "right" if side == "left" else "left"
        pairs = [("torso", f"{side}:{g}") for g in legs] + [(f"{side}:{g}", f"{other}:{h}") for g in legs for h in legs]
        for sgn in (1, -1):
            fc = first_contact(rb, pairs, lambda x: {f"{side}_hip": x}, sgn * np.arange(1, 61, 1.0))
            key = f"{side}_hip {'inward(+)' if sgn > 0 else 'outward(-)'}"
            res[key] = {f"{p[0]}~{p[1]}": v for p, v in fc.items()}
            print(f"[ranges] {key:22s} " + (", ".join(f"{p[0]}~{p[1]} {v:+.0f}" for p, v in sorted(fc.items(), key=lambda kv: abs(kv[1]))[:4]) or "none to 60"))
    # both hips inward together
    pairs = [(f"left:{g}", f"right:{h}") for g in legs for h in legs]
    fc = {} if a.skip_hip else first_contact(rb, pairs, lambda x: {"left_hip": x, "right_hip": x}, np.arange(1, 41, 1.0))
    res["both hips inward"] = {f"{p[0]}~{p[1]}": v for p, v in fc.items()}
    print("[ranges] both hips inward  " + ", ".join(f"{p[0]}~{p[1]} {v:+.0f}" for p, v in sorted(fc.items(), key=lambda kv: kv[1])[:4]))
    for sd in a.sides:
        sagittal(rb, sd, legs, a.step, res)
    (ROOT / "outputs" / "redesign" / "asbuilt_ranges.json").write_text(json.dumps(res, indent=1))
    print(f"[ranges] done in {time.time() - t0:.0f}s")


def sagittal(rb, side, legs, step, res):
    """Grid over femur x tibia for one leg: every pair within the leg and with the torso."""
    groups = ["torso"] + [f"{side}:{g}" for g in legs]
    pairs = [p for p in itertools.combinations(groups, 2)]
    P0 = rb.poses({})
    base = {p: rb.overlap(*p, P0) for p in pairs}
    F = np.arange(-90, 90 + 1e-9, step)
    Tt = np.arange(-120, 120 + 1e-9, step)
    free = np.zeros((len(F), len(Tt)), bool)
    first_pair = {}
    for i, f in enumerate(F):
        for j, tb in enumerate(Tt):
            P = rb.poses({f"{side}_femur": f, f"{side}_tibia": tb})
            hit = None
            for p in pairs:
                if rb.overlap(*p, P) - base[p] > MIN_PTS:
                    hit = p
                    break
            free[i, j] = hit is None
            if hit:
                first_pair[(float(f), float(tb))] = f"{hit[0]}~{hit[1]}"
        print(f"[ranges] {side} femur {f:+5.0f}: tibia free " + (f"{Tt[free[i]].min():+.0f}..{Tt[free[i]].max():+.0f}" if free[i].any() else "none"),
              flush=True)
    np.savez(ROOT / "outputs" / "redesign" / f"asbuilt_sagittal_free_{side}.npz", femur=F, tibia=Tt, free=free)
    res[f"sagittal_first_pair_{side}"] = {f"{k[0]},{k[1]}": v for k, v in first_pair.items()}
    ank = (F[:, None] - Tt[None, :])[free]
    res[f"sagittal_summary_{side}"] = {"femur_free_deg": [float(F[free.any(1)].min()), float(F[free.any(1)].max())],
                                       "ankle_free_deg": [float(ank.min()), float(ank.max())]}
    print(f"[ranges] {side}: femur {res[f'sagittal_summary_{side}']['femur_free_deg']}, ankle (femur - tibia) "
          f"{res[f'sagittal_summary_{side}']['ankle_free_deg']}")


if __name__ == "__main__":
    main()
