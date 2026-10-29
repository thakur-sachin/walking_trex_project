"""Mass budget of the AS-BUILT robot from the real parts (Main v24 STEP + the print STLs), per URDF
link, against the URDF and against the earlier bottom-up estimate (redesign/mass_estimates.yaml).
Material and component values: redesign/asbuilt_mass.yaml (every one labelled with its source).

Part choice (see docs/REDESIGN_LOG.md section 7):
  - legs: the LEFT leg parts of Main v24 for both legs; they are identical to the print exports
    (1_v1, 2_v1, 3_v1, coupler_*). The right-leg parts in Main v24 match no print export.
  - head / tail: the print exports head_updated.stl / tail_updated.stl (= the URDF meshes). Main v24
    contains a larger Head v1 / Tail v1 that were not printed.
  - chassis: New_Chassis, with and without its arch (the video shows no arch fitted).

    python redesign/asbuilt_mass.py   -> outputs/redesign/asbuilt_mass.json
"""

from __future__ import annotations

import collections
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.asbuilt import group_of, load_parts, to_urdf  # noqa: E402
from redesign.asbuilt_ranges import EXCLUDE  # noqa: E402

CFG = yaml.safe_load((ROOT / "redesign" / "asbuilt_mass.yaml").read_text())
STL = ROOT / "cad" / "incoming" / "final_version" / "STL Files"


def printed_g(volume_cm3: float, area_cm2: float, infill: float) -> float:
    pm = CFG["print_model"]
    shell = min(volume_cm3, area_cm2 * pm["wall_mm"] / 10.0)
    return CFG["pla_density_g_cm3"] * (shell + (volume_cm3 - shell) * infill)


def main():
    recs, z = load_parts()
    asb = json.loads((ROOT / "outputs" / "redesign" / "asbuilt.json").read_text())
    t = np.array(asb["t"])
    infills = CFG["print_model"]["infill_cases"]
    parts = collections.defaultdict(list)  # link -> [(name, V cm3, A cm2)]

    def add(link, name, mesh):
        parts[link].append((name, float(mesh.volume) / 1000, float(mesh.area) / 100))

    leg_map = {"Bracket_hip": "femur", "Bracket_Femur": "tibia", "Bracket_Tibia": "ankle", "Main_Trex_Feet": "foot",
               "Coupler_Femur": "ankle", "Femur_Tibia_Coupler": "tibia", "Coupler_Tibia_Feet": "ankle"}
    # couplers are not URDF links: the upper idler moves with the thigh's angle and the lower idler
    # with the shank's; they are booked to the link they move with (a mass-budget approximation).
    for r in recs:
        g = group_of(r["path"])
        v = z[f"v_{r['i']}"]
        if len(v) == 0:
            continue
        m = trimesh.Trimesh(v.astype(float), z[f"f_{r['i']}"], process=True)
        m.merge_vertices()
        if g.startswith("left:"):
            key = g.split(":")[1]
            for side in ("left", "right"):
                add(f"{side}_{leg_map[key]}", key, m)
        elif "Adapter" in g:
            # servo horns: printed. Hip-servo horns ride with the femur link, femur-servo horns with the
            # thigh (tibia link), tibia-servo horns with the shank (ankle link); head/tail with head/tail
            pass
        elif g == "New_Chassis":
            add("base_link", "New_Chassis", m)
            mu = trimesh.Trimesh(to_urdf(m.vertices, t), m.faces)
            top = trimesh.intersections.slice_mesh_plane(mu, [0, 0, 1], [0, 0, 245.0], cap=True)
            parts["_chassis_arch_above_245mm"].append(("arch", float(top.volume) / 1000, float(top.area) / 100))
        elif g.startswith("Torso Bracket"):
            add("base_link", "Torso Bracket", m)
        elif g.startswith("controller_case"):
            pass  # in the electronics line
    # horns: 8 servos x (shaft + motor adapter), from the CAD
    ad = {k: [] for k in ("Shaft Adapter", "Motor Adapter")}
    for r in recs:
        g = group_of(r["path"])
        for k in ad:
            if g.endswith(k) and len(z[f"v_{r['i']}"]):
                m = trimesh.Trimesh(z[f"v_{r['i']}"].astype(float), z[f"f_{r['i']}"], process=True)
                ad[k].append((float(m.volume) / 1000, float(m.area) / 100))
    hv = sum(np.mean([x[0] for x in ad[k]]) for k in ad)
    ha = sum(np.mean([x[1] for x in ad[k]]) for k in ad)
    for link in ("head", "tail", "left_femur", "right_femur", "left_tibia", "right_tibia", "left_ankle", "right_ankle"):
        parts[link].append(("servo horns (printed, assumed)", hv, ha))
    for link, f in (("head", "head_updated.stl"), ("tail", "tail_updated.stl")):
        add(link, f, trimesh.load(str(STL / f)))
    # screws seated in a placed part (within 3 mm), steel; booked as one line
    from scipy.spatial import cKDTree
    placed = np.vstack([z[f"v_{r['i']}"] for r in recs if len(z[f"v_{r['i']}"]) and group_of(r["path"]) != "screws"
                        and not group_of(r["path"]).startswith(EXCLUDE) and not r["path"][0].startswith(EXCLUDE)])
    tree = cKDTree(placed)
    screw_v = [r["volume_mm3"] for r in recs if group_of(r["path"]) == "screws" and len(z[f"v_{r['i']}"])
               and tree.query(z[f"v_{r['i']}"])[0].min() < 3.0]
    screws_g = sum(screw_v) / 1000 * CFG["steel_density_g_cm3"]
    # electronics, battery, servos
    sv = CFG["servo"]
    urdf = {l.get("name"): float(l.find("inertial/mass").get("value")) * 1000 for l in ET.parse(ROOT / "assets/robots/trex/Trex.urdf").getroot().findall("link")}
    out = {"links": {}, "infill_cases": infills}
    print(f"{'link':12s} " + " ".join(f"{'infill ' + str(int(i * 100)) + '%':>11s}" for i in infills) + "   URDF   earlier est.")
    est = json.loads((ROOT / "outputs" / "redesign" / "audit.json").read_text())["bottom_up"]
    totals = np.zeros(len(infills))
    for link in ("base_link", "head", "tail", "left_femur", "left_tibia", "left_ankle", "left_foot",
                 "right_femur", "right_tibia", "right_ankle", "right_foot"):
        pr = np.array([sum(printed_g(V, A, i) for _, V, A in parts[link]) for i in infills])
        servo = sv["count_per_link"].get(link, 0) * sv["mass_g"]
        tot = pr + servo
        row = {"printed_g": pr.round(1).tolist(), "servos_g": servo, "total_g": tot.round(1).tolist(),
               "printed_volume_cm3": round(sum(V for _, V, _ in parts[link]), 2), "urdf_g": round(urdf[link], 1),
               "parts": [p[0] for p in parts[link]]}
        out["links"][link] = row
        totals += tot
        e = est.get(link, {}).get("estimate_kg")
        print(f"{link:12s} " + " ".join(f"{x:11.0f}" for x in tot) + f"   {urdf[link] if link != 'base_link' else 1000.0:6.0f}   "
              + (f"{e[0] * 1000:.0f}-{e[1] * 1000:.0f}" if e else ""))
    el = np.array(list(CFG["electronics_g"].values())).sum(0)
    bat = np.array(CFG["battery_g"])
    arch = [printed_g(V, A, i) for _, V, A in parts["_chassis_arch_above_245mm"] for i in infills]
    out.update(electronics_g=el.tolist(), battery_g=bat.tolist(), chassis_arch_g=[round(a, 1) for a in arch],
               chassis_arch_volume_cm3=parts["_chassis_arch_above_245mm"][0][1])
    totals += screws_g
    out["screws_seated"] = {"count": len(screw_v), "g": round(screws_g, 1)}
    print(f"screws seated in parts: {len(screw_v)}, {screws_g:.1f} g steel")
    lo, hi = totals + el[0] + bat[0], totals + el[1] + bat[1]
    out["robot_total_g"] = {"low": lo.round(0).tolist(), "high": hi.round(0).tolist(),
                            "without_arch_low": (lo - np.array(arch)).round(0).tolist(),
                            "without_arch_high": (hi - np.array(arch)).round(0).tolist()}
    tb = np.array(out["links"]["base_link"]["total_g"])
    out["torso_total_g"] = {"low": (tb + el[0] + bat[0]).round(0).tolist(), "high": (tb + el[1] + bat[1]).round(0).tolist(),
                            "without_arch_low": (tb + el[0] + bat[0] - np.array(arch)).round(0).tolist(),
                            "without_arch_high": (tb + el[1] + bat[1] - np.array(arch)).round(0).tolist()}
    print(f"electronics {el.tolist()} g, battery {bat.tolist()} g, chassis arch {np.round(arch, 0).tolist()} g "
          f"({out['chassis_arch_volume_cm3']:.1f} cm3)")
    print("torso total (printed + 4 servos + electronics + battery):", out["torso_total_g"])
    print("robot total:", out["robot_total_g"], " (model 1778 g)")
    (ROOT / "outputs" / "redesign" / "asbuilt_mass.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
