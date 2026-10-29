"""Audit of the current design from the data in hand (meshes, URDF, robot_config, MuJoCo model).

    python redesign/audit.py   -> prints the tables, writes outputs/redesign/audit.json

1. URDF inertials vs the meshes (volume, implied density, centre of mass, inertia).
2. Mass as simulated (MuJoCo model) per link, total, centre of mass in three poses.
3. Bottom-up estimate of the real masses (redesign/mass_estimates.yaml: ESTIMATES).
4. Geometry: joint positions, segment lengths, hip spacing, sole patch, support polygon.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.statics import POSES, Statics  # noqa: E402

A = ROOT / "assets" / "robots" / "trex"


def urdf_vs_mesh():
    root = ET.parse(A / "Trex.urdf").getroot()
    rows = {}
    for link in root.findall("link"):
        n = link.get("name")
        inr = link.find("inertial")
        m = float(inr.find("mass").get("value"))
        com = np.array(list(map(float, inr.find("origin").get("xyz").split())))
        I = inr.find("inertia")
        Iu = np.array([[float(I.get(k)) for k in r] for r in (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))])
        off = np.array(list(map(float, link.find("visual/origin").get("xyz").split())))
        me = trimesh.load(str(A / "meshes" / f"{n}.stl"))
        me.apply_scale(0.001)
        me.apply_translation(off)
        v = me.volume
        me.density = m / v
        rows[n] = {"urdf_mass_kg": m, "mesh_volume_cm3": v * 1e6, "implied_density_g_cm3": m / v / 1000,
                   "com_diff_mm": float(np.linalg.norm(com - me.center_mass) * 1000),
                   "inertia_ratio_diag": (np.diag(me.moment_inertia) / np.diag(Iu)).round(4).tolist(),
                   "watertight": bool(me.is_watertight), "extents_mm": (me.extents * 1000).round(1).tolist()}
    return rows


def bottom_up(audit_rows):
    e = yaml.safe_load((ROOT / "redesign" / "mass_estimates.yaml").read_text())
    sv = e["servo"]
    vblock = float(np.prod(sv["cad_block_m"]))
    out = {}
    for n, cls in e["part_class"].items():
        ns = e["servos_per_link"].get(n, 0)
        vprint = audit_rows[n]["mesh_volume_cm3"] * 1e-6 - ns * vblock
        lo, hi = e["fill_fraction"][cls]
        servo = ns * (sv["mass_kg"] + sv["horn_screws_kg"])
        out[n] = {"servos": ns, "printed_volume_cm3": vprint * 1e6,
                  "estimate_kg": [servo + vprint * e["pla_density_kg_m3"] * lo, servo + vprint * e["pla_density_kg_m3"] * hi],
                  "urdf_kg": audit_rows[n]["urdf_mass_kg"]}
    t = np.array(list(e["torso_items"].values())).sum(0)
    out["base_link"] = {"estimate_kg": t.tolist(), "items": e["torso_items"], "model_kg": 1.0}
    tot = np.array([v["estimate_kg"] for v in out.values()]).sum(0)
    out["total_estimate_kg"] = tot.tolist()
    return out


def geometry(st: Statics):
    root = ET.parse(A / "Trex.urdf").getroot()
    j = {jj.get("name"): np.array(list(map(float, jj.find("origin").get("xyz").split()))) for jj in root.findall("joint")}
    g = {
        "hip_axes_y_mm": [j["left_hip_joint"][1] * 1000, j["right_hip_joint"][1] * 1000],
        "hip_axis_spacing_mm": (j["left_hip_joint"][1] - j["right_hip_joint"][1]) * 1000,
        "hip_axis_height_above_sole_mm": None,
        "hip_to_femur_axis_mm": (np.linalg.norm(j["left_femur_joint"]) * 1000, j["left_femur_joint"] * 1000),
        "thigh_femur_to_tibia_axis_mm": abs(j["left_tibia_joint"][2]) * 1000,
        "shank_tibia_to_ankle_axis_mm": abs(j["left_ankle_joint"][2]) * 1000,
        "leg_lateral_jog_mm": [j["left_femur_joint"][1] * 1000, j["left_ankle_joint"][1] * 1000],
        "head_axis_mm": (j["head_joint"] * 1000).tolist(), "tail_axis_mm": (j["tail_joint"] * 1000).tolist(),
    }
    st.set_pose({})
    sole = st.sole["left"]
    g["sole_patch"] = {"area_cm2": sole.area * 1e4, "hull_area_cm2": sole.convex_hull.area * 1e4,
                       "x_range_mm_foot_frame": [sole.bounds[0] * 1000, sole.bounds[2] * 1000],
                       "y_range_mm_foot_frame": [sole.bounds[1] * 1000, sole.bounds[3] * 1000],
                       "sole_below_ankle_axis_mm": -st.sole_z["left"] * 1000}
    hipz = st.d.xpos[st.m.body("left_femur").id][2]
    g["hip_axis_height_above_sole_mm"] = (hipz - st.sole_polygon_world("left")[1]) * 1000
    return g


def main():
    rows = urdf_vs_mesh()
    print("1. URDF inertials vs meshes")
    for n, r in rows.items():
        print(f"   {n:12s} {r['urdf_mass_kg']*1000:7.1f} g  V {r['mesh_volume_cm3']:6.1f} cm3  rho {r['implied_density_g_cm3']:.3f}  "
              f"dCOM {r['com_diff_mm']:.2f} mm  I ratio {r['inertia_ratio_diag']}  watertight {r['watertight']}")
    st = Statics()
    sim = {}
    for b in range(1, st.m.nbody):
        sim[st.m.body(b).name] = float(st.m.body_mass[b])
    print(f"2. simulated masses (MuJoCo): total {st.total_mass():.3f} kg; " + ", ".join(f"{k} {v*1000:.0f} g" for k, v in sim.items()))
    poses = {}
    for pn, p in POSES.items():
        st.set_pose(p)
        poses[pn] = {"com_mm": (st.com() * 1000).round(1).tolist(), "com_above_sole_mm": st.com_height_above_sole() * 1000}
        ds = st.double_support(p)
        poses[pn]["support_margin_mm"] = ds.margins["support_margin_m"] * 1000
        poses[pn]["support_x_mm"] = (np.array(ds.margins["support_x_m"]) * 1000).round(1).tolist()
        poses[pn]["support_y_mm"] = (np.array(ds.margins["support_y_m"]) * 1000).round(1).tolist()
        print(f"   pose {pn:10s} COM {poses[pn]['com_mm']} mm, {poses[pn]['com_above_sole_mm']:.1f} mm above soles, "
              f"double-support margin {poses[pn]['support_margin_mm']:.1f} mm, support x {poses[pn]['support_x_mm']} y {poses[pn]['support_y_mm']}")
    bu = bottom_up(rows)
    print("3. bottom-up mass ESTIMATE [low, high] kg vs model")
    for n, v in bu.items():
        if n == "total_estimate_kg":
            continue
        ref = v.get("urdf_kg", v.get("model_kg"))
        print(f"   {n:12s} {np.round(v['estimate_kg'], 3).tolist()}  (model {ref:.3f})")
    print(f"   total        {np.round(bu['total_estimate_kg'], 3).tolist()}  (model {st.total_mass():.3f})")
    geo = geometry(st)
    print("4. geometry", json.dumps(geo, default=lambda x: np.asarray(x).round(2).tolist(), indent=1))
    out = {"urdf_vs_mesh": rows, "sim_masses_kg": sim, "poses": poses, "bottom_up": bu, "geometry": geo}
    (ROOT / "outputs" / "redesign" / "audit.json").write_text(json.dumps(out, indent=1, default=lambda x: np.asarray(x).tolist()))


if __name__ == "__main__":
    main()
