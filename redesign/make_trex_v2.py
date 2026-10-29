"""Build the Trex v2 simulation asset: assets/robots/trex_v2/{Trex.urdf, meshes/*.stl, robot_config.yaml}.

Sources:
  redesign/trex_v2_design.yaml              every parameter (labelled [CAD] / [CONCEPT] / [ASSUMED] / [EST])
  outputs/redesign/cad/v2/asm/*.stl          the new parts (redesign/cad/v2_parts.py)
  outputs/redesign/step/main_v24/            the unchanged as-built parts (redesign/step_parts.py, asbuilt.py)
  assets/robots/trex/                        head/tail meshes; servo model, actuator fit, sim gains (robot_config)

URDF conventions are those of assets/robots/trex: same link and joint names, joint axes, all rpy 0,
link frames at the joints, meshes in mm in the assembly frame with visual origin = -(link position),
scale 0.001. So the policy interface (8 actuated joints, 2 passive ankles) is unchanged.

Inertials: every link is the sum of its parts, each part a uniform solid of its own mass:
  printed PLA parts: density x (shell + infill x core) from the design's print model
  sole pads (PLA): solid; servos: 52 g each; battery and electronics deck: boxes of their estimated mass
  4-bar couplers: not URDF links (as in trex); their mass is booked to the link they move with
  (upper idler -> tibia link, knee coupler + lower idler -> ankle link).

    python redesign/cad/v2_parts.py && python redesign/make_trex_v2.py
"""

from __future__ import annotations

import collections
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.asbuilt import group_of, load_parts, servo_axes, SERVO_TO_JOINT, to_urdf  # noqa: E402

D = yaml.safe_load((ROOT / "redesign" / "trex_v2_design.yaml").read_text())
TREX = ROOT / "assets" / "robots" / "trex"
OUT = ROOT / "assets" / "robots" / "trex_v2"
CAD = ROOT / "outputs" / "redesign" / "cad" / "v2" / "asm"
YS = D["frame"]["y_shift_from_trex_mm"]
LINKS = ["base_link", "head", "tail", "left_femur", "left_tibia", "left_ankle", "left_foot",
         "right_femur", "right_tibia", "right_ankle", "right_foot"]


def printed_kg(m: trimesh.Trimesh) -> float:
    pm = D["print_mass"]
    v, a = m.volume / 1000.0, m.area / 100.0
    shell = min(v, a * pm["wall_mm"] / 10.0)
    return pm["pla_g_cm3"] * (shell + (v - shell) * pm["infill"]) / 1000.0


def box_mesh(size, centre):
    b = trimesh.creation.box(extents=size)
    b.apply_translation(centre)
    return b


def mirror_y(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = m.copy()
    m.vertices[:, 1] *= -1
    m.invert()
    return m


class Part:
    def __init__(self, name, mesh, kg, in_mesh=True):
        self.name, self.mesh, self.kg, self.in_mesh = name, mesh, kg, in_mesh


def asbuilt_parts():
    """Left-leg / torso parts of Main v24 in the v2 frame (shifted +YS in y), by role."""
    recs, z = load_parts()
    asb = json.loads((ROOT / "outputs" / "redesign" / "asbuilt.json").read_text())
    t = np.array(asb["t"])
    ax = servo_axes(recs)
    role = {}
    for jn, test in SERVO_TO_JOINT.items():
        for n, a in ax.items():
            if test(a["point"], a["dir"]):
                role[n] = jn
    out = collections.defaultdict(list)
    for r in recs:
        v = z[f"v_{r['i']}"]
        if len(v) == 0:
            continue
        g = group_of(r["path"])
        m = trimesh.Trimesh(to_urdf(v.astype(float), t) + [0, YS, 0], z[f"f_{r['i']}"], process=True)
        m.merge_vertices()
        if g.startswith("servo["):
            name, tag = g.rsplit(":", 1)
            jn = role.get(name)
            if jn is None or jn.startswith("right"):
                continue
            key = jn.replace("left_", "").replace("_joint", "")
            out[f"servo_{key}_{'body' if tag == 'body' else 'horn'}"].append(m)
        elif g.startswith("left:"):
            out[g.split(":")[1]].append(m)
        elif g.startswith("Torso Bracket"):
            c = m.centroid
            if abs(c[0]) > 40:  # head / tail servo brackets (the hip-servo ones are replaced by the pelvis)
                out["torso_bracket_headtail"].append(m)
    return {k: trimesh.util.concatenate(v) for k, v in out.items()}


def rotz180_about(m: trimesh.Trimesh, x, y) -> trimesh.Trimesh:
    m = m.copy()
    m.vertices[:, 0] = 2 * x - m.vertices[:, 0]
    m.vertices[:, 1] = 2 * y - m.vertices[:, 1]
    return m


def build_parts():
    ab = asbuilt_parts()
    h, lg = D["hip"], D["leg"]
    y_asb = 70.87 + YS  # as-built left hip axis (v2 frame)
    dy = h["axis_y_mm"] - y_asb
    dz_leg = -(lg["hip_to_femur_mm"] - 55.0)  # parts below the hip bracket drop with the femur axis
    sv = D["servo"]["mass_g"] / 1000.0
    P = collections.defaultdict(list)

    def mv(m, ddy, ddz):
        m = m.copy()
        m.apply_translation([0, ddy, ddz])
        return m

    # ---- base_link
    P["base_link"].append(Part("pelvis", trimesh.load(str(CAD / "pelvis.stl")), None))
    P["base_link"][-1].kg = printed_kg(P["base_link"][-1].mesh)
    hip_body = rotz180_about(ab["servo_hip_body"], h["axis_x_mid_mm"], y_asb)  # shaft end inboard
    hip_body = mv(hip_body, dy, 0.0)
    P["base_link"] += [Part("left_hip_servo", hip_body, sv), Part("right_hip_servo", mirror_y(hip_body), sv),
                       Part("head_servo", ab["servo_head_body"], sv), Part("tail_servo", ab["servo_tail_body"], sv)]
    tb = ab["torso_bracket_headtail"]
    P["base_link"].append(Part("head_tail_servo_brackets", tb, printed_kg(tb)))
    bt = D["battery"]
    z_plate = D["pelvis"]["plate_z_mm"][1]
    bx = np.array(bt["box_mm"])
    bc = np.array([bt["centre_xy_mm"][0], bt["centre_xy_mm"][1], z_plate + bx[2] / 2])
    P["base_link"].append(Part("battery", box_mesh(bx, bc), bt["mass_g"] / 1000.0))
    el = D["electronics"]
    dx_ = np.array(el["deck_box_mm"])
    dc = np.array([bc[0], 0.0, z_plate + bx[2] + el["deck_gap_mm"] + dx_[2] / 2])
    P["base_link"].append(Part("electronics_deck", box_mesh(dx_, dc), sum(el["mass_g"].values()) / 1000.0))
    # ---- legs (left; right = mirror)
    L = collections.defaultdict(list)
    hb = trimesh.load(str(CAD / "left_hip_bracket.stl"))
    hip_horn = mv(rotz180_about(ab["servo_hip_horn"], h["axis_x_mid_mm"], y_asb), dy, 0.0)
    L["femur"] += [Part("hip_bracket", hb, printed_kg(hb)), Part("hip_servo_horns", hip_horn, printed_kg(hip_horn)),
                   Part("femur_servo", mv(ab["servo_femur_body"], dy, dz_leg), sv)]
    th = mv(ab["Bracket_Femur"], dy, dz_leg)
    fh = mv(ab["servo_femur_horn"], dy, dz_leg)
    ui = mv(ab["Femur_Tibia_Coupler"], dy, dz_leg)
    L["tibia"] += [Part("thigh_bracket", th, printed_kg(th)), Part("tibia_servo", mv(ab["servo_tibia_body"], dy, dz_leg), sv),
                   Part("femur_servo_horns", fh, printed_kg(fh)), Part("upper_idler", ui, printed_kg(ui), in_mesh=False)]
    sh = mv(ab["Bracket_Tibia"], dy, dz_leg)
    tbh = mv(ab["servo_tibia_horn"], dy, dz_leg)
    kc = mv(ab["Coupler_Femur"], dy, dz_leg)
    li = mv(ab["Coupler_Tibia_Feet"], dy, dz_leg)
    L["ankle"] += [Part("shank_bracket", sh, printed_kg(sh)), Part("tibia_servo_horns", tbh, printed_kg(tbh)),
                   Part("knee_coupler", kc, printed_kg(kc), in_mesh=False), Part("lower_idler", li, printed_kg(li), in_mesh=False)]
    ft = trimesh.load(str(CAD / "left_foot.stl"))
    pad = trimesh.load(str(CAD / "left_sole_pad.stl"))
    L["foot"] += [Part("foot", ft, printed_kg(ft)), Part("sole_pad", pad, pad.volume / 1000.0 * D["print_mass"]["pla_g_cm3"] / 1000.0)]
    for k, parts in L.items():
        P[f"left_{k}"] = parts
        P[f"right_{k}"] = [Part(p.name, mirror_y(p.mesh), p.kg, p.in_mesh) for p in parts]
    # ---- head / tail (unchanged meshes)
    for k in ("head", "tail"):
        m = trimesh.load(str(TREX / "meshes" / f"{k}.stl"))
        m.apply_translation([0, YS, 0])
        P[k].append(Part(k, m, D["head_tail"][f"{k}_kg"]))
    return P


def joint_positions():
    h, lg = D["hip"], D["leg"]
    J = {"head_joint": ("base_link", "head", [75.102, 15.379 + YS, 241.619], [0, 1, 0]),
         "tail_joint": ("base_link", "tail", [-93.508, 16.876 + YS, 239.673], [0, 1, 0])}
    for s, sg in (("left", 1), ("right", -1)):
        hip = np.array([lg["axis_x_mm"], sg * h["axis_y_mm"], h["axis_z_mm"]])
        fem = hip + [0, 0, -lg["hip_to_femur_mm"]]
        tib = fem + [0, 0, -lg["thigh_mm"]]
        ank = tib + [0, 0, -lg["shank_mm"]]
        J[f"{s}_hip_joint"] = ("base_link", f"{s}_femur", hip.tolist(), [-1, 0, 0] if s == "left" else [1, 0, 0])
        J[f"{s}_femur_joint"] = (f"{s}_femur", f"{s}_tibia", fem.tolist(), [0, -1, 0])
        J[f"{s}_tibia_joint"] = (f"{s}_tibia", f"{s}_ankle", tib.tolist(), [0, 1, 0])
        J[f"{s}_ankle_joint"] = (f"{s}_ankle", f"{s}_foot", ank.tolist(), [0, 1, 0])
    return J


def inertial(parts, origin_mm):
    """Mass, COM [m] and inertia about the COM [kg m^2] in the link frame (translation only)."""
    M = sum(p.kg for p in parts)
    coms, Is = [], []
    for p in parts:
        m = p.mesh.copy()
        m.apply_scale(0.001)
        if not m.is_volume:
            m = m.convex_hull  # robust mass properties for a non-watertight mesh (conservative inertia)
        m.density = p.kg / m.volume
        coms.append(m.center_mass)
        Is.append((p.kg, m.center_mass, m.moment_inertia))
    c = sum(k * cm for k, cm, _ in Is) / M
    I = np.zeros((3, 3))
    for k, cm, Ic in Is:
        r = cm - c
        I += Ic + k * (r @ r * np.eye(3) - np.outer(r, r))
    return M, c - np.asarray(origin_mm) / 1000.0, I


def main():
    P = build_parts()
    J = joint_positions()
    origin = {"base_link": np.zeros(3)}
    for jn, (par, ch, pos, ax) in J.items():
        origin[ch] = np.asarray(pos, float)
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    report = {}
    links_xml = []
    for ln in LINKS:
        parts = P[ln]
        M, com, I = inertial(parts, origin[ln])
        mesh = trimesh.util.concatenate([p.mesh for p in parts if p.in_mesh])
        mesh.export(str(OUT / "meshes" / f"{ln}.stl"))
        o = -origin[ln] / 1000.0
        report[ln] = {"mass_kg": float(round(M, 4)), "com_m": com.round(5).tolist(),
                      "parts": {p.name: float(round(p.kg * 1000, 1)) for p in parts}}
        links_xml.append(f'''<link name="{ln}">
  <inertial>
    <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>
    <mass value="{M:.6f}"/>
    <inertia ixx="{I[0, 0]:.9e}" iyy="{I[1, 1]:.9e}" izz="{I[2, 2]:.9e}" ixy="{I[0, 1]:.9e}" iyz="{I[1, 2]:.9e}" ixz="{I[0, 2]:.9e}"/>
  </inertial>
  <visual>
    <origin xyz="{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}" rpy="0 0 0"/>
    <geometry>
      <mesh filename="meshes/{ln}.stl" scale="0.001 0.001 0.001"/>
    </geometry>
    <material name="silver"/>
  </visual>
  <collision>
    <origin xyz="{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}" rpy="0 0 0"/>
    <geometry>
      <mesh filename="meshes/{ln}.stl" scale="0.001 0.001 0.001"/>
    </geometry>
  </collision>
</link>''')
    joints_xml = []
    for jn, (par, ch, pos, ax) in J.items():
        rel = (np.asarray(pos) - origin[par]) / 1000.0
        joints_xml.append(f'''<joint name="{jn}" type="continuous">
  <origin xyz="{rel[0]:.6f} {rel[1]:.6f} {rel[2]:.6f}" rpy="0 0 0"/>
  <parent link="{par}"/>
  <child link="{ch}"/>
  <axis xyz="{ax[0]:.1f} {ax[1]:.1f} {ax[2]:.1f}"/>
</joint>''')
    urdf = ('<?xml version="1.0" ?>\n<!-- Trex v2, written by redesign/make_trex_v2.py from redesign/trex_v2_design.yaml. '
            'Do not edit by hand. -->\n<robot name="Trex">\n\n<material name="silver">\n  <color rgba="0.700 0.700 0.700 1.000"/>\n</material>\n\n'
            + "\n\n".join(links_xml) + "\n\n" + "\n\n".join(joints_xml) + "\n\n</robot>\n")
    (OUT / "Trex.urdf").write_text(urdf)
    write_robot_config(report)
    tot = sum(r["mass_kg"] for r in report.values())
    report["total_kg"] = round(tot, 4)
    (OUT / "build_report.json").write_text(json.dumps(report, indent=1))
    for ln, r in report.items():
        if ln != "total_kg":
            print(f"[trex_v2] {ln:12s} {r['mass_kg'] * 1000:6.1f} g  COM {np.round(np.array(r['com_m']) * 1000, 1)} mm  {r['parts']}")
    print(f"[trex_v2] total {tot:.3f} kg -> {OUT}")


def write_robot_config(report):
    rc = yaml.safe_load((TREX / "robot_config.yaml").read_text())
    rc = copy.deepcopy(rc)
    jd = D["joints_deg"]
    rc["mass"] = {"base_link_kg": float(round(report["base_link"]["mass_kg"], 4)), "base_link_inertia": "urdf", "link_kg": {}}
    rc["joints"]["default_limit_deg"] = 45.0
    sides = ("left", "right")
    rc["joints"]["overrides_deg"] = {**{f"{s}_hip_joint": float(jd["scale"]["hip"]) for s in sides},
                                     **{f"{s}_tibia_joint": float(jd["scale"]["tibia"]) for s in sides}}
    rc["joints"]["range_deg"] = {**{f"{s}_hip_joint": [float(x) for x in jd["range"]["hip"]] for s in sides},
                                 **{f"{s}_tibia_joint": [float(x) for x in jd["range"]["tibia"]] for s in sides}}
    rc["ankle_linkage"]["coupled_limit_deg"] = [float(x) for x in jd["ankle_coupled"]]
    lg = D["leg"]
    rc["ankle_linkage"]["upper_parallelogram"]["z_mm"] = lg["thigh_mm"]
    rc["ankle_linkage"]["lower_parallelogram"]["z_mm"] = lg["shank_mm"]
    rc["sim_self_collision"] = {"pairs": D["self_collision"]["pairs"]}
    bt = D["battery"]
    rc["power"] = {"battery": "TalentCell YB1206000, 11.1 V nominal (12.6-9 V), 6000 mAh", "battery_kg": bt["mass_g"] / 1000.0,
                   "battery_max_current_a": bt["max_current_a"],
                   "servo_bus_v": rc["actuator_model"]["measured_voltage_v"],
                   "note": "servo bus fed by a DC-DC buck-boost: 3 A at ~11 V caps the bus near 4.5 A at 7.1 V before losses"}
    rc["source"] = {"script": "redesign/make_trex_v2.py", "design": "redesign/trex_v2_design.yaml",
                       "masses": "ESTIMATES (printed parts at the design's infill, datasheet servos, estimated electronics)"}
    header = ("# Trex v2 robot configuration. Written by redesign/make_trex_v2.py from redesign/trex_v2_design.yaml\n"
              "# (geometry, ranges, masses) and assets/robots/trex/robot_config.yaml (servo model, fitted actuator,\n"
              "# sim gains, sensors). Edit the design file and regenerate; do not edit by hand. Masses are estimates.\n")
    (OUT / "robot_config.yaml").write_text(header + yaml.safe_dump(rc, sort_keys=False, width=110))


if __name__ == "__main__":
    main()
