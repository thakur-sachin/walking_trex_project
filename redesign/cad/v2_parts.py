"""Parametric CAD (build123d) of the Trex v2 parts that change: pelvis plate with the mirrored hip-servo
mounts, hip bracket (deeper cross-plate, narrow-stance leg plane), narrow foot with its TPU sole pad.
Every dimension comes from redesign/trex_v2_design.yaml (labelled [CAD] / [CONCEPT] / [ASSUMED]).

FIRST PASS for approval: printable solids with the right interfaces (axes, horn screw pattern, pivot
holes, servo pockets), no cosmetic shaping, not yet fitted to real screws. Unchanged parts (thigh and
shank brackets, idlers, servos, head, tail, head/tail servo brackets) are reused from Main v24.

    python redesign/cad/v2_parts.py
      -> outputs/redesign/cad/v2/asm/<part>.stl     in the v2 assembly frame (mm), for the asset builder
      -> outputs/redesign/cad/v2/print/<part>.stl   laid flat for printing, checked against the bed
      -> outputs/redesign/cad/v2/parts.json         volume, area, bounding box, bed fit per part
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import shapely
import yaml
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
D = yaml.safe_load((ROOT / "redesign" / "trex_v2_design.yaml").read_text())
OUT = ROOT / "outputs" / "redesign" / "cad" / "v2"


# ----------------------------------------------------------------------------- helpers
def prism(poly: shapely.Polygon, plane: bd.Plane, depth: float) -> bd.Part:
    """Extrude a shapely polygon (plane-local u, v coordinates) along the plane normal by `depth`."""
    poly = shapely.geometry.polygon.orient(poly.simplify(0.05), 1.0)
    pts = list(poly.exterior.coords)[:-1]
    face = bd.Face(bd.Wire.make_polygon([bd.Vector(*p, 0) for p in pts], close=True))
    for ring in poly.interiors:
        hole = bd.Face(bd.Wire.make_polygon([bd.Vector(*p, 0) for p in list(ring.coords)[:-1]], close=True))
        face = face - hole
    return bd.extrude(plane * face, amount=depth)


def box(x0, x1, y0, y1, z0, z1) -> bd.Part:
    return bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * bd.Box(x1 - x0, y1 - y0, z1 - z0)


def cyl_x(y, z, r, x0, x1) -> bd.Part:
    return bd.Pos((x0 + x1) / 2, y, z) * (bd.Rot(0, 90, 0) * bd.Cylinder(r, x1 - x0))


def cyl_y(x, z, r, y0, y1) -> bd.Part:
    return bd.Pos(x, (y0 + y1) / 2, z) * (bd.Rot(90, 0, 0) * bd.Cylinder(r, y1 - y0))


def cyl_z(x, y, r, z0, z1) -> bd.Part:
    return bd.Pos(x, y, (z0 + z1) / 2) * bd.Cylinder(r, z1 - z0)


# --------------------------------------------------------------------------------- parts
def hip_bracket_left() -> bd.Part:
    """U-fork on the hip servo horns (hip roll axis along x at y = hip.axis_y), a cross-plate
    bracket_drop below the axis, femur-servo cradle side plates, and the arm to the upper idler pivot."""
    h, lg, sv = D["hip"], D["leg"], D["servo"]
    yA, zA = h["axis_y_mm"], h["axis_z_mm"]
    zF = zA - lg["hip_to_femur_mm"]  # femur axis
    w, t, aw = h["arm_half_width_mm"], h["plate_mm"], h["arm_mm"]
    z_plate_top = zA - h["bracket_drop_mm"]
    z_plate_bot = z_plate_top - t
    xb, xf = h["horn_faces_x_mm"]
    # fork arm profile in (y, z): a lobe around the axis joined to the cross-plate width
    lobe = shapely.Point(yA, zA).buffer(11.0, 64)
    foot_rect = shapely.box(yA - w, z_plate_bot, yA + w, z_plate_top + 6.0)
    arm2d = unary_union([lobe, foot_rect]).convex_hull
    arms = prism(arm2d, bd.Plane.YZ, -aw).moved(bd.Location((xb, 0, 0))) + prism(arm2d, bd.Plane.YZ, aw).moved(bd.Location((xf, 0, 0)))
    plate = box(xb - aw, xf + aw, yA - w, yA + w, z_plate_bot, z_plate_top)
    # femur servo cradle: side plates on both sides of the servo (servo shaft along y)
    sx = 12.5 + 1.5  # half the servo's 25 mm + clearance
    x0, x1 = lg["axis_x_mm"] - sx, lg["axis_x_mm"] + sx
    half = sv["body_mm"][2] / 2  # 36.2 / 2 along y
    z_side_bot = zF + lg["side_plate_end_above_axis_mm"]
    sides = box(x0, x1, yA - half - 4.0, yA - half, z_side_bot, z_plate_bot) + box(x0, x1, yA + half, yA + half + 4.0, z_side_bot, z_plate_bot)
    # arm to the upper idler pivot, in the x-z plane, on the inner side plate's y
    ly0, ly1 = yA + lg["idler_lug_y_mm"][0], yA + lg["idler_lug_y_mm"][1]
    xi = lg["axis_x_mm"] - lg["idler_offset_mm"]
    arm_xz = unary_union([shapely.Point(xi, zF).buffer(9.0, 64), shapely.box(x0, z_side_bot, x0 + 10.0, z_plate_bot)]).convex_hull
    idler_arm = prism(arm_xz, bd.Plane.XZ, -(ly1 - ly0)).moved(bd.Location((0, ly0, 0)))
    part = arms + plate + sides + idler_arm
    # holes: horn screws (pattern on both arms), the hip axis clearance, the idler pivot, servo screws
    r = sv["horn_pcd_mm"] / 2
    for dy, dz in ((-r, 0.0), (r, 0.0), (0.0, -r)):
        part -= cyl_x(yA + dy, zA + dz, sv["horn_hole_d_mm"] / 2, xb - aw - 1, xf + aw + 1)
    part -= cyl_x(yA, zA, 3.0, xb - aw - 1, xf + aw + 1)  # [ASSUMED] 6 mm clearance on the axis
    part -= cyl_y(xi, zF, lg["idler_hole_d_mm"] / 2, ly0 - 1, ly1 + 1)
    for dx in (-8.0, 8.0):  # [ASSUMED] 4 x M2 servo screws through the side plates
        for dz in (8.0, 20.0):
            part -= cyl_y(lg["axis_x_mm"] + dx, z_side_bot + dz, 1.1, yA - half - 5, yA + half + 5)
    return part


def foot_outline() -> shapely.Polygon:
    f = D["foot"]
    t = f["toes"]
    hw = f["width_mm"] / 2
    heel = shapely.Point(f["length_mm"][0] + t["heel_radius_mm"], 0).buffer(t["heel_radius_mm"], 64)
    mid = shapely.box(f["length_mm"][0] + t["heel_radius_mm"], -t["mid_half_width_mm"], 22.0, t["mid_half_width_mm"])
    tr = t["toe_width_mm"] / 2
    toes = [shapely.LineString([(10.0, 0.0), (t["mid_tip_x_mm"] - tr, 0.0)]).buffer(tr, 32)]
    for s in (1, -1):
        toes.append(shapely.LineString([(8.0, s * 6.0), (t["side_tip"][0], s * (hw - tr))]).buffer(tr, 32))
    poly = unary_union([heel, mid] + toes).buffer(2.0).buffer(-2.0)
    return poly


def foot_left():
    """Sole plate (PLA) with the heel beam carrying the ankle and idler pivots, and the TPU pad.
    Built in the ankle frame (ankle axis at the origin), then placed under the left ankle."""
    f, lg = D["foot"], D["leg"]
    zs = -lg["ankle_to_sole_mm"]  # floor
    pad_top = zs + f["pad_mm"]
    sole_top = pad_top + f["sole_pla_mm"]
    out = foot_outline()
    sole = prism(out, bd.Plane.XY, f["sole_pla_mm"]).moved(bd.Location((0, 0, pad_top)))
    c = f["edge_chamfer_mm"]
    try:
        sole = bd.chamfer(sole.edges().group_by(bd.Axis.Z)[0], c)
    except Exception:  # chamfer can fail on short toe edges: keep the plain sole and say so
        print("[v2_parts] foot: bottom chamfer skipped")
    b = f["beam"]
    beam2d = unary_union([shapely.box(b["x_mm"][0] + 8, sole_top - 1.0, b["x_mm"][1] - 3, b["top_above_axis_mm"]),
                          shapely.Point(0, 0).buffer(b["top_above_axis_mm"], 64),
                          shapely.Point(-lg["idler_offset_mm"], 0).buffer(b["top_above_axis_mm"], 64),
                          shapely.Point(b["x_mm"][0] + 8, b["top_above_axis_mm"] - 8).buffer(8, 32)]).convex_hull
    th = b["thickness_mm"]
    beam = prism(beam2d, bd.Plane.XZ, -th).moved(bd.Location((0, -th / 2, 0)))
    # the heel overhang below the beam (behind the sole) gets a web down to the sole plane
    web = box(b["x_mm"][0] + 2, f["length_mm"][0] + 10, -th / 2, th / 2, sole_top - 4, 0)
    foot = sole + beam + web
    for x in (0.0, -lg["idler_offset_mm"]):
        foot -= cyl_y(x, 0.0, b["hole_d_mm"] / 2, -th, th)
    pad = prism(out.buffer(-f["pad_inset_mm"]), bd.Plane.XY, f["pad_mm"]).moved(bd.Location((0, 0, zs)))
    place = bd.Location((lg["axis_x_mm"], D["hip"]["axis_y_mm"],
                         D["hip"]["axis_z_mm"] - lg["hip_to_femur_mm"] - lg["thigh_mm"] - lg["shank_mm"]))
    return foot.moved(place), pad.moved(place)


def pelvis() -> bd.Part:
    """Flat plate (as-built outline) with the two mirrored hip-servo mounts hanging under it:
    a top pad over the servo's outboard 25 mm and an end wall on its outer face (the fork arms sweep
    the servo's inboard end, so nothing else is allowed there)."""
    p, h, sv = D["pelvis"], D["hip"], D["servo"]
    x0, x1 = p["plate_x_mm"]
    hy = p["plate_half_y_mm"]
    z0, z1 = p["plate_z_mm"]
    L0 = sv["body_mm"][0]
    y_ear = h["axis_y_mm"] - sv["shaft_from_end_mm"] + L0 + p["hip_mount_wall_mm"]  # ears over the outboard servo ends
    xm0 = h["axis_x_mid_mm"]
    out = unary_union([shapely.box(x0, -hy, x1, hy), shapely.box(xm0 - 16.0, -y_ear, xm0 + 16.0, y_ear)])
    out = out.buffer(-p["corner_r_mm"]).buffer(p["corner_r_mm"], 32).buffer(3.0).buffer(-3.0)
    plate = prism(out, bd.Plane.XY, z1 - z0).moved(bd.Location((0, 0, z0)))
    L, S, T = sv["body_mm"]  # 45 (y, outboard), 25 (z), 36.2 (x)
    zA = h["axis_z_mm"]
    serv_top = zA + S / 2
    wall = p["hip_mount_wall_mm"]
    xm = h["axis_x_mid_mm"]
    for s in (1, -1):
        y_in = h["axis_y_mm"] - sv["shaft_from_end_mm"]  # inboard face of the servo
        y_out = y_in + L
        pad = box(xm - 13.0, xm + 13.0, s * (y_in + 18.0), s * y_out, serv_top, z0) if s > 0 else \
            box(xm - 13.0, xm + 13.0, s * y_out, s * (y_in + 18.0), serv_top, z0)
        endw = box(xm - 13.0, xm + 13.0, min(s * y_out, s * (y_out + wall)), max(s * y_out, s * (y_out + wall)), zA - S / 2, z0)
        plate = plate + pad + endw
        for xx in (xm - 8.0, xm + 8.0):  # [ASSUMED] 2 x M2.5 through the end wall into the servo
            plate -= cyl_y(xx, zA, 1.35, s * y_out - 5, s * y_out + 5) if s > 0 else cyl_y(xx, zA, 1.35, s * y_out - 5, s * y_out + 5)
    # Pi deck standoff holes and battery strap slots
    a, b = p["pi_holes_mm"]
    for dx in (-a / 2, a / 2):
        for dy in (-b / 2, b / 2):
            plate -= cyl_z(-10.0 + dx, dy, 1.35, z0 - 1, z1 + 1)
    sw, sl = p["battery_strap_slot_mm"]
    for xx in (-10.0 - 50.0, -10.0 + 50.0):
        for s in (1, -1):
            plate -= box(xx - sl / 2, xx + sl / 2, s * 40.0 - sw / 2, s * 40.0 + sw / 2, z0 - 1, z1 + 1)
    return plate


# --------------------------------------------------------------------------------- export
def lay_flat(shape: bd.Part) -> bd.Part:
    """Orient for printing: the largest planar face down (a simple heuristic), moved onto z = 0."""
    faces = [f for f in shape.faces() if f.geom_type == bd.GeomType.PLANE]
    big = max(faces, key=lambda f: f.area)
    n = big.normal_at()
    target = bd.Vector(0, 0, -1)
    axis = n.cross(target)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, n.dot(target)))))
    s = shape if axis.length < 1e-9 and ang < 1 else shape.rotate(bd.Axis((0, 0, 0), axis if axis.length > 1e-9 else (1, 0, 0)), ang)
    bb = s.bounding_box()
    return s.moved(bd.Location((-bb.min.X, -bb.min.Y, -bb.min.Z)))


def main():
    (OUT / "asm").mkdir(parents=True, exist_ok=True)
    (OUT / "print").mkdir(parents=True, exist_ok=True)
    hb = hip_bracket_left()
    ft, pad = foot_left()
    parts = {"pelvis": pelvis(), "left_hip_bracket": hb, "right_hip_bracket": hb.mirror(bd.Plane.XZ),
             "left_foot": ft, "right_foot": ft.mirror(bd.Plane.XZ), "left_sole_pad": pad, "right_sole_pad": pad.mirror(bd.Plane.XZ)}
    bed = np.array(D["printer"]["bed_mm"])
    info = {}
    for name, sh in parts.items():
        bd.export_stl(sh, str(OUT / "asm" / f"{name}.stl"), tolerance=0.05, angular_tolerance=0.2)
        flat = lay_flat(sh)
        bd.export_stl(flat, str(OUT / "print" / f"{name}.stl"), tolerance=0.05, angular_tolerance=0.2)
        bb = flat.bounding_box()
        ext = np.array([bb.size.X, bb.size.Y, bb.size.Z])
        fits = bool(np.all(np.sort(ext[:2]) <= np.sort(bed[:2])) and ext[2] <= bed[2])
        info[name] = {"volume_cm3": sh.volume / 1000, "area_cm2": sh.area / 100, "valid": bool(sh.is_valid),
                      "print_extent_mm": ext.round(1).tolist(), "fits_bed": fits,
                      "material": "TPU" if "pad" in name else "PLA"}
        print(f"[v2_parts] {name:18s} V {sh.volume / 1000:6.1f} cm3  area {sh.area / 100:6.1f} cm2  valid {sh.is_valid}  "
              f"print extent {ext.round(1)} mm  fits {D['printer']['bed_mm']} bed: {fits}")
    (OUT / "parts.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.exit(main())
