"""Generate TPU sole pads (STL, mm) matching the current feet's sole outline. Parameters: sole_pad.yaml.

    python redesign/cad/sole_pad.py   -> outputs/redesign/cad/{left,right}_sole_pad.stl
"""
import sys
from pathlib import Path

import numpy as np
import shapely
import trimesh
import yaml
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load((Path(__file__).with_suffix(".yaml")).read_text())
out = ROOT / "outputs" / "redesign" / "cad"
out.mkdir(parents=True, exist_ok=True)
for side in ("left", "right"):
    m = trimesh.load(str(ROOT / "assets" / "robots" / "trex" / "meshes" / f"{side}_foot.stl"))
    zmin = m.bounds[0][2]
    s = m.section(plane_origin=[0, 0, zmin + cfg["section_above_sole_mm"]], plane_normal=[0, 0, 1])
    P = unary_union([shapely.Polygon(p.exterior.coords) for p in s.to_2D(to_2D=np.eye(4))[0].polygons_full])
    r = cfg["min_feature_mm"] / 2
    P = P.buffer(-cfg["inset_mm"]).buffer(-r).buffer(r)  # inset, then morphological opening
    P = max(P.geoms, key=lambda g: g.area) if P.geom_type == "MultiPolygon" else P
    pad = trimesh.creation.extrude_polygon(P, cfg["thickness_mm"])
    pad.apply_translation([0, 0, zmin - cfg["thickness_mm"]])  # in the assembly frame, under the foot
    assert pad.is_watertight
    pad.export(str(out / f"{side}_sole_pad.stl"))
    print(f"{side}: area {P.area / 100:.1f} cm2 (sole {unary_union(P).area / 100:.1f}), extents {np.round(pad.extents, 1)} mm, "
          f"volume {pad.volume / 1000:.2f} cm3 (~{pad.volume / 1000 * 1.21:.1f} g TPU)")
