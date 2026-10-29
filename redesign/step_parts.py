"""Flatten a Fusion STEP assembly into its leaf solids, in the assembly frame.

For every solid: the occurrence path (names), volume, centre of mass, inertia at unit density,
bounding box, and a tessellated mesh. Output (gitignored):
  outputs/redesign/step/<tag>/parts.json        one record per solid
  outputs/redesign/step/<tag>/meshes.npz        v_<i>, f_<i> arrays (mm, assembly frame)

    python redesign/step_parts.py cad/incoming/final_version/Main_v24.step --tag main_v24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

ROOT = Path(__file__).resolve().parent.parent


def label_name(lab: TDF_Label) -> str:
    attr = TDataStd_Name()
    if lab.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def tessellate(shape, defl: float = 0.2):
    BRepMesh_IncrementalMesh(shape, defl, False, 0.3, True)
    vs, fs, off = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current()) if hasattr(TopoDS, "Face") else TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n = tri.NbNodes()
            pts = np.array([[*tri.Node(i).Transformed(trsf).Coord()] for i in range(1, n + 1)])
            tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            if face.Orientation() == 1:  # TopAbs_REVERSED
                tris = tris[:, ::-1]
            vs.append(pts)
            fs.append(tris + off)
            off += n
        exp.Next()
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(vs), np.vstack(fs)


def cylinders(shape):
    """Cylindrical faces: axis point/direction, radius, area (holes, pins, bosses), assembly frame."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current()) if hasattr(TopoDS, "Face") else TopoDS.Face_s(exp.Current())
        sa = BRepAdaptor_Surface(face)
        if sa.GetType() == GeomAbs_Cylinder:
            c = sa.Cylinder()
            ax = c.Axis()
            gp_ = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, gp_)
            out.append({"p": list(ax.Location().Coord()), "d": list(ax.Direction().Coord()), "r": c.Radius(),
                        "area": gp_.Mass(), "c": list(gp_.CentreOfMass().Coord())})
        exp.Next()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cylinders", action="store_true", help="also record cylindrical faces (pivots, holes)")
    ap.add_argument("--deflection", type=float, default=0.2, help="tessellation chord tolerance [mm]")
    a = ap.parse_args()
    t0 = time.time()
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    r = STEPCAFControl_Reader()
    r.SetNameMode(True)
    assert r.ReadFile(a.step) == IFSelect_RetDone
    r.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    print(f"[step] read in {time.time() - t0:.0f} s")
    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    out_dir = ROOT / "outputs" / "redesign" / "step" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, meshes = [], {}

    def walk(lab, path, loc):
        name = label_name(lab)
        if st.IsReference_s(lab):
            ref = TDF_Label()
            st.GetReferredShape_s(lab, ref)
            walk(ref, path + [name], loc.Multiplied(st.GetLocation_s(lab)))
            return
        if st.IsAssembly_s(lab):
            comps = TDF_LabelSequence()
            st.GetComponents_s(lab, comps)
            for i in range(1, comps.Length() + 1):
                walk(comps.Value(i), path, loc)
            return
        shape = st.GetShape_s(lab).Moved(loc)
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        k = 0
        while exp.More():
            sol = exp.Current()
            p = GProp_GProps()
            BRepGProp.VolumeProperties_s(sol, p)
            I = p.MatrixOfInertia()
            bb = Bnd_Box()
            BRepBndLib.Add_s(sol, bb)
            idx = len(recs)
            recs.append({"i": idx, "path": path + ([name] if name and (not path or name != path[-1]) else []), "solid": k,
                         "volume_mm3": p.Mass(), "com_mm": list(p.CentreOfMass().Coord()),
                         "inertia_unit_density_mm5": [[I.Value(r_, c) for c in (1, 2, 3)] for r_ in (1, 2, 3)],
                         "bbox_mm": [*bb.CornerMin().Coord(), *bb.CornerMax().Coord()],
                         "cylinders": cylinders(sol) if a.cylinders else []})
            v, f = tessellate(sol, a.deflection)
            meshes[f"v_{idx}"], meshes[f"f_{idx}"] = v.astype(np.float32), f.astype(np.int32)
            k += 1
            exp.Next()

    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), [], TopLoc_Location())
    (out_dir / "parts.json").write_text(json.dumps(recs, indent=0))
    np.savez_compressed(out_dir / "meshes.npz", **meshes)
    print(f"[step] {len(recs)} solids, total volume {sum(r['volume_mm3'] for r in recs) / 1000:.1f} cm3, "
          f"{time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
