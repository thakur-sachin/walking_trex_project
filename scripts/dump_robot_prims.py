"""Dump every prim under the imported Trex robot's root, with type, applied
APIs, and a few key attributes, to audit for import bloat.

Runs as plain USD (no Kit/SimulationApp needed -- opening the file directly
is much faster than booting Isaac Sim just to introspect a stage).

Usage:
    conda activate isaac6
    python scripts/dump_robot_prims.py [path/to/Trex.usda]
    python scripts/dump_robot_prims.py --summary-only   # just counts by type
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from pxr import Usd, UsdGeom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USD = PROJECT_ROOT / "_converted_usd" / "trex" / "Trex" / "Trex.usda"

parser = argparse.ArgumentParser()
parser.add_argument("usd_path", nargs="?", default=str(DEFAULT_USD))
parser.add_argument("--summary-only", action="store_true", help="only print per-type counts, not every prim")
args = parser.parse_args()

usd_path = Path(args.usd_path)
if not usd_path.exists():
    print(f"USD file not found: {usd_path}", file=sys.stderr)
    print("Run scripts/test_load_robot.py first to produce it.", file=sys.stderr)
    sys.exit(1)

stage = Usd.Stage.Open(str(usd_path))
stage.Load()  # force all payload arcs to compose; Stage.Open alone doesn't guarantee it
root = stage.GetDefaultPrim()
print(f"Stage: {usd_path}")
print(f"Default prim: {root.GetPath()}")

variant_sets = root.GetVariantSets()
for vset_name in variant_sets.GetNames():
    vset = variant_sets.GetVariantSet(vset_name)
    print(
        f"VariantSet '{vset_name}': active='{vset.GetVariantSelection()}', "
        f"available={vset.GetVariantNames()}"
    )
print()

type_counts: Counter[str] = Counter()
api_counts: Counter[str] = Counter()
mesh_prim_count = 0
mesh_points_total = 0

for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
    type_name = prim.GetTypeName() or "(untyped)"
    type_counts[type_name] += 1
    for api in prim.GetAppliedSchemas():
        api_counts[api] += 1

    if prim.IsA(UsdGeom.Mesh):
        mesh_prim_count += 1
        points_attr = UsdGeom.Mesh(prim).GetPointsAttr()
        pts = points_attr.Get()
        n_pts = len(pts) if pts else 0
        mesh_points_total += n_pts

    if not args.summary_only:
        extra = ""
        if prim.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            extra = f"  [{len(pts) if pts else 0} verts]"
        print(f"{prim.GetPath()}  ({type_name}){extra}")

print("\n=== Prim type counts ===")
for type_name, count in type_counts.most_common():
    print(f"  {count:5d}  {type_name}")

print("\n=== Applied API counts ===")
for api, count in api_counts.most_common():
    print(f"  {count:5d}  {api}")

print("\n=== Geometry ===")
print(f"  Mesh prims: {mesh_prim_count}")
print(f"  Total mesh vertices: {mesh_points_total}")

# File sizes on disk under the same directory tree (catches library files,
# duplicate geometry, cached materials, etc. that the prim tree alone won't show).
print("\n=== Files on disk (same import output dir) ===")
import_dir = usd_path.parent.parent  # .../trex/Trex/Trex.usda -> .../trex
total_bytes = 0
for f in sorted(import_dir.rglob("*")):
    if f.is_file():
        size = f.stat().st_size
        total_bytes += size
        print(f"  {size:>10,d} B  {f.relative_to(import_dir)}")
print(f"\n  Total: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")
