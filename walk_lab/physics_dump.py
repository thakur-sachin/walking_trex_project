"""Dump the physics-relevant USD attributes of a live stage (scene, ground, robot env_0),
so the Isaac Sim stack and the Isaac Lab port can be checked for identical physics settings.

    python walk_lab/physics_dump.py --out /tmp/isaacsim_physics.json   # Isaac Sim stack (walk_rl)
    (walk_lab/smoke_lab.py writes the Isaac Lab side with dump_stage_physics())
    python walk_lab/physics_dump.py --diff a.json b.json                # no simulator needed
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _val(v):
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return [float(x) for x in v]
    except TypeError:
        return str(v)


def _attrs(prim, prefixes=("physx", "physics")) -> dict:
    out = {}
    for a in prim.GetAttributes():
        n = a.GetName()
        if n.split(":")[0] in prefixes and a.HasAuthoredValue():
            out[n] = _val(a.Get())
    return out


def dump_stage_physics(stage, robot_root: str) -> dict:
    """Authored physics attributes: every PhysicsScene, every ground/collider outside the
    robot (with bound physics materials), and every prim under `robot_root` (env_0)."""
    from pxr import Usd, UsdPhysics, UsdShade

    d = {"scene": {}, "ground": {}, "robot": {}, "materials": {}}
    for p in stage.Traverse():
        path = str(p.GetPath())
        if p.IsA(UsdPhysics.Scene):
            d["scene"]["<scene>"] = _attrs(p)
        elif p.HasAPI(UsdPhysics.CollisionAPI) and "/envs/" not in path:
            d["ground"]["<ground collider>"] = _attrs(p)
            mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
            if mat:
                d["ground"]["<ground material>"] = _attrs(mat.GetPrim())
    root = stage.GetPrimAtPath(robot_root)
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        a = _attrs(p)
        if a:
            rel = str(p.GetPath())[len(robot_root):] or "/"
            d["robot"][rel] = a
        if p.HasAPI(UsdPhysics.CollisionAPI):
            mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
            if mat:
                d["materials"][str(p.GetPath())[len(robot_root):]] = _attrs(mat.GetPrim())
    return d


def diff(a: dict, b: dict, la="A", lb="B") -> int:
    n = 0
    for sec in sorted(set(a) | set(b)):
        sa, sb = a.get(sec, {}), b.get(sec, {})
        for key in sorted(set(sa) | set(sb)):
            ka, kb = sa.get(key, {}), sb.get(key, {})
            for attr in sorted(set(ka) | set(kb)):
                va, vb = ka.get(attr, "<unset>"), kb.get(attr, "<unset>")
                if va != vb:
                    n += 1
                    print(f"[{sec}] {key} :: {attr}\n    {la}: {va}\n    {lb}: {vb}")
    print(f"{n} differing attributes")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2)
    args = ap.parse_args()
    if args.diff:
        A, B = (json.load(open(p)) for p in args.diff)
        diff(A, B, Path(args.diff[0]).stem, Path(args.diff[1]).stem)
        sys.exit(0)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.usd

        from walk_rl.env import TrexWalkEnv

        env = TrexWalkEnv(2, device="cpu", simulation_app=simulation_app)
        d = dump_stage_physics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Trex")
        json.dump(d, open(args.out, "w"), indent=1, sort_keys=True)
        print(f"[dump] wrote {args.out}: scene {len(d['scene'].get('<scene>', {}))} attrs, robot prims {len(d['robot'])}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        simulation_app.close()
