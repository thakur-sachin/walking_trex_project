"""Bake the Trex robot USD for Isaac Lab.

Isaac Lab spawns a robot from a USD file, while the Isaac Sim stack (walk_rl) authors
robot_config.yaml onto the live stage after referencing the imported USD. This runs the
same steps (URDF import cache + tlr.apply_robot_config: actuated-joint limits, servo
torque caps, base_link mass) and writes the result as one flattened file, so both stacks
simulate the same robot.

    python walk_lab/bake_usd.py      # -> _converted_usd/trex_lab/trex_lab.usd
    python walk_lab/bake_usd.py --asset-dir assets/robots/trex_v2   # -> _converted_usd/trex_v2_lab/trex_v2_lab.usd

Re-run whenever robot_config.yaml (limits, mass, torque) or the URDF changes. The file
records the robot_config hash it was baked from; walk_lab/env.py refuses a stale bake.
"""
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DEFAULT_ASSET_DIR = PROJECT_ROOT / "assets" / "robots" / "trex"


def baked_usd_path(asset_dir: Path = DEFAULT_ASSET_DIR) -> Path:
    """Baked Lab USD of a robot asset: _converted_usd/<asset>_lab/<asset>_lab.usd."""
    name = f"{Path(asset_dir).resolve().name}_lab"
    return PROJECT_ROOT / "_converted_usd" / name / f"{name}.usd"


def config_hash(asset_dir: Path = DEFAULT_ASSET_DIR) -> str:
    h = hashlib.sha256()
    for p in (Path(asset_dir) / "robot_config.yaml", Path(asset_dir) / "Trex.urdf"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main(asset_dir: Path) -> None:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    import test_load_robot as tlr
    from common.robot_config import RobotConfig

    OUT = baked_usd_path(asset_dir)
    cfg = RobotConfig.load(asset_dir / "robot_config.yaml")
    src = tlr.convert_urdf_to_usd(cfg, asset_dir)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/Trex", "Xform")
    root.GetReferences().AddReference(str(src))
    root.Load(Usd.LoadWithDescendants)
    for _ in range(3):
        simulation_app.update()
    tlr.apply_robot_config(stage, "/Trex", cfg, asset_dir)
    stage.SetDefaultPrim(root)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    flat = stage.Flatten()  # an Sdf.Layer
    flat.defaultPrim = "Trex"
    flat.customLayerData = {"trex_robot_config_hash": config_hash(asset_dir), "source": str(src)}
    flat.Export(str(OUT))

    # sanity: one articulation root, the 10 revolute joints, base mass as configured
    check = Usd.Stage.Open(str(OUT))
    roots = [p.GetPath() for p in check.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [p.GetName() for p in check.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    base = check.GetPrimAtPath("/Trex/Geometry/base_link")
    mass = UsdPhysics.MassAPI(base).GetMassAttr().Get()
    print(f"[bake] wrote {OUT}")
    print(f"[bake] articulation roots: {roots}")
    print(f"[bake] revolute joints ({len(joints)}): {sorted(joints)}")
    print(f"[bake] base_link mass: {mass} kg (robot_config {cfg.base_link_mass_kg}); config hash {config_hash(asset_dir)}")
    assert len(roots) == 1 and len(joints) == 10 and abs(mass - cfg.base_link_mass_kg) < 1e-6  # float32 attribute


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR.relative_to(PROJECT_ROOT)),
                    help="robot asset directory, repo-relative (Trex.urdf + meshes/ + robot_config.yaml)")
    asset_dir = (PROJECT_ROOT / ap.parse_args().asset_dir).resolve()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        main(asset_dir)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        simulation_app.close()
