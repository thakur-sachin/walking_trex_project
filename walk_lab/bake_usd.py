"""Bake the Trex robot USD for Isaac Lab.

Isaac Lab spawns a robot from a USD file, while the Isaac Sim stack (walk_rl) authors
robot_config.yaml onto the live stage after referencing the imported USD. This runs the
same steps (URDF import cache + tlr.apply_robot_config: actuated-joint limits, servo
torque caps, base_link mass) and writes the result as one flattened file, so both stacks
simulate the same robot.

    python walk_lab/bake_usd.py      # -> _converted_usd/trex_lab/trex_lab.usd

Re-run whenever robot_config.yaml (limits, mass, torque) or the URDF changes. The file
records the robot_config hash it was baked from; walk_lab/env.py refuses a stale bake.
"""
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

OUT = PROJECT_ROOT / "_converted_usd" / "trex_lab" / "trex_lab.usd"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PROJECT_ROOT / "assets" / "robots" / "trex" / "robot_config.yaml",
              PROJECT_ROOT / "assets" / "robots" / "trex" / "Trex.urdf"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    import test_load_robot as tlr
    from common.robot_config import RobotConfig

    cfg = RobotConfig.load(tlr.ROBOT_CONFIG_PATH)
    src = tlr.convert_urdf_to_usd(cfg)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/Trex", "Xform")
    root.GetReferences().AddReference(str(src))
    root.Load(Usd.LoadWithDescendants)
    for _ in range(3):
        simulation_app.update()
    tlr.apply_robot_config(stage, "/Trex", cfg)
    stage.SetDefaultPrim(root)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    flat = stage.Flatten()  # an Sdf.Layer
    flat.defaultPrim = "Trex"
    flat.customLayerData = {"trex_robot_config_hash": config_hash(), "source": str(src)}
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
    print(f"[bake] base_link mass: {mass} kg (robot_config {cfg.base_link_mass_kg}); config hash {config_hash()}")
    assert len(roots) == 1 and len(joints) == 10 and abs(mass - cfg.base_link_mass_kg) < 1e-9


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        simulation_app.close()
