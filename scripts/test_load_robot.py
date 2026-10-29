"""Isaac Sim 6.0.1 smoke test: environment + lighting + Trex robot, with real
robot parameters (mass, joint limits, servo torque) applied from a swappable
config file rather than hardcoded in this script.

See assets/robots/trex/robot_config.yaml for the actual numbers and their
provenance (servo datasheet, user-provided estimates, TODOs pending real
measurement) -- this script should never hardcode a mass/limit/gain value
that belongs in that file.

Pipeline: converts the Trex URDF to USD via the full
`isaacsim.asset.importer.urdf.URDFImporter` (not the bare
`urdf_usd_converter.Converter` -- that one has no drive-gain/mass/limit
knobs and an unconfigured import is physically unstable, exploding within
1-2 physics steps), simplifies the imported asset's visual/collision Xform
naming, builds a stage with a ground plane and dome light, loads the robot
as an Articulation, applies per-joint position limits + torque caps + the
base_link mass override + green-skin/dark-claw materials from the config,
steps physics briefly, frames a camera from the robot's actual settled
bounding box, and saves a screenshot to outputs/. Pass `--remote` to also
enable live-stage access over `isaacsim.code_editor.python_server`
(127.0.0.1:8226) for the `isaac-sim-remote` skill.

Robot facts baked into this pipeline (see robot_config.yaml for detail):
- Only 8 of the 10 URDF joints are real actuated servos (Hiwonder LX-16A):
  head, tail, and hip/femur/tibia per leg. The 2 ankle joints have no servo
  -- they're set mechanically by a double 4-bar parallelogram linkage that
  keeps the foot parallel to base_link, i.e. ankle = femur - tibia
  (robot_config.yaml `ankle_linkage`). enforce_ankle_linkage() teleports the
  ankle DOFs to that relationship; the ankle drive uses the same stiffness/
  damping as the leg joints (`passive_joint_drive`) so it holds the target.
- Actuated joints get a position limit (deg, from config) and their drive's
  max force capped to the servo's real stall torque (1.667 N*m). Ankle
  joints are left unlimited and uncapped since they're not a servo.
- base_link mass is overridden to the config value (1.0 kg placeholder for
  the whole assembly: PLA shell + battery + Pi 4 + hip servos) -- the URDF's
  own mass there is wrong (never measured).

Other gotchas discovered getting this working:

- The imported asset's real content (physics schemas, geometry) sits behind
  a `payload` arc nested inside the referenced asset. A reference added
  dynamically at runtime does not reliably auto-load that payload; call
  `prim.Load(Usd.LoadWithDescendants)` explicitly before querying for the
  ArticulationRootAPI prim, or `Articulation(...)` fails with `paths=[None]`.
- `Articulation(...)` needs `reset_xform_op_properties=True` the first time a
  freshly-imported/freshly-referenced prim is wrapped, since it has no
  `xformOp:translate` authored yet.
- `UsdPhysics.RevoluteJoint` lower/upper limits are authored in **degrees**,
  not radians (confirmed from the USD Physics schema source doc strings).

Run:
    conda activate isaac6
    export OMNI_KIT_ACCEPT_EULA=YES
    python scripts/test_load_robot.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402

ROBOT_DIR = PROJECT_ROOT / "assets" / "robots" / "trex"
URDF_PATH = ROBOT_DIR / "Trex.urdf"
ROBOT_CONFIG_PATH = ROBOT_DIR / "robot_config.yaml"
USD_OUT_DIR = PROJECT_ROOT / "_converted_usd" / "trex"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--physics-steps", type=int, default=15, help="kept low for a quick smoke test")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="enable isaacsim.code_editor.python_server (127.0.0.1:8226) for live-stage access",
    )
    return parser.parse_args()


# SimulationApp creation (and the CLI parsing it depends on) only happens when
# this file is run directly -- a script that imports convert_urdf_to_usd(),
# apply_robot_config(), etc. as a library must create its own SimulationApp
# *before* importing this module (only one can exist per process), then those
# functions work the same way main() uses them below. See
# scripts/joint_sign_convention_experiment.py for an example.
if __name__ == "__main__":
    args = _parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    if args.remote:
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("isaacsim.code_editor.python_server", True)
        print("[test_load_robot] isaacsim.code_editor.python_server enabled on 127.0.0.1:8226")

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from isaacsim.core.experimental.materials import OmniPbrMaterial  # noqa: E402
from isaacsim.core.experimental.objects import Camera, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402


def _look_at_quaternion(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Quaternion (wxyz) for a USD camera at `eye` looking at `target`.

    Uses Gf.Matrix4d.SetLookAt (USD's own, tested utility) rather than a
    hand-rolled rotation-matrix-to-quaternion derivation -- an earlier
    hand-rolled version LOOKED internally consistent (manually re-deriving
    its own view direction matched the authored transform) but still
    produced a badly-aimed camera, so trust the library here instead of
    re-deriving the math a second time.
    """
    # SetLookAt builds a world-to-camera (view) matrix; invert for the
    # camera prim's local-to-world transform.
    view_matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*world_up))
    cam_to_world = view_matrix.GetInverse()
    quat = cam_to_world.ExtractRotationQuat()
    real = quat.GetReal()
    imaginary = quat.GetImaginary()
    return np.array([real, imaginary[0], imaginary[1], imaginary[2]])


def simplify_visual_collision_naming(usd_path: Path) -> None:
    """Rename each link's `<link>`/`<link>_1` visual/collision container Xforms
    to plain `visual`/`collision`. Purely cosmetic (the importer's default
    naming -- e.g. `left_femur/left_femur` + `left_femur/left_femur_1` -- reads
    as a confusing duplicate of the link name), but it has to happen by editing
    the actual defining layer (`payloads/base.usda`), not as a live-stage
    override, since these are real local prims there (not instance proxies --
    only their Mesh children are), so a namespace-edit rename is both legal
    and persists across every future load of this converted asset.

    Uses Sdf.BatchNamespaceEdit, which fixes up any same-layer path references
    (e.g. the material-binding Scope nested under each renamed Xform) -- a
    plain "create new prim at new path, copy, delete old" would not.
    """
    base_usda_path = usd_path.parent / "payloads" / "base.usda"

    # Read-only pass to find every (visual_path, collision_path) rename target
    # -- must close this stage before editing the layer directly below.
    stage = Usd.Stage.Open(str(base_usda_path))
    geometry = stage.GetPrimAtPath("/Trex/Geometry")
    renames: list[tuple[Sdf.Path, str]] = []
    for link_prim in Usd.PrimRange(geometry):
        if link_prim == geometry:
            continue
        link_name = link_prim.GetName()
        visual = link_prim.GetChild(link_name)
        collision = link_prim.GetChild(f"{link_name}_1")
        if visual and collision:
            renames.append((visual.GetPath(), "visual"))
            renames.append((collision.GetPath(), "collision"))
    stage = None

    layer = Sdf.Layer.FindOrOpen(str(base_usda_path))
    edit = Sdf.BatchNamespaceEdit()
    for path, new_name in renames:
        edit.Add(Sdf.NamespaceEdit.Rename(path, new_name))
    assert layer.Apply(edit), f"Failed to apply visual/collision rename to {base_usda_path}"
    layer.Save()
    print(f"[test_load_robot] Simplified {len(renames) // 2} links' visual/collision Xform names in {base_usda_path}")


def convert_urdf_to_usd(cfg: RobotConfig) -> Path:
    USD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = list(USD_OUT_DIR.glob(f"**/{URDF_PATH.stem}.usd*"))
    if cached:
        print(f"[test_load_robot] Reusing cached USD at {cached[0]}")
        return cached[0]
    print(f"[test_load_robot] Importing {URDF_PATH} -> {USD_OUT_DIR}")
    importer_config = URDFImporterConfig(
        urdf_path=str(URDF_PATH),
        usd_path=str(USD_OUT_DIR),
        merge_fixed_joints=True,
        fix_base=False,
        collision_from_visuals=False,
        robot_type="Quadruped",
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness=cfg.urdf_importer_stiffness_overrides(),
        override_joint_damping=cfg.urdf_importer_damping_overrides(),
    )
    result = URDFImporter(importer_config).import_urdf()
    print(f"[test_load_robot] Imported USD: {result}")
    result_path = Path(str(result))
    simplify_visual_collision_naming(result_path)
    return result_path


def apply_robot_config(stage: Usd.Stage, robot_prim_path: str, cfg: RobotConfig) -> None:
    """Author the real robot parameters onto the imported USD: per-joint position
    limits + torque caps for the 8 actuated joints, and the base_link mass override.
    Ankle joints are intentionally left untouched (unlimited, uncapped; their drive
    gains were already baked in at import time) -- see module docstring.
    """
    physics_root = f"{robot_prim_path}/Physics"
    for joint_name in cfg.actuated_joints:
        joint_prim = stage.GetPrimAtPath(f"{physics_root}/{joint_name}")
        assert joint_prim.IsValid(), f"Expected joint prim at {physics_root}/{joint_name}"

        revolute = UsdPhysics.RevoluteJoint(joint_prim)
        lo_deg, hi_deg = cfg.joint_range_deg(joint_name)
        revolute.CreateLowerLimitAttr().Set(lo_deg)
        revolute.CreateUpperLimitAttr().Set(hi_deg)

        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        assert drive, f"Expected an angular DriveAPI on {joint_name} (joint_drive_type='force' at import)"
        drive.CreateMaxForceAttr().Set(cfg.actuator_stall_torque_nm)

        print(
            f"[test_load_robot] {joint_name}: limit {lo_deg:+g}..{hi_deg:+g} deg, "
            f"max_force={cfg.actuator_stall_torque_nm} N*m"
        )

    for link, kg in cfg.link_mass_overrides_kg().items():
        # links are nested (Geometry/base_link/.../head), so find the prim by name
        found = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{robot_prim_path}/Geometry"))
                 if p.GetName() == link and p.HasAPI(UsdPhysics.MassAPI)]
        assert len(found) == 1, f"Expected one link prim named {link} with a MassAPI, found {len(found)}"
        prim = found[0]
        api = UsdPhysics.MassAPI.Apply(prim)
        old = api.GetMassAttr().Get()
        assert old, f"{link}: no authored URDF mass to scale from"
        api.CreateMassAttr().Set(kg)
        inertia = api.GetDiagonalInertiaAttr()
        if inertia.Get() is not None:
            inertia.Set(inertia.Get() * (kg / old))
        print(f"[test_load_robot] {link} mass {old:.4f} -> {kg} kg (inertia x{kg / old:.3f}; robot_config mass.link_kg)")

    base_link_prim = stage.GetPrimAtPath(f"{robot_prim_path}/Geometry/base_link")
    assert base_link_prim.IsValid(), f"Expected base_link prim at {robot_prim_path}/Geometry/base_link"
    mass_api = UsdPhysics.MassAPI.Apply(base_link_prim)
    authored_mass = mass_api.GetMassAttr().Get()  # URDF value on a fresh import
    mass_api.CreateMassAttr().Set(cfg.base_link_mass_kg)
    print(f"[test_load_robot] base_link mass set to {cfg.base_link_mass_kg} kg")
    if cfg.base_link_inertia == "scale_with_mass" and authored_mass:
        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        scaled = inertia_attr.Get() * (cfg.base_link_mass_kg / authored_mass)
        inertia_attr.Set(scaled)
        print(f"[test_load_robot] base_link inertia scaled by {cfg.base_link_mass_kg / authored_mass:.3f} "
              f"(robot_config base_link_inertia: scale_with_mass) -> {tuple(round(v, 6) for v in scaled)}")
    elif cfg.base_link_inertia in ("composite", "parts"):
        inertia, com = cfg.base_link_inertia_and_com(PROJECT_ROOT / "assets" / "robots" / "trex" / "meshes" / "base_link.stl")
        principal, axes = np.linalg.eigh(inertia)  # columns: principal axes in the base_link frame
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        rot = Gf.Matrix3d(*axes.T.flatten().tolist()).ExtractRotation().GetQuat()  # Gf: row vectors -> R^T
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*principal.tolist()))
        mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(rot.GetReal(), *rot.GetImaginary()))
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com.tolist()))
        print(f"[test_load_robot] base_link inertia ({cfg.base_link_inertia}) -> principal "
              f"{tuple(round(v, 6) for v in principal)} kg*m^2, CoM {tuple(round(v, 4) for v in com)} m")


def enforce_ankle_linkage(robot: Articulation, dof_names: list[str], cfg: RobotConfig) -> None:
    """Set both ankle DOFs to the angle the real 4-bar parallelogram linkage enforces
    given the *current* femur/tibia DOF positions (see RobotConfig.ankle_angle_from_linkage
    and robot_config.yaml's ankle_linkage section for the confirmed geometry/derivation).

    This is a per-call kinematic enforcement (teleports the ankle DOF, same mechanism as
    set_dof_positions elsewhere in this file), not a continuously-solved physics constraint
    -- PhysX's native mimic joint only supports a single reference joint, not the two-term
    sum this relationship needs. Call this again after any future change to femur/tibia
    (e.g. every step of a control loop) to keep it enforced; it does not stay valid on its
    own as physics/other code moves those joints.
    """
    positions = robot.get_dof_positions().numpy()
    for side in ("left", "right"):
        femur_idx = dof_names.index(f"{side}_femur_joint")
        tibia_idx = dof_names.index(f"{side}_tibia_joint")
        ankle_idx = dof_names.index(f"{side}_ankle_joint")
        ankle_angle = cfg.ankle_angle_from_linkage(positions[0, femur_idx], positions[0, tibia_idx])
        robot.set_dof_positions([[ankle_angle]], dof_indices=[ankle_idx])


def apply_materials(stage: Usd.Stage, robot_prim_path: str, cfg: RobotConfig) -> None:
    """Create one OmniPBR material per group in robot_config.yaml's `materials`
    section, and bind each to the *visual* mesh of every link in that group
    (not the `_1`-suffixed collision mesh -- see module docstring on why that
    duplicate exists; it doesn't need a color).

    Per-link mesh nesting depth varies with how deep the link sits in the
    kinematic chain (e.g. base_link/head vs base_link/left_femur/left_tibia/
    left_ankle/left_foot), so rather than hand-build each path, this finds
    every Mesh prim under Geometry whose own name is a known link name and
    whose immediate parent Xform is named "visual" (post
    simplify_visual_collision_naming(); the collision copy's parent is "collision").
    """
    link_to_group = {link: group for group in cfg.material_groups for link in group.links}

    materials: dict[str, OmniPbrMaterial] = {}
    for group in cfg.material_groups:
        material_path = f"/World/Looks/{group.name}"
        material = OmniPbrMaterial(material_path)
        material.set_input_values("diffuse_color_constant", list(group.color_rgb))
        material.set_input_values("reflection_roughness_constant", group.roughness)
        materials[group.name] = material
        print(f"[test_load_robot] Created material '{group.name}' {group.color_rgb} for links {group.links}")

    geometry_root = stage.GetPrimAtPath(f"{robot_prim_path}/Geometry")
    bound = set()
    for prim in Usd.PrimRange(geometry_root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        link_name = prim.GetName()
        is_visual = prim.GetParent().GetName() == "visual"
        group = link_to_group.get(link_name)
        if is_visual and group is not None:
            # The Mesh itself is an instance proxy (lives inside the imported
            # asset's instanceable reference) -- USD disallows authoring onto
            # a proxy directly. Walk up to the nearest non-proxy ancestor
            # (the visual Xform that actually introduces the reference arc,
            # one level up) and bind there instead; material binding resolves
            # down through instance boundaries for rendering.
            bind_target = prim
            while bind_target.IsInstanceProxy():
                bind_target = bind_target.GetParent()
            # Each mesh already has its own direct material binding from the
            # importer (the default grey/white material) -- that's closer to
            # the geometry than our ancestor-level binding, so "nearest wins"
            # would otherwise keep showing the old material. Force ours to
            # win regardless of proximity.
            UsdShade.MaterialBindingAPI.Apply(bind_target).Bind(
                materials[group.name].materials[0], bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            bound.add(link_name)

    missing = set(link_to_group) - bound
    if missing:
        print(f"[test_load_robot] WARNING: no visual mesh found for configured links: {missing}")


def main() -> None:
    cfg = RobotConfig.load(ROBOT_CONFIG_PATH)
    print(f"[test_load_robot] Loaded robot config from {ROBOT_CONFIG_PATH}")
    print(f"[test_load_robot] Actuated joints ({len(cfg.actuated_joints)}): {cfg.actuated_joints}")
    print(f"[test_load_robot] Passive joints ({len(cfg.passive_joints)}): {cfg.passive_joints}")

    usd_path = convert_urdf_to_usd(cfg)

    stage_utils.create_new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    if not stage.GetPrimAtPath("/physicsScene"):
        UsdPhysics.Scene.Define(stage, "/physicsScene")

    GroundPlane("/World/GroundPlane")
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(1000.0)
    dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    robot_prim_path = "/World/Trex"
    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    robot_prim.GetReferences().AddReference(str(usd_path))
    robot_prim.Load(Usd.LoadWithDescendants)  # see module docstring: payload gotcha
    for _ in range(5):
        simulation_app.update()  # let the reference + payload fully compose

    apply_robot_config(stage, robot_prim_path, cfg)
    apply_materials(stage, robot_prim_path, cfg)

    # Spawn height: small ground clearance only. The imported mesh geometry is
    # already authored at its natural standing height relative to base_link's
    # own origin (confirmed: root at world Z=0 gives a world bbox of
    # Z=[0.007, 0.347], matching the asset's own extentsHint) -- unlike the
    # older IsaacLab config (since removed from this repo), which assumed base_link's
    # origin was at the feet and needed a manually-chosen 0.35m spawn offset.
    # Don't copy that number here; it was for a different conversion pipeline.
    spawn_clearance_m = 0.02
    robot = Articulation(
        robot_prim_path, positions=np.array([[0.0, 0.0, spawn_clearance_m]]), reset_xform_op_properties=True
    )
    print(f"[test_load_robot] DOF names ({robot.num_dofs}): {robot.dof_names}")
    print(f"[test_load_robot] Link names ({robot.num_links}): {robot.link_names}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    pos_before, _ = robot.get_world_poses()
    print(f"[test_load_robot] Root position at spawn: {pos_before.numpy()}")

    for _ in range(args.physics_steps):
        simulation_app.update()

    pos_after, _ = robot.get_world_poses()
    print(f"[test_load_robot] Root position after {args.physics_steps} steps: {pos_after.numpy()}")

    dof_names = robot.dof_names
    enforce_ankle_linkage(robot, dof_names, cfg)
    for _ in range(2):
        simulation_app.update()
    print(f"[test_load_robot] Enforced ankle linkage ({cfg.raw['ankle_linkage']['relationship']}) for both legs")

    # pause(), NOT stop() -- stop() resets the whole articulation back to its
    # initial authored pose, discarding the settling + ankle enforcement done
    # above. Found live: every screenshot taken after stop() silently showed
    # the neutral pose regardless of what was actually simulated.
    timeline.pause()

    # Frame the camera from the robot's *actual* settled bounding box, not a
    # hardcoded guess -- a fixed guess silently drifts out of frame whenever
    # the settled pose differs (it did, badly, during development here).
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(robot_prim_path)).ComputeAlignedRange()
    bbox_center = np.array(world_range.GetMidpoint())
    bbox_size = np.array(world_range.GetSize())
    print(f"[test_load_robot] Robot world bbox center={bbox_center}, size={bbox_size}")

    camera_path = "/World/Camera"
    distance = 2.6 * max(bbox_size.max(), 0.2)  # scale with robot size, floor it for tiny/degenerate bboxes
    eye = bbox_center + np.array([distance * 0.7, -distance * 0.7, distance * 0.5])
    orientation_wxyz = _look_at_quaternion(eye, bbox_center)
    Camera(camera_path, positions=np.array([eye]), orientations=np.array([orientation_wxyz]))
    viewport = get_active_viewport()
    viewport.camera_path = camera_path

    for _ in range(120):  # MDL shader compilation is async and can take a while on first load
        simulation_app.update()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_path = OUTPUT_DIR / "trex_first_load.png"
    capture_viewport_to_file(viewport, str(screenshot_path))
    simulation_app.update()  # flush the capture
    print(f"[test_load_robot] Screenshot saved to {screenshot_path}")

    if not args.headless:
        print("[test_load_robot] GUI mode: window stays open until you close it.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
