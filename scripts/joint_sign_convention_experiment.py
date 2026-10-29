"""Experimentally determine, for each of Trex's 10 joints, what a positive
USD/sim joint-angle value actually does in the real world: which direction
the limb/segment moves, expressed in robot-relative terms (forward/back,
left/right, up/down) rather than raw XYZ numbers -- and specifically whether
left/right leg joints are mirrored (a positive value on both sides doing the
mirror-image motion) or not (a positive value doing the *same* raw-angle
motion, which looks different in body-relative terms on each side).

Method: pure kinematics, not dynamics. For each joint in turn, the robot is
reset to a known neutral pose (all 10 DOF at 0, root at the spawn pose),
then exactly that one joint is *teleported* (`Articulation.set_dof_positions`,
not a PD target) to +20 degrees -- safely inside every joint's limit,
including the tightest (hip, +/-30 deg). Teleporting avoids servo gain /
settling-time / gravity-sag questions entirely; those are a separate,
later concern (control), not part of defining what the sign convention *is*.
The displacement of the joint's immediate child link and the relevant
end-effector (the foot, for every leg joint) is measured in world frame,
then decomposed into the robot's own forward/right/up basis -- established
once at the start from the head/tail neutral positions, not assumed.

Because this is pure kinematics with the root re-pinned to the spawn pose
before every test, world frame and robot-body frame are identical throughout
(the robot's own orientation never changes), so no basis-rotation bookkeeping
is needed beyond the one-time forward/right/up determination.

Outputs:
- docs/JOINT_CONVENTIONS.md -- the actual documentation this experiment exists
  to produce.
- outputs/joint_experiment/<joint_name>.png -- one screenshot per joint test,
  same fixed camera throughout, for visual cross-checking.

Run:
    conda activate isaac6
    export OMNI_KIT_ACCEPT_EULA=YES
    python scripts/joint_sign_convention_experiment.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common.robot_config import RobotConfig  # noqa: E402

ROBOT_CONFIG_PATH = PROJECT_ROOT / "assets" / "robots" / "trex" / "robot_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "joint_experiment"
DOC_PATH = PROJECT_ROOT / "docs" / "JOINT_CONVENTIONS.md"

TEST_ANGLE_DEG = 20.0  # inside every joint's limit, including hip's tightest +/-30deg

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.objects import Camera, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

import test_load_robot as tlr  # noqa: E402  (must import after SimulationApp exists)

# Kinematic tree, established and verified in earlier sessions (see
# README) -- full USD path of each link's rigid-body
# container Xform (not its "visual"/"collision" child).
GEOMETRY_ROOT = "/World/Trex/Geometry/base_link"
LINK_PATHS = {
    "base_link": GEOMETRY_ROOT,
    "head": f"{GEOMETRY_ROOT}/head",
    "tail": f"{GEOMETRY_ROOT}/tail",
    "left_femur": f"{GEOMETRY_ROOT}/left_femur",
    "left_tibia": f"{GEOMETRY_ROOT}/left_femur/left_tibia",
    "left_ankle": f"{GEOMETRY_ROOT}/left_femur/left_tibia/left_ankle",
    "left_foot": f"{GEOMETRY_ROOT}/left_femur/left_tibia/left_ankle/left_foot",
    "right_femur": f"{GEOMETRY_ROOT}/right_femur",
    "right_tibia": f"{GEOMETRY_ROOT}/right_femur/right_tibia",
    "right_ankle": f"{GEOMETRY_ROOT}/right_femur/right_tibia/right_ankle",
    "right_foot": f"{GEOMETRY_ROOT}/right_femur/right_tibia/right_ankle/right_foot",
}

# joint -> (immediate child link, end-effector link to also track)
JOINT_EFFECT_LINKS = {
    "head_joint": ("head", "head"),
    "tail_joint": ("tail", "tail"),
    "left_hip_joint": ("left_femur", "left_foot"),
    "right_hip_joint": ("right_femur", "right_foot"),
    "left_femur_joint": ("left_tibia", "left_foot"),
    "right_femur_joint": ("right_tibia", "right_foot"),
    "left_tibia_joint": ("left_ankle", "left_foot"),
    "right_tibia_joint": ("right_ankle", "right_foot"),
    "left_ankle_joint": ("left_foot", "left_foot"),
    "right_ankle_joint": ("right_foot", "right_foot"),
}

JOINT_TEST_ORDER = [
    "head_joint",
    "tail_joint",
    "left_hip_joint",
    "right_hip_joint",
    "left_femur_joint",
    "right_femur_joint",
    "left_tibia_joint",
    "right_tibia_joint",
    "left_ankle_joint",
    "right_ankle_joint",
]


def link_world_position(stage: Usd.Stage, link_name: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(LINK_PATHS[link_name])
    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(xform.ExtractTranslation())


def describe_component(value: float, pos_label: str, neg_label: str, threshold: float = 0.003) -> str:
    if abs(value) < threshold:
        return ""
    label = pos_label if value > 0 else neg_label
    return f"{label} {abs(value) * 100:.1f}cm"


def main() -> None:
    cfg = RobotConfig.load(ROBOT_CONFIG_PATH)
    usd_path = tlr.convert_urdf_to_usd(cfg)

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
    robot_prim.Load(Usd.LoadWithDescendants)
    for _ in range(5):
        simulation_app.update()

    tlr.apply_robot_config(stage, robot_prim_path, cfg)
    tlr.apply_materials(stage, robot_prim_path, cfg)

    spawn_clearance_m = 0.02
    spawn_pos = np.array([[0.0, 0.0, spawn_clearance_m]])
    spawn_orient = np.array([[1.0, 0.0, 0.0, 0.0]])  # wxyz identity
    robot = Articulation(robot_prim_path, positions=spawn_pos, reset_xform_op_properties=True)
    dof_names = robot.dof_names
    print(f"[joint_experiment] DOF order: {dof_names}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(3):
        simulation_app.update()  # initialize the physics tensor view

    def reset_to_neutral() -> None:
        robot.set_world_poses(positions=spawn_pos, orientations=spawn_orient)
        robot.set_dof_positions(np.zeros((1, len(dof_names))))
        for _ in range(2):
            simulation_app.update()

    # --- Establish the robot's own forward/right/up basis from the neutral pose ---
    reset_to_neutral()
    base_pos = link_world_position(stage, "base_link")
    head_pos = link_world_position(stage, "head")
    tail_pos = link_world_position(stage, "tail")
    left_femur_pos = link_world_position(stage, "left_femur")
    right_femur_pos = link_world_position(stage, "right_femur")

    up = np.array([0.0, 0.0, 1.0])
    forward_raw = head_pos - tail_pos  # head and tail are on opposite ends of base_link
    forward = forward_raw - np.dot(forward_raw, up) * up  # project onto the horizontal plane
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Sanity check: does "right" (as computed) actually point toward the link
    # named right_femur, not left_femur? If not, something about this
    # right-hand convention or the robot's own left/right labeling disagrees
    # -- report it, don't silently assume.
    lateral = right_femur_pos - left_femur_pos
    right_check = np.dot(lateral, right)
    basis_note = (
        "computed 'right' axis points toward the right_femur link (consistent)"
        if right_check > 0
        else "WARNING: computed 'right' axis points toward left_femur, not right_femur -- "
        "the robot's left/right naming and this forward x up convention disagree, double check"
    )

    print(f"[joint_experiment] world forward = {forward}")
    print(f"[joint_experiment] world right   = {right}")
    print(f"[joint_experiment] world up      = {up}")
    print(f"[joint_experiment] {basis_note}")

    # --- Fixed camera for all screenshots, framed once from the neutral robot ---
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(robot_prim_path)).ComputeAlignedRange()
    bbox_center = np.array(world_range.GetMidpoint())
    bbox_size = np.array(world_range.GetSize())
    distance = 3.2 * max(bbox_size.max(), 0.2)
    eye = bbox_center + np.array([distance * 0.7, -distance * 0.7, distance * 0.55])
    camera_path = "/World/Camera"
    Camera(camera_path, positions=np.array([eye]), orientations=np.array([tlr._look_at_quaternion(eye, bbox_center)]))
    viewport = get_active_viewport()
    viewport.camera_path = camera_path
    for _ in range(90):  # let MDL shaders finish compiling before the first capture
        simulation_app.update()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for joint_name in JOINT_TEST_ORDER:
        dof_index = dof_names.index(joint_name)
        child_link, end_effector_link = JOINT_EFFECT_LINKS[joint_name]

        reset_to_neutral()
        child_before = link_world_position(stage, child_link)
        foot_before = link_world_position(stage, end_effector_link)

        angles = np.zeros((1, len(dof_names)), dtype=np.float32)
        angles[0, dof_index] = np.radians(TEST_ANGLE_DEG)
        robot.set_dof_positions(angles)
        for _ in range(3):
            simulation_app.update()

        child_after = link_world_position(stage, child_link)
        foot_after = link_world_position(stage, end_effector_link)

        child_delta = child_after - child_before
        foot_delta = foot_after - foot_before

        def to_body_frame(delta: np.ndarray) -> tuple[float, float, float]:
            return (float(np.dot(delta, forward)), float(np.dot(delta, right)), float(np.dot(delta, up)))

        child_fwd, child_rgt, child_up = to_body_frame(child_delta)
        foot_fwd, foot_rgt, foot_up = to_body_frame(foot_delta)

        screenshot_path = OUTPUT_DIR / f"{joint_name}.png"
        capture_viewport_to_file(viewport, str(screenshot_path))
        simulation_app.update()

        results.append(
            {
                "joint": joint_name,
                "child_link": child_link,
                "end_effector": end_effector_link,
                "child_delta_body": (child_fwd, child_rgt, child_up),
                "foot_delta_body": (foot_fwd, foot_rgt, foot_up),
                "screenshot": screenshot_path.relative_to(PROJECT_ROOT),
            }
        )
        print(
            f"[joint_experiment] {joint_name}: +{TEST_ANGLE_DEG}deg -> "
            f"{child_link} moves (fwd={child_fwd * 100:+.1f}cm, right={child_rgt * 100:+.1f}cm, up={child_up * 100:+.1f}cm), "
            f"{end_effector_link} moves (fwd={foot_fwd * 100:+.1f}cm, right={foot_rgt * 100:+.1f}cm, up={foot_up * 100:+.1f}cm)"
        )

    # Which lateral side (relative to "right") each leg sits on at neutral,
    # needed to turn a raw delta-right sign into "inward"/"outward".
    left_foot_neutral = link_world_position(stage, "left_foot")
    right_foot_neutral = link_world_position(stage, "right_foot")
    leg_side_sign = {
        "left": 1.0 if np.dot(left_foot_neutral - base_pos, right) < 0 else -1.0,
        "right": 1.0 if np.dot(right_foot_neutral - base_pos, right) < 0 else -1.0,
    }
    print(f"[joint_experiment] leg_side_sign (neutral side, for inward/outward calc): {leg_side_sign}")

    reset_to_neutral()
    timeline.stop()

    write_doc(results, forward, right, up, basis_note, leg_side_sign)
    print(f"[joint_experiment] Wrote {DOC_PATH}")


def write_doc(
    results: list[dict],
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    basis_note: str,
    leg_side_sign: dict[str, float],
) -> None:
    lines = []
    lines.append("# Joint Sign Conventions")
    lines.append("")
    lines.append(
        "Experimentally determined (not assumed from the URDF `<axis>` tags alone) by "
        "`scripts/joint_sign_convention_experiment.py`: each joint teleported to "
        f"+{TEST_ANGLE_DEG} deg from a neutral (all-zero) pose, with the resulting world-space "
        "displacement of its child link and the relevant foot measured and decomposed into the "
        "robot's own forward/right/up axes. Pure kinematics (teleported, not PD-driven), so "
        "these are sign-convention facts, not control-tuning results."
    )
    lines.append("")
    lines.append("## Global / body frame convention")
    lines.append("")
    lines.append("- **Up** = world `+Z` (standard USD/Isaac Sim up axis for this stage).")
    lines.append(
        "- **Forward** = direction from `tail` to `head` at the neutral pose, projected onto the "
        f"horizontal plane. In world coordinates at spawn (identity orientation): `{forward.tolist()}`."
    )
    lines.append(
        f"- **Right** = `forward × up` (right-hand rule). In world coordinates at spawn: `{right.tolist()}`."
    )
    lines.append(f"- {basis_note}.")
    lines.append(
        "- At spawn the robot's own orientation is identity, so world axes and robot-body axes "
        "coincide -- the forward/right/up vectors above double as both."
    )
    lines.append("")
    lines.append("## Per-joint effect of a positive angle")
    lines.append("")
    lines.append(
        "`child_link` is the link the joint directly rotates; `foot_delta` is the resulting motion "
        "of the relevant end-effector (the foot, for every leg joint) -- useful since several joints "
        "only move the foot indirectly through the kinematic chain. All deltas in cm, body frame "
        "(+forward/-back, +right/-left, +up/-down)."
    )
    lines.append("")
    lines.append("| Joint | Child link | Child delta (fwd, right, up) | Foot delta (fwd, right, up) |")
    lines.append("|---|---|---|---|")
    for r in results:
        cf, cr, cu = r["child_delta_body"]
        ff, fr, fu = r["foot_delta_body"]
        lines.append(
            f"| `{r['joint']}` | `{r['child_link']}` | "
            f"({cf * 100:+.1f}, {cr * 100:+.1f}, {cu * 100:+.1f}) | "
            f"({ff * 100:+.1f}, {fr * 100:+.1f}, {fu * 100:+.1f}) |"
        )
    lines.append("")

    lines.append("## Left vs right symmetry")
    lines.append("")
    pairs = [
        ("left_hip_joint", "right_hip_joint"),
        ("left_femur_joint", "right_femur_joint"),
        ("left_tibia_joint", "right_tibia_joint"),
        ("left_ankle_joint", "right_ankle_joint"),
    ]
    by_joint = {r["joint"]: r for r in results}
    for left_name, right_name in pairs:
        l, r_ = by_joint[left_name], by_joint[right_name]
        l_right_component = l["foot_delta_body"][1]
        r_right_component = r_["foot_delta_body"][1]
        l_inward = l_right_component * leg_side_sign["left"]
        r_inward = r_right_component * leg_side_sign["right"]
        if abs(l_right_component) < 0.003 and abs(r_right_component) < 0.003:
            note = "no significant left/right (sideways) component for either side at this test angle"
        else:
            l_word = "inward" if l_inward > 0 else "outward"
            r_word = "inward" if r_inward > 0 else "outward"
            raw_sign_note = (
                "same raw sign (not mirrored as a raw number)"
                if (l_right_component > 0) == (r_right_component > 0)
                else "opposite raw sign (mirrored as a raw number)"
            )
            note = (
                f"left foot moves **{l_word}** ({abs(l_inward) * 100:.1f}cm), right foot moves "
                f"**{r_word}** ({abs(r_inward) * 100:.1f}cm) -- {raw_sign_note}"
            )
        lines.append(f"- `{left_name}` vs `{right_name}`: {note}")
    lines.append("")

    lines.append("## Screenshots")
    lines.append("")
    lines.append("One per joint, all from the same fixed camera (`outputs/joint_experiment/`, gitignored):")
    lines.append("")
    for r in results:
        lines.append(f"- `{r['joint']}`: `{r['screenshot']}`")
    lines.append("")

    DOC_PATH.write_text("\n".join(lines))


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
