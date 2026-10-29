"""MuJoCo model of Trex as a Pi-runner backend (sim2sim: a second, independent physics engine).

Built from the same URDF and robot_config.yaml as the Isaac stacks and matched to what PhysX
actually simulates (checked 2026-10-15 by dumping the Isaac Sim articulation):
  - base_link mass = robot_config (1.0 kg placeholder); its inertia follows robot_config
    base_link_inertia like Isaac (urdf / scale_with_mass / composite shell + payload)
  - joint limits from robot_config; ankles unlimited
  - servo = implicit PD (kp 40 / kd 5 legs, 10 / 1 head and tail) capped at the LX-16A stall
    torque, tracking a goal ramped linearly over the control period (move-time command);
    or (actuator="lx16a") the fitted LX-16A torque model of walk_rl/actuator.py, nominal
    parameters, with its armature on the joints
  - ankle linkage ankle = femur - tibia: an equality on the fixed tendon (ankle - femur +
    tibia). Like the real parallelogram, it carries the foot torque without loading the
    femur/tibia servos (Isaac's ankle PD drive + teleport puts that torque on the chain).
    Tried and rejected: teleporting the ankle each substep as Isaac does, or setting its
    velocity -- both break momentum conservation (free fall: 5x Isaac's torso pitch rate,
    policy fell in 0.3 s); an ankle PD drive -- unstable on the 72 g foot. MuJoCo's soft
    constraint strength scales with the (tiny) foot inertia, so at 200 Hz walking loads
    stretched the linkage 32-35 deg; at 1 kHz it holds 0.35 deg (Isaac: 0.2-0.3 deg). Isaac
    matches this load path with robot_config ankle_linkage sim_model: teleport_routed. Hence
  - 1 kHz physics (the policy still runs at the task's control rate), no self-collision,
    nominal friction 0.5 (PhysX default material)
  - contact time constant 4 ms (near-rigid, like PhysX) by default. MuJoCo's default 20 ms
    is a noticeably soft floor: v8d falls within seconds on it but walks 30 s at 4 ms, so
    floor/sole compliance is swept as a robustness axis in sim2sim.py.

Differences that remain by construction: contact model (MuJoCo soft contacts vs PhysX),
mesh collision (MuJoCo convex hulls), solver. That is the point of a sim2sim test.
"""

from __future__ import annotations

import math
import re
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.runner import Reading  # noqa: E402

ASSET_DIR = PROJECT_ROOT / "assets" / "robots" / "trex"


def build_model(rc: RobotConfig, friction: float = 0.5, physics_hz: int = 1000, spawn_height_m: float = 0.02,
                contact_timeconst_s: float = 0.004, actuator: str = "implicit_pd",
                visual: bool = False) -> mujoco.MjModel:
    urdf = (ASSET_DIR / "Trex.urdf").read_text()
    urdf = re.sub(r"<material.*?</material>|<material[^>]*/>", "", urdf, flags=re.S)  # MuJoCo rejects repeated names
    urdf = re.sub(r"<visual>.*?</visual>", "", urdf, flags=re.S)
    urdf = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="%s" balanceinertia="true" discardvisual="true" '
                  r'fusestatic="false"/></mujoco>' % ASSET_DIR, urdf, count=1)
    spec = mujoco.MjSpec.from_string(urdf)
    spec.option.timestep = 1.0 / physics_hz
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    base = spec.body("base_link")
    base.add_freejoint(name="root")
    if rc.base_link_inertia == "scale_with_mass":  # as test_load_robot.apply_robot_config does for Isaac
        base.fullinertia = np.asarray(base.fullinertia) * (rc.base_link_mass_kg / base.mass)  # URDF: full tensor
    elif rc.base_link_inertia in ("composite", "parts"):
        inertia, com = rc.base_link_inertia_and_com(ASSET_DIR / "meshes" / "base_link.stl")
        base.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        base.ipos = com
    base.mass = rc.base_link_mass_kg
    for link, kg in rc.link_mass_overrides_kg().items():  # robot_config mass.link_kg, inertia scaled alike
        b = spec.body(link)
        b.fullinertia = np.asarray(b.fullinertia) * (kg / b.mass)
        b.mass = kg
    base.pos = [0.0, 0.0, spawn_height_m]

    for j in spec.joints:
        if j.name in rc.actuated_joints:
            j.range = list(rc.joint_range_rad(j.name))
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            act = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            if actuator == "lx16a":  # ctrl = joint torque from walk_rl/actuator.py
                act.set_to_motor()
                act.forcerange = [-3.0 * rc.actuator_stall_torque_nm, 3.0 * rc.actuator_stall_torque_nm]
                j.armature = rc.raw["actuator_model"]["armature_kgm2"]
            else:
                kp, kd = rc.sim_drive_for_joint(j.name)
                act.set_to_position(kp=kp, kv=kd)
                act.forcerange = [-rc.actuator_stall_torque_nm, rc.actuator_stall_torque_nm]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE

    for side in ("left", "right"):
        t = spec.add_tendon(name=f"{side}_ankle_linkage")
        t.wrap_joint(f"{side}_ankle_joint", 1.0)
        t.wrap_joint(f"{side}_femur_joint", -1.0)
        t.wrap_joint(f"{side}_tibia_joint", 1.0)
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_TENDON, name1=t.name, data=np.zeros(11))
        eq.solref = [2.0 / physics_hz, 1.0]  # stiffest MuJoCo allows at this timestep

    solref = [max(contact_timeconst_s, 2.0 / physics_hz), 1.0]
    for g in spec.geoms:  # robot: collides with the floor only
        g.contype, g.conaffinity = 1, 0
        g.friction = [friction, 0.005, 0.0001]
        g.solref = solref
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], contype=0,
                                    conaffinity=1, friction=[friction, 0.005, 0.0001], solref=solref)
    if visual:  # rendering only (render_mujoco.py): 5 cm checker floor, robot colour, light; no physics change
        spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         width=256, height=256, rgb1=[0.82, 0.82, 0.80], rgb2=[0.62, 0.64, 0.66])
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         width=256, height=256, rgb1=[0.75, 0.84, 0.93], rgb2=[0.95, 0.96, 0.97])
        mat = spec.add_material(name="checker", texrepeat=[10, 10], texuniform=True)
        tex = list(mat.textures)
        tex[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "checker"
        mat.textures = tex
        floor.material = "checker"
        spec.compiler.discardvisual = False  # it would drop the floor's material (visual meshes are already stripped)
        for g in spec.geoms:
            if g.name != "floor":
                g.rgba = [0.85, 0.45, 0.15, 1.0]
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    return spec.compile()


class MujocoTrex:
    """Backend: read() -> Reading, write(goal) -> runs one control period of physics.

    Hardware effects it adds (each optional): bus latency in control steps, sensor noise,
    per-run encoder offsets. Without them it is the noise-free robot."""

    def __init__(self, friction: float = 0.5, control_hz: int = 25, physics_hz: int = 1000, delay_steps: int = 0,
                 contact_timeconst_s: float = 0.004,
                 noise: dict | None = None, encoder_offset_rad: float = 0.0, seed: int = 0,
                 max_tilt_deg: float = 55.0, min_head_height_m: float = 0.12, ground_contact_force_n: float = 0.2,
                 actuator: str = "implicit_pd", visual: bool = False, robot_config: str | Path | None = None):
        self.rc = RobotConfig.load(Path(robot_config) if robot_config else ASSET_DIR / "robot_config.yaml")
        self.actuator = actuator
        self.model = build_model(self.rc, friction, physics_hz, contact_timeconst_s=contact_timeconst_s, actuator=actuator,
                                 visual=visual)
        self.data = mujoco.MjData(self.model)
        self.decimation = physics_hz // control_hz
        self.act_names = list(self.rc.actuated_joints)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in self.act_names])
        self.ctrl_idx = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.act_names])
        self.dofadr = np.array([m.jnt_dofadr[jid(n)] for n in self.act_names])
        if actuator == "lx16a":  # nominal fitted servo at the measured pack voltage (no DR here)
            sys.path.insert(0, str(PROJECT_ROOT))
            from walk_rl.actuator import LX16AParams

            self.lx = LX16AParams.from_robot_config(self.rc)
            self.control_period = 1.0 / control_hz
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in ("left", "right")}
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.delay_steps = delay_steps
        self.noise = noise or {}
        self.rng = np.random.default_rng(seed)
        self.enc_offset = (self.rng.random(len(self.act_names)) * 2 - 1) * encoder_offset_rad
        self.cos_tilt = math.cos(math.radians(max_tilt_deg))
        self.min_head_z = min_head_height_m
        self.contact_n = ground_contact_force_n
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.goal = np.zeros(len(self.act_names))
        self.queue = deque([self.goal.copy()] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        self.time = 0.0
        self.fail_reason = None

    # ------------------------------------------------------------------ state
    def _rot(self) -> np.ndarray:
        return self.data.xmat[self.base].reshape(3, 3)  # body -> world

    def base_quat_xyzw(self) -> np.ndarray:
        w, x, y, z = self.data.xquat[self.base]
        return np.array([x, y, z, w])

    def yaw(self) -> float:
        x, y, z, w = self.base_quat_xyzw()
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def base_vel_world(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()  # free joint: world-frame linear velocity

    def base_pos(self) -> np.ndarray:
        return self.data.xpos[self.base].copy()

    def read(self) -> Reading:
        R = self._rot()
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        w_b = self.data.qvel[3:6].copy()  # free joint: angular velocity in the body frame
        q = self.data.qpos[self.qadr].copy() + self.enc_offset
        n = self.noise
        if n:
            g_b = g_b + self.rng.normal(0, n.get("gravity", 0.0), 3)
            w_b = w_b + self.rng.normal(0, n.get("gyro_rad_s", 0.0), 3)
            q = q + self.rng.normal(0, n.get("joint_pos_rad", 0.0), len(q))
        return Reading(gravity_b=g_b, gyro_b=w_b, yaw=self.yaw(), q=q)

    def failed(self) -> str | None:
        """Same terminations as training: tilt, head height, any non-foot link on the ground."""
        if self.fail_reason:
            return self.fail_reason
        g_z = (self._rot().T @ np.array([0.0, 0.0, -1.0]))[2]
        if g_z > -self.cos_tilt:
            self.fail_reason = "tilt"
        elif self.data.xpos[self.head][2] < self.min_head_z:
            self.fail_reason = "head low"
        else:
            f = np.zeros(6)
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                other = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else None
                if other is None:
                    continue
                body = self.model.geom_bodyid[other]
                if body in self.feet:
                    continue
                mujoco.mj_contactForce(self.model, self.data, i, f)
                if abs(f[0]) > self.contact_n:
                    self.fail_reason = f"{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)} on the ground"
                    break
        return self.fail_reason

    def ankle_residual_deg(self) -> float:
        q = lambda n: self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
        return max(abs(math.degrees(q(f"{s}_ankle_joint") - (q(f"{s}_femur_joint") - q(f"{s}_tibia_joint"))))
                   for s in ("left", "right"))

    # ---------------------------------------------------------------- actuate
    def write(self, goal: np.ndarray) -> None:
        """Send servo goals; the servos ramp to them over one control period (after the
        bus latency). Runs that period of physics."""
        self.queue.append(np.asarray(goal, dtype=np.float64).copy())
        target = self.queue[0]
        start = self.goal
        h = self.model.opt.timestep
        for i in range(self.decimation):
            if self.actuator == "lx16a":
                from walk_rl.actuator import lx16a_torque, ramp_reference

                p, v = self.lx, self.lx.v_nominal / self.lx.v_ref
                q_ref = ramp_reference(start, target, (i + 1) * h, self.control_period, p.latency, np.clip)
                self.data.ctrl[self.ctrl_idx] = lx16a_torque(
                    q_ref, self.data.qpos[self.qadr], self.data.qvel[self.dofadr], p.kp, p.deadband, p.coulomb,
                    p.stall * v, p.no_load * v, p.armature, h, np)
            else:
                tgt = start + (target - start) * ((i + 1) / self.decimation)
                self.data.ctrl[self.ctrl_idx] = tgt
            mujoco.mj_step(self.model, self.data)
        self.goal = target.copy()
        self.time += self.decimation * self.model.opt.timestep
