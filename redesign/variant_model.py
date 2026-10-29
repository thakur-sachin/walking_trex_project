"""Design variants of the MuJoCo Trex model (pi_runner/mujoco_backend.py) for CPU sensitivity studies.

`build_variant_model(rc, variant, **kw)` takes the simulator's own MjSpec
(`pi_runner.mujoco_backend.build_spec`) and applies the design changes in `variant` before compiling.
With an empty variant the compiled model is identical to build_model's (checked by
`python redesign/variant_model.py`, which compares every model array and a 3 s physics trace).

`patched_backend(variant)` is a context manager that makes `MujocoTrex` (and so sim2sim / PolicyRunner
loops) build the variant instead of the baseline, without editing pi_runner.

Variant keys (all optional; a lever's baseline value is "absent"). Lengths in metres, masses in kg.
  base_mass_kg                  torso (base_link) mass: robot_config default
  base_com_shift_m [dx,dy,dz]   shift of the torso centre of mass (inertia kept about the new point)
  base_inertia_scale            torso inertia tensor multiplier
  head_mass_scale, tail_mass_scale     mass and inertia multiplier (shape fixed)
  head_length_scale, tail_length_scale centre-of-mass distance from the joint x s, inertia x s^2,
                                collision mesh scaled about the joint (mass unchanged)
  thigh_length_m                femur_joint -> tibia_joint span (URDF 0.065)
  shank_length_m                tibia_joint -> ankle_joint span (URDF 0.054)
  leg_mass_scale                multiplier on femur/tibia/ankle link masses and inertias
  hip_spacing_delta_m           added to |y| of both hip joints (feet follow)
  hip_axis_y_m                  both hip roll axes at +-this y (symmetric), feet follow
  foot_scale [sx, sy]           foot collision mesh scaled about the ankle axis (mass, inertia follow
                                sx*sy for mass; this is the sole footprint lever)
  foot_mass_scale               foot mass / inertia multiplier
  foot_friction                 friction of the foot geoms (MuJoCo takes the max of the pair, so the
                                floor is set to min(friction, foot_friction))
  link_mass_scale {link: s}     mass and inertia multiplier per link (shape fixed)
  joint_limits_deg {joint: [lo, hi]}  per-joint range override
  stall_scale                   servo stall torque multiplier (actuator model)
"""

from __future__ import annotations

import contextlib
import copy
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.robot_config import RobotConfig  # noqa: E402
import pi_runner.mujoco_backend as mb  # noqa: E402

ASSET_DIR = ROOT / "assets" / "robots" / "trex"
SIDES = ("left", "right")


def _scale_body_inertia(b, s_mass: float = 1.0, s_inertia: float | None = None):
    b.mass = b.mass * s_mass
    b.fullinertia = np.asarray(b.fullinertia) * (s_mass if s_inertia is None else s_inertia)
    b.inertia = np.asarray(b.inertia) * (s_mass if s_inertia is None else s_inertia)


def _geoms_of(spec, body_name):
    return [g for g in spec.geoms if g.parent.name == body_name]


def apply_variant(spec: mujoco.MjSpec, rc: RobotConfig, variant: dict, friction: float):
    v = variant or {}
    base = spec.body("base_link")

    # ---------------------------------------------------------------- torso
    if "base_mass_kg" in v:
        s = v["base_mass_kg"] / base.mass
        base.mass = v["base_mass_kg"]
        base.fullinertia = np.asarray(base.fullinertia) * s
    if "base_inertia_scale" in v:
        base.fullinertia = np.asarray(base.fullinertia) * v["base_inertia_scale"]
    if "base_com_shift_m" in v:
        base.ipos = np.asarray(base.ipos) + np.asarray(v["base_com_shift_m"], dtype=float)

    # ------------------------------------------------------------ head / tail
    for part in ("head", "tail"):
        b = spec.body(part)
        if f"{part}_mass_scale" in v:
            _scale_body_inertia(b, v[f"{part}_mass_scale"])
        if f"{part}_length_scale" in v:
            s = v[f"{part}_length_scale"]
            b.ipos = np.asarray(b.ipos) * s
            b.fullinertia = np.asarray(b.fullinertia) * s * s
            b.inertia = np.asarray(b.inertia) * s * s
            for g in _geoms_of(spec, part):  # scale the collision mesh about the joint (body origin)
                m = spec.mesh(g.meshname)
                m.scale = np.asarray(m.scale) * s
                g.pos = np.asarray(g.pos) * s

    # ------------------------------------------------------------------- legs
    for side in SIDES:
        femur, tibia, ankle, foot = (spec.body(f"{side}_{n}") for n in ("femur", "tibia", "ankle", "foot"))
        if "thigh_length_m" in v:  # ankle body = child of tibia; its pos is the tibia_joint location
            p = np.asarray(ankle.pos, dtype=float)
            p0 = abs(p[2])
            ankle.pos = [p[0], p[1], -v["thigh_length_m"]]
            _stretch_link(spec, f"{side}_tibia", v["thigh_length_m"] / p0)
        if "shank_length_m" in v:
            p = np.asarray(foot.pos, dtype=float)
            p0 = abs(p[2])
            foot.pos = [p[0], p[1], -v["shank_length_m"]]
            _stretch_link(spec, f"{side}_ankle", v["shank_length_m"] / p0)
        if "leg_mass_scale" in v:
            for b in (femur, tibia, ankle):
                _scale_body_inertia(b, v["leg_mass_scale"])
        if "hip_axis_y_m" in v:  # symmetric hip roll axes at +-y (the URDF has +70.8 / -76.6 mm)
            p = np.asarray(femur.pos, dtype=float)
            femur.pos = [p[0], np.sign(p[1]) * v["hip_axis_y_m"], p[2]]
        if "hip_spacing_delta_m" in v:
            p = np.asarray(femur.pos, dtype=float)
            femur.pos = [p[0], p[1] + np.sign(p[1]) * v["hip_spacing_delta_m"], p[2]]
        if "foot_scale" in v:
            sx, sy = v["foot_scale"]
            for g in _geoms_of(spec, f"{side}_foot"):
                m = spec.mesh(g.meshname)
                m.scale = np.asarray(m.scale) * np.array([sx, sy, 1.0])
                gp = np.asarray(g.pos, dtype=float)
                g.pos = [gp[0] * sx, gp[1] * sy, gp[2]]
            foot.ipos = np.asarray(foot.ipos) * np.array([sx, sy, 1.0])
            _scale_body_inertia(foot, sx * sy)
        if "foot_mass_scale" in v:
            _scale_body_inertia(foot, v["foot_mass_scale"])
        if "foot_friction" in v:
            for g in _geoms_of(spec, f"{side}_foot"):
                g.friction = [v["foot_friction"], 0.005, 0.0001]

    if "foot_friction" in v:  # MuJoCo uses the max of the two geoms: keep the floor at or below the sole
        fl = spec.geom("floor")
        fl.friction = [min(friction, v["foot_friction"]), 0.005, 0.0001]
        for g in spec.geoms:  # non-foot links keep the nominal value (they only matter for termination)
            pass

    # ------------------------------------------------------- per-link mass scale
    for name, sc in (v.get("link_mass_scale") or {}).items():
        _scale_body_inertia(spec.body(name), sc)

    # ------------------------------------------------------------ joint limits
    for name, (lo, hi) in (v.get("joint_limits_deg") or {}).items():
        j = spec.joint(name)
        j.range = [np.radians(lo), np.radians(hi)]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE


def _stretch_link(spec, body_name: str, s: float):
    """Stretch a leg link along its z axis about its joint (body origin): mesh, COM and inertia."""
    b = spec.body(body_name)
    ip = np.asarray(b.ipos, dtype=float)
    b.ipos = [ip[0], ip[1], ip[2] * s]
    for g in _geoms_of(spec, body_name):
        m = spec.mesh(g.meshname)
        m.scale = np.asarray(m.scale) * np.array([1.0, 1.0, s])
        gp = np.asarray(g.pos, dtype=float)
        g.pos = [gp[0], gp[1], gp[2] * s]


def build_variant_model(rc: RobotConfig, variant: dict | None = None, friction: float = 0.5, physics_hz: int = 1000,
                        spawn_height_m: float = 0.02, contact_timeconst_s: float = 0.004, actuator: str = "implicit_pd",
                        visual: bool = False, return_spec: bool = False, asset_dir: Path | None = None):
    """The simulator's own model (pi_runner.mujoco_backend.build_spec) with the variant applied."""
    spec = mb.build_spec(rc, friction, physics_hz, spawn_height_m, contact_timeconst_s, actuator, visual, asset_dir)
    sc = (variant or {}).get("stall_scale", 1.0)
    if sc != 1.0:
        for a in spec.actuators:
            a.forcerange = np.asarray(a.forcerange) * sc
    apply_variant(spec, rc, variant, friction)
    if return_spec:
        return spec
    return spec.compile()


@contextlib.contextmanager
def patched_backend(variant: dict | None, stall_scale_into_lx16a: bool = True):
    """Within the block, MujocoTrex builds `variant`. The lx16a servo parameters are read by
    MujocoTrex from robot_config; a variant stall_scale is applied to them as well."""
    orig_build = mb.build_model
    orig_init = mb.MujocoTrex.__init__

    def build(rc, friction=0.5, physics_hz=1000, spawn_height_m=0.02, contact_timeconst_s=0.004,
              actuator="implicit_pd", visual=False, asset_dir=None):
        return build_variant_model(rc, variant, friction, physics_hz, spawn_height_m, contact_timeconst_s, actuator, visual,
                                   asset_dir=asset_dir)

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if variant and "stall_scale" in variant and getattr(self, "actuator", None) == "lx16a":
            self.lx = copy.copy(self.lx)
            self.lx.stall = self.lx.stall * variant["stall_scale"]

    mb.build_model = build
    mb.MujocoTrex.__init__ = init
    try:
        yield
    finally:
        mb.build_model = orig_build
        mb.MujocoTrex.__init__ = orig_init


# ------------------------------------------------------------------------ self-check
def _compare_models(a: mujoco.MjModel, b: mujoco.MjModel) -> list[str]:
    diffs = []
    for name in dir(a):
        if name.startswith("_"):
            continue
        try:
            x, y = getattr(a, name), getattr(b, name)
        except Exception:
            continue
        if isinstance(x, np.ndarray):
            if x.shape != y.shape or not np.array_equal(x, y):
                diffs.append(name)
    return diffs


def self_check():
    rc = RobotConfig.load(ASSET_DIR / "robot_config.yaml")
    ok = True
    for act in ("implicit_pd", "lx16a"):
        m0 = mb.build_model(rc, 0.5, 1000, actuator=act)
        m1 = build_variant_model(rc, {}, 0.5, 1000, actuator=act)
        d = _compare_models(m0, m1)
        print(f"[variant] empty variant vs build_model ({act}): {'IDENTICAL arrays' if not d else 'DIFF ' + str(d)}")
        ok &= not d
        # 3 s passive trace with a fixed control
        tr = []
        for m in (m0, m1):
            dd = mujoco.MjData(m)
            dd.ctrl[:] = 0.05 if act == "lx16a" else 0.0
            for _ in range(3000):
                mujoco.mj_step(m, dd)
            tr.append(np.concatenate([dd.qpos, dd.qvel]))
        same = np.array_equal(tr[0], tr[1])
        print(f"[variant] 3 s trace ({act}): {'IDENTICAL' if same else 'max diff %.3g' % np.abs(tr[0] - tr[1]).max()}")
        ok &= same
    # a variant must actually change what it says
    m = build_variant_model(rc, {"base_mass_kg": 1.2, "head_mass_scale": 0.5, "thigh_length_m": 0.08,
                                 "hip_spacing_delta_m": -0.01, "foot_friction": 0.8}, 0.5, 1000, actuator="lx16a")
    m0 = mb.build_model(rc, 0.5, 1000, actuator="lx16a")
    bid = lambda mm, n: mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, n)
    print(f"[variant] base mass {m0.body_mass[bid(m0,'base_link')]:.3f} -> {m.body_mass[bid(m,'base_link')]:.3f}; "
          f"head {m0.body_mass[bid(m0,'head')]:.4f} -> {m.body_mass[bid(m,'head')]:.4f}; "
          f"L ankle body z {m0.body_pos[bid(m0,'left_ankle')][2]:.4f} -> {m.body_pos[bid(m,'left_ankle')][2]:.4f}; "
          f"L hip y {m0.body_pos[bid(m0,'left_femur')][1]:.4f} -> {m.body_pos[bid(m,'left_femur')][1]:.4f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)
