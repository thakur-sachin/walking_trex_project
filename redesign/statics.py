"""Quasi-static analysis of Trex design variants on the MuJoCo model (same URDF and config as the
simulators): servo torque per joint, centre of mass, support polygon and stability margins.

Method (rigid bodies, joints held still, torso level):
  r = qfrc_inverse(q, qvel=0, qacc = a on the root translation)   (gravity + d'Alembert force)
  r = S^T tau + J_link^T lambda + sum_feet J_foot^T w_foot
    tau     8 servo torques; root and ankles are unactuated
    lambda  the ankle parallelogram (tendon ankle - femur + tibia = 0): it carries the foot moment
            into the leg chain the way the real bars do, so it costs the servos nothing
    w_foot  6-D wrench of each foot in contact, at its sole centre
  single support: square system, solved exactly; a = (a_x, a_y) is chosen so that the centre of
            pressure sits at a chosen point of the sole (LIPM-consistent: the ground force passes
            through the centre of mass, the body accelerates as an inverted pendulum would)
  double support: minimum servo-torque solution (internal forces between the feet allowed; the
            friction they need is reported)

Torques are reported as a fraction of the stall torque at the measured pack voltage
(robot_config actuator.stall_torque_nm x measured_voltage_v / reference_voltage_v).

This is a static / quasi-static model. It does not include joint acceleration, impacts or the
policy's own choices, so it ranks levers; it does not predict gait performance.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import shapely
import trimesh
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from redesign.variant_model import ASSET_DIR, build_variant_model  # noqa: E402

G = 9.81
ACT = ["head_joint", "tail_joint", "left_hip_joint", "right_hip_joint", "left_femur_joint", "right_femur_joint",
       "left_tibia_joint", "right_tibia_joint"]


def sole_patch(mesh_path: Path, foot_frame_offset_m: np.ndarray, scale_xy=(1.0, 1.0), band_mm: float = 0.5):
    """Sole contact polygon in the foot body frame [m] (section 0.5 mm above the lowest point), and
    the sole height below the ankle axis [m]."""
    m = trimesh.load(str(mesh_path))
    zmin = m.bounds[0][2]
    s = m.section(plane_origin=[0, 0, zmin + band_mm], plane_normal=[0, 0, 1])
    polys = s.to_2D(to_2D=np.eye(4))[0].polygons_full
    P = unary_union([shapely.Polygon(np.asarray(p.exterior.coords) * 0.001 + foot_frame_offset_m[:2]) for p in polys])
    P = shapely.affinity.scale(P, scale_xy[0], scale_xy[1], origin=(0, 0))
    return P, zmin * 0.001 + foot_frame_offset_m[2]


@dataclass
class Result:
    tau_frac: dict
    com: np.ndarray
    cop: dict
    a_xy: tuple
    feet_force: dict
    friction_needed: float
    margins: dict


class Statics:
    def __init__(self, variant: dict | None = None, asset_dir: Path | None = None):
        self.asset_dir = Path(asset_dir) if asset_dir is not None else ASSET_DIR
        self.rc = RobotConfig.load(self.asset_dir / "robot_config.yaml")
        self.variant = variant or {}
        self.m = build_variant_model(self.rc, self.variant, actuator="lx16a", asset_dir=self.asset_dir)
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT  # we solve the constraints ourselves
        self.d = mujoco.MjData(self.m)
        am = self.rc.raw["actuator_model"]
        self.stall = self.rc.actuator_stall_torque_nm * self.variant.get("stall_scale", 1.0) * am["measured_voltage_v"] / am["reference_voltage_v"]
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.dof = {n: self.m.jnt_dofadr[jid(n)] for n in ACT + ["left_ankle_joint", "right_ankle_joint"]}
        self.qadr = {n: self.m.jnt_qposadr[jid(n)] for n in ACT + ["left_ankle_joint", "right_ankle_joint"]}
        self.foot = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in ("left", "right")}
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        # sole polygons in each foot frame (geom pos = visual origin: mesh assembly frame -> foot frame)
        self.sole, self.sole_z = {}, {}
        fs = self.variant.get("foot_scale", (1.0, 1.0))
        for s in ("left", "right"):
            g = [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == self.foot[s]][0]
            # geom_pos of a mesh geom is the mesh's centre after MuJoCo recentres it; use the URDF origin instead
            off = self._urdf_visual_origin(f"{s}_foot", self.asset_dir)
            self.sole[s], self.sole_z[s] = sole_patch(self.asset_dir / "meshes" / f"{s}_foot.stl", off, fs)

    @staticmethod
    def _urdf_visual_origin(link: str, asset_dir: Path = ASSET_DIR) -> np.ndarray:
        import xml.etree.ElementTree as ET
        root = ET.parse(asset_dir / "Trex.urdf").getroot()
        for l in root.findall("link"):
            if l.get("name") == link:
                return np.array(list(map(float, l.find("visual/origin").get("xyz").split())))
        raise KeyError(link)

    # ------------------------------------------------------------------ kinematics
    def set_pose(self, pose: dict):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qvel[:] = 0
        d.qacc[:] = 0
        for n, v in pose.items():
            d.qpos[self.qadr[n]] = math.radians(v)
        for s in ("left", "right"):
            d.qpos[self.qadr[f"{s}_ankle_joint"]] = math.radians(pose.get(f"{s}_femur_joint", 0.0) - pose.get(f"{s}_tibia_joint", 0.0))
        mujoco.mj_forward(self.m, d)

    def sole_center_world(self, side: str) -> np.ndarray:
        b = self.foot[side]
        R = self.d.xmat[b].reshape(3, 3)
        c = self.sole[side].centroid
        return self.d.xpos[b] + R @ np.array([c.x, c.y, self.sole_z[side]])

    def sole_polygon_world(self, side: str):
        b = self.foot[side]
        R = self.d.xmat[b].reshape(3, 3)
        pts = np.asarray(self.sole[side].convex_hull.exterior.coords)
        w = (np.c_[pts, np.full(len(pts), self.sole_z[side])] @ R.T) + self.d.xpos[b]
        return shapely.Polygon(w[:, :2]), w[:, 2].mean()

    def com(self) -> np.ndarray:
        return self.d.subtree_com[self.base].copy()

    def total_mass(self) -> float:
        return float(self.m.body_subtreemass[self.base])

    # --------------------------------------------------------------------- statics
    def _system(self, feet: list[str], a_xy=(0.0, 0.0)):
        m, d = self.m, self.d
        d.qacc[:] = 0
        d.qacc[0], d.qacc[1] = a_xy
        mujoco.mj_inverse(m, d)
        r = d.qfrc_inverse.copy()
        nv = m.nv
        cols = []
        for n in ACT:
            e = np.zeros(nv)
            e[self.dof[n]] = 1.0
            cols.append(e)
        S = np.array(cols).T  # nv x 8
        Jt = np.zeros((nv, 2))  # fixed tendon of each ankle linkage: ankle - femur + tibia (constant Jacobian)
        for i, s in enumerate(("left", "right")):
            Jt[self.dof[f"{s}_ankle_joint"], i] = 1.0
            Jt[self.dof[f"{s}_femur_joint"], i] = -1.0
            Jt[self.dof[f"{s}_tibia_joint"], i] = 1.0
        Jc = []
        for s in feet:
            jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(m, d, jp, jr, self.sole_center_world(s), self.foot[s])
            Jc.append(np.vstack([jp, jr]).T)  # nv x 6
        A = np.hstack([S, Jt] + Jc)
        return A, r

    def solve(self, feet: list[str], a_xy=(0.0, 0.0)):
        A, r = self._system(feet, a_xy)
        if len(feet) == 1:
            x = np.linalg.solve(A, r)
        else:  # min servo torque, wrench/linkage forces nearly free
            w = np.r_[np.ones(8), np.full(A.shape[1] - 8, 1e-6)]
            Winv = np.diag(1.0 / w)
            x = Winv @ A.T @ np.linalg.solve(A @ Winv @ A.T, r)
        tau = x[:8]
        wr = {s: x[10 + 6 * i: 16 + 6 * i] for i, s in enumerate(feet)}
        return tau, wr, np.abs(A @ x - r).max()

    def cop_local(self, side: str, wrench: np.ndarray) -> np.ndarray:
        """Centre of pressure relative to the sole centre (world-aligned axes) [m]."""
        f, mo = wrench[:3], wrench[3:]
        return np.array([-mo[1] / f[2], mo[0] / f[2]])

    def single_support(self, pose: dict, side: str, cop_offset=(0.0, 0.0)) -> Result:
        """Stance on one foot; choose a_xy so the CoP lies at sole centre + cop_offset."""
        self.set_pose(pose)
        # CoP is affine in a_xy: probe three points
        base = self._cop_for(side, (0.0, 0.0))
        ax = self._cop_for(side, (1.0, 0.0)) - base
        ay = self._cop_for(side, (0.0, 1.0)) - base
        Mx = np.c_[ax, ay]
        a = np.linalg.solve(Mx, np.asarray(cop_offset) - base)
        tau, wr, res = self.solve([side], tuple(a))
        f = wr[side][:3]
        return Result(tau_frac=dict(zip(ACT, (tau / self.stall).round(4))), com=self.com(), cop={side: self.cop_local(side, wr[side])},
                      a_xy=tuple(a), feet_force={side: f}, friction_needed=float(math.hypot(f[0], f[1]) / f[2]),
                      margins={"residual": float(res)})

    def _cop_for(self, side, a_xy):
        tau, wr, _ = self.solve([side], a_xy)
        return self.cop_local(side, wr[side])

    def double_support(self, pose: dict) -> Result:
        self.set_pose(pose)
        tau, wr, res = self.solve(["left", "right"])
        fn = max(math.hypot(w[0], w[1]) / max(w[2], 1e-9) for w in wr.values())
        poly = unary_union([self.sole_polygon_world(s)[0] for s in ("left", "right")]).convex_hull
        c = self.com()
        pt = shapely.Point(c[0], c[1])
        margin = poly.exterior.distance(pt) * (1 if poly.contains(pt) else -1)
        return Result(tau_frac=dict(zip(ACT, (tau / self.stall).round(4))), com=c, cop={s: self.cop_local(s, w) for s, w in wr.items()},
                      a_xy=(0.0, 0.0), feet_force={s: w[:3] for s, w in wr.items()}, friction_needed=fn,
                      margins={"support_margin_m": float(margin), "residual": float(res),
                               "support_x_m": poly.bounds[0::2], "support_y_m": poly.bounds[1::2]})

    def com_height_above_sole(self) -> float:
        z = min(self.sole_polygon_world(s)[1] for s in ("left", "right"))
        return float(self.com()[2] - z)

    def stance_halfwidth(self) -> float:
        return float(abs(self.sole_center_world("left")[1] - self.sole_center_world("right")[1]) / 2)

    def inertia_about_com(self) -> np.ndarray:
        """Whole-body 3x3 inertia about the centre of mass, world axes (pose as set)."""
        m, d = self.m, self.d
        c = self.com()
        I = np.zeros((3, 3))
        for b in range(1, m.nbody):
            R = d.ximat[b].reshape(3, 3)
            Ib = R @ np.diag(m.body_inertia[b]) @ R.T
            r = d.xipos[b] - c
            I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        return I


POSES = {  # degrees; the policy poses are the v9c means measured in MuJoCo (outputs/redesign/probe_baseline.json)
    "zero": {},
    "v9c_stand": {"head_joint": 45.0, "tail_joint": 44.4, "left_femur_joint": 23.5, "right_femur_joint": 23.5,
                  "left_tibia_joint": 45.0, "right_tibia_joint": 45.0},
    "v9c_walk": {"head_joint": -43.3, "tail_joint": -41.5, "left_femur_joint": -17.8, "right_femur_joint": -17.8,
                 "left_tibia_joint": -36.2, "right_tibia_joint": -36.2},  # hips ~-3.6 in the run; 0 here so the soles stay flat
}
