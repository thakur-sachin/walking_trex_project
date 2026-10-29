"""Checks of the Trex v2 asset (assets/robots/trex_v2) on CPU:

1. loads in the MuJoCo backend (MujocoTrex(asset_dir=...)), masses, ranges, self-collision pairs,
   no self-contact at the zero pose;
2. open-loop standing on the fitted servo model (zero pose and a forward-knee crouch), 10 s;
3. static numbers (redesign/levers.figures) for v2 vs the current robot model;
4. ankle contact range with the v2 foot (shank / lower idler vs foot), by mesh sweep;
5. leg/foot clearances over v9d's gait: v9d is run on the CURRENT robot in MuJoCo (the deployed loop),
   its joint trajectories are replayed kinematically on v2, and for every frame the minimum distance
   between the left and right leg parts is measured, two ways:
     - MuJoCo mj_geomDistance between the link collision geoms (convex hulls: conservative; libccd, see
       pi_runner/mujoco_backend.py: the native collider gave spurious zeros on these meshes),
     - surface samples of the real meshes incl. the 4-bar couplers (KD-tree; about +-1 mm).

    python redesign/check_trex_v2.py   -> outputs/redesign/trex_v2_check.json
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pi_runner.mujoco_backend as mb  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402
from redesign.levers import figures  # noqa: E402

V2 = ROOT / "assets" / "robots" / "trex_v2"
JN = ["head_joint", "tail_joint", "left_hip_joint", "right_hip_joint", "left_femur_joint", "right_femur_joint",
      "left_tibia_joint", "right_tibia_joint"]
SAMPLE_PER_MM2 = 1.0


# ------------------------------------------------------------------------------- 1 + 2
def load_check():
    sim = mb.MujocoTrex(actuator="lx16a", asset_dir=V2)
    m, d = sim.model, sim.data
    out = {"total_mass_kg": float(m.body_subtreemass[1]), "npair": int(m.npair),
           "ranges_deg": {n: np.degrees(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]).round(1).tolist() for n in JN}}
    mujoco.mj_forward(m, d)
    self_c = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]),
               mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2]))
              for i in range(d.ncon) if d.contact[i].geom1 != sim.floor and d.contact[i].geom2 != sim.floor]
    out["self_contacts_zero_pose"] = self_c
    return out


def stand_test(pose_deg: dict, seconds=10.0):
    sim = mb.MujocoTrex(actuator="lx16a", asset_dir=V2)
    m, d = sim.model, sim.data
    for n, v in pose_deg.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = math.radians(v)
    for s in ("left", "right"):
        a = pose_deg.get(f"{s}_femur_joint", 0.0) - pose_deg.get(f"{s}_tibia_joint", 0.0)
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_ankle_joint")]] = math.radians(a)
    # drop it onto the floor from its pose: lowest point 2 mm above the floor
    mujoco.mj_forward(m, d)
    zmin = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in range(m.ngeom) if g != sim.floor)
    goal = np.radians([pose_deg.get(n, 0.0) for n in JN])
    sim.goal = goal.copy()
    sim.queue.clear()
    sim.queue.append(goal.copy())
    tilt, selfc = 0.0, 0
    for k in range(int(seconds * 25)):
        sim.write(goal)
        g = sim._rot().T @ np.array([0, 0, -1.0])
        tilt = max(tilt, math.degrees(math.acos(-g[2])))
        selfc = max(selfc, sum(1 for i in range(d.ncon) if d.contact[i].geom1 != sim.floor and d.contact[i].geom2 != sim.floor))
        if sim.failed():
            break
    return {"pose": pose_deg, "survived_s": round(sim.time, 2), "fell": sim.failed(), "max_tilt_deg": round(tilt, 2),
            "max_self_contacts": selfc}


# ------------------------------------------------------------------------------- 4 + 5
class Geometry:
    """v2 link meshes (assembly frame, mm) + the 4-bar couplers, moved by joint angles."""

    def __init__(self):
        from redesign.make_trex_v2 import build_parts, joint_positions, mirror_y  # reuse the asset's own parts
        P = build_parts()
        self.J = joint_positions()
        self.parts = {}
        for side in ("left", "right"):
            for ln in ("femur", "tibia", "ankle", "foot"):
                ms = [p.mesh for p in P[f"{side}_{ln}"] if p.in_mesh]
                self.parts[f"{side}:{ln}"] = trimesh.util.concatenate(ms)
            for p in P[f"{side}_tibia"] + P[f"{side}_ankle"]:
                if not p.in_mesh:
                    self.parts[f"{side}:{p.name}"] = p.mesh
        self.samples = {k: trimesh.sample.sample_surface(m, int(m.area * SAMPLE_PER_MM2), seed=0)[0] for k, m in self.parts.items()}
        self.trees = {}

    def piv(self, side):
        hip = np.array(self.J[f"{side}_hip_joint"][2])
        F = np.array(self.J[f"{side}_femur_joint"][2])
        T = np.array(self.J[f"{side}_tibia_joint"][2])
        A = np.array(self.J[f"{side}_ankle_joint"][2])
        off = np.array([-50.0, 0, 0])
        return hip, F, T, A, F + off, T + off

    def transforms(self, q: dict):
        """4x4 per part (mm) for joint angles q [rad], torso fixed."""
        out = {}
        for side, ax_hip in (("left", -1.0), ("right", 1.0)):
            hip, F, T, A, Fp, Tp = self.piv(side)
            f, t, h = q.get(f"{side}_femur_joint", 0.0), q.get(f"{side}_tibia_joint", 0.0), q.get(f"{side}_hip_joint", 0.0)

            def roty(a, about):
                M = np.eye(4)
                M[:3, :3] = Rot.from_euler("y", a).as_matrix()
                M[:3, 3] = about - M[:3, :3] @ about
                return M

            def tr(dv):
                M = np.eye(4)
                M[:3, 3] = dv
                return M
            Rf = roty(-f, F)
            Tn = (Rf @ np.r_[T, 1])[:3]
            KC = tr(Tn - T)
            Sh = KC @ roty(-f + t, T)
            An = (Sh @ np.r_[A, 1])[:3]
            Rh = np.eye(4)
            Rh[:3, :3] = Rot.from_rotvec(h * np.array([ax_hip, 0, 0])).as_matrix()
            Rh[:3, 3] = hip - Rh[:3, :3] @ hip
            loc = {"femur": np.eye(4), "tibia": Rf, "ankle": Sh, "foot": tr(An - A), "upper_idler": roty(-f, Fp),
                   "knee_coupler": KC, "lower_idler": KC @ roty(-f + t, Tp)}
            for k, M in loc.items():
                out[f"{side}:{k}"] = Rh @ M
        return out

    def min_dist(self, a, b, T):
        """Min distance between parts a and b [mm] from dense surface samples (about +-1 mm): a's samples,
        moved into b's rest frame, queried against b's (static) KD-tree."""
        if b not in self.trees:
            self.trees[b] = cKDTree(self.samples[b])
        M = np.linalg.inv(T[b]) @ T[a]
        pa = self.samples[a] @ M[:3, :3].T + M[:3, 3]
        return float(self.trees[b].query(pa, distance_upper_bound=60.0)[0].min())

    def inside_count(self, a, b, T):
        """Points of a inside b (b's watertight mesh at its pose): for penetration checks."""
        pa = self.samples[a] @ T[a][:3, :3].T + T[a][:3, 3]
        Tb = np.linalg.inv(T[b])
        pl = pa @ Tb[:3, :3].T + Tb[:3, 3]
        mb_ = self.parts[b]
        lo, hi = mb_.bounds
        sel = np.all((pl > lo) & (pl < hi), axis=1)
        return int(mb_.contains(pl[sel]).sum()) if sel.any() and mb_.is_watertight else -1


def ankle_range(geo: Geometry):
    """Femur 0, sweep the tibia: the first ankle angle (femur - tibia) at which the v2 foot comes within
    0.5 mm of the shank or the lower idler (separated at the zero pose)."""
    res = {}
    T0 = geo.transforms({})
    base = {p: geo.min_dist("left:foot", p, T0) for p in ("left:ankle", "left:lower_idler")}
    for sgn in (1, -1):
        hit = None
        for a in np.arange(1, 121, 1.0):
            T = geo.transforms({"left_tibia_joint": math.radians(-sgn * a)})  # ankle = 0 - tibia
            for p in base:
                d_ = geo.min_dist("left:foot", p, T)
                if d_ < 0.5 and base[p] >= 0.5:
                    hit = (sgn * a, p)
                    break
            if hit:
                break
        res["+" if sgn > 0 else "-"] = hit
    res["zero_pose_distance_mm"] = base
    return res


def record_v9d(cmds=(0.0, 0.10, 0.20, 0.25), seconds=20.0):
    run = ROOT / "outputs" / "walk_runs" / "v9d"
    traj = {}
    for c in cmds:
        runner = PolicyRunner(run)
        tc = runner.cfg.termination
        sim = mb.MujocoTrex(friction=0.5, control_hz=runner.cfg.sim.control_hz, actuator="lx16a",
                            max_tilt_deg=tc.max_tilt_deg, min_head_height_m=tc.min_head_height_m,
                            ground_contact_force_n=tc.ground_contact_force_n)
        runner.reset_state(np.zeros(8, dtype=np.float32))
        runner.latch_heading(sim.read().yaw)
        Q = []
        for k in range(int(seconds * 25)):
            sim.write(runner.step(sim.read(), c))
            Q.append(sim.data.qpos[sim.qadr].copy())
            if sim.failed():
                break
        traj[c] = {"q": np.array(Q), "fell": sim.failed()}
    return traj


PAIRS = [("left:foot", "right:foot"), ("left:foot", "right:ankle"), ("right:foot", "left:ankle"), ("left:ankle", "right:ankle"),
         ("left:tibia", "right:tibia"), ("left:knee_coupler", "right:knee_coupler"), ("left:lower_idler", "right:lower_idler"),
         ("left:foot", "right:lower_idler"), ("right:foot", "left:lower_idler"), ("left:knee_coupler", "right:ankle"),
         ("right:knee_coupler", "left:ankle"), ("left:femur", "right:femur")]


def replay_clearance(geo: Geometry, traj, every=4):
    # MuJoCo convex-hull distances on the v2 model
    m = mb.build_model(mb.RobotConfig.load(V2 / "robot_config.yaml"), actuator="lx16a", asset_dir=V2)
    d = mujoco.MjData(m)
    gid = {b: [g for g in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == b]
           for b in ("left_foot", "right_foot", "left_ankle", "right_ankle", "left_tibia", "right_tibia")}
    mj_pairs = [("left_foot", "right_foot"), ("left_foot", "right_ankle"), ("right_foot", "left_ankle"),
                ("left_ankle", "right_ankle"), ("left_tibia", "right_tibia")]
    jadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JN + ["left_ankle_joint", "right_ankle_joint"]}
    res = {}
    for c, tr_ in traj.items():
        Q = tr_["q"][::every]
        mesh_min = {f"{a}~{b}": 1e9 for a, b in PAIRS}
        mj_min = {f"{a}~{b}": 1e9 for a, b in mj_pairs}
        for q in Q:
            qd = dict(zip(JN, q))
            T = geo.transforms(qd)
            for a, b in PAIRS:
                mesh_min[f"{a}~{b}"] = min(mesh_min[f"{a}~{b}"], geo.min_dist(a, b, T))
            d.qpos[:] = 0
            d.qpos[3] = 1
            for n in JN:
                d.qpos[jadr[n]] = qd[n]
            for s in ("left", "right"):
                d.qpos[jadr[f"{s}_ankle_joint"]] = qd[f"{s}_femur_joint"] - qd[f"{s}_tibia_joint"]
            mujoco.mj_kinematics(m, d)
            for a, b in mj_pairs:
                best = 1e9
                for g1, g2 in itertools.product(gid[a], gid[b]):
                    ft = np.zeros(6)
                    best = min(best, mujoco.mj_geomDistance(m, d, g1, g2, 0.05, ft))
                mj_min[f"{a}~{b}"] = min(mj_min[f"{a}~{b}"], best * 1000)
        qdeg = np.degrees(tr_["q"])
        res[str(c)] = {"frames": int(len(Q)), "v9d_fell_on_current_robot": tr_["fell"],
                       "hip_deg_range": [float(qdeg[:, 2:4].min()), float(qdeg[:, 2:4].max())],
                       "mesh_min_mm": {k: (round(v, 1) if v < 60 else ">60") for k, v in mesh_min.items()},
                       "mujoco_hull_min_mm": {k: round(v, 1) for k, v in mj_min.items()}}
    return res


def main():
    out = {"load": load_check()}
    print("[v2] load:", json.dumps(out["load"]))
    out["stand"] = [stand_test({}),
                    stand_test({"left_femur_joint": 20.0, "right_femur_joint": 20.0, "left_tibia_joint": 40.0, "right_tibia_joint": 40.0}),
                    stand_test({"left_femur_joint": -17.8, "right_femur_joint": -17.8, "left_tibia_joint": -36.2, "right_tibia_joint": -36.2})]
    for s in out["stand"]:
        print("[v2] stand:", s)
    f_v2, f_cur = figures({}, asset_dir=V2), figures({})
    keys = ["mass_kg", "com_h_m", "halfwidth_m", "ss_hip_switch", "ss_hip_mid", "ss_mu_switch", "ss_femur", "ss_tibia",
            "ds_stand_max", "ds_support_margin_m", "ss_static_margin_m", "sole_inner_gap_m", "roll_est_deg", "t_roll_s", "t_pitch_s"]
    out["statics"] = {"v2": {k: f_v2[k] for k in keys}, "current_model": {k: f_cur[k] for k in keys}}
    for k in keys:
        print(f"[v2] statics {k:22s} current {f_cur[k]:8.3f}  v2 {f_v2[k]:8.3f}")
    geo = Geometry()
    out["ankle_range_v2_foot"] = ankle_range(geo)
    print("[v2] ankle contact (femur 0):", out["ankle_range_v2_foot"])
    traj = record_v9d()
    out["clearance"] = replay_clearance(geo, traj)
    for c, r in out["clearance"].items():
        print(f"[v2] v9d cmd {c}: hips {np.round(r['hip_deg_range'], 1)} deg; mesh min (mm) {r['mesh_min_mm']}")
        print(f"[v2]      MuJoCo hull min (mm) {r['mujoco_hull_min_mm']}")
    (ROOT / "outputs" / "redesign" / "trex_v2_check.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
