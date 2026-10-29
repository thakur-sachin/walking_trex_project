"""Geometric joint-range check from the link meshes: sweep each joint and find the angle at which
two link meshes start to interpenetrate (beyond what they already overlap at the zero pose, e.g.
a servo horn modelled inside its bracket).

Method: surface points are sampled on each moving link (seeded, ~1 per 4 mm^2); for each pair the
count of points of A inside B and of B inside A (filled 1 mm voxel grids of the meshes) is compared
with the zero-pose count. The first
angle where the excess exceeds `min_points` is the collision angle. Pose kinematics come from the
MuJoCo model (same URDF), base fixed.

Limits of this check (say them with the numbers):
  - the 4-bar idler links, servo horns as separate parts, screws and cables are NOT in the meshes;
  - the merged link meshes may include the servo bodies as plain blocks;
  - left_femur.stl is not a clean manifold (non-manifold edges): inside tests on it are approximate.

    python redesign/range_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from redesign.variant_model import build_variant_model, ASSET_DIR  # noqa: E402

MESH_DIR = ASSET_DIR / "meshes"
VOXEL_M = 0.001


class Kin:
    def __init__(self, variant=None):
        rc = RobotConfig.load(ASSET_DIR / "robot_config.yaml")
        self.m = build_variant_model(rc, variant or {}, actuator="lx16a")
        self.d = mujoco.MjData(self.m)
        self.meshes, self.samples, self.offset, self.vox = {}, {}, {}, {}
        rng = np.random.default_rng(0)
        for b in range(1, self.m.nbody):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
            mesh = trimesh.load(str(MESH_DIR / f"{name}.stl"))
            mesh.apply_scale(0.001)
            # body frame: mesh (assembly frame) + visual origin = -(zero-pose body position)
            self.d.qpos[:] = 0
            self.d.qpos[3] = 1.0
            mujoco.mj_kinematics(self.m, self.d)
            p0 = self.d.xpos[b].copy()  # free joint qpos = 0: base at the world origin = assembly frame
            mesh.apply_translation(-p0)
            self.meshes[name] = mesh
            n = int(mesh.area / 4e-6)
            pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
            self.samples[name] = pts
        self.jid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(self.m.njnt)}

    def set_pose(self, q: dict[str, float]):
        """q: joint name -> rad (actuated and ankle; ankle follows the linkage if not given)."""
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        for n, v in q.items():
            d.qpos[self.m.jnt_qposadr[self.jid[n]]] = v
        for s in ("left", "right"):
            if f"{s}_ankle_joint" not in q:
                f = q.get(f"{s}_femur_joint", 0.0)
                t = q.get(f"{s}_tibia_joint", 0.0)
                d.qpos[self.m.jnt_qposadr[self.jid[f"{s}_ankle_joint"]]] = f - t
        mujoco.mj_kinematics(self.m, d)

    def world(self, name: str, pts: np.ndarray) -> np.ndarray:
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        R = self.d.xmat[b].reshape(3, 3)
        return pts @ R.T + self.d.xpos[b]

    def to_body(self, name: str, pts_w: np.ndarray) -> np.ndarray:
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        R = self.d.xmat[b].reshape(3, 3)
        return (pts_w - self.d.xpos[b]) @ R

    def overlap(self, a: str, b: str) -> int:
        pa = self.to_body(b, self.world(a, self.samples[a]))
        pb = self.to_body(a, self.world(b, self.samples[b]))
        return self._inside(b, pa) + self._inside(a, pb)

    def _inside(self, name, pts) -> int:
        """Points inside the link's solid: lookup in a filled 1 mm voxel grid of the mesh (body frame)."""
        if name not in self.vox:
            v = self.meshes[name].voxelized(pitch=VOXEL_M).fill()
            self.vox[name] = (v.matrix.copy(), np.linalg.inv(v.transform))
        mat, inv = self.vox[name]
        ijk = np.round(pts @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(mat.shape)), axis=1)
        return int(mat[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]].sum())


SWEEPS = [  # joint, pairs to watch, other joints held
    ("head_joint", [("base_link", "head")], {}),
    ("tail_joint", [("base_link", "tail")], {}),
    ("left_hip_joint", [("base_link", "left_femur"), ("base_link", "left_tibia"), ("left_foot", "right_foot"),
                        ("left_ankle", "right_ankle"), ("left_tibia", "right_tibia")], {}),
    ("left_femur_joint", [("left_femur", "left_tibia"), ("base_link", "left_tibia"), ("left_femur", "left_ankle"),
                          ("base_link", "left_ankle"), ("left_femur", "left_foot")], {}),
    ("left_tibia_joint", [("left_tibia", "left_ankle"), ("left_femur", "left_ankle"), ("left_tibia", "left_foot"),
                          ("left_femur", "left_foot")], {}),
]


def sweep(kin: Kin, joint: str, pairs, held: dict, angles_deg, min_points: int = 20):
    base = {}
    kin.set_pose(dict(held))
    for p in pairs:
        base[p] = kin.overlap(*p)
    first = {}
    for a in angles_deg:
        q = dict(held)
        q[joint] = np.radians(a)
        kin.set_pose(q)
        for p in pairs:
            if p in first:
                continue
            ex = kin.overlap(*p) - base[p]
            if ex > min_points:
                first[p] = a
    return base, first


def main():
    kin = Kin()
    out = {}
    print(f"[range] samples per link: { {k: len(v) for k, v in kin.samples.items()} }")
    for joint, pairs, held in SWEEPS:
        for sgn in (+1, -1):
            angles = sgn * np.arange(2.0, 122.0, 2.0)
            base, first = sweep(kin, joint, pairs, held, angles)
            for p in pairs:
                key = f"{joint} {'+' if sgn > 0 else '-'} {p[0]}~{p[1]}"
                out[key] = {"zero_pose_overlap_pts": base[p], "first_collision_deg": first.get(p)}
                print(f"[range] {key:60s} zero-overlap {base[p]:5d}  first collision at {first.get(p, 'none <= 120')}")
    # tibia sweep with the femur at its walking / standing values (policy-measured means)
    for fem in (-17.5, 23.5):
        for sgn in (+1, -1):
            angles = sgn * np.arange(2.0, 122.0, 2.0)
            pairs = [("left_tibia", "left_ankle"), ("left_femur", "left_ankle"), ("left_tibia", "left_foot"), ("left_femur", "left_foot")]
            base, first = sweep(kin, "left_tibia_joint", pairs, {"left_femur_joint": np.radians(fem)}, angles)
            for p in pairs:
                key = f"left_tibia_joint {'+' if sgn > 0 else '-'} (femur {fem:+.1f}) {p[0]}~{p[1]}"
                out[key] = {"zero_pose_overlap_pts": base[p], "first_collision_deg": first.get(p)}
                print(f"[range] {key:60s} zero-overlap {base[p]:5d}  first collision at {first.get(p, 'none <= 120')}")
    dst = ROOT / "outputs" / "redesign" / "range_check.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"[range] wrote {dst}")


if __name__ == "__main__":
    main()
