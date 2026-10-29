"""Load the exported Trex USD in Newton (SolverMuJoCo / MJWarp) and check it against the MuJoCo sim2sim model:
joints, masses, tendons, and the ankle linkage residual |ankle - (femur - tibia)| while the robot stands on PD holds.

    ~/trex_stack/IsaacLab_dev/.venv/bin/python lab3/check_newton_model.py
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import newton

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USD = PROJECT_ROOT / "outputs" / "lab3" / "trex" / "usd" / "trex" / "trex.usda"

builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
from pxr import Usd

stage = Usd.Stage.Open(str(USD))
stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("mujoco")  # the converter leaves it unset
builder.add_usd(stage, xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()))
builder.add_ground_plane()
model = builder.finalize()
print("bodies", model.body_count, "joints", model.joint_count, "dofs", model.joint_dof_count)
print("joint names", model.joint_label if hasattr(model, "joint_label") else model.joint_key)
masses = model.body_mass.numpy()
print("total mass kg", round(float(masses.sum()), 4))
solver = newton.solvers.SolverMuJoCo(model)
mj = solver.mj_model
print("mujoco model: ntendon", mj.ntendon, "tendon_limited", mj.tendon_limited[:], "range", mj.tendon_range[:].tolist(), "neq", mj.neq)
import mujoco  # noqa: E402

print("newton->mujoco: timestep", mj.opt.timestep, "cone", mj.opt.cone, "impratio", mj.opt.impratio, "noslip", mj.opt.noslip_iterations)
for i in range(min(mj.ngeom, 14)):
    print(f"  geom {i:2d} {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i)!s:18} type {mj.geom_type[i]} condim {mj.geom_condim[i]} "
          f"friction {mj.geom_friction[i].round(4).tolist()} solref {mj.geom_solref[i].round(4).tolist()} "
          f"solimp {mj.geom_solimp[i].round(3).tolist()} margin {mj.geom_margin[i]:.4f} priority {mj.geom_priority[i]}")
print("  dof armature", mj.dof_armature.round(4).tolist(), "damping", mj.dof_damping.round(4).tolist(), "frictionloss", mj.dof_frictionloss.round(4).tolist())
