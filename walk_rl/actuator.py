"""LX-16A servo model (walk_task.yaml actuator.model: lx16a): explicit joint torque per physics
substep, the same formulas for torch (task_core, both Isaac stacks) and numpy (pi_runner MuJoCo
backend). Parameters: robot_config.yaml `actuator_model` (fitted to the real servos), per-env
randomization: walk_task.yaml domain_randomization.actuator_model.

    q_ref = move-time ramp start -> goal over the control period, delayed by `latency`
    e     = q_ref - q with a deadband removed           (no drive inside +-deadband)
    duty  = clamp(kp * e, -1, 1)                        (firmware P gain, saturating)
    tau   = duty * stall - (stall / no_load) * qd       (DC motor: back-EMF torque-speed line)
            - friction                                  (Coulomb, discrete: see below)
    stall, no_load scale with the bus voltage V / V_ref.

Friction is applied so it can stop the joint but never reverse it within one substep:
|tau_f| <= coulomb and |tau_f| <= J_arm * |qd| / dt (the armature is a lower bound of the joint
inertia, so this is stable at 200 Hz, unlike a smooth tanh(qd / eps) law). A joint at rest
then holds against loads below the friction level, as the gearbox does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LX16AParams:
    """Nominal values (robot_config.yaml actuator_model + actuator)."""
    kp: float  # duty per rad
    deadband: float  # rad
    latency: float  # s
    armature: float  # kg m^2
    coulomb: float  # N m
    stall: float  # N m at v_ref
    no_load: float  # rad/s at v_ref (datasheet x no_load_speed_scale)
    v_ref: float
    v_nominal: float  # bus voltage used when not randomized

    @classmethod
    def from_robot_config(cls, rc) -> "LX16AParams":
        am = rc.raw["actuator_model"]
        return cls(kp=am["kp_duty_per_rad"], deadband=math.radians(am["deadband_deg"]), latency=am["latency_s"],
                   armature=am["armature_kgm2"], coulomb=am["coulomb_friction_nm"],
                   stall=rc.actuator_stall_torque_nm * am.get("stall_torque_scale", 1.0),
                   no_load=rc.actuator_no_load_speed_rad_per_sec * am["no_load_speed_scale"],
                   v_ref=am["reference_voltage_v"], v_nominal=am["measured_voltage_v"])


def ramp_reference(start, goal, t_in_period, period, latency, clip):
    """Servo's internal setpoint `t_in_period` seconds into the control period (end of the
    substep), `latency` late. The previous ramp ended at `start`, so before the delayed ramp
    begins the setpoint is `start`. `clip(x, lo, hi)` is torch.clamp or np.clip."""
    frac = clip((t_in_period - latency) / period, 0.0, 1.0)
    return start + (goal - start) * frac


def lx16a_torque(q_ref, q, qd, kp, deadband, coulomb, stall, no_load, armature, dt, xp):
    """Joint torque [N m]. All arrays broadcast; `stall` and `no_load` already voltage-scaled.
    `xp` is torch or numpy (needs sign, abs, clip/clamp via `_clip`, minimum)."""
    e = q_ref - q
    e = xp.sign(e) * _relu(xp.abs(e) - deadband, xp)
    duty = _clip(kp * e, -1.0, 1.0, xp)
    tau = duty * stall - (stall / no_load) * qd
    f = xp.minimum(coulomb, armature * xp.abs(qd) / dt)
    return tau - xp.sign(qd) * f


def _clip(x, lo, hi, xp):
    return xp.clamp(x, lo, hi) if hasattr(xp, "clamp") else xp.clip(x, lo, hi)


def _relu(x, xp):
    return xp.clamp(x, min=0.0) if hasattr(xp, "clamp") else xp.maximum(x, 0.0)
