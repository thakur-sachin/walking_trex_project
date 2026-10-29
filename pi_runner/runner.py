"""Backend-agnostic policy runner: what the Raspberry Pi executes every control tick.

It reproduces the actor side of walk_rl/task_core.py exactly (pi_runner/test_obs_parity.py
checks this against the simulator): observation frame, history ring buffer, finite-
difference servo velocity, gait clock, heading error, then the action pipeline (clamp,
low-pass, scale, 0.24 deg quantization, speed-limited goal, joint limits). It does NOT model
what the hardware does by itself (bus latency, servo move-time ramp, sensor noise); a
backend provides readings and executes goals:

    reading = backend.read()            # Reading(gravity_b, gyro_b, yaw, q)
    goal = runner.step(reading, cmd_vx) # 8 servo goals [rad], actuated_joints order
    backend.write(goal, move_time_s=runner.dt)

Backends: pi_runner/mujoco_backend.py (sim2sim) and pi_runner/hardware.py (BNO055 + LX-16A bus).
Only numpy + yaml are imported, so it runs on the Pi.

Arithmetic is float32, like training. It matters: a saturated action times a 7.5-degree-
multiple limit lands exactly on a half servo tick (15 deg = 62.5 ticks), and float32 vs
float64 then round to different ticks.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.policy import NumpyPolicy  # noqa: E402
from walk_rl.config import robot_asset_dir  # noqa: E402
from walk_rl.config import apply_checkpoint_interface, joint_center_rad, joint_scale_rad, load_task_config, obs_dims  # noqa: E402

ROBOT_CONFIG = PROJECT_ROOT / "assets" / "robots" / "trex" / "robot_config.yaml"


@dataclass
class Reading:
    """One tick of sensor data, in the robot's body frame (x forward, y left, z up)."""
    gravity_b: np.ndarray  # (3,) unit gravity direction (BNO055 gravity vector, normalized)
    gyro_b: np.ndarray  # (3,) rad/s
    yaw: float  # rad, BNO055 fused heading (only differences to the latched reference are used)
    q: np.ndarray  # (8,) servo positions [rad], actuated_joints order, zero = CAD neutral pose


class Safety:
    """Independent of the policy: trips on excessive tilt and holds a latched state until
    reset. The hardware loop maps a trip to "unload servos". Joint-limit and per-tick rate
    clamps are part of the action pipeline (identical to training), so they always apply."""

    def __init__(self, max_tilt_deg: float):
        self.cos_max = math.cos(math.radians(max_tilt_deg))
        self.tripped = None

    def check(self, r: Reading) -> bool:
        if self.tripped is None and r.gravity_b[2] > -self.cos_max:
            self.tripped = f"tilt beyond {math.degrees(math.acos(self.cos_max)):.0f} deg"
        return self.tripped is None


F = np.float32


class PolicyRunner:
    def __init__(self, run_dir: str | Path, policy_file: str = "policy_numpy.npz", robot_config: str | Path | None = None):
        """`robot_config`: the robot's robot_config.yaml (default: the run's `robot.asset_dir`)."""
        run_dir = Path(run_dir)
        self.policy = NumpyPolicy(run_dir / policy_file)
        meta = self.policy.meta
        cfg = load_task_config(run_dir / "walk_task.yaml")
        apply_checkpoint_interface(cfg, meta)
        self.asset_dir = robot_asset_dir(cfg)
        rc = RobotConfig.load(robot_config if robot_config is not None else self.asset_dir / "robot_config.yaml")
        self.cfg, self.rc = cfg, rc

        self.act_names = list(rc.actuated_joints)
        assert self.act_names == meta.get("actuated_joints", self.act_names), "joint order differs from the policy's"
        self.dt = 1.0 / cfg.sim.control_hz
        self.joint_limit = np.array(joint_scale_rad(cfg, rc, self.act_names), dtype=F)
        if "joint_limit_rad" in meta:  # the scale the policy was trained with
            assert np.allclose(self.joint_limit, meta["joint_limit_rad"], atol=1e-6), \
                f"joint scale {self.joint_limit} differs from the policy's {meta['joint_limit_rad']}"
        self.act_scale = self.joint_limit * F(cfg.actions.scale_frac_of_limit)
        rng = [rc.joint_range_rad(n) for n in self.act_names]  # physical (possibly asymmetric) range
        self.joint_lo = np.array([r[0] for r in rng], dtype=F)
        self.joint_hi = np.array([r[1] for r in rng], dtype=F)
        cpl = rc.ankle_coupled_limit_rad()
        self.ankle_couple = None if cpl is None else (F(cpl[0]), F(cpl[1]))
        self.fem_a = [self.act_names.index(f"{s}_femur_joint") for s in ("left", "right")]
        self.tib_a = [self.act_names.index(f"{s}_tibia_joint") for s in ("left", "right")]
        self.default_q = np.zeros(len(self.act_names), dtype=F)
        self.act_center = np.array(joint_center_rad(cfg, self.act_names), dtype=F)
        assert np.allclose(self.act_center, meta.get("act_center_rad", np.zeros_like(self.act_center)), atol=1e-6), \
            f"joint centre {self.act_center} differs from the policy's {meta.get('act_center_rad')}"
        self.servo_res = F(math.radians(rc.actuator_spec["position_accuracy_deg"]))
        self.max_step = F(rc.actuator_no_load_speed_rad_per_sec * self.dt)
        self.alpha = F(cfg.actions.smoothing_alpha)
        self.quantize = cfg.actions.quantize_to_servo_resolution
        ob = cfg.observation
        self.history_len = int(ob.history_len)
        self.use_heading = bool(getattr(getattr(ob, "heading", None), "enabled", False))
        self.use_clock = bool(cfg.gait.use_phase_clock)
        self.gyro_scale, self.jv_scale = ob.scales.gyro, ob.scales.joint_vel
        self.frame_dim, self.obs_dim, _ = obs_dims(cfg)
        assert self.obs_dim == self.policy.obs_dim, f"config gives {self.obs_dim} inputs, policy takes {self.policy.obs_dim}"
        self.safety = Safety(cfg.termination.max_tilt_deg)
        self.reset_state(np.zeros(len(self.act_names)))

    # ------------------------------------------------------------------ helpers
    def gait_frequency(self, cmd_vx: float) -> float:
        g = self.cfg.gait
        f = F(g.frequency_hz)
        k = getattr(g, "frequency_per_mps", 0.0)
        if k:
            f = f + F(k) * (F(cmd_vx) - F(g.frequency_ref_mps))
            f = min(max(f, F(g.frequency_range_hz[0])), F(g.frequency_range_hz[1]))
        return F(f)

    def _quant(self, q: np.ndarray) -> np.ndarray:
        return np.round(q / self.servo_res) * self.servo_res

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    # -------------------------------------------------------------------- state
    def reset_state(self, servo_goal: np.ndarray, phase: float = 0.0, yaw_ref: float | None = None) -> None:
        """Start of an episode. `servo_goal`: the goal the servos currently hold (the pose
        they were sent to). The first reading after this fills the history buffer."""
        A = len(self.act_names)
        self.cmd_target = np.asarray(servo_goal, dtype=F).copy()
        self.smoothed = ((self.default_q - self.act_center) / self.act_scale).astype(F)
        self.last_action = np.zeros(A, dtype=F)
        self.phase = phase
        self.yaw_ref = yaw_ref
        self.q_prev = None
        self.hist = None
        self.safety.tripped = None

    def latch_heading(self, yaw: float) -> None:
        self.yaw_ref = yaw

    # --------------------------------------------------------------------- tick
    def observation(self, r: Reading, cmd_vx: float) -> np.ndarray:
        g = np.asarray(r.gravity_b, dtype=F)
        g = g / np.linalg.norm(g)
        q_meas = self._quant(np.asarray(r.q, dtype=F))
        if self.q_prev is None:
            self.q_prev = q_meas
        dq = (q_meas - self.q_prev) / F(self.dt)
        self.q_prev = q_meas
        cmd_vx = F(cmd_vx)
        walking = cmd_vx > F(self.cfg.gait.walking_cmd_threshold_mps)
        parts = [g, np.asarray(r.gyro_b, dtype=F) * F(self.gyro_scale), [cmd_vx], (q_meas - self.act_center) / self.joint_limit,
                 dq * F(self.jv_scale), self.last_action]
        if self.use_clock:
            ang = F(2 * math.pi) * F(self.phase)
            parts.append(np.array([np.sin(ang), np.cos(ang)], dtype=F) * F(walking))
        if self.use_heading:
            if self.yaw_ref is None:
                self.yaw_ref = r.yaw
            he = self._wrap(r.yaw - self.yaw_ref)
            parts.append([math.sin(he), math.cos(he)])
        frame = np.concatenate([np.asarray(p, dtype=F).ravel() for p in parts])
        if self.hist is None:
            self.hist = np.tile(frame, (self.history_len, 1))
        else:
            self.hist = np.roll(self.hist, 1, axis=0)
            self.hist[0] = frame
        return self.hist.reshape(-1)

    def limit_goal(self, goal: np.ndarray) -> np.ndarray:
        """Same as task_core._limit_goal: joint ranges, then lo <= femur - tibia <= hi."""
        goal = np.minimum(np.maximum(goal, self.joint_lo), self.joint_hi).astype(F)
        if self.ankle_couple is None:
            return goal
        lo, hi = self.ankle_couple
        f, t = goal[self.fem_a], goal[self.tib_a]
        t = np.maximum(np.minimum(t, f - lo), f - hi)
        t = np.maximum(np.minimum(t, self.joint_hi[self.tib_a]), self.joint_lo[self.tib_a])
        f = np.maximum(np.minimum(f, t + hi), t + lo)
        goal[self.fem_a], goal[self.tib_a] = f, t
        return goal

    def step(self, r: Reading, cmd_vx: float) -> np.ndarray:
        """One control tick: reading -> new 8 servo goals [rad]."""
        obs = self.observation(r, cmd_vx)
        self.last_obs = obs
        a = np.clip(self.policy(obs.astype(np.float64)), -1.0, 1.0).astype(F)
        self.last_raw_action = a
        self.last_action = a
        self.smoothed = self.alpha * self.smoothed + (1.0 - self.alpha) * a
        raw = self.act_center + self.smoothed * self.act_scale
        if self.quantize:
            raw = self._quant(raw)
        goal = self.cmd_target + np.clip(raw - self.cmd_target, -self.max_step, self.max_step)
        goal = self.limit_goal(goal)
        self.cmd_target = goal
        self.phase = F((F(self.phase) + self.gait_frequency(cmd_vx) * F(self.dt)) % F(1.0))
        self.safety.check(r)
        return goal
