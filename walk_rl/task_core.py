"""Backend-independent Trex walking task: servo model, commands, gait clock, rewards,
terminations, observations, curricula and per-episode randomization.

Shared by both simulators so they run the *same* task:
  walk_rl/env.py    TrexWalkEnv  -- Isaac Sim core API directly (the original stack, fallback)
  walk_lab/env.py   TrexLabEnv   -- Isaac Lab DirectRLEnv

A backend subclass provides the simulator access through the `_sim_*` / `_dr_*` hooks
below and drives the step in this order (mirrors the original TrexWalkEnv.step):
  _pre_step(actions)
  for i in range(decimation): set _substep_dof_targets(i), _enforce_ankle_linkage(),
                               _apply_harness() if harness_alpha > 0, physics step
  reward, terminated, timeout, extras = _post_step()
  reset_idx(done envs); obs = _observations()

Nothing here imports a simulator. Tensors are torch on `self.device`; quaternions from
`_sim_link_state` are xyzw.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from walk_rl.config import (DEFAULT_TASK_PATH, _deep_update, _ns, apply_checkpoint_interface, joint_center_rad, joint_scale_rad,  # noqa: E402,F401
                             load_task_config, ns_to_dict, obs_dims)


def harness_alpha_at(cfg: SimpleNamespace, it: int) -> float:
    """Training harness strength at PPO iteration `it`: 1 until start_iter, linear to 0 at
    end_iter (shared by walk_rl's PPO and the env-side curriculum used with rsl_rl)."""
    h = getattr(cfg, "harness", None)
    if h is None or not h.enabled:
        return 0.0
    if it < h.start_iter:
        return 1.0
    if it >= h.end_iter:
        return 0.0
    return 1.0 - (it - h.start_iter) / float(h.end_iter - h.start_iter)


def push_scale_at(cfg: SimpleNamespace, it: int) -> float:
    """Push strength at iteration `it`: 0 until start_iter, linear to 1 at end_iter (pushes
    start after the harness has gone, so the policy first learns to walk unaided)."""
    c = getattr(cfg, "push", None)
    if c is None or not c.enabled:
        return 0.0
    if it < c.start_iter:
        return 0.0
    if it >= c.end_iter:
        return 1.0
    return (it - c.start_iter) / float(c.end_iter - c.start_iter)


def apply_weight_schedule(cfg: SimpleNamespace, it: int) -> None:
    """reward.weight_schedule {term: [[iteration, weight], ...]} (MicroDuck-style staged reward curriculum):
    from each listed iteration on, the term's weight is that value; before the first stage it keeps the
    configured weight. The task reads cfg.reward.weights every step, so setting it here takes effect at once."""
    sched = getattr(cfg.reward, "weight_schedule", None)
    if sched is None:
        return
    base = cfg.reward.__dict__.setdefault("_weights_base", dict(vars(cfg.reward.weights)))
    for term, stages in vars(sched).items():
        w = base.get(term, 0.0)
        for start, value in sorted(stages):
            if it >= start:
                w = value
        setattr(cfg.reward.weights, term, float(w))


def push_scale_final(cfg: SimpleNamespace) -> float:
    c = getattr(cfg, "push", None)
    return 1.0 if (c is not None and c.enabled) else 0.0


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vector `v` into the frame of quaternion `q` (xyzw)."""
    q_vec = q[..., :3]
    w = q[..., 3:4]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector `v` from the frame of quaternion `q` (xyzw) into the world frame."""
    q_vec = q[..., :3]
    w = q[..., 3:4]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


class TrexTaskCore:
    """Mixin. The backend sets `cfg`, `robot_cfg`, `num_envs`, `device` and calls
    `_setup_timing()` before building its scene, then `_init_buffers()`,
    `_randomize_startup()` and `reset_idx(all)`."""

    # ------------------------------------------------------- backend interface
    # state (all envs)
    def _sim_dof_names(self) -> list[str]: raise NotImplementedError
    def _sim_link_names(self) -> list[str]: raise NotImplementedError
    def _sim_dof_state(self) -> tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError  # q, qd (N, D)
    def _sim_link_state(self) -> tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError  # (N,L,7 pos+xyzw), (N,L,6 lin+ang world)
    def _sim_joint_forces(self) -> torch.Tensor: raise NotImplementedError  # (N, D) measured joint torque
    def _sim_contact_forces(self) -> torch.Tensor: raise NotImplementedError  # (N, 2+K) |F|: feet, then ground_links
    def _sim_total_mass(self) -> torch.Tensor: raise NotImplementedError  # (N,)
    # actuation
    def _sim_set_dof_targets(self, dof_tgt: torch.Tensor) -> None: raise NotImplementedError  # (N, D)
    def _sim_dof_positions_now(self) -> torch.Tensor: raise NotImplementedError  # (N, D), fresh within a substep
    def _sim_write_dof_positions(self, q: torch.Tensor) -> None: raise NotImplementedError  # (N, D), all envs
    def _sim_base_pose_vel_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: raise NotImplementedError  # link pos (N,L,3), base quat xyzw, base ang vel world
    def _sim_apply_base_wrench(self, force_w: torch.Tensor, torque_w: torch.Tensor, link_pos: torch.Tensor) -> None: raise NotImplementedError
    # root writes
    def _sim_root_velocities(self) -> torch.Tensor: raise NotImplementedError  # (N, 6) lin, ang world
    def _sim_set_root_velocities(self, ids: torch.Tensor, lin: torch.Tensor, ang: torch.Tensor) -> None: raise NotImplementedError
    def _sim_reset_state(self, ids, pos, quat_wxyz, q0) -> None: raise NotImplementedError  # pose, zero vel, dofs, targets
    # domain randomization primitives
    def _dr_set_friction(self, f: torch.Tensor) -> None: raise NotImplementedError  # (N, 1) static=dynamic
    def _dr_link_masses(self) -> torch.Tensor: raise NotImplementedError  # (N, L)
    def _dr_set_link_masses(self, m: torch.Tensor) -> None: raise NotImplementedError
    def _dr_link_inertias(self) -> torch.Tensor: raise NotImplementedError  # (N, L, 9)
    def _dr_set_link_inertias(self, i: torch.Tensor) -> None: raise NotImplementedError
    def _dr_base_com(self) -> torch.Tensor: raise NotImplementedError  # (N, 1, 3)
    def _dr_set_base_com(self, p: torch.Tensor) -> None: raise NotImplementedError
    def _dr_dof_gains(self) -> tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError
    def _dr_set_dof_gains(self, kp: torch.Tensor, kd: torch.Tensor) -> None: raise NotImplementedError
    def _dr_max_efforts(self) -> torch.Tensor: raise NotImplementedError  # (N, D)
    def _dr_set_max_efforts(self, mf: torch.Tensor) -> None: raise NotImplementedError
    def _sim_set_dof_efforts(self, e: torch.Tensor) -> None: raise NotImplementedError  # (N, D) feed-forward joint efforts
    # actuator.model lx16a only
    def _sim_dof_velocities_now(self) -> torch.Tensor: raise NotImplementedError  # (N, D), fresh within a substep
    def _dr_set_armatures(self, a: torch.Tensor) -> None: raise NotImplementedError  # (N, D) kg m^2

    # ------------------------------------------------------------------ timing
    def _setup_timing(self) -> None:
        c = self.cfg
        self.physics_dt = 1.0 / c.sim.physics_hz
        assert c.sim.physics_hz % c.sim.control_hz == 0, "control_hz must divide physics_hz"
        self.decimation = c.sim.physics_hz // c.sim.control_hz
        self.dt = self.physics_dt * self.decimation
        self.max_episode_steps = int(round(c.sim.episode_length_s / self.dt))
        self.ground_links = list(c.termination.ground_contact_links)

    # ---------------------------------------------------------------- buffers
    def _init_buffers(self) -> None:
        N, dev, rc, c = self.num_envs, self.device, self.robot_cfg, self.cfg
        dn, ln = self._sim_dof_names(), self._sim_link_names()
        self.dof_names, self.link_names = list(dn), list(ln)
        self.act_names = list(rc.actuated_joints)
        self.act_idx = torch.tensor([dn.index(n) for n in self.act_names], device=dev)
        pick = lambda names: torch.tensor([self.act_names.index(n) for n in names], device=dev)
        self.hip_a = pick(["left_hip_joint", "right_hip_joint"])
        self.num_actions = len(self.act_names)
        side_idx = lambda s, j: dn.index(f"{s}_{j}_joint")
        self.fem_idx = torch.tensor([side_idx(s, "femur") for s in ("left", "right")], device=dev)
        self.tib_idx = torch.tensor([side_idx(s, "tibia") for s in ("left", "right")], device=dev)
        self.ank_idx = torch.tensor([side_idx(s, "ankle") for s in ("left", "right")], device=dev)
        # dof indices (full 10-dof vector) -> position within the 8-action vector
        a_of = {int(d): i for i, d in enumerate(self.act_idx.tolist())}
        self.fem_a = torch.tensor([a_of[int(i)] for i in self.fem_idx.tolist()], device=dev)
        self.tib_a = torch.tensor([a_of[int(i)] for i in self.tib_idx.tolist()], device=dev)
        self.ankle_routed = rc.ankle_sim_model == "teleport_routed"
        self.linkage_effort = torch.zeros(N, len(dn), device=dev)  # routed ankle torque (teleport_routed)
        self._just_reset = torch.ones(N, dtype=torch.bool, device=dev)
        self.vy_ema = torch.zeros(N, device=dev)  # lateral_drift: 1 s average of heading-frame v_y
        self.leg_asym_ema = torch.zeros(N, 2, device=dev)  # leg_symmetry: 1 s average of left - right femur/tibia
        self.sole_pos = self.sole_pos_prev = torch.zeros(N, 2, 3, device=dev)  # foot_slip_point: sole
        self.sole_contact_prev = torch.zeros(N, 2, dtype=torch.bool, device=dev)

        lim = torch.tensor(joint_scale_rad(c, rc, self.act_names), device=dev)
        self.joint_limit = lim
        rng = [rc.joint_range_rad(n) for n in self.act_names]  # physical (possibly asymmetric) range
        self.joint_lo = torch.tensor([r[0] for r in rng], device=dev)
        self.joint_hi = torch.tensor([r[1] for r in rng], device=dev)
        self.ankle_couple = rc.ankle_coupled_limit_rad()
        self.act_scale = lim * c.actions.scale_frac_of_limit
        self.default_q = torch.zeros(self.num_actions, device=dev)  # CAD neutral pose
        self.act_center = torch.tensor(joint_center_rad(c, self.act_names, rc), device=dev)
        # smoothed action whose target is default_q: episodes start from the neutral-pose goal
        self.smooth0 = (self.default_q - self.act_center) / self.act_scale
        self.servo_res = math.radians(rc.actuator_spec["position_accuracy_deg"])
        self.max_target_step = rc.actuator_no_load_speed_rad_per_sec * self.dt
        self.tau_max = rc.actuator_stall_torque_nm
        if getattr(getattr(c, "actuator", None), "model", "implicit_pd") == "lx16a":
            # the fitted servo's holding torque (actuator_model.stall_torque_scale, 1 before 2026-10-20):
            # torque / torque_sat / saturation are relative to what the modelled servo can give
            self.tau_max = self.tau_max * rc.raw["actuator_model"].get("stall_torque_scale", 1.0)

        self.base_i = ln.index("base_link")
        self.head_i = ln.index("head")
        self.foot_i = torch.tensor([ln.index("left_foot"), ln.index("right_foot")], device=dev)
        # reward.foot_slip_point: sole -> slip measured at the sole centre (robot_config foot_sole)
        self.sole_r = (torch.tensor(self.robot_cfg.foot_sole_center_m(), device=dev)
                       if getattr(self.cfg.reward, "foot_slip_point", "origin") == "sole" else None)
        # foot.clearance_mode: sole -> swing clearance = lowest sole corner above the ground (robot_config
        # foot_sole.corners_in_foot_m); relative (default, v1-n25) -> one foot origin above the other
        self.sole_corners = (torch.tensor(self.robot_cfg.raw["foot_sole"]["corners_in_foot_m"], device=dev)
                             if getattr(self.cfg.foot, "clearance_mode", "relative") == "sole" else None)
        self.sole_min_z = torch.zeros(N, 2, device=dev)

        self.num_dofs = len(dn)
        self.all_idx = torch.arange(N, device=dev)
        self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3)

        z = lambda *s: torch.zeros(*s, device=dev)
        self.cmd_target = z(N, self.num_actions)  # servo goal after rate-limit (8)
        self.prev_target = z(N, self.num_actions)
        self.last_action = z(N, self.num_actions)  # policy output at t-1
        self.prev_action = z(N, self.num_actions)  # for action-rate penalty
        self.prev2_action = z(N, self.num_actions)  # a(t-2), for the action-acceleration (chatter) penalty
        self.smoothed_action = z(N, self.num_actions)  # low-pass filtered command (servo-input filter)
        self.air_time = z(N, 2)  # seconds each foot has been off the ground
        self.harness_alpha = 0.0  # training-only support strength in [0,1]; set by the trainer
        self.total_mass = self._sim_total_mass()  # (N,) kg, incl. DR
        self.link_count = len(ln)
        self.contact_prev = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self.foot_force = z(N, 2)
        self.ground_force = z(N, len(self.ground_links))
        # Contact views keep the LAST reported force after a teleport (no "contact lost" event
        # fires on a reset; natural lift-off does clear). Measured: tail link 25 cm up, force
        # stuck at the impact value for 25+ steps -> terminate-on-contact looped at ep_len 1.
        # So at reset record each sensor's reading and ignore it until it changes.
        K = 2 + len(self.ground_links)
        self._stale_ref = z(N, K)
        self._armed = torch.ones(N, K, dtype=torch.bool, device=dev)
        self.start_yaw = z(N)
        self.q_meas_prev = z(N, self.num_actions)
        self.delay = z(N)  # per-env bus latency in control steps (0/1)
        self.cmd_vx = z(N)
        self.phase = z(N)
        self.cmd_timer = z(N)
        self.episode_len = torch.zeros(N, dtype=torch.long, device=dev)
        self.reset_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self.start_x = z(N)

        # filled by _read_state()
        self.q = self.qd = self.root_q = self.lin_vel_b = self.ang_vel_b = None
        self.gravity_b = self.link_z = self.foot_z = self.foot_vel_xy = None
        self.torque = None

        # v8 additions (each one off/neutral when its config key is absent)
        ob = self.cfg.observation
        self.use_heading = bool(getattr(getattr(ob, "heading", None), "enabled", False))
        self.history_len = int(getattr(ob, "history_len", 1))
        self.yaw_ref = z(N)  # heading reference latched at reset (the Pi latches it when walking starts)
        self.yaw_bias = z(N)  # IMU yaw drift accumulated since the latch
        self.yaw_drift_rate = z(N)  # rad/s, per episode
        self.imu_delay = z(N)  # per-episode IMU latency in control steps (observation.imu_delay_steps_range)
        self.imu_tilt = torch.zeros(N, 2, device=self.device)  # per-episode IMU mount offset (roll, pitch) rad
        self.imu_prev = None  # the last K steps' (gravity, gyro, heading-error) readings, for imu_delay
        self.enc_offset = z(N, self.num_actions)  # per-episode servo calibration offset (readings only)
        self.root_lin_vel_w = z(N, 3)
        self.prev_root_lin_vel_w = z(N, 3)
        self.feet_y_neutral = None  # lateral foot spacing in the spawn pose, measured at first reset
        self.base_h_neutral = None  # _base_height() in the spawn pose, measured at first reset
        self.push_scale = 0.0  # training-only disturbance strength in [0,1]; set by the trainer
        self.push_timer = z(N)
        self.push_count = 0

        # actuator.model: implicit_pd (engine position drive) or lx16a (walk_rl/actuator.py)
        self.lx16a = getattr(getattr(c, "actuator", None), "model", "implicit_pd") == "lx16a"
        if self.lx16a:
            from walk_rl.actuator import LX16AParams

            p = self.lx_nominal = LX16AParams.from_robot_config(rc)
            A = self.num_actions
            full = lambda v: torch.full((N, A), float(v), device=dev)
            self.act_kp, self.act_db, self.act_fc = full(p.kp), full(p.deadband), full(p.coulomb)
            self.act_arm = full(p.armature)
            self.act_stall = full(p.stall * p.v_nominal / p.v_ref)
            self.act_noload = full(p.no_load * p.v_nominal / p.v_ref)
            self.act_lat = torch.full((N, 1), float(p.latency), device=dev)
            self.bus_voltage = torch.full((N, 1), float(p.v_nominal), device=dev)
            self.act_torque = z(N, A)  # servo torque of the last physics substep (rewards, critic)
            ac = c.actuator
            self.supply_sag = bool(getattr(ac, "supply_sag", False))
            # drive limit (N, A) at the set voltage, or None: actuator.drive_torque_limit
            self.act_drive = full(p.drive * p.v_nominal / p.v_ref) if getattr(ac, "drive_torque_limit", False) else None
        self._substep_i = 0

        self.noise = self.cfg.observation.noise
        self.use_clock = c.gait.use_phase_clock
        # critic: one clean frame + privileged state
        self.frame_dim, self.num_actor_obs, self.num_critic_obs = obs_dims(c)
        self.obs_hist = z(N, self.history_len, self.frame_dim)
        self._hist_fresh = torch.ones(N, dtype=torch.bool, device=dev)  # fill history with the first frame
        self.ep_sums: dict[str, torch.Tensor] = {}
        self.finished_episodes: list[dict] = []

    # ------------------------------------------------------ domain randomization
    def _randomize_startup(self) -> None:
        dr, N, dev = self.cfg.domain_randomization, self.num_envs, self.device
        self.dr_active = {}
        lo_hi = lambda r: torch.empty(N, 1, device=dev).uniform_(r[0], r[1])
        self.dr_active["action_delay"] = True  # resampled per episode in reset_idx()
        if self.lx16a:
            self._setup_lx16a(dr)
        if not dr.enabled:
            return

        def attempt(name, fn):
            try:
                fn()
                self.dr_active[name] = True
            except Exception as e:  # report, don't hide: a skipped DR is a real gap
                self.dr_active[name] = f"SKIPPED: {type(e).__name__}: {str(e)[:80]}"

        def friction():
            self._dr_set_friction(lo_hi(dr.friction_range))

        def mass():
            m = self._dr_link_masses().clone()  # (N, L)
            scale = torch.ones_like(m)
            scale[:, self.base_i] = lo_hi(dr.base_mass_scale_range).squeeze(-1)
            limb = getattr(dr, "limb_mass_scale_range", None)  # head/tail: printed shells vary
            if limb is not None:
                for name in ("head", "tail"):
                    scale[:, self.link_names.index(name)] = lo_hi(limb).squeeze(-1)
            self._dr_set_link_masses(m * scale)
            inertia_range = getattr(dr, "base_inertia_scale_range", None)
            if self.robot_cfg.base_link_inertia != "urdf" or inertia_range is not None:
                # same mass distribution, heavier/lighter body: the inertia scales with the mass;
                # base_inertia_scale_range adds an independent factor (battery placement unknown)
                if inertia_range is not None:
                    scale[:, self.base_i] *= lo_hi(inertia_range).squeeze(-1)
                inertia = self._dr_link_inertias().clone()  # (N, L, 9)
                self._dr_set_link_inertias(inertia * scale.unsqueeze(-1))

        def base_com():
            # base_link lumps battery + Pi + wiring whose placement is not measured
            p = self._dr_base_com().clone()  # (N, 1, 3)
            r = dr.base_com_offset_m
            p += (torch.rand_like(p) * 2 - 1) * r
            self._dr_set_base_com(p)

        def gains():
            s = lo_hi(dr.pd_gain_scale_range)
            kp, kd = self._dr_dof_gains()
            self._dr_set_dof_gains(kp * s, kd * s)

        def torque_cap():
            s = lo_hi(dr.torque_cap_scale_range)
            mf = self._dr_max_efforts().clone()
            mf[:, self.act_idx] *= s
            self._dr_set_max_efforts(mf)

        attempt("friction", friction)
        attempt("base_mass", mass)
        if getattr(dr, "base_com_offset_m", 0.0):
            attempt("base_com", base_com)
        if not self.lx16a:  # lx16a randomizes its own gain, torque and speed (domain_randomization.actuator_model)
            attempt("pd_gains", gains)
            attempt("torque_cap", torque_cap)
        self.total_mass = self._sim_total_mass()  # after mass DR (harness weight support)

    # ---------------------------------------------------------- LX-16A actuator
    def _lx16a_dr(self, dr):
        am = getattr(dr, "actuator_model", None) if dr.enabled else None
        return am

    def _setup_lx16a(self, dr) -> None:
        """Engine drives off on the actuated joints (the servo torque is explicit), effort limit
        above what the model can command (braking can reach 2x stall), armature per env and joint.
        Ankles keep their implicit drive (the passive linkage)."""
        kp, kd = self._dr_dof_gains()
        kp, kd = kp.clone(), kd.clone()
        kp[:, self.act_idx] = 0.0
        kd[:, self.act_idx] = 0.0
        self._dr_set_dof_gains(kp, kd)
        mf = self._dr_max_efforts().clone()
        mf[:, self.act_idx] = 3.0 * self.robot_cfg.actuator_stall_torque_nm
        self._dr_set_max_efforts(mf)
        am = self._lx16a_dr(dr)
        if am is not None:
            self.act_arm = self.act_arm * torch.empty_like(self.act_arm).uniform_(*am.armature_scale_range)
        arm = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        arm[:, self.act_idx] = self.act_arm
        self._dr_set_armatures(arm)
        self.dr_active["actuator_lx16a"] = True if am is None else "randomized"

    def _sample_lx16a(self, ids: torch.Tensor) -> None:
        """Per episode: each servo's gain, deadband, friction, speed and torque; the pack voltage
        and bus latency per robot (armature is set once at startup)."""
        am = self._lx16a_dr(self.cfg.domain_randomization)
        if am is None:
            return
        p, n, A, dev = self.lx_nominal, len(ids), self.num_actions, self.device
        u = lambda r, *shape: torch.empty(*shape, device=dev).uniform_(r[0], r[1])
        self.bus_voltage[ids] = u(am.bus_voltage_v_range, n, 1)
        v = self.bus_voltage[ids] / p.v_ref
        self.act_kp[ids] = p.kp * u(am.kp_scale_range, n, A)
        self.act_db[ids] = torch.deg2rad(u(am.deadband_deg_range, n, A))
        self.act_fc[ids] = u(am.coulomb_friction_nm_range, n, A)
        self.act_stall[ids] = self.robot_cfg.actuator_stall_torque_nm * u(am.stall_torque_scale_range, n, A) * v
        self.act_noload[ids] = self.robot_cfg.actuator_no_load_speed_rad_per_sec * u(am.no_load_speed_scale_range, n, A) * v
        self.act_lat[ids] = u(am.latency_s_range, n, 1)
        if self.act_drive is not None:
            ds = getattr(am, "drive_torque_scale_range", None)
            self.act_drive[ids] = (self.robot_cfg.actuator_stall_torque_nm * u(ds, n, A) if ds else p.drive) * v

    def _lx16a_torque(self) -> torch.Tensor:
        """(N, A) servo torque for the current substep from fresh joint state."""
        from walk_rl.actuator import lx16a_torque, ramp_reference, supply_ratio

        q = self._sim_dof_positions_now()[:, self.act_idx]
        qd = self._sim_dof_velocities_now()[:, self.act_idx]
        t = (self._substep_i + 1) * self.physics_dt
        q_ref = ramp_reference(self._ramp_start, self._ramp_goal, t, self.dt, self.act_lat, torch.clamp)
        stall, noload, drive = self.act_stall, self.act_noload, None
        if self.act_drive is not None:
            drive = self.act_drive
        if self.supply_sag:  # actuator.supply_sag: the bus voltage drops with last substep's total servo torque
            r = supply_ratio(self.bus_voltage, self.act_torque.abs().sum(dim=1, keepdim=True), self.lx_nominal.sag, torch)
            stall, noload = stall * r, noload * r
            drive = None if drive is None else drive * r
        tau = lx16a_torque(q_ref, q, qd, self.act_kp, self.act_db, self.act_fc, stall, noload,
                           self.act_arm, self.physics_dt, torch, drive_max=drive)
        self.act_torque = tau
        return tau

    # ----------------------------------------------------------------- reading
    def _read_state(self) -> None:
        self.q, self.qd = self._sim_dof_state()
        lt, lv = self._sim_link_state()  # (N, L, 7) pos + quat xyzw; (N, L, 6) lin, ang (world)
        self.root_q = lt[:, self.base_i, 3:7]
        self.root_pos = lt[:, self.base_i, 0:3]
        self.root_lin_vel_w = lv[:, self.base_i, 0:3]
        self.foot_pos = lt[:, self.foot_i, 0:3]
        self.foot_quat = lt[:, self.foot_i, 3:7]  # xyzw
        self.lin_vel_b = quat_rotate_inverse(self.root_q, lv[:, self.base_i, 0:3])
        self.ang_vel_b = quat_rotate_inverse(self.root_q, lv[:, self.base_i, 3:6])
        self.gravity_b = quat_rotate_inverse(self.root_q, self.gravity_w)
        self.foot_z = lt[:, self.foot_i, 2]
        self.head_z = lt[:, self.head_i, 2]
        self.foot_vel_xy = lv[:, self.foot_i, 0:2]
        if self.sole_r is not None:  # sole centre position (world)
            r_w = quat_rotate(self.foot_quat.reshape(-1, 4), self.sole_r.expand(self.num_envs * 2, 3)).reshape(-1, 2, 3)
            self.sole_pos = self.foot_pos + r_w
        if self.sole_corners is not None:  # (N, 2) lowest sole corner height (flat ground at z = 0)
            K = self.sole_corners.shape[0]
            qk = self.foot_quat.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, 4)
            c = quat_rotate(qk, self.sole_corners.repeat(self.num_envs * 2, 1)).reshape(-1, 2, K, 3)
            self.sole_min_z = (self.foot_pos[..., 2:3] + c[..., 2]).min(dim=2).values
        self.torque = self._sim_joint_forces()[:, self.act_idx]
        if self.ankle_routed:  # PhysX reports drive + applied effort; the servo's share is the drive
            self.torque = self.torque - self.linkage_effort[:, self.act_idx]
        if self.lx16a:  # the model's own output: exact, and identical on every backend
            self.torque = self.act_torque.clone()
        raw = self._sim_contact_forces()
        self._armed |= raw != self._stale_ref  # a changed reading is a fresh one
        eff = torch.where(self._armed, raw, torch.zeros_like(raw))
        self._raw_force = raw
        self.foot_force, self.ground_force = eff[:, :2], eff[:, 2:]

    def _contacts(self) -> torch.Tensor:
        """(N, 2) bool foot contact from PhysX contact forces. A foot-link height threshold is
        wrong here: the 4-bar linkage keeps each foot parallel to the base, so any body lean
        tilts the feet and shifts the link origin height by ~cm (measured: both feet carrying
        6-7 N each were flagged 'no contact')."""
        return self.foot_force > self.cfg.foot.contact_force_threshold_n

    def _sole_slip(self, contact: torch.Tensor) -> torch.Tensor:
        """(N,) sum over feet in contact now and one step ago of the squared sole-centre speed over that
        control step. Finite difference: the instantaneous PhysX foot velocity is not the foot's motion,
        because the ankle linkage moves the foot directly each substep (v9k: 36 mm/s error in contact)."""
        v = (self.sole_pos - self.sole_pos_prev)[..., 0:2] / self.dt
        both = contact & self.sole_contact_prev
        self.sole_pos_prev, self.sole_contact_prev = self.sole_pos.clone(), contact.clone()
        return (both.float() * (v ** 2).sum(-1)).sum(dim=1)

    def _base_height(self) -> torch.Tensor:
        """(N,) base origin height minus the lower foot origin height (both link frames; the base
        origin sits below the foot origin, so the value is negative and only differences count)."""
        return self.root_pos[:, 2] - self.foot_z.min(dim=1).values

    def _feet_lateral_spacing(self) -> torch.Tensor:
        """(N,) left-right foot distance across the heading direction (yaw-only frame, so
        body roll during weight shift does not change it)."""
        d = self.foot_pos[:, 0, :2] - self.foot_pos[:, 1, :2]
        yaw = self._yaw()
        return (-yaw.sin() * d[:, 0] + yaw.cos() * d[:, 1]).abs()

    def _foot_clearance(self) -> torch.Tensor:
        """(N, 2) relative mode: height of each foot origin above the other. Roll changes it (feet 0.147 m
        apart: 4 deg = 10 mm), so v9t could earn it by leaning. sole mode: lowest sole corner above the ground."""
        if self.sole_corners is not None:
            return self.sole_min_z
        return self.foot_z - self.foot_z.flip(dims=[1])

    # -------------------------------------------------------------------- step
    def _pre_step(self, actions: torch.Tensor) -> None:
        """Action processing before the physics substeps: bus delay, low-pass, quantization,
        speed-limited ramp target, training pushes."""
        a = actions.clamp(-1.0, 1.0)
        # bus latency: the servos act on the action from `delay` control steps ago (0..2)
        hist = torch.stack([a, self.last_action, self.prev_action], dim=1)  # (N, 3, A)
        applied = hist[self.all_idx, self.delay.long().clamp(0, 2)]
        self.prev2_action = self.prev_action.clone()
        self.prev_action = self.last_action.clone()
        self.last_action = a

        # First-order low-pass on the servo command (robot-side filter, runs the same in the
        # Pi loop): structurally removes Nyquist-rate chatter without penalizing the
        # policy's exploration noise the way an action-acceleration penalty alone does.
        alpha = self.cfg.actions.smoothing_alpha
        self.smoothed_action = alpha * self.smoothed_action + (1.0 - alpha) * applied
        raw = self.act_center + self.smoothed_action * self.act_scale
        if self.cfg.actions.quantize_to_servo_resolution:
            raw = torch.round(raw / self.servo_res) * self.servo_res
        # Servo ramps to the goal over the control period (move-time command),
        # at no more than its no-load speed.
        start = self.cmd_target
        goal = start + (raw - start).clamp(-self.max_target_step, self.max_target_step)
        goal = self._limit_goal(goal)
        self._ramp_start, self._ramp_goal = start, goal

        if self.push_scale > 0.0:
            self._apply_push()
        self.prev_root_lin_vel_w = self.root_lin_vel_w.clone()
        self._dof_tgt = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

    def _limit_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Joint ranges, then the coupled ankle limit (lo <= femur - tibia <= hi) by moving the
        tibia, then the femur if the tibia hit its own range. pi_runner/runner.py does the same."""
        goal = torch.max(torch.min(goal, self.joint_hi), self.joint_lo)
        if self.ankle_couple is None:
            return goal
        lo, hi = self.ankle_couple
        f, t = goal[:, self.fem_a], goal[:, self.tib_a]
        t = torch.max(torch.min(t, f - lo), f - hi)
        t = torch.max(torch.min(t, self.joint_hi[self.tib_a]), self.joint_lo[self.tib_a])
        f = torch.max(torch.min(f, t + hi), t + lo)
        goal = goal.clone()
        goal[:, self.fem_a], goal[:, self.tib_a] = f, t
        return goal

    def _substep_dof_targets(self, i: int) -> torch.Tensor:
        """(N, D) PD targets for physics substep i: linear ramp start->goal; ankle targets
        aligned with the linkage."""
        self._substep_i = i
        start, goal = self._ramp_start, self._ramp_goal
        tgt = start + (goal - start) * ((i + 1) / self.decimation)
        self._dof_tgt[:, self.act_idx] = tgt
        self._dof_tgt[:, self.ank_idx] = tgt[:, self.fem_a] - tgt[:, self.tib_a]
        return self._dof_tgt

    def _post_step(self):
        """After the physics substeps: state, commands, reward, terminations, episode stats.
        Returns (reward, terminated, timeout, extras); the caller resets `done` envs and then
        builds observations."""
        self.cmd_target = self._ramp_goal
        self.episode_len += 1
        self._read_state()
        self._nstep = getattr(self, "_nstep", 0) + 1
        if self.robot_cfg.ankle_sim_model == "constraint" and self._nstep % 10 == 0:
            # Newton holds the ankle linkage with a soft tendon limit: log how far it stretches (audit A4)
            r = (self.q[:, self.ank_idx] - (self.q[:, self.fem_idx] - self.q[:, self.tib_idx])).abs().rad2deg()
            self._ankle_res = getattr(self, "_ankle_res", []) + [(torch.quantile(r.flatten(), 0.95).item(), r.max().item())]
        self._advance_commands_and_phase()
        reward, terms, terminated = self._compute_reward()
        timeout = self.episode_len >= self.max_episode_steps
        nonfinite = ~torch.isfinite(self.q).all(dim=1) | ~torch.isfinite(self.root_pos).all(dim=1)
        terminated = terminated | nonfinite
        reward = torch.nan_to_num(reward)
        self._accumulate_episode_stats(terms, terminated | timeout, terminated)
        extras = {"time_outs": timeout & ~terminated}
        return reward, terminated, timeout, extras

    def set_harness(self, alpha: float) -> None:
        self.harness_alpha = float(max(0.0, min(1.0, alpha)))

    def set_push_scale(self, s: float) -> None:
        self.push_scale = float(max(0.0, min(1.0, s)))

    def _apply_push(self) -> None:
        """Random disturbance (BD-X / Open Duck / humanoid-gym style): every few seconds each
        env gets an instantaneous change of base velocity, xy linear + roll/pitch/yaw angular,
        uniform up to push_scale * max. Training-only; evaluate.py has its own fixed pushes."""
        p = self.cfg.push
        self.push_timer -= self.dt
        ids = (self.push_timer <= 0).nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        n, dev = len(ids), self.device
        self.push_timer[ids] = torch.empty(n, device=dev).uniform_(*p.interval_s)
        dv = torch.zeros(n, 3, device=dev)
        dv[:, :2] = (torch.rand(n, 2, device=dev) * 2 - 1) * p.max_lin_vel_mps * self.push_scale
        dw = (torch.rand(n, 3, device=dev) * 2 - 1) * p.max_ang_vel_rad_s * self.push_scale
        self.kick(ids, dv, dw)
        self.push_count += n

    def kick(self, ids: torch.Tensor, dv_w: torch.Tensor, dw_w: torch.Tensor) -> None:
        """Add (dv_w, dw_w) [world frame] to the root velocity of envs `ids`."""
        rv = self._sim_root_velocities().clone()  # (N, 6)
        lin = rv[ids, 0:3] + dv_w
        ang = rv[ids, 3:6] + dw_w
        self._sim_set_root_velocities(ids, lin, ang)
        self.root_lin_vel_w = self.root_lin_vel_w.clone()
        self.root_lin_vel_w[ids] += dv_w  # the push itself is not "base acceleration" the policy caused

    def _apply_harness(self) -> None:
        """Training-only assist (curriculum, annealed to 0 by the trainer): an upward force
        on the base (fraction of body weight) plus a righting torque on roll/pitch. It makes
        lifting a foot safe so stepping can be *discovered*; with the tail forbidden as a
        prop, v6 never lifted a foot unaided (air time 0.000 after 200 iterations).
        Applied each substep (PhysX clears external forces after every step)."""
        h = self.cfg.harness
        link_pos, q, w_w = self._sim_base_pose_vel_now()
        g_b = quat_rotate_inverse(q, self.gravity_w)  # gravity in body frame
        up_b = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(g_b)
        # Restoring rotation (body frame) is t x g_b: for nose-down pitch th, g_b=(sin th,0,-cos th),
        # t x g_b = (0,-sin th,0) = nose up. (g_b x t had the wrong sign: positive feedback.)
        err_b = torch.cross(up_b, g_b, dim=-1)
        w_b = quat_rotate_inverse(q, w_w)
        torque_b = h.torque_kp * err_b - h.torque_kd * w_b
        torque_b[:, 2] = 0.0  # never fight yaw
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=1)
        torque_w = quat_rotate_inverse(q_conj, torque_b)  # body -> world
        a = self.harness_alpha
        force = torch.zeros(self.num_envs, 3, device=self.device)
        force[:, 2] = a * h.weight_fraction * self.total_mass * 9.81
        self._sim_apply_base_wrench(force, a * torque_w, link_pos)

    def _enforce_ankle_linkage(self) -> None:
        """ankle = femur - tibia (robot_config ankle_linkage), teleported each substep.

        sim_model teleport_routed: the ankle drive stands in for the parallelogram, so its
        torque tau_a (last physics step) is also applied as the linkage's generalized force on
        the rest of the leg -- femur -tau_a, tibia +tau_a (the coefficients of
        -femur + tibia + ankle = 0). The femur/tibia servos then no longer hold the foot's
        ground moment; base_link does, as on the robot."""
        if self.robot_cfg.ankle_sim_model == "constraint":  # walk_lab3: the engine holds the linkage (tendon limit)
            if self.lx16a:
                e = torch.zeros_like(self.linkage_effort)
                e[:, self.act_idx] += self._lx16a_torque()
                self._sim_set_dof_efforts(e)
            return
        q = self._sim_dof_positions_now().clone()
        q[:, self.ank_idx] = q[:, self.fem_idx] - q[:, self.tib_idx]
        self._sim_write_dof_positions(q)
        if self.ankle_routed:
            # right after a reset the reading is from before the reset (stale on some backends)
            tau_a = self._sim_joint_forces()[:, self.ank_idx] * (~self._just_reset).unsqueeze(1)
            self._just_reset[:] = False
            e = self.linkage_effort
            e[:, self.fem_idx] = -tau_a
            e[:, self.tib_idx] = tau_a
            if not self.lx16a:
                self._sim_set_dof_efforts(e)
        if self.lx16a:  # servo torque + routed linkage torque, one effort write per substep
            e = self.linkage_effort.clone() if self.ankle_routed else torch.zeros_like(self.linkage_effort)
            e[:, self.act_idx] += self._lx16a_torque()
            self._sim_set_dof_efforts(e)

    # -------------------------------------------------------------- commands/gait
    def _advance_commands_and_phase(self) -> None:
        self.phase = (self.phase + self.gait_frequency() * self.dt) % 1.0
        if self.use_heading:
            self.yaw_bias += self.yaw_drift_rate * self.dt
        self.cmd_timer -= self.dt
        resample = (self.cmd_timer <= 0).nonzero(as_tuple=False).squeeze(-1)
        if len(resample) > 0:
            self._sample_commands(resample)

    def gait_frequency(self) -> torch.Tensor:
        """(N,) clock frequency [Hz]. Optionally rises with the command so faster walking takes
        more steps instead of only longer strides (gait.frequency_per_mps; 0 = fixed). The Pi
        runner computes the same function of the command."""
        g = self.cfg.gait
        k = getattr(g, "frequency_per_mps", 0.0)
        f = torch.full_like(self.cmd_vx, g.frequency_hz)
        if k:
            f = f + k * (self.cmd_vx - g.frequency_ref_mps)
            f = f.clamp(*g.frequency_range_hz)
        return f

    def _sample_commands(self, ids: torch.Tensor) -> None:
        lo, hi = self.cfg.command.vx_range_mps
        vx = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
        slow_p = getattr(self.cfg.command, "slow_prob", 0.0)
        if slow_p > 0:  # v9d: oversample slow walking (the slow servo made v9c round it down to standing)
            slo, shi = self.cfg.command.slow_range_mps
            slow = torch.empty(len(ids), device=self.device).uniform_(slo, shi)
            vx = torch.where(torch.rand(len(ids), device=self.device) < slow_p, slow, vx)
        stand = torch.rand(len(ids), device=self.device) < self.cfg.command.standing_prob
        self.cmd_vx[ids] = torch.where(stand, torch.zeros_like(vx), vx)
        self.cmd_timer[ids] = self.cfg.command.resample_time_s

    def _joint_limit_term(self, q_act, rcfg):
        m = getattr(rcfg, "joint_limit_range_margin_deg", None)
        if m is None:  # symmetric: beyond joint_limit_margin of +-joint_limit
            return torch.relu((q_act - self.act_center).abs() / self.joint_limit - rcfg.joint_limit_margin).sum(dim=1)
        # v9s: inside the last m deg of the physical range (joint_lo..joint_hi), 0 at m deg, 1 at the end.
        # The symmetric form never sees an asymmetric end (hips +7 deg inward vs a 30 deg limit).
        m = math.radians(m)
        return (torch.relu(q_act - (self.joint_hi - m)) + torch.relu((self.joint_lo + m) - q_act)).sum(dim=1) / m

    def _walking(self) -> torch.Tensor:
        return self.cmd_vx > self.cfg.gait.walking_cmd_threshold_mps

    def _desired_stance(self) -> torch.Tensor:
        """(N, 2) bool: should each foot be on the ground per the gait clock?"""
        ph = torch.stack([self.phase, (self.phase + 0.5) % 1.0], dim=1)
        stance = ph < self.cfg.gait.stance_ratio
        return stance | ~self._walking().unsqueeze(1)

    # ------------------------------------------------------------------ reward
    def _compute_reward(self):
        w, rcfg = self.cfg.reward.weights, self.cfg.reward
        g = self.gravity_b
        walking = self._walking().float()
        contact = self._contacts()
        stance = self._desired_stance()
        q_act = self.q[:, self.act_idx]
        foot_h = self._foot_clearance()
        # feet air time: reward the touchdown that ends a real swing (alternating steps).
        # prev_air still holds the swing duration at the touchdown step; reset it after.
        prev_air = self.air_time
        first_contact = contact & ~self.contact_prev
        air_reward = (prev_air.clamp(min=rcfg.air_time_min_s, max=rcfg.air_time_max_s) - rcfg.air_time_min_s) * first_contact.float()
        self.air_time = torch.where(contact, torch.zeros_like(prev_air), prev_air + self.dt)
        self.contact_prev = contact.clone()

        sigma = rcfg.track_vx_sigma
        rel = getattr(rcfg, "track_vx_sigma_rel", 0.0)
        if rel > 0:  # v9d: tolerance proportional to the command, so standing at 0.05 m/s earns ~nothing
            sigma = (rel * self.cmd_vx).clamp(min=rcfg.track_vx_sigma_min, max=rcfg.track_vx_sigma)
        match = contact == stance
        if getattr(rcfg, "gait_contact_mode", "per_foot") == "both":  # v9d: both feet planted no longer half-matches
            gait_contact = match.all(dim=1).float()
        else:
            gait_contact = match.float().mean(dim=1)
        terms = {
            "track_vx": torch.exp(-((self.lin_vel_b[:, 0] - self.cmd_vx) ** 2) / sigma ** 2),
            "alive": torch.ones(self.num_envs, device=self.device),
            "upright": (g[:, 0] ** 2 + g[:, 1] ** 2),
            "lateral_vel": self.lin_vel_b[:, 1] ** 2,
            "yaw_rate": self.ang_vel_b[:, 2] ** 2,
            # Heading is not observable to the policy (gyro only), so hold a straight
            # line by tracking zero yaw rate tightly rather than penalizing heading.
            "track_yaw": torch.exp(-(self.ang_vel_b[:, 2] ** 2) / rcfg.yaw_sigma_rad_s ** 2),
            "ang_vel_xy": self.ang_vel_b[:, 0] ** 2 + self.ang_vel_b[:, 1] ** 2,
            "gait_contact": gait_contact,
            "air_time": air_reward.sum(dim=1) * walking,
            "swing_clearance": ((~stance).float() * (foot_h / self.cfg.foot.swing_clearance_target_m).clamp(0, 1)).mean(dim=1) * walking,
            "foot_slip": (contact.float() * (self.foot_vel_xy ** 2).sum(-1)).sum(dim=1) if self.sole_r is None else self._sole_slip(contact),
            # swing-phase foot touching the floor (light contact counts): scuffing / dragging the swing foot
            "swing_scuff": ((~stance) & (self.foot_force > getattr(rcfg, "swing_scuff_force_n", 0.2))).float().sum(dim=1) * walking,
            "action_rate": ((self.last_action - self.prev_action) ** 2).sum(dim=1),
            # second difference: ~0 for a smooth gait, large for alternating-sign chatter
            "action_accel": ((self.last_action - 2 * self.prev_action + self.prev2_action) ** 2).sum(dim=1),
            "torque": ((self.torque / self.tau_max) ** 2).mean(dim=1),
            "joint_limit": self._joint_limit_term(q_act, rcfg),
            "hip_deviation": (q_act[:, self.hip_a] ** 2).sum(dim=1),
        }
        # ---- v8 terms (sources in walk_task.yaml). Weight absent -> term not computed.
        has = lambda k: getattr(w, k, 0.0) != 0.0
        standing = 1.0 - walking
        if has("track_heading"):  # Digit (van Marum 2024) yaw-orientation tracking
            err = self._wrap(self._yaw() - self.yaw_ref)
            terms["track_heading"] = torch.exp(-(err ** 2) / rcfg.heading_sigma_rad ** 2)
        if has("track_vx_lin"):  # humanoid-gym track_vel_hard: non-saturating, keeps pulling at 0.2+
            terms["track_vx_lin"] = (self.lin_vel_b[:, 0] - self.cmd_vx).abs()
        if has("base_acc"):  # humanoid-gym / Digit: smooth base motion (privileged)
            dv = (self.root_lin_vel_w - self.prev_root_lin_vel_w).norm(dim=1)
            terms["base_acc"] = torch.exp(-dv / rcfg.base_acc_sigma_mps)
        if has("base_rock"):  # humanoid-gym vel_mismatch_exp, roll/pitch-rate part
            terms["base_rock"] = torch.exp(-self.ang_vel_b[:, :2].norm(dim=1) / rcfg.base_rock_sigma_rad_s)
        if has("stand_still"):  # Open Duck: no joint motion when the command is 0
            terms["stand_still"] = self.qd[:, self.act_idx].abs().mean(dim=1) * standing
        if has("torque_sat"):  # LX-16A: keep clear of stall (hardware feasibility)
            terms["torque_sat"] = torch.relu(self.torque.abs() / self.tau_max - rcfg.torque_sat_frac).sum(dim=1)
        if has("feet_distance"):  # humanoid-gym: lateral foot spacing within a band
            d = self._feet_lateral_spacing()
            lo, hi = rcfg.feet_lateral_band_frac
            ref = self.feet_y_neutral
            terms["feet_distance"] = (torch.relu(lo * ref - d) + torch.relu(d - hi * ref)) / ref

        # ---- v9 lateral terms (A2): v8 crabs sideways ~1 m per 20 s while holding its heading
        if has("lateral_drift"):  # Open Duck / Microduck: penalize sustained drift, not sway
            yaw = self._yaw()
            vy_h = -torch.sin(yaw) * self.root_lin_vel_w[:, 0] + torch.cos(yaw) * self.root_lin_vel_w[:, 1]
            a = min(self.dt / rcfg.lateral_drift_tau_s, 1.0)
            self.vy_ema = (1 - a) * self.vy_ema + a * vy_h
            terms["lateral_drift"] = torch.relu(self.vy_ema.abs() - rcfg.lateral_drift_deadband_mps)
        if has("roll_band"):  # ToddlerBot: torso roll inside a dead-band
            roll = torch.atan2(g[:, 1], -g[:, 2])
            terms["roll_band"] = torch.exp(-rcfg.roll_band_scale * torch.relu(roll.abs() - rcfg.roll_band_rad))
        if has("foot_flat"):  # stance sole flat on the ground: 1 - (foot z axis . world z), contact feet
            fq = self.foot_quat
            tilt = 2 * (fq[..., 0] ** 2 + fq[..., 1] ** 2)  # 1 - R_zz for an xyzw quaternion
            terms["foot_flat"] = (contact.float() * tilt).sum(dim=1)

        # ---- v9j posture terms: v9i bent its right leg far more than its left (tibia -44 vs +28 deg mean)
        if has("base_height"):  # base lower (relative to the lower foot) than the spawn pose minus a margin, per cm
            terms["base_height"] = torch.relu(self.base_h_neutral - rcfg.base_height_drop_m - self._base_height()) / 0.01
        if has("leg_symmetry"):  # sustained left/right difference of femur and tibia, not the gait's alternation
            fa, ta = self.fem_idx, self.tib_idx
            d = torch.stack([self.q[:, fa[0]] - self.q[:, fa[1]], self.q[:, ta[0]] - self.q[:, ta[1]]], dim=1)
            a = min(self.dt / rcfg.leg_symmetry_tau_s, 1.0)
            self.leg_asym_ema = (1 - a) * self.leg_asym_ema + a * d
            terms["leg_symmetry"] = self.leg_asym_ema.abs().sum(dim=1)
        if has("joint_speed"):  # v9w: v9v moved its joints at 88 deg/s mean (v9s 48) and the servo supply
            # sagged to 4.97 V hanging; on the floor the USB adapter reset. relu(weighted mean |qd| - dead-band), rad/s
            if not hasattr(self, "_jspeed_w"):
                jw = vars(getattr(rcfg, "joint_speed_joint_weights", None) or SimpleNamespace())
                self._jspeed_w = torch.tensor([float(jw.get(n, 1.0)) for n in self.act_names], device=self.q.device)
            mean_qd = (self.qd[:, self.act_idx].abs() * self._jspeed_w).sum(dim=1) / self._jspeed_w.sum()
            terms["joint_speed"] = (mean_qd - rcfg.joint_speed_deadband_rad_s).clamp(min=0.0)
        if has("torque_sum"):  # audit 2026-10-27: current proxy. Total servo torque (N m) above a dead-band; the
            # supply sags with it (0.06 V/(N m)) and current spikes reset the USB adapter, joint speed does not
            terms["torque_sum"] = (self.torque.abs().sum(dim=1) - rcfg.torque_sum_deadband_nm).clamp(min=0.0)
        if has("joint_pose"):  # listed joints near the CAD neutral pose; with the measured masses head and
            # tail move the CoM < 2 mm, yet every policy pinned them at a range end
            idx = [self.act_names.index(n) for n in rcfg.joint_pose_joints]
            terms["joint_pose"] = ((q_act[:, idx] - self.default_q[idx]) / self.joint_limit[idx]).pow(2).sum(dim=1)

        gated = list(getattr(rcfg, "progress_gate", None) or [])
        if gated:  # v9h: posture/smoothness rewards pay only in proportion to forward progress
            walking_now = self._walking()
            ratio = self.lin_vel_b[:, 0] / self.cmd_vx.clamp(min=1e-6)
            if getattr(rcfg, "progress_gate_mode", "progress") == "tracking":  # v9i: overspeed also loses
                prog = (1.0 - (ratio - 1.0).abs()).clamp(0.0, 1.0)
            else:  # progress: clamp(v_x / cmd, 0, 1)
                prog = ratio.clamp(0.0, 1.0)
            gate = torch.where(walking_now, prog, torch.ones_like(prog))
            for k in gated:
                if k in terms:
                    terms[k] = terms[k] * gate

        tilt_exceeded = g[:, 2] > -math.cos(math.radians(self.cfg.termination.max_tilt_deg))
        fell = self.head_z < self.cfg.termination.min_head_height_m
        # any non-foot link on the ground ends the episode: the tail was being used as a
        # third point of balance (a prop), which the real robot must not rely on
        touching = self.ground_force > self.cfg.termination.ground_contact_force_n
        body_contact = touching.any(dim=1)
        terminated = tilt_exceeded | fell | body_contact
        terms["body_contact"] = body_contact.float()
        terms["terminated"] = terminated.float()

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            reward = reward + getattr(w, name, 0.0) * value
        return reward, terms, terminated

    def _accumulate_episode_stats(self, terms: dict, done: torch.Tensor, terminated: torch.Tensor) -> None:
        if not self.ep_sums:
            self.ep_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in terms}
            self.ep_sums["distance_x"] = torch.zeros(self.num_envs, device=self.device)
        for k, v in terms.items():
            self.ep_sums[k] += v
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            steps = self.episode_len[done_ids].clamp(min=1).float()
            rec = {"episode_length_s": (steps * self.dt).mean().item(),
                   "distance_x_m": (self.root_pos[done_ids, 0] - self.start_x[done_ids]).mean().item(),
                   "mean_cmd_vx": self.cmd_vx[done_ids].mean().item(),
                   "yaw_drift_deg": torch.rad2deg(self._wrap(self._yaw() - self.start_yaw)[done_ids].abs()).mean().item(),
                   "terminated_frac": terminated[done_ids].float().mean().item()}
            for k, v in self.ep_sums.items():
                if k != "distance_x":
                    rec[f"term/{k}"] = (v[done_ids] / steps).mean().item()
            self.finished_episodes.append(rec)
            for v in self.ep_sums.values():
                v[done_ids] = 0.0

    def _yaw(self) -> torch.Tensor:
        x, y, z, w = (self.root_q[:, i] for i in range(4))
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return torch.atan2(a.sin(), a.cos())

    # ------------------------------------------------------------------- reset
    def reset_idx(self, ids: torch.Tensor) -> None:
        n, dev, dr = len(ids), self.device, self.cfg.domain_randomization
        noise_q = dr.reset_joint_noise_rad if dr.enabled else 0.0
        tilt = dr.reset_tilt_noise_rad if dr.enabled else 0.0

        q0 = torch.zeros(n, self.num_dofs, device=dev)
        q0[:, self.act_idx] = self.default_q + (torch.rand(n, self.num_actions, device=dev) * 2 - 1) * noise_q
        q0[:, self.act_idx] = self._limit_goal(q0[:, self.act_idx])
        q0[:, self.ank_idx] = q0[:, self.fem_idx] - q0[:, self.tib_idx]
        self._just_reset[ids] = True
        self.vy_ema[ids] = 0.0
        self.leg_asym_ema[ids] = 0.0
        if self.sole_r is not None:
            self.sole_contact_prev[ids] = False  # no slip on the first step after a reset

        pos = self.env_origins[ids].clone()
        pos[:, 2] = self.cfg.sim.spawn_height_m
        rp = (torch.rand(n, 2, device=dev) * 2 - 1) * tilt  # roll, pitch
        yaw = (torch.rand(n, device=dev) * 2 - 1) * tilt
        cr, sr, cp, sp, cy, sy = (rp[:, 0] / 2).cos(), (rp[:, 0] / 2).sin(), (rp[:, 1] / 2).cos(), (rp[:, 1] / 2).sin(), (yaw / 2).cos(), (yaw / 2).sin()
        quat_wxyz = torch.stack([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dim=1)
        self._sim_reset_state(ids, pos, quat_wxyz, q0)

        self.cmd_target[ids] = q0[:, self.act_idx]
        self.last_action[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.prev2_action[ids] = 0.0
        self.smoothed_action[ids] = self.smooth0
        self.air_time[ids] = 0.0
        self.contact_prev[ids] = True  # episodes start with both feet down
        self.episode_len[ids] = 0
        self.phase[ids] = torch.rand(n, device=dev)
        self._sample_commands(ids)
        self.start_x[ids] = pos[:, 0]
        # refresh cached state so the first obs after reset is consistent
        self._read_state()
        self._stale_ref[ids] = self._raw_force[ids]  # whatever is reported right after the teleport is stale
        self._armed[ids] = False
        self.foot_force[ids] = 0.0
        self.ground_force[ids] = 0.0
        self.start_yaw[ids] = self._yaw()[ids]

        # ---- v8 per-episode sensor/actuator randomization
        ob = self.cfg.observation
        d = self.cfg.actions.delay_steps_range
        self.delay[ids] = torch.randint(int(d[0]), int(d[1]) + 1, (n,), device=dev).float()
        di = getattr(ob, "imu_delay_steps_range", [0, 0])
        self.imu_delay[ids] = torch.randint(int(di[0]), int(di[1]) + 1, (n,), device=dev).float()
        mo = math.radians(getattr(ob, "imu_mount_offset_deg", 0.0))
        if mo > 0:  # sample only when on: the RNG stream (and trace_env) stays as before
            self.imu_tilt[ids] = (torch.rand(n, 2, device=dev) * 2 - 1) * mo
        if self.lx16a:
            self._sample_lx16a(ids)
        eo = getattr(ob, "encoder_offset_rad", 0.0)
        self.enc_offset[ids] = (torch.rand(n, self.num_actions, device=dev) * 2 - 1) * eo
        if self.use_heading:
            h = ob.heading
            # the reference is where the robot should keep heading; start a little off it so
            # correcting a heading error is part of the task
            self.yaw_ref[ids] = self.start_yaw[ids] + (torch.rand(n, device=dev) * 2 - 1) * h.ref_init_range_rad
            self.yaw_bias[ids] = 0.0
            self.yaw_drift_rate[ids] = (torch.rand(n, device=dev) * 2 - 1) * math.radians(h.drift_deg_per_min) / 60.0
        push = getattr(self.cfg, "push", None)
        if push is not None:
            self.push_timer[ids] = torch.empty(n, device=dev).uniform_(*push.interval_s)
        self.prev_root_lin_vel_w[ids] = self.root_lin_vel_w[ids]
        if self.feet_y_neutral is None:
            self.feet_y_neutral = self._feet_lateral_spacing().mean().item()
        if self.base_h_neutral is None:
            self.base_h_neutral = self._base_height().mean().item()
        self._hist_fresh[ids] = True
        self.q_meas_prev[ids] = self._measured_q(noisy=True)[ids]

    # ------------------------------------------------------------- observations
    def _measured_q(self, noisy: bool) -> torch.Tensor:
        q = self.q[:, self.act_idx]
        if noisy:  # servo reading: calibration offset (per episode) + noise
            q = q + self.enc_offset + torch.randn_like(q) * self.noise.joint_pos_rad
        return torch.round(q / self.servo_res) * self.servo_res  # LX-16A resolution

    def _heading_err(self, measured: bool) -> torch.Tensor:
        """yaw - yaw_ref. Measured = what the Pi computes from the BNO055 fused yaw, which
        drifts slowly (yaw_bias) after the reference was latched."""
        e = self._yaw() - self.yaw_ref
        if measured:
            e = e + self.yaw_bias
        return self._wrap(e)

    def _imu_mount(self, v: torch.Tensor) -> torch.Tensor:
        """Body-frame vector as read by an IMU mounted rotated by (roll about x, then pitch about y)."""
        cr, sr = self.imu_tilt[:, 0].cos(), self.imu_tilt[:, 0].sin()
        cp, sp = self.imu_tilt[:, 1].cos(), self.imu_tilt[:, 1].sin()
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        y, z = cr * y + sr * z, -sr * y + cr * z  # R_x(roll)^T
        x, z = cp * x - sp * z, sp * x + cp * z  # R_y(pitch)^T
        return torch.stack([x, y, z], dim=1)

    def _observations(self) -> dict:
        sc = self.cfg.observation.scales
        n = self.noise
        # IMU reading (true now), optionally one control step stale (per-episode latency)
        imu_now = [self.gravity_b, self.ang_vel_b]
        if self.use_heading:
            imu_now.append(self._heading_err(measured=True).unsqueeze(1))
        imu_now = torch.cat(imu_now, dim=1)
        if self.imu_prev is None:  # (K, N, D): readings 1..K control steps old, K = max IMU delay (>= 1)
            k = max(1, int(max(getattr(self.cfg.observation, "imu_delay_steps_range", [0, 0]))))
            self.imu_prev = imu_now.unsqueeze(0).repeat(k, 1, 1)
        self.imu_prev[:, self._hist_fresh] = imu_now[self._hist_fresh]
        buf = torch.cat([imu_now.unsqueeze(0), self.imu_prev], dim=0)
        imu = buf[self.imu_delay.long().clamp(max=buf.shape[0] - 1), torch.arange(imu_now.shape[0], device=imu_now.device)]
        self.imu_prev = buf[:-1].clone()

        g, gyro = imu[:, 0:3], imu[:, 3:6]
        if getattr(self.cfg.observation, "imu_mount_offset_deg", 0.0) > 0:
            g, gyro = self._imu_mount(g), self._imu_mount(gyro)
        g = g + torch.randn_like(self.gravity_b) * n.gravity
        g = g / g.norm(dim=1, keepdim=True)
        gyro = gyro + torch.randn_like(self.ang_vel_b) * n.gyro_rad_s
        q_meas = self._measured_q(noisy=True)
        dq_est = (q_meas - self.q_meas_prev) / self.dt
        self.q_meas_prev = q_meas

        parts = [g, gyro * sc.gyro, self.cmd_vx.unsqueeze(1), (q_meas - self.act_center) / self.joint_limit, dq_est * sc.joint_vel, self.last_action]
        clock = None
        if self.use_clock:
            ang = 2 * math.pi * self.phase
            clock = torch.stack([ang.sin(), ang.cos()], dim=1) * self._walking().float().unsqueeze(1)
            parts.append(clock)
        if self.use_heading:
            he = imu[:, 6] + torch.randn_like(imu[:, 6]) * self.cfg.observation.heading.noise_rad
            parts.append(torch.stack([he.sin(), he.cos()], dim=1))
        frame = torch.cat(parts, dim=1)

        # history: newest frame first; a fresh episode starts with K copies of its first frame
        if self.history_len > 1:
            self.obs_hist = torch.roll(self.obs_hist, shifts=1, dims=1)
            self.obs_hist[:, 0] = frame
            fresh = self._hist_fresh
            self.obs_hist[fresh] = frame[fresh].unsqueeze(1).expand(-1, self.history_len, -1)
            actor = self.obs_hist.reshape(self.num_envs, -1)
        else:
            actor = frame
        self._hist_fresh[:] = False

        # critic: one clean frame + privileged state
        q_clean = self._measured_q(noisy=False)
        clean = [self.gravity_b, self.ang_vel_b * sc.gyro, self.cmd_vx.unsqueeze(1), (q_clean - self.act_center) / self.joint_limit,
                 dq_est * sc.joint_vel, self.last_action]
        if self.use_clock:
            clean.append(clock)
        if self.use_heading:
            he = self._heading_err(measured=False)
            clean.append(torch.stack([he.sin(), he.cos()], dim=1))
        priv = [self.lin_vel_b * sc.lin_vel, self._contacts().float(),
                self._foot_clearance() / self.cfg.foot.swing_clearance_target_m, self.torque / self.tau_max]
        critic = torch.cat(clean + priv, dim=1)
        return {"actor": actor, "critic": critic}

    def obs_layout(self) -> dict:
        """Actor-input contract for the Pi runner (stored in the exported policy's meta)."""
        names = ["gravity_b(3)", "gyro_b*gyro_scale(3)", "cmd_vx(1)", "servo_pos/limit(8)",
                 "finite_diff_servo_vel*joint_vel_scale(8)", "prev_action(8)"]
        if self.use_clock:
            names.append("clock sin,cos(2; zero when standing)")
        if self.use_heading:
            names.append("heading_err sin,cos(2; yaw - yaw_ref latched at start)")
        return {"frame": names, "frame_dim": self.frame_dim, "history_len": self.history_len,
                "order": "newest frame first", "gyro_scale": self.cfg.observation.scales.gyro,
                "joint_vel_scale": self.cfg.observation.scales.joint_vel}

    def reset(self) -> dict:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._observations()

    def pop_episode_stats(self) -> dict:
        """Mean of per-episode stats finished since the last call ({} if none)."""
        eps, self.finished_episodes = self.finished_episodes, []
        res, self._ankle_res = getattr(self, "_ankle_res", []), []
        extra = {} if not res else {"ankle_res_p95_deg": float(np.mean([r[0] for r in res])),
                                    "ankle_res_max_deg": float(max(r[1] for r in res))}
        if not eps:
            return extra
        return {k: float(np.mean([e[k] for e in eps])) for k in eps[0]} | {"episodes": len(eps)} | extra
