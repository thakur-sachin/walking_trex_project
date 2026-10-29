"""Vectorized Trex forward-walking environment on Isaac Sim 6.0.1 (PhysX, GPU).

Built directly on `isaacsim.core.experimental` Articulation + GridCloner (no
Isaac Lab dependency), driving the *current* pipeline: URDF import + real
robot_config.yaml (limits, servo torque cap, mass) via scripts/test_load_robot.py.

Interface is grounded in what the real robot can sense and do -- see
walk_task.yaml's header for the full contract. In short:

  action  (8):  servo position targets for robot_config `actuated_joints`.
  actor obs:    IMU gravity + gyro, 8 quantized servo positions, finite-difference
                joint velocity, previous action, forward-velocity command, gait
                clock. Nothing the real robot cannot measure.
  critic obs:   actor obs (clean) + privileged sim state (training only).

Servo model (LX-16A, from robot_config.yaml `actuator`): torque cap, position
resolution (commands + readings quantized), speed limit (target ramped across
the control period like the servo's move-time command, capped at no-load
speed), optional bus latency, randomized strength.

Ankle: passive, set by the confirmed 4-bar linkage (ankle = femur - tibia).
Enforced kinematically every physics substep (position teleport + aligned PD
target). That tracked the linkage to ~0.5 deg in testing; a PD-only ankle did
not (see docs/WALK_TASK.md).

Must be imported AFTER a SimulationApp exists (same rule as test_load_robot).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import omni.timeline
import omni.usd
import torch
import warp as wp
import yaml
from isaacsim.core.cloner import GridCloner
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import PhysxGpuCfg, SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import test_load_robot as tlr  # noqa: E402  (needs SimulationApp; reuse verified import pipeline)
from common.robot_config import RobotConfig  # noqa: E402

DEFAULT_TASK_PATH = Path(__file__).resolve().parent / "walk_task.yaml"


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def ns_to_dict(ns):
    return {k: ns_to_dict(v) if isinstance(v, SimpleNamespace) else v for k, v in vars(ns).items()}


def _deep_update(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_task_config(path: str | Path = DEFAULT_TASK_PATH) -> SimpleNamespace:
    """Load a task config, overlaid on the current defaults so that a run saved before a
    key existed (e.g. v4's walk_task.yaml has no ground_contact_links) still loads, and is
    evaluated under today's rules for anything it did not specify."""
    with open(DEFAULT_TASK_PATH) as f:
        cfg = yaml.safe_load(f)
    if Path(path) != DEFAULT_TASK_PATH:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return _ns(cfg)


def apply_checkpoint_interface(cfg: SimpleNamespace, meta: dict) -> SimpleNamespace:
    """Make the actor-input layout match a saved policy. Runs trained before v8 (no
    `obs_layout` in their meta) had one frame and no heading input; today's defaults
    would otherwise give them a bigger observation than their network takes."""
    layout = meta.get("obs_layout")
    ob = cfg.observation
    if layout is None:
        ob.history_len = 1
        if hasattr(ob, "heading"):
            ob.heading.enabled = False
    else:
        ob.history_len = layout["history_len"]
        if hasattr(ob, "heading"):
            ob.heading.enabled = any(n.startswith("heading_err") for n in layout["frame"])
    return cfg


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vector `v` into the frame of quaternion `q` (xyzw)."""
    q_vec = q[..., :3]
    w = q[..., 3:4]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def _wp(t: torch.Tensor) -> wp.array:
    return wp.from_torch(t.contiguous())


def _T(a: wp.array) -> torch.Tensor:
    return wp.to_torch(a)


class TrexWalkEnv:
    def __init__(self, num_envs: int, task_cfg: SimpleNamespace | None = None, device: str = "cuda:0",
                 simulation_app=None, render: bool = False, seed: int = 0):
        self.cfg = task_cfg or load_task_config()
        self.robot_cfg = RobotConfig.load(tlr.ROBOT_CONFIG_PATH)
        self.num_envs = num_envs
        self.device = device
        self.simulation_app = simulation_app
        self.render = render
        torch.manual_seed(seed)

        c = self.cfg
        self.physics_dt = 1.0 / c.sim.physics_hz
        assert c.sim.physics_hz % c.sim.control_hz == 0, "control_hz must divide physics_hz"
        self.decimation = c.sim.physics_hz // c.sim.control_hz
        self.dt = self.physics_dt * self.decimation
        self.max_episode_steps = int(round(c.sim.episode_length_s / self.dt))

        self._build_scene()
        self._init_buffers()
        self._randomize_startup()
        self.reset_idx(torch.arange(num_envs, device=device))

    # ------------------------------------------------------------------ scene
    def _build_scene(self) -> None:
        c, N = self.cfg, self.num_envs
        usd_path = tlr.convert_urdf_to_usd(self.robot_cfg)

        stage_utils.create_new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        SimulationManager.setup_simulation(dt=self.physics_dt, device=self.device)
        if not self.render:
            # Fabric sync of every link's USD transform every physics step costs
            # ~5x step time at 512 envs (measured) and nothing here reads USD.
            SimulationManager.enable_fabric(False)
        GroundPlane("/World/GroundPlane", sizes=1000.0)

        stage.DefinePrim("/World/envs", "Xform")
        stage.DefinePrim("/World/envs/env_0", "Xform")
        robot_prim = stage.DefinePrim("/World/envs/env_0/Trex", "Xform")
        robot_prim.GetReferences().AddReference(str(usd_path))
        robot_prim.Load(Usd.LoadWithDescendants)
        for _ in range(5):
            self.simulation_app.update()
        tlr.apply_robot_config(stage, "/World/envs/env_0/Trex", self.robot_cfg)
        self._mark_contact_reporters(stage)
        if self.render:
            # Visual-only extras. Never call simulation_app.update() while the timeline
            # plays inside step(): it would advance physics a second time. Rendering
            # scripts pause the timeline, update, capture, then play (see render_gait.py).
            from pxr import Gf, Sdf, UsdLux
            tlr.apply_materials(stage, "/World/envs/env_0/Trex", self.robot_cfg)
            light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
            light.CreateIntensityAttr(1000.0)
            light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        cloner = GridCloner(spacing=c.sim.env_spacing_m)
        cloner.define_base_env("/World/envs")
        paths = cloner.generate_paths("/World/envs/env", N)
        origins = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=paths, replicate_physics=True)
        cloner.filter_collisions("/physicsScene", "/World/collisions", paths, ["/World/GroundPlane"])
        self.env_origins = torch.as_tensor(np.asarray(origins), dtype=torch.float32, device=self.device)

        # GPU broadphase/aggregate buffers scale with env count; the defaults
        # overflow past ~1k envs and PhysX then silently misses interactions.
        cap = 1 << int(math.ceil(math.log2(max(8 * N, 1024))))  # ~4.4/env needed, measured
        scene = SimulationManager.get_physics_scenes()[0]
        scene.set_gpu_configuration(PhysxGpuCfg(
            gpu_found_lost_aggregate_pairs_capacity=cap,
            gpu_total_aggregate_pairs_capacity=cap,
            gpu_found_lost_pairs_capacity=max(262144, 64 * N),
        ))

        # Rendering: the GPU pipeline suppresses USD readback (only Fabric sees physics, and
        # the headless renderer draws from USD -> frozen robot). Render with device="cpu",
        # which writes transforms back to USD every step; Fabric sync only if available.
        self._fabric_sync = bool(self.render and SimulationManager.is_fabric_enabled())

        self.robot = Articulation("/World/envs/env_.*/Trex")
        omni.timeline.get_timeline_interface().play()
        for _ in range(5):
            self.simulation_app.update()
        self.view = self.robot._physics_articulation_view
        self._create_contact_views()

    def _mark_contact_reporters(self, stage) -> None:
        """Enable PhysX contact reports on the foot bodies and on every body link that must
        never touch the ground (env_0, before cloning)."""
        from pxr import PhysxSchema

        self.ground_links = list(self.cfg.termination.ground_contact_links)
        wanted = {"left_foot", "right_foot", *self.ground_links}
        root = stage.GetPrimAtPath("/World/envs/env_0/Trex")
        self.link_prim_paths = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.GetName() in wanted and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.IsInstanceProxy():
                    raise RuntimeError(f"rigid body {prim.GetPath()} is an instance proxy; cannot author contact API")
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
                self.link_prim_paths[prim.GetName()] = str(prim.GetPath())
        missing = wanted - set(self.link_prim_paths)
        assert not missing, f"rigid bodies not found for contact reports: {missing}"
        self.foot_prim_paths = {k: self.link_prim_paths[k] for k in ("left_foot", "right_foot")}

    def _make_contact_view(self, prim_path: str):
        """Rigid-contact view over one link across all envs, plus the row->env permutation."""
        sim_view = SimulationManager.get_physics_sim_view()
        pattern = prim_path.replace("/env_0/", "/env_*/")  # glob, not regex
        view = sim_view.create_rigid_contact_view(pattern, filter_patterns=[], max_contact_data_count=0)
        paths = list(view.sensor_paths)
        env_of = [int(pth.split("/env_")[1].split("/")[0]) for pth in paths]
        assert sorted(env_of) == list(range(self.num_envs)), f"contact view {prim_path}: unexpected sensors"
        perm = torch.empty(self.num_envs, dtype=torch.long)
        for row, env in enumerate(env_of):
            perm[env] = row
        return view, perm.to(self.device)

    def _create_contact_views(self) -> None:
        self.contact_views, self.contact_perm = [], []
        for name in ("left_foot", "right_foot"):
            v, p = self._make_contact_view(self.link_prim_paths[name])
            self.contact_views.append(v)
            self.contact_perm.append(p)
        self.ground_views = [self._make_contact_view(self.link_prim_paths[n]) for n in self.ground_links]

    def _contact_force_of(self, view, perm) -> torch.Tensor:
        f = view.get_net_contact_forces(self.physics_dt)  # (N, 3): torch on GPU, numpy/warp on CPU
        if isinstance(f, torch.Tensor):
            pass
        elif isinstance(f, np.ndarray):
            f = torch.as_tensor(f, device=self.device)
        else:
            f = _T(f)
        return f.norm(dim=-1)[perm]

    def _ground_contact_force(self) -> torch.Tensor:
        """(N, K) net contact force [N] on each link that must stay off the ground."""
        return torch.stack([self._contact_force_of(v, p) for v, p in self.ground_views], dim=1)

    def _foot_contact_force(self) -> torch.Tensor:
        """(N, 2) net contact force magnitude [N] on (left, right) foot, ordered by env."""
        return torch.stack([self._contact_force_of(v, p) for v, p in zip(self.contact_views, self.contact_perm)], dim=1)

    # ---------------------------------------------------------------- buffers
    def _init_buffers(self) -> None:
        N, dev, rc, c = self.num_envs, self.device, self.robot_cfg, self.cfg
        dn, ln = self.robot.dof_names, self.robot.link_names
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

        lim = torch.tensor([rc.joint_limit_rad(n)[1] for n in self.act_names], device=dev)
        self.joint_limit = lim
        self.act_scale = lim * c.actions.scale_frac_of_limit
        self.default_q = torch.zeros(self.num_actions, device=dev)  # CAD neutral pose
        self.servo_res = math.radians(rc.actuator_spec["position_accuracy_deg"])
        self.max_target_step = rc.actuator_no_load_speed_rad_per_sec * self.dt
        self.tau_max = rc.actuator_stall_torque_nm

        self.base_i = ln.index("base_link")
        self.head_i = ln.index("head")
        self.foot_i = torch.tensor([ln.index("left_foot"), ln.index("right_foot")], device=dev)

        self.num_dofs = self.robot.num_dofs
        self.all_idx = torch.arange(N, device=dev)
        self.all_idx_wp = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32, device=dev)
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
        self.total_mass = _T(self.view.get_masses()).sum(dim=1)  # (N,) kg, incl. DR
        self.link_count = self.robot.num_links
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
        self.imu_delay = z(N)  # per-episode IMU latency in control steps (0/1)
        self.imu_prev = None  # last step's (gravity, gyro, heading-error) reading, for imu_delay
        self.enc_offset = z(N, self.num_actions)  # per-episode servo calibration offset (readings only)
        self.root_lin_vel_w = z(N, 3)
        self.prev_root_lin_vel_w = z(N, 3)
        self.feet_y_neutral = None  # lateral foot spacing in the spawn pose, measured at first reset
        self.push_scale = 0.0  # training-only disturbance strength in [0,1]; set by the trainer
        self.push_timer = z(N)
        self.push_count = 0

        self.noise = self.cfg.observation.noise
        self.use_clock = c.gait.use_phase_clock
        self.frame_dim = 3 + 3 + 1 + 8 + 8 + 8 + (2 if self.use_clock else 0) + (2 if self.use_heading else 0)
        self.num_actor_obs = self.frame_dim * self.history_len
        self.obs_hist = z(N, self.history_len, self.frame_dim)
        self._hist_fresh = torch.ones(N, dtype=torch.bool, device=dev)  # fill history with the first frame
        # critic: one clean frame + privileged state
        self.num_critic_obs = self.frame_dim + 3 + 2 + 2 + 8
        self.ep_sums: dict[str, torch.Tensor] = {}
        self.finished_episodes: list[dict] = []

    # ------------------------------------------------------ domain randomization
    def _randomize_startup(self) -> None:
        dr, N, dev = self.cfg.domain_randomization, self.num_envs, self.device
        self.dr_active = {}
        lo_hi = lambda r: torch.empty(N, 1, device=dev).uniform_(r[0], r[1])
        self.dr_active["action_delay"] = True  # resampled per episode in reset_idx()
        if not dr.enabled:
            return

        def attempt(name, fn):
            try:
                fn()
                self.dr_active[name] = True
            except Exception as e:  # report, don't hide: a skipped DR is a real gap
                self.dr_active[name] = f"SKIPPED: {type(e).__name__}: {str(e)[:80]}"

        def friction():
            # Material setters are CPU-only in the GPU pipeline (tensors AND indices).
            mp = _T(self.view.get_material_properties()).clone().cpu()  # (N, shapes, 3)
            f = lo_hi(dr.friction_range).unsqueeze(-1).cpu()
            mp[..., 0:2] = f
            self.view.set_material_properties(_wp(mp), wp.array(np.arange(N, dtype=np.int32), device="cpu"))

        def mass():
            m = _T(self.robot.get_link_masses()).clone()  # (N, L)
            m[:, self.base_i] *= lo_hi(dr.base_mass_scale_range).squeeze(-1)
            limb = getattr(dr, "limb_mass_scale_range", None)  # head/tail: printed shells vary
            if limb is not None:
                for name in ("head", "tail"):
                    m[:, self.robot.link_names.index(name)] *= lo_hi(limb).squeeze(-1)
            self.robot.set_link_masses(_wp(m))

        def base_com():
            # base_link lumps battery + Pi + wiring whose placement is not measured
            pos, _ = self.robot.get_link_coms(link_indices=[self.base_i])  # (N, 1, 3)
            p = _T(pos).clone()
            r = dr.base_com_offset_m
            p += (torch.rand_like(p) * 2 - 1) * r
            self.robot.set_link_coms(_wp(p), link_indices=[self.base_i])

        def gains():
            s = lo_hi(dr.pd_gain_scale_range)
            kp, kd = self.robot.get_dof_gains()
            self.robot.set_dof_gains(_wp(_T(kp) * s), _wp(_T(kd) * s))

        def torque_cap():
            s = lo_hi(dr.torque_cap_scale_range)
            mf = _T(self.robot.get_dof_max_efforts()).clone()
            mf[:, self.act_idx] *= s
            self.robot.set_dof_max_efforts(_wp(mf))

        attempt("friction", friction)
        attempt("base_mass", mass)
        if getattr(dr, "base_com_offset_m", 0.0):
            attempt("base_com", base_com)
        attempt("pd_gains", gains)
        attempt("torque_cap", torque_cap)
        self.total_mass = _T(self.view.get_masses()).sum(dim=1)  # after mass DR (harness weight support)

    # ----------------------------------------------------------------- reading
    def _read_state(self) -> None:
        self.q = _T(self.robot.get_dof_positions())
        self.qd = _T(self.robot.get_dof_velocities())
        lt = _T(self.view.get_link_transforms())  # (N, L, 7) pos + quat xyzw
        lv = _T(self.view.get_link_velocities())  # (N, L, 6) lin, ang (world)
        self.root_q = lt[:, self.base_i, 3:7]
        self.root_pos = lt[:, self.base_i, 0:3]
        self.root_lin_vel_w = lv[:, self.base_i, 0:3]
        self.foot_pos = lt[:, self.foot_i, 0:3]
        self.lin_vel_b = quat_rotate_inverse(self.root_q, lv[:, self.base_i, 0:3])
        self.ang_vel_b = quat_rotate_inverse(self.root_q, lv[:, self.base_i, 3:6])
        self.gravity_b = quat_rotate_inverse(self.root_q, self.gravity_w)
        self.foot_z = lt[:, self.foot_i, 2]
        self.head_z = lt[:, self.head_i, 2]
        self.foot_vel_xy = lv[:, self.foot_i, 0:2]
        self.torque = _T(self.robot.get_dof_projected_joint_forces())[:, self.act_idx]
        raw = torch.cat([self._foot_contact_force(), self._ground_contact_force()], dim=1)
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

    def _feet_lateral_spacing(self) -> torch.Tensor:
        """(N,) left-right foot distance across the heading direction (yaw-only frame, so
        body roll during weight shift does not change it)."""
        d = self.foot_pos[:, 0, :2] - self.foot_pos[:, 1, :2]
        yaw = self._yaw()
        return (-yaw.sin() * d[:, 0] + yaw.cos() * d[:, 1]).abs()

    def _foot_clearance(self) -> torch.Tensor:
        """(N, 2) height of each foot above the other. Both feet share the base's tilt, so the
        difference is independent of lean (unlike absolute link height)."""
        return self.foot_z - self.foot_z.flip(dims=[1])

    # -------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
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
        raw = self.default_q + self.smoothed_action * self.act_scale
        if self.cfg.actions.quantize_to_servo_resolution:
            raw = torch.round(raw / self.servo_res) * self.servo_res
        # Servo ramps to the goal over the control period (move-time command),
        # at no more than its no-load speed.
        start = self.cmd_target
        goal = start + (raw - start).clamp(-self.max_target_step, self.max_target_step)
        goal = torch.max(torch.min(goal, self.joint_limit), -self.joint_limit)

        if self.push_scale > 0.0:
            self._apply_push()
        self.prev_root_lin_vel_w = self.root_lin_vel_w.clone()

        dof_tgt = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        for i in range(self.decimation):
            tgt = start + (goal - start) * ((i + 1) / self.decimation)
            dof_tgt[:, self.act_idx] = tgt
            dof_tgt[:, self.ank_idx] = tgt[:, self.fem_a] - tgt[:, self.tib_a]
            # raw physics-tensor view in the hot path: the Articulation wrapper
            # adds per-call overhead that dominates at 8 substeps/step.
            self.view.set_dof_position_targets(_wp(dof_tgt), self.all_idx_wp)
            self._enforce_ankle_linkage()
            if self.harness_alpha > 0.0:
                self._apply_harness()
            # render mode: push physics results to Fabric every substep, otherwise the
            # rendered robot lags the physics state (camera followed the *physics* pose
            # and the visual robot walked out of frame).
            SimulationManager.step(steps=1, update_fabric=self._fabric_sync)
        self.cmd_target = goal

        self.episode_len += 1
        self._read_state()
        self._advance_commands_and_phase()
        reward, terms, terminated = self._compute_reward()
        timeout = self.episode_len >= self.max_episode_steps
        nonfinite = ~torch.isfinite(self.q).all(dim=1) | ~torch.isfinite(self.root_pos).all(dim=1)
        terminated = terminated | nonfinite
        reward = torch.nan_to_num(reward)
        self._accumulate_episode_stats(terms, terminated | timeout, terminated)

        done = terminated | timeout
        extras = {"time_outs": timeout & ~terminated}
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.reset_idx(reset_ids)
        return self._observations(), reward, done, extras

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
        rv = _T(self.view.get_root_velocities()).clone()  # (N, 6)
        lin = rv[ids, 0:3] + dv_w
        ang = rv[ids, 3:6] + dw_w
        idx = wp.array(ids.int().cpu().numpy(), dtype=wp.int32, device=self.device)
        self.robot.set_velocities(_wp(lin), _wp(ang), indices=idx)
        self.root_lin_vel_w = self.root_lin_vel_w.clone()
        self.root_lin_vel_w[ids] += dv_w  # the push itself is not "base acceleration" the policy caused

    def _apply_harness(self) -> None:
        """Training-only assist (curriculum, annealed to 0 by the trainer): an upward force
        on the base (fraction of body weight) plus a righting torque on roll/pitch. It makes
        lifting a foot safe so stepping can be *discovered*; with the tail forbidden as a
        prop, v6 never lifted a foot unaided (air time 0.000 after 200 iterations).
        Applied each substep (PhysX clears external forces after every step)."""
        h = self.cfg.harness
        lt = _T(self.view.get_link_transforms())  # (N, L, 7)
        q = lt[:, self.base_i, 3:7]
        g_b = quat_rotate_inverse(q, self.gravity_w)  # gravity in body frame
        up_b = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(g_b)
        # Restoring rotation (body frame) is t x g_b: for nose-down pitch th, g_b=(sin th,0,-cos th),
        # t x g_b = (0,-sin th,0) = nose up. (g_b x t had the wrong sign: positive feedback.)
        err_b = torch.cross(up_b, g_b, dim=-1)
        w_b = quat_rotate_inverse(q, _T(self.view.get_link_velocities())[:, self.base_i, 3:6])
        torque_b = h.torque_kp * err_b - h.torque_kd * w_b
        torque_b[:, 2] = 0.0  # never fight yaw
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=1)
        torque_w = quat_rotate_inverse(q_conj, torque_b)  # body -> world
        a = self.harness_alpha
        force = torch.zeros(self.num_envs, self.link_count, 3, device=self.device)
        torque = torch.zeros_like(force)
        force[:, self.base_i, 2] = a * h.weight_fraction * self.total_mass * 9.81
        torque[:, self.base_i, :] = a * torque_w
        self.view.apply_forces_and_torques_at_position(
            _wp(force), _wp(torque), _wp(lt[:, :, 0:3].contiguous()), self.all_idx_wp, True)

    def _enforce_ankle_linkage(self) -> None:
        """ankle = femur - tibia (robot_config ankle_linkage), teleported each substep."""
        q = _T(self.view.get_dof_positions()).clone()
        q[:, self.ank_idx] = q[:, self.fem_idx] - q[:, self.tib_idx]
        self.view.set_dof_positions(_wp(q), self.all_idx_wp)

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
        stand = torch.rand(len(ids), device=self.device) < self.cfg.command.standing_prob
        self.cmd_vx[ids] = torch.where(stand, torch.zeros_like(vx), vx)
        self.cmd_timer[ids] = self.cfg.command.resample_time_s

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

        terms = {
            "track_vx": torch.exp(-((self.lin_vel_b[:, 0] - self.cmd_vx) ** 2) / rcfg.track_vx_sigma ** 2),
            "alive": torch.ones(self.num_envs, device=self.device),
            "upright": (g[:, 0] ** 2 + g[:, 1] ** 2),
            "lateral_vel": self.lin_vel_b[:, 1] ** 2,
            "yaw_rate": self.ang_vel_b[:, 2] ** 2,
            # Heading is not observable to the policy (gyro only), so hold a straight
            # line by tracking zero yaw rate tightly rather than penalizing heading.
            "track_yaw": torch.exp(-(self.ang_vel_b[:, 2] ** 2) / rcfg.yaw_sigma_rad_s ** 2),
            "ang_vel_xy": self.ang_vel_b[:, 0] ** 2 + self.ang_vel_b[:, 1] ** 2,
            "gait_contact": (contact == stance).float().mean(dim=1),
            "air_time": air_reward.sum(dim=1) * walking,
            "swing_clearance": ((~stance).float() * (foot_h / self.cfg.foot.swing_clearance_target_m).clamp(0, 1)).mean(dim=1) * walking,
            "foot_slip": (contact.float() * (self.foot_vel_xy ** 2).sum(-1)).sum(dim=1),
            "action_rate": ((self.last_action - self.prev_action) ** 2).sum(dim=1),
            # second difference: ~0 for a smooth gait, large for alternating-sign chatter
            "action_accel": ((self.last_action - 2 * self.prev_action + self.prev2_action) ** 2).sum(dim=1),
            "torque": ((self.torque / self.tau_max) ** 2).mean(dim=1),
            "joint_limit": torch.relu(q_act.abs() / self.joint_limit - rcfg.joint_limit_margin).sum(dim=1),
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
        q0[:, self.act_idx] = torch.max(torch.min(q0[:, self.act_idx], self.joint_limit), -self.joint_limit)
        q0[:, self.ank_idx] = q0[:, self.fem_idx] - q0[:, self.tib_idx]

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

        idx = wp.array(ids.int().cpu().numpy(), dtype=wp.int32, device=dev)
        self.robot.set_world_poses(_wp(pos), _wp(quat_wxyz), indices=idx)
        zeros6 = torch.zeros(n, 3, device=dev)
        self.robot.set_velocities(_wp(zeros6), _wp(zeros6), indices=idx)
        self.robot.set_dof_positions(_wp(q0), indices=idx)
        self.robot.set_dof_velocities(_wp(torch.zeros(n, self.num_dofs, device=dev)), indices=idx)
        self.robot.set_dof_position_targets(_wp(q0), indices=idx)

        self.cmd_target[ids] = q0[:, self.act_idx]
        self.last_action[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.prev2_action[ids] = 0.0
        self.smoothed_action[ids] = 0.0
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

    def _observations(self) -> dict:
        sc = self.cfg.observation.scales
        n = self.noise
        # IMU reading (true now), optionally one control step stale (per-episode latency)
        imu_now = [self.gravity_b, self.ang_vel_b]
        if self.use_heading:
            imu_now.append(self._heading_err(measured=True).unsqueeze(1))
        imu_now = torch.cat(imu_now, dim=1)
        if self.imu_prev is None:
            self.imu_prev = imu_now.clone()
        self.imu_prev[self._hist_fresh] = imu_now[self._hist_fresh]
        imu = torch.where(self.imu_delay.unsqueeze(1) > 0.5, self.imu_prev, imu_now)
        self.imu_prev = imu_now.clone()

        g = imu[:, 0:3] + torch.randn_like(self.gravity_b) * n.gravity
        g = g / g.norm(dim=1, keepdim=True)
        gyro = imu[:, 3:6] + torch.randn_like(self.ang_vel_b) * n.gyro_rad_s
        q_meas = self._measured_q(noisy=True)
        dq_est = (q_meas - self.q_meas_prev) / self.dt
        self.q_meas_prev = q_meas

        parts = [g, gyro * sc.gyro, self.cmd_vx.unsqueeze(1), q_meas / self.joint_limit, dq_est * sc.joint_vel, self.last_action]
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
        clean = [self.gravity_b, self.ang_vel_b * sc.gyro, self.cmd_vx.unsqueeze(1), q_clean / self.joint_limit,
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
        if not eps:
            return {}
        return {k: float(np.mean([e[k] for e in eps])) for k in eps[0]} | {"episodes": len(eps)}
