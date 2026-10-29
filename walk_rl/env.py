"""Vectorized Trex forward-walking environment on Isaac Sim 6.0.1 (PhysX, GPU).

Built directly on `isaacsim.core.experimental` Articulation + GridCloner (no
Isaac Lab dependency), driving the *current* pipeline: URDF import + real
robot_config.yaml (limits, servo torque cap, mass) via scripts/test_load_robot.py.
This is the original training stack and the fallback; walk_lab/ is the Isaac Lab port.

The task itself (servo model, rewards, observations, commands, curricula, per-episode
randomization) lives in walk_rl/task_core.py and is shared with the Isaac Lab port, so
both run the identical task. This file is only the Isaac Sim side: scene build, contact
views, and the `_sim_*` / `_dr_*` hooks.

Interface is grounded in what the real robot can sense and do -- see
walk_task.yaml's header for the full contract. In short:

  action  (8):  servo position targets for robot_config `actuated_joints`.
  actor obs:    IMU gravity + gyro, 8 quantized servo positions, finite-difference
                joint velocity, previous action, forward-velocity command, gait
                clock. Nothing the real robot cannot measure.
  critic obs:   actor obs (clean) + privileged sim state (training only).

Ankle: passive, set by the confirmed 4-bar linkage (ankle = femur - tibia).
Enforced kinematically every physics substep (position teleport + aligned PD
target). That tracked the linkage to ~0.5 deg in testing; a PD-only ankle did
not (see docs/PROJECT_HISTORY.md).

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
from walk_rl.config import robot_asset_dir  # noqa: E402
from walk_rl.ground import apply_ground_compliance  # noqa: E402
from walk_rl.self_collision import apply_self_collision  # noqa: E402
from walk_rl.task_core import (  # noqa: E402,F401  (re-exported: scripts import these from walk_rl.env)
    DEFAULT_TASK_PATH,
    TrexTaskCore,
    apply_checkpoint_interface,
    self_contact_partners,
    load_task_config,
    ns_to_dict,
    quat_rotate_inverse,
)


def _wp(t: torch.Tensor) -> wp.array:
    return wp.from_torch(t.contiguous())


def _T(a: wp.array) -> torch.Tensor:
    return wp.to_torch(a)


class TrexWalkEnv(TrexTaskCore):
    def __init__(self, num_envs: int, task_cfg: SimpleNamespace | None = None, device: str = "cuda:0",
                 simulation_app=None, render: bool = False, seed: int = 0):
        self.cfg = task_cfg or load_task_config()
        self.asset_dir = robot_asset_dir(self.cfg)
        self.robot_cfg = RobotConfig.load(self.asset_dir / "robot_config.yaml")
        self.num_envs = num_envs
        self.device = device
        self.simulation_app = simulation_app
        self.render = render
        torch.manual_seed(seed)

        self._setup_timing()
        self._build_scene()
        self.all_idx_wp = wp.array(np.arange(num_envs, dtype=np.int32), dtype=wp.int32, device=device)
        self._init_buffers()
        self._randomize_startup()
        self.reset_idx(torch.arange(num_envs, device=device))

    # ------------------------------------------------------------------ scene
    def _build_scene(self) -> None:
        c, N = self.cfg, self.num_envs
        usd_path = tlr.convert_urdf_to_usd(self.robot_cfg, self.asset_dir)

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
        apply_ground_compliance(stage, "/World/GroundPlane", getattr(c.sim, "ground_compliance", None))

        stage.DefinePrim("/World/envs", "Xform")
        stage.DefinePrim("/World/envs/env_0", "Xform")
        robot_prim = stage.DefinePrim("/World/envs/env_0/Trex", "Xform")
        robot_prim.GetReferences().AddReference(str(usd_path))
        robot_prim.Load(Usd.LoadWithDescendants)
        for _ in range(5):
            self.simulation_app.update()
        tlr.apply_robot_config(stage, "/World/envs/env_0/Trex", self.robot_cfg, self.asset_dir)
        if self.self_pairs:
            apply_self_collision(stage, "/World/envs/env_0/Trex", self.self_pairs)
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
        cap = 1 << int(math.ceil(math.log2(max(16 * N, 1024))))  # measured need: ~4.4/env, ~14/env with robot.self_collision
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

        self.partners = self_contact_partners(self.self_pairs)
        wanted = {"left_foot", "right_foot", *self.ground_links, *self.partners}
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

    def _make_contact_view(self, prim_path: str, filter_paths: list[str] = ()):
        """Rigid-contact view over one link across all envs, plus the row->env permutation.
        `filter_paths` (env_0 links): per-partner forces from get_contact_force_matrix."""
        sim_view = SimulationManager.get_physics_sim_view()
        glob = lambda p: p.replace("/env_0/", "/env_*/")  # glob, not regex
        pattern = glob(prim_path)
        view = sim_view.create_rigid_contact_view(pattern, filter_patterns=[glob(p) for p in filter_paths],
                                                  max_contact_data_count=0)
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
        self.partner_views = {n: self._make_contact_view(self.link_prim_paths[n], [self.link_prim_paths[m] for m in ps])
                              for n, ps in self.partners.items()}

    def _as_torch(self, f) -> torch.Tensor:
        if isinstance(f, torch.Tensor):  # torch on GPU, numpy/warp on CPU
            return f
        return torch.as_tensor(f, device=self.device) if isinstance(f, np.ndarray) else _T(f)

    def _self_forces(self) -> dict[str, torch.Tensor]:
        """Link -> (N, M, 3) contact force from each of its self-collision partners, by env."""
        return {n: self._as_torch(v.get_contact_force_matrix(self.physics_dt)).reshape(self.num_envs, -1, 3)[p]
                for n, (v, p) in self.partner_views.items()}

    def _contact_force_of(self, view, perm, name: str = "", self_f: dict | None = None) -> torch.Tensor:
        f = self._as_torch(view.get_net_contact_forces(self.physics_dt))[perm]  # (N, 3)
        if self_f and name in self_f:  # floor share only: remove the force from the other leg
            f = f - self_f[name].sum(dim=1)
        return f.norm(dim=-1)

    def _ground_contact_force(self, self_f: dict | None = None) -> torch.Tensor:
        """(N, K) contact force [N] from the floor on each link that must stay off the ground."""
        return torch.stack([self._contact_force_of(v, p, n, self_f) for n, (v, p) in zip(self.ground_links, self.ground_views)], dim=1)

    def _foot_contact_force(self, self_f: dict | None = None) -> torch.Tensor:
        """(N, 2) floor contact force magnitude [N] on (left, right) foot, ordered by env."""
        return torch.stack([self._contact_force_of(v, p, n, self_f) for n, v, p in
                            zip(("left_foot", "right_foot"), self.contact_views, self.contact_perm)], dim=1)

    # ------------------------------------------------------------ backend hooks
    def _sim_dof_names(self):
        return self.robot.dof_names

    def _sim_link_names(self):
        return self.robot.link_names

    def _sim_dof_state(self):
        return _T(self.robot.get_dof_positions()), _T(self.robot.get_dof_velocities())

    def _sim_link_state(self):
        return _T(self.view.get_link_transforms()), _T(self.view.get_link_velocities())

    def _sim_joint_forces(self):
        return _T(self.robot.get_dof_projected_joint_forces())

    def _sim_contact_forces(self):
        if not self.self_pairs:
            return torch.cat([self._foot_contact_force(), self._ground_contact_force()], dim=1)
        sf = self._self_forces()
        pair = [sf[a][:, self.partners[a].index(b)].norm(dim=-1) for a, b in self.self_pairs]
        return torch.cat([self._foot_contact_force(sf), self._ground_contact_force(sf), torch.stack(pair, dim=1)], dim=1)

    def _sim_total_mass(self):
        return _T(self.view.get_masses()).sum(dim=1)

    def _sim_set_dof_targets(self, dof_tgt):
        # raw physics-tensor view in the hot path: the Articulation wrapper
        # adds per-call overhead that dominates at 8 substeps/step.
        self.view.set_dof_position_targets(_wp(dof_tgt), self.all_idx_wp)

    def _sim_dof_positions_now(self):
        return _T(self.view.get_dof_positions())

    def _sim_write_dof_positions(self, q):
        self.view.set_dof_positions(_wp(q), self.all_idx_wp)

    def _sim_dof_velocities_now(self):
        return _T(self.view.get_dof_velocities())

    def _dr_set_armatures(self, a):
        self.robot.set_dof_armatures(_wp(a))

    def _sim_base_pose_vel_now(self):
        lt = _T(self.view.get_link_transforms())  # (N, L, 7)
        w_w = _T(self.view.get_link_velocities())[:, self.base_i, 3:6]
        return lt[:, :, 0:3], lt[:, self.base_i, 3:7], w_w

    def _sim_apply_base_wrench(self, force_w, torque_w, link_pos):
        force = torch.zeros(self.num_envs, self.link_count, 3, device=self.device)
        torque = torch.zeros_like(force)
        force[:, self.base_i] = force_w
        torque[:, self.base_i] = torque_w
        self.view.apply_forces_and_torques_at_position(
            _wp(force), _wp(torque), _wp(link_pos.contiguous()), self.all_idx_wp, True)

    def _sim_root_velocities(self):
        return _T(self.view.get_root_velocities())

    def _sim_set_root_velocities(self, ids, lin, ang):
        idx = wp.array(ids.int().cpu().numpy(), dtype=wp.int32, device=self.device)
        self.robot.set_velocities(_wp(lin), _wp(ang), indices=idx)

    def _sim_reset_state(self, ids, pos, quat_wxyz, q0):
        n, dev = len(ids), self.device
        idx = wp.array(ids.int().cpu().numpy(), dtype=wp.int32, device=dev)
        self.robot.set_world_poses(_wp(pos), _wp(quat_wxyz), indices=idx)
        zeros6 = torch.zeros(n, 3, device=dev)
        self.robot.set_velocities(_wp(zeros6), _wp(zeros6), indices=idx)
        self.robot.set_dof_positions(_wp(q0), indices=idx)
        self.robot.set_dof_velocities(_wp(torch.zeros(n, self.num_dofs, device=dev)), indices=idx)
        self.robot.set_dof_position_targets(_wp(q0), indices=idx)

    def _dr_set_friction(self, f):
        # Material setters are CPU-only in the GPU pipeline (tensors AND indices).
        mp = _T(self.view.get_material_properties()).clone().cpu()  # (N, shapes, 3)
        mp[..., 0:2] = f.unsqueeze(-1).cpu()
        self.view.set_material_properties(_wp(mp), wp.array(np.arange(self.num_envs, dtype=np.int32), device="cpu"))

    def _dr_link_masses(self):
        return _T(self.robot.get_link_masses())

    def _dr_set_link_masses(self, m):
        self.robot.set_link_masses(_wp(m))

    def _dr_link_inertias(self):
        return _T(self.robot.get_link_inertias())

    def _dr_set_link_inertias(self, i):
        self.robot.set_link_inertias(_wp(i))

    def _dr_base_com(self):
        pos, _ = self.robot.get_link_coms(link_indices=[self.base_i])  # (N, 1, 3)
        return _T(pos)

    def _dr_set_base_com(self, p):
        self.robot.set_link_coms(_wp(p), link_indices=[self.base_i])

    def _dr_dof_gains(self):
        kp, kd = self.robot.get_dof_gains()
        return _T(kp), _T(kd)

    def _dr_set_dof_gains(self, kp, kd):
        self.robot.set_dof_gains(_wp(kp), _wp(kd))

    def _dr_max_efforts(self):
        return _T(self.robot.get_dof_max_efforts())

    def _dr_set_max_efforts(self, mf):
        self.robot.set_dof_max_efforts(_wp(mf))

    def _sim_set_dof_efforts(self, e):
        self.robot.set_dof_efforts(_wp(e))

    # -------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        self._pre_step(actions)
        for i in range(self.decimation):
            self._sim_set_dof_targets(self._substep_dof_targets(i))
            self._enforce_ankle_linkage()
            if self.harness_alpha > 0.0:
                self._apply_harness()
            # render mode: push physics results to Fabric every substep, otherwise the
            # rendered robot lags the physics state (camera followed the *physics* pose
            # and the visual robot walked out of frame).
            SimulationManager.step(steps=1, update_fabric=self._fabric_sync)
        reward, terminated, timeout, extras = self._post_step()
        done = terminated | timeout
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.reset_idx(reset_ids)
        return self._observations(), reward, done, extras
