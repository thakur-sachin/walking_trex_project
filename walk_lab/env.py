"""Trex walking task on Isaac Lab 3.0 (DirectRLEnv, PhysX backend), Isaac Sim 6.0.1.

The task (servo model, rewards, observations, commands, curricula, randomization) is the
shared walk_rl/task_core.py, so this runs the *same* task as the Isaac Sim stack
(walk_rl/env.py). What Isaac Lab provides here: simulation context, InteractiveScene
cloning, the Articulation asset + implicit actuators, contact-sensor views, external
wrench composer, gymnasium registration and rsl_rl integration.

Two entry points, one object graph:
  TrexLabWalkEnv(num_envs, task_cfg, device)   same Python API as walk_rl.env.TrexWalkEnv
      (step/reset/obs dict/kick/set_harness/...), so walk_rl's PPO, evaluate and
      trace scripts run on Isaac Lab unchanged.
  TrexLabDirectEnv(cfg)                         the gymnasium/DirectRLEnv side, used by
      Isaac Lab's rsl_rl train script ("Trex-Walk-Direct-v0"); builds its own
      TrexLabWalkEnv as `.task`.

Physics parity with the Isaac Sim stack (checked with walk_lab/physics_dump.py):
  - robot: _converted_usd/trex_lab/trex_lab.usd baked by walk_lab/bake_usd.py with the
    same robot_config application; actuators take stiffness/damping/effort from USD.
  - ground: the same isaacsim GroundPlane; scene settings at PhysX schema defaults
    (Isaac Lab would otherwise author e.g. bounce threshold 0.5 m/s).
  - contact forces read raw from PhysX contact views (Isaac Lab's ContactSensor buffers
    are zeroed on reset, which would defeat the stale-contact handling in task_core).

Import after an AppLauncher / SimulationApp exists.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from walk_rl.task_core import (  # noqa: E402
    DEFAULT_TASK_PATH,
    TrexTaskCore,
    harness_alpha_at,
    load_task_config,
    obs_dims,
    push_scale_at,
)

ROBOT_CONFIG_PATH = PROJECT_ROOT / "assets" / "robots" / "trex" / "robot_config.yaml"
ROBOT_USD = PROJECT_ROOT / "_converted_usd" / "trex_lab" / "trex_lab.usd"
ROBOT_PRIM = "/World/envs/env_.*/Robot"


def _baked_body_paths() -> dict[str, str]:
    """Rigid-body name -> path relative to the robot root in the baked USD (bodies are
    nested: base_link/left_femur/left_tibia/left_ankle/left_foot). Also refuses a bake
    older than robot_config.yaml / the URDF."""
    from pxr import Usd, UsdPhysics

    sys.path.insert(0, str(PROJECT_ROOT / "walk_lab"))
    from bake_usd import config_hash  # noqa: E402  (pure function; bake_usd's app only starts under __main__)

    assert ROBOT_USD.exists(), f"{ROBOT_USD} missing: run walk_lab/bake_usd.py"
    st = Usd.Stage.Open(str(ROBOT_USD))
    baked = st.GetRootLayer().customLayerData.get("trex_robot_config_hash")
    assert baked == config_hash(), (f"{ROBOT_USD} was baked from a different robot_config/URDF "
                                    f"({baked} != {config_hash()}): re-run walk_lab/bake_usd.py")
    root = st.GetDefaultPrim()
    out = {}
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            out[p.GetName()] = str(p.GetPath())[len(str(root.GetPath())):]
    return out


@configclass
class TrexLabEnvCfg(DirectRLEnvCfg):
    """Filled from walk_task.yaml by make_env_cfg(); the defaults are placeholders."""
    task_yaml: str = str(DEFAULT_TASK_PATH)  # merged task config (defaults + experiment override)
    decimation = 8
    episode_length_s = 20.0
    action_space = 8
    observation_space = 105
    state_space = 50
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=8, use_fabric=False,
                                       physics=PhysxCfg(bounce_threshold_velocity=0.0))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path=ROBOT_PRIM,
        # activate_contact_sensors stays off: Isaac Lab's version stops at the first rigid body
        # (our URDF import nests them: base_link/left_femur/...) and also zeroes every body's
        # sleep threshold. TrexLabDirectEnv._setup_scene applies the reporters like walk_rl.
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD), activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
        # stiffness/damping/effort/velocity limits None = take the USD values, i.e. the
        # importer gains and robot_config torque caps baked by bake_usd.py
        actuators={"servos": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )


def make_env_cfg(task_cfg: SimpleNamespace, task_yaml: str | Path, num_envs: int, device: str,
                 render: bool = False) -> TrexLabEnvCfg:
    c = task_cfg
    physics_dt = 1.0 / c.sim.physics_hz
    decimation = c.sim.physics_hz // c.sim.control_hz
    _, n_actor, n_critic = obs_dims(c)
    # GPU broadphase/aggregate buffers sized exactly as walk_rl/env.py does
    cap = 1 << int(math.ceil(math.log2(max(8 * num_envs, 1024))))
    cfg = TrexLabEnvCfg()
    cfg.task_yaml = str(task_yaml)
    cfg.decimation = decimation
    # Lab's own episode timeout is never used: task_core ends episodes (its time_outs) itself
    cfg.episode_length_s = c.sim.episode_length_s + 10.0
    cfg.observation_space = n_actor
    cfg.state_space = n_critic
    cfg.sim = SimulationCfg(
        dt=physics_dt, device=device, render_interval=decimation, use_fabric=render,
        physics=PhysxCfg(
            bounce_threshold_velocity=0.0,  # PhysX schema default (what walk_rl runs with)
            # PhysX schema defaults too (Isaac Lab's are 16x / 2x larger): identical contact
            # buffers -> identical behaviour and a fair memory comparison
            gpu_max_rigid_contact_count=524288,
            gpu_max_rigid_patch_count=81920,
            gpu_found_lost_aggregate_pairs_capacity=cap,
            gpu_total_aggregate_pairs_capacity=cap,
            gpu_found_lost_pairs_capacity=max(262144, 64 * num_envs),
        ),
    )
    cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=c.sim.env_spacing_m, replicate_physics=True)
    cfg.robot = cfg.robot.replace(init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, c.sim.spawn_height_m)))
    cfg.seed = None
    return cfg


class TrexLabDirectEnv(DirectRLEnv):
    """Isaac Lab DirectRLEnv: owns the scene and the step loop; delegates the task to
    `self.task` (a TrexLabWalkEnv)."""

    cfg: TrexLabEnvCfg

    def __init__(self, cfg: TrexLabEnvCfg, render_mode: str | None = None, task=None, **kwargs):
        self.body_paths = _baked_body_paths()
        self._task_cfg = load_task_config(cfg.task_yaml)
        super().__init__(cfg, render_mode, **kwargs)
        self._substep = 0
        # Env-side curriculum + metrics, for trainers without per-iteration hooks (rsl_rl):
        # enable_curriculum(steps_per_iter, log_path). walk_rl's PPO drives the task directly.
        self._steps_per_iter = None
        self._log_path = None
        # gym/rsl_rl path: build the task here. TrexLabWalkEnv path: it attaches itself.
        self.task = task if task is not None else TrexLabWalkEnv(lab_env=self, task_cfg=self._task_cfg)

    def _setup_scene(self):
        from pxr import PhysicsSchemaTools

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        tc = self._task_cfg
        self._contact_links = ["left_foot", "right_foot", *tc.termination.ground_contact_links]
        # contact reports on exactly the links walk_rl marks (env_0, before cloning), threshold 0
        from pxr import PhysxSchema
        for name in self._contact_links:
            prim = self.sim.stage.GetPrimAtPath(f"/World/envs/env_0/Robot{self.body_paths[name]}")
            assert prim.IsValid() and not prim.IsInstanceProxy(), f"rigid body {name} not found / instance proxy"
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        self._contacts = {}
        for name in self._contact_links:
            s = ContactSensor(ContactSensorCfg(prim_path=f"{ROBOT_PRIM}{self.body_paths[name]}"))
            self._contacts[name] = s
            self.scene.sensors[f"contact_{name}"] = s
        # same ground as walk_rl: isaacsim GroundPlane(sizes=1000) is exactly this call plus
        # visual templates (that extension is not loaded by Isaac Lab's app launcher)
        PhysicsSchemaTools.addGroundPlane(self.sim.stage, "/World/GroundPlane", "Z", 500.0,
                                          (0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])

    # ---- step loop hooks (DirectRLEnv.step order: pre, decimation x [apply, write, step,
    # update], dones, rewards, reset, observations)
    def enable_curriculum(self, steps_per_iter: int, log_path: str | Path | None = None) -> None:
        """Harness/push schedules by iteration = common_step_counter // steps_per_iter (the same
        functions walk_rl's PPO uses), and one train_log.jsonl record per iteration with
        walk_rl's keys so tb_sync.py and the platform comparison read every run alike."""
        import time

        self._steps_per_iter = int(steps_per_iter)
        self._log_path = Path(log_path) if log_path else None
        self._it_rew, self._it_n = 0.0, 0
        self._t0 = self._t_it = time.time()

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._steps_per_iter:
            it = self.common_step_counter // self._steps_per_iter
            self.task.set_harness(harness_alpha_at(self._task_cfg, it))
            self.task.set_push_scale(push_scale_at(self._task_cfg, it))
        self.task._pre_step(actions)
        self._substep = 0

    def _apply_action(self):
        t = self.task
        self._robot.set_joint_position_target_index(target=t._substep_dof_targets(self._substep))
        t._enforce_ankle_linkage()
        if t.harness_alpha > 0.0:
            t._apply_harness()
        self._substep += 1

    def _get_dones(self):
        self._reward, terminated, timeout, self._task_extras = self.task._post_step()
        self.extras["time_outs"] = self._task_extras["time_outs"]
        self.extras.pop("log", None)  # extras persists across steps; log a record once
        if self._steps_per_iter:
            self._log_iteration()
        return terminated, timeout & ~terminated

    def _log_iteration(self) -> None:
        import json
        import time

        self._it_rew += self._reward.mean().item()
        self._it_n += 1
        if self.common_step_counter % self._steps_per_iter:
            return
        it = self.common_step_counter // self._steps_per_iter  # iterations completed
        now = time.time()
        ep = self.task.pop_episode_stats()
        rec = {"iter": it, "mean_step_reward": self._it_rew / max(self._it_n, 1),
               "samples": it * self.num_envs * self._steps_per_iter, "t_total": now - self._t_it,
               "elapsed_min": (now - self._t0) / 60.0, "harness_alpha": self.task.harness_alpha,
               "push_scale": self.task.push_scale, **ep}
        self.extras["log"] = {f"Trex/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}
        if self._log_path is not None:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self._it_rew, self._it_n, self._t_it = 0.0, 0, now

    def _get_rewards(self):
        return self._reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        self.task.reset_idx(env_ids.long())

    def _get_observations(self):
        o = self.task._observations()
        return {"policy": o["actor"], "critic": o["critic"]}


class TrexLabWalkEnv(TrexTaskCore):
    """Isaac Lab backend for the shared task, with walk_rl.env.TrexWalkEnv's API."""

    def __init__(self, num_envs: int | None = None, task_cfg: SimpleNamespace | None = None,
                 device: str = "cuda:0", simulation_app=None, render: bool = False, seed: int = 0,
                 lab_env: TrexLabDirectEnv | None = None, task_yaml: str | Path | None = None):
        self.robot_cfg = RobotConfig.load(ROBOT_CONFIG_PATH)
        if lab_env is None:
            torch.manual_seed(seed)
            self.cfg = task_cfg or load_task_config()
            # the DirectRLEnv reloads the task config from a file: write the merged one
            if task_yaml is None:
                import tempfile

                import yaml

                from walk_rl.task_core import ns_to_dict
                f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
                yaml.safe_dump(ns_to_dict(self.cfg), f)
                f.close()
                task_yaml = f.name
            lab_env = TrexLabDirectEnv(make_env_cfg(self.cfg, task_yaml, num_envs, device, render), task=self)
            self._own_lab = True
        else:
            self.cfg = task_cfg
            self._own_lab = False
        self.lab = lab_env
        self.num_envs = lab_env.num_envs
        self.device = str(lab_env.device)
        self.robot = lab_env._robot
        self.view = self.robot.root_view  # raw PhysX articulation view (same API walk_rl uses)
        self.env_origins = lab_env.scene.env_origins.clone()

        self._setup_timing()
        assert abs(self.physics_dt - lab_env.physics_dt) < 1e-12 and self.decimation == lab_env.cfg.decimation
        self.base_body = None
        self._contact_views = []
        for name in lab_env._contact_links:
            v = lab_env._contacts[name].contact_view
            paths = list(v.sensor_paths)
            env_of = [int(p.split("/env_")[1].split("/")[0]) for p in paths]
            assert sorted(env_of) == list(range(self.num_envs)), f"contact view {name}: unexpected sensors"
            perm = torch.empty(self.num_envs, dtype=torch.long)
            for row, e in enumerate(env_of):
                perm[e] = row
            self._contact_views.append((v, perm.to(self.device)))
        self._init_buffers()
        self._base_ids = torch.tensor([self.base_i], dtype=torch.int32, device=self.device)
        self._all_idx_wp = wp.array(np.arange(self.num_envs, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._randomize_startup()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------ backend hooks
    def _sim_dof_names(self):
        return list(self.robot.joint_names)

    def _sim_link_names(self):
        return list(self.robot.body_names)

    def _sim_dof_state(self):
        d = self.robot.data
        return d.joint_pos.torch.clone(), d.joint_vel.torch.clone()

    def _sim_link_state(self):
        d = self.robot.data
        # body_com_vel_w is PhysX get_link_velocities(), the quantity walk_rl reads
        return d.body_link_pose_w.torch.clone(), d.body_com_vel_w.torch.clone()

    def _sim_joint_forces(self):
        return wp.to_torch(self.view.get_dof_projected_joint_forces()).clone()

    def _sim_contact_forces(self):
        out = []
        for v, perm in self._contact_views:
            f = v.get_net_contact_forces(self.physics_dt)
            f = f if isinstance(f, torch.Tensor) else (torch.as_tensor(f, device=self.device)
                                                       if isinstance(f, np.ndarray) else wp.to_torch(f))
            out.append(f.reshape(self.num_envs, 3).norm(dim=-1)[perm])
        return torch.stack(out, dim=1)

    def _sim_total_mass(self):
        return self.robot.data.body_mass.torch.sum(dim=1).clone()

    def _sim_set_dof_targets(self, dof_tgt):
        self.robot.set_joint_position_target_index(target=dof_tgt)

    def _sim_dof_positions_now(self):
        return self.robot.data.joint_pos.torch

    def _sim_write_dof_positions(self, q):
        self.robot.write_joint_position_to_sim_index(position=q.contiguous(), full_data=True)

    def _sim_base_pose_vel_now(self):
        d = self.robot.data
        pose = d.body_link_pose_w.torch
        return pose[:, :, 0:3], pose[:, self.base_i, 3:7], d.body_com_vel_w.torch[:, self.base_i, 3:6]

    def _sim_apply_base_wrench(self, force_w, torque_w, link_pos):
        # The same PhysX call as walk_rl (world-frame force at the link origin). Isaac Lab's
        # wrench composer ends in this call too, but converts to the body frame first; that
        # rounding was the only difference between the stacks in trace tests (bit-identical
        # without the harness). No composer is active, so write_data_to_sim leaves external
        # forces alone, and PhysX clears them after the step, as in walk_rl.
        force = torch.zeros(self.num_envs, self.link_count, 3, device=self.device)
        torque = torch.zeros_like(force)
        force[:, self.base_i] = force_w
        torque[:, self.base_i] = torque_w
        self.view.apply_forces_and_torques_at_position(
            wp.from_torch(force.contiguous()), wp.from_torch(torque.contiguous()),
            wp.from_torch(link_pos.contiguous()), self._all_idx_wp, True)

    def _sim_root_velocities(self):
        return self.robot.data.root_com_vel_w.torch.clone()

    def _sim_set_root_velocities(self, ids, lin, ang):
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.cat([lin, ang], dim=1).contiguous(), env_ids=ids.int())

    def _sim_reset_state(self, ids, pos, quat_wxyz, q0):
        n, dev, r = len(ids), self.device, self.robot
        ids32 = ids.int()
        quat_xyzw = torch.cat([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], dim=1)
        r.write_root_link_pose_to_sim_index(root_pose=torch.cat([pos, quat_xyzw], dim=1).contiguous(), env_ids=ids32)
        r.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids32)
        r.write_joint_position_to_sim_index(position=q0.contiguous(), env_ids=ids32)
        r.write_joint_velocity_to_sim_index(velocity=torch.zeros(n, self.num_dofs, device=dev), env_ids=ids32)
        r.set_joint_position_target_index(target=q0.contiguous(), env_ids=ids32)

    def _dr_set_friction(self, f):
        # same PhysX tensor call as walk_rl (Isaac Lab's randomize_rigid_body_material uses it too)
        mp = wp.to_torch(self.view.get_material_properties()).clone().cpu()
        mp[..., 0:2] = f.unsqueeze(-1).cpu()
        self.view.set_material_properties(wp.from_torch(mp.contiguous()),
                                          wp.array(np.arange(self.num_envs, dtype=np.int32), device="cpu"))

    def _dr_link_masses(self):
        return self.robot.data.body_mass.torch.clone()

    def _dr_set_link_masses(self, m):
        self.robot.set_masses_index(masses=m.contiguous(), full_data=True)

    def _dr_base_com(self):
        return self.robot.data.body_com_pose_b.torch[:, self.base_i:self.base_i + 1, 0:3].clone()

    def _dr_set_base_com(self, p):
        pose = self.robot.data.body_com_pose_b.torch[:, self.base_i:self.base_i + 1].clone()
        pose[..., 0:3] = p
        self.robot.set_coms_index(coms=pose.contiguous(), body_ids=self._base_ids)

    def _dr_dof_gains(self):
        d = self.robot.data
        return d.joint_stiffness.torch.clone(), d.joint_damping.torch.clone()

    def _dr_set_dof_gains(self, kp, kd):
        self.robot.write_joint_stiffness_to_sim_index(stiffness=kp.contiguous(), full_data=True)
        self.robot.write_joint_damping_to_sim_index(damping=kd.contiguous(), full_data=True)

    def _dr_max_efforts(self):
        return self.robot.data.joint_effort_limits.torch.clone()

    def _dr_set_max_efforts(self, mf):
        self.robot.write_joint_effort_limit_to_sim_index(limits=mf.contiguous(), full_data=True)

    # ------------------------------------------------- walk_rl-compatible API
    def step(self, actions: torch.Tensor):
        obs, rew, terminated, truncated, extras = self.lab.step(actions)
        return {"actor": obs["policy"], "critic": obs["critic"]}, rew, terminated | truncated, \
            {"time_outs": extras["time_outs"]}

    def reset(self) -> dict:
        obs, _ = self.lab.reset()
        return {"actor": obs["policy"], "critic": obs["critic"]}

    def close(self) -> None:
        if self._own_lab:
            self.lab.close()
