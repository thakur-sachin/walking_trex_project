"""Trex walking task on Isaac Lab 3 (main, 3.0.0) with the Newton physics engine (MJWarp solver), kit-less.

The task (servo model, rewards, observations, commands, curricula, randomization) is the shared
walk_rl/task_core.py, as on the Isaac Sim stack (walk_rl/env.py) and the Isaac Lab / PhysX port
(walk_lab/env.py). The robot is lab3/export_mjcf.py's USD: the MuJoCo sim2sim model (pi_runner/mujoco_backend)
converted with the MJCF importer, so training and the sim2sim check run the same MuJoCo physics.

Differences from the PhysX stacks:
  - ankle linkage: a fixed tendon with a +-1e-4 rad length limit (ankle - femur + tibia), solved by MJWarp as a
    constraint (Newton has no tendon equality). task_core's teleport is off (robot_config ankle sim_model is
    overridden to "constraint"), so the femur/tibia servos carry the foot's load through the linkage, as on the robot.
  - friction: MuJoCo takes the larger of two geoms' coefficients, PhysX (walk_rl) their mean. The ground gets 0
    and the robot geoms the PhysX mean (randomized robot value + ground value), so the effective coefficient
    and the domain-randomization range keep their meaning.
  - contacts: Isaac Lab ContactSensor (Newton), one per link task_core watches.

Run with the Isaac Lab 3 venv:  ~/trex_stack/IsaacLab_dev/.venv/bin/python ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

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
ROBOT_USD = PROJECT_ROOT / "outputs" / "lab3" / "trex" / "usd" / "trex" / "trex.usda"
ROBOT_PRIM = "/World/envs/env_.*/Robot"
PHYSX_DEFAULT_GROUND_FRICTION = 0.5  # walk_rl's ground material when sim.ground_friction is null


def _body_paths() -> dict[str, str]:
    """Rigid-body name -> path below the robot root in the exported USD (bodies are nested)."""
    from pxr import Usd, UsdPhysics

    assert ROBOT_USD.exists(), f"{ROBOT_USD} missing: run lab3/export_mjcf.py"
    st = Usd.Stage.Open(str(ROBOT_USD))
    root = st.GetDefaultPrim()
    root.GetVariantSet("Physics").SetVariantSelection("mujoco")
    out = {}
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            out[p.GetName()] = str(p.GetPath())[len(str(root.GetPath())):]
    return out


@configclass
class TrexSceneCfg(InteractiveSceneCfg):
    """Ground + robot; make_env_cfg adds one ContactSensorCfg per watched link (contact_<link>)."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.0, dynamic_friction=0.0), debug_vis=False)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD), variants={"Physics": "mujoco"}),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
        # lx16a servo model: task_core writes joint torques; no engine drive on any joint
        actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
                                                 effort_limit_sim=100.0)},
    )


@configclass
class TrexLab3EnvCfg(DirectRLEnvCfg):
    """Filled from walk_task.yaml by make_env_cfg(); the defaults are placeholders."""
    task_yaml: str = str(DEFAULT_TASK_PATH)
    decimation = 8
    episode_length_s = 20.0
    action_space = 8
    observation_space = 105
    state_space = 50
    sim: SimulationCfg = SimulationCfg(dt=1 / 200)
    scene: TrexSceneCfg = TrexSceneCfg(num_envs=1024, env_spacing=1.0, replicate_physics=True)


def make_env_cfg(task_cfg: SimpleNamespace, task_yaml: str | Path, num_envs: int, device: str) -> TrexLab3EnvCfg:
    c = task_cfg
    assert getattr(getattr(c, "actuator", None), "model", "implicit_pd") == "lx16a", \
        "walk_lab3 runs the lx16a servo model only (no engine PD drive)"
    physics_dt = 1.0 / c.sim.physics_hz
    decimation = c.sim.physics_hz // c.sim.control_hz
    _, n_actor, n_critic = obs_dims(c)
    cfg = TrexLab3EnvCfg()
    cfg.task_yaml = str(task_yaml)
    cfg.decimation = decimation
    cfg.episode_length_s = c.sim.episode_length_s + 10.0  # task_core ends episodes itself
    cfg.observation_space = n_actor
    cfg.state_space = n_critic
    cfg.sim = SimulationCfg(
        dt=physics_dt, device=device, render_interval=decimation,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=64, nconmax=32, iterations=100, ls_iterations=50, cone="pyramidal",
                                       impratio=1.0, integrator="implicitfast", use_mujoco_contacts=False),
            num_substeps=1,
        ),
    )
    cfg.scene = TrexSceneCfg(num_envs=num_envs, env_spacing=c.sim.env_spacing_m, replicate_physics=True)
    cfg.scene.robot = cfg.scene.robot.replace(
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, c.sim.spawn_height_m)))
    paths = _body_paths()
    for name in contact_links(c):
        setattr(cfg.scene, f"contact_{name}",
                ContactSensorCfg(prim_path=f"{ROBOT_PRIM}{paths[name]}", update_period=0.0, history_length=1))
    cfg.seed = None
    return cfg


def contact_links(task_cfg) -> list[str]:
    return ["left_foot", "right_foot", *task_cfg.termination.ground_contact_links]


class TrexLab3DirectEnv(DirectRLEnv):
    """DirectRLEnv: owns the scene and the step loop; delegates the task to `self.task` (TrexLab3WalkEnv)."""

    cfg: TrexLab3EnvCfg

    def __init__(self, cfg: TrexLab3EnvCfg, render_mode: str | None = None, task=None, **kwargs):
        self._task_cfg = load_task_config(cfg.task_yaml)
        super().__init__(cfg, render_mode, **kwargs)
        self._robot = self.scene["robot"]
        self._contact_links = contact_links(self._task_cfg)
        self._contacts = {n: self.scene[f"contact_{n}"] for n in self._contact_links}
        self._substep = 0
        self._steps_per_iter = None
        self._log_path = None
        self.task = task if task is not None else TrexLab3WalkEnv(lab_env=self, task_cfg=self._task_cfg)

    def enable_curriculum(self, steps_per_iter: int, log_path: str | Path | None = None) -> None:
        """As walk_lab: harness/push schedules by iteration, one train_log.jsonl record per iteration."""
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
        t._substep_dof_targets(self._substep)  # advances the servo ramp state; no engine drive
        t._enforce_ankle_linkage()  # "constraint": only writes the servo torques
        if t.harness_alpha > 0.0:
            t._apply_harness()
        self._substep += 1

    def _get_dones(self):
        self._reward, terminated, timeout, self._task_extras = self.task._post_step()
        self.extras["time_outs"] = self._task_extras["time_outs"]
        self.extras.pop("log", None)
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
        it = self.common_step_counter // self._steps_per_iter
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


class TrexLab3WalkEnv(TrexTaskCore):
    """Isaac Lab 3 / Newton backend for the shared task, with walk_rl.env.TrexWalkEnv's API."""

    def __init__(self, num_envs: int | None = None, task_cfg: SimpleNamespace | None = None,
                 device: str = "cuda:0", simulation_app=None, render: bool = False, seed: int = 0,
                 lab_env: TrexLab3DirectEnv | None = None, task_yaml: str | Path | None = None):
        self.robot_cfg = RobotConfig.load(ROBOT_CONFIG_PATH)
        self.robot_cfg.ankle_sim_model = "constraint"  # the tendon limit holds the linkage (no teleport)
        if lab_env is None:
            torch.manual_seed(seed)
            self.cfg = task_cfg or load_task_config()
            if task_yaml is None:  # the DirectRLEnv reloads the task config from a file: write the merged one
                import tempfile

                import yaml

                from walk_rl.task_core import ns_to_dict
                f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
                yaml.safe_dump(ns_to_dict(self.cfg), f)
                f.close()
                task_yaml = f.name
            lab_env = TrexLab3DirectEnv(make_env_cfg(self.cfg, task_yaml, num_envs, device), task=self)
            self._own_lab = True
        else:
            self.cfg = task_cfg
            self._own_lab = False
        self.lab = lab_env
        self.num_envs = lab_env.num_envs
        self.device = str(lab_env.device)
        self.robot = lab_env._robot
        self.env_origins = lab_env.scene.env_origins.clone()

        self._setup_timing()
        assert abs(self.physics_dt - lab_env.physics_dt) < 1e-12 and self.decimation == lab_env.cfg.decimation
        self.base_body = None
        g = getattr(self.cfg.sim, "ground_friction", None)
        self._ground_mu = PHYSX_DEFAULT_GROUND_FRICTION if g is None else float(g)
        model = lab_env.sim.backend_manager.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        tc = getattr(self.cfg.sim, "newton_contact_timeconst_s", None)
        if tc is not None:  # every shape, ground included: MuJoCo mixes the two geoms' solref (solmix 1:1)
            self._set_contact_timeconst(model, max(float(tc), 2.0 * self.physics_dt))
        self._init_buffers()
        self._base_ids = torch.tensor([self.base_i], dtype=torch.int32, device=self.device)
        self._set_robot_mu(torch.full((self.num_envs, 1), 1.0, device=self.device))  # nominal; DR may override
        self._randomize_startup()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _set_contact_timeconst(self, model, tc: float) -> None:
        """MuJoCo contact solref (timeconst, dampratio 1) on every shape. Default: the robot geoms carry the
        exported 2 / 200 Hz = 0.01 s and Isaac Lab's ground 0.02 s, mixed to ~0.015 s, against 0.004 s in the
        MuJoCo sim2sim check (pi_runner/mujoco_backend). MuJoCo needs timeconst >= 2 physics steps."""
        from newton import ModelFlags  # noqa: PLC0415

        s = wp.to_torch(model.mujoco.solref)
        s[:, 0] = tc
        s[:, 1] = 1.0
        self.lab.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        print(f"[walk_lab3] contact solref timeconst {tc:.4f} s on {s.shape[0]} shapes", flush=True)

    def _set_robot_mu(self, f: torch.Tensor) -> None:
        """f (N, 1): the robot's PhysX friction value; MuJoCo gets the PhysX mean with the ground's."""
        from newton import ModelFlags  # noqa: PLC0415

        self._mu_view[:] = (0.5 * (f + self._ground_mu)).expand_as(self._mu_view)
        self.lab.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

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
        return d.body_link_pose_w.torch.clone(), d.body_com_vel_w.torch.clone()

    def _sim_joint_forces(self):
        # only the lx16a servo model runs here, and task_core takes its torque from the model itself
        return torch.zeros(self.num_envs, self.num_dofs, device=self.device)

    def _sim_contact_forces(self):
        out = [self.lab._contacts[n].data.net_forces_w.torch[:, 0].norm(dim=-1) for n in self.lab._contact_links]
        return torch.stack(out, dim=1)

    def _sim_total_mass(self):
        return self.robot.data.body_mass.torch.sum(dim=1).clone()

    def _sim_set_dof_targets(self, dof_tgt):
        self.robot.set_joint_position_target_index(target=dof_tgt)

    def _sim_set_dof_efforts(self, e):
        self.robot.set_joint_effort_target_index(target=e)

    def _sim_dof_positions_now(self):
        return self.robot.data.joint_pos.torch

    def _sim_write_dof_positions(self, q):
        self.robot.write_joint_position_to_sim_index(position=q.contiguous())

    def _sim_dof_velocities_now(self):
        return self.robot.data.joint_vel.torch

    def _dr_set_armatures(self, a):
        self.robot.write_joint_armature_to_sim_index(armature=a.contiguous())

    def _sim_base_pose_vel_now(self):
        d = self.robot.data
        pose = d.body_link_pose_w.torch
        return pose[:, :, 0:3], pose[:, self.base_i, 3:7], d.body_com_vel_w.torch[:, self.base_i, 3:6]

    def _sim_apply_base_wrench(self, force_w, torque_w, link_pos):
        force = torch.zeros(self.num_envs, self.link_count, 3, device=self.device)
        torque = torch.zeros_like(force)
        force[:, self.base_i] = force_w
        torque[:, self.base_i] = torque_w
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=force, torques=torque, positions=link_pos, is_global=True)

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

    def _dr_set_friction(self, f):
        self._set_robot_mu(f)

    def _dr_link_masses(self):
        return self.robot.data.body_mass.torch.clone()

    def _dr_set_link_masses(self, m):
        self.robot.set_masses_index(masses=m.contiguous())

    def _dr_link_inertias(self):
        return self.robot.data.body_inertia.torch.clone()

    def _dr_set_link_inertias(self, i):
        self.robot.set_inertias_index(inertias=i.contiguous())

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
        self.robot.write_joint_stiffness_to_sim_index(stiffness=kp.contiguous())
        self.robot.write_joint_damping_to_sim_index(damping=kd.contiguous())

    def _dr_max_efforts(self):
        return self.robot.data.joint_effort_limits.torch.clone()

    def _dr_set_max_efforts(self, mf):
        self.robot.write_joint_effort_limit_to_sim_index(limits=mf.contiguous())

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
