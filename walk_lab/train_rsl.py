"""Train the Trex walking task on Isaac Lab with rsl_rl (Isaac Lab's standard PPO trainer).

    conda activate isaaclab6; export OMNI_KIT_ACCEPT_EULA=YES
    python walk_lab/train_rsl.py --name v8e_lab_rsl --iterations 3000 --task walk_rl/experiments/v8e_long_16k.yaml

Same task (walk_rl/task_core.py), same harness/push curricula (env-side, by iteration) and
the same network sizes as walk_rl's PPO (actor 128-128-64 ELU: must run on the Pi 4;
critic 256-128-64; init std exp(ppo.init_log_std); observation normalization). The PPO
itself is rsl_rl's with its standard Isaac Lab settings (adaptive LR on a KL target), so
this run measures "the Isaac Lab stack as normally used". For the simulator-only
comparison use walk_rl/train.py --backend lab (walk_rl's PPO on the Isaac Lab env).

Outputs (outputs/walk_runs/<name>/): rsl_rl/ (rsl_rl checkpoints + TensorBoard),
train_log.jsonl (walk_rl keys, written by the env), and at the end checkpoint.pt +
policy_numpy.npz converted to walk_rl's format, so evaluate.py judges it like any run.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--task", default=str(PROJECT_ROOT / "walk_rl" / "walk_task.yaml"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

simulation_app = AppLauncher(headless=True, device=args.device).app

import math  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

try:
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
        handle_deprecated_rsl_rl_cfg,
    )
    from rsl_rl.runners import OnPolicyRunner

    from walk_lab.env import TrexLabDirectEnv, make_env_cfg
    from walk_rl.ppo import ActorCritic, export_numpy_policy
    from walk_rl.task_core import load_task_config, ns_to_dict

    cfg = load_task_config(args.task)
    p = cfg.ppo
    num_envs = args.num_envs or p.num_envs
    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    merged = run_dir / "walk_task.yaml"
    with open(merged, "w") as f:
        yaml.safe_dump(ns_to_dict(cfg), f, sort_keys=False)
    (run_dir / "train_log.jsonl").unlink(missing_ok=True)

    env_cfg = make_env_cfg(cfg, merged, num_envs, args.device)
    env_cfg.seed = args.seed
    env = TrexLabDirectEnv(env_cfg)
    env.enable_curriculum(p.rollout_steps, run_dir / "train_log.jsonl")
    task = env.task
    print(f"[rsl] envs={num_envs} actor_obs={task.num_actor_obs} critic_obs={task.num_critic_obs} "
          f"DR={task.dr_active}", flush=True)

    agent = RslRlOnPolicyRunnerCfg(
        seed=args.seed, device=args.device, num_steps_per_env=p.rollout_steps, max_iterations=args.iterations,
        empirical_normalization=False, obs_groups={"actor": ["policy"], "critic": ["critic"]},
        save_interval=50, experiment_name=args.name, logger="tensorboard",
        actor=RslRlMLPModelCfg(hidden_dims=list(p.actor_hidden), activation="elu", obs_normalization=True,
                               distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                                   init_std=math.exp(p.init_log_std))),
        critic=RslRlMLPModelCfg(hidden_dims=list(p.critic_hidden), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=p.value_coef, use_clipped_value_loss=True, clip_param=p.clip, entropy_coef=p.entropy_coef,
            num_learning_epochs=p.epochs, num_mini_batches=p.minibatches, learning_rate=p.lr, schedule="adaptive",
            gamma=p.gamma, lam=p.lam, desired_kl=0.01, max_grad_norm=p.max_grad_norm),
    )
    import importlib.metadata as md

    agent = handle_deprecated_rsl_rl_cfg(agent, md.version("rsl-rl-lib"))  # as Isaac Lab's train.py does
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    rsl_dir = run_dir / "rsl_rl"
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(rsl_dir), device=args.device)
    with open(run_dir / "rsl_agent.yaml", "w") as f:
        yaml.safe_dump(agent.to_dict(), f, sort_keys=False)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)

    # ---- convert the rsl_rl actor to walk_rl's checkpoint + Pi NumPy format
    actor = runner.alg.actor
    ac = ActorCritic(task.num_actor_obs, task.num_critic_obs, task.num_actions, p.actor_hidden, p.critic_hidden,
                     p.init_log_std).to(args.device)
    lin_src = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
    lin_dst = [m for m in ac.actor if isinstance(m, torch.nn.Linear)]
    assert len(lin_src) == len(lin_dst)
    with torch.no_grad():
        for s, d in zip(lin_src, lin_dst):
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
        nz = actor.obs_normalizer  # (x - mean) / (std + eps); ours: (x - mean) / sqrt(var + 1e-8), clip 10
        ac.actor_norm.mean.copy_(nz._mean.squeeze(0))
        ac.actor_norm.var.copy_((nz._std.squeeze(0) + nz.eps) ** 2 - 1e-8)
    with torch.inference_mode():  # rsl_rl rolls out in inference mode: env buffers are inference tensors
        obs_td = wrapped.get_observations().to(args.device)
        a_rsl = actor(obs_td)
        obs = obs_td["policy"]
        a_ours = ac.act_deterministic(obs)
    conv_err = (a_rsl - a_ours).abs().max().item()
    print(f"[rsl] converted actor: max |rsl - walk_rl| action diff {conv_err:.2e} on {len(obs)} live observations",
          flush=True)
    meta = {"iteration": args.iterations, "trainer": "rsl_rl", "backend": "lab", "obs_layout": task.obs_layout(),
            "control_hz": cfg.sim.control_hz, "actuated_joints": task.act_names, "actor_obs_dim": task.num_actor_obs,
            "joint_limit_rad": task.joint_limit.tolist(), "act_scale_rad": task.act_scale.tolist(),
            "servo_resolution_rad": task.servo_res, "use_phase_clock": task.use_clock,
            "gait_frequency_hz": cfg.gait.frequency_hz, "gait_frequency_per_mps": cfg.gait.frequency_per_mps,
            "gait_frequency_ref_mps": cfg.gait.frequency_ref_mps, "gait_frequency_range_hz": cfg.gait.frequency_range_hz,
            "smoothing_alpha": cfg.actions.smoothing_alpha, "conversion_max_action_err": conv_err}
    torch.save({"model": ac.state_dict(), "meta": meta}, run_dir / "checkpoint.pt")
    export_numpy_policy(ac, run_dir / "policy_numpy.npz", meta)
    print(f"[rsl] wrote {run_dir / 'checkpoint.pt'} and policy_numpy.npz", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
