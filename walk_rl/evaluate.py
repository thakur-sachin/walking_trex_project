"""Evaluate a trained walking policy with deterministic actions at fixed
forward-velocity commands.

    python walk_rl/evaluate.py --name v1 [--no-dr] [--num-envs 256]

Same conditions for every run (whatever it was trained with), so runs compare fairly:
  nominal  DR on (or --no-dr), action delay 0-1 step, no IMU delay, no encoder offset
  hard     action delay 0-2, IMU delay 0-1, +/-0.02 rad encoder offsets
  push     nominal, at cmd 0 and 0.10 m/s: one base-velocity kick at t=5 s in +x/-x/+y/-y
           (a quarter of the envs each), survival over the next 10 s per magnitude
Results -> outputs/walk_runs/<name>/eval_<ckpt>.json (eval_<ckpt>_lab.json with --backend lab:
the same suite on the Isaac Lab port, conda env isaaclab6; any policy can be judged on
either simulator).

Reports per command: survival (fraction of envs that complete the full episode
without falling), achieved forward speed, distance, lateral drift, servo
torque saturation (fraction of steps a servo is at >=95% of its stall torque --
a hardware-feasibility signal), action smoothness, and the ankle-linkage
residual. Also verifies the NumPy export (policy_numpy.npz, what runs on the
Pi) matches the torch actor.

v9 diagnostics (JSON, plus a second printed table):
  at_limit     per joint, fraction of steps at >=95% of the joint limit (is speed capped
               by the range, e.g. the 45 deg tibia placeholder?)
  load80 / tau_mean  per joint, fraction of steps above 80% of stall torque, and mean
               |torque|/stall (is it capped by torque instead?)
  foot_roll_deg      mean |roll| of each foot relative to the world while it is in contact
               (sole on its edge?); measured relative to the foot's roll at the spawn pose
  stance_L/R, step_L/R  contact fraction per foot and mean step length (touchdown foot
               ahead of the other, along the heading) per foot: left/right asymmetry
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="v1")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--ckpt", default="checkpoint.pt", help="file in the run dir: checkpoint.pt | best.pt | checkpoints/ckpt_0750.pt")
parser.add_argument("--no-dr", action="store_true", help="disable domain randomization (nominal robot)")
parser.add_argument("--commands", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
parser.add_argument("--no-hard", action="store_true", help="skip the hard-sensor table")
parser.add_argument("--no-push", action="store_true", help="skip the push-recovery test")
parser.add_argument("--push-mags", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
parser.add_argument("--backend", choices=["isaacsim", "lab"], default="isaacsim")
parser.add_argument("--actuator", choices=["implicit_pd", "lx16a"], default=None,
                    help="override walk_task.yaml actuator.model (lx16a: the fitted servo model; result file gets _lx16a)")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--out", default=None, help="JSON path (default: the run dir, see above)")
parser.add_argument("--ground-stiffness", type=float, default=None,
                    help="compliant ground [N/m] (walk_rl/ground.py); default: the run's sim.ground_compliance")
parser.add_argument("--ground-damping", type=float, default=0.0, help="with --ground-stiffness [N*s/m]")
args = parser.parse_args()

from walk_rl import backend  # noqa: E402

simulation_app = backend.launch(args.backend, device=args.device)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import json  # noqa: E402

from walk_rl.task_core import apply_checkpoint_interface, load_task_config  # noqa: E402

TrexWalkEnv = backend.env_class(args.backend)
from walk_rl.ppo import ActorCritic  # noqa: E402


def _yaw(q: torch.Tensor) -> torch.Tensor:
    """Heading angle of the base from an xyzw quaternion."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(a.sin(), a.cos())


def _roll(q: torch.Tensor) -> torch.Tensor:
    """Roll angle of a frame from an xyzw quaternion (gravity direction in that frame)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


class NumpyPolicy:
    """What the Pi runs: normalizer + ELU MLP in plain NumPy."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.var, self.clip = d["norm_mean"], d["norm_var"], float(d["norm_clip"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["num_layers"]))]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return x


try:
    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.name
    cfg = load_task_config(run_dir / "walk_task.yaml")
    if args.no_dr:
        cfg.domain_randomization.enabled = False
    if args.actuator:
        cfg.actuator.model = args.actuator
    ck = torch.load(run_dir / args.ckpt, map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    cfg.push.enabled = False  # training pushes off; the push test kicks explicitly
    if args.ground_stiffness is not None:
        from types import SimpleNamespace
        cfg.sim.ground_compliance = SimpleNamespace(stiffness_n_per_m=args.ground_stiffness,
                                                    damping_n_s_per_m=args.ground_damping)

    def profile(hard: bool) -> None:
        cfg.actions.delay_steps_range = [0, 2] if hard else [0, 1]
        cfg.observation.imu_delay_steps_range = [0, 1] if hard else [0, 0]
        cfg.observation.encoder_offset_rad = 0.02 if hard else 0.0
        if hasattr(cfg.observation, "heading"):
            cfg.observation.heading.ref_init_range_rad = 0.0  # reference = start heading

    profile(False)
    env = TrexWalkEnv(args.num_envs, cfg, device=args.device, simulation_app=simulation_app, seed=123)
    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std).to(env.device)
    ac.load_state_dict(ck["model"])
    from walk_rl.ppo import export_numpy_policy  # parity check uses the weights actually loaded above
    _tmp = PROJECT_ROOT / "outputs" / "walk_runs" / f"_eval_{args.name}.npz"
    export_numpy_policy(ac, _tmp, {})
    npol = NumpyPolicy(_tmp)
    _tmp.unlink()
    print(f"[eval] run={args.name} ckpt={args.ckpt} iteration={ck['meta']['iteration']} DR={'off' if args.no_dr else 'on'} "
          f"envs={args.num_envs} actor_obs={env.num_actor_obs} DR_status={env.dr_active}")
    max_np_err = 0.0

    def run(v: float, seconds: float, kick_mag: float = 0.0, kick_at_s: float = 5.0) -> dict:
        global max_np_err
        def fixed(ids, v=v):
            env.cmd_vx[ids] = v
            env.cmd_timer[ids] = 1e9
        env._sample_commands = fixed
        obs = env.reset()
        fixed(env.all_idx)
        n = env.num_envs
        fell = torch.zeros(n, dtype=torch.bool, device=env.device)
        fell_after_kick = torch.zeros(n, dtype=torch.bool, device=env.device)
        alive_steps = torch.zeros(n, device=env.device)
        z = lambda: torch.zeros(n, device=env.device)
        vx_sum, vy_abs, vy_sq, sat, arate, resid, path, qd_abs = z(), z(), z(), z(), z(), z(), z(), z()
        roll_min, roll_max = torch.full((n,), 9.0, device=env.device), torch.full((n,), -9.0, device=env.device)
        x0 = env.root_pos[:, 0].clone()
        y0 = env.root_pos[:, 1].clone()
        yaw0 = _yaw(env.root_q)
        x_end, y_end, yaw_end = x0.clone(), y0.clone(), yaw0.clone()
        prev_xy = env.root_pos[:, :2].clone()
        kick_step = int(round(kick_at_s / env.dt))
        A = env.num_actions
        at_limit, load80, tau_mean = torch.zeros(n, A, device=env.device), torch.zeros(n, A, device=env.device), torch.zeros(n, A, device=env.device)
        foot_roll, stance_steps = torch.zeros(n, 2, device=env.device), torch.zeros(n, 2, device=env.device)
        step_len, step_n = torch.zeros(n, 2, device=env.device), torch.zeros(n, 2, device=env.device)
        foot_roll0 = _roll(env._sim_link_state()[0][:, env.foot_i, 3:7]).mean(dim=0)  # spawn-pose foot roll (2,)
        prev_contact = env._contacts().clone()
        for t in range(int(round(seconds / env.dt))):
            if kick_mag > 0 and t == kick_step:
                d = torch.arange(n, device=env.device) % 4  # +x, -x, +y, -y
                dirs = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], device=env.device)
                env.kick(env.all_idx, dirs[d] * kick_mag, torch.zeros(n, 3, device=env.device))
            with torch.no_grad():
                act = ac.act_deterministic(obs["actor"])
            if t % 50 == 0:  # export parity check
                ref = npol(obs["actor"][:8].cpu().numpy().astype(np.float64))
                max_np_err = max(max_np_err, float(np.abs(ref - act[:8].cpu().numpy()).max()))
            live = ~fell
            obs, rew, done, extras = env.step(act)
            # `done` envs were already reset inside step(); a non-timeout done is a fall.
            newly_fell = done & ~extras["time_outs"] & live
            fell |= newly_fell
            if t >= kick_step:
                fell_after_kick |= newly_fell
            alive_steps += live.float()
            m = live & ~done
            mf = m.float()
            vx_sum += mf * env.lin_vel_b[:, 0]
            vy_abs += mf * env.lin_vel_b[:, 1].abs()
            vy_sq += mf * env.lin_vel_b[:, 1] ** 2
            sat += mf * ((env.torque.abs() / env.tau_max) > 0.95).float().mean(dim=1)
            arate += mf * (env.last_action - env.prev_action).abs().mean(dim=1)
            qd_abs += mf * env.qd[:, env.act_idx].abs().mean(dim=1)
            roll = torch.atan2(env.gravity_b[:, 1], -env.gravity_b[:, 2])
            roll_min = torch.where(m, torch.minimum(roll_min, roll), roll_min)
            roll_max = torch.where(m, torch.maximum(roll_max, roll), roll_max)
            q = env.q
            r = torch.rad2deg((q[:, env.ank_idx] - (q[:, env.fem_idx] - q[:, env.tib_idx])).abs()).mean(dim=1)
            resid += mf * r
            x_end = torch.where(m, env.root_pos[:, 0], x_end)
            y_end = torch.where(m, env.root_pos[:, 1], y_end)
            yaw_end = torch.where(m, _yaw(env.root_q), yaw_end)
            path += mf * (env.root_pos[:, :2] - prev_xy).norm(dim=1)
            prev_xy = env.root_pos[:, :2].clone()
            # v9 diagnostics
            mA = mf.unsqueeze(1)
            q_act = q[:, env.act_idx]
            at_limit += mA * ((q_act - env.act_center).abs() >= 0.95 * env.joint_limit).float()
            tau_n = env.torque.abs() / env.tau_max
            load80 += mA * (tau_n > 0.8).float()
            tau_mean += mA * tau_n
            contact = env._contacts()
            m2 = mA * contact.float()
            fr = _wrap(_roll(env._sim_link_state()[0][:, env.foot_i, 3:7]) - foot_roll0)
            foot_roll += m2 * fr.abs()
            stance_steps += m2
            touchdown = contact & ~prev_contact
            prev_contact = contact.clone()
            yaw = _yaw(env.root_q)
            d = env.foot_pos[:, :, :2] - env.foot_pos[:, [1, 0], :2]  # each foot relative to the other
            ahead = yaw.cos().unsqueeze(1) * d[..., 0] + yaw.sin().unsqueeze(1) * d[..., 1]
            td = mA * touchdown.float()
            step_len += td * ahead
            step_n += td
        steps = alive_steps.clamp(min=1)
        dyaw = torch.rad2deg(_wrap(yaw_end - yaw0))
        return {
            "cmd": v, "survive": 1.0 - fell.float().mean().item(),
            "survive_after_kick": 1.0 - fell_after_kick.float().mean().item(),
            "vx": (vx_sum / steps).mean().item(), "dist": (x_end - x0).mean().item(),
            "|vy|": (vy_abs / steps).mean().item(), "vy_rms": (vy_sq / steps).sqrt().mean().item(),
            "roll_p2p_deg": torch.rad2deg(roll_max - roll_min).mean().item(),
            "sat": (sat / steps).mean().item(), "d_action": (arate / steps).mean().item(),
            "|qd|": (qd_abs / steps).mean().item(), "ankle_res_deg": (resid / steps).mean().item(),
            "dy": (y_end - y0).mean().item(), "|dy|": (y_end - y0).abs().mean().item(),
            "yaw_deg": dyaw.mean().item(), "|yaw|_deg": dyaw.abs().mean().item(),
            "path": path.mean().item(),
            "at_limit": dict(zip(env.act_names, (at_limit / steps.unsqueeze(1)).mean(0).tolist())),
            "load80": dict(zip(env.act_names, (load80 / steps.unsqueeze(1)).mean(0).tolist())),
            "tau_mean": dict(zip(env.act_names, (tau_mean / steps.unsqueeze(1)).mean(0).tolist())),
            "foot_roll_deg": torch.rad2deg(foot_roll.sum(0) / stance_steps.sum(0).clamp(min=1)).tolist(),
            "stance_L": (stance_steps[:, 0] / steps).mean().item(), "stance_R": (stance_steps[:, 1] / steps).mean().item(),
            "step_L": (step_len[:, 0].sum() / step_n[:, 0].sum().clamp(min=1)).item(),
            "step_R": (step_len[:, 1].sum() / step_n[:, 1].sum().clamp(min=1)).item(),
        }

    def table(label: str) -> list:
        rows = []
        print(f"[eval] ---- {label}")
        print("[eval]  cmd  | surv | vx (%cmd)      | dist  | vy_rms | roll p2p | |yaw| deg | sat   | d_act | |qd|")
        for v in args.commands:
            r = run(v, cfg.sim.episode_length_s)
            pct = f"({100 * r['vx'] / v:4.0f}%)" if v > 0 else "      "
            print(f"[eval] {v:.2f} | {r['survive']:.2f} | {r['vx']:+.3f} {pct} | {r['dist']:+.2f} | {r['vy_rms']:.3f}  | "
                  f"{r['roll_p2p_deg']:5.1f}    | {r['|yaw|_deg']:5.1f}     | {r['sat']:.3f} | {r['d_action']:.3f} | {r['|qd|']:.2f}",
                  flush=True)
            rows.append(r)
        print("[eval]  cmd  | tibia@lim L/R | femur@lim L/R | load>80% tib/fem/hip | foot roll L/R deg | stance L/R | step L/R cm | dy m")
        for r in rows:
            j = lambda k, n: r[k][n]
            lr = lambda k, base: (j(k, f"left_{base}_joint") + j(k, f"right_{base}_joint")) / 2
            print(f"[eval] {r['cmd']:.2f} | {j('at_limit', 'left_tibia_joint'):.2f} / {j('at_limit', 'right_tibia_joint'):.2f}   | "
                  f"{j('at_limit', 'left_femur_joint'):.2f} / {j('at_limit', 'right_femur_joint'):.2f}   | "
                  f"{lr('load80', 'tibia'):.3f} / {lr('load80', 'femur'):.3f} / {lr('load80', 'hip'):.3f}  | "
                  f"{r['foot_roll_deg'][0]:5.1f} / {r['foot_roll_deg'][1]:5.1f}     | {r['stance_L']:.2f} / {r['stance_R']:.2f} | "
                  f"{100 * r['step_L']:+.1f} / {100 * r['step_R']:+.1f} | {r['dy']:+.2f}", flush=True)
        return rows

    results = {"run": args.name, "ckpt": args.ckpt, "iteration": ck["meta"]["iteration"], "dr": not args.no_dr,
               "backend": args.backend}
    results["nominal"] = table("nominal (action delay 0-1, no IMU delay, no encoder offset)")
    if not args.no_hard:
        profile(True)
        results["hard"] = table("hard sensors (action delay 0-2, IMU delay 0-1, +/-0.02 rad encoder offsets)")
        profile(False)
    if not args.no_push:
        print("[eval] ---- push recovery (kick at t=5 s, +x/-x/+y/-y quarters; survival over the next 10 s)")
        push = []
        for v in (0.0, 0.10):
            line = []
            for mag in args.push_mags:
                r = run(v, 15.0, kick_mag=mag)
                push.append({"cmd": v, "kick_mps": mag, "survive": r["survive_after_kick"], "survive_total": r["survive"]})
                line.append(f"{mag:.2f}: {r['survive_after_kick']:.2f}")
            print(f"[eval] cmd {v:.2f} | " + " | ".join(line), flush=True)
        results["push"] = push
    print(f"[eval] numpy-export max |torch - numpy| action diff: {max_np_err:.2e}")
    results["numpy_parity"] = max_np_err
    from common.provenance import provenance
    results["provenance"] = provenance()
    out = Path(args.out) if args.out else run_dir / f"eval_{Path(args.ckpt).stem}{'_nodr' if args.no_dr else ''}{'_lx16a' if args.actuator == 'lx16a' else ''}{'_lab' if args.backend == 'lab' else ''}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[eval] wrote {out}")
except BaseException:
    # Kit's fast shutdown in close() exits before Python prints a pending traceback.
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
