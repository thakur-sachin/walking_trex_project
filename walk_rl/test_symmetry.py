"""Tests for the left/right mirror map (walk_rl/symmetry.py). A wrong map actively damages
training, so it is checked two ways:

  algebra   mirror(mirror(x)) == x; the neutral standing frame maps to itself; a right
            lean maps to a left lean; dimensions match the task's observation sizes.
  physics   (--sim) two envs on CPU physics, all noise/delays/DR off. Env 1 starts as the
            mirror image of env 0 (mirrored base kick, clock half a period ahead) and gets
            the mirrored action every step. mirror(obs_0) must then match obs_1 up to the
            simulator's own asymmetry (CAD meshes, solver order). The same check with a
            deliberately WRONG map (hips negated) must fail by a wide margin, so the test
            can tell a right map from a wrong one.

    python walk_rl/test_symmetry.py                 # algebra only, no simulator
    python walk_rl/test_symmetry.py --sim [--run v8d]   # + physics (conda env isaac6)
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from walk_rl.symmetry import ACTUATED, MirrorMap, joint_perm  # noqa: E402
from walk_rl.task_core import load_task_config, obs_dims  # noqa: E402


def test_algebra(cfg) -> None:
    m = MirrorMap(cfg)
    frame, n_actor, n_critic = obs_dims(cfg)
    assert m.dims == {"actor": n_actor, "critic": n_critic, "action": 8}, m.dims
    assert joint_perm(ACTUATED) == [0, 1, 3, 2, 5, 4, 7, 6]
    for key, d in m.dims.items():
        x = torch.randn(64, d)
        assert torch.equal(getattr(m, key)(getattr(m, key)(x)), x), f"{key}: mirror twice != identity"
    # neutral standing frame: upright, no motion, joints at 0, clock 0 (standing), heading 0
    f = torch.zeros(frame)
    f[2] = -1.0  # gravity z
    if cfg.observation.heading.enabled:
        f[-1] = 1.0  # cos(0)
    x = f.repeat(cfg.observation.history_len)
    assert torch.equal(m.actor(x), x), "neutral standing frame is not mirror-invariant"
    # right lean (gravity has +y in the body frame when rolled right... either way) maps to the opposite lean
    lean = x.clone()
    lean[1] = 0.2
    assert m.actor(lean)[1] == -0.2
    # a left-hip-only action maps to a right-hip-only action of the same sign
    a = torch.zeros(8)
    a[2] = 0.5
    assert m.action(a)[3] == 0.5 and m.action(a)[2] == 0.0
    print("[sym] algebra: OK (involution, neutral invariance, lean flip, hip swap, dims "
          f"actor={n_actor} critic={n_critic})")


def test_physics(cfg_path: Path, steps: int) -> None:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from walk_rl.env import TrexWalkEnv
        from walk_rl.task_core import apply_checkpoint_interface

        cfg = load_task_config(cfg_path)
        ck_meta = None
        if cfg_path.parent.joinpath("checkpoint.pt").exists():
            ck_meta = torch.load(cfg_path.parent / "checkpoint.pt", map_location="cpu")["meta"]
            apply_checkpoint_interface(cfg, ck_meta)
        cfg.domain_randomization.enabled = False
        cfg.push.enabled = False
        cfg.harness.enabled = False
        cfg.actions.delay_steps_range = [0, 0]
        ob = cfg.observation
        ob.imu_delay_steps_range = [0, 0]
        ob.encoder_offset_rad = 0.0
        for k in vars(ob.noise):
            setattr(ob.noise, k, 0.0)
        if ob.heading.enabled:
            ob.heading.ref_init_range_rad = 0.0
            ob.heading.drift_deg_per_min = 0.0
            ob.heading.noise_rad = 0.0
        cfg.command.standing_prob = 0.0

        env = TrexWalkEnv(2, cfg, device="cpu", simulation_app=app, seed=0)
        m = MirrorMap(cfg, env.act_names)

        def fixed(ids):
            env.cmd_vx[ids] = 0.10
            env.cmd_timer[ids] = 1e9
        env._sample_commands = fixed
        env.reset()
        env.phase[1] = (env.phase[0] + 0.5) % 1.0
        # mirrored initial disturbance: a lean/sway that the policy must catch
        dv = torch.tensor([[0.05, 0.08, 0.0], [0.05, -0.08, 0.0]])
        dw = torch.tensor([[0.6, 0.2, 0.3], [-0.6, 0.2, -0.3]])
        env.kick(env.all_idx, dv, dw)
        env._hist_fresh[:] = True  # history restarts from the mirrored state (phase was set after reset)
        obs = env._observations()

        frame = env.frame_dim
        names = ["g.x", "g.y", "g.z", "gyro.x", "gyro.y", "gyro.z", "cmd"] + \
            [f"{b}.{n[:-6]}" for b in ("q", "dq", "a") for n in env.act_names] + ["clk_s", "clk_c", "hd_s", "hd_c"]
        hip_cols = [blk + j for blk in (7, 15, 23) for j in (2, 3)]  # hips in the pos / vel / prev-action blocks

        def wrong_frame(x):  # the plausible mistake: also negate the hips when swapping them
            y = m.actor(x[..., :frame]).clone() if x.shape[-1] == frame else m.actor(x)[..., :frame].clone()
            y[..., hip_cols] *= -1
            return y

        t = torch.arange(steps, dtype=torch.float32) * env.dt
        # smooth, deliberately asymmetric action sequence (left and right legs do different things)
        freq = torch.tensor([0.7, 0.5, 1.8, 1.1, 1.8, 1.3, 1.8, 0.9])
        amp = torch.tensor([0.2, 0.2, 0.15, 0.05, 0.3, 0.1, 0.3, 0.15])
        phase0 = torch.tensor([0.3, 1.1, 2.0, 4.0, 0.5, 5.0, 3.0, 1.7])
        worst = {"actor": 0.0, "critic": 0.0, "hip_ok": 0.0, "hip_wrong": 0.0}
        n_cmp = 0
        for k in range(steps):
            a0 = (amp * torch.sin(2 * torch.pi * freq * t[k] + phase0)).unsqueeze(0)
            obs, rew, done, _ = env.step(torch.cat([a0, m.action(a0)], dim=0))
            if done.any():
                print(f"[sym] episode ended at step {k} (done={done.tolist()}); compared {n_cmp} steps")
                break
            n_cmp += 1
            mir = m.actor(obs["actor"][0])
            da = (mir - obs["actor"][1]).abs()
            dc = (m.critic(obs["critic"][0]) - obs["critic"][1]).abs()
            hw = (wrong_frame(obs["actor"][0]) - obs["actor"][1][:frame]).abs()[hip_cols]
            worst["actor"] = max(worst["actor"], da.max().item())
            worst["critic"] = max(worst["critic"], dc.max().item())
            worst["hip_ok"] = max(worst["hip_ok"], da[hip_cols].max().item())
            worst["hip_wrong"] = max(worst["hip_wrong"], hw.max().item())
            if k in (0, 1, 5, 10, steps - 1):
                top = da[:frame].topk(3).indices.tolist()
                print(f"[sym] step {k:3d}: max |mirror(obs0) - obs1| actor {da.max().item():.2e} critic {dc.max().item():.2e} | "
                      f"reward {rew[0].item():+.3f} vs {rew[1].item():+.3f} | largest: " + ", ".join(
                          f"{names[i]} {mir[i].item():+.3f} vs {obs['actor'][1][i].item():+.3f}" for i in top))
        print(f"[sym] physics over {n_cmp} steps: correct map worst actor {worst['actor']:.3e}, critic {worst['critic']:.3e}; "
              f"hip entries: correct map {worst['hip_ok']:.3e} vs WRONG map (hips negated) {worst['hip_wrong']:.3e}")
        ok = n_cmp >= 10 and worst["actor"] < 0.05 and worst["hip_wrong"] > 5 * max(worst["hip_ok"], 1e-3)
        print(f"[sym] physics: {'OK' if ok else 'FAILED'} (pass: >=10 steps compared, correct map < 0.05 everywhere, "
              f"wrong map > 5x correct on the hips)")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        app.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--run", default="v8d", help="outputs/walk_runs/<run>/walk_task.yaml for the obs layout")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    run_cfg = PROJECT_ROOT / "outputs" / "walk_runs" / args.run / "walk_task.yaml"
    test_algebra(load_task_config())
    if args.sim:
        test_physics(run_cfg, args.steps)
