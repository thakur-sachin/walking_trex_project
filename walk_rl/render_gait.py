"""Render a short capture of the trained policy walking, so the gait can be
looked at (metrics alone do not show a limp, a shuffle or a lean).

    python walk_rl/render_gait.py --name v4 --cmd 0.15 --seconds 8

Writes PNG frames + an animated GIF to outputs/gait/<name>_cmd<cmd>/ and
prints per-foot step statistics (steps/s, stance fraction, left/right balance).
Runs a single env, static wide camera.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="v4")
parser.add_argument("--ckpt", default="best.pt", help="file in the run dir (best.pt | checkpoint.pt | checkpoints/ckpt_0750.pt)")
parser.add_argument("--cmd", type=float, default=0.15)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--every", type=int, default=2, help="capture every N control steps")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.experimental.objects import Camera  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

from walk_rl.env import TrexWalkEnv, apply_checkpoint_interface, load_task_config  # noqa: E402
from walk_rl.ppo import ActorCritic  # noqa: E402
import test_load_robot as tlr  # noqa: E402

try:
    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.name
    cfg = load_task_config(run_dir / "walk_task.yaml")
    cfg.domain_randomization.enabled = False
    ck = torch.load(run_dir / args.ckpt, map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    cfg.push.enabled = False
    if hasattr(cfg.observation, "heading"):
        cfg.observation.heading.ref_init_range_rad = 0.0
    # CPU physics for rendering: GPU physics keeps transforms out of USD (see env.py).
    env = TrexWalkEnv(1, cfg, device="cpu", simulation_app=simulation_app, render=True, seed=7)
    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std).to(env.device)
    ac.load_state_dict(ck["model"])

    def fixed(ids):
        env.cmd_vx[ids] = args.cmd
        env.cmd_timer[ids] = 1e9
    env._sample_commands = fixed
    obs = env.reset()
    fixed(env.all_idx)

    out_dir = PROJECT_ROOT / "outputs" / "gait" / f"{args.name}_cmd{args.cmd:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    camera = Camera("/World/GaitCam", positions=np.array([[0.0, -1.0, 0.4]]), orientations=np.array([[1.0, 0, 0, 0]]))
    viewport.camera_path = "/World/GaitCam"
    cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/GaitCam")
    # Static, wide shot (tracking the moving robot proved unreliable with Fabric on, and
    # the default 50 mm lens is ~23 deg wide: the robot fills the frame and leaves it).
    # Frame the straight-line walk starting at env 0's origin.
    cam_prim.GetAttribute("focalLength").Set(18.0)
    org = env.env_origins[0].cpu().numpy()
    walk_m = args.cmd * args.seconds
    eye = org + np.array([0.5 * walk_m, -1.3, 0.28])
    target = org + np.array([0.5 * walk_m, 0.0, 0.16])
    quat = tlr._look_at_quaternion(eye, target)  # wxyz
    t_attr, o_attr = cam_prim.GetAttribute("xformOp:translate"), cam_prim.GetAttribute("xformOp:orient")
    t_attr.Set(type(t_attr.Get())(float(eye[0]), float(eye[1]), float(eye[2])))
    o_attr.Set(type(o_attr.Get())(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
    timeline = omni.timeline.get_timeline_interface()

    steps = int(args.seconds / env.dt)
    contacts = []
    fell_any, tilt_deg, x_start = False, [], float(env.root_pos[0, 0])
    frame = 0
    for t in range(steps):
        with torch.no_grad():
            act = ac.act_deterministic(obs["actor"])
        obs, rew, done, extras = env.step(act)
        contacts.append(env._contacts()[0].cpu().numpy())
        fell_any |= bool(done[0] & ~extras["time_outs"][0])
        tilt_deg.append(float(torch.rad2deg(torch.acos((-env.gravity_b[0, 2]).clamp(-1, 1)))))
        if t % args.every == 0:
            timeline.pause()  # pause, not stop(): stop() resets the pose
            for _ in range(3 if frame else 60):  # first frame: let MDL shaders compile
                simulation_app.update()
            capture_viewport_to_file(viewport, str(out_dir / f"frame_{frame:04d}.png"))
            simulation_app.update()
            timeline.play()
            frame += 1

    c = np.array(contacts)  # (T, 2) [left, right]
    td = np.abs(np.diff(c.astype(int), axis=0))  # touchdown/liftoff events
    secs = len(c) * env.dt
    print(f"[gait] cmd {args.cmd:.2f} m/s, {secs:.1f} s, frames={frame}, fell_any={fell_any}, "
          f"forward {float(env.root_pos[0, 0]) - x_start:+.2f} m (cmd {args.cmd * secs:.2f} m), "
          f"body tilt mean {np.mean(tilt_deg):.1f} deg max {np.max(tilt_deg):.1f} deg, device={env.device}")
    print(f"[gait] stance fraction L/R: {c[:, 0].mean():.2f} / {c[:, 1].mean():.2f}")
    print(f"[gait] contact transitions per second L/R: {td[:, 0].sum() / secs:.2f} / {td[:, 1].sum() / secs:.2f}"
          f"  (clock target {2 * float(env.gait_frequency()[0]):.2f})")
    print(f"[gait] frames in {out_dir}")
    try:
        from PIL import Image
        # the last capture can be unflushed at shutdown; use only frames actually on disk
        files = sorted(out_dir.glob("frame_*.png"))
        frames = [Image.open(f).convert("RGB") for f in files]
        frames[0].save(out_dir / "gait.gif", save_all=True, append_images=frames[1:], duration=int(1000 * env.dt * args.every), loop=0)
        print(f"[gait] gif: {out_dir / 'gait.gif'}")
    except Exception as e:  # PIL missing is non-fatal; PNG frames are the artifact
        print(f"[gait] GIF skipped ({type(e).__name__}: {e})")
except BaseException:
    # Kit's fast shutdown in close() exits before Python prints a pending traceback.
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
