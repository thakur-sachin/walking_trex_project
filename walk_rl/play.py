"""Watch a trained walking policy live in the Isaac Sim window (one robot, CPU physics,
real time, camera following the robot).

    python walk_rl/play.py --name v8c --ckpt checkpoint.pt [--cmd 0.10] [--dr]

Keys (click the viewport first so it has keyboard focus):
    UP / DOWN   forward speed command +/- 0.05 m/s (0 .. 0.25)
    SPACE       stand (command 0)
    P           push the robot sideways (0.3 m/s base-velocity kick, random side)
    F           push it forward/backward
    R           reset the episode
    C           toggle camera follow

Telemetry (is it following the command?):
    3D view   GREEN arrow = commanded velocity (along the latched heading reference),
              RED arrow = actual base velocity (horizontal, instantaneous: swings with each step),
              YELLOW arrow = actual velocity averaged over the last --avg-s seconds,
              WHITE line = heading reference
    window    "Trex telemetry": forward speed (command / actual / 1 s average) and, per
              servo, the policy's raw target, the target actually sent to the servo (after
              low-pass, quantization and the speed-limited ramp) and the measured position,
              over the last 5 s (--history-s), on a +/- joint-limit scale

The policy sees exactly what it sees in evaluate.py (harness off, training pushes off).
A fall (any non-foot ground contact, tilt or head height) resets, like in training.
Needs a GPU with ~3 GB free for the RTX viewport; physics runs on CPU.
"""
import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="v8c")
parser.add_argument("--ckpt", default="checkpoint.pt", help="file in the run dir (checkpoint.pt | best.pt | checkpoints/ckpt_0750.pt)")
parser.add_argument("--cmd", type=float, default=0.10)
parser.add_argument("--dr", action="store_true", help="keep domain randomization on (random friction/mass/CoM)")
parser.add_argument("--avg-s", type=float, default=1.0, help="averaging window of the yellow arrow / plot line")
parser.add_argument("--history-s", type=float, default=5.0, help="seconds shown in the telemetry plots")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": False})

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.util.debug_draw", True)

import carb.input  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import omni.ui as ui  # noqa: E402
import torch  # noqa: E402
from collections import deque  # noqa: E402
from isaacsim.util.debug_draw import _debug_draw  # noqa: E402
from isaacsim.core.experimental.objects import Camera  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402

from walk_rl.env import TrexWalkEnv, apply_checkpoint_interface, load_task_config  # noqa: E402
from walk_rl.ppo import ActorCritic  # noqa: E402
import test_load_robot as tlr  # noqa: E402


# ui colors are 0xAABBGGRR
C_CMD, C_ACT, C_AVG, C_RAW, C_SENT, C_MEAS = 0xFFFFFFFF, 0xFFFFD000, 0xFF00D7FF, 0x806060FF, 0xFF3399FF, 0xFFFFD000


class Telemetry:
    """Live plots: forward speed (command vs actual) and, per servo, raw policy target /
    target sent to the servo / measured position."""

    def __init__(self, joint_names, limits, n):
        self.n, self.limits = n, limits
        self.vel = {k: deque([0.0] * n, maxlen=n) for k in ("cmd", "act", "avg")}
        self.jt = [{k: deque([0.0] * n, maxlen=n) for k in ("raw", "sent", "meas")} for _ in joint_names]
        self.win = ui.Window("Trex telemetry", width=620, height=980)
        self.plots, self.labels = {}, {}
        with self.win.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=4):
                    ui.Label(f"forward speed [m/s]: white = command, cyan = actual, yellow = {args.avg_s:g} s average",
                             height=16, word_wrap=True)
                    self.labels["vel"] = ui.Label("", height=16)
                    self.plots["vel"] = self._stack(90, -0.10, 0.35, [("cmd", C_CMD), ("act", C_ACT), ("avg", C_AVG)])
                    ui.Label("servo position [rad], scale = +/- limit: faint red = raw policy target, "
                             "orange = target sent to servo, cyan = measured", height=32, word_wrap=True)
                    for i, name in enumerate(joint_names):
                        self.labels[i] = ui.Label("", height=16)
                        lim = float(limits[i])
                        self.plots[i] = self._stack(60, -lim, lim, [("raw", C_RAW), ("sent", C_SENT), ("meas", C_MEAS)])
        self.names = joint_names

    def dock(self):
        """Beside the viewport, so the robot stays visible (needs a rendered frame first)."""
        return self.win.dock_in_window("Viewport", ui.DockPosition.RIGHT, 0.42)

    @staticmethod
    def _stack(h, lo, hi, series):
        out = {}
        with ui.ZStack(height=h):
            ui.Rectangle(style={"background_color": 0xFF202020})
            for key, color in series:
                out[key] = ui.Plot(ui.Type.LINE, lo, hi, style={"color": color, "background_color": 0x00000000})
        return out

    def push(self, cmd, vx, raw, sent, meas):
        self.vel["cmd"].append(cmd)
        self.vel["act"].append(vx)
        k = max(1, int(round(args.avg_s / 0.04)))
        self.vel["avg"].append(sum(list(self.vel["act"])[-k:]) / k)
        for i in range(len(self.jt)):
            self.jt[i]["raw"].append(raw[i])
            self.jt[i]["sent"].append(sent[i])
            self.jt[i]["meas"].append(meas[i])

    def draw(self):
        for key, plot in self.plots["vel"].items():
            plot.set_data(*self.vel[key])
        avg = self.vel["avg"][-1]
        cmd = self.vel["cmd"][-1]
        pct = f" ({100 * avg / cmd:.0f}% of command)" if cmd > 0.01 else ""
        self.labels["vel"].text = f"command {cmd:.2f} | actual {self.vel['act'][-1]:+.3f} | {args.avg_s:g} s avg {avg:+.3f}{pct}"
        for i, name in enumerate(self.names):
            for key, plot in self.plots[i].items():
                plot.set_data(*self.jt[i][key])
            lim = float(self.limits[i])
            m = self.jt[i]["meas"][-1]
            pinned = "  AT LIMIT" if abs(m) > 0.95 * lim else ""
            self.labels[i].text = (f"{name}: sent {self.jt[i]['sent'][-1]:+.2f}  meas {m:+.2f}  "
                                   f"(limit +/-{lim:.2f}){pinned}")


def draw_arrows(dd, base, cmd_vx, yaw_ref, vel_w, vel_avg_w, scale=2.0):
    """3D arrows at the robot: commanded velocity (green, along the heading reference),
    actual horizontal base velocity (red, instantaneous), its moving average (yellow),
    heading reference (white)."""
    import carb
    dd.clear_lines()
    z = base[2] + 0.12
    o = (base[0], base[1], z)
    starts, ends, colors, widths = [], [], [], []

    def arrow(vx, vy, color, w):
        tip = (o[0] + vx * scale, o[1] + vy * scale, z)
        starts.append(o); ends.append(tip); colors.append(color); widths.append(w)
        L = math.hypot(vx, vy) * scale
        if L > 1e-3:
            ang = math.atan2(vy, vx)
            for s in (+1, -1):  # arrow head
                a = ang + math.pi + s * 0.45
                h = min(0.05, 0.4 * L)
                starts.append(tip); ends.append((tip[0] + h * math.cos(a), tip[1] + h * math.sin(a), z))
                colors.append(color); widths.append(w)

    c, s_ = math.cos(yaw_ref), math.sin(yaw_ref)
    starts.append(o); ends.append((o[0] + 0.6 * c, o[1] + 0.6 * s_, z)); colors.append((1, 1, 1, 0.5)); widths.append(1)
    arrow(cmd_vx * c, cmd_vx * s_, (0.1, 1.0, 0.2, 1.0), 6)
    arrow(float(vel_w[0]), float(vel_w[1]), (1.0, 0.15, 0.1, 0.8), 3)
    arrow(float(vel_avg_w[0]), float(vel_avg_w[1]), (1.0, 0.85, 0.0, 1.0), 6)
    dd.draw_lines([carb.Float3(*p) for p in starts], [carb.Float3(*p) for p in ends],
                  [carb.ColorRgba(*col) for col in colors], widths)


try:
    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.name
    cfg = load_task_config(run_dir / "walk_task.yaml")
    cfg.domain_randomization.enabled = args.dr
    ck = torch.load(run_dir / args.ckpt, map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    cfg.push.enabled = False
    cfg.harness.enabled = False
    if hasattr(cfg.observation, "heading"):
        cfg.observation.heading.ref_init_range_rad = 0.0
    cfg.sim.episode_length_s = 1e6  # no time-out while watching
    # CPU physics: GPU physics keeps transforms out of USD, so the viewport would show a frozen robot.
    env = TrexWalkEnv(1, cfg, device="cpu", simulation_app=simulation_app, render=True, seed=7)
    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std).to(env.device)
    ac.load_state_dict(ck["model"])
    print(f"[play] {args.name}/{args.ckpt} (iteration {ck['meta']['iteration']}), actor_obs={env.num_actor_obs}, "
          f"DR={'on' if args.dr else 'off'}", flush=True)

    state = {"cmd": args.cmd, "reset": False, "kick": None, "follow": True}

    def fixed(ids):
        env.cmd_vx[ids] = state["cmd"]
        env.cmd_timer[ids] = 1e9
    env._sample_commands = fixed

    K = carb.input.KeyboardInput

    def on_key(event, *_):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        k = event.input
        if k == K.UP:
            state["cmd"] = min(0.25, round(state["cmd"] + 0.05, 2))
        elif k == K.DOWN:
            state["cmd"] = max(0.0, round(state["cmd"] - 0.05, 2))
        elif k == K.SPACE:
            state["cmd"] = 0.0
        elif k == K.R:
            state["reset"] = True
        elif k == K.P:
            state["kick"] = (0.0, float(np.random.choice([-1.0, 1.0])) * 0.3)
        elif k == K.F:
            state["kick"] = (float(np.random.choice([-1.0, 1.0])) * 0.3, 0.0)
        elif k == K.C:
            state["follow"] = not state["follow"]
        else:
            return True
        print(f"[play] cmd {state['cmd']:.2f} m/s | follow {state['follow']}"
              + (f" | kick {state['kick']}" if state["kick"] else ""), flush=True)
        return True

    kb_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), on_key)

    viewport = get_active_viewport()
    Camera("/World/PlayCam", positions=np.array([[0.0, -1.0, 0.4]]), orientations=np.array([[1.0, 0, 0, 0]]))
    viewport.camera_path = "/World/PlayCam"
    cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/PlayCam")
    cam_prim.GetAttribute("focalLength").Set(18.0)
    t_attr, o_attr = cam_prim.GetAttribute("xformOp:translate"), cam_prim.GetAttribute("xformOp:orient")

    def place_camera():
        p = env.root_pos[0].cpu().numpy()
        eye = p + np.array([0.30, -1.35, 0.40])  # wide enough for the narrowed (docked) viewport
        quat = tlr._look_at_quaternion(eye, p + np.array([0.0, 0.0, -0.05]))  # wxyz
        t_attr.Set(type(t_attr.Get())(*map(float, eye)))
        o_attr.Set(type(o_attr.Get())(*map(float, quat)))

    tele = Telemetry(env.act_names, env.joint_limit.cpu().numpy(), int(args.history_s / env.dt))
    dd = _debug_draw.acquire_debug_draw_interface()
    timeline = omni.timeline.get_timeline_interface()
    obs = env.reset()
    fixed(env.all_idx)
    place_camera()
    timeline.pause()  # warm-up frames must not advance physics: unpowered, the robot falls in ~1 s
    for _ in range(60):  # let shaders compile before starting the clock
        simulation_app.update()
    timeline.play()
    print(f"[play] telemetry docked: {tele.dock()}", flush=True)
    print("[play] running. Keys: UP/DOWN speed, SPACE stand, P side push, F fore/aft push, R reset, C camera follow",
          flush=True)

    falls, t_next, x0 = 0, time.perf_counter(), float(env.root_pos[0, 0])
    vel_hist = deque(maxlen=max(1, int(round(args.avg_s / env.dt))))  # world-frame xy velocity
    while simulation_app.is_running():
        if state["reset"]:
            state["reset"] = False
            obs = env.reset()
            x0 = float(env.root_pos[0, 0])
            vel_hist.clear()
        fixed(env.all_idx)
        if state["kick"] is not None:
            dx, dy = state["kick"]
            env.kick(env.all_idx, torch.tensor([[dx, dy, 0.0]]), torch.zeros(1, 3))
            state["kick"] = None
        with torch.no_grad():
            act = ac.act_deterministic(obs["actor"])
        raw_tgt = (env.act_center + act.clamp(-1, 1) * env.act_scale)[0].tolist()  # before step: policy's wish
        obs, rew, done, extras = env.step(act)
        tele.push(state["cmd"], float(env.lin_vel_b[0, 0]), raw_tgt, env.cmd_target[0].tolist(),
                  env.q[0, env.act_idx].tolist())
        tele.draw()
        vel_hist.append(env.root_lin_vel_w[0, :2].tolist())
        vel_avg = np.mean(np.array(vel_hist), axis=0)
        draw_arrows(dd, env.root_pos[0].tolist(), state["cmd"], float(env.yaw_ref[0]) if env.use_heading
                    else float(env._yaw()[0]), env.root_lin_vel_w[0].tolist(), vel_avg)
        if bool(done[0]) and not bool(extras["time_outs"][0]):
            falls += 1
            print(f"[play] fell (#{falls}); reset", flush=True)
            x0 = float(env.root_pos[0, 0])
            vel_hist.clear()
        if state["follow"]:
            place_camera()
        # render one frame per control step: pause so update() does not advance physics again
        timeline.pause()
        simulation_app.update()
        timeline.play()
        if env.episode_len[0] % 50 == 0:
            print(f"[play] cmd {state['cmd']:.2f} | vx {float(env.lin_vel_b[0, 0]):+.3f} m/s | "
                  f"walked {float(env.root_pos[0, 0]) - x0:+.2f} m", flush=True)
        # real time: one control step per env.dt
        t_next += env.dt
        slack = t_next - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            t_next = time.perf_counter()
except BaseException:
    # Kit's fast shutdown in close() exits before Python prints a pending traceback.
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
