"""Video of the deployed loop (PolicyRunner + NumPy policy) walking in MuJoCo, the same setup
as sim2sim.py: what the Pi will run, on a physics engine the policy never trained in.

    python pi_runner/render_mujoco.py --run v9o --actuator lx16a \
        --schedule 0:2 0.15:8 0.20:8 0:4

`--schedule cmd:seconds ...` is the forward-speed command over time. Tracking camera from the
robot's front-left; the overlay shows time, command, measured forward speed and distance.
Writes outputs/videos/<run>_<actuator>_<tag>.mp4 (H.264, 30 fps) and prints a summary.
Uses MUJOCO_GL=glfw when a display is available (EGL failed on this PC), else egl.
"""
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pi_runner.mujoco_backend import MujocoTrex  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", default="v9o")
parser.add_argument("--policy", default="policy_numpy.npz")
parser.add_argument("--actuator", choices=["implicit_pd", "lx16a"], default="lx16a")
parser.add_argument("--schedule", nargs="+", default=["0:2", "0.15:8", "0.20:8", "0:4"])
parser.add_argument("--friction", type=float, default=0.5)
parser.add_argument("--delay", type=int, default=1, help="bus latency in control steps")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--size", type=int, nargs=2, default=[960, 544])
parser.add_argument("--tag", default=None)
args = parser.parse_args()

run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run
schedule = [(float(c), float(s)) for c, s in (x.split(":") for x in args.schedule)]
tag = args.tag or "_".join(f"{c:.2f}" for c, _ in schedule)
out = PROJECT_ROOT / "outputs" / "videos" / f"{args.run}_{args.actuator}_{tag}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)

runner = PolicyRunner(run_dir, args.policy)
tm = runner.cfg.termination
sim = MujocoTrex(friction=args.friction, control_hz=runner.cfg.sim.control_hz, delay_steps=args.delay,
                 max_tilt_deg=tm.max_tilt_deg, min_head_height_m=tm.min_head_height_m,
                 ground_contact_force_n=tm.ground_contact_force_n, actuator=args.actuator, visual=True)
runner.reset_state(np.zeros(8, dtype=np.float32), phase=0.0)
runner.latch_heading(sim.read().yaw)

w, h = args.size
sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = max(w, 640), max(h, 480)
renderer = mujoco.Renderer(sim.model, height=h, width=w)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = sim.base
cam.distance, cam.azimuth, cam.elevation = 1.0, 135.0, -20.0

dt = 1.0 / runner.cfg.sim.control_hz
render_every = 1.0 / args.fps
next_frame, t = 0.0, 0.0
p0, yaw0 = sim.base_pos(), sim.yaw()
vx_f = 0.0
segments = []
with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8) as video:
    for cmd, secs in schedule:
        seg_p0, seg_t0, n = sim.base_pos(), t, 0
        for _ in range(int(round(secs / dt))):
            goal = runner.step(sim.read(), cmd)
            sim.write(goal)
            t += dt
            v = sim.base_vel_world()
            yaw = sim.yaw()
            vx_f += 0.15 * ((math.cos(yaw) * v[0] + math.sin(yaw) * v[1]) - vx_f)  # display only: ~0.3 s filter
            n += 1
            while t >= next_frame:
                renderer.update_scene(sim.data, camera=cam)
                img = np.ascontiguousarray(renderer.render())
                d = sim.base_pos() - p0
                fwd = math.cos(yaw0) * d[0] + math.sin(yaw0) * d[1]
                lines = [f"{args.run}  ({args.actuator} servo, MuJoCo)", f"t {t:5.1f} s   cmd {cmd:.2f} m/s",
                         f"vx {vx_f:+.3f} m/s   forward {fwd:+.2f} m"]
                if sim.failed():
                    lines.append(f"FELL: {sim.failed()}")
                for k, s in enumerate(lines):
                    cv2.putText(img, s, (14, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
                video.append_data(img)
                next_frame += render_every
            if sim.failed():
                break
        d = sim.base_pos() - seg_p0
        seg_fwd = math.cos(yaw0) * d[0] + math.sin(yaw0) * d[1]
        segments.append((cmd, t - seg_t0, seg_fwd / max(t - seg_t0, 1e-9)))
        if sim.failed():
            break

for cmd, secs, vx in segments:
    pct = f" ({100 * vx / cmd:.0f}%)" if cmd > 0 else ""
    print(f"[render] cmd {cmd:.2f} m/s for {secs:4.1f} s: mean forward speed {vx:+.3f} m/s{pct}")
dyaw = math.degrees(math.atan2(math.sin(sim.yaw() - yaw0), math.cos(sim.yaw() - yaw0)))
print(f"[render] {'FELL: ' + sim.failed() if sim.failed() else 'no fall'} at t {t:.1f} s; heading change {dyaw:+.1f} deg")
print(f"[render] wrote {out}")
