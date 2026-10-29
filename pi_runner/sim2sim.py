"""Pi-runner pass test 3: the deployed loop (runner + NumPy policy) walks in MuJoCo, a physics
engine the policy never trained in, across floor friction and bus-latency sweeps.

    python pi_runner/sim2sim.py --run v8d [--seconds 60] [--noisy]

For each friction x delay x command it runs one 60 s episode and reports survival time, the
reason for any fall, forward speed (heading frame), sideways drift and heading change.
Pass criterion (plan A6): survives the full 60 s in every walking condition.
Writes outputs/walk_runs/<run>/sim2sim_<policy>.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from common.provenance import provenance  # noqa: E402
from pi_runner.mujoco_backend import MujocoTrex  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="v8d")
parser.add_argument("--policy", default="policy_numpy.npz")
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--frictions", type=float, nargs="+", default=[0.5, 0.8, 1.2])
parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--contact-tc", type=float, nargs="+", default=[0.004],
                    help="MuJoCo contact time constant(s) [s]: 0.004 near-rigid (PhysX-like), 0.01-0.02 softer floor/sole")
parser.add_argument("--commands", type=float, nargs="+", default=[0.0, 0.10, 0.20])
parser.add_argument("--noisy", action="store_true", help="training's sensor noise + encoder offsets on the readings")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--physics-hz", type=int, default=1000, help="MuJoCo step rate (see mujoco_backend: ankle linkage)")
parser.add_argument("--actuator", choices=["implicit_pd", "lx16a"], default="implicit_pd",
                    help="lx16a: the fitted LX-16A servo model (walk_rl/actuator.py), nominal parameters")
parser.add_argument("--robot-config", default=None, help="robot_config.yaml to build the robot from (default: the current one)")
parser.add_argument("--ankle-constraint", choices=["equality", "tendon_limit"], default="equality",
                    help="ankle linkage as a tendon equality, or as a zero-width tendon length limit (Newton has no tendon equality)")
parser.add_argument("--bus-voltage", type=float, nargs="+", default=[None], help="lx16a: servo supply(s) at no load [V] (default: robot_config measured_voltage_v)")
parser.add_argument("--supply-sag", action="store_true", help="lx16a: supply sags with the total servo torque (robot_config supply_sag_v_per_nm)")
parser.add_argument("--drive-limit", action="store_true", help="lx16a: motor torque <= datasheet stall while it drives the joint")
parser.add_argument("--imu-delay", type=int, nargs="+", default=[0], help="IMU readings this many control steps old (BNO055 fusion lag)")
parser.add_argument("--out", default=None)
args = parser.parse_args()

run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run


def episode(friction: float, delay: int, cmd: float, tc: float, volt: float | None = None, imu_delay: int = 0) -> dict:
    runner = PolicyRunner(run_dir, args.policy, robot_config=args.robot_config)
    ob = runner.cfg.observation
    noise = vars(ob.noise) if args.noisy else None
    sim = MujocoTrex(friction=friction, control_hz=runner.cfg.sim.control_hz, physics_hz=args.physics_hz,
                     delay_steps=delay, contact_timeconst_s=tc, noise=noise, encoder_offset_rad=ob.encoder_offset_rad if args.noisy else 0.0,
                     seed=args.seed, max_tilt_deg=runner.cfg.termination.max_tilt_deg,
                     min_head_height_m=runner.cfg.termination.min_head_height_m,
                     ground_contact_force_n=runner.cfg.termination.ground_contact_force_n, actuator=args.actuator,
                     robot_config=args.robot_config, ankle_constraint=args.ankle_constraint, bus_voltage_v=volt,
                     supply_sag=args.supply_sag, drive_torque_limit=args.drive_limit, imu_delay_steps=imu_delay)
    runner.reset_state(np.zeros(8, dtype=np.float32), phase=0.0)
    r0 = sim.read()
    runner.latch_heading(r0.yaw)
    p0, yaw0 = sim.base_pos(), sim.yaw()
    steps = int(round(args.seconds * runner.cfg.sim.control_hz))
    vx_sum, n, res = 0.0, 0, 0.0
    warm = int(round(2.0 * runner.cfg.sim.control_hz))  # gait statistics skip the first 2 s
    qs, rolls, fz = [], [], []
    feet = sorted(sim.feet)
    contact = Contact(sim, feet)
    for k in range(steps):
        r = sim.read()
        if k >= warm:
            qs.append(np.asarray(r.q, dtype=np.float64))
            g = r.gravity_b
            rolls.append(math.degrees(math.atan2(g[1], -g[2])))
            fz.append(sim.data.xpos[feet, 2].copy())
            contact.sample()
        goal = runner.step(r, cmd)
        sim.write(goal)
        v = sim.base_vel_world()
        yaw = sim.yaw()
        vx_sum += math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
        res = max(res, sim.ankle_residual_deg())
        n += 1
        if sim.failed():
            break
    p = sim.base_pos()
    d = p - p0
    fwd = (math.cos(yaw0) * d[0] + math.sin(yaw0) * d[1])
    side = (-math.sin(yaw0) * d[0] + math.cos(yaw0) * d[1])
    dyaw = math.degrees(math.atan2(math.sin(sim.yaw() - yaw0), math.cos(sim.yaw() - yaw0)))
    return {"friction": friction, "delay": delay, "contact_tc": tc, "cmd": cmd, "bus_voltage": volt, "imu_delay": imu_delay, "survived_s": round(sim.time, 2),
            "fell": sim.failed(), "vx": vx_sum / max(n, 1), "forward_m": fwd, "side_m": side, "yaw_deg": dyaw,
            "ankle_res_max_deg": res, **gait_stats(runner, qs, rolls, fz), **contact.stats(fwd)}


class Contact:
    """Per control step after the warm-up: stance-sole slide, swing clearance of the lowest sole point,
    servo torque over the datasheet stall, total servo torque (current proxy) and supply voltage."""
    IN_CONTACT_N = 0.5  # light scuffing counts as contact (slide), a foot under this is in swing

    def __init__(self, sim, feet):
        self.sim, self.feet = sim, feet
        self.prev = {b: None for b in feet}  # (sole centre, in contact) at the previous step
        self.slide = 0.0
        self.swing_max = {b: None for b in feet}  # highest lowest-sole-point of the current swing
        self.clear = []  # one value per finished swing [m]
        self.swing_n, self.n = 0, 0
        self.over, self.tsum, self.vmin = [], [], []

    def sample(self):
        sim, F = self.sim, self.sim.foot_forces()
        for b in self.feet:
            c, on = sim.sole_center_w(b), F[b] > self.IN_CONTACT_N
            pc = self.prev[b]
            if on and pc is not None and pc[1]:
                self.slide += float(np.linalg.norm((c - pc[0])[:2]))
            if on:
                if self.swing_max[b] is not None:
                    self.clear.append(self.swing_max[b])
                self.swing_max[b] = None
            else:
                self.swing_n += 1
                z = sim.sole_min_z(b)
                self.swing_max[b] = z if self.swing_max[b] is None else max(self.swing_max[b], z)
            self.prev[b] = (c, on)
        self.n += 1
        st = sim.step_stats
        if st:
            self.over.append(st["over_stall"]), self.tsum.append(st["tau_sum"]), self.vmin.append(st["v_min"])

    def stats(self, forward_m: float) -> dict:
        if self.n < 10:
            return {}
        out = {"stance_slide_m": round(self.slide, 3),
               "slide_per_forward": round(self.slide / max(abs(forward_m), 0.05), 2),
               "swing_clearance_mm": round(1000 * float(np.median(self.clear)), 1) if self.clear else 0.0,
               "swing_share": round(self.swing_n / (self.n * len(self.feet)), 3)}
        if self.over:
            out.update({"over_stall_share": round(float(np.mean(self.over)), 3),
                        "tau_sum_mean_nm": round(float(np.mean(self.tsum)), 2),
                        "tau_sum_p95_nm": round(float(np.percentile(self.tsum, 95)), 2),
                        "v_min": round(float(np.min(self.vmin)), 2)})
        return out


def gait_stats(runner: PolicyRunner, qs: list, rolls: list, fz: list) -> dict:
    """Posture of the gait (deg): mean angle per joint, sustained left - right femur and tibia
    difference, fraction of time each joint sits within 3 deg of its reachable goal range end,
    roll 5-95 % spread, foot lift (95th percentile of foot body height above its 5th, mm), mean joint
    speed from the control-rate samples (deg/s, the same finite difference as the real-robot log)."""
    if len(qs) < 10:
        return {}
    q = np.degrees(np.stack(qs))
    lo = np.degrees(np.maximum(runner.joint_lo, runner.act_center - runner.act_scale))
    hi = np.degrees(np.minimum(runner.joint_hi, runner.act_center + runner.act_scale))
    at_lim = ((q < lo + 3.0) | (q > hi - 3.0)).mean(axis=0)
    names = [n.replace("_joint", "") for n in runner.act_names]
    fl, fr = runner.fem_a
    tl, tr = runner.tib_a
    return {"q_mean_deg": dict(zip(names, q.mean(axis=0).round(1).tolist())),
            "at_limit": dict(zip(names, at_lim.round(3).tolist())),
            "femur_lr_deg": float((q[:, fl] - q[:, fr]).mean()), "tibia_lr_deg": float((q[:, tl] - q[:, tr]).mean()),
            "roll_spread_deg": float(np.percentile(rolls, 95) - np.percentile(rolls, 5)),
            "foot_lift_mm": (1000 * (np.percentile(fz, 95, axis=0) - np.percentile(fz, 5, axis=0))).round(1).tolist(),
            "joint_speed_deg_s": dict(zip(names, (np.abs(np.diff(q, axis=0)).mean(axis=0) / runner.dt).round(1).tolist()))}


t0 = time.time()
rows = []
print(f"[sim2sim] run={args.run} policy={args.policy} engine=MuJoCo seconds={args.seconds} noisy={args.noisy}")
print("[sim2sim] tc     fric delay  cmd | survived      | vx (%cmd)       | forward | side   | yaw    | ankle res max")
import itertools  # noqa: E402

for tc, fr, dl, volt, di, c in itertools.product(args.contact_tc, args.frictions, args.delays, args.bus_voltage,
                                                 args.imu_delay, args.commands):
            r = episode(fr, dl, c, tc, volt, di)
            rows.append(r)
            pct = f"({100 * r['vx'] / c:4.0f}%)" if c > 0 else "       "
            surv = f"{r['survived_s']:5.1f}s" + (f" {r['fell']}" if r["fell"] else " ok")
            print(f"[sim2sim] {tc:.3f}  {fr:.1f}  {dl}   {c:.2f} | {surv:13s} | {r['vx']:+.3f} {pct} | {r['forward_m']:+6.2f}m | "
                  f"{r['side_m']:+.2f}m | {r['yaw_deg']:+6.1f} | {r['ankle_res_max_deg']:.2f} deg", flush=True)
walking = [r for r in rows if r["cmd"] > 0 and r["contact_tc"] == min(args.contact_tc)]
full = sum(r["fell"] is None for r in rows)
ok = all(r["fell"] is None for r in walking)
print(f"[sim2sim] survived the full {args.seconds:.0f} s: {full}/{len(rows)} episodes; "
      f"{'OK' if ok else 'FAILED'} (pass: every walking condition on the stiffest floor survives) [{time.time() - t0:.0f}s wall]")
out = Path(args.out) if args.out else run_dir / f"sim2sim_{Path(args.policy).stem}{'_noisy' if args.noisy else ''}{'_lx16a' if args.actuator == 'lx16a' else ''}.json"
out.write_text(json.dumps({"run": args.run, "policy": args.policy, "seconds": args.seconds, "noisy": args.noisy,
                           "supply_sag": args.supply_sag, "drive_limit": args.drive_limit,
                           "rows": rows, "pass": ok, "robot_config": args.robot_config,
                           "provenance": provenance(args.robot_config)}, indent=1))
print(f"[sim2sim] wrote {out}")
