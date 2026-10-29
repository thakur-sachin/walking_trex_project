"""Compare a real floor walk with MuJoCo, and estimate the real foot slip from the joint angles.

    $LAB pi_runner/replay_real_walk.py assets/robots/trex/servo_logs/walk_v9o_20261021_212933_floor.npz \
        --run v9o --real-distance 0.46 --frictions 0.3 0.5 0.8

Three walks over the same schedule:
  real          the pi_runner/walk_live.py log (servo angles, IMU gravity and yaw; no position)
  mujoco open   the real servo goals replayed open loop (lx16a servo model)
  mujoco closed the policy itself in the deployed loop (pi_runner/runner.py), with the log's command schedule

No-slip odometry: the joint angles and the IMU attitude give each sole centre (robot_config foot_sole)
relative to the torso. If the stance foot does not slide, the torso moves by minus the stance sole's
motion. The stance foot is taken as the lower sole (the robot has no foot contact sensors), after removing
each sole's own floor level (its 10th-percentile height while walking; a hip offset lifts one sole
for the whole walk). Distances are path lengths along the torso heading (summed tick by tick in the
heading frame), so a turning walk still counts its forward motion. Summed over
the walk this is the distance the robot WOULD cover without slip; the measured distance divided by it is
the fraction of the leg motion that became progress (1 = no slip). The estimator runs on the MuJoCo walks
too, where the true distance and contacts are known, to check it.
Step length: at each change of stance foot, the forward distance from the old to the new stance sole.

Limit (2026-10-21, v9o log): the soles rise only a few mm, so the stance choice depends mostly on the IMU
attitude, not on the joint angles. In MuJoCo the estimate is 0.73-0.98 of the true distance, but on the
real log it changes from -0.5 m to +0.4 m when the IMU is shifted by 1-2 ticks (the real roll lags the
simulated roll of the same joint motion by 40-80 ms). It does not measure real slip; a foot marker in the
side video does. The open- and closed-loop comparisons (distance, yaw, joint RMS) are the useful output.
"""
import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pi_runner.mujoco_backend import MujocoTrex  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402

J = 8
SIDES = ("left", "right")
STANCE_MARGIN_M = 0.002  # a sole must be this much lower than the other to count as the only stance foot


def load_log(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    R, cols = d["rows"], [str(c) for c in d["cols"]]
    c = lambda n: R[:, cols.index(n)]  # noqa: E731
    i = cols.index("q_head_joint")
    yaw = c("yaw")
    return {"t": c("t"), "cmd": c("cmd"), "q": R[:, i:i + J], "goal": R[:, i + J:i + 2 * J],
            "gravity": np.stack([c("gx"), c("gy"), c("gz")], 1), "yaw": np.unwrap(yaw)}


class Kinematics:
    """Sole centres in a torso-centred, gravity-aligned frame from joint angles + attitude."""

    def __init__(self, sim: MujocoTrex):
        self.m = sim.model
        self.d = mujoco.MjData(self.m)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.qadr = sim.qadr
        self.names = sim.act_names
        self.ankle = {s: self.m.jnt_qposadr[jid(f"{s}_ankle_joint")] for s in SIDES}
        self.foot = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in SIDES]
        self.sole_r = np.array(sim.rc.foot_sole_center_m())

    def soles(self, q: np.ndarray, gravity_b: np.ndarray, yaw: float) -> np.ndarray:
        """(2, 3) left/right sole centres, torso origin at 0, world z up, x along the yaw heading."""
        tilt = Rotation.align_vectors([[0.0, 0.0, -1.0]], [gravity_b / np.linalg.norm(gravity_b)])[0]
        R = Rotation.from_euler("z", yaw) * tilt
        x, y, z, w = R.as_quat()
        self.d.qpos[:] = 0
        self.d.qpos[3:7] = [w, x, y, z]
        self.d.qpos[self.qadr] = q
        for s in SIDES:  # ankle linkage: ankle = femur - tibia
            self.d.qpos[self.ankle[s]] = q[self.names.index(f"{s}_femur_joint")] - q[self.names.index(f"{s}_tibia_joint")]
        mujoco.mj_kinematics(self.m, self.d)
        return np.array([self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.sole_r for b in self.foot])


def odometry(kin: Kinematics, q, gravity, walk) -> dict:
    """No-slip torso travel (along the heading) and step lengths over the ticks in `walk`."""
    S = np.array([kin.soles(q[i], gravity[i], 0.0) for i in range(len(q))])
    z = S[:, :, 2] - np.percentile(S[walk][:, :, 2], 10, axis=0)
    dz = z[:, 0] - z[:, 1]
    stance = np.where(dz < -STANCE_MARGIN_M, 0, np.where(dz > STANCE_MARGIN_M, 1, -1))  # -1 = both
    travel, steps, last = np.zeros(2), {0: [], 1: []}, None
    idx = np.nonzero(walk)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        k = stance[a] if stance[a] == stance[b] else -1
        if k >= 0:
            travel -= (S[b, k, :2] - S[a, k, :2])
        else:  # both feet low or a change of stance: the mean of the two soles
            travel -= (S[b, :, :2].mean(0) - S[a, :, :2].mean(0))
        if stance[b] >= 0:
            if last is not None and stance[b] != last:
                steps[stance[b]].append(S[b, stance[b], 0] - S[b, last, 0])
            last = stance[b]
    return {"forward": travel[0], "side": travel[1], "steps": steps, "stance": stance[walk], "soles": S}


def run_mujoco(log: dict, mode: str, friction: float, run_dir: Path | None) -> dict:
    sim = MujocoTrex(friction=friction, actuator="lx16a")
    runner = None
    if mode == "closed":
        runner = PolicyRunner(run_dir)
        runner.reset_state(np.zeros(J, dtype=np.float32), phase=0.0)
        runner.latch_heading(sim.read().yaw)
    foot_ids = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in SIDES]
    out = {k: [] for k in ("q", "gravity", "yaw", "pos", "contact")}
    f6 = np.zeros(6)
    fell = None
    for i in range(len(log["t"])):
        r = sim.read()
        g = runner.step(r, float(log["cmd"][i])) if runner else log["goal"][i]
        out["q"].append(r.q); out["gravity"].append(r.gravity_b); out["yaw"].append(r.yaw)
        out["pos"].append(sim.base_pos())
        ff = np.zeros(2)
        for c in range(sim.data.ncon):
            con = sim.data.contact[c]
            other = con.geom2 if con.geom1 == sim.floor else con.geom1 if con.geom2 == sim.floor else None
            if other is not None and sim.model.geom_bodyid[other] in foot_ids:
                mujoco.mj_contactForce(sim.model, sim.data, c, f6)
                ff[foot_ids.index(sim.model.geom_bodyid[other])] += abs(f6[0])
        out["contact"].append(ff > 1.0)
        sim.write(g)
        if fell is None and sim.failed():
            fell = f"{sim.failed()} at {log['t'][i]:.1f} s"
            break
    out = {k: np.array(v) for k, v in out.items()}
    out["yaw"] = np.unwrap(out["yaw"])
    out["fell"] = fell
    out["sim"] = sim
    return out


def tracking(q_a, q_b, names, walk) -> str:
    n = min(len(q_a), len(q_b))
    e = np.degrees(q_a[:n] - q_b[:n])[walk[:n]]
    return " ".join(f"{nm.replace('_joint', '')} {math.sqrt((x ** 2).mean()):.1f}" for nm, x in zip(names, e.T))


def report(name: str, od: dict, true_fwd: float | None, contact=None) -> dict:
    sL, sR = od["steps"][0], od["steps"][1]
    st = od["stance"]
    line = (f"{name:22s} no-slip {od['forward']:+.3f} m (side {od['side']:+.3f})"
            f" | steps L {np.mean(sL) * 1000 if sL else float('nan'):5.1f} mm x{len(sL)}"
            f"  R {np.mean(sR) * 1000 if sR else float('nan'):5.1f} mm x{len(sR)}"
            f" | stance L/R/both {(st == 0).mean():.2f}/{(st == 1).mean():.2f}/{(st == -1).mean():.2f}")
    eff = None
    if true_fwd is not None:
        eff = true_fwd / od["forward"] if abs(od["forward"]) > 1e-3 else float("nan")
        line += f" | measured {true_fwd:+.3f} m -> progress/no-slip {eff:.2f}"
    if contact is not None:
        n = min(len(contact), len(st))
        lone = st[:n] >= 0
        agree = (contact[:n][lone, :][np.arange(lone.sum()), st[:n][lone]]).mean() if lone.any() else float("nan")
        line += f" | lower sole in contact {agree:.2f}"
    print(line)
    return {"no_slip_m": od["forward"], "measured_m": true_fwd, "progress_ratio": eff,
            "step_L_mm": float(np.mean(sL) * 1000) if sL else None, "step_R_mm": float(np.mean(sR) * 1000) if sR else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--run", help="policy run for the closed-loop walk (outputs/walk_runs/<run>)")
    ap.add_argument("--real-distance", type=float, help="measured forward distance during the walk (pens), m")
    ap.add_argument("--frictions", type=float, nargs="+", default=[0.5])
    a = ap.parse_args()

    log = load_log(a.log)
    t, cmd = log["t"], log["cmd"]
    walk = cmd > 0
    print(f"{a.log.name}: {walk.sum() * 0.04:.0f} s at {cmd[walk].max():.2f} m/s; yaw change while walking "
          f"{math.degrees(log['yaw'][walk][-1] - log['yaw'][walk][0]):+.1f} deg")
    kin_sim = MujocoTrex(actuator="lx16a")
    kin = Kinematics(kin_sim)
    names = kin_sim.act_names
    od = odometry(kin, log["q"], log["gravity"], walk)
    res = {"real": report("real", od, a.real_distance)}
    for fr in a.frictions:
        for mode in ("open", "closed") if a.run else ("open",):
            m = run_mujoco(log, mode, fr, PROJECT_ROOT / "outputs" / "walk_runs" / a.run if a.run else None)
            n = len(m["q"])
            w = walk[:n]
            idx = np.nonzero(w)[0]
            d = np.diff(m["pos"][idx, :2], axis=0)
            h = m["yaw"][idx[:-1]]
            fwd = float((np.cos(h) * d[:, 0] + np.sin(h) * d[:, 1]).sum())  # path along the heading
            o = odometry(kin, m["q"], m["gravity"], w)
            tag = f"mujoco {mode} mu {fr}" + (f" FELL {m['fell']}" if m["fell"] else "")
            print(f"{tag}: yaw change {math.degrees(m['yaw'][idx[-1]] - m['yaw'][idx[0]]):+.0f} deg")
            res[tag] = report(tag, o, fwd, m["contact"][w])
            print(f"{'':22s} |sim q - real q| RMS deg: {tracking(m['q'], log['q'], names, w)}")
    return res


if __name__ == "__main__":
    main()
