"""Refit the LX-16A servo model with the real robot's walking logs (loaded) plus the unloaded logs.

    $LAB walk_rl/fit_actuator_walk.py              # score the current robot_config model, then fit
    $LAB walk_rl/fit_actuator_walk.py --score 1 1 1 1

Data (assets/robots/trex/servo_logs/):
  walk_v9m_20261020_222622_floor.npz  pi_runner/walk_live.py, v9m on the floor, 0.10 m/s (loaded)
  walk_v9m_20261020_222021_hang.npz   the same policy with the robot held in the air (unloaded)
  servo_identify / servo_sweep_20261018.json  walk_rl/fit_actuator.py data (each servo alone, in the air)
The real servo goals are replayed open loop in MuJoCo (pi_runner/mujoco_backend.py, lx16a; base frozen
for the hanging log) and the sim joint angles compared with the real ones on the 6 leg joints (floor
3-9 s, before open-loop drift dominates; hanging 3-13 s). The 10-18 segments use fit_actuator.py's
one-joint simulation. Fitted: scales on the current robot_config actuator_model for stall torque, kp,
no-load speed and Coulomb friction (differential evolution on the mean of the three RMS).
Copy the result into robot_config.yaml actuator_model by hand.
"""
import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import walk_rl.fit_actuator as fa  # noqa: E402
import pi_runner.mujoco_backend as mb  # noqa: E402
from walk_rl.actuator import LX16AParams, lx16a_torque, ramp_reference  # noqa: E402

LOGS = PROJECT_ROOT / "assets" / "robots" / "trex" / "servo_logs"
J = 8
NAMES = ["stall", "kp", "no_load", "coulomb"]


def load_walk(name: str, t0: float, t1: float):
    R = np.load(LOGS / name)["rows"]
    t = R[:, 0]
    return R[:, 3 + J:3 + 2 * J], R[:, 3:3 + J], (t > t0) & (t < t1)


FLOOR = load_walk("walk_v9m_20261020_222622_floor.npz", 3, 9)
HANG = load_walk("walk_v9m_20261020_222021_hang.npz", 3, 13)
SEGS, VOLTS = fa.load_segments()
P = LX16AParams.from_robot_config(fa.rc)


def replay(x, data, hang: bool) -> np.ndarray:
    """Per-leg-joint RMS (deg) of sim q - real q, real goals replayed open loop."""
    s, k, nl, c = x
    m = mb.MujocoTrex(friction=0.5, actuator="lx16a")
    m.lx.stall *= s; m.lx.kp *= k; m.lx.no_load *= nl; m.lx.coulomb *= c
    goals, qr, w = data
    step = mujoco.mj_step
    if hang:  # robot held in the air: base frozen 10 cm up
        m.data.qpos[2] += 0.10
        mujoco.mj_forward(m.model, m.data)
        base = m.data.qpos[0:7].copy()

        def frozen(model, d):
            d.qpos[0:7] = base
            d.qvel[0:6] = 0
            step(model, d)
        mb.mujoco.mj_step = frozen
    try:
        for _ in range(25):
            m.write(np.zeros(J))
        q = []
        for g in goals:
            q.append(m.read().q)
            m.write(g)
    finally:
        mb.mujoco.mj_step = step
    e = np.degrees(np.array(q) - qr)[w][:, 2:]
    return np.sqrt((e ** 2).mean(0))


def segment_sim(seg, x, dt=0.005) -> np.ndarray:
    s, k, nl, c = x
    stall, no_load = P.stall * VOLTS / P.v_ref * s, P.no_load * VOLTS / P.v_ref * nl
    Jt, n = seg["J"] + P.armature, int(seg["t"][-1] / dt) + 2
    th, w, out = seg["p0"], 0.0, np.empty(n)
    period = max(seg["move"], 1e-9)
    for i in range(n):
        ref = ramp_reference(seg["p0"], seg["tgt"], (i + 1) * dt, period, P.latency, np.clip)
        tau = lx16a_torque(ref, th, w, P.kp * k, P.deadband, P.coulomb * c, stall, no_load, P.armature, dt, np) \
            + seg["g0"] + seg["gk"] * (th - seg["mid"])
        w += tau / Jt * dt
        th += w * dt
        out[i] = th
    return np.interp(seg["t"], (np.arange(n) + 1) * dt, out)


def scores(x) -> tuple[float, float, float]:
    e = np.concatenate([(segment_sim(sg, x) - sg["p"])[sg["mask"]] for sg in SEGS])
    old = math.degrees(math.sqrt(np.mean(e ** 2)))
    f, h = replay(x, FLOOR, False), replay(x, HANG, True)
    return old, float(np.sqrt((f ** 2).mean())), float(np.sqrt((h ** 2).mean()))


def objective(x) -> float:
    return sum(scores(x)) / 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", type=float, nargs=4, metavar=("STALL", "KP", "NO_LOAD", "COULOMB"),
                    help="only score these scales")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    fmt = lambda x: "RMS deg: 10-18 logs {:.2f}, floor walk {:.2f}, hanging walk {:.2f}".format(*scores(x))  # noqa: E731
    print("current robot_config model:", fmt([1, 1, 1, 1]))
    if a.score:
        print(f"scales {a.score}:", fmt(a.score))
        return
    r = differential_evolution(objective, [(0.5, 4), (0.2, 2), (0.5, 2.5), (0.2, 2)], seed=0, maxiter=25, popsize=10,
                               workers=a.workers, updating="deferred", polish=False, x0=[1, 1, 1, 1])
    x = r.x
    print("fit scales on the current values: " + ", ".join(f"{n} x{v:.3f}" for n, v in zip(NAMES, x)))
    print("fitted:", fmt(x))
    am = fa.rc.raw["actuator_model"]
    print(f"-> kp_duty_per_rad {am['kp_duty_per_rad'] * x[1]:.3g}, coulomb_friction_nm {am['coulomb_friction_nm'] * x[3]:.3g}, "
          f"no_load_speed_scale {am['no_load_speed_scale'] * x[2]:.3g}, "
          f"stall_torque_scale {am.get('stall_torque_scale', 1.0) * x[0]:.3g}")


if __name__ == "__main__":
    main()
