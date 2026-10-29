"""Servo-bus current budget (CPU, MuJoCo): total LX-16A current over a policy's gait vs the supply ceiling.

Supply (Sachin, 2026-10-19): TalentCell YB1206000, 11.1 V nominal (12.6-9 V), 3 A max output, feeding a
DC-DC buck-boost that holds the servo bus at ~7.1 V. So the bus current is capped at
    I_bus_max = 3 A x V_batt x efficiency / 7.1 V
(efficiency ASSUMED 0.88; ~4.1 A at 11.1 V, ~3.3 A at 9 V). The Pi's supply path is unknown (if it hangs on
the same battery, it takes another ~0.3-0.6 A of that budget).

Servo current model (ESTIMATE): I = I_noload + I_stall x |tau| / stall(V), tau the fitted model's output
torque. Datasheet values disagree: no-load 0.1 A; stall 2.4-3 A (Hiwonder manual) or 1.1-1.9 A (resellers),
so 1.5 / 2.4 / 3.0 A are reported.

    python redesign/current_budget.py --run v9d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pi_runner.mujoco_backend as mb  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402
from walk_rl.actuator import lx16a_torque, ramp_reference  # noqa: E402

I_NOLOAD = 0.1
I_STALL = [1.5, 2.4, 3.0]
EFF = 0.88


def run(run_name: str, cmd: float, seconds: float = 30.0, warmup: float = 4.0, friction: float = 0.5):
    runner = PolicyRunner(ROOT / "outputs" / "walk_runs" / run_name)
    tc = runner.cfg.termination
    sim = mb.MujocoTrex(friction=friction, control_hz=runner.cfg.sim.control_hz, actuator="lx16a",
                        max_tilt_deg=tc.max_tilt_deg, min_head_height_m=tc.min_head_height_m,
                        ground_contact_force_n=tc.ground_contact_force_n)
    runner.reset_state(np.zeros(8, dtype=np.float32))
    runner.latch_heading(sim.read().yaw)
    m, d = sim.model, sim.data
    p, v = sim.lx, sim.lx.v_nominal / sim.lx.v_ref
    stall = p.stall * v
    fr = []  # |tau| / stall per substep, (N, 8)
    for k in range(int(seconds * 25)):
        goal = runner.step(sim.read(), cmd)
        sim.queue.append(np.asarray(goal, dtype=np.float64).copy())
        tgt, st0 = sim.queue[0], sim.goal
        for i in range(sim.decimation):
            q_ref = ramp_reference(st0, tgt, (i + 1) * m.opt.timestep, sim.control_period, p.latency, np.clip)
            d.ctrl[sim.ctrl_idx] = lx16a_torque(q_ref, d.qpos[sim.qadr], d.qvel[sim.dofadr], p.kp, p.deadband, p.coulomb,
                                                stall, p.no_load * v, p.armature, m.opt.timestep, np)
            if k >= warmup * 25:
                fr.append(np.abs(d.ctrl[sim.ctrl_idx]) / stall)
            mujoco.mj_step(m, d)
        sim.goal = tgt.copy()
        sim.time += sim.decimation * m.opt.timestep
        if sim.failed():
            break
    fr = np.array(fr)
    res = {"cmd": cmd, "survived_s": round(sim.time, 2), "fell": sim.failed()}
    for ist in I_STALL:
        tot = (I_NOLOAD + ist * np.minimum(fr, 1.0)).sum(1)  # 8 servos, per 1 ms substep
        win = tot[: len(tot) // 20 * 20].reshape(-1, 20).mean(1)  # 20 ms averages (a supply's short-term limit)
        res[f"I_stall_{ist}"] = {"mean_a": float(tot.mean()), "p95_a": float(np.percentile(tot, 95)), "max_20ms_a": float(win.max()),
                                 "frac_20ms_over_4.1a": float((win > 3 * 11.1 * EFF / 7.1).mean()),
                                 "frac_20ms_over_3.3a": float((win > 3 * 9.0 * EFF / 7.1).mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v9d")
    a = ap.parse_args()
    rows = [run(a.run, c) for c in (0.0, 0.10, 0.20, 0.25)]
    for r in rows:
        print(f"[current] {a.run} cmd {r['cmd']:.2f}: " + "; ".join(
            f"Istall {k[8:]} A: mean {v['mean_a']:.2f} p95 {v['p95_a']:.2f} max20ms {v['max_20ms_a']:.2f} A, "
            f">4.1 A {100 * v['frac_20ms_over_4.1a']:.1f} %, >3.3 A {100 * v['frac_20ms_over_3.3a']:.1f} %"
            for k, v in r.items() if k.startswith("I_stall")))
    (ROOT / "outputs" / "redesign" / f"current_budget_{a.run}.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
