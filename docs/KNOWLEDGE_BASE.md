# Trex knowledge base

The current state of the Trex walking robot project: what exists, how it fits together and how to use it. Last updated 2026-10-19 (branch `pi-bringup`).

The full development record (experiments, failures, decisions and the reasons for them) is in [PROJECT_HISTORY.md](PROJECT_HISTORY.md).

---

## 1. The robot

Trex is a 3D-printed (PLA) bipedal dinosaur robot. A walking policy is trained in simulation and runs on the robot's Raspberry Pi.

| | |
|---|---|
| Actuators | 8 Hiwonder LX-16A serial bus servos: head, tail, and hip / femur / tibia on each leg |
| Servo datasheet | 1.667 N·m stall and 375 °/s no-load at 7.4 V, 0.24° resolution, 115200 baud half-duplex bus, position / voltage / temperature feedback |
| Ankles | passive, driven by a double 4-bar parallelogram that keeps each foot parallel to the torso: `ankle = femur − tibia` |
| Sensors | BNO055 IMU in the torso, and the 8 servo position readings. No foot contact sensors, no camera |
| Compute | Raspberry Pi 4 Model B, 8 GB, no GPU |
| Power | 2S battery pack (7.1 V at rest), glued under the torso plate |
| Servo adapter | Hiwonder BusLinker-style USB board (CH340, `1a86:7523`) |
| Mass | 1.778 kg in the model; `base_link` (torso with battery, Pi, hip servos) is a 1.0 kg estimate until weighed |
| Floor | indoor wood, PLA soles |

### Joints

| joint | servo id | range (model) | positive angle moves |
|---|---|---|---|
| `head_joint` | 7 | ±45° | nose down |
| `tail_joint` | 8 | ±45° | tail tip up |
| `left_hip_joint` | 1 | ±30° | left foot inward |
| `right_hip_joint` | 4 | ±30° | right foot inward |
| `left_femur_joint` | 2 | ±45° | left foot forward |
| `right_femur_joint` | 5 | ±45° | right foot forward |
| `left_tibia_joint` | 3 | ±45° | left foot back |
| `right_tibia_joint` | 6 | ±45° | right foot back |

- A positive hip angle moves **either** foot inward. The hips are mirrored in raw sign, so never assume that left and right share a sign. The measured table is in [JOINT_CONVENTIONS.md](JOINT_CONVENTIONS.md), produced by `scripts/joint_sign_convention_experiment.py`.
- The joint ranges are working values. The real mechanical ranges have not been measured.
- The all-zero pose is the robot standing straight. It is not statically stable: balance comes from the policy.
- The body frame is x forward (tail to head), y left, z up.

### Hardware calibration (`assets/robots/trex/hardware.yaml`)

| joint | id | dir | zero tick | stored servo limits (ticks) |
|---|---|---|---|---|
| head | 7 | +1 | 448 | 248–648 |
| tail | 8 | +1 | 495 | 295–695 |
| left hip | 1 | −1 | 330 | 192–468 |
| right hip | 4 | +1 | 409 | 271–547 |
| left femur | 2 | +1 | 200 | 0–400 |
| right femur | 5 | +1 | 408 | 208–608 |
| left tibia | 3 | +1 | 386 | 186–586 |
| right tibia | 6 | +1 | 344 | 144–544 |

- Conversion: `tick = zero_tick + dir · deg(q) · 1000/240`. Ticks run 0–1000 over 240°.
- The stored limits are each joint's range plus 3°, written to the servos' EEPROM. The servos clamp every command to them on their own.
- IMU mounting: `sensor_to_body = [[0,1,0],[-1,0,0],[0,0,1]]`, i.e. chip +y is forward and chip +x is right.
- Gravity convention: g points the way gravity pulls, so nose down gives g_x > 0 and left side down gives g_y > 0.

---

## 2. Repository layout

```
assets/robots/trex/
  Trex.urdf, meshes/          robot geometry (11 links, 10 joints)
  robot_config.yaml           every physical parameter: masses, inertia model, joint limits,
                              servo spec, fitted servo model, linkage, sim drive gains
  hardware.yaml               Pi wiring and calibration: bus, servo id/dir/zero, IMU mounting
  servo_logs/                 real servo moves used to fit the servo model
common/robot_config.py        loader for robot_config.yaml (use it, never hardcode numbers)
walk_rl/                      the walking task, Isaac Sim stack (fallback), shared task logic
  task_core.py                all task logic shared by both simulators
  env.py                      Isaac Sim backend (conda env isaac6)
  actuator.py                 LX-16A servo model (torch and numpy)
  walk_task.yaml              task / reward / randomization / PPO defaults
  experiments/*.yaml          per-run overrides (v8e, v9p, v9a, v9b, v9c)
  train.py, ppo.py            in-repo asymmetric PPO
  evaluate.py                 fixed evaluation suite
  play.py, render_gait.py     live viewer, offline gait capture
  fit_actuator.py             fits the servo model to servo_logs/
  symmetry.py                 left/right mirror map (augmentation)
  trace_env.py, smoke_env.py  regression trace, sanity check
  test_*.py                   ankle linkage and symmetry tests
walk_lab/                     the same task on Isaac Lab 3.0 (conda env isaaclab6), rsl_rl trainer
pi_runner/                    everything that runs on, or stands in for, the robot
  runner.py, policy.py        the deployed control loop (NumPy, float32)
  hardware.py                 real backend: BNO055 + LX-16A bus
  bno055.py, lx16a.py         device drivers
  mujoco_backend.py           MuJoCo backend (sim2sim)
  sim2sim.py, render_mujoco.py  MuJoCo robustness sweep, MuJoCo video
  hw_check.py                 Pi bring-up checks (never moves a servo)
  servo_identify.py, servo_hold.py, servo_limits.py   servo tools (move servos)
  deploy.sh, pi_setup.sh      PC → Pi deployment, Pi system setup
  test_obs_parity.py          runner vs simulator parity test
scripts/                      robot import pipeline, prim audit, joint-sign experiment
docs/                         this file, PROJECT_HISTORY.md, JOINT_CONVENTIONS.md
outputs/                      (gitignored) runs, evaluations, videos, logs
```

---

## 3. Environments

### PC
- RTX 4060 Laptop (8 GB). The NVIDIA driver must be ≥ 550.90.07, or the RTX renderer crashes.
- **conda env `isaac6`**: Python 3.12, Isaac Sim 6.0.1 (pip), torch 2.11+cu128. Used for the Isaac Sim stack.
- **conda env `isaaclab6`**: a clone of isaac6 plus Isaac Lab 3.0.0-beta2.patch1 (from `~/IsaacLab3`), rsl-rl-lib 5.0.1, MuJoCo 3.8 and SciPy. Used for the Lab stack, MuJoCo and fitting.
- Every Isaac script needs `export OMNI_KIT_ACCEPT_EULA=YES`.
- The GPU is shared with other jobs. Stop your own processes by exact PID.
- Run outputs go to `outputs/walk_runs/<name>/`:
  - `train_log.jsonl`;
  - `checkpoint.pt` (latest) and `best.pt`;
  - `checkpoints/`;
  - `policy_numpy.npz`, the deployable actor;
  - the merged `walk_task.yaml`;
  - `eval_*.json`, `sim2sim_*.json`.

### Raspberry Pi
| | |
|---|---|
| Access | `ssh pi@trex.local`: key auth, passwordless sudo, static IP on your Wi-Fi (NetworkManager) |
| OS | Raspberry Pi OS Bookworm 64-bit, Python 3.11 |
| Code | `~/trex`, venv `~/trex/.venv` (numpy, pyyaml, pyserial, smbus2; no torch) |
| IMU bus | bit-banged I²C bus 3 on GPIO2/3 (`dtoverlay=i2c-gpio,bus=3`), hardware I²C off, BNO055 at 0x28 in IMUPLUS mode |
| Servo bus | `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` (`port: auto` finds it) |
| System | CPU governor `performance` at boot, serial console off, UART on |
| Logs | `~/trex/logs/` |

---

## 4. The walking task

### Interface
| | |
|---|---|
| Action (8) | position targets for the 8 servos, `target = a · joint_limit`, a ∈ [−1, 1]. Ankles follow the linkage |
| Actor observation (105) | the 3 latest frames of 35 values: IMU gravity (3), gyro (3), speed command (1), 8 servo positions / limit (quantised to 0.24°), 8 finite-difference joint velocities, previous action (8), gait clock sin/cos (2, zero when standing), heading error sin/cos (2) |
| Critic observation (50) | one clean frame plus privileged state: base velocity, foot contacts, foot heights, servo torques |
| Command | forward speed 0.05–0.25 m/s; 0 (stand) 20% of the time; resampled every 5 s |
| Heading | error to a reference yaw latched from the BNO055 fused yaw when walking starts |
| Control | 25 Hz policy, 200 Hz physics, 20 s episodes |

The observation layout is stored in each exported policy (`obs_layout` in `policy_numpy.npz`). `pi_runner/runner.py` rebuilds exactly this input.

### Action pipeline (identical in training and on the Pi)
clamp → low-pass (`cmd = 0.5·cmd_prev + 0.5·action`) → scale → quantise to 0.24° → speed-limited goal → joint limits → servo move command with move time = one control period.

### Servo models (`walk_task.yaml actuator.model`)
- `implicit_pd` (default): a PhysX/MuJoCo position drive toward the move-time ramp, capped at stall torque. Used for v1–v9b.
- `lx16a` (v9c): an explicit torque model fitted to the real servos, `walk_rl/actuator.py`. Per physics substep:
  - `q_ref` = the move-time ramp from the previous goal, delayed by `latency`;
  - `e` = `q_ref − q`, minus a deadband;
  - `duty` = `clamp(kp·e, −1, 1)`;
  - `τ = duty·stall(V) − stall(V)/no_load(V)·q̇ − Coulomb friction`;
  - armature is added on each joint.
  - Friction can stop a joint but not reverse it within one substep.
  - Nominal values are in `robot_config.yaml actuator_model` (kp 12.2/rad, deadband 0.41°, latency 4 ms, armature 0.021 kg·m², friction 0.31 N·m, 0.46 × datasheet speed). Per-episode randomization is in `domain_randomization.actuator_model`.
  - Torch (both Isaac stacks) and NumPy (MuJoCo) run the same code.

### Rules and training mechanisms
- **Only the feet may touch the ground.** Ground contact by any other link ends the episode with a penalty.
- **Contact** comes from PhysX contact forces (> 1 N for a foot, > 0.2 N for other links), never from link height.
  - A reading is ignored after a reset until it changes, because PhysX keeps reporting the pre-reset force.
- **Harness (training only):**
  - an upward force of 70% of body weight on the torso, plus a righting torque;
  - full strength until iteration 150, then down to zero by iteration 450.
- **Pushes:** random base-velocity kicks every 4–8 s, ramped in after the harness is gone (v9 configs: up to 0.35 m/s from iterations 500–1200).
- **Randomization:**
  - friction;
  - torso mass ×0.8–1.3, inertia ×0.6–1.6 and centre of mass ±1.5 cm;
  - head/tail mass ±15%;
  - action delay 0–2 steps and IMU delay 0–1 step;
  - encoder offsets ±0.02 rad and sensor noise;
  - heading drift;
  - servo parameters (`lx16a`) or PD gains and torque cap (`implicit_pd`).
- **Rewards** (`walk_task.yaml reward.weights`):
  - speed tracking (exponential and linear), heading tracking, yaw rate;
  - gait clock contact pattern, swing clearance, air time;
  - smoothness (action rate and acceleration), torque and torque saturation, joint limits;
  - upright, base acceleration and rocking, stand-still;
  - foot spacing, foot slip, foot flat;
  - lateral drift (1 s average), roll band;
  - alive and termination.
- **PPO:**
  - asymmetric actor-critic: actor 128-128-64 ELU (small enough for the Pi), critic 256-128-64;
  - LR 5e-4 decaying linearly to 1e-4, with a per-minibatch KL guard;
  - action std capped at 0.5;
  - `best.pt` is chosen only after the harness and push ramps finish.

### The current policy: v9i
v9i is v9d plus a progress gate, trained on the measured masses (2026-10-20):
- `reward.progress_gate` scales the heading, yaw, roll, rocking, acceleration and gait-contact rewards by the forward progress while walking;
- `progress_gate_mode: tracking` makes the gate peak at the commanded speed, so going faster than the command earns nothing more.

Without the gate, every run on the measured model (v9e, v9f, v9g) drifted to standing after the training pushes started: standing earned almost as much reward per step as walking. v9i was trained for 1500 iterations at 4096 envs. Its files are `outputs/walk_runs/v9i/checkpoint.pt` and `policy_numpy.npz`; the config is `walk_rl/experiments/v9i_tracking_gate.yaml`.

| check | result |
|---|---|
| Isaac, fitted servo, randomization on, 20 s, 0–0.25 m/s | survival 0.97–1.00; hard sensors 0.95–0.99 |
| speed (Isaac) | 0.05 → 93%, 0.10 → 86%, 0.15 → 85%, 0.20 → 76%, 0.25 → 67% of the command |
| push recovery | ≥ 0.94 up to 0.3 m/s kicks; 0.74 standing, 0.89 walking at 0.4 m/s |
| MuJoCo, fitted servo, 60 s, friction 0.3/0.5/0.8 × bus delay 0–2 × cmd 0/0.05/0.10/0.20 | 32/36 survive; all 4 falls at 0.20 on friction 0.5/0.8; 0.10: 12/12 at 53–63% of the command |
| roll peak-to-peak while walking | 10–16° |
| runner parity | 0/4800 servo goals differ |

Known limits:
- Use 0.10–0.15 m/s for the first steps. At 0.20 it can fall in MuJoCo.
- MuJoCo speed is about 55–60% of the Isaac speed at the same command; at 0.05 MuJoCo hardly moves.
- Standing puts the tibias at their limit.
- The policy needs the `lx16a` servo model.

The previous candidate, v9d, makes only 51–55% of the command on the measured model (survival 0.86–0.89).

Videos are in `outputs/videos/`:
- `v9i_lx16a_speeds.mp4`: v9i in MuJoCo, commands 0 → 0.05 → 0.10 → 0.15 → 0.20 → 0;
- `v9c_lx16a_walk.mp4`: MuJoCo, commands 0 → 0.15 → 0.20 → 0;
- `v9c_lx16a_low_speed.mp4`: MuJoCo, 0.05 / 0.10 / 0.25;
- `v9a_lx16a_before.mp4`: the previous policy falling on the fitted servo;
- `v9c_isaacsim_cmd0.15.mp4`: Isaac Sim.
- `v9d_lx16a_speeds.mp4`: v9d in MuJoCo, commands 0 → 0.05 → 0.10 → 0.15 → 0.20 → 0.

---

## 5. Configuration files

| file | holds | rule |
|---|---|---|
| `assets/robots/trex/robot_config.yaml` | physical robot: masses (`base_link_inertia: composite`), joint limits, servo datasheet, `actuator_model`, ankle linkage (`sim_model: teleport_routed`), sim drive gains, materials | every physical number lives here, never in code |
| `assets/robots/trex/hardware.yaml` | Pi side only: bus port and timing (`request_gap_s` 2 ms, `read_timeout_s` 6 ms), servo id/dir/zero, IMU bus/address/mode/mounting | `HardwareBackend` refuses to drive until every id/dir/zero is set and every joint range fits 0–1000 ticks |
| `walk_rl/walk_task.yaml` | task defaults: rates, command, gait clock, actions, actuator model, observation, termination, rewards, harness, pushes, randomization, PPO | each run writes its merged copy into its run directory |
| `walk_rl/experiments/<run>.yaml` | overrides on top of the defaults, one file per run | `train.py --task` |

After changing `robot_config.yaml`, re-bake the Isaac Lab robot with `python walk_lab/bake_usd.py`. The Lab env refuses a stale bake.

Robot choice and self-collision (task config `robot:`):
- `asset_dir`: the robot asset (`Trex.urdf`, `meshes/`, `robot_config.yaml`), default `assets/robots/trex`. Training, evaluation, the Pi runner and MuJoCo all load the run's asset. Each asset has its own Lab bake: `python walk_lab/bake_usd.py --asset-dir assets/robots/trex_v2`.
- `self_collision`: when true, PhysX collides exactly the link pairs in the asset's `sim_self_collision.pairs`, in both Isaac stacks. A pair touching above `termination.self_contact_force_n` ends the episode. Foot and ground-link forces then count the floor only. Check: `python redesign/test_self_collision.py --backend sim|lab`.
- Self-collision needs about 3x more GPU aggregate pairs (~14 per env against ~4.4). With the old 8-per-env buffer, PhysX printed `increase PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity` once and then missed contacts: at 256 envs the robots fell through contacts and every episode ended in the first steps. Both stacks now use 16 per env. After a run, look for `PhysX error` in its log; that message shows only one time.

Per-joint action range (task config `actions:`): `joint_scale_deg` sets the half-range and `joint_center_deg` its centre. Target = centre + a × scale; observation = (q − centre) / scale. Both are part of the policy interface, and the Pi runner refuses a policy whose scale or centre differs.

---

## 6. How to

All PC commands run from the repo root with `export OMNI_KIT_ACCEPT_EULA=YES`. `PY=~/miniconda3/envs/isaac6/bin/python` is the Isaac Sim env and `LAB=~/miniconda3/envs/isaaclab6/bin/python` is the Lab/MuJoCo env.

### Train and evaluate
```bash
$PY scripts/test_load_robot.py                                # load the robot with robot_config, screenshot to outputs/
$PY walk_rl/smoke_env.py --num-envs 256                       # env sanity check
$PY walk_rl/train.py --name v9c --iterations 1500 --task walk_rl/experiments/v9c_actuator.yaml
$PY walk_rl/train.py --name X --resume                        # continue a run
$LAB walk_rl/train.py --backend lab --name X ...              # same trainer on Isaac Lab
$LAB walk_lab/train_rsl.py --name X --iterations 3000 --task ...   # rsl_rl on Isaac Lab
$PY walk_rl/evaluate.py --name v9c --actuator lx16a           # nominal + hard sensors + pushes, ~10 min
walk_rl/start_tensorboard.sh start                            # http://localhost:6006
```
- `evaluate.py` runs the same suite for every run:
  - nominal: action delay 0–1;
  - hard: action delay 0–2, IMU delay 0–1, encoder offsets;
  - push: kicks of 0.1–0.4 m/s at 5 s.
- Options: `--actuator` overrides the servo model, `--no-dr` gives the nominal robot, `--backend lab` evaluates on Lab, and `--ground-stiffness` gives a soft floor.
- Compare runs only on `evaluate.py`, never on training reward or the training termination rate.

### Watch a policy
```bash
$LAB pi_runner/render_mujoco.py --run v9c --schedule 0:2 0.15:8 0.20:8 0:4    # MuJoCo MP4, deployed loop
$PY walk_rl/render_gait.py --name v9c --ckpt checkpoint.pt --cmd 0.15 --seconds 8   # Isaac frames + GIF
$PY walk_rl/play.py --name v9c --ckpt checkpoint.pt          # live Isaac viewer, keyboard command and pushes
```
- `render_mujoco.py` runs the exact Pi loop (`PolicyRunner` + NumPy policy) on the fitted servo model.
- `render_gait.py` and `play.py` use CPU physics, which is required for rendering, and need a free GPU for the RTX renderer.

### Check sim-to-sim transfer
```bash
$LAB pi_runner/sim2sim.py --run v9c --frictions 0.3 0.5 0.8 --actuator lx16a   # 27 × 60 s episodes in MuJoCo
$PY pi_runner/test_obs_parity.py --run v9c                    # runner == simulator observations and goals
```
The pass criterion is that every walking condition on the stiffest floor survives 60 s.

### Check that a code change did not alter the task
```bash
$PY walk_rl/trace_env.py --out /tmp/a.npz                     # before the change
$PY walk_rl/trace_env.py --out /tmp/b.npz --compare /tmp/a.npz  # after: must print IDENTICAL
$LAB walk_lab/smoke_lab.py --out /tmp/lab.npz --compare /tmp/b.npz   # Lab vs Isaac Sim: IDENTICAL at 16 CPU envs
```

### Refit the servo model
```bash
$LAB walk_rl/fit_actuator.py            # 48 logged segments → outputs/actuator_fit.json (RMS 0.457°)
```
Copy the values into `robot_config.yaml actuator_model` by hand.

---

## 7. Deployment on the Pi

### Setup and deploy (from the PC)
```bash
pi_runner/deploy.sh pi@trex.local v9c   # rsync code, configs and policies to ~/trex, update the venv
ssh pi@trex.local 'bash ~/trex/pi_runner/pi_setup.sh'   # one-time system setup (idempotent)
```

### Tools (on the Pi, `cd ~/trex && .venv/bin/python -m pi_runner.<tool>`)
| command | moves servos | does |
|---|---|---|
| `hw_check info` | no | board, clocks, throttling, buses, adapters |
| `hw_check imu --seconds N` | no | live gravity, gyro, yaw; rate, latency, register integrity |
| `hw_check bus [--scan]` | no | servo ids, positions, voltage, temperature, read timing |
| `hw_check watch --seconds N` | no | records all joint angles while joints are moved by hand |
| `hw_check bench RUN_DIR [--dump f.npy]`, `compare a b` | no | policy step time; PC↔Pi goal parity |
| `hw_check dryrun RUN_DIR --cmd V [--bus]` | no | the full 25 Hz loop with the real IMU; `--bus` adds real servo reads and sends moves to an absent id |
| `servo_limits [--apply \| --restore FILE]` | no | shows, writes (with backup and read-back) or restores the EEPROM angle limits |
| `servo_identify --check \| --go` | yes | one servo at a time: ±10° ramp and step, logged |
| `servo_hold --check \| --go \| --sweep` | yes | zero pose: ramp in 2 s and hold 10 s; `--sweep` moves each joint to +10° and back |

Every tool that moves servos switches torque off at the end, on any error, on Ctrl-C and on ssh disconnect. All of them refuse to move if any servo reads below the minimum voltage (battery off).

### Measured on the robot
- **Policy step:** 0.76 ms on the Pi. Pi and PC goals are bit-identical.
- **Full 25 Hz loop:** IMU, 8 pipelined servo reads, policy and 8 moves take 20.4 ms mean (23.4 ms max) of the 40 ms tick, with 0 overruns and 0 missing replies over 12,000 reads on battery.
- **BNO055:** 3.4 ms burst read, fresh 100 Hz fusion data every tick.
- **Servos (unloaded, 10° step):** 90% reached in ~110 ms at ~97 °/s mid-move, final error within ±1.2°.
- **Calibration:** standing straight, the robot reads 0 ± 0.2° on every joint.
- **Battery:** 7.06–7.16 V at rest.

### Staged bring-up on the robot
Each stage needs the operator's go-ahead.
1. ✓ Read-only checks: IMU, bus, calibration.
2. ✓ Hold the zero pose under torque (robot in the air).
3. ✓ Each joint to +10° and back (robot in the air); all directions confirmed.
4. Policy with the robot hanging, feet clear.
5. Assisted standing.
6. First steps.

A hardware runner that drives the policy on the servos (stage 4) is not written yet. `PolicyRunner` + `HardwareBackend(servos=True)` are the parts it will combine.

---

## 8. Open items

- **Isaac–MuJoCo speed gap:** v9i makes 86% of 0.10 in Isaac but 53–63% in MuJoCo, and falls at 0.20 on friction 0.5/0.8 in MuJoCo.
- **Hip inward range:** the redesign audit found the femur servo meets the hip servo at 8–9° inward in the CAD, while config and EEPROM allow +30°. Measure by hand, then tighten `robot_config` and the servo limits. v9d uses at most +5.5°.
- **Measurements still needed:**
  - real joint ranges (the tibia limit caps speed);
  - torso / battery mass and size;
  - a loaded servo bench test to narrow the friction / gain / stall ranges;
  - battery voltage under load;
  - floor friction.
- **Mechanical check:** the head and right hip stopped 2.3–2.8° short of +10° when holding, possibly from drag.
- **Battery current (hypothesis, decided to keep the current pack for now):**
  - The TalentCell YB1206000 supplies 3 A at most, about 4.1 A at the servos through the DC-DC converter.
  - The estimated walking draw is 4.2–6.2 A mean with 7–11 A peaks (uncertain: the LX-16A stall current is quoted at 1.1–3 A).
  - On the first powered steps, watch the servo-reported voltage and log the current if possible.
  - If it sags or cuts out, the parked fix is a 2S 7.4 V LiPo straight to the BusLinker (e.g. [OVONIC 2S 2200 mAh 50C XT60](https://www.amazon.com/OVONIC-7-4V-Lipo-Battery-Packs/dp/B0CF9JRPZL), 118 g), with the existing DC-DC converter set to 5.1–5.2 V for the Pi only.
- **Roll:** 13–17° peak-to-peak while walking, against a 12° target.
- **Isaac Lab:** the GPU evaluation gap is unexplained, and the rsl_rl run is unfinished.

---

## 9. Practical notes

- **Isaac Sim:**
  - Render with CPU physics. GPU physics keeps transforms out of USD, so the renderer shows a frozen robot.
  - Use `timeline.pause()`, not `stop()`, to freeze a pose; `stop()` resets the articulation.
  - Disable Fabric for headless training (5× faster) and size the PhysX GPU pair buffers to the env count, or contacts are silently missed.
  - Kit's fast shutdown hides Python tracebacks, so the scripts print them explicitly.
- **MuJoCo:**
  - The ankle linkage is a tendon equality and needs 1 kHz physics.
  - Use a 4 ms (near-rigid) contact time constant.
  - Never teleport the ankle or set its velocity.
  - Rendering on this PC works with `MUJOCO_GL=glfw` (EGL fails).
- **Pi runner:**
  - It must compute in float32. Saturated actions land on half ticks, and float64 rounds them differently from training.
  - The command low-pass, history, heading latch and gait clock must stay identical to `task_core`. `test_obs_parity.py` checks this.
- **BNO055:** it clock-stretches. Keep it on bit-banged I²C, because the Pi's hardware I²C corrupts bit 7.
- **LX-16A bus:**
  - Read only the bytes a reply still lacks; asking for more waits out the timeout.
  - Pipeline position requests 2.0 ms apart (1.6 ms loses replies).
  - Let queued writes drain before reading.
