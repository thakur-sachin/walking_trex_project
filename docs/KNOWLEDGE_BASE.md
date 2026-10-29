# Trex knowledge base

The current state of the Trex walking robot project: what exists, how it fits together and how to use it. Last updated 2026-10-21 (branch `pi-bringup`).

The full development record (experiments, failures, decisions and the reasons for them) is in [PROJECT_HISTORY.md](PROJECT_HISTORY.md). Every training run, with its change, targets, results and verdict, is in [ABLATION_STUDY.md](ABLATION_STUDY.md).

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
| Mass | 1.307 kg weighed (2026-10-20); torso with its 4 servos and all electronics 0.805 kg, each leg 193 g, head 78 g, tail 38 g |
| Floor | indoor wood, PLA soles |

### Joints

| joint | servo id | range (model) | positive angle moves |
|---|---|---|---|
| `head_joint` | 7 | −45…+45° | nose down |
| `tail_joint` | 8 | −45…+45° | tail tip up |
| `left_hip_joint` | 1 | −30…+7° | left foot inward |
| `right_hip_joint` | 4 | −30…+7° | right foot inward |
| `left_femur_joint` | 2 | −45…+45° | left foot forward |
| `right_femur_joint` | 5 | −45…+45° | right foot forward |
| `left_tibia_joint` | 3 | −45…+90° | left foot back |
| `right_tibia_joint` | 6 | −45…+90° | right foot back |

- A positive hip angle moves **either** foot inward. The hips are mirrored in raw sign, so never assume that left and right share a sign. The measured table is in [JOINT_CONVENTIONS.md](JOINT_CONVENTIONS.md), produced by `scripts/joint_sign_convention_experiment.py`.
- The ranges come from the as-built CAD (the hip meets the torso at +10° inward; the right hip stopped at ~7.5° on the robot). Femur − tibia (the ankle angle) is limited to −55…+37° where the foot meets the shank and idler. The tibia and femur ranges are not yet confirmed by hand.
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
common/provenance.py          code commit + robot model hash stored with every run and result
walk_rl/                      the walking task, Isaac Sim stack (fallback), shared task logic
  task_core.py                all task logic shared by both simulators
  env.py                      Isaac Sim backend (conda env isaac6)
  actuator.py                 LX-16A servo model (torch and numpy)
  walk_task.yaml              task / reward / randomization / PPO defaults
  experiments/*.yaml          per-run overrides of the current runs (v9n, v9o); older ones: git show <commit>
  experiments/registry.yaml   every run: parent, change, why, targets, verdict, lesson
  ablation_study.py           registry + measured results -> docs/ABLATION_STUDY.md
  gait_scorecard.py           the 10 fixed gait targets every run is judged on
  run_experiment.sh           train + the full fixed evaluation of one run
  train.py, ppo.py            in-repo asymmetric PPO
  evaluate.py                 fixed evaluation suite
  play.py, render_gait.py     live viewer, offline gait capture
  fit_actuator.py, fit_actuator_walk.py   fit the servo model to servo_logs/ (unloaded moves; real walks)
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
  walk_live.py, servo_load_test.py   policy on the robot, standing load test (move servos)
  deploy.sh, pi_setup.sh      PC → Pi deployment, Pi system setup
  test_obs_parity.py          runner vs simulator parity test
scripts/                      robot import pipeline, prim audit, joint-sign experiment
docs/                         this file, PROJECT_HISTORY.md, ABLATION_STUDY.md, JOINT_CONVENTIONS.md
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
  - the merged `walk_task.yaml`, a copy of `robot_config.yaml` and `provenance.json` (commit, robot model hash);
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
| Action (8) | position targets for the 8 servos, `target = centre + a · scale`, a ∈ [−1, 1]; scale = the symmetric joint limit unless `actions.joint_scale_deg` sets it (stored in the policy). Ankles follow the linkage |
| Actor observation (105) | the 3 latest frames of 35 values: IMU gravity (3), gyro (3), speed command (1), 8 servo positions / limit (quantised to 0.24°), 8 finite-difference joint velocities, previous action (8), gait clock sin/cos (2, zero when standing), heading error sin/cos (2) |
| Critic observation (50) | one clean frame plus privileged state: base velocity, foot contacts, foot heights, servo torques |
| Command | forward speed (current runs 0.03–0.25 m/s, 35% drawn from 0.03–0.12); 0 (stand) 20% of the time; resampled every 5 s |
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
  - Nominal values are in `robot_config.yaml actuator_model`, refit 2026-10-20 to the real robot's walking logs (`walk_rl/fit_actuator_walk.py`): kp 4.56/rad, deadband 0.41°, latency 4 ms, armature 0.021 kg·m², friction 0.116 N·m, 0.775 × datasheet speed, 1.84 × datasheet stall torque (an effective holding torque). Open-loop replay of the real walks: 1.4° RMS on the floor, 0.5° hanging. Per-episode randomization is in `domain_randomization.actuator_model` (v9n ranges are centred on these values; older configs keep the 10-18 ranges).
  - With `lx16a`, the torque, torque-saturation and saturation measures are relative to the fitted holding torque.
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
  - IMU mount offset (`observation.imu_mount_offset_deg`, v9n ±2° roll and pitch);
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

### Current policies
Judge every run with `walk_rl/gait_scorecard.py <runs>`, the MuJoCo video, and the real robot. All runs and their results: [ABLATION_STUDY.md](ABLATION_STUDY.md).

| | v9m | v9n | v9o |
|---|---|---|---|
| change | sole-centre foot slip −10 | + servo model refit to the real walks, IMU mount offset ±2° | + mirror loss 0.5 |
| Isaac speed at 0.10 m/s | 88% | 94% | 91% |
| MuJoCo speed at 0.10 (refit servo model) | 52% | 72% | 70% |
| MuJoCo survival | 29/36 | 33/36 | 35/36 |
| scorecard | 6/10 | 6/10 | 6/10 |
| **real robot, floor, 0.10 m/s for 10 s** | **0.41 m (41%)**, yaw +7° | **0.25 m (25%)**, turned +49° | **0.46 m (46%)**, yaw −8°, straight |

- **v9m** (2026-10-20): 41% on the robot, straight.
- **v9n** (2026-10-21) learned a one-sided posture: head −14°, tail +18°, left hip near its +7° end 61% of the time, right hip −9°. On the floor it turned +49° in 10 s; in MuJoCo the same posture turned only 5°.
- **v9o** centres head, tail and hips (MuJoCo heading change −0.2°). In MuJoCo it moves sideways 0.8–2.9 m per ~4 m forward, always to its left. On the floor (2026-10-21) it walked 0.46 m in 10 s (46%, the best real result so far), heading held (−8° net), with a slight drift to the right and no crab. Mean head −1°, tail −2°, hips −7°/−3°.
- Configs: `walk_rl/experiments/v9n_servo_refit.yaml`, `v9o_mirror_loss.yaml`; v9m: `git show 3a021e3:walk_rl/experiments/v9m_slip10.yaml` (it trained on robot model M4; today's robot_config is M5).
- These policies need the `lx16a` servo model.

Videos: `outputs/videos/<run>_lx16a_speeds.mp4` (MuJoCo, commands 0 → 0.05 → 0.10 → 0.15 → 0.20 → 0); real robot: `media/Robot_walk_v1.MOV` (v9m), `media/Robot_walk_v2.MOV` (v9n), `media/Robot_walk_v3.MOV` (v9o, side view at floor level).

---

## 5. Configuration files

| file | holds | rule |
|---|---|---|
| `assets/robots/trex/robot_config.yaml` | physical robot: masses (`base_link_inertia: composite`), joint limits, servo datasheet, `actuator_model`, ankle linkage (`sim_model: teleport_routed`), sim drive gains, materials | every physical number lives here, never in code |
| `assets/robots/trex/hardware.yaml` | Pi side only: bus port and timing (`request_gap_s` 2 ms, `read_timeout_s` 6 ms), servo id/dir/zero, IMU bus/address/mode/mounting | `HardwareBackend` refuses to drive until every id/dir/zero is set and every joint range fits 0–1000 ticks |
| `walk_rl/walk_task.yaml` | task defaults: rates, command, gait clock, actions, actuator model, observation, termination, rewards, harness, pushes, randomization, PPO | the defaults are the v8 task with every later option off, so `trace_env.py` stays bit-identical; they do not walk on today's robot. Start new runs from the newest experiment file |
| `walk_rl/experiments/<run>.yaml` | overrides on top of the defaults, one file per run, self-contained (not chained) | `train.py --task`; only the current runs are kept in the tree |
| `walk_rl/experiments/registry.yaml` | one entry per run: parent, change, why, targets, result, verdict, lesson; the robot model versions M0–M5 | `walk_rl/ablation_study.py` turns it into `docs/ABLATION_STUDY.md` |

After changing `robot_config.yaml`, re-bake the Isaac Lab robot with `python walk_lab/bake_usd.py`. The Lab env refuses a stale bake.

---

## 6. How to

All PC commands run from the repo root with `export OMNI_KIT_ACCEPT_EULA=YES`. `PY=~/miniconda3/envs/isaac6/bin/python` is the Isaac Sim env and `LAB=~/miniconda3/envs/isaaclab6/bin/python` is the Lab/MuJoCo env.

### Train and evaluate
```bash
$PY scripts/test_load_robot.py                                # load the robot with robot_config, screenshot to outputs/
$PY walk_rl/smoke_env.py --num-envs 256                       # env sanity check
$PY walk_rl/train.py --name v9o --iterations 1500 --task walk_rl/experiments/v9o_mirror_loss.yaml
$PY walk_rl/train.py --name X --resume                        # continue a run
$LAB walk_rl/train.py --backend lab --name X ...              # same trainer on Isaac Lab
$LAB walk_lab/train_rsl.py --name X --iterations 3000 --task ...   # rsl_rl on Isaac Lab
$PY walk_rl/evaluate.py --name v9o --actuator lx16a           # nominal + hard sensors + pushes, ~10 min
walk_rl/start_tensorboard.sh start                            # http://localhost:6006
```
- `evaluate.py` runs the same suite for every run:
  - nominal: action delay 0–1;
  - hard: action delay 0–2, IMU delay 0–1, encoder offsets;
  - push: kicks of 0.1–0.4 m/s at 5 s.
- Options: `--actuator` overrides the servo model, `--no-dr` gives the nominal robot, `--backend lab` evaluates on Lab, and `--ground-stiffness` gives a soft floor.
- Compare runs only on `evaluate.py`, never on training reward or the training termination rate.

### Run an experiment (the ablation loop)
1. Find the cause in data first (replay, scorecard, real logs). Write the new entry in `walk_rl/experiments/registry.yaml`: parent, the one change, why, and measurable targets.
2. Copy the parent's experiment file, make the one change, and write the evidence in its header.
3. `walk_rl/run_experiment.sh NAME TASK_YAML PARENT...`: trains, then runs the parity test, `evaluate.py`, `sim2sim.py`, `render_mujoco.py`, `gait_scorecard.py` and `ablation_study.py` (about 2 h).
4. Fill result, verdict and lesson in the registry, then `python3 walk_rl/ablation_study.py`.
5. Only a candidate that meets its targets goes to the robot, with the operator's go-ahead.

### Watch a policy
```bash
$LAB pi_runner/render_mujoco.py --run v9o --schedule 0:2 0.05:6 0.10:6 0.15:6 0.20:6 0:3 --tag speeds   # MuJoCo MP4, deployed loop
$PY walk_rl/render_gait.py --name v9o --ckpt checkpoint.pt --cmd 0.10 --seconds 8   # Isaac frames + GIF
$PY walk_rl/play.py --name v9o --ckpt checkpoint.pt          # live Isaac viewer, keyboard command and pushes
```
- `render_mujoco.py` runs the exact Pi loop (`PolicyRunner` + NumPy policy) on the fitted servo model.
- `render_gait.py` and `play.py` use CPU physics, which is required for rendering, and need a free GPU for the RTX renderer.

### Check sim-to-sim transfer
```bash
$LAB pi_runner/sim2sim.py --run v9o --frictions 0.3 0.5 0.8 --commands 0.0 0.05 0.10 0.20 --actuator lx16a   # 36 × 60 s episodes
$PY pi_runner/test_obs_parity.py --run v9o                    # runner == simulator observations and goals
$LAB pi_runner/replay_real_walk.py assets/robots/trex/servo_logs/walk_v9o_20261021_212933_floor.npz --run v9o --real-distance 0.46 --frictions 0.3 0.5 0.8   # real walk vs MuJoCo open/closed loop
$LAB pi_runner/contact_sweep.py assets/robots/trex/servo_logs/walk_v9o_20261021_212933_floor.npz --run v9o --real-distance 0.46   # which friction / contact / play reproduces the real gait
$LAB pi_runner/sim2sim.py --run v9q --frictions 0.15 0.2 0.27 --commands 0.0 0.05 0.10 0.20 --actuator lx16a --out outputs/walk_runs/v9q/sim2sim_lowmu_lx16a.json   # the measured low-friction floor (own file)
$LAB pi_runner/sim2sim_home_model.py [--runs v9m v9o]            # every policy on the robot model it trained on (rebuilt from its robot_config)
python3 walk_rl/gait_scorecard.py v9m v9n v9o                # the 10 fixed targets, one column per run
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
$LAB walk_rl/fit_actuator.py            # 48 unloaded segments (10-18) → outputs/actuator_fit.json
$LAB walk_rl/fit_actuator_walk.py       # + the real floor and hanging walks: scales on the current values
```
Copy the values into `robot_config.yaml actuator_model` by hand.

---

## 7. Deployment on the Pi

### Setup and deploy (from the PC)
```bash
pi_runner/deploy.sh pi@trex.local v9m v9n v9o   # rsync code, configs and policies to ~/trex, update the venv
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
| `walk_live RUN_DIR --schedule 0:3 0.10:10 0:3 [--floor]` | yes | runs the policy at 25 Hz and logs every tick (`~/trex/logs/walk_*.npz`); hanging mode stops on a 20° swing, floor mode on the training tilt limit; up to 2 consecutive missed servo replies use the last good reading |
| `servo_load_test [--sim]` | yes | robot standing: squats and a squat sine, logged; `--sim` runs the same goals in MuJoCo on the PC |

Every tool that moves servos switches torque off at the end, on any error, on Ctrl-C and on ssh disconnect. All of them refuse to move if any servo reads below the minimum voltage (battery off).

### Measured on the robot
- **Policy step:** 0.76 ms on the Pi. Pi and PC goals are bit-identical.
- **Full 25 Hz loop:** IMU, 8 pipelined servo reads, policy and 8 moves take 20.4 ms mean (23.4 ms max) of the 40 ms tick, with 0 overruns and 0 missing replies over 12,000 reads on battery.
- **BNO055:** 3.4 ms burst read, fresh 100 Hz fusion data every tick.
- **Servos (unloaded, 10° step):** 90% reached in ~110 ms at ~97 °/s mid-move, final error within ±1.2°.
- **Calibration:** standing straight, the robot reads 0 ± 0.2° on every joint.
- **Battery:** 7.06–7.16 V at rest; 7.08–7.16 V after a 16 s walk.
- **Servo tracking while walking (v9m):** goals reached about 2 ticks (80 ms) later; error with that delay removed 1.4–3.7° on the floor, 1.0–2.9° hanging; mean goal − q within 0.5°.
- **Attitude standing at the zero pose:** roll −0.6°, pitch +1.2° (sim 0°); pitch changes ~1° with a crouch.
- **Servo temperature:** 25–28 °C → 28–32 °C over a 16 s walk.
- **Missed servo replies:** one double miss (left tibia) in the first floor run; since then 2 single misses in ~2300 ticks (tolerated).
- **v9n on the floor:** head and tail goal − q 12°/12° mean (v9m 4°/2°) with ±40° swings; legs 4–6°.
- **Friction and backward stability (tilt test on a wood plank, 2026-10-21, zero pose under torque, `servo_hold.py --go --hold-s 120`):** head downhill the feet slid at 15° (static μ 0.27, below the 0.30 training minimum). Tail downhill the robot toppled at 7° before it slid; the MuJoCo model tips at 12.8° (centre of mass 41 mm in front of the heel edge; the real robot about 22 mm). The joints held within 1.2°. Heel contact under load (paper test): the heel edges carry load on both feet. Balance point (robot on a pencil across the soles): the centre of mass is 30 mm ahead of the heel edge (model was 40.8 mm). Robot model M6 moves the estimated battery and electronics positions 25 mm back to match (30.3 mm; backward tip angle 9.6°, tilt test 7°). Nothing on the robot moved since the 2026-10-20 weighing.
- **v9o on the floor:** goal(t−1) − q 4–7° mean on every joint (left hip highest, 7.5°); head/tail 6°/5°. Gait cycle 1.8 Hz (18 cycles in 10 s), so 25 mm per cycle. Servo temperature 24–26 → 28–31 °C; battery 7.08–7.16 V before and after.
- **Foot slip (v9o side video, first try):** the sole tracked by colour gives the total distance (0.46–0.48 m, consistent with the pens, scale from the 140 mm sole length) but not the slip per step. From the side the two soles overlap, and the colour mask fails in the darker half of the frame. Stance-foot creep reads 0–15 mm against 12.5 mm per step, which is inside the noise. A marker on one foot (bright tape) and more light are necessary to measure slip.

### Staged bring-up on the robot
Each stage needs the operator's go-ahead.
1. ✓ Read-only checks: IMU, bus, calibration.
2. ✓ Hold the zero pose under torque (robot in the air).
3. ✓ Each joint to +10° and back (robot in the air); all directions confirmed.
4. ✓ Policy with the robot hanging, feet clear (v9m, 2026-10-20).
5. Assisted standing (not done separately).
6. ✓ First steps: v9m walked 0.41 m in 10 s on the floor (2026-10-20); v9n 0.25 m with a +49° turn (2026-10-21); v9o 0.46 m, straight (2026-10-21).

Floor test method: 3 s at 0, 10 s at 0.10 m/s, 3 s at 0; pens every 5 in along the path; the camera fixed at floor level, at the side, so it sees the robot from the side for the whole walk.

---

## 8. Open items

- **Sim-to-real speed gap.** At 0.10 m/s: v9m Isaac 88%, MuJoCo 52%, real 41%; v9n MuJoCo 72%, real 25% (with a 49° turn); v9o Isaac 91%, MuJoCo 70%, real 46%. Found and addressed: foot skating in PhysX (v9m), the servo model (refit, M5), attitude sensitivity (IMU offset, v9n), a one-sided posture (mirror loss, v9o). Not measured: real foot slip (the first side video cannot resolve it; needs a marker on one foot).
- **v9o uses its joints differently on the floor than in MuJoCo** (same policy, mean while walking): left hip −6.5° with ±5.9° swing (MuJoCo −2°, ±2.9°), right femur −5° (MuJoCo +1°), right tibia swing ±7° (MuJoCo ±10°), roll +1.9° (MuJoCo 0.1°; standing −0.8°). Ruled out in MuJoCo as the single cause: a torso CoM offset up to ±15 mm sideways, an IMU roll offset of ±3°, an IMU lag of up to 3 ticks (v9o keeps 55–78%). Contact sweep (`pi_runner/contact_sweep.py`): friction μ 0.15 (47%) or 3° of joint play (50–57%) brings the speed to the real 46%, but no symmetric contact or play setting, and no one-sided encoder offset (±4°) or one-sided play (4°), gives the real one-sided posture. Training friction is 0.30–0.85; the real static μ is 0.27. The first model change that reproduces the real left-hip pattern (−5°, ±5° swing; real −6.5°, ±5.9°): torso CoM 20–31 mm further back with μ 0.18–0.22 (score 1.6–1.9, speed 54–68%); at μ 0.27 the gait snaps back to the nominal pattern, and at 31–35 mm most runs fall onto the tail.
- **Real IMU attitude lags MuJoCo by 40–80 ms** on all three floor walks (roll of the real goals replayed in MuJoCo; joint angles align at zero lag). Training covers 0–1 tick (task_core only supports 0/1). v9m and v9n slow down with 2–3 ticks in MuJoCo; v9o does not. Unknown: sensor latency or body compliance.
- **v9o sideways drift in MuJoCo** (always to its left, heading held). It did not transfer: the real robot drifted slightly to the right with no crab. The MuJoCo-only cause is not known.
- **Measurements still needed:** tibia and femur ranges by hand; battery voltage under load.
- **Training friction vs the measured floor:** PhysX averages robot and ground friction; with the default ground 0.5 the effective minimum is 0.25 (real static μ 0.27, sliding μ lower). `sim.ground_friction` (default null = unchanged) sets the ground; v9q uses 0.1 with robot 0.14–1.6 (0.12–0.85 effective) and walks at 80–91% of 0.10 in MuJoCo at μ 0.15–0.27.
- **Surface matters more than any sim setting so far (2026-10-22):** v9s made 0.36 m on wood and > 0.70 m on carpet with the same joint motion. Its 6 mm foot lift slides on wood. MuJoCo at μ 0.15–0.27 does not show this loss.
- **Hip goal clip on M6 (2026-10-22):** the goal is clipped to the physical range (hips −30…+7°), but the default joint-limit penalty only acts beyond 0.9 of the symmetric ±30° limit, so the +7° inward end costs nothing. On M6, v9q keeps its right hip there 49–51% of the walk and v9r parks both hips there while standing. `reward.joint_limit_range_margin_deg: N` penalizes the last N° of the physical range instead (0 at N°, 1 per joint at the end); v9s uses 3°.
- **Mechanical check:** the head and right hip stopped 2.3–2.8° short of +10° when holding, possibly from drag.
- **Battery current (hypothesis, current pack kept):** the TalentCell YB1206000 supplies 3 A at most (~4.1 A at the servos); walking may draw 4.2–6.2 A mean. No sag seen so far (7.08–7.16 V after a 16 s walk). Parked fix: a 2S 7.4 V LiPo straight to the servo board.
- **Isaac Lab:** the GPU evaluation gap (v8e era) is unexplained; the rsl_rl run is unfinished.
- **Task defaults:** `walk_task.yaml` defaults are the v8 task (kept for the regression trace). Making the current recipe the default needs a new reference trace.

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
