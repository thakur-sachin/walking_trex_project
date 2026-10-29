# v9 improvement plan

Source: Sachin's research brief "Trex biped walking improvements" (2026-10-15), checked against the code and the v8d evaluation. The work runs in three tiers, and each tier starts only when the one before it has been used up:

- **A.** Free work in simulation and software. I can do this now.
- **B.** Free measurements on the robot and parts Sachin already has.
- **C.** Cheap upgrades: sensors and mechanical changes.

## 1. What the brief gets right and wrong for Trex

Each claim in the brief was checked against our data. v8d `eval_checkpoint.json` is the v8e baseline until v8e is evaluated.

| Brief's claim | Our data | Verdict |
|---|---|---|
| Sway ~0.2 m/s is mostly physics (linear-inverted-pendulum floor 0.2–0.48 m/s) | v8d vy_rms 0.14–0.21 m/s while walking, 0.015 standing | **Agree.** Stop penalising instantaneous vy (`lateral_vel` −2·vy² today) |
| Penalise the ~25° roll instead | roll is 15–24° **peak-to-peak** (±8–12°), largest at slow speed | **Agree with the correction** that the figure is peak-to-peak. A ±0.1 rad (±5.7°) dead-band will bite |
| Heading drift needs symmetry | yaw change is already 0–3° per 20 s, **but lateral drift dy ≈ +1.0 m per 20 s at every speed, always to the same side** | **Agree, for a different reason.** The real defect is a constant sideways crab of ~5 cm/s, a left/right asymmetry the brief did not see. Symmetry and drift-based lateral terms target it directly |
| 84% speed tracking comes from a loose velocity kernel | our kernel is σ = 0.06 m/s, plus a linear term `track_vx_lin`. At v8d's 0.054 m/s error (73% of 0.20) the kernel already loses 56% of its reward | **Disagree.** There is plenty of gradient. The known cap is the tibia pinned on the 45° placeholder limit (v8 finding). Test it as a sensitivity study (A1); don't retune the kernel |
| Check for phantom root mass (Open Duck Jetson bug) | checked 2026-10-15 on the Isaac Sim stack: link masses sum to 1.778 kg = PhysX total, so no phantom mass. base_link = the 1.0 kg placeholder; the URDF says 2.59 kg | **No bug.** Note: body weight is 17.4 N, not the brief's ~13 N, so the knee moment-arm budget is ~25% tighter than it says |
| Servo model lacks armature, friction, backlash and back-EMF | we have delay, a 0.5 low-pass, 0.24° quantisation and a no-load-speed target ramp on an implicit PD drive (kp 40, kd 5). There is **no** armature, joint friction, backlash, torque-speed line or voltage coupling, and the torque cap is randomised independently of speed | **Agree. This is the biggest free item** |
| Action filter (−3 dB at ~2.9 Hz) eats swing motion | `smoothing_alpha` 0.5 at 25 Hz, as stated | **Agree.** Worth an ablation; whatever is chosen must be mirrored in the Pi loop |
| 3-frame history is too short | `history_len` 3 (120 ms) | **Agree.** Try 10–15 frames |
| Foot-distance band | `feet_distance` exists (band 0.7–1.5× the spawn spacing) | Already have it; the band may need narrowing |
| Mirror map: swap hips without negating | JOINT_CONVENTIONS: positive hip angle drives either leg inward, and the femur/tibia axes are not mirrored | **Agree.** For all 8 joints the map is a pure L/R swap; IMU roll, yaw and vy flip sign. Unit-test before use |
| MuJoCo sim2sim before hardware | `mujoco` 3.8.0 is already installed in `isaac6`. MuJoCo can express the ankle linkage exactly as a joint equality | **Agree.** Free |

## 2. Tier A: free, simulation and software (me)

Each item can be switched on or off in `walk_task.yaml` so it can be ablated. All new task logic goes into `walk_rl/task_core.py`, so both stacks get it. The CPU trace parity check (`trace_env.py` vs `smoke_lab.py`) must still read IDENTICAL after every change.

**A0. Evaluation additions**, done first, since they decide everything else.
- Tibia time-at-limit fraction and per-joint torque percentiles: is speed limited by the joint range or by torque?
- Foot roll relative to the world during stance: is the sole on its edge?
- Left/right stance-time and step-length asymmetry, next to dy.
- An "actuator-model physics" eval mode (after A4), so every policy, v8e included, also gets a transfer-gap number.

**A1. Sensitivity studies.** These are information for Sachin, not policies.
- Short runs: 4096 envs, 1000 iterations, ~1 h each.
- Tibia limit 45° → 60° → 75°.
- base_link mass 0.8 / 1.0 / 1.3 kg.
- Answers "how much speed would the real joint range buy?" and "how much does every 100 g matter?" before he measures.
- Flagged hypothetical; robot_config is not changed.

**A2. Lateral rewards.**
- Replace instantaneous `lateral_vel` with a dead-banded drift term: |1 s EMA of v_y| minus 0.10 m/s, clipped at 0 (Microduck/Open Duck).
- Add a torso-roll dead-band (±0.1 rad, exp(−100·excess), weight 1–2; ToddlerBot).
- Add a foot-flat-in-stance term.
- Keep `ang_vel_xy` low (−0.05).
- Target: dy from ~1 m to < 0.3 m per 20 s, and roll peak-to-peak down, with no loss of survival.

**A3. Symmetry.**
- Mirror data augmentation in walk_rl PPO: mirrored samples added to each batch.
- The mirror map gets a unit test:
  - neutral pose maps to itself;
  - a right lean maps to a left lean;
  - mirror applied twice is the identity;
  - the gait clock shifts by half a period.
- Targets: the constant sideways crab and the standing yaw (3°).
- rsl_rl has the same feature built in, so the Lab rsl_rl path gets it too.

**A4. LX-16A actuator model.** Biggest transfer item.
- An explicit torque actuator in task_core: firmware P-gain with duty saturation, a back-EMF torque-speed line (stall 1.67 N·m, no-load 5.5–6.5 rad/s, both scaled by bus voltage), Coulomb friction, backlash dead zone and deadband.
- Armature (0.005–0.03 kg·m², randomised ×0.5–2).
- A **coupled** battery-voltage DR (6.6–8.4 V) replaces the independent torque-cap DR.
- Needs two new backend hooks, `_sim_set_dof_efforts` and `_dr_set_armature`, in both stacks.
- Every prior is flagged as an estimate until B4 replaces it.
- I expect simulated numbers to get worse; that is the point. The comparison is v8e vs v9 under actuator-model physics.

**A5. Action and observation pipeline.**
- Smoothing α 0.5 → 0.25 → 0, with an action-rate penalty ramped −0.1 → −1.0 after the gait emerges (Microduck).
- History 3 → 10 → 15 frames.
- Actor stays small enough for the Pi (≤ ~100k MACs; a 15-frame input is ~83k).

**A6. Pi runner and MuJoCo sim2sim.**
- A backend-agnostic runner: observation builder, frozen normaliser, ring buffer, low-pass, clock, NumPy policy and a safety layer.
- Backends: MuJoCo (now) and the LX-16A bus (later).
- Three pass tests:
  1. Recorded Isaac observations go through the runner's builder unchanged.
  2. NumPy matches PyTorch to 1e-5.
  3. 60 s of walking in MuJoCo across friction and delay sweeps.
- This also gives a third, independent simulator for the platform comparison.

**A7. 50 Hz variant.** Only if B3 shows the bus can do it. Train a 50 Hz policy as an alternative.

### Experiment order (each one compared on `evaluate.py`, never on training reward)

| Run | Change | Main metric |
|---|---|---|
| S1–S2 | sensitivity: tibia limit, base mass | speed and saturation vs parameter |
| v9a | v8e + A2 | dy, roll p2p, survival |
| v9b | v9a + A3 | dy, stance asymmetry, standing yaw |
| v9c | v9b + A4 | everything, under actuator-model physics |
| v9d | v9c + A5 (best α and history) | tracking at 0.20, saturation, d_action |
| v9e | winners, long run (16384 envs, 3000 iterations) | candidate for the Pi runner |

**GPU schedule.**
- The Isaac Sim vs Isaac Lab comparison Sachin already asked for runs first: v8e finishes ~04:00 UTC, then benchmarks, then v8e_lab and v8e_lab_rsl.
- A0, A3's unit test and A6 are CPU work and run alongside.
- If the GPU benchmark shows the Lab stack is faster, the v9 ablations run on it. Parity makes this safe.

## 3. Tier B: free, with parts Sachin already has

I write the scripts and data sheets; Sachin does the measuring.

| # | Measurement | How (no purchases) | Feeds |
|---|---|---|---|
| B1 | Mass and CoM | weigh base_link assembly, head, tail and each leg part on a kitchen scale; hang the torso from two points and photograph the plumb lines for the CoM | robot_config `mass`, CoM DR range, A1 |
| B2 | Real joint ranges | move each joint to its mechanical stop, read the servo position | robot_config limits: removes the speed cap |
| B3 | Bus timing | on the Pi: which USB-serial adapter, `cat /sys/bus/usb-serial/devices/ttyUSB*/latency_timer`, and a timing script for 8 writes + 8 reads over 1000 cycles; try `latency_timer=1` | delay DR re-centred, 25 vs 50 Hz decision |
| B4 | Servo bench | one spare LX-16A, a printed lever, known masses (coins, water bottle), Pi logging commanded vs read position, phone slow-motion video. Protocol: static hold under load, 2–5° and 30–60° steps at two loads, chirp 0.2–8 Hz, 5–10 min tibia-load hold logging temperature and voltage, and whether the move-time ramp starts from the measured angle or the setpoint | fit of A4's parameters (CMA-ES against the same servo model) |
| B5 | IMU | BNO055 mode (IMUPLUS vs NDOF), I²C vs UART, actual gyro and quaternion rate and latency, gyro bias at rest | IMU noise, delay and drift DR |
| B6 | Floor friction | tilt a board with a foot on it until it slips: μ = tan θ, on the floor Trex will walk on | friction DR range |
| B7 | Battery voltage range | the pack's voltage full vs under load (multimeter) | V_bus DR range for A4 |

## 4. Tier C: cheap upgrades, after A and B show where the gap is

| # | Upgrade | Cost | Why | Condition |
|---|---|---|---|---|
| C1 | Foot contact switch per foot (SS-10 microswitch, or FSR 402 thresholded to a bit with hysteresis) | ~€4–15 | contact is the signal Trex can least infer (position-only servos); enables contact-phase rewards on hardware | after A5: if longer history still leaves weight-transfer timing poor |
| C2 | 74HC126 buffers + second Pi UART (one leg per bus) | < €2 | ~8.4 ms wire time per cycle makes 50 Hz possible | if B3 shows a USB-latency or bus bottleneck |
| C3 | INA219/INA226 voltage monitor | ~€3–5 | torque and speed scale with bus voltage; log it, and optionally use it as an input | after A4 |
| C4 | Mechanics: battery low between the hips, lighter torso (gyroid infill), TPU soles with rounded edges, ankle re-indexed by the gait's forward lean, flanged bearings on the 4-bar pivots | filament + bearings | knee torque margin, roll, slip | prototype each in sim first (URDF or robot_config change + retrain) |
| C5 | Knee spring, sideways tail | small | knee torque, roll | sim study only; the brief ranks them low and I agree |
| C6 | BNO085 | ~€20 | heading without the magnetometer | only if B5 shows the BNO055 is inadequate |

**Not planned:** optical-flow or ToF as policy inputs (bench odometry only), on-robot RL fine-tuning, or an actuator network (no torque labels).

## 5. Success targets for v9 (on the eval suite with DR)

- **Lateral:** dy < 0.3 m per 20 s at every speed; roll p2p ≤ 12° at 0.10 m/s.
- **Robustness:** survival and push survival ≥ v8e.
- **Speed:** tracking at 0.20 m/s ≥ v8e. The real gain is expected from B2; A1 says how much.
- **Transfer:** under actuator-model physics, v9 survival is well above v8e's.
- **Runner:** the Pi runner passes all three sim2sim tests.
