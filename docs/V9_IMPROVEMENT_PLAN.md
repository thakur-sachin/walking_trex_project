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

## 6. Tier A progress: CPU-only work (2026-10-15, branch `v9-tier-a`)

All of this runs on CPU physics, alongside v8e on the GPU. The Isaac Sim stack is unchanged: `trace_env.py` is IDENTICAL to `isaaclab-port`, and the Lab port is still bit-identical to it.

### A0. Evaluation diagnostics (`walk_rl/evaluate.py`)
New metrics:
- time at each joint limit;
- fraction of steps above 80% of stall torque, and mean |τ|/stall, per joint;
- foot roll in stance;
- stance fraction and step length per foot.

They print as a second table. A `--out` flag writes the JSON elsewhere. First reading on v8d (16 envs, CPU, DR on):

| cmd | tibia at limit L/R | load > 80% stall: tibia / femur / hip | step length L/R | dy per 20 s |
|---|---|---|---|---|
| 0.00 | 0.98 / 0.98 | 0.000 / 0.000 / 0.003 | – | −0.04 m |
| 0.10 | 0.59 / 0.37 | 0.012 / 0.018 / **0.159** | **+4.3 / −0.4 cm** | +1.03 m |
| 0.20 | 0.59 / 0.51 | 0.011 / 0.023 / **0.176** | **+4.7 / +0.4 cm** | +0.88 m |

What the numbers show:
- **Tibia range binds.** The tibias sit on the 45° placeholder limit 98% of the time standing and 37–59% walking, so the real range (B2) matters.
- **The hips carry the load.** They spend 16–18% of walking steps above 80% of stall torque.
- **v8d limps.** The left foot lands 4.3–4.7 cm ahead of the right, while the right barely passes the left. This asymmetry is behind the sideways crab.

### A3. Symmetry (`walk_rl/symmetry.py`, `walk_rl/test_symmetry.py`)
- **Mirror map:** swap left and right joints without negating any of them; flip gravity y, gyro roll/yaw, base vy and heading sin; shift the gait clock by half a period. Swap feet in the critic.
- **Algebra test:** passes (mirroring twice is the identity, the neutral frame is invariant, a lean flips, the hips swap).
- **Physics test (`--sim`):**
  - Setup: two CPU envs, the second the mirror image of the first, driven with a deliberately asymmetric action sequence.
  - Correct map: max observation error 0.03 over 15 steps, which is the simulator's own asymmetry while the robot topples.
  - Hip entries: correct map 0.008, against **0.30 for the plausible wrong map** (hips negated).
- **CAD:** symmetric about a centre line ~3 mm off the origin, with the hips 2 mm apart fore–aft. The limp is learned, not built in.
- **Training:** `ppo.symmetry.augment` (mirrored copy of every sample) and `ppo.symmetry.mirror_loss_coef` are added to walk_rl PPO, and to rsl_rl through Isaac Lab's `RslRlSymmetryCfg`. With either enabled, the normalizers are also fed mirrored observations. Both trainers are smoke-tested on CPU. Off by default.

### A6. Pi runner and sim2sim (`pi_runner/`)
**What it is.** `runner.py` is the deployed loop, NumPy only and float32 like training:
- observation frame, history, finite-difference servo velocity, gait clock and heading;
- then clamp → low-pass → scale → 0.24° quantisation → speed-limited goal → joint limits;
- plus a tilt safety latch.

`walk_rl/config.py` holds the torch-free config loading it needs; `task_core` re-exports it.

**Test 1 + 2 (`test_obs_parity.py`).** The runner is fed the Isaac env's raw sensor values for 3 envs × 200 steps at commands 0 / 0.10 / 0.20.
- Observations match to 2e-6.
- NumPy vs torch actor: 7e-7.
- **Servo goals are identical in 4800 of 4800 cases.** In float64 they were not: a saturated action lands exactly on a half tick (15° = 62.5 ticks, 45° = 187.5 ticks), and float64 rounds those differently from training. So the Pi runner must compute in float32.

**Test 3 (`sim2sim.py` + `mujoco_backend.py`).** MuJoCo 3.8 model built from the same URDF and robot_config:
- masses and inertias as PhysX has them;
- ankle linkage as a tendon equality;
- implicit-PD servos with a move-time ramp;
- termination rules from training.

Verified against Isaac on the first control step in free fall:
- torso gyro (0.124, −0.503, −0.003) vs Isaac (0.116, −0.469, −0.025);
- joint positions within 0.001–0.004 rad.

Three modelling lessons along the way:
- Teleporting the ankle each substep, as Isaac does, breaks momentum conservation in MuJoCo (5× the torso pitch rate).
- MuJoCo's soft linkage constraint is weak on a 72 g foot at 200 Hz (32–35° stretch while walking). It needs 1 kHz physics, which gives 0.35°.
- Contact stiffness decides the outcome. On MuJoCo's default 20 ms contact v8d falls within seconds; at 4 ms (near-rigid, like PhysX) it walks.

Result for v8d, 60 s episodes, friction {0.5, 0.8, 1.2} × bus delay {0, 1, 2 steps} × command {0, 0.10, 0.20}:

| floor | survived 60 s | speed (% of command) | sideways drift |
|---|---|---|---|
| near-rigid (4 ms) | **26/27** (one fall at 33 s: delay 2, friction 0.5, 0.10 m/s) | 46–88% | +1.0 to +5.2 m per minute, always +y |
| softer (10 ms) | 24/27 | 41–96% | similar |

- **Standing:** perfect.
- **Speed:** lower in MuJoCo than in Isaac (71–87%).
- **The crab** goes the same direction in both engines, so it is a property of the policy, which A2/A3 target.
- **Test 3 status:** not yet passed, because of that single fall.

### Findings for the model (not changed yet: they would change the task the comparison runs are using)
1. **base_link inertia is 2.6× too large for its mass.** Isaac keeps the URDF inertia, which belongs to 2.59 kg, but sets the mass to 1.0 kg. This probably makes balance easier than reality. Proposed fix: scale inertia with the mass override (a robot_config switch), retrain, and compare.
2. **Isaac's ankle model loads the servos with foot torque.** The ankle PD drive plus teleport reacts foot torque into the femur/tibia chain. The real parallelogram carries it through the linkage bars: by virtual work, the servos feel none of it. MuJoCo's tendon equality models the real mechanism.
3. **The policy needs a near-rigid floor.** Survival drops as contact softens, and MuJoCo's default floor already fails it. A TPU sole (C4) or carpet would be softer. Possible fixes: add compliant-contact DR in training, or keep the PLA sole.

## 7. Fixes for the model findings and A2 (2026-10-16, branch `v9-tier-a`)

The robot model is corrected in `robot_config.yaml` on this branch only. `isaaclab-port` keeps the v1–v8 model, so the Isaac Sim vs Isaac Lab comparison runs are untouched. The v9 worktree has its own `_converted_usd/` copy, since the Lab bake records the robot_config hash.

Every new option has an off setting that reproduces the old behaviour exactly. With both fixes switched off, the Isaac Sim `trace_env.py` is IDENTICAL to `isaaclab-port`. With them on, Isaac Sim and the Lab port are still bit-identical to each other (`smoke_lab.py --compare`, diff 0.000 on every channel).

### Finding 1: torso inertia (`mass.base_link_inertia`)
- **The URDF torso is a steel shell.** base_link's mesh is a closed hollow shell: 330.6 cm³ of material inside a 2362 cm³ outline. Its URDF mass of 2.59 kg is 7.85 g/cm³, Fusion's default steel; every other link is ~1.06 g/cm³ plastic. On the robot the shell is printed plastic and most of the mass is the battery inside it, with the Pi, hip servos and wiring (Sachin, 2026-10-16).
- `composite` (new default): the shell mesh at PLA density (0.41 kg), plus the rest of base_link_kg (0.59 kg) as one solid 10 × 7 × 4 cm box centred on the shell's centre of mass. Principal inertia comes out at (0.0016, 0.0023, 0.0025) kg·m².
- That is ~5× below the URDF tensor (0.0081, 0.0110, 0.0111) and ~1.8× below `scale_with_mass` (0.0031, 0.0042, 0.0043), which put all 1.0 kg into the walls.
- The tensor and its principal axes are authored in USD (the reconstruction error is 4e-10) and in MuJoCo (`fullinertia`).
- **Still an estimate.** The battery's mass, size and position (TODO in robot_config) would firm it up. Training randomizes it: `domain_randomization.base_inertia_scale_range` ×0.6–1.6, on top of the mass scaling. Mass DR now scales the randomised links' inertia along with their mass.
- v8d on this model (Isaac, CPU, no DR) still walks 10 s at every command.

### Finding 2: ankle load path (`ankle_linkage.sim_model`)
- `teleport_routed` (new default):
  - keeps the stable teleport + implicit ankle PD;
  - applies the ankle drive's torque τₐ (from the last physics step) as the linkage's generalized force on the rest of the leg: femur −τₐ, tibia +τₐ (the coefficients of −femur + tibia + ankle = 0);
  - the femur/tibia servos stop carrying foot moments; base_link does, as on the robot;
  - PhysX reports drive + applied effort as the joint force, so the servo torque reading (rewards, critic, eval) subtracts the routed part;
  - the routed torque is zero on the first substep after a reset, because Isaac Sim still reports the pre-reset force there and Lab does not. This keeps the two stacks bit-identical.
- Sign check (v8d, CPU, no DR, `walk_rl/test_ankle_linkage.py`): the femur+tibia servos' own mean |τ|/stall fell from 0.126 → 0.102 standing, 0.151 → 0.103 at 0.10 m/s, and 0.150 → 0.114 at 0.20 m/s. The wrong sign would raise it. The ankle carries 0.15–0.18 of stall on average, so this load path matters. v8d still walks 10 s at every command with either model.
- **Tried and rejected: a PhysX fixed tendon** over the three joints (the exact model, like MuJoCo's tendon equality). On the 72 g foot it is either too soft to hold the foot (stiffness ≤ 0.5: 2–4° error, falls within 0.5 s) or goes NaN (≥ 1, also at 1 kHz physics; damping makes it worse).

### Finding 3: soft floor (`sim.ground_compliance`, `walk_rl/ground.py`)
- PhysX compliant contact on the ground plane in both stacks, off (rigid) by default.
- `evaluate.py --ground-stiffness K [--ground-damping C]` evaluates any policy on a softer floor.
- Scale: k ≈ m/tc², so MuJoCo's 20 ms default is ≈ 4.5e3 N/m and 10 ms is ≈ 1.8e4 N/m.
- PhysX has no per-env material compliance in the tensor API, so it is one value per run (v9s), not per-env DR.

### A2: lateral rewards (weights 0 in `walk_task.yaml`, on in `experiments/v9a+`)
| term | form | v8d value per step | weight in v9a |
|---|---|---|---|
| `lateral_drift` | relu(\|1 s EMA of heading-frame v_y\| − 0.01 m/s) | 0.043 (EMA 0.052 m/s) | −10 |
| `roll_band` | exp(−100 · relu(\|roll\| − 0.1 rad)) | 0.84 | +1 |
| `foot_flat` | Σ over feet in contact of 1 − cos(sole tilt) | 0.007 | −20 |

The plan's 0.10 m/s dead-band would never trigger: v8d drifts at ~0.05 m/s. So it is 0.01 m/s, below the 0.015 m/s target (0.3 m per 20 s).

### Floor: indoor wood, PLA sole (Sachin, 2026-10-16)
- **Rigid contact is right.** Soft-floor training is not needed for the first floor, so the soft-floor run is dropped. `sim.ground_compliance` stays available for carpet later.
- **Friction is the risk.** PhysX averages robot and ground (0.5) friction, so v1–v8's DR range 0.4–1.2 trained only on 0.45–0.85 effective. PLA on wood is probably ~0.3–0.4, an estimate until B-tier tilt tests: tilt a board with a foot on it until it slides, μ = tan θ.
- In MuJoCo at friction 0.3, v8d makes almost no progress at 0.20 m/s (−3 to 21% of the command, turning 26–40°). The v9 configs widen the range to 0.1–1.2 (0.30–0.85 effective), and sim2sim sweeps 0.3 / 0.5 / 0.8.

### Ablation configs (`walk_rl/experiments/`, 4096 envs, 1500 iterations, pushes ramped 500→1200 to 0.35 m/s, action std ≤ 0.5)
| run | content |
|---|---|
| v9p | v8e task on the corrected model + inertia and wood-friction DR: the baseline |
| v9a | v9p + A2 |
| v9b | v9a + mirror augmentation (A3) |

Order (Sachin): v9 runs first, then the Lab comparison runs.

`evaluate.py` takes the robot model from the current `robot_config.yaml`, not from the run. Evaluating v8e on this branch therefore measures its transfer to the corrected model.

### Parity note (found 2026-10-16, predates v9)
On CPU physics the Isaac Sim and Lab stacks are bit-identical at 16 envs, but at 32 envs they drift apart from step 3 onward, starting at the 1e-5 level. `isaaclab-port` does the same (mean reward diff 0.065–0.088 over 150 steps), so it is not caused by the v9 changes. It looks like floating-point rounding that contact dynamics amplify (likely the env-origin layout differs between GridCloner and Lab's cloner), not a modeling difference. Each stack is deterministic run to run. Parity checks therefore use 16 envs; the GPU check after the v9 runs will show how it behaves at training scale.

## 8. v9 ablation results (2026-10-16, GPU, 4096 envs, 1500 iterations)

Every run is evaluated on the corrected model with `evaluate.py` on Isaac Sim (DR on, 256 envs, 20 s). "hard" adds the per-episode sensor errors (encoder offset, IMU delay). Push survival is the mean over the 0.1–0.4 m/s kicks at 0 and 0.10 m/s. v8e is listed both on its own (v8) model and on the corrected one, since only the second is a fair baseline.

| run | survival, hard (worst cmd) | v_x @ 0.20 | \|dy\| @ 0.20 (m / 20 s) | roll p2p @ 0.10 | \|yaw\| standing | push mean (0.4 m/s standing / walking) | MuJoCo falls (of 27) |
|---|---|---|---|---|---|---|---|
| v8e, v8 model | 0.98 | 0.174 | 3.35 | 21° | 1° | 0.973 (0.97 / 0.84) | 4 |
| v8e, corrected model | **0.83** | 0.182 | 2.62 | 20° | 3° | 0.911 (0.75 / 0.88) | — |
| v9p (corrected model + DR) | 0.98 | 0.172 | 1.13 | 22° | 1° | 0.934 (0.72 / 0.87) | 3 |
| **v9a** (+ A2 lateral rewards) | **0.99** | **0.184** | **0.26** | **15°** | 0° | **0.942** (0.76 / 0.86) | 7 |
| v9b (+ mirror augmentation) | 0.82 | 0.190 | 0.17 | 19° | 8° | 0.890 (0.68 / 0.75) | 9 |

**What this says**
- **The model fix matters.** v8e trained on the old model loses up to 17% survival on the corrected one. v9p, retrained on it, gets that back.
- **A2 fixes the crab.** Lateral drift at 0.20 m/s goes from 3.35 m (v8e) and 1.13 m (v9p) to 0.26 m per 20 s, under the 0.3 m target at every speed. Roll p2p drops from ~22° to 14–15°, still above the 12° target. Survival and push survival hold.
- **Mirror augmentation evens the gait but costs robustness.** v9b's stance times are symmetric (0.63/0.62 vs v9p's 0.56/0.46) and drift is lowest, but hard-sensor survival falls to 0.82, push survival drops, and it turns 6–8° while standing. Not adopted as is; a lower mirror-loss weight is a later option.
- **v9a is the base for v9c (A4 actuator model).**
- **MuJoCo transfer is the open risk.** In the 60 s sim2sim sweep (friction 0.3/0.5/0.8 × action delay 0–2 × cmd 0/0.10/0.20), v9a falls in 7 runs and v9b in 9, against 3 for v9p. Standing never falls. The falls spread over all frictions, mostly at delay 0 or 2, so the A2 gait seems tuned more tightly to PhysX contact. A4 and the MuJoCo sweep are the check for v9c.
- Speed is still capped near 0.19 m/s at the 0.25 command (tibia at its limit 48% of the time). B2's real joint ranges are expected to lift it.

Against the §5 targets: lateral ✓, roll ✗ (14–15° vs 12°), robustness ✓ (v9a ≥ v8e on the corrected model), speed ✓, transfer and runner not yet (A4 and sim2sim falls).

## 9. Isaac Sim vs Isaac Lab, first GPU results (2026-10-16, v8 model, `isaaclab-port`)

- **Benchmark (1024 envs):** Lab 46.4 ms/step vs Isaac Sim 52.4 (+13% samples/s), 562 vs 1842 MiB peak GPU memory, 2.5 vs 8.2 s app start.
- **Training parity:** v8e trained on Lab (same PPO) evaluates on Isaac Sim at 0.98–1.00 survival at every command, like v8e.
- **Evaluation gap (open):** the same checkpoint survives less when evaluated on Lab at 0.20–0.25 m/s. v8e: 1.00 → 0.96 / 0.82 (nominal), 0.99 → 0.91 / 0.74 (hard). v8e_lab: 1.00 → 0.84 / 0.86. CPU traces are bit-identical at 16 envs, so the gap is likely in the evaluation path at GPU scale (DR sampling, push timing, reset). Next step: diff the two evaluators' DR, push and reset logs for one seed.
- v8e_lab_rsl (rsl_rl on Lab) is still training; the comparison report in `docs/PLATFORM_COMPARISON.md` gets its numbers when the queue finishes.
