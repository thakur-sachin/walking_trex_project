# Trex project history

The complete record of the Trex walking robot project: what was tried, what worked, what failed and why each decision was made, in the order it happened. Numbers are kept as measured.

For the current state and how to use it, see [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md).

| Period | Phase | Branch |
|---|---|---|
| 2026-10-08 | Consolidation into one repo | `main` |
| 2026-10-09 | Robot pipeline in Isaac Sim 6.0.1 | `main` |
| 2026-10-09 → 13 | Walking task v1–v7 | `main` |
| 2026-10-15 | v8: robustness, observation history, heading | `v8-walk-robustness` |
| 2026-10-15 → 16 | Isaac Lab port, platform comparison | `isaaclab-port` |
| 2026-10-15 → 16 | v9 Tier A: diagnostics, symmetry, Pi runner, MuJoCo, model fixes | `v9-tier-a` |
| 2026-10-17 → 18 | Raspberry Pi bring-up, servo calibration | `pi-bringup` |
| 2026-10-18 → 19 | A4: servo model fitted to the real servos, v9c–v9e | `pi-bringup` |
| 2026-10-20 | measured masses, standing trap, gait scorecard, v9f–v9m, first real walks, servo refit | `pi-bringup` |
| 2026-10-21 | v9n and v9o, second real walk, ablation study, pruning | `pi-bringup` |

Every training run, with its change, targets, results and verdict, is also in [ABLATION_STUDY.md](ABLATION_STUDY.md).

---

## 1. Consolidation (2026-10-08)

Earlier work was spread over `~/trex-sim`, `~/RL_SB3`, `~/RexAI`, `~/RexAI_Direct`, `~/robot_learning_framework` and an `~/IsaacLab` 2.3 fork with uncommitted RexAI work.

- **Decision:** Sachin asked to stop the scattered exploration and keep one git-tracked repo. `walking_trex_project` was created; the old locations were left untouched.
- **Rule set from the start:** every physical parameter lives in `robot_config.yaml`, never in code. Values not yet measured are flagged as estimates.

## 2. Robot pipeline in Isaac Sim 6.0.1 (2026-10-09)

Result: `scripts/test_load_robot.py` takes the URDF to USD, applies the real config, sets materials and physics, and takes a screenshot. The robot loads and stands.

### Problems solved
| problem | fix |
|---|---|
| RTX renderer hard-crashed with no traceback | NVIDIA driver 535 → 595 (Isaac Sim 6.0.1 needs ≥ 550.90.07) |
| raw URDF import exploded in 1–2 steps | full `URDFImporter` with `override_joint_stiffness/damping`, not the bare converter |
| imported content behind a payload arc | explicit `Load()`; `Articulation(reset_xform_op_properties=True)` on first wrap |
| materials not showing | meshes are instance proxies: bind on the nearest non-proxy ancestor with `strongerThanDescendants` |
| auto-generated Fusion2URDF names | renamed to `head`, `left_femur`, … and simplified `visual`/`collision` children |
| screenshot after inspection showed the wrong pose | `timeline.stop()` resets the articulation; use `pause()` |

### Joint sign conventions
- **Method:** each joint was teleported to +20°; the experiment measured child and foot motion in the body frame (`scripts/joint_sign_convention_experiment.py` → `docs/JOINT_CONVENTIONS.md`).
- **Finding:** the hips are mirrored in raw sign, but a positive angle moves **either** foot inward.
- **Lesson:** never assume left/right sign symmetry.

### Ankle linkage
- Sachin confirmed that the ankle has no servo: a double 4-bar parallelogram keeps the foot parallel to the torso.
  - Lower parallelogram: 50 × 54 mm.
  - Upper parallelogram: 50 × 65 mm.
  - The idler links are not in the URDF.
- **First hypothesis `ankle = −(femur + tibia)` was wrong.** It left ~44° of foot tilt.
  - The URDF axes are femur (0, −1, 0), but tibia and ankle are (0, +1, 0).
  - Correct relation: **`ankle = femur − tibia`**, verified analytically (0.0000°) and live (foot within 0.35° of parallel).
  - **Lesson:** check signed axis vectors, not just axis direction.
- PhysX mimic joints take one reference joint and this relation needs two, so the ankle is enforced kinematically.

### Action space decision
**8 DOF:** the 8 real servos. Ankles are derived, never commanded. The observations are limited to what the robot measures.

## 3. Walking task v1–v7 (2026-10-09 → 13)

### Design decisions
- **A fresh task instead of porting the old IsaacLab `rl_task/`**, which was later removed. Isaac Lab was not on PyPI for Isaac Sim 6.0.1 at the time. The task is therefore a vectorized env directly on `isaacsim.core.experimental` with GridCloner, plus an in-repo PPO.
- **Asymmetric actor-critic.** The actor gets only deployable signals: IMU gravity and gyro, quantized servo positions, finite-difference velocities, previous action, command and gait clock. The critic also gets privileged state.
- **25 Hz control.** The LX-16A bus needs ~17 ms of wire time per cycle (8 writes plus 8 reads at 115200 baud), so 25 Hz leaves about 2× margin.
- **Small actor (128-128-64)** that exports as plain NumPy for the Pi.
- **Servo model:**
  - torque cap at stall;
  - 0.24° quantization;
  - a target ramp limited to the 375 °/s datasheet no-load speed;
  - bus delay;
  - a first-order command low-pass.

### Run log
| run | change | outcome |
|---|---|---|
| v1 | adaptive LR, weak yaw and chatter terms | circled ~100–116° per 20 s and chattered while standing; LR pinned at its floor; later found to be tail-propped |
| v1b | v1 + LR reset to 1e-3 on resume | regressed toward standing |
| v2 | + yaw tracking, `action_accel` −0.5 | never learned: the penalty charged ~−3 per step for exploration noise |
| v3 | command low-pass, `action_accel` −0.05, LR 1e-3 | best curve up to iteration ~750, then collapsed (one update with KL 0.24 at action std 0.13); best weights lost |
| v3b | v3 resumed with a KL guard | recovered too slowly; stopped |
| v4 | v3 recipe from scratch, KL guard, best checkpoints | strong numbers, but **walking on a tripod** (below) |
| v5 | + force-based foot contact, air-time term | killed early: tail still allowed on the ground |
| v6 | + any non-foot contact ends the episode | stood still (foot air time 0.000 at iteration 200) |
| v7 | + stale-contact fix, harness curriculum | **first valid biped** |

### What went wrong and what it taught
- **v4 used its tail as a third leg.**
  - Its metrics read 100% survival and ~97% tracking. Sachin spotted the tail in the rendered frames.
  - Under "any non-foot contact = fall", v4 survives 0% at every command.
  - The same check found that the height-based foot contact was wrong: both feet carrying 6–7 N read "no contact", because the linkage keeps the feet parallel to the torso and lean tilts them.
  - **Lessons:**
    - Any rule implied by the intent ("only feet touch the ground") must be an explicit termination.
    - When a metric looks too good, look at the frames and check what the robot touches.
    - Use contact forces, not link height.
- **Stale contact.** PhysX contact views keep the last force after a teleport. Terminate-on-contact then looped at episode length 1.
  - Fix: record each reading at reset and ignore it until it changes.
- **Exploration barrier.** With the tail forbidden, lifting a foot unaided was never discovered.
  - Fix: a training-only harness (70% of body weight upward plus a righting torque, full until iteration 150, zero by 450).
  - The first version's righting torque had the wrong sign, which made it positive feedback; the correct form is `up × g_b`.
- **Smoothness penalties on sampled actions** charge exploration noise. The fix is a structural command low-pass plus a mild penalty.
- **Adaptive LR** (rsl_rl style) pins at its floor when action std is small, because Adam keeps KL above target even at 1e-5.
  - Replaced by a fixed LR with linear decay and a per-minibatch KL guard.
  - Also: `best.pt` plus periodic checkpoints, and never resuming a collapsed run at a high LR.
- **Rendering:** GPU physics keeps transforms out of USD, so the headless renderer showed a frozen robot. The fix is CPU physics, one env and a static wide camera. The RTX renderer also runs out of memory next to a 4096-env training run.
- **Throughput:** disabling Fabric made headless training 5× faster. The GPU pair buffers must be sized to the env count, or contacts are silently missed.
- **The neutral pose is not statically stable** (it falls in ~1.2 s), so balance is learned.
- **Standing local optimum:** around iterations 100–250 runs pass through standing still, because standing already earns about half the tracking reward at low commands.

### v7 result (20 s, randomization on)
| command | survival | speed | heading change |
|---|---|---|---|
| 0.00 | 100% | −0.003 | +4° |
| 0.10 | 99% | 97% | −3° |
| 0.20 | 98% | 84% | −25° |

- Gait at 0.15 m/s: stance L/R 0.49/0.53, 3.8/4.0 contact transitions per second, mean lean 7°, tail up as a counterweight.
- **Weak spots:**
  - speed at 0.20;
  - heading drift;
  - lateral sway (|vy| ≈ 0.18 m/s);
  - standing jitter (`d_action` 0.10);
  - saturation 1.5–3%.
- The run directories of v1–v4 were deleted in the 2026-10-13 cleanup, together with the old `rl_task/` and REXAI docs.

## 4. v8: robustness (2026-10-15)

### Reward research
Rewards are computed in simulation and may use privileged state. What matters is whether the actor can *infer* the rewarded quantity from its sensors and their history.

| source | borrowed |
|---|---|
| Disney BD-X (arXiv 2501.05204) | per-episode encoder offset ±0.02 rad; pushes ramped in over training |
| Open Duck Mini v2 (BNO055 + hobby servos + Pi, the closest match) | `stand_still`; 0–2 step action delay and IMU delay; pushes; action history |
| humanoid-gym (XBot) | observation history; linear non-saturating speed term; `base_acc`; roll/pitch-rate term; foot-spacing band |
| Digit (arXiv 2404.19173) | heading tracking; push-recovery evaluation protocol |

Not adopted:
- Reference motion: no reference gait exists for Trex.
- Base-height tracking: no height sensor.
- Contact observations: no foot sensors.

### Changes
- **Heading:** heading-error observation and reward. The yaw reference is latched from the BNO055 fused yaw, with drift and an initial offset randomized. Sachin approved this sensor-contract change.
- **History:** 3-frame observation history (105 inputs), with the layout stored in the exported policy.
- **New rewards:** `track_heading`, `track_vx_lin`, `base_acc`, `base_rock`, `stand_still`, `torque_sat`, `feet_distance`. The speed kernel σ went from 0.08 to 0.06, and the command range widened to 0.05–0.25 m/s.
- **Robustness:**
  - per-episode action delay 0–2 and IMU delay 0–1;
  - encoder offsets;
  - pushes (0.25 m/s, ramped over iterations 500–800);
  - head/tail mass ±15%;
  - base centre of mass ±1.5 cm.
- **`walk_rl/play.py`:** a live viewer with keyboard speed command, pushes and telemetry. Green is the commanded velocity, red the actual, yellow the averaged actual, and the white line the heading reference.

### Base-mass finding
Widening the base mass randomization to ×0.7–1.5 made standing unlearnable. Falls per env per 10 s were 0.00 / 0.47 / 1.76 / 3.34 for ×0.7–0.9 / 0.9–1.1 / 1.1–1.3 / 1.3–1.5.

- Action delay made no difference.
- The range was kept at ×0.8–1.3.
- **If the real torso is well over ~1.2 kg, the legs are underpowered:** a hardware issue, not a policy one.

### Results (iteration 1000, same evaluation for all runs)
| metric | v7 | v8c (all v8) | v8d (+ speed-scaled clock) | v8b (no pushes) |
|---|---|---|---|---|
| survival, nominal | 81–94% | 97–100% | 99–100% | 98–100% |
| survival, hard sensors | 35–58% | 88–97% | 91–98% | 95–100% |
| speed at 0.25 | 62% | 67% | 64% | 71% |
| heading change per 20 s | 10–34° | 0.7–2.4° | 1.4–3.2° | 0.6–1.9° |
| standing jitter | 0.103 | 0.127 | 0.054 | 0.049 |
| push survival walking, 0.3 m/s kick | 0.84 | 0.95 | 0.93 | 0.88 |

- **What worked:**
  - Heading input fixed drift.
  - History plus per-episode delays fixed sensor robustness.
  - Pushes improved recovery up to 0.3 m/s.
  - The speed-scaled clock (v8d) cut sway at speed and halved standing jitter.
- **The v8b ablation:** the observation, reward and sensor changes, not the pushes, did the heavy lifting. Pushes buy 7–13 points of push survival for a small cost in speed.
- **What did not work: speed capped at ~0.15–0.17 m/s in every run.** At 0.25 m/s both tibias sat on their +45° limit 81–100% of the time. The 45° is a placeholder; the real range is needed.
- **Process mistakes caught:**
  - v8c's `best.pt` was iteration 250, a harness-assisted policy. `best.pt` is now chosen only after both curricula finish.
  - The training termination rate (30–60%) is not the fall rate: deterministic evaluation of a policy at 57% training termination survived 96–99%.
  - v8a (interface change only) was configured but never trained.

## 5. Isaac Lab port (2026-10-15 → 16)

**Goal (Sachin):** keep the Isaac Sim stack as the fallback, port the task to Isaac Lab and compare the two quantitatively.

### How
- The task logic moved into `walk_rl/task_core.py` (`TrexTaskCore`). Each stack only implements `_sim_*`/`_dr_*` hooks.
- The refactor was verified bit-identical with `walk_rl/trace_env.py`: 150 CPU steps with harness, pushes, a kick and resets; max diff 0.0.
- `walk_lab/env.py` is a `DirectRLEnv` on Isaac Lab 3.0.0-beta2.patch1 (the build for Isaac Sim 6.0.1), in conda env `isaaclab6`.
- The robot USD is baked by `walk_lab/bake_usd.py` with the same config steps and stores the config hash.
- `walk_lab/physics_dump.py` diffs every authored physics attribute between the two scenes.
- **Result:** the Lab port is **bit-identical** to the Isaac Sim stack on CPU physics (every observation, reward and done over 150 steps with 48 resets).

### Isaac Lab pitfalls found (each would have silently changed the task)
1. `activate_contact_sensors=True` skips nested rigid bodies, so only `base_link` reported contacts. It also zeroes sleep thresholds.
2. `ContactSensor` zeroes its data on reset, which would arm the stale-contact bug. The port reads raw contact views.
3. `PhysxCfg()` authors its own scene values, for example a bounce threshold of 0.5.
4. The ground-plane extension is not loaded by the Lab launcher.
5. The wrench composer adds rounding (a 0.09% reward drift). The harness now uses the direct PhysX call.
6. Smaller ones:
   - rsl_rl's `inference_mode` tensors;
   - persistent `extras`;
   - `handle_deprecated_rsl_rl_cfg`.

### Results
- **Benchmark (1024 envs):** Lab 46.4 ms/step vs Isaac Sim 52.4 ms (+13% samples/s), 562 vs 1842 MiB GPU memory, 2.5 vs 8.2 s start-up.
- **v8e** (16384 envs, 3000 iterations, Isaac Sim): survival 0.98–1.00. Its action std ran away from ~0.27 to 3.89 under 0.35 m/s pushes, hence `ppo.max_action_std` 0.5 in later configs.
- **v8e_lab** (same trainer on Lab), evaluated on Isaac Sim: 0.98–1.00, so training is at parity.
- **Open gap:** the same checkpoint survives less when evaluated *on Lab* at 0.20–0.25 m/s (v8e 0.96/0.82 nominal). CPU traces are identical, so the suspect is the GPU-scale evaluation path (randomization sampling, push timing, resets). Not resolved.
- **Determinism note:** at 32 CPU envs the stacks drift apart from step 3 at the 1e-5 level, also on the original branch. This is likely floating-point rounding amplified by contacts. Parity checks use 16 envs.
- **v8e_lab_rsl** (rsl_rl): stopped at iteration 2475 of 3000 when the GPU was given to the v9 runs; not evaluated.

## 6. v9 plan and Tier A (2026-10-15 → 16)

### The plan
Source: Sachin's research brief ("Trex biped walking improvements"), checked claim by claim against the data. Ordering rule (Sachin): **A** free simulation/software work → **B** free measurements with parts he has → **C** cheap upgrades.

| brief's claim | verdict after checking our data |
|---|---|
| sway ~0.2 m/s is mostly physics | agree: stop penalizing instantaneous vy |
| penalize the ~25° roll | agree; the figure is peak-to-peak (±8–12°) |
| heading drift needs symmetry | yaw was already fine (0–3° per 20 s), **but v8d crabs ~1 m per 20 s sideways, always to +y**: a learned asymmetry |
| 84% speed tracking comes from a loose velocity kernel | **disagree:** the kernel is already steep; the cap is the tibia limit |
| check for phantom root mass | checked: no phantom mass (1.778 kg total) |
| the servo model lacks armature, friction and back-EMF | **agree: the biggest free item (A4)** |
| 3-frame history is too short | agree, worth an ablation |
| mirror map: swap hips without negating | agree, unit-test it |
| MuJoCo sim2sim before hardware | agree |

Tier A items:
- A0: evaluation diagnostics.
- A1: sensitivity studies (tibia limit, base mass). **Not run yet.**
- A2: lateral rewards.
- A3: symmetry.
- A4: actuator model.
- A5: action and observation pipeline. **Not run yet.**
- A6: Pi runner and MuJoCo.
- A7: 50 Hz variant. **Not run yet.**

### A0: evaluation diagnostics
New metrics: time at each joint limit, load above 80% of stall per joint, foot roll in stance, stance fraction and step length per foot. First reading on v8d:
- **The tibias sit on the 45° limit** 98% of the time standing and 37–59% walking.
- **The hips carry the load:** 16–18% of walking steps are above 80% of stall.
- **v8d limps:** the left foot lands 4.3–4.7 cm ahead of the right, while the right barely passes the left. This is behind the crab.

### A3: symmetry
- **Mirror map** (`walk_rl/symmetry.py`): swap left and right joints without negating, flip gravity y, gyro roll/yaw, vy and heading sin, and shift the gait clock by half a period.
- **Algebra tests:** pass.
- **Physics test:** two mirrored CPU envs. The correct map gives 0.008 error on the hip entries, against **0.30 for the plausible wrong map** (hips negated).
- **CAD:** symmetric to ~3 mm, so the limp is learned.
- **Trainers:** both got mirror augmentation and mirror loss, off by default.

### A6: Pi runner and MuJoCo sim2sim
- `pi_runner/runner.py` is the deployed loop in NumPy float32.
- **Parity with the simulator:**
  - observations match to 2e-6;
  - NumPy vs torch actor 7e-7;
  - servo goals identical in 4800 of 4800 cases.
  - **In float64 they were not identical:** a saturated action lands exactly on a half tick (15° = 62.5 ticks), and float64 rounds it differently from training. **The Pi must compute in float32.**
- MuJoCo model built from the same URDF and config:
  - Teleporting the ankle as Isaac does broke momentum conservation (5× the torso pitch rate).
  - MuJoCo's soft linkage constraint stretched 32–35° at 200 Hz; at **1 kHz** it holds 0.35°.
  - On MuJoCo's default 20 ms contact, v8d fell within seconds; at **4 ms** (near-rigid, like PhysX) it walks.
- **v8d in MuJoCo:** 26 of 27 60 s episodes survive. It crabs +y in both engines, so the crab is a property of the policy.

### Model findings and fixes (`v9-tier-a`)
- **Torso inertia.**
  - The URDF torso is a hollow shell authored as Fusion's default **steel** (2.59 kg).
  - Isaac kept that inertia at 1.0 kg mass, i.e. 2.6× too large.
  - New default `base_link_inertia: composite`: the shell at PLA density (0.41 kg) plus a 10 × 7 × 4 cm battery/electronics box (0.59 kg). This is ~5× below the URDF tensor.
  - Randomized ×0.6–1.6 since it is an estimate.
- **Ankle load path.**
  - Isaac's ankle drive reacted foot torque into the femur/tibia servos; the real parallelogram carries it through the bars.
  - New default `sim_model: teleport_routed`: the ankle torque is applied as femur −τ, tibia +τ, and subtracted from the servo torque reading.
  - Servo load fell 19–32%.
  - **A PhysX fixed tendon was tried and rejected:** on the 72 g foot it was too soft (falls in 0.5 s) or produced NaN.
- **Floor:** indoor wood with PLA soles (Sachin).
  - Rigid contact is right, so the soft-floor run was dropped.
  - Friction is the risk: PhysX averages robot and ground friction, so v1–v8 trained only on 0.45–0.85 effective.
  - v8d barely moves at friction 0.3 in MuJoCo.
  - v9 trains on 0.30–0.85 effective.
- **A2 lateral rewards:**
  - `lateral_drift`: 1 s EMA of v_y with a 0.01 m/s dead-band. The planned 0.10 m/s dead-band would never trigger, since v8d drifts at 0.05.
  - `roll_band`: ±0.1 rad.
  - `foot_flat`.
- Every new option has an off setting that reproduces the old behaviour bit-identically.

### v9 ablation results (4096 envs, 1500 iterations, evaluated on the corrected model)
| run | survival hard (worst) | v_x at 0.20 | \|dy\| per 20 s at 0.20 | roll p2p at 0.10 | MuJoCo falls of 27 |
|---|---|---|---|---|---|
| v8e on its own model | 0.98 | 0.174 | 3.35 m | 21° | 4 |
| v8e on the corrected model | **0.83** | 0.182 | 2.62 m | 20° | — |
| v9p (corrected model + randomization) | 0.98 | 0.172 | 1.13 m | 22° | 3 |
| **v9a** (+ A2) | **0.99** | 0.184 | **0.26 m** | **15°** | 7 |
| v9b (+ mirror augmentation) | 0.82 | 0.190 | 0.17 m | 19° | 9 |

- **The model fix matters:** v8e loses up to 17% survival on the corrected model.
- **A2 fixes the crab** (3.35 m → 0.26 m) and cuts roll. **v9a became the base.**
- **Mirror augmentation** evens the gait but costs robustness (hard survival 0.82, standing yaw 8°). Not adopted.
- **The open risk was MuJoCo transfer:** v9a fell in 7 of 27 episodes.

## 7. Raspberry Pi bring-up (2026-10-17 → 18)

Live robot work was staged, and each stage was started on Sachin's go-ahead.

### Machine
- Pi 4 (8 GB), Bookworm 64-bit.
- Key-based ssh from the PC, static IP via NetworkManager.
- Locale fixed, CPU governor `performance`.
- Python venv without torch.
- `pi_setup.sh` (idempotent system setup) and `deploy.sh` (rsync plus venv).
- **Compute is ample:** policy step 0.76 ms (2% of a tick). Pi and PC goals are bit-identical (0 of 24,000 differ).

### BNO055
- **Problem:** the Pi's hardware I²C corrupted bit 7 of bytes read (OPR_MODE 0x08 read as 0x88), giving 0.14 rad/s gyro spikes. The BNO055 clock-stretches, and the hardware block mishandles it.
- **Fix without rewiring:** bit-banged I²C bus 3 on the same pins. 0 of 1000 integrity reads wrong.
- Burst read 3.4 ms; fresh 100 Hz fusion data every tick. A 30 s 25 Hz dry run had 0 overruns.

### IMU orientation
- **Method:** two tilt rounds with the robot held: nose down, left side down, turn left.
- **My error along the way:** I told Sachin that left side down gives g_y < 0. It is **g_y > 0**, because g points the way gravity pulls. The check prompt was fixed.
- **Result:** chip +y is forward and chip +x is right: `sensor_to_body = [[0,1,0],[-1,0,0],[0,0,1]]`. Both rounds agree.

### Servo bus
- **Adapter:** CH340. Ids 1–8 answer.
- With the battery off, the servo logic runs from USB 5 V: enough to read, not to move.
- **Reply reader bug:** it asked for more bytes than a reply has, so every read waited out the timeout. Fixed by reading only the missing bytes.
- **Speed problem:** one position read takes 3.6 ms round trip, so all 8 took 28.9 ms. Together with 8 moves, the IMU and the policy that is ~41 ms, **over the 40 ms tick**. Options considered:
  1. IMU in a thread.
  2. One-tick-old joint readings (would need a retrain).
  3. A 74HC126 buffer on the Pi UART.
  4. 20 Hz.
- **Fix: pipelining.** The servo answers ~1.7 ms after a request; the rest is USB latency. Requests now go out every 2.0 ms and replies are matched by id.
  - Sweep: a 1.6 ms gap lost replies in 8 of 300 runs, 1.7 ms in 1, 1.8–2.2 ms in none.
  - All 8 read in 15.7 ms instead of 29 ms. Moves go out as one write, with a TX-drain guard before reads.
- **Full loop with real servo reads, on battery, 60 s:** 20.4 ms mean, 23.3 ms max, 0 overruns, 0 missing replies in 12,000 reads.

### First powered moves (robot held in the air)
- **Test:** `servo_identify`, each servo alone: 10° ramps over 500 ms and full-speed steps out and back, logged every ~3.75 ms. All 8 reached every target.
- **Step response (unloaded, 10°):**
  - 20% of the move after ~37 ms, 90% after ~110 ms;
  - **~97 °/s mid-move, against the 375 °/s the simulator allowed** (15° per tick vs ~4° measured);
  - overshoot ≤ 1.4°, final error ±1.2°.
  - **This became the main input for A4.**
- Battery 7.06–7.16 V; sag < 0.02 V.

### Mapping and calibration
- **Id map** (from the order Sachin saw them move): 1 L hip, 2 L femur, 3 L tibia, 4 R hip, 5 R femur, 6 R tibia, 7 head, 8 tail.
- **Directions,** measured by hand with torque off (`hw_check watch`): +1 for all except the left hip (−1).
  - **Correction to `JOINT_CONVENTIONS`:** positive tail is tip **up**. The doc's "down" was the tail's centre-of-mass motion.
  - The femur and tibia servo horns drive the parent link.
  - One round was redone because two joints were moved out of order.
- **Zero pose (first pass):** the tail level was at tick −49, outside the servo's 0–1000 range. The left femur could not reach −45°.
  - Sachin detached both links while I set the servos to tick 500 (tail) and 375 (left femur).
  - The tail was refitted level. The left femur could not be refitted to the reference pose as exactly.
  - Zeros re-measured with the robot standing straight. Every joint's range now fits; the tightest is the left femur at tick 12 at −45°.
- **Stored limits:** the servos' EEPROM limits did not match the joint ranges on 7 of 8 servos (id 4 rested outside its limit). They were rewritten to each range plus 3°, read back and verified. The old limits were backed up (`~/trex/logs/servo_limits_backup_20261018_164543.json`).
- **Safety in `HardwareBackend`:** it refuses to drive until every id, direction and zero is set, and every joint range fits 0–1000 ticks.
- **Verification:** standing straight, every joint reads 0 ± 0.2°. Gravity shows the robot standing ~5° left side down.

### Stand stages
- **Stage 1** (`servo_hold --go`): ramp to zero, hold 10 s. Every joint within 0.2°.
- **Stage 1b** (`servo_hold --sweep`): each joint alone to +10°. Reached +7.2 to +9.4°, others within 1.0°. **All 8 directions confirmed by Sachin.**
  - The head and right hip settled 2.3–2.8° short. **Open:** likely mechanical drag, still to check.

## 8. A4: servo model fitted to the real servos, v9c (2026-10-18 → 19)

### Model (`walk_rl/actuator.py`)
An explicit torque per physics substep. The same code runs in torch (both Isaac stacks) and NumPy (MuJoCo):
- the setpoint is the move-time ramp, delayed by a latency;
- `duty = clamp(kp · (error − deadband), ±1)`: the firmware P gain, saturating;
- `τ = duty · stall(V) − stall(V)/no_load(V) · q̇`: a DC motor torque-speed line, scaled by battery voltage;
- Coulomb friction that can stop the joint but never reverse it within a substep (`min(fc, J_arm · |q̇| / dt)`). It holds static loads like the gearbox and is stable at 200 Hz, where a smooth tanh law was not;
- armature on each joint.

### Fit (`walk_rl/fit_actuator.py`, data in `assets/robots/trex/servo_logs/`)
- **Data:** 32 logged moves plus the settled parts of 16 sweep holds.
- **Method:** each segment is simulated as one joint, with inertia and gravity from the MuJoCo model at its mid-angle. SciPy `differential_evolution` minimizes the RMS position error with the exact simulator code at 200 Hz.
- **Result:** **RMS 0.46°**, one parameter set for all 8 servos.
  - First fits with a smooth friction law gave 0.41–0.44° and spread the poorly separated parameters, so randomization covers that spread.
  - The friction parameter hit its bound in one fit; the bounds were widened and the fit redone with the exact discrete friction.

| parameter | nominal | spread of fits | randomization |
|---|---|---|---|
| kp | 12.2 /rad | 10.3–15.2 | ×0.75–1.3 |
| deadband | 0.41° | 0.29–0.54° | 0.1–1.0° |
| latency | 4 ms | 1.4–4.1 ms | 0–10 ms |
| armature | 0.021 kg·m² | 0.014–0.021 | ×0.6–1.6 |
| Coulomb friction | 0.31 N·m | 0.15–0.49 | 0.1–0.55 N·m |
| no-load speed | 0.46 × datasheet | 0.44–0.57 | 0.38–0.65 |
| stall torque | datasheet | not identifiable unloaded | ×0.75–1.05 |
| battery | 7.11 V | — | 6.6–8.4 V |

**Against the old servo model,** the real servo has:
- about half the speed;
- duty saturation within ~5° of the goal;
- friction and a deadband;
- an armature 4–45× the link inertia, so joints accelerate far more slowly.

### Checks
- `implicit_pd` (still the default) traces identical to before A4.
- With `lx16a`, Isaac Sim and Isaac Lab are identical (16 CPU envs, every channel; Lab USD re-baked).
- **MuJoCo replay** of the 28 non-tail moves on the full robot: 0.20–0.72° RMS per move.
  - The tail moves are off by ~45°. They were logged before its horn was re-indexed, outside the model's joint range, so this is an artifact.

### Transfer gap: existing policies on the fitted servo (20 s survival, randomization on)
| policy | walking, old servo | walking, fitted servo | standing, fitted servo |
|---|---|---|---|
| v8e | 0.86–0.98 | 0.08–0.19 | 0.88 |
| v9p | 0.98–1.00 | 0.02–0.29 | 0.79 |
| v9a | 1.00 | 0.05–0.09 | 0.56 |
| v9b | 0.95–0.99 | 0.21–0.46 | 1.00 |

**No existing policy could walk on the real servo's dynamics.** v9a, the best on the old servo, was among the worst. In MuJoCo it fell within 1.4 s at 0.10 m/s.

### v9c = v9a + `actuator.model: lx16a` (1500 iterations, 83 min)
| cmd | survival nominal / hard | speed | roll p2p |
|---|---|---|---|
| 0.00 | 1.00 / 1.00 | — | 1.8° |
| 0.05 | 1.00 / 1.00 | stands still | 2.0° |
| 0.10 | 1.00 / 1.00 | 42% | 8.5° |
| 0.15 | 1.00 / 0.99 | 100% | 17.2° |
| 0.20 | 1.00 / 0.99 | 88% | 15.1° |
| 0.25 | 1.00 / 0.99 | 77% | 13.4° |

- **The transfer gap is closed in Isaac:** walking survival went from 0.05–0.09 to 0.99–1.00 on the fitted servo.
- **Push recovery:** ≥ 0.98 up to 0.3 m/s kicks.
- **MuJoCo with the same servo model: 27 of 27 episodes survive** (v9a: 10 of 27). Speed there is 49–70% of a 0.20 command.
- **v9c no longer walks on the old servo model.** It is specialized to the slow servo, which is expected, since the old model is not the real servo.
- **Visual check (2026-10-19):**
  - MuJoCo videos of the deployed loop (`pi_runner/render_mujoco.py`) and an Isaac Sim render show a feet-only gait with the tail low as a counterweight. At 0.15 m/s in Isaac: stance 0.53/0.51, 4.2 steps/s, mean tilt 4.5°.
  - When standing, the policy pitches the head down and the tail up, with its goals at the +45° limits of head, tail and tibias. When walking it switches to head and tail back.
- **Open:**
  - **No gait below ~0.12 m/s.** The likely cause is that on the slow servo a slow gait costs more than standing, so the policy rounds low commands down to standing. That is safe, but the first steps on the robot should be slow, so this needs fixing first.
  - Roll is 13–17°, against the 12° target.

### v9d: slow-gait fix (2026-10-19)
- **Diagnosis.** At a 0.05 m/s command, standing still still earned ~50% of `track_vx` (fixed σ 0.06) and 60% of `gait_contact`, because each planted foot was scored against its stance phase on its own. On the slow servo that beat the cost of stepping.
- **Changes** (`experiments/v9d_slow_gait.yaml`, switches in `task_core`, off by default, trace IDENTICAL with them off):
  - `track_vx` σ = 30% of the command (0.015–0.06 m/s);
  - `gait_contact` pays only when both feet match the clock;
  - 35% of walking commands drawn from 0.03–0.12 m/s, range starting at 0.03.
- **Result** (fitted servo, randomization on, 20 s):

| cmd | 0.03 | 0.05 | 0.07 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|---|
| v9c speed | — | stands | — | 42% | 100% | 88% | 77% |
| v9d speed | stands | stands | stands | **99%** | 102% | 88% | 76% |

  - Survival 0.99–1.00 at every command, also with hard sensors (0.10: 94% of the command).
  - **MuJoCo: 36 of 36 one-minute episodes survive** (friction 0.3/0.5/0.8 × delay 0–2 × cmd 0/0.05/0.10/0.20). At 0.10 it moves at 14–82% of the command (v9c: 0%), at 0.20 45–60%.
  - Costs vs v9c: servo saturation 3–5% (was 1–3%), push survival walking at a 0.4 m/s kick 0.84 (was 0.96).
  - Hip roll stays within −11…+5.5° in MuJoCo, inside the 8–9° inward contact found in the as-built CAD.
- **What it means.** The policy now has a minimum walking speed of about 0.08–0.10 m/s. Below that it stands, which is safe. The sharp, command-relative tolerance gives no credit to a gait that is too fast for a 0.05 command, so a dynamic gait with a minimum speed loses to standing there. 0.10 m/s is a usable first walking speed for the robot.
- **v9d replaces v9c as the deployment candidate.**

### As-built limits, mass estimates and v9e (2026-10-19)
- **From a check of the as-built CAD (Fusion Main v24):**
  - the hip meets the torso at +10° inward, so the hip range is now −30…+7° and the hip servos' stored limits −31…+8°, written and read back on the robot;
  - the foot meets the shank at femur − tibia = +39° and the lower idler meets the foot at −57°, so a coupled limit of −55…+37° is enforced identically in `task_core` and the Pi runner (bit-identical on 20,000 random goals);
  - the tibia range is opened to −45…+90°.
- **Mass estimates** with the real battery (TalentCell YB1206000, 370 g): torso 0.9 kg, head 0.105 kg, tail 0.058 kg (printed parts, 30% infill assumed). Total 1.50 kg, was 1.78.
- **A per-run action scale** (`actions.joint_scale_deg`) separates the policy's interface from the physical range. The Pi runner checks it against the scale stored in the policy.
- **Checks on the new configuration:** Isaac Sim vs Isaac Lab trace IDENTICAL; runner vs simulator 0/4800 servo goals differ.
- **v9d on the new model:**
  - Isaac (fitted servo, randomization): survival 0.98–0.99; 0.10 → 91%, 0.20 → 80% of the command (was 99% / 88%);
  - MuJoCo: 35 of 36 episodes (one fall at 0.10); speed somewhat lower than on the old model.
  - **It is still the deployment candidate.**
- **v9e** = v9d recipe on the new model with a 90° tibia action scale, to reach the forward-knee range. **Worse:**
  - it walks only from 0.15 m/s (52%) and makes 36–42% at 0.20–0.25;
  - push survival at a 0.4 m/s kick 0.69–0.71 (v9d 0.84–0.91);
  - MuJoCo 34 of 36, and almost no progress below 0.20 in the video.
  - **Likely cause:** doubling the tibia scale doubles the joint-space exploration noise and halves the action resolution around the posture the gait needs.
  - **Not adopted.** A later attempt should shift the tibia's action centre forward instead (scale ±67.5° around +22.5° covers −45…+90° at almost today's resolution).

### Measured masses, the standing trap and v9f–v9i (2026-10-20)
- **Measured robot** (Sachin, scale): whole robot 1307 g; torso with its 4 servos and all electronics 805 g; head and tail are the smaller versions; the legs are mirrored; no chassis arch.
  - The parts weighed alone add up to more than 1307 g. The whole-robot weight is the anchor: head 0.078, tail 0.038, each leg 193 g (URDF leg links ×0.893).
  - Torso inertia from parts (`base_link_inertia: parts`): battery 370 g as a box under the plate, electronics 179 g, 4 servos, printed rest. Torso CoM z 0.2478 → 0.2236 m.
- **v9d on the measured model:** 51–55% of the command, survival 0.86–0.89, MuJoCo 32 of 36.
- **v9f** (tibia ±67.5° about +22.5°, `actions.joint_center_deg`): runner parity OK, but it learned to stand at every command.
- **The standing trap.** v9e, v9f and v9g (v9d recipe on the measured model, control) all drifted to standing once the training pushes started (~iteration 500). Standing earned 4.95 per step against 5.26 for walking: the heading and yaw rewards pay almost the same whether the robot walks or not.
- **v9h** (`reward.progress_gate`: posture and smoothness rewards scale with v_x / cmd while walking): it walks again (survival 0.97–1.00), but at 0.10–0.16 m/s whatever the command (199% at 0.05, 61% at 0.25). MuJoCo 21 of 36: falls at friction 0.5 and 0.8, 1–3 m sideways drift per minute.
- **v9i** (`progress_gate_mode: tracking`: the gate peaks at the commanded speed):

| cmd | 0.05 | 0.075 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|
| Isaac speed | 93% | 89% | 86% | 85% | 76% | 67% |
| Isaac survival | 0.98 | 0.98 | 1.00 | 0.98 | 0.98 | 0.97 |

  - Hard sensors: 0.05 → 90%, 0.10 → 86%, survival 0.95–0.99. Push at 0.4 m/s: 0.74 standing, 0.89 walking. Roll 10–16° p2p; saturation 1–6%.
  - Runner parity OK (0/4800 servo goals differ).
  - **MuJoCo 32 of 36:** the 4 falls are all at 0.20 on friction 0.5/0.8 (head on the ground). At 0.10 all 12 survive at 53–63% of the command. At 0.05 it moves at 1–35%. The video shows a feet-only gait; MuJoCo speed is ~40% of the command in the render.
  - **v9i replaces v9d as the deployment candidate**, with commands limited to 0.10–0.15 m/s for the first steps. The Isaac–MuJoCo speed gap (86% vs ~55–60% at 0.10) stays open.

### Gait scorecard, v9j and v9k (2026-10-20)
- Sachin: v9i's video looked worse than v9c/v9d; train with clear objectives driven by data and first principles.
- **`walk_rl/gait_scorecard.py`**: 10 fixed targets per run (MuJoCo survival and speed, Isaac speed, left-right femur/tibia, time at a range end, roll spread, drift, step symmetry, saturation, hard-sensor survival). `sim2sim.py` now records the gait statistics it needs.
- **Findings on the measured robot:**
  - v9c and v9d hardly walk (MuJoCo 0% / 15% of 0.10); their good videos were on the older, heavier model.
  - v9i scooted: left foot stepping backward (−1.1 cm) and tibias 61° apart, a gait PhysX allows and MuJoCo does not.
  - head and tail move the CoM < 2 mm with the measured masses, yet every policy pinned them at a range end; v9j's pinned-down tail caused two MuJoCo falls.
- **v9j** = v9i + `leg_symmetry` (1 s mean of left − right femur and tibia) + weak `base_height`: legs symmetric (−1/+4°), drift 9% of forward distance (v9i 142%); 4/10 targets.
- **v9k** = v9j + `joint_pose` on head and tail: head/tail near neutral, steps 1.1/1.1 cm at 0.10, **MuJoCo 24/24 at friction 0.3–0.5**, 9/12 at 0.8; 5/10 targets. Runner parity: 1 of 4800 goals 1 tick off (NumPy/torch rounding), now allowed up to 0.1%.
- **Open:** MuJoCo speed (43% of 0.10) and the slow forward-pitch falls at friction 0.8 (feet in contact, femurs move but no catch step over 0.5 s).

### Replay diagnosis, v9l / v9m and the first real walks (2026-10-20)
- **Replay diagnosis (v9k, 0.10 m/s, friction 0.5, no delay):** the Isaac servo goals replayed open loop in MuJoCo, and the policy run closed loop in both.
  - Servo tracking error is the same in both engines, so the servo model does not explain the Isaac–MuJoCo gap.
  - In Isaac the sole centres slid forward 541 mm while in contact, against 889 mm of body advance: about 60% of Isaac's progress was skating. Double support 57% in Isaac, 4–8% in MuJoCo.
  - The old `foot_slip` measured the foot origin and cost ~0.002/step. PhysX foot link velocities are not the foot's motion, because the ankle linkage is teleported each substep.
  - Fix: `reward.foot_slip_point: sole` (sole centre from `robot_config foot_sole.center_in_foot_m`, slip as a finite difference over each control step).
- **v9l** (v9k + sole slip at −100) learned to stand: v9k's slip × 100 was −1.5/step, half the reward. 1/10 scorecard targets.
- **v9m** (sole slip at −10): Isaac 88/88/82% of 0.05/0.10/0.15, MuJoCo 59% at 0.10, sole slide 140 mm (v9k 541), double support 22%, hard-sensor survival 0.96; 5/10 targets (fails: drift 34%, right step 0.7 cm at 0.10, saturation 5.1%, MuJoCo speed, 3/36 MuJoCo falls).
- **`pi_runner/walk_live.py`** runs a policy on the robot and logs every tick (stages 4 and 6). Tested on the PC with a fake MuJoCo backend first.
- **Real robot, v9m (logs in `assets/robots/trex/servo_logs/`):**
  - Hanging, 16 s: 40.0 ms ticks, none late, tick work ≤ 23.6 ms; servos reach a goal about 2 ticks later, error 1.0–2.9° without that delay.
  - First floor run: aborted at 5.4 s; the left tibia missed its position reply twice in one tick (cable fine, no power fault). `HardwareBackend(max_missed_reads=2)` now keeps the last good reading for up to 2 ticks and logs every miss; the next runs had none.
  - Floor, 16 s at 0.10 m/s: no fall, **0.41 m in 10 s = 41% of the command** (Isaac 88%, MuJoCo 52–59%). Roll spread 4°, pitch 3°, legs symmetric (±1.6°), yaw +7° in 13 s. Actions at their limit 11–14% on the left hip and tibias. Knees 6–8° more bent than in sim.
  - Video (front view, hand-held): alternating steps with clear foot lift; foot slip not visible from the front.
- **Servo load test (`pi_runner/servo_load_test.py`, standing squats and sine):** real and MuJoCo agree within 0.6° static and 0.3° on the sine. The load is too light to tell the models apart.
- **Real walking load vs the model:** mean goal − q (delay removed) on the left hip and tibias: real +0.4 / +0.5 / −0.1°, MuJoCo +3.9 / +2.4 / +2.9°, Isaac +5.5 / +2.6 / +4.4°. The real servos hold their goals; both sims gave way 2–5°. The real knees are bent more because the policy sends deeper goals there.
- **Servo refit (`walk_rl/fit_actuator_walk.py`):** the real floor and hanging goals replayed open loop in MuJoCo plus the 10-18 logs, differential evolution on 4 scales. kp 12.2 → 4.56, Coulomb 0.31 → 0.116 N·m, no-load ×0.46 → ×0.775, stall ×1 → ×1.84 (effective). RMS: floor 3.2° → 1.4°, hanging 2.9° → 0.5°, 10-18 logs 0.46° → 0.47°. With it v9m makes 52% of 0.10 in MuJoCo (was 59%) and falls 7/9 at 0.20.
- **Attitude sensitivity:** standing, the real robot reads roll −0.6°, pitch +1.2° (sim 0°), and its pitch changes ~1° with the crouch (sim: none); walking roll differs ~2° from MuJoCo. In MuJoCo, IMU offsets of 1–3° move v9m between 0 and 0.08 m/s at 0.10, and a pitch offset makes it crouch. New `observation.imu_mount_offset_deg` (per-episode roll/pitch offset on gravity and gyro; off by default, trace IDENTICAL).
- **v9n** = v9m + refit servo model (DR re-centred) + IMU mount offset ±2°; torque terms now relative to the fitted holding torque.

### v9n, the second real walk, and v9o (2026-10-21)
- **v9n in sim:** MuJoCo 72% of 0.10 (v9m 52% on the same model), speed stable under 1–3° IMU offsets (v9m dropped to 0.014 m/s), saturation 0.6%. Fails: left hip near its +7° end 55% of the time, Isaac steps 0.4 / 2.7 cm. Posture changed: tibias +35° (v9m −14°).
- **Parity test failure was the test:** `test_obs_parity.py` did not switch off the new IMU mount offset. Fixed; v9n 0/4800 goals differ.
- **v9n on the floor (0.10 m/s, 10 s):** 0.25 m (25%) and a steady **+49° turn** (12, 15, 4, 11, 7° per 2 s). The turn did not follow the head/tail swing (correlation 0.1); it came from a one-sided posture: head −14°, tail +18°, left hip +3° (61% near its end), right hip −9°. MuJoCo showed the same offsets but turned only 5°. Head and tail goal − q 12° mean (v9m 2–4°) with ±40° swings.
- **v9o** = v9n + mirror loss 0.5 (the soft form; v9b's mirror augmentation had cost robustness). Head/tail −1/−1°, hips −3/−5°, no joint near a range end, MuJoCo heading change −0.2°, MuJoCo 70% of 0.10, survival 35/36, hard-sensor survival 0.97. New: it moves sideways 0.8–2.9 m per ~4 m forward in MuJoCo, always to the same side. Deployed on the Pi.
- **v9o on the floor (0.10 m/s, 10 s):** **0.46 m (46%)**, the best real result so far (v9m 41%, v9n 25%). Heading held: −8° net, ±12° per 2 s window, no steady turn. Slight drift to the right, no crab (the MuJoCo drift was to the left, so it did not transfer). Mean head −1°, tail −2°, hips −7/−3°, femurs +2/−5°, tibias +18/+19°; left hip never above +4°. Tracking goal − q 4–7° on every joint (v9n head/tail 12°). Gait cycle 1.8 Hz, ~25 mm per cycle. Temperatures 24–26 → 28–31 °C. One missed reply (left tibia).
- **First side video (`media/Robot_walk_v3.MOV`):** colour tracking of the soles confirms the distance (0.46–0.48 m) but cannot resolve slip per step: the two soles overlap from the side and the mask fails in the darker half of the frame. Next try: a bright marker on one foot and more light.
- **Replay of the real v9o walk (`pi_runner/replay_real_walk.py`):**
  - The real goals replayed open loop in MuJoCo give joint angles within 0.6–2.5° RMS of the real ones: the M5 servo model holds for v9o.
  - Open loop, the MuJoCo robot walks 0.56–0.62 m along its heading (real 0.46 m) but spins 270–306° (real −8°). The real heading corrections do not fit the simulated body.
  - Closed loop (the policy in MuJoCo, same schedule): 0.62–0.76 m, but the joints differ from the real ones by 5–10° RMS. On the floor the policy uses the left hip more (−6.5° mean, twice the swing), puts the right femur 6° further back and swings the right tibia 30% less. The real robot rolls +1.9° while walking (MuJoCo 0.1°).
  - Not the single cause, in MuJoCo: torso CoM ±15 mm sideways, IMU roll offset ±3°, IMU lag up to 3 ticks (v9o keeps 55–78%; v9m and v9n drop to 19–66%).
  - The real roll lags the simulated roll of the same joint motion by 40–80 ms on all three floor walks.
  - A no-slip odometry from joint angles and IMU attitude works in MuJoCo (0.73–0.98 of the true distance) but not on the real log: the soles rise only a few mm, so the stance choice follows the IMU, and a 1–2 tick shift moves the result from −0.5 to +0.4 m. Real slip still needs a foot marker.
- **Contact sweep (`pi_runner/contact_sweep.py`, v9o closed loop, 45 settings + one-sided tests):** friction μ 0.15 gives 47% of the command (μ 0.5: 63%), 3° of joint play 50–57%; contact softness (4–20 ms) and torsional friction (no effect with condim 3) change little. No setting reproduces the real one-sided posture; one-sided encoder offsets (±4°) and one-sided play (4°) on the hips and right leg do not either, because the policy compensates them. So the speed gap fits a low real friction (below the 0.30 training minimum) or joint play; the posture has a cause outside these models. Next: measure μ (tilt test) and compare a second surface (carpet).
- **Tilt test on a wood plank** (zero pose under torque, `servo_hold.py --go --hold-s 120`, which now logs the IMU tilt): feet slid at 15° head downhill (static μ 0.27, below the 0.30 training minimum); tail downhill the robot toppled at 7° (model: 12.8°), with the joints within 1.2°. So the real centre of mass sits ~22 mm in front of the heel edge (model 41 mm), or the heel edge carries no load.
- **MuJoCo with these numbers:** moving the torso CoM 20–31 mm back with μ 0.18–0.22 is the first change that gives the real left-hip pattern (−5°, ±5° swing). At μ 0.27 the gait snaps back to the nominal one; at 31–35 mm most runs fall onto the tail. A lead, not yet a model: the CoM along x needs a direct measurement.
- **Heel contact and balance point:** a paper strip does not go under either heel edge with the robot standing under torque, so the whole sole carries load. On a pencil across the soles the robot balances 30 mm ahead of the heel edge (model 40.8 mm). **Robot model M6:** the estimated battery and electronics positions move 25 mm back (30.3 mm). With M6 alone, MuJoCo does not show the real left-hip pattern at any friction (0.15–0.5): the pattern needed a ~20 mm torso shift with μ ≤ 0.22, and M6 is ~17 mm on the torso. The real posture still has no reproduced cause.
- `servo_hold.py` now tolerates 2 consecutive missed replies (a single left-femur miss stopped a 300 s hold at 164 s). Missed replies today: left tibia 2, left femur 1.

### v9q, v9r and the hip goal clip (2026-10-22)
- **v9q** = v9o + training friction down to the measured floor (`sim.ground_friction: 0.1`, robot 0.14–1.6 → 0.12–0.85 effective), on M6. MuJoCo at μ 0.15 / 0.20 / 0.27: 80–91 / 82–91 / 81–84% of 0.10, 36/36 (v9o on M6: 37–50 / 49–66 / 72–76%, 31/36). Fails: right hip within 3° of its +7° inward end 49–51% of the walk (hips −8 / +2°), Isaac 46% at 0.05, hard survival 0.91. Scorecard 6/10; sideways drift 15% (v9o 43%).
- **v9r** = the v9o recipe unchanged on M6 (control, trained in parallel). The hypothesis (keeps v9o's low-friction weakness) was false: 68–144% of 0.10 at μ 0.15–0.27 (overspeed). But it stands with both hips at +6° (100% at the end), does not walk at 0.05 (1%, tail +25°), roll spread 22°. Rejected.
- **Cause of the hip clip:** the goal is clipped to the physical range (−30…+7°), the joint-limit penalty only sees ±30°. v9o on M5 never used the clip; on M6 (CoM 25 mm further back) the policy does. New option `reward.joint_limit_range_margin_deg` (trace identical); config `v9s_range_limit.yaml` = v9q + 3°.
- **v9s** (trained 11:06–12:46Z): no joint at a range end (hips ≤ 3%), hips −2 / −2° while walking, heading change 0° in the MuJoCo video. MuJoCo at μ 0.15 / 0.20 / 0.27: 61–71 / 56–66 / 81–87% of 0.10, 36/36. Isaac 99 / 90 / 82% at 0.05 / 0.10 / 0.15. Scorecard **8/10**, the best so far; fails step ratio (0.6 / 1.4 cm) and hard survival 0.78 at 0.25 m/s (0.95 at 0.10). Head +15° while standing (+6° walking). Deployed on the Pi.
- **v9s on the floor (0.10 m/s, 10 s, `media/Robot_walk_v4.mov`):** **0.36 m (36 %)**, below v9o (46 %). Two runs, same posture: hips −2 / −2° (as in MuJoCo), head +6°, tail +3°, heading −1° and +6° net. The hip fix transferred. But the gait shuffles: joint swing 2–3° std (v9o 5–8°) and foot lift 5.8 mm in MuJoCo (v9o 14–16 mm, v9m 13 mm). MuJoCo still gives 83–104 % with the same 2–3° swings, so its speed comes from contact effects the real floor does not give. The scorecard has no foot-lift target; Isaac steps were already short (0.6 / 1.4 cm). One battery-off abort and one USB adapter reset (nothing moved) before the good run; missed replies: left tibia 2, right tibia 2.
- **v9s on carpet (same schedule, two runs):** **more than 0.70 m (> 70 %)**, twice the wood result and the best real distance so far. Heading +54° and +15° net (both to the left, growing over the walk); left hip about 2° more inward than on wood (+0.3 / −0.3° vs −2°); joint swing 2.6–3.3°. Same policy, same joint motion: the wood loss is foot slip. MuJoCo at μ 0.15–0.27 still gives v9s 61–87 %, so the MuJoCo contact does not reproduce the wood slip of a 6 mm shuffle (sliding μ below the static 0.27, or contact softness). The Pi dropped off Wi-Fi twice today for ~20 s.

### v9t: foot lift (2026-10-23)
- **Cause:** v9s shuffled; its swing_clearance reward was 0.024 per step against 0.093 for v9o. The scorecard gets an 11th target, foot lift min @0.10 >= 12 mm (median over the MuJoCo grid): v9m 16, v9o 10, v9s 6 mm. New counts: v9m 7/11, v9o 6/11, v9s 8/11.
- **v9t** = v9s + `swing_clearance: 8.0` (was 2). Targets before training: foot lift >= 12 mm, femur and tibia swing >= 4 deg, keep the v9s targets, real wood >= 50 %.
- **Result:** foot lift 39 mm; swing std hips 8, femurs 10, tibias 12 deg (v9s 2-3) - a stepping gait. MuJoCo 36/36, 100 % of 0.10 (own model 119 / 91 / 59 % at 0.05 / 0.10 / 0.20); at mu 0.15 / 0.20 / 0.27: 58-65 / 55-72 / 58-77 % (two of nine conditions below 60). Misses: hips -6 / -6 deg (target +-5), roll spread 19 deg (v9s 5), tibias at the -45 deg end 17 % median (33 % worst episode), Isaac 70 % at 0.15. Head -22 to -35 deg while walking. Scorecard 8/11.
- **v9t on wood (0.10 m/s, 10 s):** barely moved forward; it turned on the spot, **+78 deg** left by the IMU. Half of it (+37 deg) in the first 2 s, while the legs go from the standing pose (tibias +45, femurs +10 deg) to the walking crouch (femurs -26, tibias -10 to -13 deg); the rest slowly over 9 s. Joint swing 5-7 deg (v9s 2-3) and roll -2..+6 deg, so the bigger steps transferred, but the feet pivot instead of push. MuJoCo at mu 0.15-0.27 also turns v9t (-38..+25 deg in 60 s) but still walks. Rejected.
- **v9t on carpet:** run 1 fell 0.7 s after the stop (head swung to +41 deg, tipped forward-left; safety stop at 55 deg). Run 2: **~0.70 m forward and ~0.40 m to the left** (0.81 m straight line), yaw about +34 deg net, roll -12..+14 deg, no fall. Same forward progress as v9s on carpet, with more side drift and rocking. Before run 1, the USB servo adapter reset twice at servo power-on (nothing moved; fixed by reseating the cable). A later zero-pose hold stopped on 3 missed right-hip replies.
- **v9t on wood, run 2:** **~0.50 m forward (50 %)** and ~0.20 m to the right, then it hit a wall on its right. The best wood result so far (v9o 46 %), and it meets the v9t real target. Yaw at most -17 deg, +4 deg net; roll +2..+8 deg; joint swing 4-8 deg; no fall. Run 1 made ~0 m with a +78 deg turn, so the wood result varies strongly from run to run. Verdict back to candidate; more wood runs are needed.

### MuJoCo on each policy's own robot model (2026-10-22)
- `pi_runner/sim2sim_home_model.py` rebuilds each run's robot from the `robot_config.yaml` it trained with (the run's copy, else its registry commit, else the model's first commit; the URDF is unchanged since 10-09) and runs the standard sim2sim sweep with the servo model it trained with. `MujocoTrex`, `PolicyRunner` and `sim2sim.py` take an optional robot config; provenance hashes the config actually used.
- Checks: v9s reproduces its standard sweep bit for bit; v9n and v9o on M5 match their original results (33/36, 72 %; 35/36, 70 %).
- 25 policies scored; the ablation study has two new columns (own model: survived, speed @0.10). Only M0 cannot be rebuilt exactly (MuJoCo always routes the ankle torque).

### Ablation study and pruning (2026-10-21)
- **Ablation study:** `walk_rl/experiments/registry.yaml` (one entry per run: parent, change, why, targets, verdict, lesson) and `walk_rl/ablation_study.py`, which adds the measured results from the run directories and writes `docs/ABLATION_STUDY.md`. The simulated robot changed six times (models M0–M5); the study labels each result with the model it was measured on and marks results that were re-run on a later model.
- **Provenance:** every new run, evaluation and sim2sim result stores the code commit and a hash of the parsed robot model (`common/provenance.py`); each run also keeps a copy of `robot_config.yaml`.
- **`walk_rl/run_experiment.sh`:** training plus the full fixed evaluation as one command (it replaces an ad-hoc chain script).
- **Pruned:** the experiment configs of v8a–v9m and v9p (they would now train on today's robot model, not the one they ran on; each is in the registry as `git show <commit>:<path>`), stale examples and statuses in the README and knowledge base, the leftover `_eval_*.npz` files (evaluate.py now deletes its temporary policy), and references to the stopped CAD redesign.
- **Found while building the study:** v9f trained and was evaluated on model M3 (0.9 kg torso), not on the measured masses as written earlier. Several older evaluation files (v9c, v9d) were re-run on later models and overwrote the originals; the study marks them.

---

## 9. Decision log

| date | decision | why |
|---|---|---|
| 10-08 | one consolidated repo | scattered work could not be tracked |
| 10-08 | physical parameters only in `robot_config.yaml` | the hardware is real and its measurements change |
| 10-09 | 8-DOF action, ankles derived | only 8 servos exist; ankles are a linkage |
| 10-09 | kinematic ankle enforcement each substep | PD-only ankle 8–14° off; mimic joints can't express two terms; teleport + aligned PD gives 0.5° |
| 10-09 | in-repo env + PPO on Isaac Sim | Isaac Lab was not installable for Isaac Sim 6.0.1 then; full control over PPO safeguards |
| 10-09 | 25 Hz control | bus wire time ~17 ms per cycle |
| 10-13 | only feet may touch the ground | v4 used its tail as a leg |
| 10-13 | contact from forces, stale-contact guard | the height test was wrong; PhysX reports stale force after a reset |
| 10-13 | training-only harness | foot lifting was an exploration barrier |
| 10-13 | fixed LR + KL guard | adaptive LR pinned at its floor; v3 collapsed |
| 10-15 | heading observation from BNO055 fused yaw | heading drift up to 34° per 20 s; approved by Sachin |
| 10-15 | base mass randomization kept ×0.8–1.3 | wider made standing infeasible |
| 10-15 | keep Isaac Sim as fallback, port to Lab, compare | Sachin's requirement; platform choice made on data |
| 10-15 | v9 tier order A → B → C | Sachin: free work before measurements before purchases |
| 10-16 | composite torso inertia, routed ankle torque | the URDF torso was steel; the real linkage carries foot torque |
| 10-16 | wood friction range, no soft-floor training | the real floor is wood (Sachin) |
| 10-16 | v9a as base, mirror augmentation not adopted | A2 fixed drift; symmetry cost robustness |
| 10-17 | BNO055 on bit-banged I²C | hardware I²C corrupts bit 7 |
| 10-18 | pipelined servo reads at 2.0 ms | sequential reads overran the 40 ms tick |
| 10-18 | servo EEPROM limits = joint range + 3° | servos enforce limits even if the Pi misbehaves |
| 10-18 | fitted LX-16A model, train v9c on it | the real servo is ~4× slower than the old sim; every old policy failed on it |
| 10-19 | fix the low-speed gait before first steps | first steps on the robot should be slow |
| 10-20 | whole-robot weight anchors the mass fit | parts weighed alone overcount (Sachin: fit approved) |
| 10-20 | progress gate in tracking mode (v9i) | standing nearly matched walking on reward; v9i tracks 0.05–0.15 |
| 10-20 | sole-centre slip by finite difference, weight −10 (v9m) | ~60% of Isaac progress was skating; −100 (v9l) made standing pay more |
| 10-20 | tolerate 2 consecutive missed servo replies (Sachin) | one missed reply aborted the first floor run; misses are logged |
| 10-20 | servo model refit to the real walks, train v9n | the 10-18 model gave way 2–5° under walking load, the real servos 0.5° |
| 10-20 | IMU mount offset randomization ±2° | 1–3° attitude offsets swing v9m's MuJoCo speed from 0 to 0.08 m/s |
| 10-21 | v9o = v9n + mirror loss, not mirror augmentation | the real v9n turn came from a learned one-sided posture; augmentation cost robustness in v9b |
| 10-21 | registry + generated ablation study; provenance in every result | results from six robot models were being compared without labels |
| 10-21 | keep only the current experiment configs in the tree | old configs train on today's robot, not the one they ran on |
| 10-21 | v9o is the deployed policy | 46% of 0.10 on the floor, straight; best real result |
| 10-21 | robot model M6: CoM anchored to the measured balance point | the estimated payload positions gave a CoM 11 mm too far forward |
| 10-21 | training ground friction option, v9q at 0.12–0.85 effective | the measured wood μ 0.27 was below the reachable training range |
| 10-22 | v9r rejected; joint-limit penalty over the physical range (v9s) | on M6 both runs ride the +7° hip goal clip, which the symmetric penalty does not see |
| 10-22 | v9s is the floor-test candidate | 8/10 scorecard, hips off the clip, keeps low-friction speed |
| 10-23 | foot-lift target on the scorecard; v9t (clearance x4); wood 0 m then 0.50 m | v9s 36 % on wood vs > 70 % on carpet: a 6 mm shuffle slips |

## 10. Cross-cutting lessons

- Look at the robot (renders, videos) before believing any metric. Check which bodies touch the ground.
- Judge policies only with the fixed evaluation suite, never on training reward or termination rate.
- Make every implicit rule explicit, or RL finds the loophole.
- Check signed axes and conventions by experiment, not by assumption: hips, ankle linkage, tail direction and IMU sign each had a wrong first assumption.
- Measure the real hardware early. The servo speed measurement (~97 °/s vs 375 °/s modelled) invalidated every policy trained before it.
- Keep a bit-identical regression trace (`trace_env.py`) for every refactor and port. It caught every unintended change.
- When several reward variants hit the same ceiling, check joint limits and saturation first (the tibia limit).
- Compute on the Pi exactly as in training, down to float32 rounding.
- Replay real logs in the simulator (real goals in, compare joint angles) before changing a model. A test that seemed decisive (standing squats) loaded the servos too little, and the first explanation (servos give way more on the robot) was the opposite of the data.
- Sample random numbers only when a new option is on, or the regression trace changes even with the option off.
- A small learned asymmetry that a simulator barely shows (5° of turn in MuJoCo) can dominate on the real floor (49°). Check posture symmetry (mean head, tail and hip angles) before a robot test.
- A simulator-only artefact can go the other way too: v9o's one-sided drift in MuJoCo did not appear on the floor. Test on the robot before spending a training run on a sim-only symptom.
- An estimator that works in simulation can fail on real data for a reason the simulation hides (here: IMU timing). Check its sensitivity to each real input before trusting its number.
- Label every result with the robot model it was measured on. Results overwritten after a model change look like regressions or gains that never happened.
