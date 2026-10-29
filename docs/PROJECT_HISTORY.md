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
| 2026-10-18 → 19 | A4: servo model fitted to the real servos, v9c | `pi-bringup` |

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
  - Hip roll stays within −11…+5.5° in MuJoCo, inside the 8–9° inward contact the redesign audit found in the CAD.
- **What it means.** The policy now has a minimum walking speed of about 0.08–0.10 m/s. Below that it stands, which is safe. The sharp, command-relative tolerance gives no credit to a gait that is too fast for a 0.05 command, so a dynamic gait with a minimum speed loses to standing there. 0.10 m/s is a usable first walking speed for the robot.
- **v9d replaces v9c as the deployment candidate.**

### As-built limits, mass estimates and v9e (2026-10-19)
- **From the redesign audit of the as-built CAD (Fusion Main v24):**
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
- **v9e** = v9d recipe on the new model with a 90° tibia action scale, to reach the forward-knee range (redesign study T1). **Worse:**
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

## 10. Cross-cutting lessons

- Look at the robot (renders, videos) before believing any metric. Check which bodies touch the ground.
- Judge policies only with the fixed evaluation suite, never on training reward or termination rate.
- Make every implicit rule explicit, or RL finds the loophole.
- Check signed axes and conventions by experiment, not by assumption: hips, ankle linkage, tail direction and IMU sign each had a wrong first assumption.
- Measure the real hardware early. The servo speed measurement (~97 °/s vs 375 °/s modelled) invalidated every policy trained before it.
- Keep a bit-identical regression trace (`trace_env.py`) for every refactor and port. It caught every unintended change.
- When several reward variants hit the same ceiling, check joint limits and saturation first (the tibia limit).
- Compute on the Pi exactly as in training, down to float32 rounding.
