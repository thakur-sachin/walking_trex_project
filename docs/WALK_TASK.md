# Trex walking RL task (`walk_rl/`)

Forward-velocity-tracking walking task on the current pipeline (Isaac Sim 6.0.1,
PhysX GPU, real `robot_config.yaml`). No Isaac Lab dependency: a vectorized env
directly on `isaacsim.core.experimental` + `GridCloner`, and an in-repo PPO. It
replaced an older, un-ported IsaacLab task that was removed from this repo (see git
history before the commit that adds this note).

```
conda activate isaac6; export OMNI_KIT_ACCEPT_EULA=YES
python walk_rl/smoke_env.py --num-envs 256                 # env sanity check
python walk_rl/train.py --name run0 --iterations 1000      # ~3 s/iter @ 4096 envs
python walk_rl/train.py --name run0 --resume --reset-lr    # continue
python walk_rl/evaluate.py --name run0 --ckpt best.pt [--no-dr]   # fixed-command sweep
python walk_rl/render_gait.py --name run0 --cmd 0.15       # gait capture (CPU physics)
walk_rl/start_tensorboard.sh [start|stop|status] [port]    # http://localhost:6006
```

Outputs land in `outputs/walk_runs/<name>/` (gitignored): `train_log.jsonl`,
`best.pt` + `best_numpy.npz` (best smoothed reward; the NumPy file is the
deployable actor), `checkpoint.pt` (latest), `checkpoints/ckpt_*.pt` (every 50
iterations), and `walk_task.yaml` (the exact config used). TensorBoard uses the
system `tensorboard`: `tb_sync.py` mirrors each run's jsonl into event files, so no
change to the `isaac6` env; each run's config is attached under Text.

## Interface (what the real robot can sense and do)

| | |
|---|---|
| **Action (8)** | position targets for `actuated_joints` (head, tail, hip/femur/tibia x2). Ankles are never commanded; they follow `ankle = femur - tibia`. |
| **Actor obs (v8: 105 = 3 x 35)** | the last 3 frames (newest first) of: IMU gravity (3), gyro (3), command vx (1), 8 servo positions / limit (quantized to 0.24 deg), 8 finite-difference joint velocities, previous action (8), gait clock sin/cos (2, zeroed when standing), heading error sin/cos (2). v7 was one 33-dim frame without heading. The exact layout is stored in each exported policy's meta (`obs_layout`). |
| **Critic obs (v8: 50)** | one clean frame + base linear velocity, foot contacts, relative foot heights, servo torques (privileged, training only) |
| **Heading error (v8)** | `yaw - yaw_ref`, where the Pi latches `yaw_ref` from the BNO055 fused yaw when walking starts. Requires the BNO055 in a fusion mode (IMU or NDOF); training models a slow yaw drift (+/-3 deg/min, estimate) and a 0.15 rad initial offset from the reference. Sachin approved this sensor-contract change on 2026-10-15. |
| **Not observed** | base linear velocity, foot contact (no sensors on the robot), ankle position (no ankle servo/feedback) |

Servo model (all values from `robot_config.yaml` `actuator`): torque cap (1.667 N*m,
randomly scaled 0.85-1.0), position resolution (commands and readings quantized),
target ramped across the control period at no more than the no-load speed (like the
LX-16A move-time command), 0-1 control-step bus delay, and a first-order low-pass on
the command (`actions.smoothing_alpha`, must be mirrored in the Pi loop).

Control rate is 25 Hz (`walk_task.yaml sim.control_hz`). One cycle at 115200 baud is
~6.9 ms writing 8 targets + ~9.7 ms reading 8 positions, ~17 ms of wire time, so
25 Hz leaves ~2x margin before Pi/USB-serial latency. Raise it only once the real
driver loop rate is measured.

## Task rules and training method

- **Only the feet may touch the ground.** Contact by the head, tail, base, thighs,
  shins or ankles ends the episode with a penalty. Head and tail exist to help
  balance, not to be props. (Without this rule an earlier policy used its tail as a
  third leg; see "What went wrong".)
- **Contact comes from PhysX contact forces** (>1 N for a foot, >0.2 N for body
  links), not link height. The 4-bar linkage keeps each foot parallel to the base,
  so body lean tilts the feet and a link-origin height threshold misreports contact.
  Contact is privileged information: the actor never sees it.
- **Stale-contact guard.** Contact views keep the last reported force after a
  teleport (no "contact lost" event fires on a reset; natural lift-off does clear
  it). Each sensor's reading is recorded at reset and ignored until it changes
  (stale readings are bit-identical, real ones are not). Without it,
  terminate-on-contact looped at episode length 1.
- **Training-only harness curriculum.** With the tail forbidden the policy stood
  still (foot air time stayed 0.000): lifting a foot unaided is an exploration
  barrier. The harness applies an upward force on the base (70% of body weight) plus
  a righting torque, full strength until iteration 150, annealed to 0 by 450, so
  stepping is safe to discover and the final policy balances unassisted. The
  restoring torque is `up x g_b` (body frame); the first version had the sign flipped
  and was positive feedback. `evaluate.py` never turns it on.
- **Stepping rewards**: swing clearance measured relative to the other foot (both feet
  share the base's tilt, so the difference is lean-independent), a feet-air-time term
  that pays the touchdown that ends a real swing, and a phase-clock contact schedule.
- **PPO safeguards** (all learned the hard way): observation normalizer frozen during
  a rollout; fixed LR with linear decay; a per-minibatch KL guard that stops updating
  before a step that would overshoot (a single update at lr=1e-3 once moved KL to
  0.24 and collapsed a run); `best.pt` and periodic checkpoints so a collapse is
  recoverable. An rsl_rl-style adaptive LR was removed: at small action std Adam's
  steps keep KL above target even at lr=1e-5, so the controller pins the LR at its
  floor and learning stalls.
- **Chatter** is removed structurally (the command low-pass above) with only a mild
  second-difference penalty: a smoothness penalty on *sampled* actions cannot tell
  exploration noise from chatter, and at -0.5 it charged ~-3/step for exploration
  alone and the run never learned.

## Reward research (2026-10-15, for v8)

Rewards are computed in simulation only, so they may use privileged state; the limited
sensors constrain the actor's *observations*. What matters is whether the actor can infer
the rewarded quantity from IMU + servo readings + their history, otherwise the term is noise.

| source | robot | borrowed for v8 |
|---|---|---|
| Disney BD-X, [arXiv 2501.05204](https://arxiv.org/html/2501.05204) | small biped, 50 Hz PD | per-episode encoder offset +/-0.02 rad; pushes ramped in over training; action rate + acceleration penalties (already had both) |
| Open Duck Mini v2, [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) | BNO055 + hobby servos + Pi (closest to Trex) | `stand_still` (joint motion at cmd 0); 0-2 step action delay and IMU delay; pushes every few seconds; action history in the obs |
| humanoid-gym (XBot), [repo](https://github.com/roboterax/humanoid-gym) | full-size humanoid | observation history; `track_vel_hard` (linear, non-saturating speed term); `base_acc`; roll/pitch-rate part of `vel_mismatch_exp`; `feet_distance` band |
| Digit, van Marum et al. 2024, [arXiv 2404.19173](https://arxiv.org/html/2404.19173v1) | full-size humanoid | heading (yaw-orientation) tracking; push-recovery evaluation protocol |
| Isaac Lab H1 velocity task | full-size humanoid | nothing new (feet slide, air time, joint deviation already covered) |

Not adopted: reference-motion imitation (no reference gait exists for Trex; the clock
reward shapes the gait), base-height tracking (no height sensor, unclear the actor can
infer it), foot-contact observations (no foot sensors; history lets the policy infer contact).

v8 reward additions (all in `walk_task.yaml reward.weights`, 0 disables): `track_heading`
1.5, `track_vx_lin` -2.0, `base_acc` 0.5, `base_rock` 0.5, `stand_still` -0.3,
`torque_sat` -2.0 (relu above 80% of stall), `feet_distance` -1.0; `track_vx_sigma`
0.08 -> 0.06; command range 0.05-0.25 m/s. Robustness: per-episode action delay 0-2 steps,
IMU delay 0-1, encoder offsets, pushes (0.25 m/s, 0.5 rad/s at full scale, ramped in over
iterations 500-800), head/tail mass +/-15%, base CoM +/-1.5 cm.

**Base mass range finding.** Widening `base_mass_scale_range` to 0.7-1.5 (because the real
mass is unknown) made training fail to stand: under the full harness with small random
actions, falls per env per 10 s were 0.00 / 0.47 / 1.76 / 3.34 for base mass x0.7-0.9 /
0.9-1.1 / 1.1-1.3 / 1.3-1.5. Action delay made no difference (fall prob/step 0.005-0.006
for 0, 1, 2 steps). The servos cannot hold up a base much heavier than ~1.1-1.2 kg in this
model, so the range is back to 0.8-1.3. **If the real base_link weighs well over ~1.2 kg,
the legs are likely underpowered: a hardware issue, not a policy one.**

## Ankle model (experiment-backed decision)

The ankle has no servo; the 4-bar linkage forces `ankle = femur - tibia` (confirmed,
see `robot_config.yaml ankle_linkage`). Tried on 64 envs with random femur/tibia
targets, measuring foot-vs-base tilt (should be 0):

| approach | tilt mean | note |
|---|---|---|
| PD to f-t target, stiffness 40 | 11 deg | cannot track under ground load |
| same, stiffness 400 | 14 deg | |
| + effort compensation (route drive torque via femur/tibia) | 8 deg | halves servo load, still loose |
| **kinematic enforcement every physics substep + aligned PD target** | **0.5 deg** | adopted |

Caveat: the kinematic teleport is not energy-conserving and lets the foot torque be
reacted "for free". A real parallelogram reacts foot torque into the base (no servo
load), so this is the closest of the options, but still an approximation.

## Results (v7 `best.pt`, iteration 1000; unassisted, feet-only contact)

`evaluate.py`, 256 envs, 20 s, deterministic actions, domain randomization on:

| command | survive | vx (% of cmd) | distance | heading change | servo sat. |
|---|---|---|---|---|---|
| 0.00 m/s | 100% | -0.003 | -0.10 m | +4 deg | 0.0% |
| 0.05 | 100% | 0.055 (110%) | +1.01 m | -0.3 deg | 1.5% |
| 0.10 | 99% | 0.097 (97%) | +1.84 m | -3 deg | 1.5% |
| 0.15 | 98% | 0.137 (91%) | +2.58 m | -13 deg | 1.5% |
| 0.20 | 98% | 0.168 (84%) | +3.11 m | -25 deg | 1.6% |

DR off is similar (98-100% survival, 0.154 m/s at 0.20). NumPy export matches torch
to 4e-7. Rendered at 0.15 m/s (CPU physics, 6 s): stance fraction L/R **0.49 / 0.53**
(symmetric), contact transitions **3.8 / 4.0 per s** (clock target 3.6), mean body
lean 7 deg (max 13), tail held up and back as a counterweight and never touching
down, no falls.

Weak spots: tracking falls off at speed (84% at 0.20 m/s); heading drifts -25 deg
over 20 s at 0.20 m/s; lateral sway is large (|vy| ~0.18 m/s, path 6.7 m for 1.8 m
forward: weight-shifting side to side); standing still is a little jittery
(`d_action` 0.10, path 0.9 m); servo saturation 1.5-3%. The DR/noise ranges are
estimates and nothing has run on hardware.

## What went wrong on the way (so it is not repeated)

| run | change | outcome |
|---|---|---|
| v1 | adaptive LR, weak yaw/chatter terms | circled (~100-116 deg/20 s) and chattered at standing; LR pinned at floor; later found tail-propped |
| v1b | v1 + LR reset to 1e-3 on resume | regressed toward standing |
| v2 | + yaw tracking, `action_accel` -0.5 | never learned: the penalty charged exploration noise |
| v3 | command low-pass, `action_accel` -0.05, LR 1e-3 | best curve to it ~750, then collapsed (single-update KL 0.24); best weights lost (only the latest ckpt was kept) |
| v3b | v3 resumed with the KL guard | too slow to recover; stopped |
| v4 | v3 recipe from scratch, KL guard, best ckpts | smooth, strong tracking numbers, but **tail-propped** (below) |
| v5 | + force-based foot contact, air-time term | killed early: tail still allowed on the ground |
| v6 | + terminate on any non-foot contact | stood still (air time 0.000 at it 200) |
| v7 | + stale-contact fix, harness curriculum | **first valid biped result** (above) |

**v4 walked on a tripod.** Its headline numbers (100% survival, ~97% tracking) were
real but meaningless: the tail was a permanent third contact point. Sachin spotted
it in the rendered frames. My task only terminated on head height and tilt, so
nothing prevented it; a zero-action trace ends propped on the tail at ~43 deg tilt,
under the old 55 deg limit, and re-evaluating v4 under "any non-foot contact = fall"
gives **0% survival** at every command. The same investigation found the height-based
foot contact was wrong (both feet carrying 6-7 N read "no contact"), so the gait
terms were fed a bad signal too.

Lessons: tracking metrics and 100% survival said "walks"; only looking at frames and
checking *which bodies touch the ground* exposed the cheat. Any rule implicit in the
user's intent (feet only on the ground) must be an explicit termination, or RL will
find the cheat. And when a metric looks too good, check what the robot is touching.

## Other findings

1. **The neutral pose is not statically stable.** With all targets at 0 the robot falls
   in ~1.2 s. Balance is learned, not inherited. The placeholder 1.0 kg base mass
   makes the tibia servo saturate just to stand under a PD-only ankle, so the real
   base-mass reading matters.
2. **Throughput**: ~35k env-steps/s for the full env at 25 Hz control (8 substeps),
   ~3 s per 24-step PPO iteration at 4096 envs. Needs Fabric disabled (5x), GPU
   aggregate-pair buffers sized to the env count (else PhysX silently misses
   contacts), and raw-view calls in the substep hot path. Timing varies up to 3x with
   GPU power state (battery) and other load on the machine.
3. **Rendering**: with GPU physics, transforms reach Fabric but not the USD stage
   (readback is suppressed), and the headless renderer draws from USD, so the robot
   looks frozen while physics runs. Render with `device="cpu"`, one env, a static wide
   camera (focalLength 18). The policy transfers to CPU physics. The RTX renderer also
   runs out of GPU memory next to a 4096-env training run.
4. **Standing local optimum**: runs pass through an upright "stand still" phase around
   iterations 100-250 because standing already earns about half the tracking reward at
   the mean command (~0.08 m/s). A higher minimum walking command or a direct
   forward-progress term are the first things to try if a run stays stuck.
5. Kit's fast shutdown swallows Python tracebacks; scripts print them explicitly.

## Next steps

1. Sim-to-real is the big unknown. Get the real `base_link` mass (scale reading) and
   servo/IMU noise to replace the estimated DR ranges, then try `best_numpy.npz` at
   25 Hz with the same 0.24 deg quantization and alpha=0.5 command low-pass (all in
   `walk_task.yaml`).
2. Speed tracking at 0.15-0.20 m/s and heading drift at 0.20 m/s: more training with a
   wider command range, a tighter yaw sigma, or a forward-progress term.
3. Standing jitter and lateral sway: a mild standing-pose penalty when the command is
   0; check the sway amplitude against what the real hips can do (+/-30 deg,
   servo saturation 1.5-3%).
4. Robustness: push disturbances, uneven ground, observation delay, per-episode
   mass/CoM variation; always re-evaluate with the harness OFF.
5. Review the gait at more speeds (only one render has been inspected).

## Known gaps

- Foot/body contact (rewards, termination, critic) is privileged; the real robot has no
  contact sensing, so the policy must infer ground contact from IMU + joint readings.
- DR ranges and sensor noise are estimates, not measurements.
- `base_link` mass is the 1.0 kg placeholder; joint limits are a working assumption.
- Straight-line walking only (hips are abduction joints, no yaw command).
- Kinematic ankle enforcement is not energy-conserving (see Ankle model).
- Not validated on hardware.
