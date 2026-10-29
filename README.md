# Walking Trex Project

Clean, consolidated home for the Trex bipedal (T-Rex style) robot RL project:
the robot description, the Isaac Sim pipeline that loads it with real hardware
parameters, and a walking RL task grounded in what the real robot can sense and do.
It consolidates earlier experiments (trex-sim, RexAI and an SB3 prototype) into one repo.

## Layout

```
assets/robots/trex/       Canonical robot asset: Trex.urdf + STL meshes
                           (10 DoF: head, tail, L/R hip-femur-tibia-ankle)
  robot_config.yaml          Swappable real-robot params: mass, joint limits,
                              servo spec, which joints are actuated, ankle
                              linkage (see "Robot" below)

common/
  robot_config.py            Loader for robot_config.yaml -- read this, not
                              hardcoded numbers, wherever drive gains, limits
                              or mass are needed

walk_rl/                  Walking RL task (Isaac Sim 6.0.1, PhysX GPU, no Isaac Lab)
  env.py                     Vectorized env: servo model, sensor model, rewards
  ppo.py / train.py          In-repo asymmetric PPO + training entry point
  evaluate.py                Fixed-command evaluation sweep
  render_gait.py             Render a short gait capture (CPU physics)
  smoke_env.py               Env sanity check
  tb_sync.py, start_tensorboard.sh   Live TensorBoard from the training logs
  walk_task.yaml             Task / reward / randomization / PPO config

docs/
  WALK_TASK.md               Walking task: interface, design, experiments,
                              results, bugs found, next steps
  JOINT_CONVENTIONS.md       Experimentally measured sign convention per joint

scripts/                  Standalone Isaac Sim utilities
  test_load_robot.py         Main pipeline: URDF -> USD -> config -> physics -> screenshot
  dump_robot_prims.py        Audit every prim under the imported robot
  joint_sign_convention_experiment.py   Regenerates docs/JOINT_CONVENTIONS.md
```

## Environment

Python env: conda `isaac6` (Python 3.12, Isaac Sim 6.0.1.0 pip install, torch 2.11.0+cu128).
The NVIDIA driver must be 550.90.07+ (the RTX renderer hard-crashes older drivers).

```bash
conda activate isaac6
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/test_load_robot.py                  # load the robot, screenshot to outputs/
python walk_rl/train.py --name run0 --iterations 1000
python walk_rl/evaluate.py --name run0 --ckpt best.pt
walk_rl/start_tensorboard.sh                       # http://localhost:6006
```

## Robot

Bipedal, dinosaur-styled, 3D-printed (PLA) robot. **8 real actuated DOF**,
Hiwonder LX-16A serial bus servos (datasheet:
https://cdn.compacttool.ru/downloads/LX16A.pdf): 17 kg·cm (1.667 N·m) stall
torque @ 7.4 V, ~375°/s no-load speed, 0.24° position accuracy, position/
temperature/voltage feedback over a 115200-baud serial bus.

- Actuated joints (8): `head_joint`, `tail_joint`, `left_hip_joint`,
  `right_hip_joint`, `left_femur_joint`, `right_femur_joint`,
  `left_tibia_joint`, `right_tibia_joint`.
- Links (11): `base_link`, `head`, `tail`, `left_femur`, `right_femur`,
  `left_tibia`, `right_tibia`, `left_ankle`, `right_ankle`, `left_foot`,
  `right_foot` (renamed from the Fusion2URDF export's auto-generated names).
  Each link's imported USD has plain `visual`/`collision` mesh children,
  renamed automatically by `scripts/test_load_robot.py`
  (`simplify_visual_collision_naming`).
- **Ankle is not a servo.** It is a passive joint set mechanically by a double
  4-bar parallelogram linkage that keeps the foot parallel to `base_link`
  (lower parallelogram 50 mm x 54 mm, upper 50 mm x 65 mm, confirmed
  2026-10-09; the idler links are not in the URDF). **Confirmed + implemented:**
  `ankle_joint_angle = femur_joint_angle - tibia_joint_angle`. Note this is NOT
  `-(femur+tibia)`: femur_joint's URDF axis is opposite in sign to tibia_joint/
  ankle_joint, so a naive same-sign sum is off by ~44°
  (`common/robot_config.py: ankle_angle_from_linkage()` has the derivation).
  It is enforced kinematically, not by a native PhysX constraint (PhysX mimic
  joints take a single reference joint, this needs two terms), so any loop that
  steps physics must re-apply it every step. The ankle drive uses the same gains
  as the leg joints ("passive" means no servo, not "no drive").
- Sensors: BNO055 IMU in `base_link` + per-servo position feedback (8 channels).
  No foot contact sensing. No camera (a RealSense D435 is possible later).
- Compute: Raspberry Pi 4 (8 GB), no GPU: any deployed policy must fit a Pi 4's
  CPU budget (the trained actor exports as plain NumPy weights).
- Mass, joint limits (±45°, hips ±30°), the servo spec and the visual
  materials live in `assets/robots/trex/robot_config.yaml`, not in Python.
  `base_link` mass is still a 1.0 kg placeholder pending a scale reading, and
  the joint limits are a working assumption.

## Joint sign conventions

What a positive angle on each joint does, in robot-relative terms
(forward/back, left/right, up/down), measured experimentally (pure kinematics,
not PD-driven), not assumed from the URDF `<axis>` tags. Full table and
screenshots: `docs/JOINT_CONVENTIONS.md`; regenerate with
`scripts/joint_sign_convention_experiment.py`.

**Key finding:** the hip joints are mirrored in raw sign but **both drive the
foot inward** on a positive angle, so a positive value on *either* hip swings
that leg toward the centerline. Femur/tibia are mostly sagittal-plane. Do not
assume a shared positive direction across left/right joints when writing a
symmetric gait.

## Walking RL task

`walk_rl/` trains a forward-velocity-tracking policy with an 8-DOF action space
(the actuated joints; the ankles are derived) and an actor that only sees what
the real robot can measure: IMU gravity + gyro, quantized servo positions,
its previous action, the command and a gait clock. The critic additionally
sees privileged simulator state during training. Servo behaviour (torque cap,
0.24° quantization, speed limit, bus latency) comes from `robot_config.yaml`.

Rule enforced in the task: **only the feet may touch the ground**. The head and
tail exist to help balance; any other link on the ground ends the episode.
(An earlier policy was found to be propping itself on its tail; the details and
the other bugs found on the way are in `docs/WALK_TASK.md`.)

Latest validated result (v7, unassisted, domain randomization on): 98-100%
survival at 0-0.2 m/s commands, ~97% speed tracking at 0.10 m/s and ~84% at
0.20 m/s, symmetric alternating stance, tail held up as a counterweight. See
`docs/WALK_TASK.md` for the table, weak spots and next steps. **Not yet run on
hardware**, and the robot mass/noise parameters behind it are estimates.

## Gotchas (cost real time; details in script docstrings and `robot_config.yaml`)

- Raw URDF import with untuned joint drives is unstable (explodes in 1-2
  steps): use the full `URDFImporter` with `override_joint_stiffness/damping`,
  not the bare `urdf_usd_converter.Converter`.
- The imported USD's content sits behind a `payload` arc needing an explicit
  `Load()`; `Articulation()` needs `reset_xform_op_properties=True` on first wrap.
- The mesh geometry is authored at its natural standing height (root at Z=0
  puts the feet near Z≈0); use only a small spawn clearance.
- Mesh geometry lives behind instance proxies: bind materials on the nearest
  non-proxy ancestor with `bindingStrength=strongerThanDescendants`.
- A joint's URDF `<axis>` sign is not guaranteed consistent with its neighbours
  even when they share a rotation direction. Check the signed axis vectors.
- `omni.timeline`'s `stop()` resets the whole articulation to its initial pose.
  Use `pause()` to freeze a pose for inspection or screenshots.
- GPU physics keeps transforms out of the USD stage (Fabric only), so the
  headless renderer shows a frozen robot: render with CPU physics.
- PhysX contact views keep their last value after a teleport; ignore a reading
  until it changes (handled in `walk_rl/env.py`).
- A foot-link height threshold is not contact: the linkage keeps the feet
  parallel to the base, so body lean tilts them. Use contact forces.
- Kit's fast shutdown swallows Python tracebacks on exit: scripts print them
  explicitly before closing the app.
- Live-stage access: run `scripts/test_load_robot.py --remote` to enable
  `isaacsim.code_editor.python_server` (127.0.0.1:8226) for the
  `isaac-sim-remote` skill.
