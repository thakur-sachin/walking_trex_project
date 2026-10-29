# Joint Sign Conventions

Experimentally determined (not assumed from the URDF `<axis>` tags alone) by `scripts/joint_sign_convention_experiment.py`: each joint teleported to +20.0 deg from a neutral (all-zero) pose, with the resulting world-space displacement of its child link and the relevant foot measured and decomposed into the robot's own forward/right/up axes. Pure kinematics (teleported, not PD-driven), so these are sign-convention facts, not control-tuning results.

## Global / body frame convention

- **Up** = world `+Z` (standard USD/Isaac Sim up axis for this stage).
- **Forward** = direction from `tail` to `head` at the neutral pose, projected onto the horizontal plane. In world coordinates at spawn (identity orientation): `[0.9999609150207881, -0.008841291239874592, 0.0]`.
- **Right** = `forward × up` (right-hand rule). In world coordinates at spawn: `[-0.008841291239874592, -0.9999609150207881, 0.0]`.
- computed 'right' axis points toward the right_femur link (consistent).
- At spawn the robot's own orientation is identity, so world axes and robot-body axes coincide -- the forward/right/up vectors above double as both.

## Per-joint effect of a positive angle

`child_link` is the link the joint directly rotates; `foot_delta` is the resulting motion of the relevant end-effector (the foot, for every leg joint) -- useful since several joints only move the foot indirectly through the kinematic chain. All deltas in cm, body frame (+forward/-back, +right/-left, +up/-down).

| Joint | Child link | Child delta (fwd, right, up) | Foot delta (fwd, right, up) |
|---|---|---|---|
| `head_joint` | `head` | (-0.1, +0.0, -2.1) | (-0.1, +0.0, -2.1) |
| `tail_joint` | `tail` | (+0.2, -0.0, -1.8) | (+0.2, -0.0, -1.8) |
| `left_hip_joint` | `left_femur` | (+0.3, -0.0, -2.3) | (-0.0, +6.3, -1.1) |
| `right_hip_joint` | `right_femur` | (+0.3, -0.3, -1.9) | (-0.5, -6.5, -1.0) |
| `left_femur_joint` | `left_tibia` | (+0.3, +0.1, -1.8) | (+3.7, +0.2, -1.4) |
| `right_femur_joint` | `right_tibia` | (+0.8, -0.2, -1.7) | (+4.1, -0.3, -1.4) |
| `left_tibia_joint` | `left_ankle` | (-0.7, +0.3, -1.2) | (-2.6, +0.3, -0.8) |
| `right_tibia_joint` | `right_ankle` | (-1.2, -0.3, -1.3) | (-3.2, -0.3, -0.7) |
| `left_ankle_joint` | `left_foot` | (-1.6, +0.2, -0.8) | (-1.6, +0.2, -0.8) |
| `right_ankle_joint` | `right_foot` | (-1.5, -0.2, -0.7) | (-1.5, -0.2, -0.7) |

## Left vs right symmetry

- `left_hip_joint` vs `right_hip_joint`: left foot moves **inward** (6.3cm), right foot moves **inward** (6.5cm) -- opposite raw sign (mirrored as a raw number)
- `left_femur_joint` vs `right_femur_joint`: no significant left/right (sideways) component for either side at this test angle
- `left_tibia_joint` vs `right_tibia_joint`: left foot moves **inward** (0.3cm), right foot moves **inward** (0.3cm) -- opposite raw sign (mirrored as a raw number)
- `left_ankle_joint` vs `right_ankle_joint`: no significant left/right (sideways) component for either side at this test angle

## Screenshots

One per joint, all from the same fixed camera (`outputs/joint_experiment/`, gitignored):

- `head_joint`: `outputs/joint_experiment/head_joint.png`
- `tail_joint`: `outputs/joint_experiment/tail_joint.png`
- `left_hip_joint`: `outputs/joint_experiment/left_hip_joint.png`
- `right_hip_joint`: `outputs/joint_experiment/right_hip_joint.png`
- `left_femur_joint`: `outputs/joint_experiment/left_femur_joint.png`
- `right_femur_joint`: `outputs/joint_experiment/right_femur_joint.png`
- `left_tibia_joint`: `outputs/joint_experiment/left_tibia_joint.png`
- `right_tibia_joint`: `outputs/joint_experiment/right_tibia_joint.png`
- `left_ankle_joint`: `outputs/joint_experiment/left_ankle_joint.png`
- `right_ankle_joint`: `outputs/joint_experiment/right_ankle_joint.png`
