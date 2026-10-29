# Trex mechanical redesign: log

The running record of the redesign effort (brief: [REDESIGN_BRIEF.md](REDESIGN_BRIEF.md)), in the style of [PROJECT_HISTORY.md](PROJECT_HISTORY.md). The proposal that came out of it is [REDESIGN_CONCEPT.md](REDESIGN_CONCEPT.md).

Every number below is labelled by where it comes from:
- **measured**: on the real robot, from earlier work (PROJECT_HISTORY);
- **CAD**: computed from the URDF/meshes exported from Fusion;
- **model**: from the MuJoCo model (URDF + `robot_config.yaml`, fitted LX-16A servo);
- **estimate / assumption**: stated as such, with the reason.

| Period | Phase | Branch |
|---|---|---|
| 2026-10-19 | Leg 1: audit, CPU sensitivity studies, concept | `redesign` |
| 2026-10-19 | Leg 2: the as-built CAD (Fusion Main v24) | `redesign` |
| 2026-10-19 | Leg 3: trex_v2 simulation asset, first-pass CAD, checks (after Sachin's go-ahead) | `redesign` (merged `pi-bringup` 8c2e1cb) |

---

## 1. Setup (2026-10-19)

- Worktree `~/walking_trex_redesign`, branch `redesign` from `d05b2e2`. Nothing was written to the main trees; the v9c policy files (`policy_numpy.npz`, `walk_task.yaml`, its sim2sim log) were **copied** into this tree's `outputs/walk_runs/v9c/` to run it.
- **CPU only.** New conda env `trex_cad` (Python 3.12; mujoco 3.8.0, the same version as `isaaclab6`; trimesh 5.1.1, shapely 2.2, scipy 1.18, matplotlib 3.11, manifold3d). `isaac6` and `isaaclab6` were not touched.
- Media: the 3 photos show the electronics stack only; frames were extracted from `IMG_8708.MOV` (33 s, robot held and turned by hand) with the ffmpeg binary bundled in `isaaclab6` (read-only use), to `outputs/media_frames/`.

### Baseline reproduced before varying anything
| check | result |
|---|---|
| `pi_runner/sim2sim.py --run v9c --actuator lx16a`, friction 0.5 / 0.3, delay 0, cmd 0.20 / 0 | vx 0.133 / 0.110 m/s, 4/4 survive: **identical** to the main effort's log |
| `redesign/policy_probe.py` (my instrumented copy of the same loop), 60 s, friction 0.5, cmd 0.20 | vx 0.13302670705329042 vs 0.13302670705329048 (summation order) |
| `redesign/variant_model.py` with an empty variant vs `pi_runner.mujoco_backend.build_model` | every compiled model array identical; 3 s physics traces identical (PD and LX-16A models) |
| statics on the model | total mass 1.778 kg (= KNOWLEDGE_BASE); vertical foot forces sum to m·g to 1e-13; a pure pitch moment on a foot gives **0** servo torque and 1.0 on the linkage (the parallelogram routing, as `sim_model: teleport_routed` assumes) |

## 2. Audit of the current design

### 2.1 URDF inertials vs the meshes (CAD)
Each link mesh, placed by its URDF visual origin and given the density m/V, reproduces the URDF centre of mass to ≤ 0.05 mm and the inertia diagonal to ≤ 0.1 %. So the URDF inertials are exactly Fusion's solid bodies:

| link | URDF mass | mesh volume | implied density | contents (meshes + video) |
|---|---|---|---|---|
| base_link | 2595 g | 330.6 cm³ | **7.85 g/cm³ (steel)** | plate, arched shell, 4 servo bodies (2 hip, head, tail), mounts |
| head | 231 g | 218.2 cm³ | 1.06 | printed head |
| tail | 115 g | 108.0 cm³ | 1.06 | printed tail |
| femur (×2) | 67 g | 63.0 cm³ | 1.06 | hip bracket (U-fork on the hip horn) + femur servo body |
| tibia (×2) | 54 g | 50.5 cm³ | 1.06 | tibia servo body + thigh structure (65 mm span) |
| ankle (×2) | 24 g | 22.6 cm³ | 1.06 | shank (54 mm span) |
| foot (×2) | 72 g | 67.6 cm³ | 1.06 | three-toed foot with the idler lever at the heel |

- 1.06 g/cm³ is a Fusion default plastic. **Every URDF mass is a solid-volume figure, not a printed or weighed one**: printed PLA is 1.24 g/cm³ with walls and infill, and a servo modelled as a solid block at 1.06 g/cm³ weighs ~43 g instead of the real 52 g.
- `left_femur.stl` has non-manifold edges (not watertight); its volume still matches the URDF.
- The simulator uses base_link = 1.0 kg (estimate) with the composite inertia; total **1.778 kg (model)**.

### 2.2 Bottom-up mass estimate (estimate)
`redesign/mass_estimates.yaml` + `redesign/audit.py`: servo count per link from the meshes and video, 52 g + 4 g per servo, printed volume = mesh volume − servo blocks, PLA at an assumed fill of 0.6–1.0 (thin parts) or 0.25–0.6 (head, tail), and the torso contents seen in the photos/video.

| | estimate [low, high] | model |
|---|---|---|
| femur link | 73–84 g | 67 g |
| tibia link | 63–68 g | 54 g |
| ankle link | 17–28 g | 24 g |
| foot | 50–84 g | 72 g |
| head | 68–162 g | 231 g |
| tail | 33–80 g | 115 g |
| torso (base_link) | 485–882 g | 1000 g |
| **total** | **0.99–1.65 kg** | **1.78 kg** |

Reading: the legs are probably *heavier* than modelled (real servos) and the head, tail and torso probably *lighter*. The range is wide because nothing has been weighed; this is the first item of the data request.

### 2.3 What the video shows (`IMG_8708.MOV`, frames in `outputs/media_frames/`)
- The torso is a **flat white plate**. The two hip servos hang under it; the head and tail servos sit on its front and rear ends.
- The electronics are a stack **on top of** the plate: Pi 4, two acrylic plates on standoffs, the BNO055, a large red DC-DC converter board with a toroid, a USB Wi-Fi dongle with a long antenna. The battery is not visible (glued under the plate, per the brief).
- **The arched shell is not fitted in the video.**
- Legs are black prints; the parallelogram idler bars are thin black bars and white "bone" links. Feet are blue, three-toed.
- The robot is filmed on a rug; the project's floor of record is indoor wood.

### 2.4 Geometry (CAD)
| | value |
|---|---|
| hip roll axes | y = +70.8 / −76.6 mm, spacing 147.4 mm; 198.2 mm above the soles. **Corrected in section 7:** the hips are symmetric about the chassis; the URDF base origin sits 2.9 mm off the centre line |
| hip axis → femur (pitch) axis | 55.0 mm down, 18.5 mm forward, 18.2 mm outward |
| thigh span (femur → tibia axis) | 65.0 mm |
| shank span (tibia → ankle axis) | 54.0 mm |
| ankle axis → sole | 24.2 mm; the foot steps back inward by 18.75 mm, so each sole centre sits under its hip axis |
| sole patch (section 0.5 mm above the lowest point) | 45.8 cm² (convex hull 60.4 cm²), x −37…+79 mm about the ankle axis, y ±42.4 mm; flat to ~0.3 mm |
| feet | sole centres 147.4 mm apart (half-width 73.7 mm); 63 mm gap between the inner toe tips |
| head / tail axes | +75 / −94 mm from the base origin, ~36 mm above the hip axes |
| centre of mass, zero pose (model) | (+6.1, −3.2, 209.5) mm; 202 mm above the soles. v9c standing pose: 189 mm; walking pose: 201 mm |
| double-support margin (model) | 47–52 mm from the support polygon edge in the three poses |

Note (cosmetic): `robot_config.yaml ankle_linkage` lists the 65 mm parallelogram as `[tibia_joint, ankle_joint]`; in the URDF the 65 mm span is femur_joint→tibia_joint and the 54 mm span is tibia_joint→ankle_joint. The dimensions are right, the joint labels are shifted by one.

### 2.5 Geometric joint ranges from the meshes (CAD)
`redesign/range_check.py`: sweep each joint, count sampled surface points of one link inside the other (1 mm voxel grids) beyond the zero-pose overlap; first angle with > 20 new points. Idler links, horns, screws and cables are not in the meshes.

| joint / pair | first contact | model range |
|---|---|---|
| head vs torso | ±74° | ±45° |
| tail vs torso | −52° (tip down) / +86° | ±45° |
| **hip inward (+) vs torso** | **8–9°, both sides** (sharp onset: 0 → 27 → 57 points at 8/9/10°). As-built CAD (section 7): **10°** | +30° |
| hip outward (−) | none to −40° | −30° |
| both hips inward together | the feet touch at ~10° | — |
| femur (thigh vs hip bracket) | −62° / +102° | ±45° |
| shank angle (femur − tibia): thigh link vs foot | −85° / +70° | not limited; the ±45° box allows ±90° |
| **(missed in leg 1)** shank vs foot, lower idler vs foot | **not checked in leg 1**; as-built CAD (section 7): **+39° / −57°** | — |
| tibia: thigh vs shank | none to ±120° | ±45° |

- The hip contact is the **top of the femur servo body hitting the bottom of the hip servo body** (y 45–55 mm, z 193–195 mm).
- **Corroboration (measured):** in stage 1b the right hip stopped 2.3–2.8° short of a +10° (inward) goal, i.e. at ~7.2–7.7°; PROJECT_HISTORY lists it as possible drag. The CAD puts the first contact at 8–9°; screws, horns or cables, which are not in the meshes, would make contact slightly earlier. This is the likely explanation, to be confirmed by hand on the robot.
- Servo ticks (from `hardware.yaml`) leave the tibias room to about −82° (right) / −93° (left) and the left femur only to −48°. So a wider negative tibia range needs **no new servo travel**; whether the idler links allow it is unknown.
- **Correction (leg 2):** the leg-1 sweep never paired the shank with the foot. With the as-built parts, that pair and the lower idler set the ankle range (femur − tibia) to −57…+39°, so the negative tibia range is **not** free (section 7.4).

### 2.6 The parallelogram limits (CAD + assumption)
- Upper parallelogram angle = femur, lower = femur − tibia (the ankle joint angle). With the 50 mm idler offset perpendicular at the zero pose, each goes flat (singular) at 90°, and the bar force for a given foot moment grows as 1/cos(angle).
- **The current ±45°/±45° joint box contains |femur − tibia| = 90°**, a flat lower parallelogram the real bars cannot reach. The simulators do not model the idlers, so nothing stops a policy there. v9c stays inside ±40° (walking: femur −27…−9, tibia −44…−24), so this is a latent mismatch, not a current one.

### 2.7 Leg kinematics (model)
`redesign/kinematics.py`, sole-centre reach in the sagittal plane, torso level, usable parallelogram angle ≤ 60° (**assumption**: bar force ×2):

| joint ranges | body heights possible with the foot under its neutral spot | foot lift at mid-swing from knee bend, body at 190 / 195 mm | stride range at 190 mm |
|---|---|---|---|
| current ±45 / ±45 | 189–198 mm (**9 mm of crouch**) | **1 / 6 mm** | −12…+82 mm |
| tibia −70…+45 | 177–198 mm | 13 / 18 mm | same |
| tibia ±60, femur ±50 | 182–198 mm | 8 / 13 mm | same |

- The knee bend is exactly |tibia|. v9c walks at ~192 mm hip-to-sole with the knee bent ~36° backwards (femur −18, tibia −36). Lifting the swing foot at mid-swing needs more knee bend, and −45° allows almost none.
- **Policy evidence (model, v9c in MuJoCo):** the swing foot does clear the floor (mean 13 mm, p90 21 mm at 0.20 m/s), but the left−right foot height difference correlates at −0.98 with 2·half-width·sin(torso roll), slope 1.2. **About 80 % of the swing clearance comes from rolling the torso**, not from the leg. Rolling is how the policy lifts a foot with this knee range, which ties the tibia limit to the 13–17° roll p2p.
- Stride is not the limit: at the walking height the foot can reach −12…+82 mm, while v9c takes ~30 mm steps.

### 2.8 The simulated standing pose rests on joint stops the robot does not have (model)
Standing (cmd 0), v9c drives head and tibias to +45°. In MuJoCo the joint-limit constraint then carries the load: limit torque / stall = head 0.10, tibias 0.06 / 0.15, while those servos carry 0.005–0.02. On the robot ±45° is only the servo's EEPROM clamp, so the servos hold it. The loads are small (≤ 0.15 stall), so this is a modelling gap, not a danger. Walking: no limit torque. (Reported for the main effort; for the redesign it suggests printed hard stops, see the concept.)

### 2.9 Mass that need not be there
Head + tail are 346 g in the model (19 % of the total) and are modelled solid. The arched shell (121 cm³ above the plate) is not fitted in the video. On the torso, the long Wi-Fi antenna, the two-level acrylic stack and the large DC-DC board are candidates for consolidation. All of these await weights.

## 3. Quasi-static studies (model)

### 3.1 Method (`redesign/statics.py`, `redesign/levers.py`, sweeps in `redesign/studies.yaml`)
- Rigid bodies, torso level, joints still. Generalized forces from `mj_inverse` (gravity plus a d'Alembert force on the root) = servo torques + linkage force (ankle − femur + tibia) + foot wrenches.
- **Single support:** the system is square. The body acceleration is chosen so the centre of pressure sits at the sole centre: the ground force passes through the centre of mass, as in an inverted pendulum (LIPM). With the centre of mass at the midline this is the **step switch**, the worst instant of a periodic LIPM gait.
- **Mid-stance:** scaled by 1/cosh(ωT/2) = 0.75 at the v9c gait clock of 2.2 Hz (ω = √(g/h) = 7.0 rad/s).
- **Double support:** minimum servo torque.
- Torque is a fraction of stall at the measured pack voltage (1.60 N·m).

### 3.2 Baseline (model)
| quantity | value |
|---|---|
| stance hip, step switch / mid-stance | **0.72 / 0.54 of stall** (left 0.716, right 0.711; the URDF hip offsets are a base-origin offset, see 7.2) |
| friction the step switch needs (\|F_xy\|/F_z) | **0.37**: the same as the estimated PLA-on-wood μ of 0.3–0.4 |
| stance femur / tibia, walking pose, CoP at the CoM ±20 mm | 0.14 / 0.23 |
| standing (v9c pose), worst leg servo | 0.10 |
| LIPM centre-of-mass sway, and the torso roll that produces it | 18 mm, 5.2° half amplitude (v9c MuJoCo: 13–15° p2p) |
| single support, static: centre of mass to the stance sole | **39 mm outside**: standing on one foot needs an 11° lean, so the gait must be dynamic |

### 3.3 One-at-a-time levers (model; full table in `outputs/redesign/levers.txt`)
| lever (change) | stance hip at switch | μ needed | stance tibia | roll (LIPM) | other |
|---|---|---|---|---|---|
| baseline | 0.716 | 0.366 | 0.229 | 5.2° | static single-support margin −39 mm |
| **hip spacing −30 mm** | **0.426 (−41 %)** | **0.217** | 0.229 | 3.1° | margin −11 mm; the soles would touch (gap 3 mm) unless the feet narrow |
| hip spacing −40 mm | 0.329 (−54 %) | 0.167 | 0.229 | 2.4° | the soles overlap (−17 mm) |
| hips ±44 mm symmetric, feet ×0.8 wide | 0.439 (−39 %) | 0.226 | 0.227 | 3.0° | sole gap 20 mm, margin −20 mm |
| torso 1.0 → 0.8 kg | 0.643 (−10 %) | 0.374 | 0.203 | 5.4° | |
| torso 1.0 → 0.6 kg | 0.571 (−20 %) | 0.385 | 0.177 | 5.7° | |
| head + tail ×0.5 | 0.660 (−8 %) | 0.378 | 0.207 | 5.5° | −173 g |
| head + tail removed | 0.605 (−16 %) | 0.394 | 0.185 | 5.9° | upper bound of that lever |
| torso CoM −30 mm (lower) | 0.781 (+9 %) | 0.400 | 0.234 | 6.1° | **lowering the CoM hurts** here |
| torso CoM +30 mm (higher) | 0.661 (−8 %) | 0.338 | 0.225 | 4.5° | |
| torso CoM x ±20 mm | 0.716 | 0.37 | 0.229 | 5.2° | no effect |
| head/tail length ×0.7 / ×1.3 | 0.711 / 0.721 | 0.36 | 0.23 | 5.3° | negligible |
| thigh 65 → 75 / 85 mm | 0.722 / 0.728 | 0.35 / 0.34 | 0.227 | 4.9° | stance femur 0.14 → 0.18 / 0.21; swing-foot effective mass 1.55 → 1.29 / 1.15 kg; swing-foot speed 0.48 → 0.50 / 0.53 m/s |
| shank 54 → 64 mm | 0.722 | 0.35 | 0.269 | 4.8° | foot speed 0.53 m/s; stance force capacity 94 → 79 N |
| leg link mass ±30 % | 0.671 / 0.761 | 0.36 / 0.37 | 0.22 / 0.24 | 5.0° | through total mass only |
| foot length ×0.8 / ×1.2 | 0.70 / 0.73 | 0.36 / 0.37 | 0.23 | 5.1° | through foot mass only (statics cannot see CoP range) |
| stall torque ×0.75 (the low end of the randomization) | 0.955 | | 0.306 | | the hip is at stall at the switch |

Reading:
- **Stance width is the dominant lever for the hips.** The stance-hip torque is ≈ m·g·(lateral CoM-to-foot distance)·(hip height / CoM height), and that distance is the half-width at the step switch. It is linear: −10 mm of half-width gives −13 % hip torque. The same lever cuts the friction the gait needs and the lateral sway.
- **Mass** acts on every servo proportionally (−10 % torso mass ≈ −4 to −6 % on each joint, because the legs stay).
- **CoM height:** lowering it does not help in this model. Higher h means a smaller lateral force for the same offset and a slower lateral pendulum.

### 3.4 Do the static scalings hold under a real gait? (model, v9c in MuJoCo)
`redesign/policy_sweep.py`: v9c's deployed loop on design variants, fitted LX-16A servo, 30 s episodes (4 s warm-up excluded), commands 0 / 0.15 / 0.20 / 0.25, bus delay 0/1/2, friction 0.5 and 0.3. 432 episodes, 210 s on 8 CPU processes. Table: `outputs/redesign/policy_sweep.txt`.

| check | statics predicts | v9c gait in MuJoCo |
|---|---|---|
| torso 1.0 → 1.3 kg, hip load | ×1.15 | mean ×1.16 (0.37 → 0.43); time above 80 % stall 2.2 % → 4.2 % (1.5 kg: 9.7 %) |
| hips −20 mm (outside training), hip load | ×0.73 | mean ×0.76 (0.37 → 0.28) |
| friction 0.5 → 0.3 | step switch needs 0.37 | speed −15 to −20 % at every command, foot slip in stance 0.035 → 0.063 m/s, no falls |

- The load scalings agree with statics to within a few %.
- The **performance** numbers (speed, roll, survival) do not rank designs here:
  - A heavier torso made v9c faster with less roll (more traction for a policy tuned at 1.0 kg).
  - Weaker servos (×0.8) did the same.
  - Narrower hips (outside training) made it slower with more roll.
  - Survival dips of 1–3 falls in 9 (0.67–0.89) appear in unrelated cells at friction 0.5 and follow no lever.
- These are the responses of one fixed policy, not properties of the designs. **Comparing walking performance needs a policy trained per design** (section 6 of the concept).

### 3.5 Where the servo torque goes while walking (model)
At 0.20 m/s, tick-averaged |torque|/stall:
- hip 0.35, femur 0.33, tibia 0.31;
- of which the torque that accelerates the rotor (armature × mean joint acceleration over the tick) is 0.17 / 0.26 / 0.24.

**The pitch servos spend most of their walking torque accelerating their own rotors.**

`redesign/swing.py` (one leg in the air, torso pinned, Pi-runner goal stream, fitted servo):

| variant | 40 mm step | 80 mm step |
|---|---|---|
| baseline | 0.20 s | 0.36 s |
| leg link mass ×0.7 / ×1.3 | 0.20 s | 0.36 s (**identical**) |
| thigh 75 or 85 mm (or shank 64 mm) | 0.20 s | 0.32 s |
| thigh 55 mm | 0.24 s | (cannot reach) |

This confirms the brief's point 1: **lighter links do not make joints faster**. Longer segments do, slightly, because each degree moves the foot further (40 ms tick resolution).

### 3.6 Open-loop tests (model, no policy; `redesign/openloop.py`)
- Every variant tested **stands open-loop for 10 s** on the fitted servo, both in the zero pose and in the v9c standing pose (double-support margin 47–52 mm, held by servo friction/deadband).
- No variant can lift a foot without first leaning: the lifted foot stays loaded and the body tips 3.6–7° toward it. The static single-support margin is negative for all (baseline −39 mm, narrow concept −14 to −20 mm).

## 4. Ranking of the design levers

Order by expected effect on walking (torque margin, speed, stability, robustness), with the evidence for each:

1. **Stance width** (hip axes closer to the midline, feet under them, feet narrower to keep a gap).
   - Stance-hip torque −39 to −41 % for −30 mm half-width (model, linear).
   - Friction needed 0.37 → 0.22. Lateral sway / roll −40 %.
   - The single-support lean needed falls from 11° to ~6°, which is what a slow (quasi-static) gait needs; v9c has no gait below 0.12 m/s.
   - Load scaling confirmed in the v9c gait.
   - Risks: leg/foot self-collision, the sim has no self-collision, and hip packaging.
2. **Knee (tibia) range**, re-indexed to about −70…+45° with a coupled parallelogram limit.
   - Mid-swing foot lift 1 → 13 mm at the walking height, crouch range 9 → 21 mm.
   - Removes the need to roll the torso for foot clearance (80 % of it today).
   - ~~The printed parts allow it~~ **Corrected in leg 2 (7.4):** the foot hits the shank at femur − tibia = +39°, so the backward-knee side is capped; the forward-knee side is free (see the revised ranking in 7.7).
   - May be testable on the current robot.
3. **Hip inward range** (8–9° by the merged meshes, 10° by the as-built CAD, +30° in the sim).
   - Needed for lateral foot placement and push recovery, and to make a narrow stance safe.
   - The CAD contact matches the stage-1b right-hip shortfall.
4. **Total mass** (head/tail hollow shells, consolidated electronics, no shell).
   - Every stance torque scales with it; −0.2 kg ≈ −6 to −10 % per joint.
   - But the real mass is unknown (bottom-up 0.99–1.65 kg vs 1.78 modelled), so the size of this lever is the most uncertain.
5. **Sole friction** (TPU sole or pad).
   - The step switch needs μ ≈ 0.37 now (0.22–0.24 with a narrow stance) against an estimated 0.3–0.4 for PLA on wood.
   - In the v9c gait, μ 0.3 vs 0.5 costs 15–20 % speed and doubles slip.
   - Cheap; parametric pads generated (`redesign/cad/sole_pad.py`).
6. **Leg segment length** (thigh +10–20 mm).
   - −11 % swing time for 80 mm steps, −17 to −26 % swing-foot effective mass, CoM +8 to +16 mm (slightly helps μ and the hips).
   - Costs: stance femur torque +26 to +50 % (static, from a low 0.14) and a taller robot.
   - Optional.
7. **CoM height:** do not lower it (+9 % hip torque, +9 % μ for −30 mm). Place mass for packaging.
8. **Not worth design effort:** CoM x position (±20 mm: no effect), head/tail length (±30 %: ±1 %), link mass for speed (no effect), foot length (statics blind; keep the current length).

## 5. Decisions and open items
| date | decision / finding | why |
|---|---|---|
| 10-19 | work in a separate env `trex_cad`, CPU only | brief; the GPU is training |
| 10-19 | studies run on a variant builder that is bit-identical to the simulator's model when empty | changes must be the only difference |
| 10-19 | rank levers on statics + policy-free tests; use the trained policy only for load scaling | a fixed policy's speed/roll response is not a design property (3.4) |
| 10-19 | no CAD before approval, except the sole pad (cheap, useful now) | brief |

**Open (needs Sachin's data):** real masses; real joint ranges (above all hip inward and tibia negative); idler link geometry and clearances; floor friction for PLA and TPU; printer and material; separated CAD.

**For the main effort (not redesign work, found on the way):** *(superseded by the as-built numbers in 7.8)*
- The hip inward range is probably ~8° on the robot, not 30°.
- The tibia range could be wider on the current robot if the idlers allow it.
- Standing in sim rests on joint stops the robot lacks.
- The ±45° box allows a flat parallelogram.

## 6. How to reproduce
```bash
PY=~/miniconda3/envs/trex_cad/bin/python
$PY redesign/variant_model.py          # self-check: empty variant == simulator model (arrays + trace)
$PY redesign/audit.py                  # 2.1, 2.2, 2.4      -> outputs/redesign/audit.json
$PY redesign/range_check.py            # 2.5                -> outputs/redesign/range_check.json (~30 s)
$PY redesign/kinematics.py             # 2.7                -> outputs/redesign/kinematics_baseline.json, workspace_baseline.png
$PY redesign/levers.py                 # 3.2, 3.3           -> outputs/redesign/levers.json (sweeps: redesign/studies.yaml)
$PY redesign/policy_sweep.py --procs 8 # 3.4 (needs outputs/walk_runs/v9c/) -> policy_sweep.json; summarize_sweep.py
$PY redesign/swing.py                  # 3.5
$PY redesign/openloop.py               # 3.6
$PY redesign/cad/sole_pad.py           # TPU sole pads      -> outputs/redesign/cad/*.stl
# leg 2 (as-built CAD; cad/incoming/ must exist, see cad/README.md)
$PY redesign/step_parts.py cad/incoming/final_version/Main_v24.step --tag main_v24 --cylinders   # 7.1 (~40 s)
$PY redesign/asbuilt.py                # 7.2, 7.3 (part/link map, joint residuals, pivots)
$PY redesign/asbuilt_ranges.py         # 7.4 (~3 min)
$PY redesign/asbuilt_stops_probe.py    # 7.4 v9c with the as-built stops
$PY redesign/asbuilt_mass.py           # 7.6
```

## 7. Leg 2: the as-built CAD (2026-10-19)

Sachin confirmed that the robot as built is Fusion project "T-Rex Final Version", design `Main` v24 (2024-04-01). Files: `cad/incoming/` (see `cad/README.md`).

### 7.1 Reading the STEP (`redesign/step_parts.py`)
- OCP (OpenCascade 7.8, pip `cadquery-ocp`) in `trex_cad`.
- `Main_v24.step`: 915 solids, read with names and occurrence transforms, in 36 s. For each solid: volume, centre of mass, inertia, cylindrical faces (pivots, holes) and a tessellated mesh. The meshes are watertight for every printed part except the feet; the mesh volumes equal the B-rep volumes to < 0.5 %.
- Contents:
  - New_Chassis (plate + arch in one body);
  - 4 torso brackets;
  - TalentCell battery;
  - velcro;
  - 8 LX-16A servo models, each with a Shaft and a Motor Adapter (the horns);
  - per leg: Bracket_hip, Bracket_Femur, Bracket_Tibia, Coupler_Femur (the 50 mm knee coupler, two plates on the left leg), Femur_Tibia_Coupler (upper idler), Coupler_Tibia_Feet (lower idler), Main_Trex_Feet;
  - Head v1, Tail v1;
  - 84 screws;
  - Pi 4, acrylic plate, BusLinker, a printed controller case, an LM2596 DC-DC module.
- **The electronics are not placed on the robot**: they float ~100–250 mm to its right. Some screws float with them.
- One duplicate foot solid.

### 7.2 Frame map and URDF joints vs as built (`redesign/asbuilt.py`)
- The STEP frame is +y forward, +z up, left = −x: a fixed 90° permutation of the URDF frame.
- The translation was fitted on the joint axes. Each servo axis comes from the line through its Shaft and Motor Adapter; each ankle axis from the shank's lower hole. Only offsets normal to an axis count, since the position along an axis is arbitrary.

| joint | as-built axis vs URDF: offset normal to the axis | axis angle |
|---|---|---|
| hips (both) | 0.30 mm | 0.022° (the URDF has a 0.000376 tilt) |
| femur, tibia, ankle | 0.96–1.08 mm: **±1.0 mm fore-aft, opposite on the two legs**, 0.29 mm vertical | 0° |
| right ankle | 1.42 mm (the right shank's lower hole is 1.3 mm higher, see 7.3) | 0° |
| head | 0.67 mm | 0° |
| tail | **1.72 mm (vertical)** | 0° |

- **The URDF matches the as-built joints to ≤ 1.7 mm and 0.02°. No URDF change is needed for the simulation.**
- The URDF puts the right leg 2.0 mm ahead of the left; the CAD has them level.
- The hip axes are 147.4 mm apart in both. In the CAD they are **symmetric about the chassis** (±73.7 mm). The URDF base_link origin just sits 2.9 mm off the chassis centre line, so the "5.7 mm left/right asymmetry" of section 2.4 is retracted.
- The "18.5 mm forward / 18.2 mm outward" offsets of the femur joint origins (2.4) lie along the joint axes and carry no geometry.

### 7.3 Which parts are the printed ones (shape signatures vs `STL Files/`; ICP vs the URDF meshes)

| STEP part | matches | notes |
|---|---|---|
| New_Chassis | URDF base_link mesh (ICP 0.02°, 0.2 mm, 0.10 mm mean residual) | no print STL of the chassis; it **includes the arch** (97 cm³ above z = 245 mm) |
| left leg: Bracket_hip, Bracket_Femur, Bracket_Tibia, couplers | print exports `1_v1`, `2_v1`, `3_v1`, `coupler_side`, `coupler_main`, `coupler_bottom_v1` (signature diff ≤ 0.05; hip bracket 1.7) | URDF links match: shank ICP 0.06°, 1 mm |
| **right leg brackets and couplers** | **no print export** (signature diff 1–19) | bulkier: femur bracket 16.3 vs 13.4 cm³, hip bracket 25.5 vs 25.0; the right shank's lower hole is 1.3 mm higher (52.7 vs 54.0 mm span) |
| Main_Trex_Feet (both) | `Feet.stl` and the URDF feet (ICP 0.00° / 0.11°, ~1 mm) | the sole is exactly the URDF sole |
| Torso Bracket | `Mounting_Bracket*.stl` | |
| **Head v1 / Tail v1** | **not** the printed parts: 782 / 382 cm³, 277 / 358 mm long | the print exports `head_updated` / `tail_updated` (218.7 / 108.5 cm³) **are** the URDF head and tail |

**Working assumption:** both legs were printed from the exported STLs, the right one mirrored, so the left leg of Main v24 stands for both. To confirm with Sachin.

### 7.4 The 4-bar and the joint ranges, as built (`redesign/asbuilt_ranges.py`)
**Pivots** (cylindrical faces; URDF frame x, z in mm, left leg):

| pivot | position |
|---|---|
| femur axis F | (−9.5, 150.4) |
| tibia axis T | (−9.5, 85.4) |
| ankle A | (−9.5, 31.4) |
| idler line | the same heights at x = −59.5 |

- **Idler offset 50.0 mm (behind the main line); spans 65.0 and 54.0 mm, exact.** The idlers are parallel to the main links at the zero pose.
- Assignment: the upper idler (Femur_Tibia_Coupler) runs from the hip bracket to the knee coupler; the knee coupler (Coupler_Femur) turns on the tibia axis; the lower idler (Coupler_Tibia_Feet) runs from the knee coupler to the foot's heel lever.

**Collision sweep with every real part:** hip bracket, thigh, both idlers, knee coupler, shank, foot, servos with horns, seated screws (52 of 84; the rest float by the electronics), and the torso. Method as in 2.5: 1 mm voxels, more than 20 new points.

| joint | as-built contact (left leg = print parts) | right-leg CAD variant | leg-1 merged meshes | sim |
|---|---|---|---|---|
| **hip inward (+)** | **+10°**, hip bracket vs the hip servo body (y 46–74 mm, z 192–214 mm); both sides | +10° | 8–9° | +30° |
| hip outward (−) | none to −60° | none to −60° | none to −40° | −30° |
| both hips inward together | **the feet touch at +10°** | | ~10° | |
| one hip inward | its foot reaches the other shank at +32° | | | |
| femur | −60° (hip bracket vs thigh) / +63° (thigh vs knee coupler, upper idler) | ±57° | −62 / +102° | ±45° |
| **ankle = femur − tibia** | **+39° (shank vs foot) / −57° (lower idler vs foot)** | +36° / −54° | not checked | box allows ±90° |
| tibia alone (thigh vs shank) | none within ±100° | **±57°** | none | ±45° |

- The tibia range at a given femur angle is therefore tibia ∈ [femur − 39°, femur + 57°] (left / print parts). At femur 0 that is −39…+57°; at the v9c walking femur (−18°) it is −57…+39°.
- **The backward-bending knee is capped by the foot.**
- **v9c walking already reaches it:** in MuJoCo, femur − tibia peaks at +38 to +41° at 0.15–0.25 m/s (above +39° for 1 % of the time at 0.15). Standing uses −19 to −28°, far from −57°.
- Adding the as-built stops to MuJoCo as hard limits (ankle −57…+39°, hip inward +10°; `redesign/asbuilt_stops_probe.py`): **v9c still walks**, 18/18 walking and 6/6 standing episodes, speeds unchanged to 0.002 m/s.
- On the robot the shank will touch the foot at the end of the backward-knee stance; the hip stays below +5°.

### 7.5 Kinematics with the as-built limits (`redesign/kinematics.py`, new range sets)

| joint limits | body heights with the foot at its neutral spot | mid-swing foot lift at 180 / 190 mm |
|---|---|---|
| config ±45 / ±45 (+ as-built contacts) | 189–198 mm | 0 / 1 mm |
| as-built contacts only (left/print legs) | 156–198 mm | 24 / 34 mm |
| **current robot, tibia −40…+90°, ankle −55…+37° (2° margin), femur −45…+55°** | **164–198 mm** | **16 / 26 mm** |
| current robot, tibia −40…+75°, same ankle limits | 174–198 mm | 6 / 16 mm |
| leg-1 concept: tibia −70…+45° (needs the foot to clear +60°) | 177–198 mm | 3 / 13 mm |

- The lift that the backward-bending knee cannot give is available by bending the knee **forwards** (the standing pose already does this), with no new parts.
- The tibia servo travel allows +147° / +157°; the left femur servo allows +192° and −48°.
- Bending forwards needs:
  - the tibia upper limit raised from +45° to +75…+90°;
  - a **coupled** limit −55 ≤ femur − tibia ≤ +37° in the action pipeline (simulator and Pi alike); per-servo EEPROM limits cannot express it;
  - a policy trained with it.
- If the right leg really is the Main v24 variant, its knee stops at ±57°, which caps this.

### 7.6 Mass budget from the real parts (`redesign/asbuilt_mass.py`, inputs `redesign/asbuilt_mass.yaml`)
**Method:**
- Printed mass = PLA density × (shell + infill × core). The shell is surface area × 1.2 mm (3 perimeters, assumed), capped at the part volume. **Infill is unknown**, so 15 %, 30 % and 100 % are shown.
- 52 g per servo.
- The horns are printed (assumed).
- Electronics from datasheet values and estimates; they are not placed in the CAD.
- Battery: the CAD "TalentCell Battery v1" envelope is 109.5 × 63 × 24 mm. The closest product listing found is a TalentCell 12 V 3000 mAh pack, ~105 × 64 × 24 mm, 180 g (retailer listing, unreliable), so 150–220 g is used.
- The 52 seated screws weigh 8.4 g.

| link | infill 15 % / 30 % / 100 % | URDF | leg-1 estimate |
|---|---|---|---|
| torso: printed + 4 servos only | 389 / 397 / 437 g | (1000, all-in) | |
| torso all-in (+ electronics 137–246 g + battery 150–220 g) | **0.68–0.86 kg** at 30 %; **0.58–0.76 kg without the arch** | 1.00 kg | 0.49–0.88 |
| head (`head_updated`) | 89 / 122 / 274 g | 231 | 68–162 |
| tail (`tail_updated`) | 50 / 66 / 137 g | 115 | 33–80 |
| femur link (hip bracket + femur servo + horns) | 76 / 78 / 86 g | 67 | 73–84 |
| tibia link (thigh + tibia servo + upper idler + horns) | 72 / 73 / 76 g | 54 | 63–68 |
| ankle link (shank + knee coupler + lower idler + horns) | 29 / 31 / 37 g | 24 | 17–28 |
| foot | 35 / 44 / 84 g | 72 | 50–84 |
| **robot total** | **1.33–1.51 kg at 30 % (1.23–1.41 without the arch)**; 1.25–1.43 at 15 %; 1.71–1.89 solid | **1.78 kg** | 0.99–1.65 |

Reading:
- The simulated robot is probably **0.3–0.5 kg too heavy**.
- The torso estimate (0.58–0.86 kg) sits at or below the bottom of the training randomization (×0.8–1.3 of 1.0 kg).
- The leg links are heavier than the URDF (+15 to +35 %, real servos plus idlers); the feet, head and tail lighter (≈0.5–0.6×).
- The battery is the largest single torso item and sits under the plate on the centre line. The arch, if printed, is ~100 g.
- **All of this is still an estimate: weighing remains the first data request.**
- Statics on this estimate (`studies.yaml: asbuilt_mass`, torso 0.67–0.77 kg; robot 1.31–1.41 kg): stance hip at the step switch is 0.55–0.59 of stall (model 0.72); friction needed 0.39 (lower CoM); stance tibia 0.17–0.18. The narrow-stance concept on the same basis: hip 0.34 (−39 %), μ 0.24.

### 7.7 What changes in the findings and the ranking
- **Unchanged, now confirmed on real geometry:**
  - stance width (147.4 mm, symmetric); stance width stays the top lever;
  - foot and sole geometry;
  - the parallelogram dimensions (50 / 65 / 54 mm);
  - the hip inward contact (+10° vs +30° in the sim);
  - the URDF joints.
- **Corrected:**
  - the knee: the backward-bending side is limited by the foot at femur − tibia = +39°, and the current gait already touches it. The forward-bending side is free to about tibia +100° (left/print parts). The knee-range lever becomes **"forward knee range on the current robot"**: config, a coupled limit and training, no new parts. The redesign only needs to move the heel/shank contact if the backward-knee gait is to be kept.
  - the 5.7 mm asymmetry: retracted.
  - masses: the model is probably 20–30 % heavy; the head/tail lever is smaller than thought (printed head/tail ≈ 0.5× the URDF).
- **Ranking:**
  1. stance width;
  2. knee range (now cheapest: current robot, forward-knee range);
  3. hip inward range;
  4. sole friction;
  5. total mass (smaller than in leg 1: the as-built estimate is already lighter);
  6. segment length;
  7. CoM height: unchanged (do not lower).

### 7.8 For the main walking effort (current robot)

| item | as-built CAD | recommendation (pending a hand measurement) |
|---|---|---|
| hip inward | contact at +10°, both sides; the feet touch at +10° when both hips roll in together; stage 1b stopped the right hip at ~7.2–7.7° | joint limit −30…**+7°**, EEPROM inward ≈ +8° (the usual +3° margin would sit on the contact). v9c uses ≤ +5° |
| hip outward | free to −60° | keep −30° (more is possible) |
| femur | −60 / +63° (right-leg CAD variant ±57°); the left femur servo travel is −48° | keep ±45° |
| ankle = femur − tibia | **+39° (foot vs shank) / −57° (lower idler vs foot)**; the ±45° box allows ±90° | a coupled limit −55 ≤ femur − tibia ≤ +37° in the action pipeline (sim and Pi identical). Not expressible as EEPROM limits. v9c reaches +38…+41° walking but still walks with the stops modelled |
| tibia | not limited by itself (left/print parts); right-leg CAD variant ±57° | the +45° limit can be raised to +75…+90° if the hand test agrees: 16–26 mm of mid-swing lift with a forward-knee gait (needs training) |
| masses | the model is probably 0.3–0.5 kg heavy; torso 0.58–0.86 kg | weigh before changing `robot_config`; consider widening the torso randomization downwards (to ~0.55 kg) |

## 8. Leg 3: the trex_v2 asset and the first-pass CAD (2026-10-19)

**Context:**
- Sachin gave the go-ahead. All measurements are parked; best approximations are used and flagged.
- New facts:
  - **Battery:** TalentCell YB1206000 (11.1 V nominal, 12.6–9 V, 6000 mAh, ~370 g, **3 A max output**). It feeds a DC-DC buck-boost; the 7.1 V the servos report is the converter output.
  - The torso carries 4 servos, the battery, Pi, IMU and the DC-DC.
- `pi-bringup` 8c2e1cb was merged with no conflicts: robot_config ranges, coupled ankle limit, mass estimates (1.50 kg), per-run action scale.
- `redesign/variant_model.py` was synced and re-checked: identical arrays and trace.

### 8.1 Backend option (backwards compatible)
- `pi_runner/mujoco_backend.py`:
  - `build_model(..., asset_dir=None)`, `build_spec(...)` (the uncompiled model) and `MujocoTrex(..., asset_dir=None)`;
  - optional robot_config `sim_self_collision.pairs`: explicit link-pair contacts, with libccd for them (see 8.4).
- `pi_runner/runner.py`: `PolicyRunner(..., robot_config=None)`.
- The default asset is unchanged: every compiled model array and a 3 s trace are identical to the pre-change build. v9d sim2sim smoke run OK.
- `redesign/variant_model.py` now builds on `build_spec`, so it no longer duplicates the backend.

### 8.2 What was built
**`redesign/trex_v2_design.yaml`** holds every parameter, each labelled [CAD] / [CONCEPT] / [ASSUMED] / [EST].

**`redesign/cad/v2_parts.py`** (build123d 0.13 in `trex_cad`) is the first-pass parametric CAD of the parts that change. All parts are valid solids and fit the **assumed 220 × 220 × 250 mm bed** (printer unknown).

| part | volume | print extent | content |
|---|---|---|---|
| pelvis plate | 95.8 cm³ | 162 × 164 × 33 mm | as-built plate outline (4 mm); ears and pads over the two **mirrored** hip servos (shaft inboard); Pi-deck holes; battery strap slots |
| hip bracket (L/R) | 26.4 cm³ | 82 × 44 × 82 mm | fork on the hip horns (as-built 14 mm horn pattern); cross-plate **24 mm** below the hip axis (as-built 15); femur-servo cradle; arm to the upper-idler pivot (50 mm offset kept) |
| foot (L/R) | 35.2 cm³ PLA | 142 × 68 × 31 mm | **68 mm** wide three-toed sole (85 as built), same length; as-built heel beam with the ankle and idler pivots (Ø10, 50 mm apart, 24.1 mm above the floor) |
| TPU sole pad (L/R) | 6.8 cm³ | 115 × 67 × 2 mm | 2 mm, inset 0.5 mm |

- Not changed: thigh and shank brackets, couplers, servos, head, tail, and the head/tail servo brackets. These are reused from Main v24, the left leg mirrored for the right (leg-2 assumption).
- Assumed: screw holes (M2 / M2.5), the 6 mm axis clearance, wall/infill, pad thickness, edge chamfer and the print orientation (largest face down; not optimised).

**`redesign/make_trex_v2.py`** writes `assets/robots/trex_v2/` (`Trex.urdf`, `meshes/`, `robot_config.yaml`, `build_report.json`):
- **The trex URDF conventions are kept:** same link and joint names, axes, rpy 0, mm meshes in the assembly frame, scale 0.001. The policy interface is unchanged.
- **The frame** is shifted +2.85 mm in y so the chassis centre line is y = 0.
- **Joints:**
  - hips at y = ±44 mm, z 205.38;
  - **hip-to-femur 62 mm** (55 as built, so the femur servo hangs below the deeper cross-plate): the leg is 7 mm longer;
  - thigh 65 / shank 54 mm;
  - all leg pivots at x = −9.5 (both legs; the trex URDF had them 2 mm apart).
- **Inertials:** each link is the sum of its parts, each a uniform solid of its own mass.
  - printed PLA at 30 % infill, 1.2 mm walls (assumed); TPU solid; servos 52 g;
  - battery 370 g as a 130 × 76 × 25 mm box. The size is a retailer figure ([desertcart](https://www.desertcart.nz/products/9799070)); the maker's page gives none.
  - electronics deck 167 g (estimates);
  - the couplers' mass is booked to the link they move with.
- **Battery on top of the plate**, centred. At 76 mm wide it no longer fits between the mirrored hip mounts, whose inner faces are 68 mm apart.
- **Masses (estimate):**

  | link | mass |
  |---|---|
  | base_link | 0.838 kg |
  | head / tail | 0.105 / 0.058 kg |
  | femur / tibia / ankle / foot (each side) | 80 / 73 / 31 / 36 g |
  | **total** | **1.441 kg** (current model 1.495) |

- **robot_config:**
  - generated from the design and trex's config (servo model, fitted actuator, gains);
  - `base_link_inertia: urdf`;
  - hip scale 20° and range −16…+20°;
  - tibia scale 90° and range −45…+90°;
  - coupled ankle −52…+33°;
  - `sim_self_collision` pairs;
  - a `power` block with the 3 A note.

### 8.3 Checks (`redesign/check_trex_v2.py`, `outputs/redesign/trex_v2_check.log`)
**Loading:**
- Loads in `MujocoTrex(asset_dir=...)` with the fitted servo model: 1.441 kg, 7 self-collision pairs, no self-contact at the zero pose.
- **Open-loop standing, 10 s:** zero pose, forward-knee crouch (femur 20°, tibia 40°) and v9c's backward-knee walking pose all hold; maximum tilt ≤ 0.7°, no self-contact.

**Static numbers** (`levers.figures`, current model = the merged pi-bringup config):

| | current model | trex_v2 |
|---|---|---|
| mass | 1.495 kg | 1.441 kg |
| CoM above the soles (walking pose) | 192 mm | 195 mm |
| stance half-width | 73.7 mm | 44 mm |
| **stance hip at the step switch / mid-stance** | 0.62 / 0.46 of stall | **0.37 / 0.27** (−41 %) |
| **friction needed at the step switch** | 0.385 | **0.24** |
| stance femur / tibia | 0.12 / 0.19 | 0.11 / 0.19 |
| standing, worst leg servo | 0.08 | 0.08 |
| LIPM roll (half amplitude) | 5.7° | 3.3° |
| lean to stand on one foot | 11.8° (−40 mm) | 6.4° (−22 mm) |
| double-support polygon, zero pose | x −47…+70, y −119…+113 mm; CoM margin 46.5 mm | x −46…+69, y ±77.5 mm; margin 41.6 mm |
| sole contact patch (hull) | 45.8 (60.4) cm² | 34.1 (51.4) cm² |
| roll / pitch fall time constant | 0.178 / 0.167 s | 0.164 / 0.167 s |

**Ankle range with the v2 foot** (shank and lower idler unchanged): contact at femur − tibia = **+35° / −54°** (the as-built foot: +39 / −57). The v2 config uses −52…+33. The beam/heel top of the v2 foot could be relieved to regain the 4°.

**Hip roll:**
- Hip bracket vs pelvis/servo: **outward contact at −18°**, inward free to 45° (the battery now sits on the plate). Hence the −16° limit. A deeper cross-plate would buy more outward roll at the cost of leg length.
- Feet: with the feet side by side (20 mm gap) **they touch after +6° of one hip inward, or +4° of both**. On the current robot the feet touch at +10° (both) and one foot reaches the other shank at +32°.
- The narrow stance trades inward foot placement for hip torque. A trained policy has to live with this: self-collision on in training.

**Clearance over v9d's gait replayed on v2:**
- v9d was run on the current model in MuJoCo with its deployed loop: 20 s each at 0 / 0.10 / 0.20 / 0.25 m/s, no falls. Its joint trajectories were then replayed kinematically on v2 (every 4th frame).
- Minimum distances, from mesh samples including the 4-bar couplers (±1 mm) and MuJoCo hull distances (`mj_geomDistance`):

| pair | standing (cmd 0) | walking (0.10–0.25 m/s) |
|---|---|---|
| foot ~ foot | **10.3 mm** | 24.0–25.1 mm |
| shank ~ shank | 15.6 mm | 25.0–25.7 mm |
| knee coupler ~ other shank | 18.5 mm | 28.0–28.7 mm |
| knee coupler ~ knee coupler | 21.5 mm | 30.9–31.6 mm |
| thigh ~ thigh | 38 mm | 44 mm |
| foot ~ other shank | 33–41 mm | 44–55 mm |
| idlers | > 54 mm | > 60 mm |

- The two methods agree to 0.3 mm on the pairs they share.
- v9d stands with its hips 1–4° inward, hence the 10 mm. It walks with the hips 12–16° **outward**, at the edge of v2's −16° stop, so a v2 policy has to find another lateral strategy. The replay checks clearances only; it is not a gait prediction.
- **Simulator note:** MuJoCo 3.8's native convex collider returned spurious 0.0 distances for these mesh pairs at some poses; libccd (`mjDSBL_NATIVECCD`) was consistent. The backend uses libccd only when self-collision pairs exist (the floor contacts are plane-mesh and unaffected).

### 8.4 Servo-bus current vs the battery's 3 A (`redesign/current_budget.py`)
**Model (estimate):**
- Per-servo current = 0.1 A + I_stall × |τ| / stall(V).
- I_stall: 2.4–3 A in the Hiwonder manual, 1.1–1.9 A in reseller listings ([manual](https://cdn.robotshop.com/rbm/a5e9ab51-86bc-497b-adac-003de3088fb7/9/92896585-5e83-4c7c-a663-650b81e5978b/67e9729b_lx-16a-serial-bus-servo-user-manual.pdf), [listing](https://openelab.io/a/s/products/lx-16a-bus-servo)).
- Supply ceiling: 3 A × V_batt × 0.88 (assumed efficiency) / 7.1 V ≈ **4.1 A at 11.1 V, 3.3 A at 9 V**, less whatever the Pi takes if it runs from the same pack.

v9d on the current model in MuJoCo, 30 s per command; percentages are of 20 ms windows:

| cmd | I_stall 1.5 A: mean / 20 ms max | I_stall 2.4 A: mean / 20 ms max | time > 4.1 A (1.5 / 2.4 A) |
|---|---|---|---|
| 0 (standing) | 1.7 / 2.0 A | 2.3 / 2.8 A | 0 % |
| 0.10 m/s | 3.0 / 6.5 A | 4.3 / 9.9 A | 16 % / 58 % |
| 0.20 m/s | 4.2 / 7.2 A | 6.2 / 11.0 A | 47 % / 96 % |
| 0.25 m/s | 4.3 / 8.5 A | 6.4 / 13.1 A | 55 % / 94 % |

- **If these estimates hold, walking needs more current than a 3 A 11 V pack can deliver.** A pack with over-current protection may cut out, or the bus voltage sags (slower servos).
- The current is mostly the femur/tibia servos accelerating their own rotors (3.5), which geometry does not change. v2 lowers the hip share (−41 % static).
- This is a model estimate, not a measurement: the first powered walking tests should log the bus current.
- Options: a pack/regulator rated for ≥ 10 A peaks (e.g. a 2S pack straight to the 6–8.4 V servo bus, or a high-current UBEC), and/or a current-aware training term (study v2d).

### 8.5 Training studies for trex_v2 (`redesign/experiments/*.yaml`, v9e recipe; 1500 iterations, 4096 envs)

| file | study | compare with |
|---|---|---|
| `v2a_base.yaml` | v9e recipe on trex_v2, self-collision on, base-mass DR 0.75–1.3 | v9e on the current robot (same recipe): isolates the body |
| `v2b_noselfcol.yaml` | v2a without self-collision | v2a: does the policy cross its legs when allowed? |
| `v2c_tpu.yaml` | v2a with friction DR 0.5–1.2 (TPU, assumed) | v2a (0.1–1.2) |
| `v2d_current_cap.yaml` | v2a + `servo_current` penalty: relu(I_bus − 4.0 A), current model of 8.4 | v2a: speed vs bus current |
| seeds | v2a and v9e, 3 seeds each | the go / no-go for printing |

**Pipeline changes these need** (not made here; they touch the Isaac stacks, which need the GPU to test):
1. a `robot.asset_dir` key read where `assets/robots/trex` is hardcoded: `scripts/test_load_robot.py` (ROBOT_DIR, USD_OUT_DIR, composite path), `walk_lab/bake_usd.py`, the PolicyRunner call sites;
2. articulation self-collision for the configured pairs (PhysX `enabledSelfCollisions`, all other pairs filtered), treated as a termination like other non-foot contacts;
3. for v2d only, a `servo_current` reward term.

Evaluate with `walk_rl/evaluate.py` and MuJoCo sim2sim on `MujocoTrex(asset_dir=assets/robots/trex_v2)`. Compare speed, minimum speed with a gait, roll p2p, servo time above 80 % of stall, self-contacts per episode, push survival, slip, and the estimated bus current.

### 8.6 What is assumed in leg 3 (all flagged in the design file)
- printer bed 220 × 220 × 250;
- battery size;
- PLA 30 % infill / 1.2 mm walls;
- electronics deck mass and size;
- screw holes;
- TPU pad 2 mm and friction;
- both legs printed from the same STLs;
- LX-16A stall current;
- converter efficiency;
- the Pi's supply path.


### 8.7 v2a / v2b results (2026-10-20, v9i recipe, measured masses)
- The first v2a run overflowed the PhysX GPU aggregate-pair buffer (self-collision needs ~14 pairs per env) and missed contacts. Buffers are now 16 per env; the runs below have no PhysX errors.
- Runner parity OK for both. Isaac, fitted servo, randomization on:

| run | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | survival | roll p2p | push 0.4 walking |
|---|---|---|---|---|---|---|---|---|
| v9i (current body) | 93% | 86% | 85% | 76% | 67% | 0.97–1.00 | 10–16° | 0.89 |
| v2a (self-collision) | 98% | 91% | 93% | 88% | 82% | 0.98–1.00 | 15–20° | 0.75 |
| v2b (no self-collision) | 79% | 89% | 93% | 89% | 83% | 0.99–1.00 | 14–15° | 0.74 |

- MuJoCo (60 s, friction 0.3/0.5/0.8 × delay 0–2 × cmd 0/0.05/0.10/0.20):
  - v9i 32/36 (falls at 0.20), 53–63% of 0.10;
  - v2a 33/36: 3 tail falls at friction 0.3/0.5 after a large turn (~150°); 56–84% of 0.10;
  - v2b 36/36; 73–99% of 0.10, 66–89% of 0.20.
- **The v2 body tracks the command better in both simulators and transfers better to MuJoCo.** Self-collision termination did not help: v2a turns and falls in MuJoCo, v2b does not.
- Older runs kept for reference: `v2a_overflow` (buffer overflow), `v2b_progress` (v9h recipe: overspeed 260% at 0.05, MuJoCo 31/36).
