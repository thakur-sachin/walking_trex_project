# CAD from Fusion 360 (hub Dev_designs)

Fetched read-only with `redesign/aps_fusion.py` (login, hubs, download) and `redesign/aps_step.py`
(STEP export). The files are gitignored because of their size; the commands below re-create them.

| folder | source | content |
|---|---|---|
| `incoming/final_version/` | project "T-Rex Final Version" | **the robot as built (Sachin, 2026-10-19): `Main` v24 (2024-04-01)**. `Main_v24.step` = full assembly (837 solids, 270 occurrences: legs with couplers/idlers, LX-16A servos and adapters, brackets, chassis, head, tail, feet, battery, BusLinker, DC-DC, Pi). Native `.f3d` of Main/Head/Tail/Bracket_mount/IMU/Leg_Metric_Left. `STL Files/` = the print exports of 2024-03-23..31 (`*_v1`/`*_v2` are the later re-exports) |
| `incoming/urdf_trex/` | project "URDF_Trex" | simplified model used for the simulation URDF (`Trex` v9, last edited 2025-08-03), parts, couplers, and `export/Trex_main_no_couplers.step` |

```
python3 redesign/aps_fusion.py download "T-Rex Final Version" --out cad/incoming/final_version
python3 redesign/aps_step.py "T-Rex Final Version" Main --modified 2024-04-01 --out cad/incoming/final_version/Main_v24.step
python3 redesign/aps_fusion.py download URDF_Trex --out cad/incoming/urdf_trex
```
Version comments are not exposed by the APS APIs; the version choice was confirmed by Sachin.
