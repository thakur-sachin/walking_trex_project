"""Write the Trex MJCF (pi_runner/mujoco_backend.build_spec, the MuJoCo sim2sim model) for Isaac Lab 3 / Newton,
then convert it to USD with the Isaac Sim MJCF importer (kit-less). One model source for both simulators:
robot_config masses, joint ranges, and the ankle linkage tendon equality (ankle = femur - tibia).

    ~/trex_stack/IsaacLab_dev/.venv/bin/python lab3/export_mjcf.py [--out outputs/lab3/trex]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.mujoco_backend import build_spec  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "lab3" / "trex"))
ap.add_argument("--no-usd", action="store_true")
a = ap.parse_args()

rc = RobotConfig.load(ROBOT_CONFIG)
spec = build_spec(rc, physics_hz=200, spawn_height_m=0.0, ankle_constraint="tendon_limit")
spec.delete(spec.geom("floor"))  # Isaac Lab adds the terrain
for act in list(spec.actuators):  # the actuator model comes from the Isaac Lab config (BAM)
    spec.delete(act)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
xml = out / "trex.xml"
xml.write_text(spec.to_xml())
print(f"wrote {xml}")
if not a.no_usd:
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    cfg = MJCFImporterConfig(mjcf_path=str(xml), usd_path=str(out / "usd"))
    print("usd:", MJCFImporter(cfg).import_mjcf())
