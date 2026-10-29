"""Loader for a robot's robot_config.yaml -- the swappable source of truth for
physical/actuation parameters that the raw URDF doesn't capture correctly
(mass, joint limits, real servo specs, which joints are actually actuated).

Usage:
    from common.robot_config import RobotConfig
    cfg = RobotConfig.load(PROJECT_ROOT / "assets" / "robots" / "trex" / "robot_config.yaml")
    cfg.joint_limit_rad("left_hip_joint")   # -> (-0.5236, 0.5236)
    cfg.sim_drive_for_joint("left_femur_joint")  # -> (stiffness=40.0, damping=5.0)
    cfg.ankle_angle_from_linkage(femur_angle_rad, tibia_angle_rad)  # confirmed 4-bar relationship
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SimDriveGroup:
    joint_pattern: str
    stiffness: float
    damping: float


@dataclass
class MaterialGroup:
    name: str
    color_rgb: tuple[float, float, float]
    roughness: float
    links: list[str]


@dataclass
class RobotConfig:
    base_link_mass_kg: float
    default_limit_deg: float
    joint_limit_overrides_deg: dict[str, float]
    actuated_joints: list[str]
    passive_joints: list[str]
    passive_joint_stiffness: float
    passive_joint_damping: float
    actuator_stall_torque_nm: float
    actuator_no_load_speed_rad_per_sec: float
    actuator_spec: dict
    sim_drive_groups: list[SimDriveGroup]
    material_groups: list[MaterialGroup]
    sensors: dict
    compute: dict
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "RobotConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        joints = raw["joints"]
        actuator = raw["actuator"]
        sim_drive = raw["sim_drive"]

        return cls(
            base_link_mass_kg=raw["mass"]["base_link_kg"],
            default_limit_deg=joints["default_limit_deg"],
            joint_limit_overrides_deg=joints.get("overrides_deg", {}),
            actuated_joints=list(joints["actuated_joints"]),
            passive_joints=list(joints["passive_joints"]),
            passive_joint_stiffness=joints.get("passive_joint_drive", {}).get("stiffness", 0.0),
            passive_joint_damping=joints.get("passive_joint_drive", {}).get("damping", 0.0),
            actuator_stall_torque_nm=actuator["stall_torque_nm"],
            actuator_no_load_speed_rad_per_sec=actuator["no_load_speed_rad_per_sec"],
            actuator_spec=actuator,
            sim_drive_groups=[
                SimDriveGroup(joint_pattern=g["joint_pattern"], stiffness=g["stiffness"], damping=g["damping"])
                for g in sim_drive.values()
            ],
            material_groups=[
                MaterialGroup(
                    name=name,
                    color_rgb=tuple(g["color_rgb"]),
                    roughness=g.get("roughness", 0.5),
                    links=list(g["links"]),
                )
                for name, g in raw.get("materials", {}).items()
            ],
            sensors=raw.get("sensors", {}),
            compute=raw.get("compute", {}),
            raw=raw,
        )

    def joint_limit_deg(self, joint_name: str) -> float:
        """Symmetric +/- limit in degrees for one joint (overrides win over default)."""
        return self.joint_limit_overrides_deg.get(joint_name, self.default_limit_deg)

    def joint_limit_rad(self, joint_name: str) -> tuple[float, float]:
        """(lower, upper) symmetric limit in radians for one joint."""
        limit = math.radians(self.joint_limit_deg(joint_name))
        return (-limit, limit)

    def sim_drive_for_joint(self, joint_name: str) -> tuple[float, float]:
        """(stiffness, damping) for one joint, by matching its sim_drive group pattern.
        Raises if a joint doesn't match any group and isn't in passive_joints -- fail loud
        rather than silently drive a joint with made-up gains.
        """
        if joint_name in self.passive_joints:
            return (self.passive_joint_stiffness, self.passive_joint_damping)
        for group in self.sim_drive_groups:
            if re.fullmatch(group.joint_pattern, joint_name):
                return (group.stiffness, group.damping)
        raise ValueError(f"No sim_drive group or passive-joint entry matches joint '{joint_name}'")

    def is_actuated(self, joint_name: str) -> bool:
        return joint_name in self.actuated_joints

    def urdf_importer_stiffness_overrides(self) -> dict[str, float]:
        """{regex: stiffness} dict in the shape isaacsim's URDFImporterConfig expects,
        built from sim_drive_groups plus the passive joints' own drive (see
        passive_joint_drive in robot_config.yaml -- "passive" means no independent
        servo, not "no drive": the ankle needs real stiffness to hold the target
        computed by the linkage relationship, same as any other joint)."""
        overrides = {g.joint_pattern: g.stiffness for g in self.sim_drive_groups}
        if self.passive_joints:
            overrides["|".join(self.passive_joints)] = self.passive_joint_stiffness
        return overrides

    def urdf_importer_damping_overrides(self) -> dict[str, float]:
        overrides = {g.joint_pattern: g.damping for g in self.sim_drive_groups}
        if self.passive_joints:
            overrides["|".join(self.passive_joints)] = self.passive_joint_damping
        return overrides

    def material_group_for_link(self, link_name: str) -> MaterialGroup | None:
        for group in self.material_groups:
            if link_name in group.links:
                return group
        return None

    def ankle_angle_from_linkage(self, femur_angle_rad: float, tibia_angle_rad: float) -> float:
        """Ankle joint angle enforced by the real 4-bar parallelogram linkage (see
        `ankle_linkage` in robot_config.yaml for the confirmed bar dimensions).

        femur_joint, tibia_joint, and ankle_joint all rotate about the same *physical*
        axis, and the linkage keeps the foot parallel to base_link at every femur/tibia
        configuration -- so the three relative rotations about that axis must sum to
        zero (true at the neutral pose, where the foot is flat and all three are 0).
        BUT the URDF's `<axis>` for femur_joint is (0,-1,0) while tibia_joint and
        ankle_joint are both (0,+1,0) -- opposite sign from femur, same sign as each
        other. A naive -(femur+tibia) (same-sign assumption) is wrong by a large margin
        (verified: produces ~44deg of residual foot tilt instead of 0 -- see git history
        for the live-session debugging that caught this). Correctly accounting for the
        sign flip:
            -femur_angle + tibia_angle + ankle_angle = 0  =>  ankle_angle = femur_angle - tibia_angle
        Verified analytically (Gf.Rotation composition, zero floating point residual)
        and confirmed live in Isaac Sim 2026-10-09.

        Pure kinematics, no Isaac Sim dependency -- callers wire this to the actual
        DOF positions (see scripts/test_load_robot.py's enforce_ankle_linkage()).
        """
        return femur_angle_rad - tibia_angle_rad
