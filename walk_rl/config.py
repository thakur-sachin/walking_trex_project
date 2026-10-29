"""Task-config loading without torch (shared by training, evaluation and the Pi runner).

The deployed runner (pi_runner/) needs the same config overlay and checkpoint-interface
rules as training but must not import torch, so they live here; walk_rl.task_core
re-exports them.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_TASK_PATH = Path(__file__).resolve().parent / "walk_task.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET_DIR = PROJECT_ROOT / "assets" / "robots" / "trex"


def robot_asset_dir(cfg: SimpleNamespace) -> Path:
    """The run's robot asset directory (task config `robot.asset_dir`, repo-relative)."""
    d = getattr(getattr(cfg, "robot", None), "asset_dir", None)
    return (PROJECT_ROOT / d).resolve() if d else DEFAULT_ASSET_DIR


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def ns_to_dict(ns):
    return {k: ns_to_dict(v) if isinstance(v, SimpleNamespace) else v for k, v in vars(ns).items()}


def _deep_update(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_task_config(path: str | Path = DEFAULT_TASK_PATH) -> SimpleNamespace:
    """Load a task config, overlaid on the current defaults so that a run saved before a
    key existed (e.g. v4's walk_task.yaml has no ground_contact_links) still loads, and is
    evaluated under today's rules for anything it did not specify."""
    with open(DEFAULT_TASK_PATH) as f:
        cfg = yaml.safe_load(f)
    if Path(path) != DEFAULT_TASK_PATH:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return _ns(cfg)


def apply_checkpoint_interface(cfg: SimpleNamespace, meta: dict) -> SimpleNamespace:
    """Make the actor-input layout match a saved policy. Runs trained before v8 (no
    `obs_layout` in their meta) had one frame and no heading input; today's defaults
    would otherwise give them a bigger observation than their network takes."""
    layout = meta.get("obs_layout")
    ob = cfg.observation
    if layout is None:
        ob.history_len = 1
        if hasattr(ob, "heading"):
            ob.heading.enabled = False
    else:
        ob.history_len = layout["history_len"]
        if hasattr(ob, "heading"):
            ob.heading.enabled = any(n.startswith("heading_err") for n in layout["frame"])
    return cfg


def joint_scale_rad(cfg: SimpleNamespace, rc, names: list[str]) -> list[float]:
    """Per-joint action scale and observation normalization [rad] of the policy: robot_config's
    symmetric joint limit unless `actions.joint_scale_deg` overrides it for this run (e.g. a wider
    tibia range). It is part of the policy interface, so it lives with the run's config."""
    over = vars(getattr(cfg.actions, "joint_scale_deg", None) or SimpleNamespace())
    return [math.radians(float(over[n])) if n in over else rc.joint_limit_rad(n)[1] for n in names]


def joint_center_rad(cfg: SimpleNamespace, names: list[str]) -> list[float]:
    """Per-joint action/observation centre [rad]: target = centre + a * scale, obs = (q - centre) / scale.
    0 unless `actions.joint_center_deg` shifts it for this run (e.g. a tibia range that is not
    symmetric about zero). Part of the policy interface, like joint_scale_rad."""
    over = vars(getattr(cfg.actions, "joint_center_deg", None) or SimpleNamespace())
    return [math.radians(float(over.get(n, 0.0))) for n in names]


def obs_dims(cfg: SimpleNamespace) -> tuple[int, int, int]:
    """(frame_dim, actor_obs, critic_obs) for a task config; what _init_buffers builds."""
    ob = cfg.observation
    heading = bool(getattr(getattr(ob, "heading", None), "enabled", False))
    frame = 3 + 3 + 1 + 8 + 8 + 8 + (2 if cfg.gait.use_phase_clock else 0) + (2 if heading else 0)
    return frame, frame * int(getattr(ob, "history_len", 1)), frame + 3 + 2 + 2 + 8
