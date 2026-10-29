"""Where a result came from: code commit and the robot model it ran on.

Every training run, evaluation and sim2sim sweep stores this, so a number can always be traced to the
robot model it was measured on. Results from different `robot_config` hashes are not comparable.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOT_FILES = [PROJECT_ROOT / "assets" / "robots" / "trex" / f for f in ("robot_config.yaml", "Trex.urdf")]


def robot_model_hash(robot_config: str | Path | None = None) -> str:
    """sha256 of the parsed robot_config.yaml (default: the current one) + Trex.urdf, 16 hex chars.
    Comments do not change it."""
    h = hashlib.sha256()
    h.update(json.dumps(yaml.safe_load(Path(robot_config or ROBOT_FILES[0]).read_text()), sort_keys=True).encode())
    h.update(ROBOT_FILES[1].read_bytes())
    return h.hexdigest()[:16]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def provenance(robot_config: str | Path | None = None) -> dict:
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "robot_model_hash": robot_model_hash(robot_config)}
