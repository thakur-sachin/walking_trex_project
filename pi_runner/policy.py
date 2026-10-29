"""The deployable policy: observation normalizer + ELU MLP in plain NumPy (no torch on the Pi).
Loads `policy_numpy.npz` as written by walk_rl.ppo.export_numpy_policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, path: str | Path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.var, self.clip = d["norm_mean"], d["norm_var"], float(d["norm_clip"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["num_layers"]))]
        self.meta = json.loads(str(d["meta"])) if "meta" in d else {}
        self.obs_dim = self.mean.shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return x
