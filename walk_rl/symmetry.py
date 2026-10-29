"""Left/right mirror map for the Trex walking task (symmetry augmentation / mirror loss).

Every quantity in the actor and critic observations and in the action maps to its mirror
image by a permutation plus a sign, so mirroring is `x[..., perm] * sign`.

Joint conventions (docs/JOINT_CONVENTIONS.md): a positive hip angle drives EITHER leg
inward, and femur/tibia axes are the same for both legs, so mirroring a joint vector SWAPS
left and right without negating anything. Head and tail (pitch joints) map to themselves.
IMU and base quantities in the body frame (x forward, y left, z up): y components of
linear vectors flip; x (roll) and z (yaw) components of angular vectors flip.
The gait clock shifts by half a period (left foot <-> right foot), which negates sin and cos.
Heading error yaw - yaw_ref flips sign: sin flips, cos stays.

walk_rl/test_symmetry.py checks this map algebraically AND against the simulator (a
mirrored action sequence from a mirrored state must give mirrored observations).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

ACTUATED = ["head_joint", "tail_joint", "left_hip_joint", "right_hip_joint",
            "left_femur_joint", "right_femur_joint", "left_tibia_joint", "right_tibia_joint"]


def joint_perm(names: list[str]) -> list[int]:
    """Index of each joint's mirror partner (left_* <-> right_*, others fixed)."""
    swap = lambda n: n.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
    return [names.index(swap(n)) for n in names]


class _Block:
    def __init__(self):
        self.perm: list[int] = []
        self.sign: list[float] = []

    def add(self, perm_local: list[int], sign: list[float]) -> None:
        base = len(self.perm)
        self.perm += [base + p for p in perm_local]
        self.sign += sign

    def extend_frames(self, k: int) -> None:
        """Repeat the current layout k times (stacked history frames)."""
        d = len(self.perm)
        self.perm = [f * d + p for f in range(k) for p in self.perm]
        self.sign = self.sign * k


class MirrorMap:
    """Mirror maps for one task config. Use `.actor(obs)`, `.critic(obs)`, `.action(a)`."""

    def __init__(self, cfg: SimpleNamespace, act_names: list[str] = ACTUATED, device="cpu"):
        jp = joint_perm(act_names)
        A = len(act_names)
        ob = cfg.observation
        heading = bool(getattr(getattr(ob, "heading", None), "enabled", False))
        clock = bool(cfg.gait.use_phase_clock)

        def frame(b: _Block) -> None:  # layout of task_core._observations' frame
            b.add([0, 1, 2], [1, -1, 1])  # gravity_b
            b.add([0, 1, 2], [-1, 1, -1])  # gyro_b (roll, pitch, yaw)
            b.add([0], [1])  # cmd_vx
            for _ in range(3):  # servo pos, finite-diff vel, previous action
                b.add(jp, [1.0] * A)
            if clock:
                b.add([0, 1], [-1, -1])  # phase + 0.5
            if heading:
                b.add([0, 1], [-1, 1])  # sin, cos of heading error

        a = _Block()
        frame(a)
        a.extend_frames(int(getattr(ob, "history_len", 1)))
        c = _Block()
        frame(c)
        c.add([0, 1, 2], [1, -1, 1])  # base linear velocity (body frame)
        c.add([1, 0], [1, 1])  # foot contacts L, R
        c.add([1, 0], [1, 1])  # foot clearance L, R
        c.add(jp, [1.0] * A)  # servo torques
        ac = _Block()
        ac.add(jp, [1.0] * A)

        T = lambda v, dt=torch.long: torch.tensor(v, dtype=dt, device=device)
        self._p = {k: T(b.perm) for k, b in (("actor", a), ("critic", c), ("action", ac))}
        self._s = {k: T(b.sign, torch.float32) for k, b in (("actor", a), ("critic", c), ("action", ac))}
        self.dims = {k: len(v) for k, v in self._p.items()}

    def to(self, device) -> "MirrorMap":
        self._p = {k: v.to(device) for k, v in self._p.items()}
        self._s = {k: v.to(device) for k, v in self._s.items()}
        return self

    def _apply(self, key: str, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dims[key], f"{key}: got {x.shape[-1]}, map built for {self.dims[key]}"
        return x[..., self._p[key]] * self._s[key]

    def actor(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply("actor", x)

    def critic(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply("critic", x)

    def action(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply("action", x)


def rsl_rl_augmentation(env, obs, actions):
    """rsl_rl `data_augmentation_func`: returns (original + mirrored) along the batch dim.
    `obs` is a TensorDict with 'policy' and 'critic' groups (walk_lab/env.py)."""
    m = getattr(env, "_mirror_map", None)
    if m is None:  # env = rsl_rl's RslRlVecEnvWrapper around walk_lab's TrexLabDirectEnv
        task = env.unwrapped.task
        m = env._mirror_map = MirrorMap(task.cfg, list(task.act_names), device=task.device)
    obs_out = act_out = None
    if obs is not None:
        obs_out = torch.cat([obs, obs.clone()], dim=0)
        n = obs.batch_size[0]
        obs_out["policy"][n:] = m.actor(obs["policy"])
        obs_out["critic"][n:] = m.critic(obs["critic"])
    if actions is not None:
        act_out = torch.cat([actions, m.action(actions)], dim=0)
    return obs_out, act_out
