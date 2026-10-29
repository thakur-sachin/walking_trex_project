"""Pick the simulator stack for a script: the original Isaac Sim stack (walk_rl/env.py,
conda env isaac6) or the Isaac Lab port (walk_lab/env.py, conda env isaaclab6). Both run
the same task (walk_rl/task_core.py) behind the same env API. lab3: Isaac Lab 3 + Newton (MJWarp),
kit-less (walk_lab3/env.py, the ~/trex_stack/IsaacLab_dev/.venv venv).

    app = backend.launch(args.backend, device=args.device)   # before any omni/isaac import
    EnvCls = backend.env_class(args.backend)
"""

BACKENDS = ("isaacsim", "lab", "lab3")


class _NoApp:
    """lab3 is kit-less: nothing to launch or close."""

    def close(self) -> None:
        pass


def launch(name: str, headless: bool = True, device: str = "cuda:0"):
    if name == "isaacsim":
        from isaacsim import SimulationApp

        return SimulationApp({"headless": headless})
    if name == "lab":
        try:
            from isaaclab.app import AppLauncher
        except ImportError as e:
            raise SystemExit("--backend lab needs the isaaclab6 conda env "
                             "(the isaaclab6 conda env)") from e
        return AppLauncher(headless=headless, device=device).app
    if name == "lab3":
        return _NoApp()
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")


def env_class(name: str):
    if name == "isaacsim":
        from walk_rl.env import TrexWalkEnv

        return TrexWalkEnv
    if name == "lab3":
        from walk_lab3.env import TrexLab3WalkEnv

        return TrexLab3WalkEnv
    from walk_lab.env import TrexLabWalkEnv

    return TrexLabWalkEnv
