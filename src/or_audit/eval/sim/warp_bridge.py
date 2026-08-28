"""NVIDIA Warp and Isaac Lab Simulation Bridge for GPU-Accelerated Rollouts.

Provides high-throughput parallel physics simulation for surgical robotics,
guidewire dynamics, and large-batch policy evaluation. A synthetic stand-in exists
for headless CI, but it is refused unless the task opts in, and it is stamped into
every artifact it touches.
"""

from __future__ import annotations

from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.sim.base import (
    BACKEND_REAL,
    BACKEND_SYNTHETIC_STUB,
    BaseSimulationBridge,
    SimulationEngine,
    module_distribution_version,
)
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import world_kind_key

_WARP_MODULES = "'warp' / 'isaaclab'"


def _refuse_synthetic_warp(kind: str) -> str:
    return (
        f"world kind {kind!r} has no NVIDIA Warp backend attached: the GPU physics "
        f"modules ({_WARP_MODULES}) did not yield a runnable simulation. A synthetic "
        "stand-in would invent joint positions, penetration, and haptic overshoot "
        "numbers that this task's hard safety gates would then score as physical "
        "evidence. Install warp-lang and Isaac Lab, or set "
        "environment.synthetic_stub = true in task.toml to accept a non-physical "
        'stand-in (artifacts are stamped backend="synthetic-stub" and export-rl refuses '
        "the run)."
    )


class WarpBridge(BaseSimulationBridge):
    """Bridge for NVIDIA Warp / Isaac Lab GPU-accelerated simulation environments."""

    world_kind: WorldKind | str = WorldKind.WARP

    def __init__(
        self,
        env_name: str,
        *,
        parameters: dict[str, Any] | None = None,
        world_pin: str = "",
        num_envs: int = 1,
        warp_env: Any = None,
        allow_synthetic: bool = False,
        backend_version: str = "",
        world_kind: WorldKind | str | None = None,
        max_steps: int = 100,
    ) -> None:
        kind = world_kind if world_kind is not None else self.world_kind
        if warp_env is None and not allow_synthetic:
            raise TaskContractError(_refuse_synthetic_warp(world_kind_key(kind)))
        self.world_kind = kind
        self.env_name = env_name
        self.parameters = parameters or {}
        self.world_pin = world_pin
        self.num_envs = int(parameters.get("num_envs", num_envs)) if parameters else num_envs
        self._env = warp_env
        self._backend_version = backend_version
        #: Harness step limit, passed by the factory from ``task.harness.max_steps``.
        #: Only the synthetic stand-in consults it; a real engine terminates itself.
        self.max_steps = max_steps

        self._step_count = 0

    def engine_provenance(self) -> dict[str, Any]:
        """Report whether a real Warp simulation or the synthetic stand-in produced data."""
        synthetic = self._env is None
        return {
            "engine": world_kind_key(self.world_kind),
            "backend": BACKEND_SYNTHETIC_STUB if synthetic else BACKEND_REAL,
            "backend_version": "" if synthetic else self._backend_version,
            "world_pin": self.world_pin,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset Warp vectorized simulation."""
        self._step_count = 0
        if self._env is not None and hasattr(self._env, "reset"):
            return self._env.reset(seed=seed, options=options)  # type: ignore[no-any-return]
        obs = {
            "warp_env": self.env_name,
            "num_envs": self.num_envs,
            "robot_joint_pos": [0.0] * 7,
            "tool_ee_pos": [0.0, 0.0, 0.0],
        }
        # Same invariant as the Isaac and SOFA stand-ins: no physical safety key
        # is synthesized. `max_pen` and `haptic_overshoot_mm` are penetration
        # and haptic measurements, and an unsolved GPU scene has neither.
        info = {
            "warp_initialized": True,
            "gpu_device": self.parameters.get("device", "cuda:0"),
            "world_pin": self.world_pin,
            "seed": seed,
            "backend": BACKEND_SYNTHETIC_STUB,
        }
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one parallel Warp / PhysX 5 step."""
        self._step_count += 1
        if self._env is not None and hasattr(self._env, "step"):
            return self._env.step(action)  # type: ignore[no-any-return]

        # Step budget comes from the harness, not from `environment.parameters`:
        # that dict is forwarded verbatim to a real engine's constructor, so a
        # magic `max_steps` key there is both an invalid kwarg for most envs and
        # a second source of truth for a limit the harness already owns.
        max_steps = self.max_steps
        terminated = self._step_count >= max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0

        obs = {
            "warp_env": self.env_name,
            "num_envs": self.num_envs,
            "robot_joint_pos": [0.01 * self._step_count] * 7,
            "tool_ee_pos": [0.1 * self._step_count, 0.0, 0.0],
        }
        info = {
            "step": self._step_count,
            "raw_success": terminated,
            "backend": BACKEND_SYNTHETIC_STUB,
        }
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        """Return GPU tensor state snapshot."""
        if self._env is not None and hasattr(self._env, "get_state"):
            return self._env.get_state()  # type: ignore[no-any-return]
        return {
            "env_name": self.env_name,
            "step_count": self._step_count,
            "num_envs": self.num_envs,
            "backend": BACKEND_SYNTHETIC_STUB,
        }

    def close(self) -> None:
        """Release GPU context and Warp memory pools."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def _acquire_warp_env(task: TaskSpec) -> tuple[Any, str]:
    """Best-effort acquisition of a real Warp/Isaac Lab env: returns (env, version)."""
    try:
        import warp
    except ImportError:
        return None, ""
    detected = module_distribution_version("warp") or str(getattr(warp, "__version__", "") or "")
    env_id = task.environment.gym_id
    if not env_id:
        return None, detected
    try:
        import gymnasium
        import isaaclab  # noqa: F401
    except ImportError:
        return None, detected
    kwargs: dict[str, Any] = dict(task.environment.parameters)
    try:
        return gymnasium.make(env_id, **kwargs), detected
    except Exception:  # Isaac Lab raises engine-specific errors for an unknown env id
        return None, detected


def make_warp_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating a WarpBridge for a task, preferring a real Warp simulation."""
    warp_env, backend_version = _acquire_warp_env(task)
    return WarpBridge(
        env_name=task.environment.gym_id or task.id,
        parameters=dict(task.environment.parameters),
        world_pin=task.environment.world_pin,
        warp_env=warp_env,
        allow_synthetic=task.environment.synthetic_stub,
        backend_version=backend_version,
        world_kind=task.environment.kind,
        max_steps=task.harness.max_steps,
    )
