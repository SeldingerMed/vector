"""Gymnasium and Lumen Simulation Bridge."""

from __future__ import annotations

from typing import Any

from or_audit.eval.enums import WorldKind
from or_audit.eval.gym_world import GymEnv, make_gym
from or_audit.eval.sim.base import (
    BACKEND_REAL,
    BaseSimulationBridge,
    SimulationEngine,
    module_distribution_version,
)
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import world_kind_key


class GymnasiumBridge(BaseSimulationBridge):
    """Bridge for standard Gymnasium, PyBullet, and Lumen environments."""

    def __init__(
        self,
        env: GymEnv,
        *,
        world_kind: WorldKind | str = WorldKind.GYM,
        world_pin: str = "",
    ) -> None:
        self.env = env
        self.world_kind = world_kind
        self.world_pin = world_pin

    @property
    def unwrapped(self) -> Any:
        """Return the unwrapped base environment."""
        return getattr(self.env, "unwrapped", self.env)

    def engine_provenance(self) -> dict[str, Any]:
        """Report the wrapped environment's own distribution as the backend."""
        return {
            "engine": world_kind_key(self.world_kind),
            "backend": BACKEND_REAL,
            "backend_version": module_distribution_version(type(self.unwrapped).__module__),
            "world_pin": self.world_pin,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset Gymnasium environment."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step Gymnasium environment."""
        return self.env.step(action)

    def render(self, mode: str = "rgb_array") -> Any:
        """Render frame if environment supports rendering."""
        render_fn = getattr(self.env, "render", None)
        if callable(render_fn):
            try:
                return render_fn()
            except TypeError:
                return render_fn(mode=mode)
        return None

    def close(self) -> None:
        """Close underlying environment."""
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()

    def get_state(self) -> dict[str, Any]:
        """Extract state dictionary if available from underlying env."""
        unwrapped = getattr(self.env, "unwrapped", self.env)
        state_fn = getattr(unwrapped, "get_state", None)
        if callable(state_fn):
            return state_fn()  # type: ignore[no-any-return]
        return {}


def make_gym_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating a GymnasiumBridge for a task."""
    env = make_gym(task)
    return GymnasiumBridge(
        env,
        world_kind=task.environment.kind,
        world_pin=task.environment.world_pin,
    )
