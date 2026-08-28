"""SOFA Framework and SofaGym Simulation Bridge for Soft-Tissue and Biomechanics.

Provides biomechanical simulation for catheter Cosserat rods, vascular elasticity,
and soft-tissue deformation. A synthetic stand-in exists for headless CI, but it is
refused unless the task opts in, and it is stamped into every artifact it touches.
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
    missing_world_errors,
    module_distribution_version,
    refuse_unbuildable_world,
    require_step_scalar,
)
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import world_kind_key

_SOFA_MODULES = "'Sofa' / 'SofaRuntime'"


def _refuse_synthetic_sofa(kind: str) -> str:
    return (
        f"world kind {kind!r} has no SOFA backend attached: the SOFA python bindings "
        f"({_SOFA_MODULES}) did not yield a runnable scene. A synthetic stand-in would "
        "invent tissue stress, wall force, and penetration numbers that this task's hard "
        "safety gates would then score as physical evidence. Install SOFA, or set "
        "environment.synthetic_stub = true in task.toml to accept a non-physical "
        'stand-in (artifacts are stamped backend="synthetic-stub" and export-rl refuses '
        "the run)."
    )


class SofaBridge(BaseSimulationBridge):
    """Bridge for SOFA Framework / SofaGym biomechanical simulations."""

    world_kind: WorldKind | str = WorldKind.SOFA

    def __init__(
        self,
        scene_name: str,
        *,
        parameters: dict[str, Any] | None = None,
        world_pin: str = "",
        sofa_env: Any = None,
        allow_synthetic: bool = False,
        backend_version: str = "",
        max_steps: int = 100,
    ) -> None:
        if sofa_env is None and not allow_synthetic:
            raise TaskContractError(_refuse_synthetic_sofa(world_kind_key(self.world_kind)))
        self.scene_name = scene_name
        self.parameters = dict(parameters or {})
        self.world_pin = world_pin
        self._env = sofa_env
        self._backend_version = backend_version
        #: Harness step limit, passed by the factory from ``task.harness.max_steps``.
        #: Only the synthetic stand-in consults it; a real engine terminates itself.
        self.max_steps = max_steps

        self._step_count = 0

    def engine_provenance(self) -> dict[str, Any]:
        """Report whether a real SOFA scene or the synthetic stand-in produced the data."""
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
        """Reset SOFA simulation scene."""
        self._step_count = 0
        if self._env is not None and hasattr(self._env, "reset"):
            return self._env.reset(seed=seed, options=options)  # type: ignore[no-any-return]
        # No physical key is synthesized. `tissue_stress_kpa` and the rest were
        # invented numbers with physical units, which is the most convincing
        # possible lie: a gate binding to them would resolve pass against a
        # scene that was never solved. A stand-in reports only that it started.
        obs = {
            "beam_elements": 20,
            "scene": self.scene_name,
        }
        info = {
            "sofa_initialized": True,
            "scene_name": self.scene_name,
            "world_pin": self.world_pin,
            "seed": seed,
            "backend": BACKEND_SYNTHETIC_STUB,
        }
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one SOFA FEA / BeamFEM step, refusing a batch rather than leaking it."""
        self._step_count += 1
        if self._env is not None and hasattr(self._env, "step"):
            obs, reward, terminated, truncated, info = self._env.step(action)
            # The engine tuple was returned verbatim through this
            # `-> float, bool, bool` signature, so a non-scalar reward or flag
            # reached the rollout loop unchecked, where `bool()` of a non-empty
            # sequence is unconditionally True. Coerce only what is genuinely
            # scalar; refuse the rest by name.
            return (
                obs,
                float(require_step_scalar(reward, field="reward", world=self.scene_name)),
                bool(require_step_scalar(terminated, field="terminated", world=self.scene_name)),
                bool(require_step_scalar(truncated, field="truncated", world=self.scene_name)),
                info if isinstance(info, dict) else {},
            )

        insertion = 0.0
        if isinstance(action, (int, float)):
            insertion = float(action)
        elif isinstance(action, dict):
            raw_ins = action.get("insertion_step_mm")
            if raw_ins is None:
                raw_ins = action.get("insertion")
            insertion = float(raw_ins) if raw_ins is not None else 1.0
        # Step budget comes from the harness, not from `environment.parameters`:
        # that dict is forwarded verbatim to a real engine's constructor, so a
        # magic `max_steps` key there is both an invalid kwarg for most envs and
        # a second source of truth for a limit the harness already owns.
        max_steps = self.max_steps
        terminated = self._step_count >= max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0

        # Step bookkeeping only: `raw_success` is the stand-in's own progress
        # counter and is honest about being that. `max_pen`, `wall_force_n`,
        # `tissue_deformation_energy`, and `safe_success` are deliberately
        # absent - penetration and wall force are exactly what an unsolved
        # scene cannot know, and a gated task abstaining is the correct result.
        obs = {
            "insertion_command": insertion,
            "beam_elements": 20,
        }
        info = {
            "step": self._step_count,
            "raw_success": terminated,
            "backend": BACKEND_SYNTHETIC_STUB,
        }
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of underlying SOFA FEA nodes and tissue stresses."""
        if self._env is not None and hasattr(self._env, "get_state"):
            return self._env.get_state()  # type: ignore[no-any-return]
        return {
            "scene": self.scene_name,
            "step_count": self._step_count,
            "backend": BACKEND_SYNTHETIC_STUB,
        }

    def close(self) -> None:
        """Release SOFA simulation context."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def _acquire_sofa_env(task: TaskSpec) -> tuple[Any, str]:
    """Acquire a real SOFA-backed scene: returns ``(env, version)``.

    "Best-effort" only covers a runtime that is *absent*. A scene that is
    registered and still fails to build is a task configuration failure and is
    refused, because the old bare ``except Exception`` made the two look alike:
    a task with ``synthetic_stub = true`` then measured stand-in numbers under a
    real scene's name, and SOFA is the one wrap target that can host a real hard
    gate today, so the substitution mattered most here.

    ``Sofa`` / ``SofaRuntime`` are imported only to probe the installed version;
    the scene itself comes from ``gymnasium.make``, because SofaGym registers its
    scenes as gymnasium envs. An unknown scene id therefore surfaces as
    gymnasium's own registration error and needs no SOFA-specific signal.
    """
    try:
        import Sofa
        import SofaRuntime
    except ImportError:
        return None, ""
    detected = module_distribution_version("Sofa")
    if not detected:
        detected = str(
            getattr(Sofa, "__version__", "") or getattr(SofaRuntime, "__version__", "") or ""
        )
    scene_id = task.environment.gym_id
    if not scene_id:
        return None, detected
    try:
        import gymnasium
        import sofagym  # noqa: F401
    except ImportError:
        return None, detected
    kwargs: dict[str, Any] = dict(task.environment.parameters)
    # The harness owns the step limit; a stray `max_steps` here is an unexpected
    # keyword for the scene constructor, and swallowing that error is how a
    # correctly-pinned real scene became "no backend" and then a silent stub.
    kwargs.pop("max_steps", None)
    try:
        return gymnasium.make(scene_id, **kwargs), detected
    except missing_world_errors(gymnasium):
        # The scene is genuinely not registered here: an honest no-backend
        # condition, which the constructor's synthetic-stub refusal then judges.
        return None, detected
    except Exception as exc:
        raise TaskContractError(refuse_unbuildable_world("SOFA / SofaGym", scene_id, exc)) from exc


def make_sofa_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating a SofaBridge for a task, preferring a real SOFA scene."""
    sofa_env, backend_version = _acquire_sofa_env(task)
    return SofaBridge(
        scene_name=task.environment.gym_id or task.id,
        parameters=dict(task.environment.parameters),
        world_pin=task.environment.world_pin,
        sofa_env=sofa_env,
        allow_synthetic=task.environment.synthetic_stub,
        backend_version=backend_version,
        max_steps=task.harness.max_steps,
    )
