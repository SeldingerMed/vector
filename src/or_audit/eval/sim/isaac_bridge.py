"""Isaac Lab world bridge (N2): ORBIT-Surgical-class worlds under the vector contract.

Isaac Lab is a *supplier* of worlds, not an evaluation contract: its envs report
a scalar task reward, and the safety-relevant physics (contact force,
penetration depth, workspace violation) lives in observation/state fields that a
reward-shaped wrapper silently discards. This adapter enforces the raw/safe
split at the seam:

* the engine's scalar reward is carried through as ``raw_reward`` and is never
  copied into a safety field;
* safety state is lifted from an explicit allowlist of engine keys into the
  canonical ``info`` names gates bind to, and *only* from keys the engine
  actually reported — an absent signal stays absent, so a gate abstains
  (``not_assessable``) instead of passing on a fabricated value;
* every step stamps ``safety_state_reported`` so the conformance suite's
  gate-state availability check can tell a real mapping from a silent gap;
* a real backend must match the task's pinned Isaac revision before the first
  step, because Isaac version churn is the top wrap risk for this shelf.

Isaac Sim itself is not redistributable (see ``surgeval worlds install`` and the
``vendor-runtime`` install strategy), so a synthetic stand-in exists for
headless CI. It is refused unless the task opts in and it is stamped into every
artifact it touches.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
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

_ISAAC_MODULES = "'isaaclab' / 'isaacsim'"

#: Engine state keys the adapter is allowed to lift into a canonical safety
#: name. Nothing outside this mapping becomes safety evidence, and the scalar
#: reward is deliberately absent from it.
SAFETY_KEY_SOURCES: dict[str, tuple[str, ...]] = {
    "max_pen": ("max_pen", "penetration_depth", "max_penetration", "penetration"),
    "contact_force_n": (
        "contact_force_n",
        "contact_force",
        "max_contact_force",
        "net_contact_force",
    ),
    "wall_force_n": ("wall_force_n", "wall_force", "tissue_force_n"),
    "workspace_violation": ("workspace_violation", "out_of_workspace", "joint_limit_violation"),
    "unsafe": ("unsafe", "safety_violation"),
    "diverged": ("diverged", "physics_diverged"),
}

#: Outcome keys the adapter passes through when the engine reports them. These
#: are task metrics, not gates: a missing one is an unassessable metric.
OUTCOME_KEYS: tuple[str, ...] = ("raw_success", "safe_success", "success", "is_success")


def _refuse_synthetic_isaac(kind: str, env_id: str) -> str:
    return (
        f"world kind {kind!r} has no Isaac Lab backend attached: the Isaac modules "
        f"({_ISAAC_MODULES}) did not yield a runnable env for {env_id!r}. A synthetic "
        "stand-in would invent contact forces, penetration depth, and workspace "
        "violations that this task's hard safety gates would then score as physical "
        "evidence. Install the vendor runtime (`surgeval worlds install "
        "orbit-surgical`), or set environment.synthetic_stub = true in task.toml to "
        'accept a non-physical stand-in (artifacts are stamped backend="synthetic-stub" '
        "and export-rl refuses the run)."
    )


def isaac_revision() -> str:
    """Installed Isaac Lab revision: VCS commit if present, else distribution version."""
    try:
        metadata = distribution("isaaclab").read_text("direct_url.json")
    except PackageNotFoundError:
        return module_distribution_version("isaaclab")
    if metadata:
        try:
            commit = json.loads(metadata)["vcs_info"]["commit_id"]
        except (KeyError, TypeError, json.JSONDecodeError):
            commit = ""
        if commit:
            return str(commit)
    return module_distribution_version("isaaclab")


def assert_isaac_pin(expected: str) -> str:
    """Require the installed Isaac Lab revision to match a task's ``world_pin``.

    Returns the detected revision. Raises when the task pins a revision and the
    installed one differs: an Isaac API break between revisions changes the
    physics a gate is scoring, so a mismatched pin is a contract error, not a
    warning.
    """
    detected = isaac_revision()
    if not expected:
        raise TaskContractError(
            "an Isaac Lab world must pin the engine revision in environment.world_pin; "
            "Isaac version churn changes the physics behind every gate"
        )
    if not detected:
        raise TaskContractError(
            "isaaclab is importable but reports no distribution version or VCS commit, "
            f"so the task pin {expected!r} cannot be verified"
        )
    if detected != expected:
        raise TaskContractError(
            f"Isaac Lab pin mismatch: task requires {expected}, installed {detected}"
        )
    return detected


class IsaacBridge(BaseSimulationBridge):
    """Bridge for Isaac Lab / ORBIT-Surgical worlds with an enforced raw/safe split."""

    world_kind: WorldKind | str = WorldKind.ISAAC_LAB

    def __init__(
        self,
        env_id: str,
        *,
        parameters: dict[str, Any] | None = None,
        world_pin: str = "",
        isaac_env: Any = None,
        allow_synthetic: bool = False,
        backend_version: str = "",
        world_kind: WorldKind | str | None = None,
        max_steps: int = 100,
    ) -> None:
        kind = world_kind if world_kind is not None else self.world_kind
        if isaac_env is None and not allow_synthetic:
            raise TaskContractError(
                _refuse_synthetic_isaac(world_kind_key(kind), env_id or "(unnamed)")
            )
        self.world_kind = kind
        self.env_id = env_id
        self.parameters = dict(parameters or {})
        self.world_pin = world_pin
        self.num_envs = int(self.parameters.get("num_envs", 1))
        self._env = isaac_env
        self._backend_version = backend_version
        #: Harness step limit, passed by the factory from ``task.harness.max_steps``.
        #: Only the synthetic stand-in consults it; a real engine terminates itself.
        self.max_steps = max_steps

        self._step_count = 0

    @property
    def synthetic(self) -> bool:
        """Whether this bridge is serving a non-physical stand-in."""
        return self._env is None

    def engine_provenance(self) -> dict[str, Any]:
        """Report whether a real Isaac Lab env or the synthetic stand-in produced data."""
        return {
            "engine": world_kind_key(self.world_kind),
            "backend": BACKEND_SYNTHETIC_STUB if self.synthetic else BACKEND_REAL,
            "backend_version": "" if self.synthetic else self._backend_version,
            "world_pin": self.world_pin,
        }

    def safety_projection(self, engine_info: dict[str, Any]) -> dict[str, Any]:
        """Lift declared safety state out of engine info without inventing any.

        Only keys in :data:`SAFETY_KEY_SOURCES` are read, the first reported
        alias wins, and nothing is defaulted: an engine that reports no safety
        state yields ``{"safety_state_reported": False}`` and gates bound to
        those names abstain.
        """
        projected: dict[str, Any] = {}
        for canonical, aliases in SAFETY_KEY_SOURCES.items():
            for alias in aliases:
                if alias in engine_info:
                    projected[canonical] = engine_info[alias]
                    break
        reported = sorted(projected)
        projected["safety_state_reported"] = bool(reported)
        projected["safety_state_keys"] = reported
        return projected

    def _compose_info(self, engine_info: dict[str, Any], *, reward: float | None) -> dict[str, Any]:
        info = dict(engine_info)
        if reward is not None:
            info["raw_reward"] = float(reward)
        for outcome in OUTCOME_KEYS:
            if outcome in engine_info:
                info[outcome] = engine_info[outcome]
        info.update(self.safety_projection(engine_info))
        info["backend"] = BACKEND_SYNTHETIC_STUB if self.synthetic else BACKEND_REAL
        info["world_pin"] = self.world_pin
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the Isaac Lab env (or the stand-in) and project its safety state."""
        self._step_count = 0
        if self._env is not None:
            obs, engine_info = self._env.reset(seed=seed, options=options)
            info = self._compose_info(
                engine_info if isinstance(engine_info, dict) else {}, reward=None
            )
            info["seed"] = seed
            return obs, info
        obs = {
            "isaac_env": self.env_id,
            "num_envs": self.num_envs,
            "robot_joint_pos": [0.0] * 7,
            "tool_ee_pos": [0.0, 0.0, 0.0],
            "object_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        # A stand-in reports that it started and nothing else. Emitting
        # `max_pen`/`contact_force_n`/`unsafe` here would manufacture physical
        # evidence out of an `if` statement: gates would resolve pass against a
        # world that never touched anything. With the keys absent, a gated task
        # abstains - which is the honest outcome and the one `abstain_ok`
        # exists for.
        info = self._compose_info(
            {
                "isaac_initialized": True,
                "device": str(self.parameters.get("device", "cuda:0")),
            },
            reward=None,
        )
        info["seed"] = seed
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the env, keeping the scalar reward strictly out of the safety fields."""
        self._step_count += 1
        if self._env is not None:
            obs, reward, terminated, truncated, engine_info = self._env.step(action)
            info = self._compose_info(
                engine_info if isinstance(engine_info, dict) else {},
                reward=float(reward),
            )
            info["step"] = self._step_count
            return obs, float(reward), bool(terminated), bool(truncated), info
        # Step budget comes from the harness, not from `environment.parameters`:
        # that dict is forwarded verbatim to a real engine's constructor, so a
        # magic `max_steps` key there is both an invalid kwarg for most envs and
        # a second source of truth for a limit the harness already owns.
        max_steps = self.max_steps
        terminated = self._step_count >= max_steps
        reward = 1.0 if terminated else 0.0
        obs = {
            "isaac_env": self.env_id,
            "num_envs": self.num_envs,
            "robot_joint_pos": [0.01 * self._step_count] * 7,
            "tool_ee_pos": [0.01 * self._step_count, 0.0, 0.0],
            "object_pose": [0.0, 0.0, 0.01 * self._step_count, 1.0, 0.0, 0.0, 0.0],
        }
        # Task *progress* is the stand-in's own bookkeeping and may be reported;
        # `safe_success` may not, because safety is exactly what a stand-in
        # cannot observe. No physical safety key is synthesized.
        info = self._compose_info(
            {"raw_success": terminated},
            reward=reward,
        )
        info["step"] = self._step_count
        return obs, reward, terminated, False, info

    def get_state(self) -> dict[str, Any]:
        """Return the engine's state snapshot, or the stand-in's step bookkeeping."""
        if self._env is not None and hasattr(self._env, "get_state"):
            return self._env.get_state()  # type: ignore[no-any-return]
        return {
            "env_id": self.env_id,
            "step_count": self._step_count,
            "num_envs": self.num_envs,
            "backend": BACKEND_SYNTHETIC_STUB if self.synthetic else BACKEND_REAL,
        }

    def close(self) -> None:
        """Release the Isaac simulation app and GPU context."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def _acquire_isaac_env(task: TaskSpec) -> tuple[Any, str]:
    """Best-effort acquisition of a real Isaac Lab env: returns ``(env, revision)``.

    The pin check runs only when a real env is actually obtained: an absent
    Isaac install is a missing-backend condition (handled by the synthetic-stub
    refusal), not a pin violation.
    """
    env_id = task.environment.gym_id
    if not env_id:
        return None, ""
    try:
        import gymnasium
        import isaaclab  # noqa: F401
    except ImportError:
        return None, ""
    kwargs: dict[str, Any] = dict(task.environment.parameters)
    kwargs.pop("max_steps", None)
    try:
        env = gymnasium.make(env_id, **kwargs)
    except Exception:  # Isaac Lab raises engine-specific errors for an unknown env id
        return None, isaac_revision()
    revision = assert_isaac_pin(task.environment.world_pin)
    return env, revision


def make_isaac_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating an :class:`IsaacBridge`, preferring a real Isaac Lab env."""
    isaac_env, revision = _acquire_isaac_env(task)
    return IsaacBridge(
        env_id=task.environment.gym_id or task.id,
        parameters=dict(task.environment.parameters),
        world_pin=task.environment.world_pin,
        isaac_env=isaac_env,
        allow_synthetic=task.environment.synthetic_stub,
        backend_version=revision,
        world_kind=task.environment.kind,
        max_steps=task.harness.max_steps,
    )
