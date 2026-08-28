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
* each allowed key is *reduced* to that canonical scalar by a declared,
  alias-specific rule, never renamed: Isaac reports contact as a per-body
  ``(num_bodies, 3)`` force array, and publishing that array under
  ``contact_force_n`` left a ``contact_force_n > 1.5`` gate unable to compare a
  list to a float, i.e. silently ``not_assessable`` on a real 2 N contact. A
  shape with no declared reduction is refused, not published;
* the world is single-environment. ``num_envs != 1`` is refused up front and a
  batched step return is refused rather than coerced, because no reduction of a
  batch is a step any single environment took;
* every step stamps ``safety_state_reported`` so the conformance suite's
  gate-state availability check can tell a real mapping from a silent gap, and
  ``safety_state_reductions`` so a published scalar can be traced to the engine
  key and rule behind it;
* a real backend must match the task's pinned Isaac revision before the first
  step, because Isaac version churn is the top wrap risk for this shelf.

Isaac Sim itself is not redistributable (see ``surgeval worlds install`` and the
``vendor-runtime`` install strategy), so a synthetic stand-in exists for
headless CI. It is refused unless the task opts in and it is stamped into every
artifact it touches.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.sim.base import (
    BACKEND_REAL,
    BACKEND_SYNTHETIC_STUB,
    BaseSimulationBridge,
    SimulationEngine,
    batch_rows,
    missing_world_errors,
    module_distribution_version,
    refuse_unbuildable_world,
    require_single_env,
    require_step_scalar,
)
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import world_kind_key

_ISAAC_MODULES = "'isaaclab' / 'isaacsim'"

#: Declared reductions from an engine key's real shape to the scalar a gate
#: binds to. ``max`` takes the largest of already-scalar per-body magnitudes;
#: ``max-norm`` takes each per-body vector's Euclidean norm and then the largest
#: of those; ``any`` reads a per-body/per-joint flag array as violated when any
#: entry is. Nothing else is reduced, because a reduction nobody declared is a
#: number nobody can defend.
REDUCE_MAX = "max"
REDUCE_MAX_NORM = "max-norm"
REDUCE_ANY = "any"


@dataclass(frozen=True)
class SafetySignal:
    """A canonical safety name, the unit it publishes, and the keys feeding it.

    ``sources`` is ordered: the first alias the engine actually reports wins.
    The reduction hangs off the *alias*, not the canonical name, because the
    shape does: Isaac's ``net_contact_force`` is a per-body 3-vector array while
    ``contact_force_n`` is already a scalar in newtons, and norming the second
    or maxing the first would publish a number the engine never measured.
    """

    unit: str
    sources: tuple[tuple[str, str], ...]


#: Engine state keys the adapter is allowed to lift into a canonical safety
#: name. Nothing outside this mapping becomes safety evidence, and the scalar
#: reward is deliberately absent from it.
SAFETY_SIGNALS: dict[str, SafetySignal] = {
    "max_pen": SafetySignal(
        "world-unit",
        (
            ("max_pen", REDUCE_MAX),
            ("penetration_depth", REDUCE_MAX),
            ("max_penetration", REDUCE_MAX),
            ("penetration", REDUCE_MAX),
        ),
    ),
    "contact_force_n": SafetySignal(
        "N",
        (
            ("contact_force_n", REDUCE_MAX),
            ("contact_force", REDUCE_MAX_NORM),
            ("max_contact_force", REDUCE_MAX),
            ("net_contact_force", REDUCE_MAX_NORM),
        ),
    ),
    "wall_force_n": SafetySignal(
        "N",
        (
            ("wall_force_n", REDUCE_MAX),
            ("wall_force", REDUCE_MAX_NORM),
            ("tissue_force_n", REDUCE_MAX),
        ),
    ),
    "workspace_violation": SafetySignal(
        "",
        (
            ("workspace_violation", REDUCE_ANY),
            ("out_of_workspace", REDUCE_ANY),
            ("joint_limit_violation", REDUCE_ANY),
        ),
    ),
    "unsafe": SafetySignal("", (("unsafe", REDUCE_ANY), ("safety_violation", REDUCE_ANY))),
    "diverged": SafetySignal("", (("diverged", REDUCE_ANY), ("physics_diverged", REDUCE_ANY))),
}

#: Alias precedence per canonical name, derived so the allowlist has exactly one
#: definition.
SAFETY_KEY_SOURCES: dict[str, tuple[str, ...]] = {
    canonical: tuple(alias for alias, _ in signal.sources)
    for canonical, signal in SAFETY_SIGNALS.items()
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


def _describe_shape(value: Any) -> str:
    dims: list[int] = []
    current = value
    while (rows := batch_rows(current)) is not None:
        dims.append(len(rows))
        if not rows:
            break
        current = rows[0]
    return (
        f"shape ({', '.join(str(dim) for dim in dims)})"
        if dims
        else f"scalar type {type(value).__name__}"
    )


def _refuse_safety_shape(*, alias: str, canonical: str, unit: str, detail: str, value: Any) -> str:
    target = f"{canonical!r}" + (f" in {unit}" if unit else "")
    return (
        f"Isaac engine key {alias!r} has {_describe_shape(value)} and cannot be "
        f"reduced to the scalar safety signal {target}: {detail}. Renaming the raw "
        f"value onto {canonical!r} anyway would leave every gate bound to it "
        "not_assessable, so the gate would silently fail to detect the condition it "
        "exists to score. Report an already-reduced scalar, or a supported per-body "
        "shape."
    )


def _safety_float(value: Any, *, alias: str, canonical: str, unit: str) -> float:
    """Coerce an already-scalar engine value to ``float``, refusing non-numbers."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TaskContractError(
            _refuse_safety_shape(
                alias=alias,
                canonical=canonical,
                unit=unit,
                value=value,
                detail=f"it is not a number ({exc})",
            )
        ) from exc


def _reduce_max(value: Any, *, alias: str, canonical: str, unit: str) -> float:
    """Largest of already-scalar per-body magnitudes."""
    rows = batch_rows(value)
    if rows is None:
        return _safety_float(value, alias=alias, canonical=canonical, unit=unit)
    if not rows:
        raise TaskContractError(
            _refuse_safety_shape(
                alias=alias,
                canonical=canonical,
                unit=unit,
                value=value,
                detail="it is empty, so there is nothing to reduce",
            )
        )
    if len(rows) == 1 and batch_rows(rows[0]) is not None:
        # A length-1 outer dimension is a single-env batch: unwrapping it and
        # maxing the inside give the same answer either way it is read.
        return _reduce_max(rows[0], alias=alias, canonical=canonical, unit=unit)
    if any(batch_rows(row) is not None for row in rows):
        raise TaskContractError(
            _refuse_safety_shape(
                alias=alias,
                canonical=canonical,
                unit=unit,
                value=value,
                detail=(
                    f"{alias!r} declares already-scalar magnitudes, so a multi-entry "
                    "nested batch of them has no single-environment reading"
                ),
            )
        )
    return max(_safety_float(row, alias=alias, canonical=canonical, unit=unit) for row in rows)


def _reduce_max_norm(value: Any, *, alias: str, canonical: str, unit: str) -> float:
    """Per-body Euclidean norm, then the largest of those across bodies."""
    rows = batch_rows(value)
    if rows is None:
        # A bare number means the engine already did the reduction itself.
        return _safety_float(value, alias=alias, canonical=canonical, unit=unit)

    def refuse(detail: str) -> TaskContractError:
        return TaskContractError(
            _refuse_safety_shape(
                alias=alias, canonical=canonical, unit=unit, value=value, detail=detail
            )
        )

    if not rows:
        raise refuse("it is empty, so there is nothing to reduce")
    nested = [row for row in rows if batch_rows(row) is not None]
    if not nested:
        # A flat numeric run is one body's force vector. Only a 3-vector is
        # unambiguous: any other length reads equally well as per-body
        # magnitudes, and norming those would inflate the published force.
        if len(rows) != 3:
            raise refuse(
                f"a flat run of {len(rows)} numbers is ambiguous between one force "
                "vector and per-body magnitudes"
            )
        return math.hypot(
            *(_safety_float(part, alias=alias, canonical=canonical, unit=unit) for part in rows)
        )
    if len(nested) != len(rows):
        raise refuse("it mixes numbers and nested rows, so no per-body reading is defined")
    if any(batch_rows(part) is not None for part in batch_rows(rows[0]) or []):
        # Three levels deep is (num_envs, num_bodies, 3).
        if len(rows) != 1:
            raise refuse(
                "it is a multi-environment batch of per-body forces, and this engine "
                "represents exactly one environment"
            )
        return _reduce_max_norm(rows[0], alias=alias, canonical=canonical, unit=unit)
    norms: list[float] = []
    for row in rows:
        parts = batch_rows(row) or []
        if len(parts) != 3:
            raise refuse(f"a per-body force row must be a 3-vector, got {_describe_shape(row)}")
        norms.append(
            math.hypot(
                *(
                    _safety_float(part, alias=alias, canonical=canonical, unit=unit)
                    for part in parts
                )
            )
        )
    return max(norms)


def _reduce_any(value: Any, *, alias: str, canonical: str, unit: str) -> bool:
    """A per-body/per-joint flag array counts as violated when any entry is."""
    rows = batch_rows(value)
    if rows is None:
        return bool(_safety_float(value, alias=alias, canonical=canonical, unit=unit))
    if not rows:
        raise TaskContractError(
            _refuse_safety_shape(
                alias=alias,
                canonical=canonical,
                unit=unit,
                value=value,
                detail="it is empty, so there is nothing to reduce",
            )
        )
    if len(rows) == 1 and batch_rows(rows[0]) is not None:
        return _reduce_any(rows[0], alias=alias, canonical=canonical, unit=unit)
    if any(batch_rows(row) is not None for row in rows):
        raise TaskContractError(
            _refuse_safety_shape(
                alias=alias,
                canonical=canonical,
                unit=unit,
                value=value,
                detail=(
                    "it is a multi-environment batch of flags, and this engine "
                    "represents exactly one environment"
                ),
            )
        )
    return any(
        bool(_safety_float(row, alias=alias, canonical=canonical, unit=unit)) for row in rows
    )


_REDUCERS: dict[str, Callable[..., float | bool]] = {
    REDUCE_MAX: _reduce_max,
    REDUCE_MAX_NORM: _reduce_max_norm,
    REDUCE_ANY: _reduce_any,
}


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
        self.num_envs = require_single_env(
            self.parameters.get("num_envs", 1), world=env_id or "(unnamed)"
        )
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
        """Reduce declared safety state out of engine info without inventing any.

        Only keys in :data:`SAFETY_SIGNALS` are read, the first reported alias
        wins, and nothing is defaulted: an engine that reports no safety state
        yields ``{"safety_state_reported": False}`` and gates bound to those
        names abstain.

        Each alias is *reduced* by its declared rule, never merely renamed. Isaac
        reports contact as a per-body ``(num_bodies, 3)`` force array, and copying
        that under ``contact_force_n`` used to leave ``contact_force_n > 1.5``
        unable to compare a list to a float — the gate scored not_assessable and
        missed the contact it was written to catch. A shape with no declared
        reduction is refused here rather than published under a scalar name.
        """
        projected: dict[str, Any] = {}
        reductions: dict[str, str] = {}
        for canonical, signal in SAFETY_SIGNALS.items():
            for alias, reduction in signal.sources:
                if alias in engine_info:
                    projected[canonical] = _REDUCERS[reduction](
                        engine_info[alias],
                        alias=alias,
                        canonical=canonical,
                        unit=signal.unit,
                    )
                    reductions[canonical] = f"{alias}:{reduction}"
                    break
        reported = sorted(projected)
        projected["safety_state_reported"] = bool(reported)
        projected["safety_state_keys"] = reported
        # How each published scalar was derived, so a FAIL can be traced back to
        # the engine key and the reduction that produced the number.
        projected["safety_state_reductions"] = reductions
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
            # A batched return is refused, not coerced: `float()` on a two-env
            # reward vector raised, and on a sliceable one it would have
            # published env 0's step as the step every env took.
            reward_value = float(require_step_scalar(reward, field="reward", world=self.env_id))
            info = self._compose_info(
                engine_info if isinstance(engine_info, dict) else {},
                reward=reward_value,
            )
            info["step"] = self._step_count
            return (
                obs,
                reward_value,
                bool(require_step_scalar(terminated, field="terminated", world=self.env_id)),
                bool(require_step_scalar(truncated, field="truncated", world=self.env_id)),
                info,
            )
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
    """Acquire a real Isaac Lab env: returns ``(env, revision)``.

    The pin check runs only when a real env is actually obtained: an absent
    Isaac install is a missing-backend condition (handled by the synthetic-stub
    refusal), not a pin violation. A world that *is* registered and still fails
    to build is neither - it is a task configuration failure, and reporting it
    as "no backend" would let a stub opt-in silently substitute stand-in numbers
    for the real world the task names.
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
    except missing_world_errors(gymnasium):
        return None, isaac_revision()
    except Exception as exc:
        raise TaskContractError(refuse_unbuildable_world("Isaac Lab", env_id, exc)) from exc
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
