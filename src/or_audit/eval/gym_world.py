"""Gymnasium-shaped worlds for gym-policy tasks.

Lumen is optional. CI injects a factory. A published row still requires a
world pin; Newton is not a default dependency.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, Protocol, cast

import numpy as np

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import PerturbationSpec
from or_audit.eval.enums import WorldKind
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import world_kind_key

GymFactory = Callable[[TaskSpec], "GymEnv"]

HARNESS_PERTURBATION_KINDS = frozenset(
    {"harness-observation-zero", "harness-observation-gaussian-noise", "harness-action-hold"}
)


class GymEnv(Protocol):
    """Reset/step world. Lumen's NavEnv matches this without subclassing Gymnasium."""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Start an episode."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """One transition: obs, reward, terminated, truncated, info."""


def make_gym(task: TaskSpec) -> GymEnv:
    """Build the world named by ``task.environment``.

    Raises:
        TaskContractError: If Lumen/gymnasium is missing or the gym id is unknown.
    """
    kind = task.environment.kind
    if kind is WorldKind.LUMEN_GYM:
        return _make_lumen(
            task.environment.gym_id,
            world_pin=task.environment.world_pin,
            parameters=task.environment.parameters,
        )
    if kind is WorldKind.GYM:
        return _make_gymnasium(task.environment.gym_id, parameters=task.environment.parameters)
    msg = f"task {task.id} world kind {world_kind_key(kind)} is not a gym-policy world"
    raise TaskContractError(msg)


def _make_lumen(
    gym_id: str,
    *,
    world_pin: str,
    parameters: dict[str, bool | int | float | str],
) -> GymEnv:
    try:
        from lumen.envs.registration import LUMEN_ENVS, register_gym_envs
    except ImportError as exc:
        msg = (
            "this task requires Lumen. Install seldinger-lumen and Newton to "
            "run gym-policy evals (BUILD.md P1). Default CI does not import them."
        )
        raise TaskContractError(msg) from exc
    _require_lumen_pin(world_pin)
    if gym_id not in LUMEN_ENVS:
        known = ", ".join(sorted(LUMEN_ENVS))
        msg = f"unknown Lumen gym_id {gym_id!r}; known: {known}"
        raise TaskContractError(msg)
    try:
        import gymnasium

        register_gym_envs()
        kwargs: dict[str, Any] = dict(parameters)
        env = cast(GymEnv, gymnasium.make(gym_id, **kwargs))
    except ImportError:
        factory: Callable[..., Any] = LUMEN_ENVS[gym_id]
        env = cast(GymEnv, factory(**parameters))
    return env


def _require_lumen_pin(expected: str) -> None:
    """Require the installed VCS package to match the task's immutable world pin."""
    try:
        metadata = distribution("seldinger-lumen").read_text("direct_url.json")
    except PackageNotFoundError as exc:
        raise TaskContractError("seldinger-lumen distribution metadata is missing") from exc
    if not metadata:
        raise TaskContractError("seldinger-lumen is not installed from a pinned VCS revision")
    try:
        actual = json.loads(metadata)["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TaskContractError("seldinger-lumen direct_url.json has no VCS commit") from exc
    if actual != expected:
        raise TaskContractError(f"Lumen pin mismatch: task requires {expected}, installed {actual}")


def _make_gymnasium(gym_id: str, *, parameters: dict[str, bool | int | float | str]) -> GymEnv:
    try:
        import gymnasium
    except ImportError as exc:
        msg = (
            f"task names gym {gym_id!r} but gymnasium is not installed; "
            f"pip install 'or-audit[lumen]' (gymnasium only) or the env's extra"
        )
        raise TaskContractError(msg) from exc
    kwargs: dict[str, Any] = dict(parameters)
    return cast(GymEnv, gymnasium.make(gym_id, **kwargs))


def sample_action(env: GymEnv, *, seed: int, step: int) -> Any:
    """Deterministic random action for a ``kind=random`` agent.

    Discrete spaces are handled before the ``low``/``high`` box path: they
    expose ``n`` rather than bounds, and a real ``Discrete`` env indexes its
    transition table with the action, so handing it a float array raises
    ``TypeError: unhashable type: 'numpy.ndarray'`` inside the env. The seeded
    ``default_rng`` is kept rather than ``space.sample()`` because the harness
    owns reproducibility here; ``space.sample()`` would draw from the env's own
    RNG and make the action depend on env internals instead of the trial seed.
    """
    rng = np.random.default_rng(seed * 1_000_003 + step)
    # The runner hands us the *bridge*, which does not forward `action_space`;
    # without unwrapping, every space looked unbounded and every action became a
    # 2-vector of floats - silently wrong for Box envs and a hard TypeError for
    # Discrete ones.
    space = getattr(env, "action_space", None)
    if space is None:
        inner = getattr(env, "unwrapped", None)
        space = getattr(inner, "action_space", None)
    n = getattr(space, "n", None)
    if n is not None:
        return int(rng.integers(0, int(n)))
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is not None and high is not None:
        return rng.uniform(np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
    return rng.uniform(-1.0, 1.0, size=2)


def split_perturbations(
    perturbations: tuple[PerturbationSpec, ...],
) -> tuple[tuple[PerturbationSpec, ...], tuple[PerturbationSpec, ...]]:
    """Separate portable harness faults from world-native perturbations."""
    unknown = sorted(
        {item.kind for item in perturbations if item.kind.startswith("harness-")}
        - HARNESS_PERTURBATION_KINDS
    )
    if unknown:
        raise TaskContractError(f"unsupported harness perturbation kinds: {unknown}")
    harness = tuple(item for item in perturbations if item.kind in HARNESS_PERTURBATION_KINDS)
    world = tuple(item for item in perturbations if item.kind not in HARNESS_PERTURBATION_KINDS)
    return harness, world


def _numeric_array(value: Any, *, label: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise TaskContractError(f"{label} requires a finite numeric array")
    return array


def _apply_harness_observation(
    observation: Any,
    perturbations: tuple[PerturbationSpec, ...],
    *,
    seed: int,
    step: int,
) -> Any:
    result = observation
    for index, perturbation in enumerate(perturbations):
        if perturbation.kind == "harness-observation-zero":
            result = np.zeros_like(_numeric_array(result, label=perturbation.kind))
        elif perturbation.kind == "harness-observation-gaussian-noise":
            std = perturbation.parameters.get("std")
            if (
                isinstance(std, bool)
                or not isinstance(std, int | float)
                or not np.isfinite(std)
                or std <= 0
            ):
                raise TaskContractError(f"{perturbation.kind} requires parameters.std > 0")
            array = _numeric_array(result, label=perturbation.kind)
            noise = np.random.default_rng([seed, step, index]).normal(0.0, float(std), array.shape)
            result = array + noise
    return result


#: Prefix marking a float the engine reported as ``nan``/``+inf``/``-inf``.
#: Deliberately not a number: nothing downstream may coerce it back into one,
#: so a gate bound to it abstains (``float(...)`` and ``x > threshold`` both
#: raise, which ``gate_dsl`` maps to ``not_assessable``) instead of reading a
#: value the engine never reported.
NONFINITE_TAG = "__nonfinite__:"


def nonfinite_tag(value: float) -> str:
    """Canonical tag for a non-finite float, distinct per kind of divergence."""
    if value != value:  # NaN is the only value unequal to itself.
        return f"{NONFINITE_TAG}nan"
    return f"{NONFINITE_TAG}{'-inf' if value < 0 else '+inf'}"


def nonfinite_kind(value: Any) -> str:
    """Name the divergence a value reports, ``""`` when it is a finite number.

    Two routes must reach the same verdict wherever a signal or metric is read.
    The recorder tags a non-finite float as a string so the divergence survives
    into the trajectory and its digest instead of being flattened to ``0.0``; a
    value read straight out of raw engine output or a ``task://`` JSON artifact
    is still a real ``float('nan')``, because ``json.loads`` accepts the
    ``NaN``/``Infinity`` literals.
    """
    if isinstance(value, str) and value.startswith(NONFINITE_TAG):
        return value[len(NONFINITE_TAG) :] or "non-finite"
    if isinstance(value, float) and not np.isfinite(value):
        return nonfinite_tag(value)[len(NONFINITE_TAG) :]
    return ""


def jsonable(value: Any) -> Any:
    """Convert numpy values so a trajectory can be JSON and canonical.

    Non-finite floats become distinct tagged strings rather than ``0.0``. They
    were previously flattened, which was the fabrication the generated
    verifiers explicitly refuse one layer up: "a signal the engine did not
    report is emitted as None, never 0.0 or False: a defaulted safety number is
    a fabricated one". A diverged solver reporting ``nan`` contact force was
    written into the trajectory as ``0.0`` newtons - the safest possible
    reading, invented here, then hashed into the head as evidence. It also
    erased the divergence signal from the determinism measurement: ``nan`` and
    ``+inf`` normalized to the same bytes, so two runs whose physics blew up
    differently digested identically and measured ``bitwise``.

    ``nan`` is JSON-illegal, so a tag is required either way; the choice made
    here is which tag, and a string is one no consumer can silently average.
    """
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not np.isfinite(item):
            return nonfinite_tag(item)
        return item
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return nonfinite_tag(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def run_gym_episode(
    env: GymEnv,
    *,
    seed: int,
    action_fn: Callable[[GymEnv, Any, int], Any],
    reset_options: dict[str, Any] | None = None,
    harness_perturbations: tuple[PerturbationSpec, ...] = (),
    max_steps: int = 10_000,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Roll out one episode. Returns final info and the trajectory.

    The returned ``info`` is canonicalized through :func:`jsonable`, the same
    bytes the trajectory carries. Live scoring and ``reconstitute`` must decide
    a gate on the same values or the replay check compares two different
    episodes; returning raw engine output here meant a diverged ``nan`` was
    scored live and ``0.0`` on replay, and both happened to pass.
    """
    obs, reset_info = env.reset(seed=seed, options=reset_options)
    if reset_options is not None:
        expected = reset_options.get("or_audit")
        acknowledged = reset_info.get("or_audit") if isinstance(reset_info, dict) else None
        if acknowledged != expected:
            raise TaskContractError(
                "gym ignored or changed the declared scenario/perturbation contract; "
                "reset info must echo options['or_audit'] exactly"
            )
    steps: list[dict[str, Any]] = []
    info: Any = {}
    previous_action: Any = None
    for step_i in range(max_steps):
        active = tuple(
            item
            for item in harness_perturbations
            if item.at_step == step_i or (item.at_step is None and step_i == 0)
        )
        observation_events = tuple(item for item in active if "observation" in item.kind)
        policy_obs = _apply_harness_observation(obs, observation_events, seed=seed, step=step_i)
        action = action_fn(env, policy_obs, step_i)
        hold_action = any(item.kind == "harness-action-hold" for item in active)
        applied_action = previous_action if hold_action and previous_action is not None else action
        if hold_action and previous_action is None:
            applied_action = np.zeros_like(_numeric_array(action, label="harness-action-hold"))
        obs, reward, terminated, truncated, info = env.step(applied_action)
        previous_action = applied_action
        recorded_info = jsonable(info) if isinstance(info, dict) else {}
        if active:
            raw_audit = recorded_info.get("or_audit", {})
            if not isinstance(raw_audit, dict) or not isinstance(
                raw_audit.get("applied_perturbations", []), list
            ):
                raise TaskContractError("gym reported malformed or_audit perturbation evidence")
            audit = dict(raw_audit)
            audit["applied_perturbations"] = [
                *audit.get("applied_perturbations", []),
                *(item.model_dump(mode="json") for item in active),
            ]
            recorded_info["or_audit"] = audit
        steps.append(
            {
                "action": jsonable(action),
                "obs": jsonable(obs),
                "reward": jsonable(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": recorded_info,
                **({"policy_observation": jsonable(policy_obs)} if observation_events else {}),
                **({"applied_action": jsonable(applied_action)} if hold_action else {}),
                **({"reset_info": jsonable(reset_info)} if step_i == 0 else {}),
            }
        )
        if terminated or truncated:
            break
    if not isinstance(info, dict):
        info = {}
    return cast(dict[str, Any], jsonable(info)), tuple(steps)
