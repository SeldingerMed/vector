"""Universal Simulation Engine Protocol and Registry for Physical Healthcare AI.

Bridges diverse physics engines (Gymnasium, Lumen, SOFA Framework, NVIDIA Warp/Isaac Lab,
PyBullet) into a unified reset/step/render/inspect interface for procedural evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    entry_points,
    packages_distributions,
    version,
)
from typing import Any, Protocol, runtime_checkable

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import (
    BUILTIN_WORLD_CAPABILITIES,
    WORLD_KIND_ENTRY_POINT_GROUP,
    WorldCapabilities,
    attach_world_adapter,
    require_world_kind,
    reset_default_world_kinds,
    world_kind_key,
    world_kind_spec,
)

#: A real physics/world library produced the observations.
BACKEND_REAL = "real"
#: A synthetic stand-in produced the observations; not physical evidence.
BACKEND_SYNTHETIC_STUB = "synthetic-stub"
#: The world exposes no provenance reporter, so the backend cannot be attested.
BACKEND_UNKNOWN = "unknown"


def module_distribution_version(module_name: str) -> str:
    """Version of the installed distribution providing ``module_name``, else ``""``."""
    root = module_name.split(".")[0]
    for dist_name in packages_distributions().get(root) or []:
        try:
            return version(dist_name)
        except PackageNotFoundError:  # pragma: no cover - metadata race only
            continue
    return ""


def module_distribution_revision(module_name: str) -> str:
    """Observed VCS commit or version of the distribution providing a module."""
    root = module_name.split(".")[0]
    for dist_name in packages_distributions().get(root) or []:
        try:
            dist = distribution(dist_name)
        except PackageNotFoundError:  # pragma: no cover - metadata race only
            continue
        metadata = dist.read_text("direct_url.json")
        if metadata:
            try:
                commit = json.loads(metadata).get("vcs_info", {}).get("commit_id", "")
            except (AttributeError, TypeError, json.JSONDecodeError):
                commit = ""
            if commit:
                return str(commit)
        return dist.version
    return ""


#: Deepest nesting a single-env batch unwrap will walk before refusing to guess
#: which level holds the scalar the engine must publish.
MAX_BATCH_UNWRAP_DEPTH = 4


def batch_len(value: Any) -> int | None:
    """Outer length of a batch-like engine value, or ``None`` when it is scalar.

    Asked rather than type-checked: a 0-d torch tensor or numpy array defines
    ``__len__`` and raises on it, so no ``hasattr`` test can tell a scalar from a
    vector for the types a GPU engine actually hands back. Strings and mappings
    are scalars here — a ``str`` is indexable but indexing it never ends.
    """
    if isinstance(value, str | bytes | Mapping):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def batch_rows(value: Any) -> list[Any] | None:
    """The value's outer dimension as a list, or ``None`` when it is scalar."""
    size = batch_len(value)
    return None if size is None else [value[index] for index in range(size)]


def require_step_scalar(value: Any, *, field: str, world: str) -> Any:
    """Unwrap a length-1 batch dimension off a step return; refuse anything wider.

    A single-env GPU engine still returns shape-``(1,)`` tensors, and taking the
    one entry out of a one-entry batch invents nothing. Two or more entries are a
    batch that ``SimulationEngine`` cannot represent: one observation, one
    reward, one terminated flag, one info dict per step. Coercing such a batch is
    never a reduction — ``float()`` raises on a list, and ``bool()`` on a
    non-empty list of per-env flags is unconditionally ``True``, which silently
    ends an episode the environments never ended. A named refusal is the only
    honest outcome.
    """
    current = value
    for _ in range(MAX_BATCH_UNWRAP_DEPTH):
        size = batch_len(current)
        if size is None:
            return current
        if size != 1:
            raise TaskContractError(
                f"world {world!r} returned {size} {field} values for one step; this "
                "SimulationEngine is a scalar contract and no reduction of a batch is "
                "a step a single environment took. Build the env with num_envs = 1."
            )
        current = current[0]
    raise TaskContractError(
        f"world {world!r} returned a {field} nested more than {MAX_BATCH_UNWRAP_DEPTH} "
        "levels deep; refusing to guess which level holds the scalar this engine must "
        "publish."
    )


def require_single_env(raw: Any, *, world: str) -> int:
    """Accept only ``num_envs == 1``.

    ``SimulationEngine`` is a scalar contract, so there is no well-defined scalar
    to publish for a batch: env 0 discards the rest, an average describes a world
    that does not exist, and a max turns one env's contact into every env's
    contact. Refusing is the honest outcome; repetition belongs to
    ``environment.n_eval_episodes`` and the seed policy, which give independently
    scored trials instead of one collapsed number.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TaskContractError(
            f"world {world!r}: environment.parameters.num_envs must be an integer, got {raw!r}"
        )
    if raw != 1:
        raise TaskContractError(
            f"world {world!r} requests num_envs={raw}, but this SimulationEngine is a "
            "scalar contract (one observation, reward, terminated flag, and info dict "
            "per step). A batched env returns a vector for each of those and no "
            "reduction of it is a step a single environment took, so the batch is "
            "refused rather than silently collapsed. Set "
            "environment.parameters.num_envs = 1 and use environment.n_eval_episodes "
            "with the task's seed policy for repetition."
        )
    return raw


#: ``gymnasium.make`` errors that mean "this world is not registered here", as
#: opposed to "this world exists and could not be built". Resolved by name
#: because gymnasium is an optional dependency and older versions omit some.
MISSING_WORLD_ERROR_NAMES = (
    "NameNotFound",
    "NamespaceNotFound",
    "VersionNotFound",
    "UnregisteredEnv",
)


def missing_world_errors(gymnasium_module: Any) -> tuple[type[BaseException], ...]:
    """Exception types from ``gymnasium.make`` that mean the world is not installed.

    Anything outside this set is a world that *is* registered and still failed to
    build, which is a task configuration failure. Both were caught by one bare
    ``except Exception`` that reported "no backend", so a task opting into the
    synthetic stub silently measured a stand-in under a real world's name. An
    empty tuple (a gymnasium exposing no ``error`` module) never matches, so
    nothing is misclassified as missing.
    """
    errors = getattr(gymnasium_module, "error", None)
    return tuple(
        candidate
        for name in MISSING_WORLD_ERROR_NAMES
        if isinstance(candidate := getattr(errors, name, None), type)
        and issubclass(candidate, BaseException)
    )


def refuse_unbuildable_world(engine: str, env_id: str, exc: BaseException) -> str:
    """Refusal for a world that is registered but could not be constructed."""
    return (
        f"{engine} world {env_id!r} is registered but could not be built: "
        f"{type(exc).__name__}: {exc}. That is a task configuration failure, not a "
        "missing vendor runtime, and the two must not look alike: this used to be "
        "caught and reported as 'no backend', so a task with "
        "environment.synthetic_stub = true silently measured a non-physical stand-in "
        "while naming a world that implies real physics. Fix environment.gym_id or "
        "environment.parameters, or drop the world; do not accept stand-in numbers "
        "under a real world's name."
    )


@runtime_checkable
class SimulationEngine(Protocol):
    """Protocol for physics simulators and procedural worlds."""

    world_kind: WorldKind | str

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Start an episode: returns (initial_observation, info_dict)."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one physics step: returns (obs, reward, terminated, truncated, info)."""
        ...

    def render(self, mode: str = "rgb_array") -> Any:
        """Render the current simulation state (image frame, point cloud, or mesh)."""
        ...

    def close(self) -> None:
        """Release simulator resources and subprocess/GPU handles."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of underlying physics state for oracle/verifier evaluation."""
        ...

    def engine_provenance(self) -> dict[str, Any]:
        """Report which engine and backend actually produced the observations."""
        ...


class BaseSimulationBridge:
    """Base class for simulation bridges providing default protocol behaviors."""

    world_kind: WorldKind | str = WorldKind.GYM
    world_pin: str = ""

    def engine_provenance(self) -> dict[str, Any]:
        """Report which engine and backend actually produced the observations."""
        return {
            "engine": world_kind_key(self.world_kind),
            "backend": BACKEND_REAL,
            "backend_version": "",
            "world_pin": self.world_pin,
        }

    def render(self, mode: str = "rgb_array") -> Any:
        """Default render implementation (returns None if headless)."""
        return None

    def close(self) -> None:
        """Default close implementation (no-op)."""

    def get_state(self) -> dict[str, Any]:
        """Default state extractor (returns empty dictionary)."""
        return {}


SimFactory = Callable[[TaskSpec], SimulationEngine]
_SIM_ENGINE_REGISTRY: dict[str, SimFactory] = {}


@dataclass(frozen=True)
class WorldAdapter:
    """A world adapter a distribution publishes: kind, capabilities, factory.

    This is the third-party publication unit for N3's open world-kind
    extension point. An adapter declares what its world is eligible for; the
    kernel pins the adapter's identity by content digest and records it, so a
    result names the exact adapter that produced it.
    """

    kind: str
    capabilities: WorldCapabilities
    factory: SimFactory
    provider: str = ""


@dataclass(frozen=True)
class AdapterDiscovery:
    """Outcome of loading one ``or_audit.world_kinds`` entry point."""

    name: str
    ok: bool
    kind: str = ""
    provider: str = ""
    adapter_identity: str = ""
    error: str = ""


_DISCOVERY: tuple[AdapterDiscovery, ...] = ()


def register_simulation_engine(
    kind: WorldKind | str,
    factory: SimFactory,
    *,
    capabilities: WorldCapabilities | None = None,
    provider: str = "",
    override: bool = False,
) -> None:
    """Register a simulation engine factory for a world kind.

    ``capabilities`` declares what the world is eligible for and is registered
    alongside the factory's digest-pinned adapter identity, which is what lets a
    non-Gym third-party world publish tasks without a kernel release. It may be
    omitted only when the kind already carries a declaration to inherit.

    Replacing a factory always re-pins the identity, reusing the already
    registered capabilities when the caller omits them: a spec left pointing at
    the previous factory would make every job head attest an adapter the
    dispatcher no longer runs, which is exactly the substitution the digest
    exists to detect. ``provider`` is *not* inherited from the replaced spec -
    a new factory is published by whoever registers it, and silently keeping
    the old distribution's name would be a provenance claim nobody made.

    A kind with no declaration anywhere is refused. An adapter that declares
    nothing withholds everything, so installing a runnable engine for it would
    leave ``resolve_world_capabilities`` with only the task's own word and let
    a package grant itself ``physics`` (and a physics oracle with it) - the
    inversion the adapter-authoritative rule exists to prevent.
    """
    key = world_kind_key(kind)
    if key in _SIM_ENGINE_REGISTRY and not override:
        raise TaskContractError(f"simulation engine already registered for {key!r}")
    existing = world_kind_spec(key)
    declared = capabilities
    if declared is None and existing is not None:
        declared = existing.capabilities
    if declared is None:
        raise TaskContractError(
            f"world kind {key!r} has no declared capabilities, so an engine for it "
            "cannot be registered: an undeclared adapter withholds every eligibility, "
            "and a task would be left granting itself physics or closed-loop on its own "
            "word. Pass capabilities= to register_simulation_engine, or declare the kind "
            f"first with register_world_kind(WorldKindSpec(kind={key!r}, capabilities=...))"
        )
    attach_world_adapter(
        key,
        capabilities=declared,
        factory=factory,
        provider=provider,
    )
    _SIM_ENGINE_REGISTRY[key] = factory


def register_world_adapter(adapter: WorldAdapter, *, override: bool = False) -> None:
    """Register a :class:`WorldAdapter` as both a world kind and a sim factory."""
    register_simulation_engine(
        adapter.kind,
        adapter.factory,
        capabilities=adapter.capabilities,
        provider=adapter.provider,
        override=override,
    )


def _load_adapter(loaded: Any, *, name: str) -> WorldAdapter:
    adapter = loaded() if callable(loaded) and not isinstance(loaded, WorldAdapter) else loaded
    if not isinstance(adapter, WorldAdapter):
        raise TaskContractError(
            f"world-kind entry point {name!r} must resolve to a WorldAdapter "
            f"(or a callable returning one), got {type(adapter).__name__}"
        )
    return adapter


def discover_world_adapters(*, override: bool = False) -> tuple[AdapterDiscovery, ...]:
    """Load every ``or_audit.world_kinds`` entry point and register its adapter.

    A broken or colliding third-party adapter is recorded as a failed
    discovery rather than raised: one bad plugin must not make the kernel
    unusable. Failures are reported by :func:`world_adapter_discovery`, by
    ``surgeval doctor``, and in the error raised when a task names a world kind
    that has no engine.
    """
    global _DISCOVERY
    results: list[AdapterDiscovery] = []
    for entry in entry_points(group=WORLD_KIND_ENTRY_POINT_GROUP):
        try:
            adapter = _load_adapter(entry.load(), name=entry.name)
            register_world_adapter(adapter, override=override)
        except Exception as exc:  # third-party import errors are data, not crashes
            results.append(AdapterDiscovery(name=entry.name, ok=False, error=str(exc)))
            continue
        spec = require_world_kind(adapter.kind)
        results.append(
            AdapterDiscovery(
                name=entry.name,
                ok=True,
                kind=spec.kind,
                provider=spec.provider,
                adapter_identity=spec.adapter_identity,
            )
        )
    _DISCOVERY = tuple(results)
    return _DISCOVERY


def world_adapter_discovery() -> tuple[AdapterDiscovery, ...]:
    """The most recent plugin-discovery report."""
    return _DISCOVERY


def get_simulation_engine(task: TaskSpec) -> SimulationEngine | None:
    """Get an instantiated simulation engine for a task, or None if not registered."""
    key = world_kind_key(task.environment.kind)
    factory = _SIM_ENGINE_REGISTRY.get(key)
    if factory is None:
        return None
    return factory(task)


def require_simulation_engine(task: TaskSpec) -> SimulationEngine:
    """Get a simulation engine for a task or raise TaskContractError if missing."""
    engine = get_simulation_engine(task)
    if engine is None:
        key = world_kind_key(task.environment.kind)
        known = ", ".join(sorted(_SIM_ENGINE_REGISTRY.keys()))
        failed = ", ".join(f"{item.name} ({item.error})" for item in _DISCOVERY if not item.ok)
        detail = f"; failed world-kind plugins: {failed}" if failed else ""
        raise TaskContractError(
            f"task {task.id} world kind {key!r} has no registered simulation engine; "
            f"known: {known}{detail}"
        )
    return engine


def list_simulation_engines() -> dict[str, str]:
    """Return dictionary of registered world kinds and their factory names."""
    return {k: getattr(v, "__name__", "factory") for k, v in sorted(_SIM_ENGINE_REGISTRY.items())}


def clear_simulation_registry() -> None:
    """Reset the simulation registry (primarily for test isolation)."""
    _SIM_ENGINE_REGISTRY.clear()


def reset_default_simulation_engines() -> None:
    """Reset and re-register standard built-in simulation engine bridges."""
    _SIM_ENGINE_REGISTRY.clear()
    reset_default_world_kinds()
    from or_audit.eval.sim.gym_bridge import make_gym_bridge
    from or_audit.eval.sim.isaac_bridge import make_isaac_bridge
    from or_audit.eval.sim.sofa_bridge import make_sofa_bridge
    from or_audit.eval.sim.warp_bridge import make_warp_bridge

    builtins: tuple[tuple[WorldKind, SimFactory], ...] = (
        (WorldKind.LUMEN_GYM, make_gym_bridge),
        (WorldKind.GYM, make_gym_bridge),
        (WorldKind.SOFA, make_sofa_bridge),
        (WorldKind.WARP, make_warp_bridge),
        (WorldKind.ISAAC_LAB, make_isaac_bridge),
        (WorldKind.PYBULLET, make_gym_bridge),
    )
    for kind, factory in builtins:
        register_simulation_engine(
            kind,
            factory,
            capabilities=BUILTIN_WORLD_CAPABILITIES[kind],
            provider="surgeval",
            override=True,
        )
