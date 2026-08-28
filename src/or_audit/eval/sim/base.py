"""Universal Simulation Engine Protocol and Registry for Physical Healthcare AI.

Bridges diverse physics engines (Gymnasium, Lumen, SOFA Framework, NVIDIA Warp/Isaac Lab,
PyBullet) into a unified reset/step/render/inspect interface for procedural evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
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

    Passing ``capabilities`` also (re-)registers the world kind's declared
    eligibility with digest-pinned adapter identity, which is what lets a
    non-Gym third-party world publish tasks without a kernel release.
    """
    key = world_kind_key(kind)
    if key in _SIM_ENGINE_REGISTRY and not override:
        raise TaskContractError(f"simulation engine already registered for {key!r}")
    if capabilities is not None:
        attach_world_adapter(
            key,
            capabilities=capabilities,
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
