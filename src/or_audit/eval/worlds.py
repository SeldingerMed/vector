"""Open world-kind registry: declared capabilities, determinism, adapter identity.

``WorldKind`` remains the set of world kinds the kernel ships adapters for, but
it is no longer the closed gate on what a task package may declare. Kernel
behaviour that used to test enum-set membership (``physics`` oracle
eligibility, closed-loop eligibility, which fields a world requires) now reads
:class:`WorldCapabilities` from this registry, so a third-party world can
publish a task through a plugin-registered adapter without a core release.

Two declaration paths exist and they must agree:

* an installed adapter registers a :class:`WorldKindSpec` (authoritative,
  carries digest-pinned adapter identity);
* a task package may declare ``[environment.capabilities]`` so a package
  remains loadable, describable, and reviewable on a machine where the adapter
  is not installed.

When both exist, :func:`resolve_world_capabilities` refuses a mismatch: a task
author cannot claim physics or closed-loop eligibility the installed adapter
does not grant.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind

#: Grammar a world-kind key is *written* in. Underscores are accepted on input
#: and folded to hyphens by :func:`world_kind_key`, so the canonical key a
#: registry is keyed by and provenance records never contains one.
WorldKindSlug = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]

#: Entry-point group a third-party distribution publishes adapters under.
WORLD_KIND_ENTRY_POINT_GROUP = "or_audit.world_kinds"


class DeterminismClass(StrEnum):
    """Measured reproducibility of a seeded world rerun.

    Recorded, never assumed. ``unmeasured`` is the honest default: the
    conformance suite measures a class per world (Tier-1 requirement) and
    refuses a declaration stronger than the measurement.
    """

    #: A seeded rerun reproduces the trace byte-for-byte (canonical digest).
    BITWISE = "bitwise"
    #: A seeded rerun reproduces the vector, with trace floats inside tolerance.
    TOLERANCE = "tolerance"
    #: A seeded rerun does not reproduce the vector.
    NONDETERMINISTIC = "nondeterministic"
    #: No execution-determinism measurement has been recorded for this world.
    UNMEASURED = "unmeasured"


#: Ordering used to refuse a declared class stronger than the measured one.
_DETERMINISM_STRENGTH: dict[DeterminismClass, int] = {
    DeterminismClass.NONDETERMINISTIC: 0,
    DeterminismClass.UNMEASURED: 1,
    DeterminismClass.TOLERANCE: 2,
    DeterminismClass.BITWISE: 3,
}


def determinism_at_least(measured: DeterminismClass, declared: DeterminismClass) -> bool:
    """Whether ``measured`` is at least as strong as ``declared``."""
    return _DETERMINISM_STRENGTH[measured] >= _DETERMINISM_STRENGTH[declared]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorldCapabilities(_Frozen):
    """What a world kind is eligible for, declared rather than inferred."""

    #: Eligible to back a ``physics`` oracle: the world reports physical state.
    physics: bool = False
    #: Eligible for closed-loop interaction: the world steps under a policy.
    closed_loop: bool = False
    #: Eligible for counterfactual interfaces.
    counterfactual: bool = False
    #: The world is addressed by a ``gym_id``.
    requires_gym_id: bool = False
    #: A runnable task on this world must pin the world.
    requires_world_pin: bool = False
    #: The world is addressed by a package-relative contract file.
    requires_contract: bool = False
    #: Measured execution-determinism class; ``unmeasured`` until conformance runs.
    #:
    #: Safety-state availability is deliberately absent here: a generic bridge
    #: (Gym, PyBullet, SOFA, Warp) passes through whatever a particular env
    #: reports, so instrumentation is a property of the wrapped world instance,
    #: not of the world kind. It is declared per task in
    #: ``WorldSpec.metrics_only`` and verified per env by the conformance
    #: suite's gate-state availability check (§2.2).
    determinism_class: DeterminismClass = DeterminismClass.UNMEASURED

    def gates(self) -> tuple[bool, ...]:
        """The eligibility flags a declaration must not overstate."""
        return (
            self.physics,
            self.closed_loop,
            self.counterfactual,
            self.requires_gym_id,
            self.requires_world_pin,
            self.requires_contract,
        )


class WorldKindSpec(_Frozen):
    """A registered world kind: capabilities plus digest-pinned adapter identity."""

    kind: WorldKindSlug
    capabilities: WorldCapabilities
    #: ``module:symbol`` of the adapter factory, empty until an adapter attaches.
    adapter_id: str = ""
    #: SHA-256 of the adapter factory's module source; empty when unattached.
    adapter_digest: str = ""
    #: Distribution that published the adapter, for provenance.
    provider: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _canonical_kind(cls, value: Any) -> Any:
        # A spec is keyed by ``spec.kind``, and a task looks its world up by
        # ``world_kind_key``. Canonicalizing here keeps a registered plugin kind
        # reachable from the task that names it: registering ``steve_sofa`` and
        # loading a task declaring ``steve_sofa`` must hit the same entry.
        return world_kind_key(value) if isinstance(value, str) else value

    @property
    def adapter_identity(self) -> str:
        """Stable adapter identity, or ``"unattached"`` when no adapter is registered."""
        if not self.adapter_id or not self.adapter_digest:
            return "unattached"
        return f"{self.adapter_id}+{self.adapter_digest}"


def world_kind_key(kind: WorldKind | str) -> str:
    """Canonical registry / provenance key for a declared or authored world kind.

    Folds underscores to hyphens exactly as ``WorldSpec._normalize_kind`` does.
    The two normalizations must stay identical: a registry keyed differently
    from the task that names it silently loses the adapter, and a lost adapter
    means an unpinned, unattested run rather than a refusal.
    """
    if isinstance(kind, WorldKind):
        return kind.value
    return str(kind).replace("_", "-").lower()


def adapter_identity(factory: Callable[..., Any]) -> tuple[str, str]:
    """Return ``(module:symbol, sha256-of-module-source)`` for an adapter factory.

    The digest pins adapter *behaviour*, not just its name: two distributions
    publishing the same world kind produce different identities, and a patched
    adapter changes the identity recorded on every artifact it touches.

    Fails closed. A factory with no importable ``module:qualname`` or no
    readable source cannot be pinned at all, and hashing its name instead would
    mint an identity that pins nothing: two distinct callable objects of the
    same class, or two C callables, would share one digest and could be
    swapped for each other without moving a single job head.
    """
    module = getattr(factory, "__module__", "") or ""
    symbol = getattr(factory, "__qualname__", "") or getattr(factory, "__name__", "") or ""
    try:
        source_file = inspect.getsourcefile(factory)
    except TypeError:  # builtins / C callables / callable instances carry no source
        source_file = None
    payload: bytes | None = None
    if source_file is not None:
        try:
            payload = Path(source_file).read_bytes()
        except OSError:
            payload = None
    if not module or not symbol or payload is None:
        missing = "an importable module:qualname" if not module or not symbol else "readable source"
        raise TaskContractError(
            f"adapter factory {factory!r} cannot be content-pinned: it has no {missing}. "
            "A world adapter must be a module-level function (or another callable whose "
            "source file is readable) so its digest changes when its behaviour does; "
            "wrap a callable object or a partial in a named module-level factory."
        )
    return f"{module}:{symbol}", hashlib.sha256(payload).hexdigest()


#: Capabilities of the world kinds the kernel ships. These are declarations,
#: not validator branches: the same table shape a plugin registers.
BUILTIN_WORLD_CAPABILITIES: dict[WorldKind, WorldCapabilities] = {
    WorldKind.LUMEN_GYM: WorldCapabilities(
        physics=True,
        closed_loop=True,
        requires_gym_id=True,
        requires_world_pin=True,
    ),
    WorldKind.LUMEN_REPLAY: WorldCapabilities(physics=True, closed_loop=True),
    WorldKind.GYM: WorldCapabilities(
        physics=True,
        closed_loop=True,
        requires_gym_id=True,
        requires_world_pin=True,
    ),
    WorldKind.SOFA: WorldCapabilities(physics=True, closed_loop=True, requires_world_pin=True),
    WorldKind.WARP: WorldCapabilities(physics=True, closed_loop=True, requires_world_pin=True),
    WorldKind.ISAAC_LAB: WorldCapabilities(physics=True, closed_loop=True, requires_world_pin=True),
    WorldKind.PYBULLET: WorldCapabilities(
        physics=True,
        closed_loop=True,
        requires_gym_id=True,
        requires_world_pin=True,
    ),
    WorldKind.ANGIOSTRESS_CONTRACT: WorldCapabilities(requires_contract=True),
    WorldKind.FRAME_SOURCE: WorldCapabilities(),
    WorldKind.COUNTERFACTUAL: WorldCapabilities(counterfactual=True),
    WorldKind.VIDEO_STREAM: WorldCapabilities(),
    WorldKind.CT_AIRWAY: WorldCapabilities(),
}

_WORLD_KIND_REGISTRY: dict[str, WorldKindSpec] = {}


def register_world_kind(spec: WorldKindSpec, *, override: bool = False) -> None:
    """Register a world kind's declared capabilities and adapter identity.

    ``spec.kind`` is already canonical (see ``WorldKindSpec._canonical_kind``),
    so the key stored here is the same key :func:`world_kind_spec` looks up.
    """
    if spec.kind in _WORLD_KIND_REGISTRY and not override:
        raise TaskContractError(f"world kind already registered for {spec.kind!r}")
    _WORLD_KIND_REGISTRY[spec.kind] = spec


def world_kind_spec(kind: WorldKind | str) -> WorldKindSpec | None:
    """Registered spec for a world kind, or ``None`` when nothing is registered."""
    return _WORLD_KIND_REGISTRY.get(world_kind_key(kind))


def require_world_kind(kind: WorldKind | str) -> WorldKindSpec:
    """Registered spec for a world kind, or raise with the known kinds listed."""
    spec = world_kind_spec(kind)
    if spec is None:
        known = ", ".join(sorted(_WORLD_KIND_REGISTRY)) or "(none)"
        raise TaskContractError(
            f"world kind {world_kind_key(kind)!r} is not registered; known: {known}"
        )
    return spec


def list_world_kinds() -> dict[str, WorldKindSpec]:
    """Every registered world kind, sorted by key."""
    return dict(sorted(_WORLD_KIND_REGISTRY.items()))


def clear_world_kind_registry() -> None:
    """Reset the world-kind registry (primarily for test isolation)."""
    _WORLD_KIND_REGISTRY.clear()


def reset_default_world_kinds() -> None:
    """Reset and re-register the built-in world-kind capability declarations."""
    _WORLD_KIND_REGISTRY.clear()
    for kind, capabilities in BUILTIN_WORLD_CAPABILITIES.items():
        _WORLD_KIND_REGISTRY[kind.value] = WorldKindSpec(
            kind=kind.value,
            capabilities=capabilities,
            provider="surgeval",
        )


def attach_world_adapter(
    kind: WorldKind | str,
    *,
    capabilities: WorldCapabilities,
    factory: Callable[..., Any],
    provider: str = "",
) -> WorldKindSpec:
    """Register (or re-register) a world kind with digest-pinned adapter identity.

    An adapter is authoritative over a task's own declaration, so attaching one
    for an already-registered kind replaces the capability declaration rather
    than appending a second, ambiguous source of truth.
    """
    key = world_kind_key(kind)
    identity, digest = adapter_identity(factory)
    spec = WorldKindSpec(
        kind=key,
        capabilities=capabilities,
        adapter_id=identity,
        adapter_digest=digest,
        provider=provider,
    )
    register_world_kind(spec, override=True)
    return spec


def resolve_world_capabilities(
    kind: WorldKind | str,
    declared: WorldCapabilities | None,
) -> WorldCapabilities:
    """Capabilities the kernel gates on for ``kind``.

    An installed adapter wins. A task-declared block is accepted only when it
    agrees with the adapter on every eligibility flag, and is the sole source
    when no adapter is installed. An unregistered kind with no declaration is a
    contract error, never a silent grant.
    """
    key = world_kind_key(kind)
    spec = world_kind_spec(key)
    if spec is None:
        if declared is None:
            raise TaskContractError(
                f"world kind {key!r} has no installed adapter and the task declares no "
                "[environment.capabilities]; install the world adapter or declare the "
                "world's physics/closed-loop eligibility in the task package"
            )
        return declared
    if declared is not None and declared.gates() != spec.capabilities.gates():
        raise TaskContractError(
            f"world kind {key!r} capability declaration disagrees with the installed "
            f"adapter ({spec.adapter_identity}); a task cannot grant itself eligibility "
            "the adapter withholds"
        )
    return spec.capabilities


reset_default_world_kinds()
