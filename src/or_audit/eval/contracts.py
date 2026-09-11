"""Versioned v0.3 contracts shared by tasks, agents, runners, and scorecards."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, NoReturn, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from or_audit.errors import TaskContractError

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
]


class InteractionMode(StrEnum):
    """Shape of interaction between a harness and an agent."""

    CLOSED_LOOP = "closed-loop"
    SINGLE_TURN = "single-turn"
    INTERACTIVE = "interactive"
    COUNTERFACTUAL = "counterfactual"


class RuntimeKind(StrEnum):
    """Portable agent runtime identities represented by v0.3."""

    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE = "openai-compatible"
    CONTAINER = "container"
    TRUSTED_IN_PROCESS = "trusted-in-process"


class MetricKind(StrEnum):
    """Metric value shapes with distinct aggregation rules."""

    BOOLEAN = "boolean"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class MetricDirection(StrEnum):
    """How a metric should move when results improve."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class GateProjectionPolicy(StrEnum):
    """Projection behavior when a hard gate is not cleanly satisfied."""

    ZERO = "zero"
    REFUSE = "refuse"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


#: SHA-256 lowercase hex — a 64-char immutable content pin.
SHA256Hex = Annotated[
    str, StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]
#: Observation source locator: ``"$"`` addresses the whole observation,
#: otherwise a JSON pointer (``/camera``) or dotted path (``camera.frame``).
SourceLocator = Annotated[
    str, StringConstraints(min_length=1, max_length=400, pattern=r"^[a-zA-Z0-9_./$-]+$")
]


class _FrozenDict(dict[str, Any]):
    """Deeply immutable JSON object used for pinned calibration payloads.

    Built only through :func:`_freeze_json`, which recursively replaces nested
    mappings with this class and nested sequences with tuples, so no mutation
    path — shallow or deep — survives construction. Equality stays
    content-based (``dict.__eq__``), so a frozen and a plain mapping with the
    same content still compare equal.
    """

    def __setitem__(self, key: str, value: Any) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    def __delitem__(self, key: str) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    def setdefault(self, key: str, default: Any = None) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    def pop(self, *args: Any) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    def popitem(self) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    def clear(self) -> NoReturn:
        raise TypeError("camera_calibration is deeply immutable")

    # CPython resolves ``x |= y`` through the C-level ``dict.__ior__``, which
    # mutates in place and bypasses ``__setitem__`` — the override is required
    # for soundness. typeshed models ``dict.__or__`` as overloads no raising
    # override can match, so the misc override diagnostic is suppressed here
    # (the sibling ``__or__`` is inherited and produces a fresh dict anyway).
    def __ior__(self, value: Any) -> NoReturn:  # type: ignore[misc]
        raise TypeError("camera_calibration is deeply immutable")


def _calibration_is_declared(value: Mapping[str, Any]) -> bool:
    """Whether a calibration carries any meaningful calibration data.

    An empty mapping is "not declared". A mapping with keys is declared even
    if a value is ``0`` or ``False`` — those are real calibration values
    (e.g. zero principal-point offset, disabled skew), so emptiness is tested
    by key count rather than by truthiness of the payload.
    """
    return len(value) > 0


def _calibration_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Content equality between two calibration trees.

    Re-canonicalizes both sides first: a tree that arrived through
    ``model_copy(update=...)`` skipped validation and still carries lists
    where a validated one holds tuples, and raw ``==`` would call that a
    mismatch. A side that cannot even be canonicalized is not a calibration
    anyone should bind against, so it compares unequal rather than raising
    inside a predicate that selection loops (``bind.py``) call casually.
    """
    try:
        left_frozen: Any = _freeze_json(left, where="camera_calibration")
        right_frozen: Any = _freeze_json(right, where="camera_calibration")
        return bool(left_frozen == right_frozen)
    except TaskContractError:
        return False


def _freeze_json(value: Any, *, where: str) -> Any:
    """Canonicalize ``value`` into a deeply immutable JSON tree.

    Rejects anything a JSON document could not carry: non-string object keys,
    non-finite floats, and unsupported leaf types. Keys are sorted so two
    calibrations with the same content canonicalize to the same order.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskContractError(f"{where}: non-finite number {value!r} is not valid JSON")
        return value
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for key, sub in value.items():
            if not isinstance(key, str):
                raise TaskContractError(f"{where}: object keys must be strings, got {key!r}")
            items[key] = _freeze_json(sub, where=f"{where}.{key}")
        return _FrozenDict(sorted(items.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, where=f"{where}[]") for item in value)
    raise TaskContractError(
        f"{where}: value of type {type(value).__name__} has no JSON representation"
    )


class StreamSpec(_Frozen):
    """One sensor/data channel a task presents to an agent.

    ``schema_id`` names the observation schema (data shape) this channel
    carries, so agent capabilities bind to data shape — not to a closed
    procedure taxonomy. ``adapter`` is the adapter-plugin identity: an open
    plugin id resolvable in the adapter registry and pinned immutably by
    ``adapter_digest``, the SHA-256 of the plugin content, verified at task
    load. ``source`` locates the observation slice this stream consumes:
    ``"$"`` for the whole observation, otherwise a JSON pointer / dotted path
    into it. A stream with an unresolvable required source is a contract
    error, never a silent pass-through.
    """

    id: Slug
    schema_id: Slug
    adapter: Slug
    adapter_digest: SHA256Hex
    source: SourceLocator = "$"
    role: Slug | None = None
    dtype: str = ""
    shape: tuple[int, ...] = ()
    unit: str = ""
    coordinate_frame: str = ""
    valid_range: tuple[float, float] | None = None
    controller_id: str = ""
    #: Deeply immutable pinned calibration; see :func:`_freeze_json`. Declared as
    #: ``Mapping`` so pydantic never hands back a caller-retained mutable dict.
    camera_calibration: Mapping[str, Any] = Field(default_factory=_FrozenDict)
    joint_order: tuple[str, ...] = ()
    invalid_depth_encoding: str = ""
    privileged: bool = False

    @field_validator("shape", mode="before")
    def _reject_impossible_shape(cls, value: Any) -> Any:
        # ``tuple[int, ...]`` is lax in pydantic: ``True`` launders into ``1``
        # and a zero/negative extent passes the type check, so the raw input
        # must be screened before coercion. A bool is not a dimension, and a
        # non-positive axis is a buffer no tensor can be allocated against.
        if isinstance(value, list | tuple):
            for dim in value:
                if isinstance(dim, bool) or (isinstance(dim, int) and dim <= 0):
                    raise TaskContractError(f"stream declares non-positive shape dimension {dim!r}")
        return value

    @model_validator(mode="after")
    def _plugin_needs_schema(self) -> Self:
        if not self.adapter:
            raise TaskContractError(f"stream {self.id} requires an adapter plugin id")
        if not self.schema_id:
            raise TaskContractError(f"stream {self.id} requires an observation schema_id")
        return self

    @model_validator(mode="after")
    def _geometry_is_sane(self) -> Self:
        # Shape soundness (bool laundering, non-positive axes) is screened on
        # the raw input by ``_reject_impossible_shape`` before pydantic can
        # coerce; what is left here is range semantics.
        if self.valid_range is not None:
            low, high = self.valid_range
            if math.isnan(low) or math.isnan(high):
                raise TaskContractError(
                    f"stream {self.id} valid_range {self.valid_range!r} contains a NaN bound; "
                    "every comparison against it is False, so the range could never be enforced"
                )
            if low > high:
                raise TaskContractError(
                    f"stream {self.id} valid_range {self.valid_range!r} is reversed"
                )
            # A range is a constraint only if it rules something out. Fully
            # finite ranges are the ordinary case and always admissible.
            # One-sided ranges are legitimate physics (depth has no ceiling, a
            # joint angle may be open below), so an infinite endpoint is fine
            # while the other side is finite. But when *both* endpoints are
            # infinite the range admits every value: it asserts nothing, so it
            # must be omitted rather than declared-vacuous.
            if not math.isfinite(low) and not math.isfinite(high):
                raise TaskContractError(
                    f"stream {self.id} valid_range {self.valid_range!r} bounds neither side; "
                    "omit valid_range when the value is unconstrained"
                )
        return self

    @field_validator("camera_calibration", mode="before")
    def _pin_calibration(cls, value: Any) -> Any:
        # A pinned calibration a caller can still mutate is not a pin: digest
        # the stream, ship it, then edit the dict, and a binding that was
        # refused now holds. Freeze deep, and reject anything JSON could not
        # round-trip so the pinned form survives serialization unchanged.
        # Screening happens before pydantic's own ``Mapping[str, Any]`` check
        # so every malformed calibration surfaces as one TaskContractError,
        # not sometimes-a-ValidationError depending on which field is bad.
        if not isinstance(value, Mapping):
            raise TaskContractError(
                f"camera_calibration must be a mapping, got {type(value).__name__}"
            )
        if type(value) is _FrozenDict:
            return value
        return _freeze_json(value, where="camera_calibration")

    @model_validator(mode="after")
    def _refreeze_after_field_validation(self) -> Self:
        # Safety net under the before-validator: ``Mapping[str, Any]`` field
        # validation rebuilds the top-level container, so pydantic can hand
        # back a plain dict even when the input was already canonical (the
        # default_factory path also never runs the before-validator). Any
        # container that is not the frozen type is re-canonicalized here, so
        # a stored stream never exposes a mutable calibration — which is the
        # whole point of pinning it.
        if type(self.camera_calibration) is not _FrozenDict:
            frozen: Any = _freeze_json(
                self.camera_calibration, where=f"stream {self.id}.camera_calibration"
            )
            object.__setattr__(self, "camera_calibration", frozen)
        return self


class InterfaceSpec(_Frozen):
    """Requirements a task exposes to compatible agents."""

    id: Slug
    interaction_mode: InteractionMode
    protocol_version: str = "1"
    observations: tuple[Slug, ...] = ()
    actions: tuple[Slug, ...] = ()
    outputs: tuple[Slug, ...] = ()
    features: tuple[Slug, ...] = ()
    #: Composed modality streams. Each stream binds an observation schema to a
    #: digest-pinned adapter plugin. Authoritative: there is no closed
    #: ``ModalityKind`` dispatch; ``modalities`` on a capability is the open
    #: plugin-id list an agent implements.
    streams: tuple[StreamSpec, ...] = ()

    @model_validator(mode="after")
    def _shape_matches_mode(self) -> Self:
        if self.interaction_mode is InteractionMode.CLOSED_LOOP and not self.actions:
            raise TaskContractError(f"closed-loop interface {self.id} must declare an action")
        if self.interaction_mode is not InteractionMode.CLOSED_LOOP and not self.outputs:
            raise TaskContractError(
                f"{self.interaction_mode.value} interface {self.id} needs output"
            )
        seen_ids: set[str] = set()
        known_schemas = set(self.observations) | set(self.features)
        for stream in self.streams:
            if stream.id in seen_ids:
                raise TaskContractError(
                    f"interface {self.id} declares duplicate stream id {stream.id!r}"
                )
            seen_ids.add(stream.id)
            if stream.schema_id not in known_schemas:
                raise TaskContractError(
                    f"interface {self.id} stream {stream.id} schema {stream.schema_id!r} "
                    "is not among the declared observations/features"
                )
        return self


class CapabilitySpec(_Frozen):
    """One interface implementation declared by an agent package."""

    interface: Slug
    interaction_modes: tuple[InteractionMode, ...]
    protocol_versions: tuple[str, ...] = ("1",)
    observations: tuple[Slug, ...] = ()
    actions: tuple[Slug, ...] = ()
    outputs: tuple[Slug, ...] = ()
    features: tuple[Slug, ...] = ()
    modalities: tuple[Slug, ...] = ()
    schema_wildcard: bool = False
    stream_profiles: tuple[StreamSpec, ...] = ()
    accepts_privileged: bool = False

    @model_validator(mode="after")
    def _validate_capability(self) -> Self:
        if not self.interaction_modes:
            raise TaskContractError(f"capability {self.interface} declares no interaction mode")
        seen_ids: set[str] = set()
        for s in self.stream_profiles:
            if s.id in seen_ids:
                raise TaskContractError(
                    f"capability {self.interface} declares duplicate stream_profile id {s.id!r}"
                )
            seen_ids.add(s.id)
        return self

    def _profile_index(self) -> tuple[dict[Slug, StreamSpec], dict[Slug, list[StreamSpec]]]:
        """Index profiles by exact stream id and by schema id.

        Two maps, never one merged: keying a single dict by both ``s.id`` and
        ``s.schema_id`` lets one channel's profile satisfy another channel's
        requirements whenever a stream id collides with some schema name, which
        is legal because both namespaces are open slugs.
        """
        by_id: dict[Slug, StreamSpec] = {}
        by_schema: dict[Slug, list[StreamSpec]] = {}
        for s in self.stream_profiles:
            by_id[s.id] = s
            by_schema.setdefault(s.schema_id, []).append(s)
        return by_id, by_schema

    def satisfies(self, interface: InterfaceSpec) -> bool:
        """Return whether this declaration satisfies every task requirement."""
        own_schemas = set(self.observations) | set(self.features)
        schemas_match = self.schema_wildcard or (
            set(interface.observations) <= set(self.observations)
            and set(interface.actions) <= set(self.actions)
            and set(interface.outputs) <= set(self.outputs)
            and set(interface.features) <= set(self.features)
            and all(
                stream.schema_id in own_schemas and stream.adapter in self.modalities
                for stream in interface.streams
            )
        )
        if not (
            self.interface == interface.id
            and interface.interaction_mode in self.interaction_modes
            and interface.protocol_version in self.protocol_versions
            and schemas_match
        ):
            return False

        by_id, by_schema = self._profile_index()
        interface_stream_ids = {stream.id for stream in interface.streams}
        for intf_stream in interface.streams:
            if intf_stream.privileged and not self.accepts_privileged:
                return False
            has_semantics = bool(
                intf_stream.unit
                or intf_stream.coordinate_frame
                or intf_stream.controller_id
                or intf_stream.dtype
                or intf_stream.shape
                or intf_stream.joint_order
                or intf_stream.invalid_depth_encoding
                or intf_stream.valid_range is not None
                or _calibration_is_declared(intf_stream.camera_calibration)
            )
            # Resolve the declaration that speaks for this channel. Exact channel
            # identity wins. A schema-level route is admissible only under
            # declared semantics and only when it cannot be confused with
            # another channel, because profiles are declared per data shape and
            # two channels may legitimately share one shape — guessing which
            # channel a shape referred to is how a probe's profile ends up
            # vouching for a camera.
            matching_profile = by_id.get(intf_stream.id)
            if matching_profile is None:
                if not has_semantics:
                    continue
                candidates = by_schema.get(intf_stream.schema_id)
                if not candidates or len(candidates) != 1:
                    return False
                candidate = candidates[0]
                # A profile named for a *different* channel of this same
                # interface is that channel's declaration, not this one's.
                if candidate.id != intf_stream.id and candidate.id in interface_stream_ids:
                    return False
                matching_profile = candidate

            # Channel identity is compared unconditionally: adapter + digest
            # are required fields on both sides, so "I serve this channel"
            # means "through this pinned plugin content". A differing digest
            # is a different plugin build, i.e. a different contract.
            if (
                matching_profile.schema_id != intf_stream.schema_id
                or matching_profile.adapter != intf_stream.adapter
                or matching_profile.adapter_digest != intf_stream.adapter_digest
            ):
                return False
            if matching_profile.source != intf_stream.source:
                # ``source`` locates the observation slice the channel carries;
                # whole-observation ("$") vs a pointer are different channels
                # even under one id, so compare it on both sides.
                return False
            if intf_stream.role is not None and matching_profile.role != intf_stream.role:
                return False
            if intf_stream.unit and matching_profile.unit != intf_stream.unit:
                return False
            if (
                intf_stream.coordinate_frame
                and matching_profile.coordinate_frame != intf_stream.coordinate_frame
            ):
                return False
            if (
                intf_stream.controller_id
                and matching_profile.controller_id != intf_stream.controller_id
            ):
                return False
            if intf_stream.dtype and matching_profile.dtype != intf_stream.dtype:
                return False
            if intf_stream.shape and matching_profile.shape != intf_stream.shape:
                return False
            if intf_stream.joint_order and matching_profile.joint_order != intf_stream.joint_order:
                return False
            if (
                intf_stream.invalid_depth_encoding
                and matching_profile.invalid_depth_encoding != intf_stream.invalid_depth_encoding
            ):
                return False
            if (
                intf_stream.valid_range is not None
                and matching_profile.valid_range != intf_stream.valid_range
            ):
                return False
            if _calibration_is_declared(intf_stream.camera_calibration) and not _calibration_equal(
                matching_profile.camera_calibration, intf_stream.camera_calibration
            ):
                return False
        return True


class HarnessSpec(_Frozen):
    """Task-owned execution mode and protocol limits."""

    interaction_mode: InteractionMode
    protocol_version: str = "1"
    max_steps: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000


class ScenarioSpec(_Frozen):
    """Versioned initial condition for a procedural trial."""

    id: Slug
    version: str = "1"
    description: str = ""
    seed: Annotated[int, Field(ge=0)] = 0
    inputs: dict[str, Any] = Field(default_factory=dict)


class PerturbationSpec(_Frozen):
    """Task-owned disturbance applied to a scenario."""

    id: Slug
    version: str = "1"
    description: str = ""
    scenario_id: Slug | None = None
    kind: Slug
    at_step: Annotated[int, Field(ge=0)] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class RuntimeDescriptor(_Frozen):
    """Pinned description of where and how an agent executes."""

    kind: RuntimeKind
    protocol_version: str = "1"
    command: tuple[str, ...] = ()
    entrypoint: str = ""
    model: str = ""
    revision: str = ""
    base_url: str = ""
    image: str = ""
    image_digest: str = ""
    timeout_sec: Annotated[float, Field(gt=0.0)] = 120.0
    #: Identity of the intake gate that admitted this runtime (N11 capability 0).
    #: Empty for locally authored packages; populated for a hosted-concierge
    #: runtime so the scorecard's execution identity names the intake-passed
    #: artifact (format, digests, sandbox report) rather than a bare upload.
    intake_identity: str = ""
    #: Digest of the sandbox policy the intake ran under, so a run cannot claim
    #: an intake that happened under weaker isolation.
    sandbox_policy_digest: str = ""
    #: Container resource envelope, part of the digest-covered identity so a
    #: run cannot claim different limits than it executed under (B4).
    container_memory: str = "2g"
    container_cpus: str = "2.0"
    container_pids_limit: str = "256"

    @model_validator(mode="after")
    def _identity_is_pinned(self) -> Self:
        if self.kind in {RuntimeKind.LOCAL, RuntimeKind.TRUSTED_IN_PROCESS}:
            if not self.command and not self.entrypoint:
                raise TaskContractError(f"{self.kind.value} runtime needs command or entrypoint")
        elif self.kind is RuntimeKind.HUGGINGFACE:
            if not self.model or not self.revision:
                raise TaskContractError("huggingface runtime needs model and revision")
        elif self.kind is RuntimeKind.OPENAI_COMPATIBLE:
            if not self.model or not self.base_url:
                raise TaskContractError("openai-compatible runtime needs model and base_url")
        elif self.kind is RuntimeKind.CONTAINER and (not self.image or not self.image_digest):
            raise TaskContractError("container runtime needs image and image_digest")
        return self

    @property
    def identity(self) -> str:
        """Stable runtime identity covered by package and job digests."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def legacy_interface(port: dict[str, Any]) -> InterfaceSpec:
    """Translate a v0.2 port mapping into the canonical v0.3 interface."""
    port_id = str(port.get("id", ""))
    if port_id == "gym-policy":
        return InterfaceSpec(
            id="gym-policy",
            interaction_mode=InteractionMode.CLOSED_LOOP,
            observations=(str(port.get("observation") or "state"),),
            actions=(str(port.get("action") or "continuous-action"),),
        )
    if port_id == "video-predict":
        return InterfaceSpec(
            id="video-predict",
            interaction_mode=InteractionMode.SINGLE_TURN,
            observations=(str(port.get("observation") or "video-clip"),),
            outputs=(str(port.get("prediction") or "structured-prediction"),),
            features=("abstention",),
        )
    raise TaskContractError(f"unknown legacy port {port_id!r}")


def legacy_capability(port_id: str) -> CapabilitySpec:
    """Translate a v0.2 agent port into a capability declaration."""
    if port_id == "gym-policy":
        return CapabilitySpec(
            interface="gym-policy",
            interaction_modes=(InteractionMode.CLOSED_LOOP,),
            schema_wildcard=True,
        )
    if port_id == "video-predict":
        return CapabilitySpec(
            interface="video-predict",
            interaction_modes=(InteractionMode.SINGLE_TURN,),
            schema_wildcard=True,
        )
    raise TaskContractError(f"unknown legacy port {port_id!r}")
