"""Versioned v0.3 contracts shared by tasks, agents, runners, and scorecards."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

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
    camera_calibration: dict[str, Any] = Field(default_factory=dict)
    joint_order: tuple[str, ...] = ()
    invalid_depth_encoding: str = ""
    privileged: bool = False

    @model_validator(mode="after")
    def _plugin_needs_schema(self) -> Self:
        if not self.adapter:
            raise TaskContractError(f"stream {self.id} requires an adapter plugin id")
        if not self.schema_id:
            raise TaskContractError(f"stream {self.id} requires an observation schema_id")
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
    def _non_empty_modes(self) -> Self:
        if not self.interaction_modes:
            raise TaskContractError(f"capability {self.interface} declares no interaction mode")
        return self

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
        cap_profiles: dict[Slug, StreamSpec] = {}
        if self.stream_profiles:
            for s in self.stream_profiles:
                if s.schema_id:
                    cap_profiles[s.schema_id] = s
                if s.id:
                    cap_profiles[s.id] = s
        for intf_stream in interface.streams:
            has_semantics = bool(
                intf_stream.unit
                or intf_stream.coordinate_frame
                or intf_stream.controller_id
                or intf_stream.dtype
                or intf_stream.shape
                or intf_stream.joint_order
                or intf_stream.invalid_depth_encoding
                or intf_stream.valid_range is not None
            )
            if has_semantics:
                matching_profile = cap_profiles.get(intf_stream.id) or cap_profiles.get(
                    intf_stream.schema_id
                )
                if matching_profile is None:
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
                if (
                    intf_stream.joint_order
                    and matching_profile.joint_order != intf_stream.joint_order
                ):
                    return False
                if (
                    intf_stream.invalid_depth_encoding
                    and matching_profile.invalid_depth_encoding
                    != intf_stream.invalid_depth_encoding
                ):
                    return False
                if (
                    intf_stream.valid_range is not None
                    and matching_profile.valid_range != intf_stream.valid_range
                ):
                    return False
            if intf_stream.privileged and not self.accepts_privileged:
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
