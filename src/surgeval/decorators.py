"""SurgEval decorators: turn a plain Python class into a bindable agent.

The on-ramp contract is that a user writes ``predict`` or ``act`` and gets a
real :class:`~or_audit.eval.contracts.CapabilitySpec` inferred from the class —
not a blanket wildcard. A wildcard binding stays available as the zero-config
fallback, but it is *recorded* on the binding so the runner marks the job
``binding_mode = "wildcard"`` and :func:`describe_agent` tells the user their
binding was never verified against the task's schemas.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import CapabilitySpec, InteractionMode, RuntimeDescriptor, RuntimeKind
from or_audit.eval.enums import AgentKind

T = TypeVar("T")

#: Attribute stamped on a decorated class holding everything ``@agent`` inferred.
BINDING_ATTR = "__surgeval_binding__"

#: SHA-256 of zero bytes. ``to_agent_package`` describes an in-memory model that
#: has no weights file on disk yet, so the only honest pin is the empty digest;
#: ``surgeval init-agent`` writes a real file and pins that file's real digest.
EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

#: ``act(observation, step=...)`` is the policy wire protocol: one interaction mode.
_ACT_MODES: tuple[InteractionMode, ...] = (InteractionMode.CLOSED_LOOP,)
#: ``predict(item)`` is the predictor wire protocol, which the harness drives for
#: three task shapes. Declaring all three is not over-claiming: it is the literal
#: set of modes that speak this method signature. Schema binding — not the method
#: name — decides whether a *particular* task in one of those modes is compatible.
_PREDICT_MODES: tuple[InteractionMode, ...] = (
    InteractionMode.SINGLE_TURN,
    InteractionMode.COUNTERFACTUAL,
    InteractionMode.INTERACTIVE,
)

_SCHEMA_FIELDS = ("observations", "actions", "outputs", "features", "modalities")


@dataclass(frozen=True, slots=True)
class AgentBinding:
    """Everything ``@agent`` inferred about one decorated class.

    ``schema_wildcard`` is the honesty flag: ``True`` means the class declared no
    schemas, so the capability binds to any task on ``interface`` without the
    kernel ever checking that the model speaks that task's data shapes.

    There is deliberately no single ``kind``/``entrypoint`` field: a class that
    implements both ``act`` and ``predict`` has two runtime identities, and a
    stored one would be the primary mode's, which is what let a predictor package
    ship as a ``policy``. Ask for one with :meth:`kind_for` and
    :meth:`entrypoint_for`, or resolve it with :func:`publication_mode`.
    """

    cls_name: str
    interface: str
    interaction_modes: tuple[InteractionMode, ...]
    primary_mode: InteractionMode
    methods: tuple[str, ...]
    capability: CapabilitySpec
    schema_wildcard: bool
    agent_id: str
    version: str
    #: Explicit ``kind=`` from the decorator; empty means "derive it from the mode".
    declared_kind: str

    def identities(self) -> tuple[InteractionMode, ...]:
        """Return one representative mode per distinct runtime identity."""
        seen: dict[str, InteractionMode] = {}
        for mode in self.interaction_modes:
            seen.setdefault(entrypoint_symbol_for(mode), mode)
        return tuple(seen.values())

    def capability_for_interface(self, interface: str) -> CapabilitySpec:
        """Return this capability rebound to ``interface`` (re-validated)."""
        if interface == self.capability.interface:
            return self.capability
        payload = self.capability.model_dump()
        payload["interface"] = interface
        return _capability(payload)

    def published_capability(
        self, mode: InteractionMode, interface: str | None = None
    ) -> CapabilitySpec:
        """Return this capability narrowed to what one entrypoint can drive.

        A package names a single factory, so it may only advertise the modes that
        factory serves. Anything wider is a claim the package cannot honour: the
        task binds on the mode, then refuses on the kind the same package
        declared.
        """
        payload = self.capability.model_dump()
        payload["interface"] = interface or self.capability.interface
        payload["interaction_modes"] = served_modes(self, mode)
        return _capability(payload)

    def entrypoint_for(self, mode: InteractionMode) -> str:
        """Return the runner symbol the harness must call to drive ``mode``."""
        return entrypoint_symbol_for(mode)

    def kind_for(self, mode: InteractionMode) -> str:
        """Return the agent-kind slug matching how ``mode`` drives this class."""
        return self.declared_kind or agent_kind_for(mode)

    def supports(self, mode: InteractionMode) -> bool:
        return mode in self.interaction_modes


def entrypoint_symbol_for(mode: InteractionMode) -> str:
    """Return the runner factory symbol the harness calls for ``mode``."""
    return "load_policy" if mode is InteractionMode.CLOSED_LOOP else "load_predictor"


def agent_kind_for(mode: InteractionMode) -> str:
    """Return the agent-kind slug for ``mode``."""
    return (
        AgentKind.POLICY.value
        if mode is InteractionMode.CLOSED_LOOP
        else AgentKind.FROZEN_MODEL.value
    )


def served_modes(binding: AgentBinding, mode: InteractionMode) -> tuple[InteractionMode, ...]:
    """Return the declared modes one entrypoint symbol can actually drive.

    ``predict(item)`` speaks single-turn, counterfactual, and interactive through
    the same factory, so publishing one of them publishes all three. ``act()``
    speaks closed-loop and nothing else.
    """
    symbol = entrypoint_symbol_for(mode)
    return tuple(
        candidate
        for candidate in binding.interaction_modes
        if entrypoint_symbol_for(candidate) == symbol
    )


def publication_mode(
    binding: AgentBinding,
    requested: InteractionMode | str | None,
    *,
    option: str = "mode=",
) -> InteractionMode:
    """Return the one mode a package built from ``binding`` may declare.

    A package carries a single ``kind`` and a single entrypoint, so it can only
    honestly advertise the modes that pair reaches. A class implementing both
    ``act`` and ``predict`` spans two such identities, and publishing the union
    emits a ``policy``/``load_policy`` package that also claims predictor modes —
    which every predictor task then rejects at bind time, after the mode check
    already passed. Refuse instead of picking one silently; ``option`` names the
    caller's way of choosing (``--mode`` for the CLI, ``mode=`` for the SDK).
    """
    if requested is not None:
        try:
            mode = InteractionMode(requested)
        except ValueError as exc:
            raise TaskContractError(f"unknown interaction mode {requested!r}") from exc
        if not binding.supports(mode):
            declared = ", ".join(item.value for item in binding.interaction_modes)
            raise TaskContractError(
                f"requested mode {mode.value!r} is not one of the modes "
                f"{binding.cls_name} declares ({declared}); implement the matching "
                "method, or ask for a mode it has"
            )
        return mode
    if len(binding.identities()) > 1:
        choices = ", ".join(item.value for item in binding.interaction_modes)
        raise TaskContractError(
            f"{binding.cls_name} implements {' and '.join(binding.methods)}, which are two "
            "different runtime identities: act() publishes as a closed-loop policy "
            "(load_policy) and predict() as a predictor (load_predictor). One package "
            f"declares one identity, so pass {option} with one of: {choices}. Publish "
            "twice to ship both."
        )
    return binding.primary_mode


def capability_toml(capability: CapabilitySpec) -> str:
    """Render a ``[[capabilities]]`` block for a generated ``agent.toml``.

    Only non-empty schema lists are emitted, so a generated package reads the
    same way a hand-written one does: what is written down is what was declared.
    """
    lines = [
        "[[capabilities]]",
        f'interface = "{capability.interface}"',
        f"interaction_modes = {json.dumps([m.value for m in capability.interaction_modes])}",
        f"protocol_versions = {json.dumps(list(capability.protocol_versions))}",
    ]
    for field in _SCHEMA_FIELDS:
        values: tuple[str, ...] = getattr(capability, field)
        if values:
            lines.append(f"{field} = {json.dumps(list(values))}")
    if capability.schema_wildcard:
        lines.append("schema_wildcard = true")
    lines.append("")
    return "\n".join(lines)


def _capability(payload: dict[str, Any]) -> CapabilitySpec:
    try:
        return CapabilitySpec.model_validate(payload)
    except TaskContractError:
        raise
    except Exception as exc:  # pydantic validation: surface as a contract refusal
        raise TaskContractError(f"@agent could not build a capability: {exc}") from exc


def _declared_schemas(cls: type[Any], field: str) -> tuple[str, ...]:
    """Read an optional ``field`` class attribute as a tuple of schema slugs."""
    raw = getattr(cls, field, None)
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if callable(raw) or not isinstance(raw, Iterable):
        raise TaskContractError(
            f"@agent: {cls.__name__}.{field} must be a string or a list of schema "
            f"ids, got {type(raw).__name__}. Rename the attribute if it is not a "
            "schema declaration."
        )
    values = tuple(raw)
    if not all(isinstance(item, str) for item in values):
        raise TaskContractError(
            f"@agent: every entry of {cls.__name__}.{field} must be a schema id string"
        )
    return values


def _infer_modes(
    cls: type[Any], explicit: InteractionMode | None
) -> tuple[tuple[InteractionMode, ...], InteractionMode, tuple[str, ...]]:
    has_act = callable(getattr(cls, "act", None))
    has_predict = callable(getattr(cls, "predict", None))
    if not has_act and not has_predict:
        raise TaskContractError(
            f"@agent: {cls.__name__} implements neither act(observation, step=...) "
            "nor predict(item). Add one: act() binds closed-loop tasks, predict() "
            "binds single-turn, counterfactual, and interactive tasks."
        )
    methods = tuple(
        name for name, present in (("act", has_act), ("predict", has_predict)) if present
    )
    modes: tuple[InteractionMode, ...] = ()
    if has_act:
        modes += _ACT_MODES
    if has_predict:
        modes += _PREDICT_MODES
    primary = InteractionMode.CLOSED_LOOP if has_act else InteractionMode.SINGLE_TURN
    if explicit is None:
        return modes, primary, methods
    if explicit not in modes:
        wanted = (
            "act(observation, step=...)"
            if explicit is InteractionMode.CLOSED_LOOP
            else "predict(item)"
        )
        raise TaskContractError(
            f"@agent: {cls.__name__} declares interaction_mode={explicit.value!r} but "
            f"implements {', '.join(methods)}. A {explicit.value} agent must implement {wanted}."
        )
    return (explicit,), explicit, methods


def _build_binding(
    cls: type[Any],
    *,
    interface: str,
    explicit_mode: InteractionMode | None,
    agent_id: str,
    version: str,
    kind: str,
) -> AgentBinding:
    modes, primary, methods = _infer_modes(cls, explicit_mode)
    declared = {field: _declared_schemas(cls, field) for field in _SCHEMA_FIELDS}
    wildcard = not any(declared.values())
    capability = _capability(
        {
            "interface": interface,
            "interaction_modes": modes,
            "schema_wildcard": wildcard,
            **declared,
        }
    )
    return AgentBinding(
        cls_name=cls.__name__,
        interface=interface,
        interaction_modes=modes,
        primary_mode=primary,
        methods=methods,
        capability=capability,
        schema_wildcard=wildcard,
        agent_id=agent_id if "/" in agent_id else f"custom/{agent_id}",
        version=version,
        declared_kind=kind,
    )


def agent(
    interface: str = "gym-policy",
    *,
    interaction_mode: InteractionMode | str | None = None,
    agent_id: str = "custom-agent",
    version: str = "0",
    kind: str = "",
) -> Callable[[type[T]], type[T]]:
    """Mark a Python class as a SurgEval agent and infer its capability.

    The interaction modes come from the methods the class actually implements —
    ``act`` binds closed-loop tasks, ``predict`` binds the predictor-driven task
    shapes, both declares both. Pass ``interaction_mode`` only to *narrow* that
    inference. Optional class attributes ``observations``, ``actions``,
    ``outputs``, ``features``, and ``modalities`` (each a string, list, or tuple
    of schema ids) become the capability's schema declaration; a class that
    declares none falls back to a wildcard binding, which is recorded so the job
    and :func:`describe_agent` say so out loud. ``kind`` overrides the agent-kind
    slug for tasks that accept a narrower kind than the method shape implies —
    ``world-model``, ``vlm``, ``panel``.
    """
    declared_mode = (
        None
        if interaction_mode is None
        else (
            interaction_mode
            if isinstance(interaction_mode, InteractionMode)
            else InteractionMode(interaction_mode)
        )
    )

    def decorator(cls: type[T]) -> type[T]:
        binding = _build_binding(
            cls,
            interface=interface,
            explicit_mode=declared_mode,
            agent_id=agent_id,
            version=version,
            kind=kind,
        )

        def to_agent_package(
            override_interface: str | None = None,
            *,
            mode: InteractionMode | str | None = None,
        ) -> AgentPackage:
            """Describe this class as an in-memory package for one interaction mode.

            ``mode`` is required when the class implements both ``act`` and
            ``predict``: the package it returns names one entrypoint, so it must
            not claim the other identity's modes. Single-identity classes need no
            argument, which keeps the ordinary decorator path one line.
            """
            published = publication_mode(binding, mode)
            iface = override_interface or binding.interface
            return AgentPackage(
                format_version="1",
                id=binding.agent_id,
                agent_version=binding.version,
                kind=binding.kind_for(published),
                capabilities=(binding.published_capability(published, iface),),
                weights_path="weights.json",
                weights_pin=EMPTY_FILE_SHA256,
                runtime=RuntimeDescriptor(
                    kind=RuntimeKind.LOCAL,
                    entrypoint=f"runner.py:{binding.entrypoint_for(published)}",
                ),
            )

        setattr(cls, BINDING_ATTR, binding)
        cls.to_agent_package = staticmethod(to_agent_package)  # type: ignore[attr-defined]
        return cls

    return decorator


def is_agent(target: Any) -> bool:
    """Return whether ``target`` (class or instance) carries an ``@agent`` binding."""
    return isinstance(getattr(target, BINDING_ATTR, None), AgentBinding)


def binding_for(target: Any) -> AgentBinding:
    """Return the ``@agent`` binding for a class or instance, or refuse.

    The refusal names the fix rather than silently synthesizing a wildcard,
    because an undecorated class has told the harness nothing it can verify.
    """
    binding = getattr(target, BINDING_ATTR, None)
    if not isinstance(binding, AgentBinding):
        name = getattr(target, "__name__", type(target).__name__)
        raise TaskContractError(
            f"{name} is not a SurgEval agent: decorate it with "
            '@surgeval.agent(interface="<task interface id>").'
        )
    return binding


def capability_for(target: Any) -> CapabilitySpec:
    """Return the capability ``@agent`` inferred for a decorated class."""
    return binding_for(target).capability


def describe_agent(target: Any) -> str:
    """Render a human summary of what ``@agent`` inferred, including honesty flags."""
    binding = binding_for(target)
    cap = binding.capability
    signatures = {"act": "act(observation, step=...)", "predict": "predict(item)"}
    lines = [
        f"agent: {binding.agent_id}@{binding.version}",
        f"  class        {binding.cls_name}",
        f"  interface    {binding.interface}",
        f"  methods      {', '.join(signatures[name] for name in binding.methods)}",
        f"  modes        {', '.join(mode.value for mode in binding.interaction_modes)}",
    ]
    # A class with one runtime identity states it; a class with both methods has
    # two, and naming only the primary one is what let a predictor ship as a
    # policy. List them instead, and say who chooses.
    identities = binding.identities()
    if len(identities) == 1:
        lines.append(f"  kind         {binding.kind_for(identities[0])}")
        lines.append(f"  entrypoint   runner.py:{binding.entrypoint_for(identities[0])}")
    else:
        for mode in identities:
            served = ", ".join(item.value for item in served_modes(binding, mode))
            lines.append(
                f"  identity     kind {binding.kind_for(mode)} via "
                f"runner.py:{binding.entrypoint_for(mode)} ({served})"
            )
        lines.append(
            "               two identities: publishing picks one "
            "(`init-agent --mode`, `to_agent_package(mode=...)`)"
        )
    if binding.schema_wildcard:
        lines.extend(
            [
                "  binding      WILDCARD (unverified)",
                "               This class declares no schemas, so it binds to any task on",
                f"               interface {binding.interface!r} and the kernel cannot check that",
                "               the model speaks that task's data shapes. Every job records",
                '               binding_mode = "wildcard". To get a verified binding, declare',
                "               observations / actions / outputs / features / modalities as class",
                "               attributes, or run with strict_schemas=True to refuse instead.",
            ]
        )
    else:
        lines.append("  binding      verified against declared schemas")
        for field in _SCHEMA_FIELDS:
            values: tuple[str, ...] = getattr(cap, field)
            if values:
                lines.append(f"  {field:<12} {', '.join(values)}")
    return "\n".join(lines)


__all__ = [
    "BINDING_ATTR",
    "EMPTY_FILE_SHA256",
    "AgentBinding",
    "agent",
    "agent_kind_for",
    "binding_for",
    "capability_for",
    "capability_toml",
    "describe_agent",
    "entrypoint_symbol_for",
    "is_agent",
    "publication_mode",
    "served_modes",
]
