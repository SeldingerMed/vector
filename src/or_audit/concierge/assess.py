"""Capability 1: assess an intake-passed model and *draft* a capability.

The agent probes I/O schema, modality, and action space from inside the same
sandbox boundary intake established, then writes down what it saw. It does not
acquire authority by doing so. Two things stay with the kernel:

* satisfaction is decided by :func:`or_audit.eval.bind.assert_bind` over a
  declared :class:`~or_audit.eval.contracts.CapabilitySpec` — never by this
  module, which only proposes the declaration; and
* a proposal is inert until a named party confirms it.

Inertness is structural, not advisory. :class:`AssessProposal` is a distinct
type: it is not a ``CapabilitySpec``, has no ``satisfies`` method, and cannot
be passed where a confirmed capability is required — mypy rejects it, and
:func:`assert_confirmed` raises at runtime for the untyped callers.
"""

from __future__ import annotations

from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.audit.canonical import digest
from or_audit.concierge.intake import IntakeResult, runtime_descriptor
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    CapabilitySpec,
    InteractionMode,
    RuntimeDescriptor,
    SHA256Hex,
    Slug,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelProbeReport(_Frozen):
    """What a sandboxed probe actually observed about a model.

    ``evidence`` is required and non-empty: an assessment with no record of
    what was probed is a guess wearing a schema, and it would be indexed by
    the eventual scorecard as though it had been measured.
    """

    interface: Slug
    interaction_modes: tuple[InteractionMode, ...]
    protocol_versions: tuple[str, ...] = ("1",)
    observations: tuple[Slug, ...] = ()
    actions: tuple[Slug, ...] = ()
    outputs: tuple[Slug, ...] = ()
    features: tuple[Slug, ...] = ()
    #: Modality adapter plugin ids the model consumes, as probed.
    modalities: tuple[Slug, ...] = ()
    #: Free-text action-space description, recorded for the human confirming.
    action_space: str = ""
    evidence: tuple[str, ...]

    @model_validator(mode="after")
    def _probed_something(self) -> Self:
        if not self.interaction_modes:
            raise TaskContractError(
                f"probe of interface {self.interface} observed no interaction mode; "
                "an unprobed shape cannot be drafted into a capability"
            )
        if not self.evidence:
            raise TaskContractError(
                f"probe of interface {self.interface} returned no evidence; refusing "
                "to draft a capability from an unrecorded assessment"
            )
        if not (self.actions or self.outputs):
            raise TaskContractError(
                f"probe of interface {self.interface} found neither an action nor an "
                "output; there is nothing for a task to bind to"
            )
        return self

    @property
    def probe_digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class ModelProbe(Protocol):
    """Injected sandboxed prober. This module performs no model I/O itself."""

    def __call__(self, intake: IntakeResult, /) -> ModelProbeReport: ...


class AssessProposal(_Frozen):
    """A *drafted* capability. Deliberately not a capability.

    Holds the same content a package would declare, but nothing accepts it as
    a capability: it has no ``satisfies``, and every scored path demands a
    :class:`ConfirmedCapability`.
    """

    intake_identity: SHA256Hex
    capability: CapabilitySpec
    runtime: RuntimeDescriptor
    evidence: tuple[str, ...]
    action_space: str = ""
    probe_digest: SHA256Hex

    @property
    def proposal_digest(self) -> str:
        """Identity of the exact draft a confirmation applies to."""
        return digest(self.model_dump(mode="json"))

    def describe(self) -> str:
        lines = [
            f"Proposed capability {self.capability.interface} (UNCONFIRMED)",
            f"  modes      {', '.join(mode.value for mode in self.capability.interaction_modes)}",
            f"  observes   {', '.join(self.capability.observations) or 'none'}",
            f"  acts       {', '.join(self.capability.actions) or 'none'}",
            f"  outputs    {', '.join(self.capability.outputs) or 'none'}",
            f"  modalities {', '.join(self.capability.modalities) or 'none'}",
            f"  action sp. {self.action_space or 'unreported'}",
            f"  intake     {self.intake_identity}",
            f"  runtime    {self.runtime.kind.value} ({self.runtime.identity})",
        ]
        lines.extend(f"  probed     {item}" for item in self.evidence)
        lines.append("  NOT USABLE until confirmed: a draft binds nothing.")
        return "\n".join(lines)


class ConfirmedCapability(_Frozen):
    """A proposal a named party accepted. The only input scored runs accept."""

    intake_identity: SHA256Hex
    capability: CapabilitySpec
    runtime: RuntimeDescriptor
    confirmed_by: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    proposal_digest: SHA256Hex

    @model_validator(mode="after")
    def _named_authority(self) -> Self:
        if not self.confirmed_by.strip():
            raise TaskContractError(
                "a confirmation must name who confirmed it: the authority record "
                "is the point of the step"
            )
        return self


def assess_model(intake: IntakeResult, *, probe: ModelProbe) -> AssessProposal:
    """Probe an intake-passed model and draft a capability for confirmation.

    Refuses a failed intake outright: assessment runs *inside* the boundary
    intake established, so probing a rejected artifact would execute exactly
    the code the gate refused.
    """
    if not intake.passed:
        detail = "; ".join(intake.refusals)
        raise TaskContractError(f"refusing to assess failed intake {intake.identity}: {detail}")
    runtime = runtime_descriptor(intake)
    report = probe(intake)
    capability = CapabilitySpec(
        interface=report.interface,
        interaction_modes=report.interaction_modes,
        protocol_versions=report.protocol_versions,
        observations=report.observations,
        actions=report.actions,
        outputs=report.outputs,
        features=report.features,
        modalities=report.modalities,
    )
    return AssessProposal(
        intake_identity=intake.identity,
        capability=capability,
        runtime=runtime,
        evidence=report.evidence,
        action_space=report.action_space,
        probe_digest=report.probe_digest,
    )


def confirm(proposal: AssessProposal, *, confirmed_by: str) -> ConfirmedCapability:
    """Accept one exact draft on behalf of a named party."""
    return ConfirmedCapability(
        intake_identity=proposal.intake_identity,
        capability=proposal.capability,
        runtime=proposal.runtime,
        confirmed_by=confirmed_by,
        proposal_digest=proposal.proposal_digest,
    )


def assert_confirmed(candidate: object) -> ConfirmedCapability:
    """Return ``candidate`` if it is a confirmation; otherwise refuse.

    The runtime half of the type-level rule, for callers that reached this
    point through JSON, ``Any``, or a dynamic plugin edge.
    """
    if isinstance(candidate, ConfirmedCapability):
        return candidate
    if isinstance(candidate, AssessProposal):
        raise TaskContractError(
            f"refusing to score against an unconfirmed assessment of interface "
            f"{candidate.capability.interface} (proposal "
            f"{candidate.proposal_digest}): the agent proposes, "
            "or_audit.eval.bind.assert_bind disposes, and a named party confirms "
            "before the first scored run"
        )
    raise TaskContractError(f"expected a confirmed capability, got {type(candidate).__name__}")
