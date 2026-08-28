"""Typed gate and metric vectors with declarative projection rules."""

from __future__ import annotations

import math
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.domain.enums import GateStatus
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.contracts import (
    GateProjectionPolicy,
    MetricDirection,
    MetricKind,
)
from or_audit.eval.evidence import EvidenceReference
from or_audit.eval.task import ProjectionSpec


class GateOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    status: GateStatus
    reason: str = ""
    #: Declared realization class that produced this outcome (enum value or slug).
    realization: str = "scalar-dsl"
    #: Human description of the oracle/evidence path.
    provenance: str = ""
    #: Kernel-resolved, kernel-hashed evidence backing the outcome.
    evidence: tuple[EvidenceReference, ...] = ()
    #: Optional uncertainty/confidence for non-DSL realizations.
    confidence: float | None = None
    abstained: bool = False

    @model_validator(mode="after")
    def _confidence_is_a_number(self) -> Self:
        """``None`` means "not reported"; ``nan`` is a number that is not one.

        Same defect as a non-finite metric value, one field over: this is
        hashed into the job head, ``canonical_json`` cannot render it, and
        ``score_context``'s ``min(max(float(c), 0.0), 1.0)`` clamp does not
        catch it - every comparison against ``nan`` is False, so it passes
        through unchanged.
        """
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise TaskContractError(
                f"gate {self.id} confidence {self.confidence!r} is not a number; report null "
                "when no confidence was measured"
            )
        return self


class MetricOutcome(BaseModel):
    """One typed metric; ``None`` is explicitly unassessable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    value: bool | float | str | None
    kind: MetricKind | None = None
    unit: str = ""
    direction: MetricDirection = MetricDirection.NEUTRAL
    headline: bool = False

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_kind(cls, raw: Any) -> Any:
        if not isinstance(raw, dict) or raw.get("kind") is not None:
            return raw
        data = dict(raw)
        value = data.get("value")
        if isinstance(value, bool) or value is None:
            data["kind"] = MetricKind.BOOLEAN.value
        elif isinstance(value, str):
            data["kind"] = MetricKind.CATEGORICAL.value
        else:
            data["kind"] = MetricKind.CONTINUOUS.value
        return data

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Self:
        if self.value is None:
            return self
        if self.kind is MetricKind.BOOLEAN and not isinstance(self.value, bool):
            raise TaskContractError(f"boolean metric {self.id} requires true, false, or null")
        if self.kind is MetricKind.CONTINUOUS and (
            isinstance(self.value, bool) or not isinstance(self.value, int | float)
        ):
            raise TaskContractError(f"continuous metric {self.id} requires a number or null")
        if self.kind is MetricKind.CATEGORICAL and not isinstance(self.value, str):
            raise TaskContractError(f"categorical metric {self.id} requires text or null")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            # A diverged solver's nan/inf is not a measurement, and the one
            # thing it must never become is a number. Refused on the published
            # record rather than only at the boundary because this value is
            # hashed into the job head: `canonical_json` cannot render it, so a
            # non-finite metric used to escape as a ValueError from inside head
            # computation - the operator got a canonicalization error rather
            # than a statement about their metric. ``score_context`` records
            # the honest answer (null, unassessable), so reaching here means a
            # construction path that bypassed it.
            raise TaskContractError(
                f"metric {self.id} value {self.value!r} is not a measurement; a non-finite "
                "value must be recorded as null (unassessable), never as a number"
            )
        return self


class TrialVector(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: str
    agent_identity: str
    seed: Annotated[int, Field(ge=0)]
    gates: tuple[GateOutcome, ...]
    metrics: tuple[MetricOutcome, ...]

    @model_validator(mode="after")
    def _one_headline(self) -> Self:
        headlines = [metric for metric in self.metrics if metric.headline]
        if len(headlines) != 1:
            raise TaskContractError(
                f"a trial vector must mark exactly one headline metric, got {len(headlines)}"
            )
        return self

    @property
    def headline(self) -> MetricOutcome:
        return next(metric for metric in self.metrics if metric.headline)

    def metric(self, metric_id: str) -> MetricOutcome | None:
        return next((metric for metric in self.metrics if metric.id == metric_id), None)

    def gate(self, gate_id: str) -> GateOutcome | None:
        return next((gate for gate in self.gates if gate.id == gate_id), None)

    @property
    def any_gate_failed(self) -> bool:
        return any(gate.status is GateStatus.FAIL for gate in self.gates)

    @property
    def any_gate_unassessable(self) -> bool:
        return any(gate.status is GateStatus.NOT_ASSESSABLE for gate in self.gates)

    def __float__(self) -> float:
        raise ScoreContractError(
            "a trial vector has no scalar value; use a pinned declarative projection"
        )

    def __int__(self) -> int:
        return int(self.__float__())

    def __bool__(self) -> bool:
        raise ScoreContractError(
            "a trial vector has no truth value; read a named metric or gate outcome"
        )


def project(vector: TrialVector, spec: ProjectionSpec) -> float:
    """Apply a digestable declarative rule to an authoritative vector.

    Every return below is finite by construction: TOML spells ``inf`` and
    ``nan``, so the declared reward values are checked here, and the source
    metric cannot hold a non-finite number (see
    :meth:`MetricOutcome._value_matches_kind`). This is the one point every
    projection path passes through, and its result is hashed into the job head
    as ``TrialRecord.projection`` - a non-finite reward would surface as a
    canonicalization error three frames down rather than as a statement about
    the projection.
    """
    for field, value in (("true_value", spec.true_value), ("false_value", spec.false_value)):
        if not math.isfinite(value):
            raise TaskContractError(
                f"projection {spec.identity} declares {field}={value!r}; a reward must be a "
                "finite number"
            )
    if vector.any_gate_unassessable:
        if spec.gate_unassessable is GateProjectionPolicy.ZERO:
            return spec.false_value
        raise ScoreContractError("cannot project a trial whose gates are unassessable")
    if vector.any_gate_failed:
        if spec.gate_failure is GateProjectionPolicy.ZERO:
            return spec.false_value
        raise ScoreContractError("projection refuses a failed hard gate")
    for metric_id in spec.require_false_metrics:
        outcome = vector.metric(metric_id)
        if outcome is None:
            raise TaskContractError(f"{spec.identity} requires a {metric_id!r} metric")
        if outcome.value is None:
            raise ScoreContractError(f"projection metric {metric_id!r} is unassessable")
        if not isinstance(outcome.value, bool):
            raise TaskContractError(f"projection guard {metric_id!r} must be boolean")
        if outcome.value:
            return spec.false_value
    source = vector.metric(spec.source_metric)
    if source is None:
        raise TaskContractError(f"{spec.identity} requires a {spec.source_metric!r} source metric")
    if source.value is None:
        raise ScoreContractError(f"projection source {spec.source_metric!r} is unassessable")
    if isinstance(source.value, bool):
        return spec.true_value if source.value else spec.false_value
    if isinstance(source.value, int | float):
        return float(source.value)
    raise TaskContractError("categorical metrics cannot be projected to a reward")
