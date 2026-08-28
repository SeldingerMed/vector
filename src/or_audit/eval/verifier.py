"""Run task-owned verifiers separately and validate typed vector output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.evidence import _MISSING, EvidenceReference, resolve_binding
from or_audit.eval.gate_dsl import evaluate_gate, is_scalar_realization
from or_audit.eval.gym_world import nonfinite_kind
from or_audit.eval.plugins import VerifierRuntime, load_verifier_runtime
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import GateOutcome, MetricOutcome, TrialVector


def score_context(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent_identity: str,
    seed: int,
    context: dict[str, Any],
    runtime: VerifierRuntime | None = None,
) -> TrialVector:
    """Score oracle evidence in a verifier process that is separate from the agent."""
    if not task.verifier.entrypoint:
        raise TaskContractError(f"task {task.id} has no verifier entrypoint")
    verifier = runtime or load_verifier_runtime(task_dir, task.verifier.entrypoint)
    owns_runtime = runtime is None
    try:
        raw = verifier.score(context)
    finally:
        if owns_runtime:
            close = getattr(verifier, "close", None)
            if callable(close):
                close()
    if not isinstance(raw, dict):
        raise TaskContractError(f"task {task.id} verifier must return an object")
    raw_gates = raw.get("gates")
    raw_metrics = raw.get("metrics")
    if not isinstance(raw_gates, dict) or not isinstance(raw_metrics, dict):
        raise TaskContractError("verifier output requires gates and metrics objects")

    declared_gates = [gate.id for gate in task.verifier.gates]
    declared_metrics = [metric.id for metric in task.verifier.metrics]
    scalar_ids = {gate.id for gate in task.verifier.gates if is_scalar_realization(gate)}
    # scalar-dsl gates are evaluated by the kernel over evidence bindings, so
    # the verifier need not (and must not) self-report their status. Only
    # non-DSL declared-realization gates are attributed to the verifier.
    required_gates = [gid for gid in declared_gates if gid not in scalar_ids]
    missing_gates = [gid for gid in required_gates if gid not in raw_gates]
    if missing_gates:
        raise TaskContractError(
            f"verifier did not report required declared gates {sorted(missing_gates)}"
        )
    if set(raw_metrics) != set(declared_metrics):
        raise TaskContractError(
            f"verifier metrics {sorted(raw_metrics)} do not match declared metrics "
            f"{sorted(declared_metrics)}"
        )

    gates = []
    for gate_spec in task.verifier.gates:
        gate_id = gate_spec.id
        dsl_outcome = evaluate_gate(gate_spec, context, task_root=task_dir)
        if dsl_outcome is not None:
            # scalar-dsl: outcome is fully kernel-resolved and kernel-hashed.
            gates.append(dsl_outcome)
            continue
        # Non-DSL declared realization (learned/spatial/temporal/human/sim):
        # accept the realization's outcome, but stamp its declared realization
        # and provenance so it is never an opaque self-reported scalar status.
        outcome = raw_gates[gate_id]
        if not isinstance(outcome, dict):
            raise TaskContractError(f"gate {gate_id} outcome must be an object")
        raw_status = outcome.get("status")
        if not isinstance(raw_status, str):
            raise TaskContractError(f"gate {gate_id} has invalid status")
        try:
            status = GateStatus(raw_status)
        except ValueError as exc:
            raise TaskContractError(f"gate {gate_id} has invalid status") from exc
        reason = outcome.get("reason", "")
        if not isinstance(reason, str):
            raise TaskContractError(f"gate {gate_id} reason must be text")
        realization = getattr(gate_spec.realization, "value", gate_spec.realization)
        # The realization's inputs are still kernel-resolved and kernel-hashed,
        # so an opaque verdict cannot fabricate the evidence it consumed.
        evidence: list[EvidenceReference] = []
        missing_input = False
        for name, locator in gate_spec.evidence_bindings.items():
            value, uri, digest = resolve_binding(
                locator,
                context=context,
                task_root=task_dir,
                absent_default=gate_spec.input_defaults.get(name, _MISSING),
            )
            if value is _MISSING or value is None:
                missing_input = True
                evidence.append(EvidenceReference(id=name, uri=uri, digest="", signal=name))
            else:
                evidence.append(EvidenceReference(id=name, uri=uri, digest=digest, signal=name))
        # Accept PASS/FAIL only when every declared input has kernel evidence.
        # A missing/null input forces abstention, never an evidence-free verdict.
        if missing_input:
            gates.append(
                GateOutcome(
                    id=gate_id,
                    status=GateStatus.NOT_ASSESSABLE,
                    reason="declared realization missing kernel-resolved evidence input(s)",
                    realization=str(realization),
                    provenance=gate_spec.provenance,
                    evidence=tuple(evidence),
                    abstained=True,
                )
            )
            continue
        confidence = outcome.get("confidence")
        if confidence is not None and not isinstance(confidence, int | float):
            raise TaskContractError(f"gate {gate_id} confidence must be numeric")
        if confidence is not None:
            confidence = min(max(float(confidence), 0.0), 1.0)
        abstained = bool(outcome.get("abstained", False))
        if abstained:
            # Abstention is never an implicit pass/fail. A verifier that opts
            # out yields NOT_ASSESSABLE regardless of any status it also
            # reported; downstream reward logic must treat the gate as
            # unassessable, never as evidence of a pass.
            status = GateStatus.NOT_ASSESSABLE
            if not reason:
                reason = f"gate {gate_id} verifier abstained"
        gates.append(
            GateOutcome(
                id=gate_id,
                status=status,
                reason=reason,
                realization=str(realization),
                provenance=gate_spec.provenance,
                evidence=tuple(evidence),
                confidence=confidence,
                abstained=abstained,
            )
        )

    metrics = []
    for metric_id in declared_metrics:
        definition = task.metric(metric_id)
        value = raw_metrics[metric_id]
        if value is not None and not isinstance(value, bool | int | float | str):
            raise TaskContractError(f"metric {metric_id} returned an unsupported value")
        if nonfinite_kind(value):
            # A diverged solver reported nan/+inf/-inf, or handed back the
            # recorder's tag for one. Either way there is no number here, and
            # the one thing it must never become is a number: 0.0 would be a
            # fabricated safety reading, and the raw value cannot be hashed
            # into the job head at all. Null is the same answer the gate gives
            # for the same evidence, and it keeps a diverged run reportable
            # instead of unscoreable. The trajectory still carries the tagged
            # value, so *why* the metric is unassessable stays on the record.
            value = None
        if (
            definition.kind.value == "categorical"
            and value is not None
            and value not in definition.categories
        ):
            raise TaskContractError(
                f"categorical metric {metric_id} returned undeclared category {value!r}"
            )
        metrics.append(
            MetricOutcome(
                id=metric_id,
                value=value,
                kind=definition.kind,
                unit=definition.unit,
                direction=definition.direction,
                headline=metric_id == task.verifier.headline,
            )
        )
    return TrialVector(
        task_id=task.id,
        task_version=task.task_version,
        agent_identity=agent_identity,
        seed=seed,
        gates=tuple(gates),
        metrics=tuple(metrics),
    )
