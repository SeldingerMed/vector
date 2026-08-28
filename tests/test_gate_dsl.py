"""Conformance tests for the declarative gate DSL evaluator."""

from __future__ import annotations

import pytest

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.gate_dsl import evaluate_gate
from or_audit.eval.task import GateSpec, ThresholdBasis


def _gate(
    gate_id: str = "g",
    source: str = "contact_force",
    fail_when: str = "contact_force > 1.5",
    kind: str = "force-threshold",
    threshold: float | None = 1.5,
) -> GateSpec:
    """A gate whose cited number is the number its predicate enforces.

    ``threshold`` is declared by default because the schema now requires the
    two to agree: a bare ``contact_force > 1.5`` is an uncited safety number,
    which is precisely what §2.2 refuses. Pass ``threshold=None`` to build a
    boolean gate.
    """
    basis = (
        ThresholdBasis(value=threshold, unit="N", citation="fixture force envelope")
        if threshold is not None
        else None
    )
    return GateSpec(
        id=gate_id,
        source=source,
        fail_when=fail_when,
        kind=kind,
        threshold=threshold,
        threshold_basis=basis,
    )


def test_gate_passes_when_below_threshold() -> None:
    gate = _gate()
    outcome = evaluate_gate(gate, {"contact_force": 1.2})
    assert outcome is not None
    assert outcome.status is GateStatus.PASS


def test_gate_fails_when_above_threshold() -> None:
    gate = _gate()
    outcome = evaluate_gate(gate, {"contact_force": 2.0})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_gate_not_assessable_when_signal_missing() -> None:
    gate = _gate()
    outcome = evaluate_gate(gate, {})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE


def test_gate_resolves_dotted_source_to_short_name() -> None:
    gate = _gate(source="oracle.catheter.contact_force", fail_when="contact_force > 1.5")
    outcome = evaluate_gate(gate, {"oracle": {"catheter": {"contact_force": 2.0}}})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_gate_resolves_dotted_source_to_full_path() -> None:
    gate = _gate(source="oracle.catheter.contact_force", fail_when="contact_force > 1.5")
    outcome = evaluate_gate(gate, {"oracle": {"catheter": {"contact_force": 0.8}}})
    assert outcome is not None
    assert outcome.status is GateStatus.PASS


def test_gate_boolean_equality() -> None:
    gate = _gate(
        source="cbd_violation",
        fail_when="cbd_violation == true",
        kind="spatial-exclusion",
        threshold=None,
    )
    fail = evaluate_gate(gate, {"cbd_violation": True})
    assert fail is not None
    assert fail.status is GateStatus.FAIL
    passing = evaluate_gate(gate, {"cbd_violation": False})
    assert passing is not None
    assert passing.status is GateStatus.PASS


def test_gate_with_threshold_field() -> None:
    gate = GateSpec(
        id="overshoot",
        source="max_overshoot_mm",
        fail_when="max_overshoot_mm > 0.5",
        kind="spatial-exclusion",
        threshold=0.5,
        threshold_basis=ThresholdBasis(
            value=0.5, unit="mm", citation="haptic boundary tolerance v1"
        ),
    )
    outcome = evaluate_gate(gate, {"max_overshoot_mm": 0.7})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_a_compound_gate_may_not_inline_two_uncited_numbers() -> None:
    """One gate carries one cited threshold, so two inline bounds cannot both be cited.

    Splitting is also the more legible artifact: two gates tell a reader which
    bound failed, where one compound gate reports a single opaque FAIL.
    """
    with pytest.raises(TaskContractError, match="declares no threshold"):
        GateSpec(
            id="g",
            inputs={"force": "force", "speed": "speed"},
            fail_when="force > 1.5 and speed > 10",
            kind="force-threshold",
        )


def test_a_compound_gate_over_declared_bounds_is_supported() -> None:
    """The pattern ``lumen-nav-safe`` ships: bounds are inputs, not literals.

    Nothing about compound evaluation is restricted - only inline numbers that
    no basis explains. Declaring the bound as an input makes it a value the
    package states and the kernel resolves, which is what a gate should do.
    """
    gate = GateSpec(
        id="g",
        inputs={
            "force": "force",
            "speed": "speed",
            "force_limit": "force_limit",
            "speed_limit": "speed_limit",
        },
        fail_when="force > force_limit and speed > speed_limit",
        kind="force-threshold",
    )
    bounds = {"force_limit": 1.5, "speed_limit": 10}
    fail = evaluate_gate(gate, {"force": 2.0, "speed": 15, **bounds})
    assert fail is not None
    assert fail.status is GateStatus.FAIL
    passing = evaluate_gate(gate, {"force": 2.0, "speed": 5, **bounds})
    assert passing is not None
    assert passing.status is GateStatus.PASS


def test_gate_returns_none_when_not_scalar_realization() -> None:
    gate = GateSpec(
        id="manual_gate",
        inputs={"x": "label.x"},
        realization="human-panel",
        provenance="human panel review of x",
    )
    assert evaluate_gate(gate, {"label": {"x": 1}}) is None


def test_gate_rejects_unbound_name_at_load() -> None:
    """A typo such as 'unsfae' must be a load-time error, never a silent PASS."""
    with pytest.raises(TaskContractError, match="unknown signal"):
        _gate(fail_when="unsfae")


def test_gate_malformed_expression_is_refused_at_load() -> None:
    """A malformed or unbound fail_when is a load-time contract error, not a silent PASS."""
    with pytest.raises(TaskContractError):
        _gate(fail_when="contact_force >>> 1.5")


def test_conformance_two_tasks_same_kind_compute_identically() -> None:
    """Two tasks declaring force_threshold must compute the same outcome from the same signal."""
    gate_a = _gate(gate_id="airway_wall", source="contact_force", fail_when="contact_force > 1.5")
    gate_b = _gate(gate_id="vessel_wall", source="contact_force", fail_when="contact_force > 1.5")
    for value in [0.5, 1.5, 2.0, 3.0]:
        out_a = evaluate_gate(gate_a, {"contact_force": value})
        out_b = evaluate_gate(gate_b, {"contact_force": value})
        assert out_a is not None
        assert out_b is not None
        assert out_a.status is out_b.status, f"mismatch at contact_force={value}"


def test_gate_runtime_error_is_not_assessable() -> None:
    gate = _gate(fail_when="contact_force > 1.5")
    # A string where a number is expected is a runtime error, never a PASS.
    outcome = evaluate_gate(gate, {"contact_force": "2.0"})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE
