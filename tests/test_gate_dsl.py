"""Conformance tests for the declarative gate DSL evaluator."""

from __future__ import annotations

import ast

import pytest

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.gate_dsl import evaluate_gate
from or_audit.eval.task import GateSpec, ThresholdBasis, numeric_boundaries


def _gate(
    gate_id: str = "g",
    source: str = "contact_force",
    fail_when: str = "contact_force > 1.5",
    kind: str = "force-threshold",
    threshold: float | None = 1.5,
    unit: str = "N",
) -> GateSpec:
    """A gate whose cited number is the number its predicate enforces.

    ``threshold`` is declared by default because the schema now requires the
    two to agree: a bare ``contact_force > 1.5`` is an uncited safety number,
    which is precisely what §2.2 refuses. Pass ``threshold=None`` to build a
    boolean gate.

    ``unit`` is stated on the gate as well as on the basis: the two are
    compared unconditionally, so a gate publishing no unit against a basis in
    N is a dropped dimension, not a shorthand.
    """
    basis = (
        ThresholdBasis(value=threshold, unit=unit, citation="fixture force envelope")
        if threshold is not None
        else None
    )
    return GateSpec(
        id=gate_id,
        source=source,
        fail_when=fail_when,
        kind=kind,
        threshold=threshold,
        unit=unit if threshold is not None else "",
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
        unit="mm",
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_raw_non_finite_signal_abstains_instead_of_exploding(value: float) -> None:
    """A diverged solver's number is not a verdict, and not a crash either.

    Original defect: ``canonical.py`` refuses to hash a non-finite float, so
    ``resolve_binding`` raised ``ValueError: non-finite float nan cannot be
    canonicalized`` and that escaped ``evaluate_gate`` unhandled - aborting a
    whole job from three frames below the gate, where every other unassessable
    condition in this system produces NOT_ASSESSABLE with a reason. The digest
    layer's refusal was accidentally preventing a fabricated pass; here it is
    a written-down outcome instead.
    """
    gate = _gate()
    outcome = evaluate_gate(gate, {"contact_force": value})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE
    assert "contact_force" in outcome.reason
    assert "non-finite" in outcome.reason


@pytest.mark.parametrize(
    ("tagged", "kind"),
    [
        ("__nonfinite__:nan", "nan"),
        ("__nonfinite__:+inf", "+inf"),
        ("__nonfinite__:-inf", "-inf"),
    ],
)
def test_a_recorded_non_finite_tag_abstains_and_names_the_divergence(
    tagged: str, kind: str
) -> None:
    """The trajectory route reaches the same verdict as the raw-float route.

    The recorder tags a non-finite float so the divergence survives into the
    trace and its digest instead of being flattened to ``0.0``. Comparing the
    tag against the threshold would raise and be reported as a broken
    *expression*, and coercing it would invent a number; the gate abstains and
    says which signal diverged and how.
    """
    gate = _gate()
    outcome = evaluate_gate(gate, {"contact_force": tagged})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE
    assert f"contact_force reported {kind}" in outcome.reason
    # The tag is canonical, so its digest is kept: "we saw a nan here" is
    # evidence, even though it supports no verdict.
    assert [ref.digest for ref in outcome.evidence] != [""]


def test_a_non_finite_boolean_signal_abstains_rather_than_reading_true() -> None:
    """``bool("__nonfinite__:nan")`` is True; that would be an invented verdict."""
    gate = _gate(
        source="info.diverged",
        fail_when="diverged",
        kind="divergence",
        threshold=None,
    )
    outcome = evaluate_gate(gate, {"info": {"diverged": "__nonfinite__:nan"}})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE


def _numeric_gate(fail_when: str, threshold: float | None = 1.5, unit: str = "N") -> GateSpec:
    """A gate over one input ``x``, so a predicate can be judged in isolation."""
    payload: dict[str, object] = {
        "id": "g",
        "inputs": {"x": "info.x"},
        "fail_when": fail_when,
    }
    if threshold is not None:
        payload["threshold"] = threshold
        payload["unit"] = unit
        payload["threshold_basis"] = {"value": threshold, "unit": unit, "citation": "spec"}
    return GateSpec.model_validate(payload)


@pytest.mark.parametrize(
    "fail_when",
    [
        "x > false or x > 1.5",
        "x > (not 999) or x > 1.5",
        "x >= true",
        "x < False",
        "x <= true and x > 1.5",
    ],
)
def test_ordering_against_a_boolean_is_refused(fail_when: str) -> None:
    """An ordering comparison against a boolean is an uncited numeric boundary.

    The DSL evaluates Python ordering, where ``False`` is 0. ``x > false or
    x > 1.5`` with ``threshold=1.5`` was accepted and returned FAIL at 0.5:
    the gate enforced a boundary at 0 while publishing a 1.5 N citation, and
    the threshold check could not see it, because reading a boolean as a
    number would refuse the honest ``unsafe == true`` gate.
    """
    with pytest.raises(TaskContractError, match="Python ordering"):
        _numeric_gate(fail_when)


def test_ordering_against_a_boolean_is_refused_without_a_threshold_too() -> None:
    """A gate declaring no threshold still may not enforce the boolean-as-0 bound.

    Without a threshold there is not even a citation to contradict: the
    predicate applied a boundary at 0 that no field on the gate mentions, and
    the "uncited inline number" check saw nothing to report.
    """
    with pytest.raises(TaskContractError, match="Python ordering"):
        _numeric_gate("x > false", threshold=None)


def test_equality_against_a_boolean_is_still_legal() -> None:
    """The boolean-gate pattern is the reason booleans are not read as numbers."""
    for fail_when in ("x == true", "x != false", "x == True", "x != False"):
        gate = _numeric_gate(fail_when, threshold=None)
        outcome = evaluate_gate(gate, {"info": {"x": True}})
        assert outcome is not None


def test_two_sided_inline_bounds_are_still_refused() -> None:
    """Both spellings of a two-sided predicate enforce a number no basis cites."""
    with pytest.raises(TaskContractError, match=r"enforces \[999\.0\]"):
        _numeric_gate("1.5 < x < 999")
    with pytest.raises(TaskContractError, match=r"enforces \[-999\.0\]"):
        _numeric_gate("x < -999 or x > 1.5")


@pytest.mark.parametrize("fail_when", ["x > 1", "x >= 1.0", "x != 1", "not x > 1"])
def test_int_float_and_negation_spellings_share_one_boundary(fail_when: str) -> None:
    """1 and 1.0 are the same cited number, and ``not`` does not hide the compare."""
    assert numeric_boundaries(ast.parse(fail_when, mode="eval")) == {1.0}
    assert _numeric_gate(fail_when, threshold=1.0).threshold == 1.0


def test_unknown_callable_is_refused() -> None:
    """``abs(x)`` is refused for the unbound name, so no boundary escapes unread."""
    with pytest.raises(TaskContractError, match="unknown signal"):
        _numeric_gate("abs(x) > 1.5")


def test_a_dropped_or_invented_unit_is_refused() -> None:
    """Unit agreement is not skippable by leaving one side empty.

    Guarding the comparison on both sides being truthy accepted a gate that
    published no unit against a basis in N (the dimension is dropped from the
    published claim) and its inverse (the dimension is invented), both while
    keeping the cited value.
    """
    for gate_unit, basis_unit in (("", "N"), ("N", "")):
        with pytest.raises(TaskContractError, match="drops or"):
            GateSpec.model_validate(
                {
                    "id": "g",
                    "inputs": {"x": "info.x"},
                    "fail_when": "x > 1.5",
                    "threshold": 1.5,
                    "unit": gate_unit,
                    "threshold_basis": {
                        "value": 1.5,
                        "unit": basis_unit,
                        "citation": "spec",
                    },
                }
            )


def test_a_dimensionless_threshold_is_legal() -> None:
    """Two empty units agree: a ratio has no physical dimension to publish."""
    gate = _numeric_gate("x > 1.5", unit="")
    assert gate.unit == ""
