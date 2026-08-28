"""Declarative gate evaluation from kernel-resolved evidence bindings.

The kernel evaluates ``scalar-dsl`` gates declaratively. Each gate declares
a named map of evidence locators (``inputs``) resolved against the scoring
``context`` — or a verified ``task://`` artifact — and a ``fail_when``
expression. The kernel resolves and hashes every binding itself, so a
verifier cannot fabricate a scalar signal; the DSL merely thresholds
kernel-owned evidence.

Gate discipline is three-valued (Kleene): a missing binding is *unknown*,
never an implicit pass. Adverse evidence outranks missing evidence — if the
expression is already TRUE (fail) regardless of the unknown term, it fails;
it is unassessable only when the truth truly cannot be determined; it
passes only when fully determined and not adverse.
"""

from __future__ import annotations

import ast
import functools
import operator as op
from pathlib import Path
from typing import Any

from or_audit.domain.enums import GateStatus
from or_audit.eval.enums import VerifierRealizationKind
from or_audit.eval.evidence import (
    _MISSING,
    EvidenceReference,
    normalize_locator,
    resolve_binding,
)
from or_audit.eval.gym_world import nonfinite_kind
from or_audit.eval.task import GateSpec
from or_audit.eval.vector import GateOutcome

#: A tri-state sentinel distinct from ``None`` and ``False``.
UNKNOWN = object()

_OPS: dict[type[ast.AST], Any] = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.In: lambda value, seq: value in seq,
    ast.NotIn: lambda value, seq: value not in seq,
}


def _truth(value: Any) -> Any:
    """Map a signal value to a tri-state truth: UNKNOWN never reads as False."""
    if value is UNKNOWN or value is None:
        return UNKNOWN
    if nonfinite_kind(value):
        # `bool()` of the tag is True and `bool(nan)` is True, either of which
        # would be a verdict invented from a divergence. We abstain when we
        # cannot assess.
        return UNKNOWN
    try:
        return bool(value)
    except Exception:
        return UNKNOWN


def _knot(t: Any) -> Any:
    if t is UNKNOWN:
        return UNKNOWN
    return not t


def _kand(a: Any, b: Any) -> Any:
    if a is False or b is False:
        return False
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    return True


def _kor(a: Any, b: Any) -> Any:
    if a is True or b is True:
        return True
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    return False


def _eval_node(node: ast.AST, signals: dict[str, Any]) -> Any:
    """Evaluate an expression node to a signal value or :data:`UNKNOWN`."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, signals)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id in ("null", "none"):
            return None
        # Missing bindings are UNKNOWN (abstention), never a silent falsy PASS.
        return signals.get(node.id, UNKNOWN)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, signals)
        if isinstance(node.op, ast.Not):
            return _knot(_truth(operand))
        if isinstance(node.op, ast.USub):
            return -operand if operand is not UNKNOWN else UNKNOWN
    if isinstance(node, ast.BoolOp):
        values = [_truth(_eval_node(v, signals)) for v in node.values]
        if isinstance(node.op, ast.And):
            return functools.reduce(_kand, values, True)
        return functools.reduce(_kor, values, False)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, signals)
        for comparator, comparator_node in zip(node.ops, node.comparators, strict=True):
            right = _eval_node(comparator_node, signals)
            if left is UNKNOWN or right is UNKNOWN or left is None or right is None:
                return UNKNOWN
            check = _OPS.get(type(comparator))
            if check is None:
                raise ValueError(f"unsupported operator {type(comparator).__name__}")
            if not check(left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def is_scalar_realization(gate: GateSpec) -> bool:
    """Whether a gate is a transparent ``scalar-dsl`` realization."""
    return (
        isinstance(gate.realization, VerifierRealizationKind)
        and gate.realization is VerifierRealizationKind.SCALAR_DSL
    )


def evaluate_gate(
    gate: GateSpec,
    context: dict[str, Any],
    *,
    task_root: Path | None = None,
) -> GateOutcome | None:
    """Evaluate a ``scalar-dsl`` gate over kernel-resolved evidence.

    Every named input binding is resolved against ``context`` (or a verified
    ``task://`` artifact) and hashed by the kernel; ``fail_when`` is
    evaluated over those values with three-valued gate discipline. Returns
    ``None`` for gates that are not ``scalar-dsl`` so the caller can route
    them to their declared realization.
    """
    if not is_scalar_realization(gate):
        return None
    bindings = gate.evidence_bindings
    if not bindings or not gate.fail_when:
        return None
    signals: dict[str, Any] = {}
    evidence: list[EvidenceReference] = []
    diverged: list[str] = []
    for name, locator in bindings.items():
        try:
            value, uri, digest = resolve_binding(
                locator,
                context=context,
                task_root=task_root,
                absent_default=gate.input_defaults.get(name, _MISSING),
            )
        except ValueError as exc:
            # `canonical.py` refuses to hash a non-finite float, and that
            # refusal is right: a value that is not a number is not a
            # measurement and must not enter the evidence chain as one. But it
            # used to escape three frames below the gate and abort a whole job
            # over one diverged step, so the operator saw a canonicalization
            # error instead of "this gate could not be assessed". Abstaining is
            # the honest outcome, and writing it down here is what keeps it
            # one: a refusal that is only a side effect of the digest layer is
            # one refactor away from becoming a fabricated pass.
            diverged.append(f"{name}: {exc}")
            signals[name] = UNKNOWN
            evidence.append(
                EvidenceReference(id=name, uri=normalize_locator(locator), digest="", signal=name)
            )
            continue
        kind = nonfinite_kind(value)
        if kind:
            # Reported, canonical, hashable - and still not a measurement: the
            # solver diverged, so there is no number for the predicate to
            # enforce. The digest is kept, because "we saw a nan here" is
            # itself evidence; only the verdict abstains.
            diverged.append(f"{name} reported {kind}")
            signals[name] = UNKNOWN
            evidence.append(EvidenceReference(id=name, uri=uri, digest=digest, signal=name))
        elif value is _MISSING or value is None:
            # Missing or explicitly null evidence is abstention, never falsy:
            # normalize to UNKNOWN so `x == true` cannot read as a PASS.
            signals[name] = UNKNOWN
            evidence.append(EvidenceReference(id=name, uri=uri, digest="", signal=name))
        else:
            signals[name] = value
            evidence.append(EvidenceReference(id=name, uri=uri, digest=digest, signal=name))
    try:
        tree = ast.parse(gate.fail_when, mode="eval")
        result = _truth(_eval_node(tree, signals))
    except Exception as exc:
        return GateOutcome(
            id=gate.id,
            status=GateStatus.NOT_ASSESSABLE,
            reason=f"fail_when expression error: {exc}",
            realization=VerifierRealizationKind.SCALAR_DSL.value,
            provenance=gate.provenance,
            evidence=tuple(evidence),
        )
    bound_digests = {ref.signal: ref.digest for ref in evidence if ref.digest}
    if result is True:
        return GateOutcome(
            id=gate.id,
            status=GateStatus.FAIL,
            reason=f"{gate.fail_when} holds (evidence digests {bound_digests})",
            realization=VerifierRealizationKind.SCALAR_DSL.value,
            provenance=gate.provenance,
            evidence=tuple(evidence),
        )
    if result is UNKNOWN:
        # From the tri-state signals, not from an empty digest: a non-finite
        # binding keeps its digest and is still unassessable.
        undetermined = sorted(name for name, value in signals.items() if value is UNKNOWN)
        reason = f"gate cannot be determined; missing evidence: {undetermined}"
        if diverged:
            reason = f"{reason}; non-finite engine output: {sorted(diverged)}"
        return GateOutcome(
            id=gate.id,
            status=GateStatus.NOT_ASSESSABLE,
            reason=reason,
            realization=VerifierRealizationKind.SCALAR_DSL.value,
            provenance=gate.provenance,
            evidence=tuple(evidence),
        )
    return GateOutcome(
        id=gate.id,
        status=GateStatus.PASS,
        reason=f"not({gate.fail_when}) (evidence digests {bound_digests})",
        realization=VerifierRealizationKind.SCALAR_DSL.value,
        provenance=gate.provenance,
        evidence=tuple(evidence),
    )
