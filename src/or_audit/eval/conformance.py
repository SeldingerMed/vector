"""Tier-1 conformance: the machine-checkable definition of a curated wrap.

next.md §2.2 promises three tiers, and only Tier 1 claims "pinned commit +
gate mapping + determinism/replay validation + license audit". This module is
that claim's implementation, so the promotion decision is a run, not an
opinion. Four checks, all four required:

``gate-state-availability``
    Every hard gate the package declares must actually resolve a ``pass`` or
    ``fail`` on this world at least once. A gate that is ``not_assessable``
    across every trial is a gate scoring state the world never reports — the
    §2.2/§2.3 forbidden case, whose honest fix is instrumenting the world or
    shipping the package as ``environment.metrics_only``, never synthesizing a
    gate. A metrics-only package is checked for the opposite property: that it
    declares no gates at all.

``license-audit``
    The package's declared SPDX license, through
    :func:`or_audit.eval.licensing.classify_license`. Restricted *and*
    unreviewed both fail: "some envs carry restrictive or contaminating
    licenses" (§2.1) is only firewalled if an unknown license is a refusal.

``evidence-replay``
    The stored trajectory must reconstitute the published vector through the
    task-owned verifier (``reconstitute.py``, which never steps the world).

``execution-determinism``
    The identical job is run twice into separate output directories and a
    :class:`~or_audit.eval.worlds.DeterminismClass` is *measured*: ``bitwise``
    when every trial's canonical trajectory digest matches, ``tolerance`` when
    the vectors match and the trajectories differ only by floats inside
    ``tolerance``, ``nondeterministic`` otherwise. Many wrapped engines (SOFA,
    PhysX) are only best-effort deterministic, so this is measured and
    recorded per world rather than assumed — and a declaration stronger than
    the measurement is refused.

License declaration contract (enforced, in resolution order):

1. ``wrap.json`` at the package root, top-level string field ``license``
   holding an SPDX id. This is what ``surgeval wrap`` writes.
2. ``license.toml`` at the package root with ``spdx = "<id>"``.
3. a ``license:<id>`` entry in ``task.toml`` ``[metadata].tags``. Tags are the
   open field; ``TaskMetadata`` forbids unknown keys, so this is the only
   in-``task.toml`` path.
4. an ``SPDX-License-Identifier: <id>`` marker line inside a bundled
   ``LICENSE``/``LICENSE.txt``/``LICENSE.md``.

A bundled license *text* with no SPDX marker is not classified from its prose,
and a package with no declaration at all is ``unknown`` — both fail the audit.
Tier 1 additionally requires the package to pin its world adapter
(``environment.adapter`` + ``adapter_digest``), to map gates rather than ship
``environment.metrics_only`` (§2.2 places a metrics-only wrap at Tier 0 by
construction: a passing gate-state check on a metrics-only package proves it
has no gates, which is honest and still not curated), and a measured
determinism class that is not ``nondeterministic``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.audit.canonical import digest
from or_audit.domain.enums import GateStatus
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.job import JobResult, read_job_config
from or_audit.eval.licensing import (
    LicenseStatus,
    LicenseVerdict,
    classify_license,
    declared_package_license,
)
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.reconstitute import assert_trajectory_matches_vector
from or_audit.eval.runner import builtin_random_agent, run_job
from or_audit.eval.sim.base import BACKEND_REAL, BACKEND_SYNTHETIC_STUB, BACKEND_UNKNOWN
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import DeterminismClass, determinism_at_least, world_kind_spec

#: Largest absolute float difference two trajectories may show and still be
#: called ``tolerance``-deterministic. Recorded in every report: a tolerance
#: class without its tolerance is not a measurement.
DEFAULT_TOLERANCE = 1e-9

CHECK_GATE_STATES = "gate-state-availability"
CHECK_LICENSE = "license-audit"
CHECK_EVIDENCE_REPLAY = "evidence-replay"
CHECK_DETERMINISM = "execution-determinism"

#: Every check a report must carry. A partial suite cannot mint a tier.
REQUIRED_CHECKS: tuple[str, ...] = (
    CHECK_GATE_STATES,
    CHECK_LICENSE,
    CHECK_EVIDENCE_REPLAY,
    CHECK_DETERMINISM,
)

#: Subdirectories of ``workdir`` the two measured runs are written to.
RUN_A = "run-a"
RUN_B = "run-b"

GymFactory = Callable[[TaskSpec], Any]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GateStateCount(_Frozen):
    """How often one declared gate reached each status across the trials."""

    id: str
    n_pass: int = Field(default=0, ge=0)
    n_fail: int = Field(default=0, ge=0)
    n_not_assessable: int = Field(default=0, ge=0)
    n_not_applicable: int = Field(default=0, ge=0)

    @property
    def assessed(self) -> int:
        """Trials in which the gate actually decided something."""
        return self.n_pass + self.n_fail


class DeterminismEvidence(_Frozen):
    """What the two-run measurement actually observed."""

    measured: DeterminismClass
    #: Strongest class the task or its installed adapter claims; ``unmeasured``
    #: means nothing was claimed and there is nothing to violate.
    declared: DeterminismClass = DeterminismClass.UNMEASURED
    tolerance: float
    #: Both runs produced byte-identical canonical trajectory digests.
    identical_digests: bool
    #: Both runs produced equal trial vectors (gates and metrics).
    vectors_equal: bool
    #: Largest absolute float delta seen between the two trajectories.
    max_float_delta: float = 0.0
    #: First difference that tolerance does not excuse; ``""`` when none.
    first_difference: str = ""


class ConformanceCheck(_Frozen):
    """One check's verdict plus the evidence it was decided on."""

    id: str
    passed: bool
    detail: str
    #: ``gate-state-availability`` evidence: per-gate status counts.
    gate_states: tuple[GateStateCount, ...] = ()
    #: ``license-audit`` evidence: the classified declaration.
    license: LicenseVerdict | None = None
    #: ``license-audit`` evidence: where the declaration was found.
    license_source: str = ""
    #: ``execution-determinism`` evidence: the two-run measurement.
    determinism: DeterminismEvidence | None = None


class ConformanceReport(_Frozen):
    """A package's Tier-1 verdict, with every check it was decided on."""

    task_id: str
    task_version: str
    task_digest: str
    world_kind: str
    #: ``module:symbol+sha256`` of the installed adapter, or ``"unattached"``.
    adapter_identity: str
    #: The package pins its world adapter, so a swapped adapter cannot run it.
    adapter_pinned: bool
    #: Tier-0 honesty label carried from ``WorldSpec.metrics_only``. A
    #: metrics-only wrap has no gate mapping, so §2.2 places it at Tier 0 no
    #: matter how clean the rest of the package is.
    metrics_only: bool = False
    #: Whether this world is one a policy *steps*, taken from the world kind's
    #: declared ``physics`` capability. A dataset-backed kind (frame-source,
    #: counterfactual, angiostress-contract) runs no engine at all, so its
    #: evidence is the pinned data and its verifier, not a physics backend;
    #: demanding engine provenance there would refuse a whole legitimate class
    #: of package for failing to attest something it never used.
    #: Required, never defaulted: a default would let a physical package omit
    #: the field, be read as dataset-backed, and earn Tier 1 on an unknown or
    #: synthetic backend. Fail-open is not available on a provenance gate.
    stepped_world: bool
    #: Backend actually observed, read from both runs' head-covered
    #: ``JobResult.world_engine``. The task's ``synthetic_stub`` flag is a
    #: *permission*, not an observation - ``make_isaac_bridge`` prefers a real
    #: engine whenever one is attached - so tiering on the flag would demote a
    #: genuine GPU run. On a stepped world only ``real`` can carry a Tier-1
    #: claim: a stub reports whatever it likes (deterministically, because it
    #: is fake), and ``unknown`` means the bridge exposes no provenance at all.
    backend: str = BACKEND_UNKNOWN
    #: Measured, never assumed.
    determinism_class: DeterminismClass
    tolerance: float
    checks: tuple[ConformanceCheck, ...]
    tier: Literal[0, 1]
    #: Names exactly what is missing for Tier 1, or why it was earned.
    tier_reason: str

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.id for check in self.checks if not check.passed)

    @property
    def kind_registered(self) -> bool:
        """Whether an adapter for this world kind is installed and registered.

        Without one there is nothing to cross-check ``stepped_world`` against,
        so the package would be certifying its own world class. That is the one
        hole the registry equality check cannot close, and it closes here.
        """
        return world_kind_spec(self.world_kind) is not None

    @property
    def engine_attested(self) -> bool:
        """Whether the run's engine claim holds for the kind of world it was."""
        if not self.stepped_world:
            return True
        return self.backend == BACKEND_REAL

    @property
    def tier1_eligible(self) -> bool:
        """Whether every Tier-1 condition holds. The tier field cannot exceed this."""
        return (
            not self.failed_checks
            and self.adapter_pinned
            and not self.metrics_only
            and self.kind_registered
            and self.engine_attested
            and self.determinism_class is not DeterminismClass.NONDETERMINISTIC
        )

    def check(self, check_id: str) -> ConformanceCheck:
        """One check by id, or raise naming the ids present."""
        found = next((item for item in self.checks if item.id == check_id), None)
        if found is None:
            present = ", ".join(item.id for item in self.checks) or "(none)"
            raise TaskContractError(f"report has no check {check_id!r}; present: {present}")
        return found

    @model_validator(mode="after")
    def _stepped_world_matches_the_installed_adapter(self) -> Self:
        """The report may not self-certify what kind of world it ran.

        ``stepped_world`` is the switch that decides whether engine provenance
        is required, so a report that could *declare* it False would waive its
        own provenance gate. The registered adapter's capability is the
        authority; only a kind with no adapter installed has no cross-check,
        and that case cannot reach Tier 1 (see ``tier1_eligible``).
        """
        spec = world_kind_spec(self.world_kind)
        if spec is None:
            return self
        declared = spec.capabilities.physics
        if self.stepped_world != declared:
            raise TaskContractError(
                f"world kind {self.world_kind!r} declares physics={declared} in the installed "
                f"adapter, but the report claims stepped_world={self.stepped_world}; a report "
                "cannot reclassify the world it ran in order to waive the backend requirement"
            )
        return self

    @model_validator(mode="after")
    def _tier_is_earned(self) -> Self:
        ids = [check.id for check in self.checks]
        missing = [name for name in REQUIRED_CHECKS if name not in ids]
        if missing:
            raise TaskContractError(
                f"a conformance report must carry every check; missing {missing}"
            )
        if len(set(ids)) != len(ids):
            raise TaskContractError("a conformance report must not repeat a check id")
        if self.tier == 1 and not self.tier1_eligible:
            raise TaskContractError(
                "tier 1 requires all four checks passed, a pinned world adapter, a gate "
                "mapping (not metrics-only), an observed real backend on any world that is "
                "stepped, and a measured determinism class better than nondeterministic"
            )
        if not self.tier_reason:
            raise TaskContractError("a conformance report must state why it earned its tier")
        return self


def _is_number(value: Any) -> bool:
    """Numeric for tolerance purposes; ``bool`` is a status, not a magnitude."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compare(a: Any, b: Any, *, tolerance: float, path: str) -> tuple[str, float]:
    """First difference tolerance does not excuse, and the largest float delta.

    Structural comparison rather than digest equality, because the whole point
    of the ``tolerance`` class is to distinguish "this engine's floats wobble
    in the last bits" from "this engine ran a different episode".
    """
    if _is_number(a) and _is_number(b):
        delta = abs(float(a) - float(b))
        if delta <= tolerance:
            return "", delta
        return f"{path or '$'}: {a!r} != {b!r} (delta {delta:.3e})", delta
    if isinstance(a, dict) and isinstance(b, dict):
        worst = 0.0
        if set(a) != set(b):
            only = sorted(set(a) ^ set(b))
            return f"{path or '$'}: key set differs ({only})", worst
        for key in sorted(a):
            found, delta = _compare(a[key], b[key], tolerance=tolerance, path=f"{path}.{key}")
            worst = max(worst, delta)
            if found:
                return found, worst
        return "", worst
    if isinstance(a, list) and isinstance(b, list):
        worst = 0.0
        if len(a) != len(b):
            return f"{path or '$'}: length {len(a)} != {len(b)}", worst
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            found, delta = _compare(left, right, tolerance=tolerance, path=f"{path}[{index}]")
            worst = max(worst, delta)
            if found:
                return found, worst
        return "", worst
    if a != b:
        return f"{path or '$'}: {a!r} != {b!r}", 0.0
    return "", 0.0


def _gate_state_check(task: TaskSpec, result: JobResult) -> ConformanceCheck:
    """Verify every declared gate resolves a real status on this world."""
    declared = task.verifier.gates
    if task.environment.metrics_only:
        passed = not declared
        detail = (
            f"metrics-only package declares no hard gates; {result.n} trials report "
            "metrics only and attest nothing about safety"
            if passed
            else f"metrics-only package declares gates {[gate.id for gate in declared]}"
        )
        return ConformanceCheck(id=CHECK_GATE_STATES, passed=passed, detail=detail)
    if not declared:
        return ConformanceCheck(
            id=CHECK_GATE_STATES,
            passed=False,
            detail=(
                "package declares no hard gates and does not declare "
                "environment.metrics_only; a Tier-1 wrap either maps gates to state the "
                "world reports or labels itself metrics-only"
            ),
        )
    counts: list[GateStateCount] = []
    for gate in declared:
        tally: dict[GateStatus, int] = dict.fromkeys(GateStatus, 0)
        for trial in result.trials:
            outcome = trial.vector.gate(gate.id)
            if outcome is None:
                raise TaskContractError(
                    f"trial seed {trial.seed} vector omits declared gate {gate.id!r}"
                )
            tally[outcome.status] += 1
        counts.append(
            GateStateCount(
                id=gate.id,
                n_pass=tally[GateStatus.PASS],
                n_fail=tally[GateStatus.FAIL],
                n_not_assessable=tally[GateStatus.NOT_ASSESSABLE],
                n_not_applicable=tally[GateStatus.NOT_APPLICABLE],
            )
        )
    never = [count.id for count in counts if count.assessed == 0]
    if never:
        detail = (
            f"gate(s) {never} never resolved pass or fail across {result.n} trials: this "
            "world does not report the state they score. Instrument the world (upstream "
            "PR) or ship the package as environment.metrics_only; never synthesize a "
            "gate from state the env does not report"
        )
        return ConformanceCheck(
            id=CHECK_GATE_STATES, passed=False, detail=detail, gate_states=tuple(counts)
        )
    return ConformanceCheck(
        id=CHECK_GATE_STATES,
        passed=True,
        detail=(
            f"{len(counts)} declared gate(s) each resolved pass or fail on this world "
            f"across {result.n} trials"
        ),
        gate_states=tuple(counts),
    )


def _license_check(task: TaskSpec, task_dir: Path) -> ConformanceCheck:
    """Classify the package's declared SPDX license; unknown is a refusal."""
    declaration = declared_package_license(task_dir, task.metadata.tags)
    verdict = classify_license(declaration.spdx)
    passed = verdict.status is LicenseStatus.ALLOWED
    where = declaration.source or "no declaration found"
    detail = f"{verdict.status.value} ({where}): {verdict.reason}"
    return ConformanceCheck(
        id=CHECK_LICENSE,
        passed=passed,
        detail=detail,
        license=verdict,
        license_source=declaration.source,
    )


def _evidence_replay_check(
    job_dir: Path, *, task: TaskSpec, task_dir: Path, result: JobResult
) -> ConformanceCheck:
    """Reconstitute every published vector from its stored trajectory."""
    try:
        assert_trajectory_matches_vector(
            job_dir,
            task=task,
            task_dir=task_dir,
            result=result,
            config=read_job_config(job_dir),
        )
    except (TaskContractError, ScoreContractError) as exc:
        return ConformanceCheck(id=CHECK_EVIDENCE_REPLAY, passed=False, detail=str(exc))
    return ConformanceCheck(
        id=CHECK_EVIDENCE_REPLAY,
        passed=True,
        detail=(
            f"all {result.n} trial vectors reconstitute from their stored trajectory "
            "through the task-owned verifier"
        ),
    )


def _declared_determinism(task: TaskSpec) -> DeterminismClass:
    """Strongest determinism class claimed for this world.

    Both the package's ``[environment.capabilities]`` and the installed
    adapter's registered capabilities are claims a published artifact carries,
    and ``resolve_world_capabilities`` deliberately ignores
    ``determinism_class`` when it compares eligibility flags. So the check must
    honour whichever claim is *strongest*: a measurement weaker than any
    standing claim is a claim the artifact cannot keep.
    """
    claims = [DeterminismClass.UNMEASURED]
    declared = task.environment.capabilities
    if declared is not None:
        claims.append(declared.determinism_class)
    spec = world_kind_spec(task.environment.kind)
    if spec is not None:
        claims.append(spec.capabilities.determinism_class)
    strongest = DeterminismClass.UNMEASURED
    for claim in claims:
        if not determinism_at_least(strongest, claim):
            strongest = claim
    return strongest


def _determinism_check(
    task: TaskSpec,
    first: JobResult,
    second: JobResult,
    *,
    tolerance: float,
) -> tuple[ConformanceCheck, DeterminismEvidence]:
    """Measure a determinism class from two identical runs; never assume one."""
    if tolerance < 0:
        raise TaskContractError(f"tolerance must be non-negative, got {tolerance}")
    seeds_a = [trial.seed for trial in first.trials]
    seeds_b = [trial.seed for trial in second.trials]
    declared = _declared_determinism(task)
    if seeds_a != seeds_b:
        evidence = DeterminismEvidence(
            measured=DeterminismClass.NONDETERMINISTIC,
            declared=declared,
            tolerance=tolerance,
            identical_digests=False,
            vectors_equal=False,
            first_difference=f"seed schedule {seeds_a} != {seeds_b}",
        )
        return _determinism_verdict(evidence), evidence

    identical_digests = True
    vectors_equal = True
    first_difference = ""
    max_delta = 0.0
    for left, right in zip(first.trials, second.trials, strict=True):
        left_trace = list(left.trajectory)
        right_trace = list(right.trajectory)
        if digest(left_trace) != digest(right_trace):
            identical_digests = False
        if left.vector != right.vector:
            vectors_equal = False
            if not first_difference:
                found, _ = _compare(
                    left.vector.model_dump(mode="json"),
                    right.vector.model_dump(mode="json"),
                    tolerance=0.0,
                    path=f"seed {left.seed} vector",
                )
                first_difference = found or f"seed {left.seed} vector differs"
            continue
        found, delta = _compare(
            left_trace, right_trace, tolerance=tolerance, path=f"seed {left.seed} trajectory"
        )
        max_delta = max(max_delta, delta)
        if found and not first_difference:
            first_difference = found

    if identical_digests:
        measured = DeterminismClass.BITWISE
    elif vectors_equal and not first_difference:
        measured = DeterminismClass.TOLERANCE
    else:
        measured = DeterminismClass.NONDETERMINISTIC
    evidence = DeterminismEvidence(
        measured=measured,
        declared=declared,
        tolerance=tolerance,
        identical_digests=identical_digests,
        vectors_equal=vectors_equal,
        max_float_delta=max_delta,
        first_difference=first_difference,
    )
    return _determinism_verdict(evidence), evidence


def _determinism_verdict(evidence: DeterminismEvidence) -> ConformanceCheck:
    """Turn a measurement into a verdict: only a broken claim fails this check.

    A world measured ``nondeterministic`` that never claimed otherwise has not
    failed a check — it has failed Tier 1, which the tier rule states
    separately. Conflating the two would report a Tier-2 world as broken.
    """
    measured = evidence.measured
    detail = f"measured {measured.value} from two identical runs (tolerance {evidence.tolerance:g})"
    if evidence.first_difference:
        detail = f"{detail}; first difference {evidence.first_difference}"
    if evidence.declared is DeterminismClass.UNMEASURED:
        return ConformanceCheck(
            id=CHECK_DETERMINISM, passed=True, detail=detail, determinism=evidence
        )
    if not determinism_at_least(measured, evidence.declared):
        return ConformanceCheck(
            id=CHECK_DETERMINISM,
            passed=False,
            detail=(
                f"{detail}; the world declares {evidence.declared.value}, which this "
                "measurement does not support. Fix the engine's seeding or weaken the "
                "declared determinism_class to what it can hold"
            ),
            determinism=evidence,
        )
    return ConformanceCheck(
        id=CHECK_DETERMINISM,
        passed=True,
        detail=f"{detail}; declared {evidence.declared.value} holds",
        determinism=evidence,
    )


def _tier_reason(
    *,
    failed: tuple[str, ...],
    adapter_pinned: bool,
    metrics_only: bool,
    kind_registered: bool,
    stepped_world: bool,
    backend: str,
    measured: DeterminismClass,
) -> tuple[Literal[0, 1], str]:
    """Tier plus a reason naming exactly what is missing."""
    missing: list[str] = []
    if failed:
        missing.append("failed check(s) " + ", ".join(failed))
    if not adapter_pinned:
        missing.append("no world-adapter pin (declare environment.adapter and adapter_digest)")
    if metrics_only:
        missing.append(
            "environment.metrics_only, which is Tier 0 by §2.2: a metrics-only wrap maps "
            "no gates and is explicitly not safety-attested"
        )
    if not kind_registered:
        missing.append(
            "no adapter is installed for this world kind, so the package's own "
            "[environment.capabilities] declaration was never cross-checked and its world "
            "class rests on its own word"
        )
    if stepped_world and backend == BACKEND_SYNTHETIC_STUB:
        missing.append(
            "the observed backend was a synthetic stand-in, so these checks describe the "
            "stand-in and not the world. A stub is deterministic because it is fake, which "
            "is the opposite of evidence"
        )
    elif stepped_world and backend != BACKEND_REAL:
        missing.append(
            f"the observed backend is {backend!r}: the bridge exposes no engine_provenance "
            "reporter, so no real-world claim can be attested from this run"
        )
    if measured is DeterminismClass.NONDETERMINISTIC:
        missing.append("measured determinism class is nondeterministic")
    if missing:
        return 0, "tier 0: " + "; ".join(missing)
    evidence = "a real backend" if stepped_world else f"declared data ({backend})"
    return 1, (
        f"tier 1: all {len(REQUIRED_CHECKS)} checks passed on {evidence}, world adapter "
        f"pinned, gates mapped, determinism measured {measured.value}"
    )


def _observed_backend(first: JobResult, second: JobResult) -> str:
    """Backend both runs actually used, from head-covered provenance.

    Disagreement between the two runs is refused rather than resolved. If run A
    reached the real engine and run B fell back to a stand-in, the pair does not
    measure one world's determinism at all, and picking either answer would
    describe a run that never happened.
    """
    backends = []
    for result in (first, second):
        engine = result.world_engine
        backends.append(engine.backend if engine is not None else BACKEND_UNKNOWN)
    if backends[0] != backends[1]:
        raise TaskContractError(
            f"the two conformance runs used different backends ({backends[0]!r} then "
            f"{backends[1]!r}), so they do not measure one world. Fix: make the engine "
            "available (or unavailable) for both runs and re-measure."
        )
    return backends[0]


def _resolve_agent(
    task: TaskSpec,
    *,
    agent: AgentPackage | None,
    agent_dir: Path | None,
) -> AgentPackage:
    if agent is not None:
        return agent
    if agent_dir is not None:
        return load_agent(agent_dir)
    return builtin_random_agent(task.interface.id)


def run_conformance(
    task_dir: Path,
    *,
    agent_dir: Path | None = None,
    agent: AgentPackage | None = None,
    n: int = 2,
    workdir: Path,
    gym_factory: GymFactory | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ConformanceReport:
    """Run the four Tier-1 checks against a task package and report a tier.

    The job is executed twice into ``workdir/run-a`` and ``workdir/run-b`` with
    identical task, agent, seeds, and ``n``, because execution determinism
    cannot be read off a single run and the harness refuses to assume it.
    """
    if n < 1:
        raise TaskContractError(f"conformance needs at least one trial, got n={n}")
    root = task_dir if task_dir.is_dir() else task_dir.parent
    task = load_task(root)
    package = _resolve_agent(task, agent=agent, agent_dir=agent_dir)
    job_dir = workdir / RUN_A
    first = run_job(
        task=task,
        task_dir=root,
        agent=package,
        agent_dir=agent_dir,
        out=job_dir,
        n=n,
        gym_factory=gym_factory,
    )
    second = run_job(
        task=task,
        task_dir=root,
        agent=package,
        agent_dir=agent_dir,
        out=workdir / RUN_B,
        n=n,
        gym_factory=gym_factory,
    )
    determinism_check, determinism = _determinism_check(task, first, second, tolerance=tolerance)
    checks = (
        _gate_state_check(task, first),
        _license_check(task, root),
        _evidence_replay_check(job_dir, task=task, task_dir=root, result=first),
        determinism_check,
    )
    spec = world_kind_spec(task.environment.kind)
    adapter_pinned = bool(task.environment.adapter and task.environment.adapter_digest)
    failed = tuple(check.id for check in checks if not check.passed)
    backend = _observed_backend(first, second)
    # From the *declared* capability, not from whether an env object happened to
    # exist: a physics world whose bridge silently failed to build must still be
    # judged as a physics world, or the missing engine would excuse itself.
    stepped_world = task.environment.resolved_capabilities.physics
    tier, reason = _tier_reason(
        failed=failed,
        adapter_pinned=adapter_pinned,
        metrics_only=task.environment.metrics_only,
        kind_registered=spec is not None,
        stepped_world=stepped_world,
        backend=backend,
        measured=determinism.measured,
    )
    return ConformanceReport(
        task_id=task.id,
        task_version=task.task_version,
        task_digest=first.task_digest,
        world_kind=task.environment.kind_key,
        adapter_identity=spec.adapter_identity if spec is not None else "unregistered",
        adapter_pinned=adapter_pinned,
        metrics_only=task.environment.metrics_only,
        stepped_world=stepped_world,
        backend=backend,
        determinism_class=determinism.measured,
        tolerance=tolerance,
        checks=checks,
        tier=tier,
        tier_reason=reason,
    )


def _cell(text: str) -> str:
    """Table-safe cell text: a world-reported value must not break the table."""
    return " ".join(text.split()).replace("|", "\\|")


def render_markdown(report: ConformanceReport) -> str:
    """Short human summary; the JSON stays the machine surface."""
    lines = [
        f"# Conformance: {report.task_id}@{report.task_version}",
        "",
        f"- Tier: **{report.tier}** — {report.tier_reason}",
        f"- World kind: `{report.world_kind}`",
        f"- Adapter identity: `{report.adapter_identity}`",
        f"- Adapter pinned: `{str(report.adapter_pinned).lower()}`",
        f"- Metrics-only: `{str(report.metrics_only).lower()}`",
        f"- Stepped world: `{str(report.stepped_world).lower()}`",
        f"- Backend (observed): `{report.backend}`",
        f"- Determinism (measured): `{report.determinism_class.value}`",
        f"- Tolerance: `{report.tolerance:g}`",
        f"- Task digest: `{report.task_digest}`",
        "",
        "## Checks",
        "",
        "| Check | Verdict | Detail |",
        "|---|:---:|---|",
    ]
    lines.extend(
        f"| {_cell(check.id)} | {'pass' if check.passed else 'FAIL'} | {_cell(check.detail)} |"
        for check in report.checks
    )
    gate_states = report.check(CHECK_GATE_STATES).gate_states
    if gate_states:
        lines.extend(
            [
                "",
                "## Gate-state availability",
                "",
                "| Gate | Pass | Fail | Not assessable | Not applicable |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {_cell(count.id)} | {count.n_pass} | {count.n_fail} | "
            f"{count.n_not_assessable} | {count.n_not_applicable} |"
            for count in gate_states
        )
    lines.extend(
        [
            "",
            "> Tier 1 is a measurement, not a label: gate states, license, evidence "
            "replay, and execution determinism were each run against this package.",
            "",
        ]
    )
    return "\n".join(lines)


def write_conformance_report(report: ConformanceReport, out: Path) -> Path:
    """Write deterministic ``conformance.json`` plus a human ``conformance.md``."""
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    target = out / "conformance.json"
    target.write_text(payload, encoding="utf-8")
    (out / "conformance.md").write_text(render_markdown(report), encoding="utf-8")
    return target
