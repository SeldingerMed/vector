"""Tier-1 conformance suite (next.md N3).

Every test runs the real harness against an in-tree example package: the point
of this suite is that Tier 1 is a *measurement*, so a test that mocked the
measurement would prove nothing. The refusals pinned here are the ones that
keep the curated shelf honest — a gate the world never resolves, a license we
cannot redistribute, a trajectory that does not reconstitute its vector, and a
determinism claim the engine cannot keep.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from or_audit.commands.conformance import register
from or_audit.domain.enums import GateStatus
from or_audit.eval import licensing
from or_audit.eval.conformance import (
    CHECK_DETERMINISM,
    CHECK_EVIDENCE_REPLAY,
    CHECK_GATE_STATES,
    CHECK_LICENSE,
    REQUIRED_CHECKS,
    RUN_A,
    ConformanceReport,
    run_conformance,
    write_conformance_report,
)
from or_audit.eval.contracts import InteractionMode
from or_audit.eval.job import read_job_result
from or_audit.eval.licensing import (
    LicenseStatus,
    classify_license,
    declared_package_license,
)
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.sim import reset_default_simulation_engines
from or_audit.eval.sim.base import BACKEND_REAL, BACKEND_SYNTHETIC_STUB, BACKEND_UNKNOWN
from or_audit.eval.task import TaskSpec
from or_audit.eval.worlds import (
    DeterminismClass,
    WorldCapabilities,
    attach_world_adapter,
    world_kind_spec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_TASK = REPO_ROOT / "docs/examples/tasks/video-nextstep"
VIDEO_AGENT = REPO_ROOT / "docs/examples/agents/example-video-predictor"
BRONCHO_TASK = REPO_ROOT / "docs/examples/tasks/broncho-airway-nav"


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    yield
    reset_default_simulation_engines()


def _frame_source_adapter(task: TaskSpec) -> object:
    """A frame source is not stepped; this exists only to pin adapter identity."""
    del task
    return object()


def _copy(source: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def _declare_license(task_dir: Path, spdx: str) -> None:
    """Declare a license the way ``surgeval wrap`` does."""
    (task_dir / "wrap.json").write_text(
        json.dumps({"license": spdx}, indent=2) + "\n", encoding="utf-8"
    )


def _rewrite_environment(task_dir: Path, lines: list[str]) -> None:
    """Replace the package's ``[environment]`` block with ``lines``."""
    toml_path = task_dir / "task.toml"
    original = toml_path.read_text(encoding="utf-8")
    head, _, tail = original.partition("[environment]")
    _, _, rest = tail.partition("\n[interface]")
    body = "\n".join(["[environment]", *lines])
    toml_path.write_text(f"{head}{body}\n\n[interface]{rest}", encoding="utf-8")


def _pinned_video_task(tmp_path: Path, *, spdx: str = "MIT") -> Path:
    """A ``video-nextstep`` copy that pins its world adapter and declares a license."""
    spec = attach_world_adapter(
        "frame-source",
        capabilities=WorldCapabilities(),
        factory=_frame_source_adapter,
        provider="conformance-test",
    )
    task_dir = _copy(VIDEO_TASK, tmp_path / "video-pinned")
    _rewrite_environment(
        task_dir,
        [
            'kind = "frame-source"',
            "n_eval_episodes = 3",
            'seed_policy = "held-out-split"',
            'inputs_path = "inputs.json"',
            'labels_path = "labels.json"',
            f'adapter = "{spec.adapter_id}"',
            f'adapter_digest = "{spec.adapter_digest}"',
        ],
    )
    _declare_license(task_dir, spdx)
    return task_dir


def _broncho_task(
    tmp_path: Path,
    *,
    name: str = "broncho",
    spdx: str = "MIT",
    determinism: str = "",
) -> Path:
    """A closed-loop ``broncho-airway-nav`` copy pinned to the installed gym adapter."""
    spec = world_kind_spec("gym")
    assert spec is not None
    task_dir = _copy(BRONCHO_TASK, tmp_path / name)
    lines = [
        'kind = "gym"',
        'gym_id = "Broncho/AirwayNav-v0"',
        'world_pin = "broncho-synthetic-v1"',
        "parameters = { max_steps = 4 }",
        "n_eval_episodes = 2",
        'seed_policy = "deterministic-eval-2"',
        f'adapter = "{spec.adapter_id}"',
        f'adapter_digest = "{spec.adapter_digest}"',
    ]
    if determinism:
        lines.extend(
            [
                "",
                "[environment.capabilities]",
                "physics = true",
                "closed_loop = true",
                "requires_gym_id = true",
                "requires_world_pin = true",
                f'determinism_class = "{determinism}"',
            ]
        )
    _rewrite_environment(task_dir, lines)
    _declare_license(task_dir, spdx)
    return task_dir


class _BronchoGym:
    """Injected closed-loop world with tunable instrumentation and determinism."""

    def __init__(
        self,
        *,
        report_force: bool = True,
        force: float = 0.5,
        jitter: float = 0.0,
        backend: str = BACKEND_REAL,
        world_pin: str = "broncho-synthetic-v1",
    ) -> None:
        self._step = 0
        self._report_force = report_force
        self._force = force
        self._jitter = jitter
        self._backend = backend
        self._world_pin = world_pin

    def engine_provenance(self) -> dict[str, Any]:
        """What a bridge must answer: which engine ran, and was it the real one.

        The double declares this because a real bridge must. Tests that want the
        provenance-less or stub-backed cases pass ``backend=`` explicitly rather
        than relying on this class staying silent, and ``world_pin=`` covers the
        bridge that ran a revision the package never pinned.
        """
        return {
            "engine": "gym",
            "backend": self._backend,
            "backend_version": "0.0.0-test-double",
            "world_pin": self._world_pin,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del options
        self._step = 0
        return {"airway_id": "RB1"}, {"seed": seed}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        self._step += 1
        done = self._step >= 2
        info: dict[str, Any] = {
            "target_reached": done,
            "safe_navigation": done,
            "diverged": False,
        }
        if self._report_force:
            info["max_contact_force_n"] = self._force
        if self._jitter:
            # Telemetry no gate or metric binds: trace-only float noise.
            info["telemetry_ms"] = 12.0 + self._jitter
        return {"airway_id": "RB1_sub"}, 1.0 if done else 0.0, done, False, info


def _factory(**kwargs: Any) -> Callable[[TaskSpec], _BronchoGym]:
    """A stable world: every construction produces the identical episode."""

    def make(task: TaskSpec) -> _BronchoGym:
        del task
        return _BronchoGym(**kwargs)

    return make


def _drifting_factory(
    *, force_step: float = 0.0, jitter_step: float = 0.0
) -> Callable[[TaskSpec], _BronchoGym]:
    """A world that differs on every construction, i.e. between the two runs.

    The drift is counted rather than sampled so the measurement is a fact
    about the suite, not a coin flip: two identical jobs are guaranteed to see
    different worlds, which is exactly what a nondeterministic engine does.
    """
    runs = itertools.count(1)

    def make(task: TaskSpec) -> _BronchoGym:
        del task
        run = next(runs)
        return _BronchoGym(force=0.5 + force_step * run, jitter=jitter_step * run)

    return make


def _diverging_factory() -> Callable[[TaskSpec], _BronchoGym]:
    """Two runs whose solver blows up differently: ``nan``, then ``+inf``.

    Under the old recorder both normalized to ``0.0``, so the pair digested
    identically and measured ``bitwise`` - the worst available reading, since
    non-finite telemetry *is* the divergence signal.
    """
    runs = itertools.count(1)

    def make(task: TaskSpec) -> _BronchoGym:
        del task
        return _BronchoGym(force=float("nan") if next(runs) == 1 else float("inf"))

    return make


def test_a_conformant_package_earns_tier_1(tmp_path: Path) -> None:
    task_dir = _pinned_video_task(tmp_path)
    report = run_conformance(
        task_dir,
        agent_dir=VIDEO_AGENT,
        n=2,
        workdir=tmp_path / "work",
    )

    assert [check.id for check in report.checks] == list(REQUIRED_CHECKS)
    assert report.failed_checks == ()
    assert report.adapter_pinned is True
    assert report.adapter_identity == world_kind_spec("frame-source").adapter_identity  # type: ignore[union-attr]
    assert report.determinism_class is DeterminismClass.BITWISE
    assert report.tolerance == pytest.approx(1e-9)
    assert report.tier == 1
    assert "all 4 checks passed" in report.tier_reason

    gate_states = report.check(CHECK_GATE_STATES).gate_states
    assert [count.id for count in gate_states] == ["unsafe_prediction"]
    # The example predictions are one safe, one unsafe: the gate really bites.
    assert (gate_states[0].n_pass, gate_states[0].n_fail) == (1, 1)
    verdict = report.check(CHECK_LICENSE).license
    assert verdict is not None
    assert verdict.status is LicenseStatus.ALLOWED
    assert report.check(CHECK_LICENSE).license_source == "wrap.json:license"


def test_tier_1_requires_the_adapter_pin(tmp_path: Path) -> None:
    """The unmodified example package is clean but unpinned: Tier 0, said plainly."""
    report = run_conformance(
        VIDEO_TASK,
        agent_dir=VIDEO_AGENT,
        n=2,
        workdir=tmp_path / "work",
    )
    assert report.adapter_pinned is False
    assert report.tier == 0
    assert "no world-adapter pin" in report.tier_reason
    assert report.check(CHECK_GATE_STATES).passed is True
    assert report.check(CHECK_EVIDENCE_REPLAY).passed is True


def test_a_gate_the_world_never_resolves_fails_and_drops_to_tier_0(tmp_path: Path) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-blind")
    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_factory(report_force=False),
    )

    check = report.check(CHECK_GATE_STATES)
    assert check.passed is False
    counts = {count.id: count for count in check.gate_states}
    assert counts["airway_wall_puncture"].assessed == 0
    assert counts["airway_wall_puncture"].n_not_assessable == 2
    assert "never resolved pass or fail" in check.detail
    assert "environment.metrics_only" in check.detail
    # Everything else about the package is fine, so the tier names only this.
    assert report.tier == 0
    assert report.tier_reason == f"tier 0: failed check(s) {CHECK_GATE_STATES}"


def test_a_gate_that_resolves_earns_the_gate_state_check(tmp_path: Path) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-instrumented")
    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_factory(),
    )
    check = report.check(CHECK_GATE_STATES)
    assert check.passed is True
    counts = {count.id: count for count in check.gate_states}
    assert counts["airway_wall_puncture"].n_pass == 2
    assert report.determinism_class is DeterminismClass.BITWISE
    assert report.tier == 1


def test_a_nondeterministic_world_is_measured_and_refused_tier_1(tmp_path: Path) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-drifting")
    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        # Contact force differs per job: gates still pass, the vector does not reproduce.
        gym_factory=_drifting_factory(force_step=0.1),
    )

    evidence = report.check(CHECK_DETERMINISM).determinism
    assert evidence is not None
    assert evidence.measured is DeterminismClass.NONDETERMINISTIC
    assert evidence.vectors_equal is False
    assert evidence.identical_digests is False
    assert "vector" in evidence.first_difference
    # The world never claimed determinism, so the *check* passes: an honest
    # Tier-2 world is not a broken package. The tier is what refuses it.
    assert report.check(CHECK_DETERMINISM).passed is True
    assert report.determinism_class is DeterminismClass.NONDETERMINISTIC
    assert report.tier == 0
    assert "measured determinism class is nondeterministic" in report.tier_reason


def test_float_jitter_inside_tolerance_measures_the_tolerance_class(tmp_path: Path) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-jitter")
    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_drifting_factory(jitter_step=1e-11),
    )

    evidence = report.check(CHECK_DETERMINISM).determinism
    assert evidence is not None
    assert evidence.measured is DeterminismClass.TOLERANCE
    assert evidence.identical_digests is False
    assert evidence.vectors_equal is True
    assert evidence.first_difference == ""
    assert 0.0 < evidence.max_float_delta <= 1e-9
    assert report.tier == 1


def test_a_declared_class_stronger_than_the_measurement_fails(tmp_path: Path) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-overclaim", determinism="bitwise")
    assert load_task(task_dir).environment.capabilities is not None

    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_drifting_factory(jitter_step=1e-11),
    )

    check = report.check(CHECK_DETERMINISM)
    evidence = check.determinism
    assert evidence is not None
    assert evidence.declared is DeterminismClass.BITWISE
    assert evidence.measured is DeterminismClass.TOLERANCE
    assert check.passed is False
    assert "declares bitwise" in check.detail
    assert report.tier == 0
    assert CHECK_DETERMINISM in report.tier_reason


def test_a_declared_class_the_measurement_supports_passes(tmp_path: Path) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-honest", determinism="tolerance")
    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_drifting_factory(jitter_step=1e-11),
    )
    check = report.check(CHECK_DETERMINISM)
    assert check.passed is True
    assert "declared tolerance holds" in check.detail
    assert report.tier == 1


def test_a_restricted_license_fails_the_audit(tmp_path: Path) -> None:
    task_dir = _pinned_video_task(tmp_path, spdx="GPL-3.0-only")
    report = run_conformance(
        task_dir,
        agent_dir=VIDEO_AGENT,
        n=2,
        workdir=tmp_path / "work",
    )

    check = report.check(CHECK_LICENSE)
    assert check.passed is False
    verdict = check.license
    assert verdict is not None
    assert verdict.status is LicenseStatus.RESTRICTED
    assert "reciprocal copyleft" in verdict.reason
    assert report.tier == 0
    assert CHECK_LICENSE in report.tier_reason


def test_an_unreviewed_license_is_never_assumed_permissive(tmp_path: Path) -> None:
    task_dir = _pinned_video_task(tmp_path, spdx="CathSim-Research-Only-1.0")
    report = run_conformance(
        task_dir,
        agent_dir=VIDEO_AGENT,
        n=2,
        workdir=tmp_path / "work",
    )
    verdict = report.check(CHECK_LICENSE).license
    assert verdict is not None
    assert verdict.status is LicenseStatus.UNKNOWN
    assert report.tier == 0


def test_report_json_is_byte_identical_across_runs(tmp_path: Path) -> None:
    task_dir = _pinned_video_task(tmp_path)
    first = run_conformance(task_dir, agent_dir=VIDEO_AGENT, n=2, workdir=tmp_path / "w1")
    second = run_conformance(task_dir, agent_dir=VIDEO_AGENT, n=2, workdir=tmp_path / "w2")

    left = write_conformance_report(first, tmp_path / "out-1")
    right = write_conformance_report(second, tmp_path / "out-2")
    assert left.read_bytes() == right.read_bytes()
    assert left.read_text(encoding="utf-8").endswith("}\n")
    payload = json.loads(left.read_text(encoding="utf-8"))
    assert list(payload) == sorted(payload)
    assert payload["tier"] == 1
    assert (
        (tmp_path / "out-1" / "conformance.md")
        .read_text(encoding="utf-8")
        .startswith("# Conformance: video-nextstep@0")
    )


def test_metrics_only_packages_are_checked_for_having_no_gates(tmp_path: Path) -> None:
    """§2.2: a metrics-only wrap is honest and still Tier 0 — it maps no gates."""
    task_dir = _broncho_task(tmp_path, name="broncho-metrics-only")
    toml_path = task_dir / "task.toml"
    body = toml_path.read_text(encoding="utf-8")
    body = body.replace("safety_critical = true", "safety_critical = false")
    body = body.replace('kind = "gym"', 'kind = "gym"\nmetrics_only = true', 1)
    toml_path.write_text(_strip_gates(body), encoding="utf-8")

    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_factory(),
    )
    check = report.check(CHECK_GATE_STATES)
    assert check.passed is True
    assert check.gate_states == ()
    assert "metrics-only package declares no hard gates" in check.detail
    assert report.metrics_only is True
    assert report.tier == 0
    assert "Tier 0 by §2.2" in report.tier_reason


def test_a_package_with_neither_gates_nor_the_metrics_only_label_is_refused(
    tmp_path: Path,
) -> None:
    task_dir = _broncho_task(tmp_path, name="broncho-silent")
    toml_path = task_dir / "task.toml"
    body = toml_path.read_text(encoding="utf-8")
    body = body.replace("safety_critical = true", "safety_critical = false")
    toml_path.write_text(_strip_gates(body), encoding="utf-8")

    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_factory(),
    )
    check = report.check(CHECK_GATE_STATES)
    assert check.passed is False
    assert "does not declare environment.metrics_only" in check.detail
    assert report.tier == 0


def test_evidence_replay_refuses_a_tampered_trajectory(tmp_path: Path) -> None:
    """The check reads the stored trajectory, not the in-memory result."""
    task_dir = _pinned_video_task(tmp_path)
    workdir = tmp_path / "work"
    run_conformance(task_dir, agent_dir=VIDEO_AGENT, n=2, workdir=workdir)

    trial = workdir / "run-a" / "trial-video-nextstep-0"
    trajectory = json.loads((trial / "trajectory.json").read_text(encoding="utf-8"))
    trajectory[0]["prediction"]["unsafe"] = True
    (trial / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2) + "\n", encoding="utf-8"
    )

    from or_audit.eval.conformance import _evidence_replay_check
    from or_audit.eval.job import read_job_result

    check = _evidence_replay_check(
        workdir / "run-a",
        task=load_task(task_dir),
        task_dir=task_dir,
        result=read_job_result(workdir / "run-a"),
    )
    assert check.passed is False
    assert "reconstitutes a different vector" in check.detail


def _passing_checks(
    *,
    measured: DeterminismClass = DeterminismClass.BITWISE,
    tolerance: float = 1e-9,
    failing: str = "",
) -> tuple[Any, ...]:
    """Four checks with real determinism evidence behind the class they claim.

    The evidence is part of the fixture because Tier 1 requires it: a report
    naming a determinism class the ``execution-determinism`` check never
    recorded is a header without a measurement.
    """
    from or_audit.eval.conformance import ConformanceCheck, DeterminismEvidence

    return tuple(
        ConformanceCheck(
            id=name,
            passed=name != failing,
            detail="fixture",
            determinism=(
                DeterminismEvidence(
                    measured=measured,
                    tolerance=tolerance,
                    identical_digests=measured is DeterminismClass.BITWISE,
                    vectors_equal=True,
                )
                if name == CHECK_DETERMINISM
                else None
            ),
        )
        for name in REQUIRED_CHECKS
    )


def test_a_report_cannot_claim_a_tier_it_did_not_earn() -> None:
    from or_audit.errors import TaskContractError

    checks = _passing_checks(failing=CHECK_LICENSE)
    with pytest.raises(TaskContractError, match="tier 1 requires"):
        ConformanceReport(
            task_id="t",
            task_version="0",
            task_digest="d",
            world_kind="gym",
            adapter_identity="mod:fn+abc",
            adapter_pinned=True,
            interaction_mode=InteractionMode.CLOSED_LOOP,
            stepped_world=True,
            backend=BACKEND_REAL,
            determinism_class=DeterminismClass.BITWISE,
            tolerance=1e-9,
            checks=checks,
            tier=1,
            tier_reason="fixture",
        )

    with pytest.raises(TaskContractError, match="must carry every check"):
        ConformanceReport(
            task_id="t",
            task_version="0",
            task_digest="d",
            world_kind="gym",
            adapter_identity="mod:fn+abc",
            adapter_pinned=False,
            interaction_mode=InteractionMode.CLOSED_LOOP,
            stepped_world=True,
            determinism_class=DeterminismClass.BITWISE,
            tolerance=1e-9,
            checks=checks[:2],
            tier=0,
            tier_reason="fixture",
        )


def test_a_stepped_world_cannot_claim_tier_1_without_a_real_backend() -> None:
    """The provenance gate is not defaultable, forgeable, or self-certifiable.

    Four ways a package could otherwise slip through: omit ``stepped_world`` so
    a stepped run reads as dataset-backed, *declare* it False for a closed-loop
    route, report ``unknown`` because the bridge exposes no reporter, or report
    a synthetic stand-in. All four must refuse.
    """
    from or_audit.errors import TaskContractError

    checks = _passing_checks()
    fields: dict[str, Any] = {
        "task_id": "t",
        "task_version": "0",
        "task_digest": "d",
        "world_kind": "gym",
        "adapter_identity": "mod:fn+abc",
        "adapter_pinned": True,
        "interaction_mode": InteractionMode.CLOSED_LOOP,
        "determinism_class": DeterminismClass.BITWISE,
        "tolerance": 1e-9,
        "checks": checks,
        "tier": 1,
        "tier_reason": "fixture",
    }

    # Omission is not a pass: the field has no default to fall back to.
    with pytest.raises(ValidationError):
        ConformanceReport(**fields, backend=BACKEND_REAL)

    for backend in (BACKEND_UNKNOWN, BACKEND_SYNTHETIC_STUB):
        with pytest.raises(TaskContractError, match="tier 1 requires"):
            ConformanceReport(**fields, stepped_world=True, backend=backend)

    # The same package on a dataset-backed route is legitimate: no engine ran,
    # and none was claimed.
    report = ConformanceReport(
        **{
            **fields,
            "world_kind": "frame-source",
            "interaction_mode": InteractionMode.SINGLE_TURN,
        },
        stepped_world=False,
        backend=BACKEND_UNKNOWN,
    )
    assert report.tier == 1
    assert report.engine_attested is True

    # Falsification: a closed-loop route steps a world, so a report cannot
    # reclassify itself as dataset-backed to escape the backend requirement.
    with pytest.raises(TaskContractError, match="cannot reclassify the world"):
        ConformanceReport(**fields, stepped_world=False, backend=BACKEND_UNKNOWN)

    # Self-certification: with no adapter installed there is nothing to
    # cross-check against, so Tier 1 is not available at all.
    with pytest.raises(TaskContractError, match="tier 1 requires"):
        ConformanceReport(
            **{**fields, "world_kind": "steve-arch-nav"},
            stepped_world=True,
            backend=BACKEND_REAL,
        )


def test_an_unmeasured_determinism_class_cannot_carry_tier_1() -> None:
    """``unmeasured`` is the honest default; Tier 1 names a measurement.

    Original defect: ``tier1_eligible`` excluded only ``nondeterministic``, so
    a serialized report could claim Tier 1 with ``determinism_class`` at its
    honest default and no ``DeterminismEvidence`` at all - the strongest claim
    this system makes, resting on the field that says nothing was measured.
    """
    from or_audit.errors import TaskContractError
    from or_audit.eval.conformance import ConformanceCheck, DeterminismEvidence

    fields: dict[str, Any] = {
        "task_id": "t",
        "task_version": "0",
        "task_digest": "d",
        "world_kind": "frame-source",
        "adapter_identity": "mod:fn+abc",
        "adapter_pinned": True,
        "interaction_mode": InteractionMode.SINGLE_TURN,
        "stepped_world": False,
        "backend": BACKEND_UNKNOWN,
        "tolerance": 1e-9,
        "tier": 1,
        "tier_reason": "fixture",
    }

    # The reviewer's probe verbatim: four bare passing checks and the default
    # class. This validated with tier=1 and tier1_eligible True.
    bare = tuple(ConformanceCheck(id=name, passed=True, detail="ok") for name in REQUIRED_CHECKS)
    with pytest.raises(TaskContractError, match="tier 1 requires"):
        ConformanceReport(**fields, determinism_class=DeterminismClass.UNMEASURED, checks=bare)

    # Naming a strong class is not enough either: the check must carry the
    # evidence that class was measured from.
    with pytest.raises(TaskContractError, match="tier 1 requires"):
        ConformanceReport(**fields, determinism_class=DeterminismClass.BITWISE, checks=bare)

    # Evidence that measured something else does not back the header.
    weaker = _passing_checks(measured=DeterminismClass.TOLERANCE)
    with pytest.raises(TaskContractError, match="tier 1 requires"):
        ConformanceReport(**fields, determinism_class=DeterminismClass.BITWISE, checks=weaker)

    # Nor does evidence measured at a different tolerance than the one the
    # report publishes: two tolerances are two claims.
    mismatched = _passing_checks(measured=DeterminismClass.TOLERANCE, tolerance=1e-3)
    with pytest.raises(TaskContractError, match="tier 1 requires"):
        ConformanceReport(**fields, determinism_class=DeterminismClass.TOLERANCE, checks=mismatched)

    # Backed by its own evidence, the same report earns the tier.
    report = ConformanceReport(
        **fields,
        determinism_class=DeterminismClass.TOLERANCE,
        checks=_passing_checks(measured=DeterminismClass.TOLERANCE),
    )
    assert report.tier == 1
    assert report.determinism_measured is True
    evidence = report.check(CHECK_DETERMINISM).determinism
    assert isinstance(evidence, DeterminismEvidence)
    assert evidence.measured is DeterminismClass.TOLERANCE


def test_a_non_finite_tolerance_is_refused_at_every_surface(tmp_path: Path) -> None:
    """``--tolerance inf`` excuses every difference, so it measures nothing.

    Original defect: the guard tested only ``tolerance < 0``, and ``inf`` is
    not negative. ``delta <= inf`` holds for any pair of floats, so two
    arbitrarily different traces classified as tolerance-deterministic - and
    the report then serialized non-standard ``Infinity``.
    """
    from or_audit.errors import TaskContractError
    from or_audit.eval.conformance import DeterminismEvidence, _compare, _determinism_check
    from or_audit.eval.job import read_job_result

    # The reviewer's probe: an infinite tolerance must not excuse this.
    found, delta = _compare(0.0, 1e300, tolerance=float("inf"), path="trace")
    assert found
    assert delta == pytest.approx(1e300)

    task_dir = _broncho_task(tmp_path, name="broncho-tolerance")
    task = load_task(task_dir)
    measured = tmp_path / "measured"
    run_conformance(task_dir, n=2, workdir=measured, gym_factory=_factory())
    ran = read_job_result(measured / RUN_A)
    for bad in (float("inf"), float("nan"), -1.0):
        with pytest.raises(TaskContractError, match="finite non-negative"):
            _determinism_check(task, ran, ran, tolerance=bad)
        with pytest.raises(TaskContractError, match="finite non-negative"):
            run_conformance(task_dir, n=2, workdir=tmp_path / "work", tolerance=bad)
        with pytest.raises(TaskContractError, match="finite non-negative"):
            DeterminismEvidence(
                measured=DeterminismClass.TOLERANCE,
                tolerance=bad,
                identical_digests=False,
                vectors_equal=True,
            )
        with pytest.raises(TaskContractError, match="finite non-negative"):
            ConformanceReport(
                task_id="t",
                task_version="0",
                task_digest="d",
                world_kind="frame-source",
                adapter_identity="mod:fn+abc",
                adapter_pinned=True,
                interaction_mode=InteractionMode.SINGLE_TURN,
                stepped_world=False,
                determinism_class=DeterminismClass.TOLERANCE,
                tolerance=bad,
                checks=_passing_checks(measured=DeterminismClass.TOLERANCE),
                tier=0,
                tier_reason="fixture",
            )


def test_an_engine_on_a_different_world_revision_is_refused(tmp_path: Path) -> None:
    """Tier 1 promises a pinned commit; ``backend`` cannot check which one ran.

    Original defect: conformance read ``world_engine.backend`` and never
    compared the head-covered ``world_pin``. A task declaring
    ``broncho-synthetic-v1`` against a real factory reporting
    ``different-revision`` returned Tier 1 with all four checks passed.
    """
    from or_audit.errors import TaskContractError

    task_dir = _broncho_task(tmp_path, name="broncho-repinned")
    with pytest.raises(TaskContractError, match="is not a pin"):
        run_conformance(
            task_dir,
            n=2,
            workdir=tmp_path / "work",
            gym_factory=_factory(world_pin="different-revision"),
        )

    # A bridge that attests no revision at all cannot certify the pin either,
    # and that is a tier, not a crash: the run happened, it just proves less.
    quiet = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "quiet",
        gym_factory=_factory(world_pin=""),
    )
    assert quiet.world_pin == "broncho-synthetic-v1"
    assert quiet.world_pin_observed == ""
    assert quiet.world_pin_attested is False
    assert quiet.tier == 0
    assert "pinned-commit half" in quiet.tier_reason

    # The honest case still earns it: the engine attested the pinned revision.
    good = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "good",
        gym_factory=_factory(),
    )
    assert good.world_pin_observed == "broncho-synthetic-v1"
    assert good.world_pin_attested is True
    assert good.tier == 1


def test_non_finite_engine_output_is_not_flattened_before_it_is_compared(
    tmp_path: Path,
) -> None:
    """A diverged solver's ``nan`` is neither 0.0 newtons nor bitwise agreement.

    Original defect: the recorder normalized every ``nan``/``+inf``/``-inf`` to
    ``0.0`` before the trajectory was built, so two runs whose physics blew up
    differently digested identically and measured ``bitwise`` - and the gate
    bound to the force read 0.0 N and passed. Non-finite telemetry *is* the
    divergence signal, which makes this the worst case rather than a rounding
    detail.
    """
    from or_audit.audit.canonical import digest
    from or_audit.eval.gym_world import jsonable

    # The reviewer's probe: the two divergences must not share a digest.
    assert digest([{"telemetry": jsonable(float("nan"))}]) != digest(
        [{"telemetry": jsonable(float("inf"))}]
    )

    # A NaN force is not a passing gate: it is a gate nobody could assess.
    abstained = run_conformance(
        _broncho_task(tmp_path, name="broncho-nan"),
        n=2,
        workdir=tmp_path / "nan",
        gym_factory=_factory(force=float("nan")),
    )
    counts = abstained.check(CHECK_GATE_STATES).gate_states
    assert [count.id for count in counts] == ["airway_wall_puncture"]
    assert (counts[0].n_pass, counts[0].n_fail) == (0, 0)
    assert counts[0].n_not_assessable == 2
    assert abstained.check(CHECK_GATE_STATES).passed is False
    # Replay still reconstitutes: the tagged value is what both the live
    # scoring pass and the stored trajectory hand the verifier.
    assert abstained.check(CHECK_EVIDENCE_REPLAY).passed is True
    assert abstained.tier == 0

    # Two runs that diverged differently are not one reproducible world.
    diverged = run_conformance(
        _broncho_task(tmp_path, name="broncho-divergent"),
        n=2,
        workdir=tmp_path / "divergent",
        gym_factory=_diverging_factory(),
    )
    assert diverged.determinism_class is DeterminismClass.NONDETERMINISTIC
    evidence = diverged.check(CHECK_DETERMINISM).determinism
    assert evidence is not None
    assert evidence.identical_digests is False
    # The divergence reaches the *published vector*, not only the trace: the
    # gate's kernel-hashed evidence digest differs because `nan` and `+inf` are
    # finally different bytes. Under the old recorder both were 0.0 and every
    # digest matched.
    assert evidence.vectors_equal is False
    assert "evidence[0].digest" in evidence.first_difference
    stored = json.loads(
        (
            tmp_path / "divergent" / RUN_A / "trial-broncho-airway-nav-0" / "trajectory.json"
        ).read_text(encoding="utf-8")
    )
    assert stored[-1]["info"]["max_contact_force_n"] == "__nonfinite__:nan"
    assert diverged.tier == 0


#: A verifier that reports a diverged force as a raw ``nan`` metric value. The
#: shipped example verifiers abstain instead, so this is the only way to drive
#: the boundary a third-party package can still reach.
_NAN_METRIC_VERIFIER = '''"""Fixture verifier reporting a diverged force as a raw NaN."""

from __future__ import annotations

from typing import Any


class NanMetricVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        info = context.get("info", {})
        reached = bool(info.get("target_reached", False))
        return {
            "gates": {},
            "metrics": {
                "safe_navigation": reached,
                "target_reached": reached,
                "max_contact_force_n": float("nan"),
                "diverged": bool(info.get("diverged", False)),
            },
        }


def load_verifier(*, root: Any = None) -> NanMetricVerifier:
    del root
    return NanMetricVerifier()
'''


def test_a_verifier_reporting_a_non_finite_metric_yields_null_not_a_traceback(
    tmp_path: Path,
) -> None:
    """A NaN metric is unassessable, never a number and never a crash.

    Original defect: ``MetricOutcome`` accepted ``float('nan')``, so this
    package passed verifier validation and then aborted the whole job with
    ``ValueError: non-finite float nan cannot be canonicalized`` from inside
    ``compute_head``. The canonicalizer refusing to hash it was the only thing
    preventing a published NaN, and that was an accident rather than a
    contract - the operator saw a digest-layer error instead of a statement
    about their metric.
    """
    task_dir = _broncho_task(tmp_path, name="broncho-nan-metric")
    (task_dir / "verifier.py").write_text(_NAN_METRIC_VERIFIER, encoding="utf-8")

    report = run_conformance(
        task_dir,
        n=2,
        workdir=tmp_path / "work",
        gym_factory=_factory(),
    )

    result = read_job_result(tmp_path / "work" / RUN_A)
    force = result.trials[0].vector.metric("max_contact_force_n")
    assert force is not None
    assert force.value is None
    # The head still computes, so the run is publishable rather than aborted.
    assert result.head
    # And the gate is untouched: the kernel reads the real 0.5 N from `info`,
    # not the verifier's self-reported number.
    gate = result.trials[0].vector.gate("airway_wall_puncture")
    assert gate is not None
    assert gate.status is GateStatus.PASS
    assert report.check(CHECK_EVIDENCE_REPLAY).passed is True
    scorecard = json.loads((tmp_path / "work" / RUN_A / "scorecard.json").read_text())
    row = next(item for item in scorecard["metrics"] if item["id"] == "max_contact_force_n")
    assert (row["assessed"], row["unassessable"]) == (0, 2)
    assert row["mean"] is None


def test_a_closed_loop_adapter_cannot_waive_provenance_by_declaring_no_physics() -> None:
    """``stepped_world`` follows the route, not the ``physics`` capability.

    Original defect: it was derived from ``capabilities.physics``, so an
    adapter declaring ``closed_loop=True, physics=False`` - a world a policy
    genuinely steps - read as dataset-backed. Its report could then declare
    ``stepped_world=False``, pass the registry cross-check, and certify its own
    synthetic stand-in as Tier 1 with ``engine_attested`` True.
    """
    from or_audit.errors import TaskContractError

    spec = attach_world_adapter(
        "kinematic-loop",
        capabilities=WorldCapabilities(closed_loop=True, physics=False),
        factory=_frame_source_adapter,
        provider="conformance-test",
    )
    fields: dict[str, Any] = {
        "task_id": "t",
        "task_version": "0",
        "task_digest": "d",
        "world_kind": "kinematic-loop",
        "adapter_identity": spec.adapter_identity,
        "adapter_pinned": True,
        "interaction_mode": InteractionMode.CLOSED_LOOP,
        "determinism_class": DeterminismClass.BITWISE,
        "tolerance": 1e-9,
        "checks": _passing_checks(),
        "tier": 1,
        "tier_reason": "fixture",
    }

    # The reviewer's probe verbatim: this validated at Tier 1.
    with pytest.raises(TaskContractError, match="cannot reclassify the world"):
        ConformanceReport(**fields, stepped_world=False, backend=BACKEND_SYNTHETIC_STUB)

    # Told the truth about the route, the stand-in cannot carry the claim.
    honest = ConformanceReport(
        **{**fields, "tier": 0, "tier_reason": "stub"},
        stepped_world=True,
        backend=BACKEND_SYNTHETIC_STUB,
    )
    assert honest.stepped_world is True
    assert honest.engine_attested is False
    assert honest.tier1_eligible is False

    # And a closed-loop report on a kind whose adapter does not grant
    # closed_loop is refused: it attests a world the adapter never offered.
    attach_world_adapter(
        "still-frames",
        capabilities=WorldCapabilities(),
        factory=_frame_source_adapter,
        provider="conformance-test",
    )
    with pytest.raises(TaskContractError, match="does not declare closed_loop"):
        ConformanceReport(
            **{**fields, "world_kind": "still-frames"},
            stepped_world=True,
            backend=BACKEND_REAL,
        )


def test_license_declaration_contract_resolution_order(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    assert declared_package_license(root).spdx == ""

    (root / "LICENSE").write_text("Copyright.\nSPDX-License-Identifier: Apache-2.0\n")
    assert declared_package_license(root).spdx == "Apache-2.0"

    # A tag beats a bare LICENSE marker; wrap.json and license.toml beat the tag.
    assert declared_package_license(root, ("video", "license:BSD-3-Clause")).spdx == "BSD-3-Clause"
    (root / "license.toml").write_text('spdx = "ISC"\n')
    assert declared_package_license(root, ("license:BSD-3-Clause",)).spdx == "ISC"
    (root / "wrap.json").write_text(json.dumps({"license": "MIT"}))
    assert declared_package_license(root, ("license:BSD-3-Clause",)).spdx == "MIT"


def test_a_license_text_without_an_spdx_marker_is_not_guessed(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "LICENSE").write_text("Permission is hereby granted, free of charge, ...\n")
    declaration = declared_package_license(root)
    assert declaration.spdx == ""
    assert "no SPDX-License-Identifier" in declaration.source
    assert classify_license(declaration.spdx).status is LicenseStatus.UNKNOWN


def test_spdx_expression_composition() -> None:
    numpy_expression = "BSD-3-Clause AND MIT AND 0BSD AND Zlib AND CC0-1.0"
    assert classify_license(numpy_expression).status is LicenseStatus.ALLOWED
    assert classify_license("MIT AND GPL-3.0-only").status is LicenseStatus.RESTRICTED
    assert classify_license("MIT OR GPL-3.0-only").status is LicenseStatus.ALLOWED
    assert classify_license("AGPL-3.0-only OR SSPL-1.0").status is LicenseStatus.RESTRICTED
    assert classify_license("mit").status is LicenseStatus.ALLOWED
    assert classify_license("NOASSERTION").status is LicenseStatus.RESTRICTED
    assert classify_license("  ").status is LicenseStatus.UNKNOWN


def test_cli_reports_the_tier_and_gates_on_require_tier1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command"))
    task_dir = _pinned_video_task(tmp_path)

    args = parser.parse_args(
        [
            "conformance",
            str(task_dir),
            "-a",
            str(VIDEO_AGENT),
            "-n",
            "2",
            "--out",
            str(tmp_path / "out"),
            "--require-tier1",
        ]
    )
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "[pass] execution-determinism" in out
    assert "determinism  bitwise (measured, tol 1e-09)" in out
    assert "tier         1" in out
    assert (tmp_path / "out" / "conformance.json").is_file()

    unpinned = parser.parse_args(
        [
            "conformance",
            str(VIDEO_TASK),
            "-a",
            str(VIDEO_AGENT),
            "--out",
            str(tmp_path / "out-2"),
            "--require-tier1",
        ]
    )
    assert unpinned.func(unpinned) == 1
    captured = capsys.readouterr()
    assert "tier         0" in captured.out
    assert "REFUSED: tier 0" in captured.err

    # Without --require-tier1 a Tier-0 measurement is a successful report.
    tolerant = parser.parse_args(
        [
            "conformance",
            str(VIDEO_TASK),
            "-a",
            str(VIDEO_AGENT),
            "--out",
            str(tmp_path / "out-3"),
        ]
    )
    assert tolerant.func(tolerant) == 0


def test_cli_refuses_an_unloadable_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        ["conformance", str(tmp_path / "nope"), "--out", str(tmp_path / "out")]
    )
    assert args.func(args) == 1
    assert "REFUSED:" in capsys.readouterr().err


def test_cli_tolerance_is_recorded_in_the_report(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command"))
    task_dir = _pinned_video_task(tmp_path)
    out = tmp_path / "out"
    args = parser.parse_args(
        [
            "conformance",
            str(task_dir),
            "-a",
            str(VIDEO_AGENT),
            "--out",
            str(out),
            "--tolerance",
            "1e-6",
        ]
    )
    assert args.func(args) == 0
    payload = json.loads((out / "conformance.json").read_text(encoding="utf-8"))
    assert payload["tolerance"] == pytest.approx(1e-6)
    assert payload["checks"][3]["determinism"]["tolerance"] == pytest.approx(1e-6)


def test_agent_package_loads_for_the_pinned_task(tmp_path: Path) -> None:
    """Guard the fixture itself: the pinned copy still binds the real agent."""
    task = load_task(_pinned_video_task(tmp_path))
    agent = load_agent(VIDEO_AGENT)
    assert task.environment.adapter
    assert agent.id == "example/video-predictor"


def test_the_license_script_still_gates_the_runtime_closure() -> None:
    """The CI gate keeps its behaviour after moving its data into the library."""
    completed = subprocess.run(
        [sys.executable, "scripts/check_license_allowlist.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "runtime dependencies within commercial allowlist" in completed.stdout


def test_the_license_script_owns_no_copy_of_the_allowlist() -> None:
    """One table, two consumers: a second copy is how the two verdicts drift apart."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import check_license_allowlist  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))
    assert check_license_allowlist.THIRD_PARTY is licensing.THIRD_PARTY
    assert check_license_allowlist.ALLOWLIST is licensing.ALLOWLIST


def _strip_gates(body: str) -> str:
    """Remove every ``[[verifier.gates]]`` block, sub-tables included."""
    out: list[str] = []
    skipping = False
    for line in body.splitlines():
        if line.startswith("[[verifier.gates]]"):
            skipping = True
            continue
        if skipping:
            if line.startswith("[verifier.gates"):
                continue
            if line.startswith("["):
                skipping = False
            else:
                continue
        out.append(line)
    return "\n".join(out) + "\n"
