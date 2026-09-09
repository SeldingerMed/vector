"""Unit tests for statistical uncertainty intervals and scorecard views (Phase D4/D5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import MetricKind
from or_audit.eval.job import JobResult, TrialRecord
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import run_job
from or_audit.eval.scorecard import render_markdown, scorecard_data
from or_audit.eval.trace import ProceduralTrace, TraceStep
from or_audit.eval.uncertainty import (
    bootstrap_mean_ci,
    clustered_bootstrap_mean_ci,
    normal_z,
    wilson_score_interval,
)
from or_audit.eval.vector import GateOutcome, MetricOutcome, TrialVector

ROOT = Path(__file__).resolve().parents[1]
VIDEO_TASK = ROOT / "docs/examples/tasks/video-nextstep"
VIDEO_AGENT = ROOT / "docs/examples/agents/example-video-predictor"


def test_normal_z_values() -> None:
    assert normal_z(0.95) == pytest.approx(1.95996, rel=1e-3)
    assert normal_z(0.90) == pytest.approx(1.64485, rel=1e-3)
    with pytest.raises(TaskContractError, match="must be in"):
        normal_z(0.0)
    with pytest.raises(TaskContractError, match="must be in"):
        normal_z(1.0)


def test_wilson_zero_failures_is_not_zero_risk() -> None:
    low, high = wilson_score_interval(3, 3, confidence=0.95)
    assert high == 1.0
    assert low < 0.50

    low, high = wilson_score_interval(0, 3, confidence=0.95)
    assert low == 0.0
    assert high > 0.50

    with pytest.raises(TaskContractError, match="cannot exceed total"):
        wilson_score_interval(5, 3)
    with pytest.raises(TaskContractError, match="must be non-negative"):
        wilson_score_interval(-1, 3)


def test_bootstrap_mean_ci() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = bootstrap_mean_ci(values, confidence=0.95, seed=42)
    assert 1.0 <= low <= high <= 5.0
    assert bootstrap_mean_ci([3.14]) == (3.14, 3.14)

    with pytest.raises(TaskContractError, match="needs at least one"):
        bootstrap_mean_ci([])


def test_clustered_bootstrap_mean_ci() -> None:
    clusters = {
        "patient-1": [10.0, 10.5, 9.8],
        "patient-2": [2.0, 2.2],
        "patient-3": [5.0, 5.1, 4.9],
    }
    low, high = clustered_bootstrap_mean_ci(clusters, confidence=0.95, seed=42)
    assert 2.0 <= low <= high <= 10.5

    with pytest.raises(TaskContractError, match="needs at least one cluster"):
        clustered_bootstrap_mean_ci({})


def test_scorecard_data_includes_uncertainty_intervals(tmp_path: Path) -> None:
    out = tmp_path / "job"
    result = run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=out,
        n=3,
        split="test",
    )
    data = scorecard_data(result)
    assert "metrics" in data
    for metric in data["metrics"]:
        if metric["kind"] in ("boolean", "continuous"):
            assert "ci_95" in metric
            if metric["assessed"] > 0:
                assert metric["ci_95"] is not None
                assert len(metric["ci_95"]) == 2
                assert metric["ci_95"][0] <= metric["ci_95"][1]

    # No declared scenario metadata in video-nextstep -> subgroups must NOT invent seed groups!
    assert data["subgroups"] == []
    assert data["worst_case"] is None

    markdown = render_markdown(result)
    assert "| 95% CI |" in markdown
    assert "Subgroups" not in markdown


def test_scorecard_coverage_and_subgroup_analysis() -> None:
    vector_pass = TrialVector(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        seed=0,
        gates=(GateOutcome(id="g1", status=GateStatus.PASS),),
        metrics=(MetricOutcome(id="m1", kind=MetricKind.BOOLEAN, headline=True, value=True),),
    )
    vector_abstained = TrialVector(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        seed=1,
        gates=(GateOutcome(id="g1", status=GateStatus.NOT_ASSESSABLE, abstained=True),),
        metrics=(
            MetricOutcome(id="m1", kind=MetricKind.BOOLEAN, headline=True, value=None),
            MetricOutcome(id="abstained", kind=MetricKind.BOOLEAN, headline=False, value=True),
        ),
    )
    vector_fail = TrialVector(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        seed=2,
        gates=(GateOutcome(id="g1", status=GateStatus.FAIL),),
        metrics=(MetricOutcome(id="m1", kind=MetricKind.BOOLEAN, headline=True, value=False),),
    )

    step_sc1 = TraceStep.model_validate(
        {
            "index": 0,
            "interaction_mode": "single-turn",
            "scenario": {"id": "scenario-a", "seed": 0},
        }
    )
    step_sc2 = TraceStep.model_validate(
        {
            "index": 0,
            "interaction_mode": "single-turn",
            "scenario": {"id": "scenario-b", "seed": 1},
        }
    )

    trial0 = TrialRecord(seed=0, vector=vector_pass, trajectory=ProceduralTrace((step_sc1,)))
    trial1 = TrialRecord(seed=1, vector=vector_abstained, trajectory=ProceduralTrace((step_sc1,)))
    trial2 = TrialRecord(seed=2, vector=vector_fail, trajectory=ProceduralTrace((step_sc2,)))

    result = JobResult(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=3,
        headline="m1",
        trials=(trial0, trial1, trial2),
        headline_true=1,
        headline_false=1,
        headline_unassessable=1,
        any_gate_failed=1,
    )

    data = scorecard_data(result)
    assert data["coverage"]["abstained"] == 1
    assert data["coverage"]["coverage"] == pytest.approx(2 / 3, rel=1e-3)
    assert data["coverage"]["risk_at_coverage"] == 0.5

    assert len(data["subgroups"]) == 2
    sg_a = next(sg for sg in data["subgroups"] if sg["subgroup"] == "scenario-a")
    sg_b = next(sg for sg in data["subgroups"] if sg["subgroup"] == "scenario-b")
    assert sg_a["count"] == 2
    assert sg_a["is_underpowered"] is True
    assert sg_b["count"] == 1
    assert sg_b["rate"] == 0.0
    assert data["worst_case"]["subgroup"] == "scenario-b"

    markdown = render_markdown(result)
    assert "## Risk vs coverage" in markdown
    assert "## Subgroups and worst-case analysis" in markdown
    assert "**Worst-case subgroup:** `scenario-b`" in markdown
