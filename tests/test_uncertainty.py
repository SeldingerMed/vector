"""Unit tests for statistical uncertainty intervals and scorecard views (Phase D4/D5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import MetricDirection, MetricKind
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
    from or_audit.eval.contracts import MetricDirection

    vector_pass = TrialVector(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        seed=0,
        gates=(GateOutcome(id="g1", status=GateStatus.PASS),),
        metrics=(
            MetricOutcome(
                id="m1",
                kind=MetricKind.BOOLEAN,
                headline=True,
                value=True,
                direction=MetricDirection.MAXIMIZE,
            ),
        ),
    )
    vector_abstained = TrialVector(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        seed=1,
        gates=(GateOutcome(id="g1", status=GateStatus.NOT_ASSESSABLE, abstained=True),),
        metrics=(
            MetricOutcome(
                id="m1",
                kind=MetricKind.BOOLEAN,
                headline=True,
                value=None,
                direction=MetricDirection.MAXIMIZE,
            ),
            MetricOutcome(id="abstained", kind=MetricKind.BOOLEAN, headline=False, value=True),
        ),
    )
    vector_fail = TrialVector(
        task_id="test-task",
        task_version="1",
        agent_identity="agent",
        seed=2,
        gates=(GateOutcome(id="g1", status=GateStatus.FAIL),),
        metrics=(
            MetricOutcome(
                id="m1",
                kind=MetricKind.BOOLEAN,
                headline=True,
                value=False,
                direction=MetricDirection.MAXIMIZE,
            ),
        ),
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

    trial0 = TrialRecord(
        seed=0,
        vector=vector_pass,
        trajectory=ProceduralTrace((step_sc1,)),
        patient_id="patient-1",
        subgroups={"anatomy": "bifurcation"},
    )
    trial1 = TrialRecord(
        seed=1,
        vector=vector_abstained,
        trajectory=ProceduralTrace((step_sc1,)),
        patient_id="patient-1",
        subgroups={"anatomy": "bifurcation"},
    )
    trial2 = TrialRecord(
        seed=2,
        vector=vector_fail,
        trajectory=ProceduralTrace((step_sc2,)),
        patient_id="patient-2",
        subgroups={"anatomy": "straight"},
    )

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
    sg_bif = next(sg for sg in data["subgroups"] if sg["subgroup"] == "bifurcation")
    sg_str = next(sg for sg in data["subgroups"] if sg["subgroup"] == "straight")
    assert sg_bif["count"] == 2
    assert sg_bif["is_underpowered"] is True
    assert sg_str["count"] == 1
    assert sg_str["rate"] == 0.0
    assert data["worst_case"]["subgroup"] == "straight"
    assert data["worst_case"]["axis"] == "anatomy"
    # Patient identifiers must never be exposed as subgroups
    subgroup_names = [sg["subgroup"] for sg in data["subgroups"]]
    assert "patient-1" not in subgroup_names
    assert "patient-2" not in subgroup_names

    markdown = render_markdown(result)
    assert "## Risk vs coverage" in markdown
    assert "## Subgroups and worst-case analysis" in markdown
    assert "**Worst-case subgroup:** `anatomy=straight`" in markdown


def test_generated_benchmark_report_fixture(tmp_path: Path) -> None:
    import json
    import shutil

    from or_audit.eval.scorecard import write_scorecards

    task_src = ROOT / "docs/examples/tasks/video-nextstep"
    agent_src = ROOT / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "benchmark-report-task"
    shutil.copytree(task_src, task_dir)

    # Manifest defining explicit cohorts and patient clusters
    splits_data = {
        "format_version": "1",
        "dataset_id": "benchmark-video-dataset",
        "dataset_revision": "1.0",
        "disjoint_by": ["case", "patient"],
        "entries": [
            {
                "case_id": "case-01",
                "episode_id": "ep-01",
                "patient_id": "patient-A",
                "site_id": "site-1",
                "split": "test",
                "item_ids": ["clip-001"],
                "subgroups": {"anatomy": "tortuous", "scanner": "siemens"},
            },
            {
                "case_id": "case-02",
                "episode_id": "ep-02",
                "patient_id": "patient-A",
                "site_id": "site-1",
                "split": "test",
                "item_ids": ["clip-002"],
                "subgroups": {"anatomy": "tortuous", "scanner": "siemens"},
            },
            {
                "case_id": "case-03",
                "episode_id": "ep-03",
                "patient_id": "patient-B",
                "site_id": "site-2",
                "split": "test",
                "item_ids": ["clip-003"],
                "subgroups": {"anatomy": "straight", "scanner": "ge"},
            },
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    out = tmp_path / "benchmark-out"
    result = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(agent_src),
        agent_dir=agent_src,
        out=out,
        n=3,
        split="test",
    )

    write_scorecards(out, result)

    scorecard_json_path = out / "scorecard.json"
    scorecard_md_path = out / "scorecard.md"
    scorecard_html_path = out / "scorecard.html"
    assert scorecard_json_path.is_file()
    assert scorecard_md_path.is_file()
    assert scorecard_html_path.is_file()

    report = json.loads(scorecard_json_path.read_text(encoding="utf-8"))
    assert report["n"] == 3
    assert report["independent_cases"] == 3
    assert report["split_manifest_digest"] == result.split_manifest_digest

    # Verify uncertainty intervals and methods on metrics
    headline_metric = next(m for m in report["metrics"] if m["headline"])
    assert "ci_95" in headline_metric
    assert headline_metric["ci_method"] == "wilson"
    assert headline_metric["confidence"] == 0.95

    # Verify subgroup analysis: must be grouped along declared cohort axes, NOT patient IDs!
    assert len(report["subgroups"]) >= 2
    axes = {sg["axis"] for sg in report["subgroups"]}
    assert "anatomy" in axes or "scanner" in axes
    assert "patient" not in axes

    md = scorecard_md_path.read_text(encoding="utf-8")
    assert "| 95% CI |" in md
    assert "## Subgroups and worst-case analysis" in md
    assert "Worst-case subgroup (anatomy):" in md
    assert "Worst-case subgroup (scanner):" in md


def test_leaderboard_renders_uncertainty_intervals(tmp_path: Path) -> None:
    from or_audit.eval.leaderboard import leaderboard_data, render_html

    task_src = ROOT / "docs/examples/tasks/video-nextstep"
    agent_src = ROOT / "docs/examples/agents/example-video-predictor"

    out = tmp_path / "lb-job"
    run_job(
        task=load_task(task_src),
        task_dir=task_src,
        agent=load_agent(agent_src),
        agent_dir=agent_src,
        out=out,
        n=3,
        split="test",
    )

    data = leaderboard_data([out])
    row = data["rows"][0]
    headline_metric = row["metrics"][row["headline"]]
    assert "ci_95" in headline_metric
    assert headline_metric["ci_95"] is not None

    html_text = render_html(data)
    assert "[" in html_text
    assert "]" in html_text


def test_subgroups_validation_constraints() -> None:
    from or_audit.eval.split import validate_subgroups

    # Valid mapping passes
    valid = validate_subgroups({"anatomy": "bifurcation", "scanner_model": "siemens-1"})
    assert valid == {"anatomy": "bifurcation", "scanner_model": "siemens-1"}

    # Reserved keys rejected
    with pytest.raises(TaskContractError, match="reserved identifier"):
        validate_subgroups({"patient": "p1"})
    with pytest.raises(TaskContractError, match="reserved identifier"):
        validate_subgroups({"site_id": "s1"})
    with pytest.raises(TaskContractError, match="reserved identifier"):
        validate_subgroups({"case": "c1"})

    # Non-slug key rejected
    with pytest.raises(TaskContractError, match="must match slug pattern"):
        validate_subgroups({"Invalid Key!": "val"})

    # Non-string value rejected
    with pytest.raises(TaskContractError, match="keys and values must be strings"):
        validate_subgroups({"anatomy": 123})

    # Empty key or value rejected
    with pytest.raises(TaskContractError, match="cannot be empty"):
        validate_subgroups({"": "val"})
    with pytest.raises(TaskContractError, match="cannot be empty"):
        validate_subgroups({"anatomy": ""})

    # Exceeding maximum 16 axes rejected
    too_many = {f"axis-{i}": f"val-{i}" for i in range(17)}
    with pytest.raises(TaskContractError, match="exceeds maximum of 16 axes"):
        validate_subgroups(too_many)


def test_trial_binding_model_strictness() -> None:
    from pydantic import ValidationError

    from or_audit.eval.job import TrialBinding

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        TrialBinding.model_validate({"case_id": "c1", "unknown_field": "val"})


def test_crash_resume_restores_bindings_intact(tmp_path: Path) -> None:
    from or_audit.eval.job import read_partial_trials

    task_src = ROOT / "docs/examples/tasks/video-nextstep"
    agent_src = ROOT / "docs/examples/agents/example-video-predictor"

    out = tmp_path / "job-resume-bindings"
    original = run_job(
        task=load_task(task_src),
        task_dir=task_src,
        agent=load_agent(agent_src),
        agent_dir=agent_src,
        out=out,
        n=2,
        split="test",
    )
    assert original.trials[0].case_id
    assert original.trials[0].patient_id

    # Simulate crash by removing result.json
    (out / "result.json").unlink()

    # Verify read_partial_trials recovers bindings
    records, _ = read_partial_trials(out, "video-nextstep", require_binding=True)
    assert len(records) == 2
    assert records[0].case_id == original.trials[0].case_id
    assert records[0].patient_id == original.trials[0].patient_id
    assert records[0].site_id == original.trials[0].site_id

    # Resume the job
    resumed = run_job(
        task=load_task(task_src),
        task_dir=task_src,
        agent=load_agent(agent_src),
        agent_dir=agent_src,
        out=out,
        n=2,
        split="test",
        resume=True,
    )
    assert resumed.head == original.head
    assert resumed.trials[0].case_id == original.trials[0].case_id
    assert resumed.trials[0].patient_id == original.trials[0].patient_id


def test_corrupt_or_missing_binding_json_refuses(tmp_path: Path) -> None:
    from or_audit.eval.job import read_partial_trials

    task_src = ROOT / "docs/examples/tasks/video-nextstep"
    agent_src = ROOT / "docs/examples/agents/example-video-predictor"

    out = tmp_path / "job-corrupt-binding"
    run_job(
        task=load_task(task_src),
        task_dir=task_src,
        agent=load_agent(agent_src),
        agent_dir=agent_src,
        out=out,
        n=2,
        split="test",
    )
    (out / "result.json").unlink()

    # Corrupt binding.json
    b_path = out / "trial-video-nextstep-0" / "binding.json"
    assert b_path.is_file()
    b_path.write_text("invalid-json{", encoding="utf-8")
    with pytest.raises(TaskContractError, match=r"invalid binding\.json"):
        read_partial_trials(out, "video-nextstep", require_binding=True)

    # Remove binding.json when required
    b_path.unlink()
    with pytest.raises(TaskContractError, match=r"missing binding\.json"):
        read_partial_trials(out, "video-nextstep", require_binding=True)


def test_subgroup_worst_case_respects_minimize_direction() -> None:
    from or_audit.eval.contracts import MetricDirection

    m_low = MetricOutcome(
        id="max_pen",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=0.05,
        direction=MetricDirection.MINIMIZE,
    )
    m_high = MetricOutcome(
        id="max_pen",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=0.25,
        direction=MetricDirection.MINIMIZE,
    )

    trial_low = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="pen-task",
            task_version="1",
            agent_identity="agent",
            seed=0,
            gates=(),
            metrics=(m_low,),
        ),
        trajectory=ProceduralTrace(()),
        subgroups={"scanner": "scanner-A"},
    )
    trial_high = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="pen-task",
            task_version="1",
            agent_identity="agent",
            seed=1,
            gates=(),
            metrics=(m_high,),
        ),
        trajectory=ProceduralTrace(()),
        subgroups={"scanner": "scanner-B"},
    )

    result = JobResult(
        task_id="pen-task",
        task_version="1",
        agent_identity="agent",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=2,
        headline="max_pen",
        trials=(trial_low, trial_high),
        headline_true=0,
        headline_false=0,
        headline_unassessable=0,
        any_gate_failed=0,
    )

    data = scorecard_data(result)
    assert len(data["subgroups"]) == 2
    # For a minimize metric, worst-case is the HIGHEST value (scanner-B: 0.25)
    assert data["worst_case"]["subgroup"] == "scanner-B"
    assert data["worst_case"]["rate"] == 0.25


def test_subgroup_neutral_metric_produces_no_worst_case() -> None:
    from or_audit.eval.contracts import MetricDirection

    m0 = MetricOutcome(
        id="neutral_metric",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=1.0,
        direction=MetricDirection.NEUTRAL,
    )
    m1 = MetricOutcome(
        id="neutral_metric",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=5.0,
        direction=MetricDirection.NEUTRAL,
    )
    t0 = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=0, gates=(), metrics=(m0,)
        ),
        subgroups={"anatomy": "bifurcation"},
    )
    t1 = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=1, gates=(), metrics=(m1,)
        ),
        subgroups={"anatomy": "straight"},
    )
    result = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=2,
        headline="neutral_metric",
        trials=(t0, t1),
        headline_true=0,
        headline_false=0,
        headline_unassessable=0,
        any_gate_failed=0,
    )
    data = scorecard_data(result)
    assert len(data["subgroups"]) == 2
    # Neutral metric must NOT produce any worst case
    assert data["worst_cases"] == {}
    assert data["worst_case"] is None


def test_subgroups_multi_axis_ranks_within_each_axis() -> None:
    m_pass = MetricOutcome(
        id="m",
        kind=MetricKind.BOOLEAN,
        headline=True,
        value=True,
        direction=MetricDirection.MAXIMIZE,
    )
    m_fail = MetricOutcome(
        id="m",
        kind=MetricKind.BOOLEAN,
        headline=True,
        value=False,
        direction=MetricDirection.MAXIMIZE,
    )

    t0 = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=0, gates=(), metrics=(m_pass,)
        ),
        subgroups={"anatomy": "bifurcation", "scanner": "siemens"},
    )
    t1 = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=1, gates=(), metrics=(m_fail,)
        ),
        subgroups={"anatomy": "straight", "scanner": "ge"},
    )
    result = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=2,
        headline="m",
        trials=(t0, t1),
        headline_true=1,
        headline_false=1,
        headline_unassessable=0,
        any_gate_failed=0,
    )
    data = scorecard_data(result)
    assert "anatomy" in data["worst_cases"]
    assert "scanner" in data["worst_cases"]
    assert data["worst_cases"]["anatomy"]["subgroup"] == "straight"
    assert data["worst_cases"]["scanner"]["subgroup"] == "ge"
    # Legacy worst_case is None when multiple axes exist
    assert data["worst_case"] is None


def test_clustered_bootstrap_rejects_empty_cluster() -> None:
    clusters = {"patient-A": [1.0, 2.0], "patient-B": []}
    with pytest.raises(TaskContractError, match="requires non-empty observations"):
        clustered_bootstrap_mean_ci(clusters)


def test_clustered_bootstrap_unequal_size_discriminates_from_pooled() -> None:
    # 4 clusters with 1 observation of 100.0 (cluster mean = 100.0)
    # 1 cluster with 20 observations of 0.0 each (cluster mean = 0.0)
    clusters = {
        "patient-A1": [100.0],
        "patient-A2": [100.0],
        "patient-A3": [100.0],
        "patient-A4": [100.0],
        "patient-B1": [0.0] * 20,
    }
    # Cluster-level mean estimand is (100 + 100 + 100 + 100 + 0) / 5 = 80.0
    low_cl, high_cl = clustered_bootstrap_mean_ci(clusters, confidence=0.95, seed=42)
    assert low_cl >= 40.0
    assert high_cl <= 100.0

    # Raw pooled bootstrap on all 24 observations: 4 * 100.0 + 20 * 0.0 -> pooled mean is 16.67!
    pooled_values = [100.0] * 4 + [0.0] * 20
    _low_pool, high_pool = bootstrap_mean_ci(pooled_values, confidence=0.95, seed=42)
    assert high_pool < 35.0
    assert high_pool < low_cl  # Completely discriminates cluster-mean weighting!


def test_subgroup_no_assessment_row_reports_na_and_excluded_from_worst_case() -> None:
    # Subgroup A: all trials unassessable (value=None)
    # Subgroup B: 1 pass, 1 fail
    m_unassessable = MetricOutcome(
        id="m",
        kind=MetricKind.BOOLEAN,
        headline=True,
        value=None,
        direction=MetricDirection.MAXIMIZE,
    )
    m_pass = MetricOutcome(
        id="m",
        kind=MetricKind.BOOLEAN,
        headline=True,
        value=True,
        direction=MetricDirection.MAXIMIZE,
    )
    m_fail = MetricOutcome(
        id="m",
        kind=MetricKind.BOOLEAN,
        headline=True,
        value=False,
        direction=MetricDirection.MAXIMIZE,
    )

    t_un1 = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="t",
            task_version="1",
            agent_identity="a",
            seed=0,
            gates=(),
            metrics=(m_unassessable,),
        ),
        subgroups={"anatomy": "unassessed-group"},
    )
    t_un2 = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t",
            task_version="1",
            agent_identity="a",
            seed=1,
            gates=(),
            metrics=(m_unassessable,),
        ),
        subgroups={"anatomy": "unassessed-group"},
    )
    t_b1 = TrialRecord(
        seed=2,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=2, gates=(), metrics=(m_pass,)
        ),
        subgroups={"anatomy": "assessed-group"},
    )
    t_b2 = TrialRecord(
        seed=3,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=3, gates=(), metrics=(m_fail,)
        ),
        subgroups={"anatomy": "assessed-group"},
    )

    result = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=4,
        headline="m",
        trials=(t_un1, t_un2, t_b1, t_b2),
        headline_true=1,
        headline_false=1,
        headline_unassessable=2,
        any_gate_failed=0,
    )

    data = scorecard_data(result)
    assert len(data["subgroups"]) == 2
    sg_un = next(sg for sg in data["subgroups"] if sg["subgroup"] == "unassessed-group")
    sg_b = next(sg for sg in data["subgroups"] if sg["subgroup"] == "assessed-group")

    assert sg_un["assessed"] == 0
    assert sg_un["unassessable"] == 2
    assert sg_un["rate"] is None
    assert sg_un["ci_95"] is None

    assert sg_b["assessed"] == 2
    assert sg_b["unassessable"] == 0
    assert sg_b["rate"] == 0.5

    # Worst-case must be assessed-group (rate=0.5), NEVER unassessed-group!
    assert data["worst_case"]["subgroup"] == "assessed-group"
    assert data["worst_case"]["rate"] == 0.5

    md = render_markdown(result)
    assert "| unassessed-group | 2 | 0 | 2 | n/a | n/a |" in md
    assert "**Worst-case subgroup:** `anatomy=assessed-group`" in md


def test_continuous_subgroup_pass_counts_assessed_only() -> None:
    m_val1 = MetricOutcome(
        id="metric",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=1.0,
        direction=MetricDirection.MAXIMIZE,
    )
    m_val2 = MetricOutcome(
        id="metric",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=2.0,
        direction=MetricDirection.MAXIMIZE,
    )
    m_unassessable = MetricOutcome(
        id="metric",
        kind=MetricKind.CONTINUOUS,
        headline=True,
        value=None,
        direction=MetricDirection.MAXIMIZE,
    )

    t0 = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="t",
            task_version="1",
            agent_identity="a",
            seed=0,
            gates=(GateOutcome(id="g", status=GateStatus.FAIL),),
            metrics=(m_val1,),
        ),
        subgroups={"cohort": "group-A"},
    )
    t1 = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t",
            task_version="1",
            agent_identity="a",
            seed=1,
            gates=(GateOutcome(id="g", status=GateStatus.PASS),),
            metrics=(m_val2,),
        ),
        subgroups={"cohort": "group-A"},
    )
    t2 = TrialRecord(
        seed=2,
        vector=TrialVector(
            task_id="t",
            task_version="1",
            agent_identity="a",
            seed=2,
            gates=(GateOutcome(id="g", status=GateStatus.PASS),),
            metrics=(m_unassessable,),
        ),
        subgroups={"cohort": "group-A"},
    )
    t3 = TrialRecord(
        seed=3,
        vector=TrialVector(
            task_id="t",
            task_version="1",
            agent_identity="a",
            seed=3,
            gates=(GateOutcome(id="g", status=GateStatus.PASS),),
            metrics=(m_val1,),
        ),
        subgroups={"cohort": "group-B"},
    )

    result = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=4,
        headline="metric",
        trials=(t0, t1, t2, t3),
        headline_true=0,
        headline_false=0,
        headline_unassessable=1,
        any_gate_failed=1,
    )

    data = scorecard_data(result)
    sg_a = next(sg for sg in data["subgroups"] if sg["subgroup"] == "group-A")
    assert sg_a["count"] == 3
    assert sg_a["assessed"] == 2
    assert sg_a["unassessable"] == 1
    # s_pass must be 1 (only trial 1 passed among assessed trials), NOT 2!
    assert sg_a["pass"] == 1
    assert sg_a["rate"] == 1.5

    md = render_markdown(result)
    assert "| Estimate |" in md
    assert "with estimate" in md


def test_scorecard_continuous_metric_cluster_activation_and_seed() -> None:
    m0 = MetricOutcome(id="cont", kind=MetricKind.CONTINUOUS, headline=True, value=10.0)
    m1 = MetricOutcome(id="cont", kind=MetricKind.CONTINUOUS, headline=True, value=20.0)

    # 1. Without declared patient/case bindings -> must be "bootstrap", NOT "clustered_bootstrap"!
    t_unclustered_0 = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=0, gates=(), metrics=(m0,)
        ),
    )
    t_unclustered_1 = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=1, gates=(), metrics=(m1,)
        ),
    )
    res_unclustered = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=2,
        headline="cont",
        trials=(t_unclustered_0, t_unclustered_1),
        headline_true=0,
        headline_false=0,
        headline_unassessable=0,
        any_gate_failed=0,
    )
    data_unclustered = scorecard_data(res_unclustered)
    m_data = next(m for m in data_unclustered["metrics"] if m["headline"])
    assert m_data["ci_method"] == "bootstrap"
    assert m_data["resampling_seed"] == 0
    assert m_data["draws"] == 1000
    assert m_data["confidence"] == 0.95

    # 2. With declared patient clusters -> activates "clustered_bootstrap"
    t_clustered_0 = TrialRecord(
        seed=0,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=0, gates=(), metrics=(m0,)
        ),
        patient_id="patient-1",
    )
    t_clustered_1 = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=1, gates=(), metrics=(m1,)
        ),
        patient_id="patient-2",
    )
    res_clustered = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=2,
        headline="cont",
        trials=(t_clustered_0, t_clustered_1),
        headline_true=0,
        headline_false=0,
        headline_unassessable=0,
        any_gate_failed=0,
        independent_case_unit="patient",
    )
    data_clustered = scorecard_data(res_clustered)
    m_data_cl = next(m for m in data_clustered["metrics"] if m["headline"])
    assert m_data_cl["ci_method"] == "clustered_bootstrap"
    assert m_data_cl["resampling_seed"] == 0
    assert m_data_cl["draws"] == 1000
    assert m_data_cl["confidence"] == 0.95

    # 3. If independent_case_unit="patient" but a trial lacks patient_id -> raises TaskContractError
    t_no_patient = TrialRecord(
        seed=1,
        vector=TrialVector(
            task_id="t", task_version="1", agent_identity="a", seed=1, gates=(), metrics=(m1,)
        ),
    )
    res_missing = JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a",
        world_pin="pin",
        interface_id="intf",
        interaction_mode="single-turn",
        task_digest="td",
        agent_digest="ad",
        n=2,
        headline="cont",
        trials=(t_clustered_0, t_no_patient),
        headline_true=0,
        headline_false=0,
        headline_unassessable=0,
        any_gate_failed=0,
        independent_case_unit="patient",
    )
    with pytest.raises(TaskContractError, match="missing on trial"):
        scorecard_data(res_missing)
