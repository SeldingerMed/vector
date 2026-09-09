"""Paired comparison: shared seeds, dropped unknowns, seeded intervals."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.compare import bootstrap_ci, compare_jobs
from or_audit.eval.job import JobResult, TrialRecord
from or_audit.eval.vector import TrialVector


def _job(values: Mapping[int, Any], *, kind: str = "boolean") -> JobResult:
    trials = tuple(
        TrialRecord(
            seed=seed,
            vector=TrialVector(
                task_id="t",
                task_version="1",
                agent_identity="a@0",
                seed=seed,
                gates=(),
                metrics=({"id": "m", "value": value, "kind": kind, "headline": True},),
            ),
        )
        for seed, value in sorted(values.items())
    )
    n = len(trials)
    return JobResult(
        task_id="t",
        task_version="1",
        agent_identity="a@0",
        world_pin="",
        task_digest="d",
        agent_digest="d",
        n=n,
        headline="m",
        trials=trials,
        headline_true=0,
        headline_false=0,
        headline_unassessable=0,
        any_gate_failed=0,
    )


def test_paired_difference_with_known_effect() -> None:
    result = compare_jobs(
        _job({0: False, 1: False, 2: False}), _job({0: True, 1: True, 2: True}), "m"
    )
    assert result.paired_seeds == (0, 1, 2)
    assert result.dropped_unassessable == 0
    assert result.mean_diff == 1.0
    assert result.ci_low <= 1.0 <= result.ci_high


def test_unassessable_pairs_drop_and_count() -> None:
    result = compare_jobs(_job({0: False, 1: False}), _job({0: True, 1: None}), "m")
    assert result.paired_seeds == (0,)
    assert result.dropped_unassessable == 1
    assert result.mean_diff == 1.0


def test_disjoint_seeds_refuse() -> None:
    with pytest.raises(TaskContractError, match="identical seed sets"):
        compare_jobs(_job({0: True}), _job({1: True}), "m")


def test_categorical_metric_refuses() -> None:
    with pytest.raises(TaskContractError, match="categorical"):
        compare_jobs(_job({0: "a"}, kind="categorical"), _job({0: "b"}, kind="categorical"), "m")


def _alternating_pair() -> tuple[JobResult, JobResult]:
    even = {i: float(i % 2) for i in range(8)}
    ones = dict.fromkeys(range(8), 1.0)
    return _job(even, kind="continuous"), _job(ones, kind="continuous")


def test_intervals_reproduce_under_seed() -> None:
    first = compare_jobs(*_alternating_pair(), "m", confidence=0.9, draws=500)
    second = compare_jobs(*_alternating_pair(), "m", confidence=0.9, draws=500)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_bootstrap_needs_data() -> None:
    with pytest.raises(TaskContractError, match="at least one paired difference"):
        bootstrap_ci([])


def test_cli_compare_reports_paired_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from or_audit.cli import main
    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_dir = root / "docs" / "examples" / "tasks" / "video-nextstep"
    agent_dir = root / "docs" / "examples" / "agents" / "example-video-predictor"
    first = tmp_path / "a"
    second = tmp_path / "b"
    for out in (first, second):
        run_job(
            task=load_task(task_dir),
            task_dir=task_dir,
            agent=load_agent(agent_dir),
            agent_dir=agent_dir,
            out=out,
            n=3,
        )
    assert main(["compare", str(first), str(second), "--metric", "next_step_correct"]) == 0
    printed = capsys.readouterr().out
    assert "mean_diff:" in printed
    assert "CI:" in printed


def test_cli_compare_refuses_mismatched_tasks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from or_audit.cli import main

    assert main(["compare", str(tmp_path), str(tmp_path), "--metric", "m"]) == 1
    assert "COMPARE REFUSED" in capsys.readouterr().err


def test_task_mismatch_refuses() -> None:
    other = _job({0: True}).model_copy(update={"task_id": "other"})
    with pytest.raises(TaskContractError, match="identical task_id"):
        compare_jobs(_job({0: True}), other, "m")


def test_duplicate_seeds_refuse() -> None:
    job = _job({0: True})
    doubled = job.model_copy(update={"trials": job.trials + job.trials, "n": 2})
    with pytest.raises(TaskContractError, match="duplicate trial seeds"):
        compare_jobs(doubled, _job({0: True}), "m")


def test_all_dropped_refuses() -> None:
    with pytest.raises(TaskContractError, match="no assessable pairs"):
        compare_jobs(_job({0: None}), _job({0: None}), "m")


def test_zero_draws_refuses() -> None:
    with pytest.raises(TaskContractError, match="draws"):
        compare_jobs(_job({0: True}), _job({0: False}), "m", draws=0)
