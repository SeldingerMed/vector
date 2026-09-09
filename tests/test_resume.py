"""Episode-boundary resume: interrupted jobs keep completed trials."""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.job import JobResult, read_job_result
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import run_job

ROOT = Path(__file__).resolve().parents[1]
VIDEO_TASK = ROOT / "docs/examples/tasks/video-nextstep"
VIDEO_AGENT = ROOT / "docs/examples/agents/example-video-predictor"


def _run(out: Path, n: int, *, resume: bool = False) -> JobResult:
    return run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=out,
        n=n,
        resume=resume,
    )


def test_resume_completes_missing_seeds_only(tmp_path: Path) -> None:
    out = tmp_path / "job"
    first = _run(out, 1)
    assert [trial.seed for trial in first.trials] == [0]
    second = _run(out, 3, resume=True)
    assert [trial.seed for trial in second.trials] == [0, 1, 2]
    assert second.trials[0].vector == first.trials[0].vector
    assert read_job_result(out).head == second.head


def test_resume_without_prior_result_refuses(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match=r"missing result\.json"):
        _run(tmp_path / "job", 1, resume=True)


def test_resume_with_shrunk_n_refuses(tmp_path: Path) -> None:
    out = tmp_path / "job"
    _run(out, 2)
    with pytest.raises(TaskContractError, match="exceeds requested n=1"):
        _run(out, 1, resume=True)


def test_resume_with_changed_task_refuses(tmp_path: Path) -> None:
    import shutil

    out = tmp_path / "job"
    _run(out, 1)
    task_copy = tmp_path / "task"
    shutil.copytree(VIDEO_TASK, task_copy, ignore=shutil.ignore_patterns("__pycache__"))
    (task_copy / "instruction.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="task package changed"):
        run_job(
            task=load_task(task_copy),
            task_dir=task_copy,
            agent=load_agent(VIDEO_AGENT),
            agent_dir=VIDEO_AGENT,
            out=out,
            n=1,
            resume=True,
        )


def test_cli_resume_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from or_audit.cli import main

    out = tmp_path / "job"
    base = ["run", "-t", str(VIDEO_TASK), "-a", str(VIDEO_AGENT), "--out", str(out)]
    assert main([*base, "-n", "1"]) == 0
    first = capsys.readouterr().out
    assert main([*base, "-n", "1", "--resume"]) == 0
    second = capsys.readouterr().out
    assert "n=1" in second
    assert len(read_job_result(out).trials) == 1
    assert first.split("head ")[1] == second.split("head ")[1]
