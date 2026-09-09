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
    with pytest.raises(TaskContractError, match=r"missing config\.json"):
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


def test_kill_recovery_reuses_completed_trials(tmp_path: Path) -> None:
    out = tmp_path / "job"
    whole = _run(out, 2)
    # Simulate a kill after trial I/O but before result.json: drop only it.
    (out / "result.json").unlink()
    resumed = _run(out, 2, resume=True)
    assert [trial.seed for trial in resumed.trials] == [0, 1]
    assert resumed.head == whole.head


def test_resume_across_backend_change_refuses(tmp_path: Path) -> None:
    from or_audit.eval.loader import load_task
    from or_audit.eval.runner import builtin_random_agent, run_job
    from tests.test_eval_run import LUMEN_TASK, FakeLumenEnv

    class OtherBackend(FakeLumenEnv):
        def engine_provenance(self) -> dict[str, str]:
            return {**super().engine_provenance(), "backend": "real"}

    out = tmp_path / "job"
    run_job(
        task=load_task(LUMEN_TASK),
        task_dir=LUMEN_TASK,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=1,
        gym_factory=lambda task: FakeLumenEnv(),
    )
    with pytest.raises(TaskContractError, match="world engine changed"):
        run_job(
            task=load_task(LUMEN_TASK),
            task_dir=LUMEN_TASK,
            agent=builtin_random_agent(),
            agent_dir=None,
            out=out,
            n=2,
            gym_factory=lambda task: OtherBackend(),
            resume=True,
        )


def test_resume_with_foreign_seed_refuses(tmp_path: Path) -> None:
    import shutil

    out = tmp_path / "job"
    _run(out, 2)
    (out / "result.json").unlink()
    foreign = out / "trial-video-nextstep-99"
    shutil.copytree(out / "trial-video-nextstep-0", foreign)
    with pytest.raises(TaskContractError, match="outside schedule"):
        _run(out, 2, resume=True)


def test_resume_with_corrupt_provenance_refuses(tmp_path: Path) -> None:
    out = tmp_path / "job"
    _run(out, 2)
    (out / "result.json").unlink()
    prov_path = out / "trial-video-nextstep-0" / "provenance.json"
    prov_path.write_text('{"backend": "not-a-valid-backend"}', encoding="utf-8")
    with pytest.raises(TaskContractError, match=r"invalid provenance\.json"):
        _run(out, 2, resume=True)


def test_resume_with_conflicting_provenance_refuses(tmp_path: Path) -> None:
    import json

    out = tmp_path / "job"
    _run(out, 2)
    (out / "result.json").unlink()
    prov0 = out / "trial-video-nextstep-0" / "provenance.json"
    prov1 = out / "trial-video-nextstep-1" / "provenance.json"
    data0 = {
        "engine": "dummy",
        "backend": "synthetic-stub",
        "backend_version": "1.0",
        "world_pin": "",
        "adapter_id": "",
        "adapter_digest": "",
        "metrics_only": False,
    }
    data1 = {**data0, "backend": "real"}
    prov0.write_text(json.dumps(data0), encoding="utf-8")
    prov1.write_text(json.dumps(data1), encoding="utf-8")
    with pytest.raises(TaskContractError, match="conflicting provenance"):
        _run(out, 2, resume=True)


def test_resume_with_mixed_provenance_refuses(tmp_path: Path) -> None:
    from or_audit.eval.job import read_partial_trials

    out = tmp_path / "job"
    _run(out, 2)
    (out / "result.json").unlink()
    prov0 = out / "trial-video-nextstep-0" / "provenance.json"
    prov1 = out / "trial-video-nextstep-1" / "provenance.json"
    assert prov0.is_file(), "trial 0 must carry provenance"
    assert prov1.is_file(), "trial 1 must carry provenance"
    prov1.unlink()
    assert not prov1.is_file()
    with pytest.raises(TaskContractError, match=r"mixed provenance"):
        read_partial_trials(out, "video-nextstep")
    with pytest.raises(TaskContractError, match="mixed provenance"):
        _run(out, 2, resume=True)


def test_kill_recovery_pre_write_drift_refuses(tmp_path: Path) -> None:
    from or_audit.eval.loader import load_task
    from or_audit.eval.runner import builtin_random_agent, run_job
    from tests.test_eval_run import LUMEN_TASK, FakeLumenEnv

    class OtherBackend(FakeLumenEnv):
        def engine_provenance(self) -> dict[str, str]:
            return {**super().engine_provenance(), "backend": "real"}

    out = tmp_path / "job"
    run_job(
        task=load_task(LUMEN_TASK),
        task_dir=LUMEN_TASK,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=1,
        gym_factory=lambda task: FakeLumenEnv(),
    )
    (out / "result.json").unlink()
    trial1_dir = out / "trial-lumen-nav-safe-1"
    assert not trial1_dir.exists()
    with pytest.raises(TaskContractError, match="world engine changed"):
        run_job(
            task=load_task(LUMEN_TASK),
            task_dir=LUMEN_TASK,
            agent=builtin_random_agent(),
            agent_dir=None,
            out=out,
            n=2,
            gym_factory=lambda task: OtherBackend(),
            resume=True,
        )
    assert not trial1_dir.exists(), "drifted trial was written before rejection"


def test_resume_with_corrupt_projection_refuses(tmp_path: Path) -> None:
    out = tmp_path / "job"
    _run(out, 2)
    (out / "result.json").unlink()
    proj = out / "trial-video-nextstep-0" / "projection.json"
    proj.write_text("not-valid-json{", encoding="utf-8")
    with pytest.raises(TaskContractError, match=r"invalid projection\.json"):
        _run(out, 2, resume=True)


def test_resume_with_non_object_projection_refuses(tmp_path: Path) -> None:
    out = tmp_path / "job"
    _run(out, 2)
    (out / "result.json").unlink()
    proj = out / "trial-video-nextstep-0" / "projection.json"
    proj.write_text('["not", "an", "object"]', encoding="utf-8")
    with pytest.raises(TaskContractError, match=r"projection\.json is not an object"):
        _run(out, 2, resume=True)
