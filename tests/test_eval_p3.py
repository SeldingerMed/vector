"""P3: job.toml cartesian runs, export-rl, trajectory reconstitution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from or_audit.cli import main
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.cartesian import (
    pair_dir_name,
    read_manifest,
    replay_cartesian,
    run_cartesian_job,
)
from or_audit.eval.enums import ProjectionId
from or_audit.eval.export_rl import export_rl
from or_audit.eval.job import JobResult, compute_head, read_job_result
from or_audit.eval.job_config import load_job_config, resolve_job
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.reconstitute import reconstitute_trial_vector
from or_audit.eval.runner import replay_job, run_job
from tests.test_eval_run import (
    ANGIO_TASK,
    CATH_SEG,
    ROOT,
    VIDEO_AGENT,
    VIDEO_TASK,
    _fake,
    _pinned_lumen,
)

EXAMPLE_JOB = ROOT / "docs" / "examples" / "jobs" / "lumen-nav-random"
RANDOM_AGENT = ROOT / "docs" / "examples" / "agents" / "seldingermed-random"


def _write_job(
    tmp_path: Path,
    task_dir: Path,
    *,
    n: int = 2,
    agents: list[str] | None = None,
    extra: str = "",
) -> Path:
    dest = tmp_path / "job-config"
    dest.mkdir()
    agent_list = agents if agents is not None else ["random"]
    agents_lit = ", ".join(json.dumps(a) for a in agent_list)
    (dest / "job.toml").write_text(
        "\n".join(
            [
                'format_version = "1"',
                'id = "lumen-nav-random"',
                f"n = {n}",
                f"tasks = [{json.dumps(str(task_dir))}]",
                f"agents = [{agents_lit}]",
                extra,
                "[projection]",
                'id = "gated_reach_v0"',
                'version = "0"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dest


def test_example_job_toml_loads() -> None:
    cfg = load_job_config(EXAMPLE_JOB)
    assert cfg.id == "lumen-nav-random"
    assert cfg.agents == ("random",)
    assert cfg.projection is not None
    assert cfg.projection.id is ProjectionId.GATED_REACH_V0
    resolved = resolve_job(EXAMPLE_JOB)
    assert resolved.task_paths[0].name == "lumen-nav-safe"
    assert resolved.agent_refs == ("random",)


def test_job_refuses_empty_agents(tmp_path: Path) -> None:
    dest = tmp_path / "empty"
    dest.mkdir()
    (dest / "job.toml").write_text(
        'format_version = "1"\nid = "empty-job"\ntasks = ["x"]\nagents = []\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="at least one agent"):
        load_job_config(dest)


def test_job_refuses_duplicate_tasks(tmp_path: Path) -> None:
    dest = tmp_path / "dup"
    dest.mkdir()
    (dest / "job.toml").write_text(
        'format_version = "1"\nid = "dup-job"\ntasks = ["a", "a"]\nagents = ["random"]\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="same task path twice"):
        load_job_config(dest)


def test_job_refuses_extra_keys(tmp_path: Path) -> None:
    dest = tmp_path / "extra"
    dest.mkdir()
    (dest / "job.toml").write_text(
        'format_version = "1"\nid = "extra-job"\n'
        'tasks = ["a"]\nagents = ["random"]\nreward = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="failed validation"):
        load_job_config(dest)


def test_stage_contract_validates_sequence_and_independence(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    stage = """
[stage]
name = "qualification"
evaluation_unit = "seeded simulator episode"
target_units = 2
independent_case_unit = "scenario-target seed"
independent_case_key = "$seed"
independent_cases = 2
scenarios = ["lumen-nav-safe"]
operator_contexts = ["autonomous"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    resolved = resolve_job(_write_job(tmp_path, task_dir, n=2, extra=stage))
    assert resolved.config.stage is not None
    assert resolved.config.stage.independent_cases == 2

    (tmp_path / "wrong-cases").mkdir()
    wrong_cases = stage.replace("independent_cases = 2", "independent_cases = 1")
    with pytest.raises(TaskContractError, match="identifies 2"):
        run_cartesian_job(
            resolve_job(_write_job(tmp_path / "wrong-cases", task_dir, n=2, extra=wrong_cases)),
            out=tmp_path / "wrong-cases-out",
            gym_factory=_fake,
        )

    (tmp_path / "bad").mkdir()
    bad = stage.replace('prerequisites = ["integration-smoke", "pilot"]', "prerequisites = []")
    with pytest.raises(TaskContractError, match="prerequisites must be"):
        load_job_config(_write_job(tmp_path / "bad", task_dir, n=2, extra=bad))


def test_stage_run_enforces_units_supported_axes_and_head_covers_outcome(
    tmp_path: Path,
) -> None:
    task_dir = _pinned_lumen(tmp_path)
    stage = """
[stage]
name = "qualification"
evaluation_unit = "seeded simulator episode"
target_units = 2
independent_case_unit = "scenario-target seed"
independent_case_key = "$seed"
independent_cases = 2
scenarios = ["lumen-nav-safe"]
operator_contexts = ["autonomous"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    resolved = resolve_job(_write_job(tmp_path, task_dir, n=2, extra=stage))
    out = tmp_path / "staged"
    manifest = run_cartesian_job(resolved, out=out, gym_factory=_fake)
    assert manifest.gate_outcome == "failed"
    assert manifest.stage is not None
    assert manifest.observed_units == 2
    assert manifest.stage.target_units == sum(pair.n for pair in manifest.pairs)
    assert read_manifest(out).head == manifest.head

    with pytest.raises(TaskContractError, match="schedules 3"):
        run_cartesian_job(resolved, out=tmp_path / "wrong-n", n=3, gym_factory=_fake)

    (tmp_path / "unsupported").mkdir()
    unsupported = stage.replace('scenarios = ["lumen-nav-safe"]', 'scenarios = ["invented-world"]')
    with pytest.raises(TaskContractError, match="unsupported scenarios"):
        run_cartesian_job(
            resolve_job(_write_job(tmp_path / "unsupported", task_dir, n=2, extra=unsupported)),
            out=tmp_path / "unsupported-out",
            gym_factory=_fake,
        )

    (tmp_path / "steps").mkdir()
    step_stage = stage.replace(
        'evaluation_unit = "seeded simulator episode"',
        'evaluation_unit = "observed simulator transition"\nunit_source = "trajectory-steps"',
    ).replace("target_units = 2", "target_units = 6")
    step_manifest = run_cartesian_job(
        resolve_job(_write_job(tmp_path / "steps", task_dir, n=2, extra=step_stage)),
        out=tmp_path / "steps-out",
        gym_factory=_fake,
    )
    assert step_manifest.observed_units == 6


def test_dataset_stage_recomputes_independent_cases_from_input_field(tmp_path: Path) -> None:
    job = tmp_path / "dataset-stage"
    job.mkdir()
    body = f"""format_version = "1"
id = "video-stage"
n = 3
tasks = [{json.dumps(str(VIDEO_TASK))}]
agents = [{json.dumps(str(VIDEO_AGENT))}]

[stage]
name = "qualification"
evaluation_unit = "scored clip"
target_units = 3
independent_case_unit = "held-out clip"
independent_case_key = "id"
independent_cases = 3
scenarios = ["video-nextstep"]
operator_contexts = ["offline"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    (job / "job.toml").write_text(body, encoding="utf-8")
    manifest = run_cartesian_job(resolve_job(job), out=tmp_path / "dataset-out")
    assert sum(pair.n for pair in manifest.pairs) == 3

    wrong = tmp_path / "dataset-stage-wrong"
    wrong.mkdir()
    (wrong / "job.toml").write_text(
        body.replace("independent_cases = 3", "independent_cases = 2"), encoding="utf-8"
    )
    with pytest.raises(TaskContractError, match="'id' identifies 3"):
        run_cartesian_job(resolve_job(wrong), out=tmp_path / "dataset-wrong-out")


def test_stage_supports_heterogeneous_per_task_trial_counts(tmp_path: Path) -> None:
    first = _pinned_lumen(tmp_path)
    second = tmp_path / "lumen-task-2"
    shutil.copytree(first, second)
    task_file = second / "task.toml"
    task_file.write_text(
        task_file.read_text(encoding="utf-8").replace(
            'id = "lumen-nav-safe"', 'id = "lumen-nav-safe-2"', 1
        ),
        encoding="utf-8",
    )
    job = tmp_path / "heterogeneous-stage"
    job.mkdir()
    first_ref, second_ref = str(first), str(second)
    (job / "job.toml").write_text(
        f"""format_version = "1"
id = "heterogeneous-stage"
tasks = [{json.dumps(first_ref)}, {json.dumps(second_ref)}]
agents = ["random"]

[task_trials]
{json.dumps(first_ref)} = 1
{json.dumps(second_ref)} = 2

[stage]
name = "qualification"
evaluation_unit = "seeded simulator episode"
target_units = 3
independent_case_unit = "scenario seed"
independent_case_key = "$seed"
independent_cases = 3
scenarios = ["lumen-nav-safe", "lumen-nav-safe-2"]
operator_contexts = ["autonomous"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
""",
        encoding="utf-8",
    )

    manifest = run_cartesian_job(
        resolve_job(job), out=tmp_path / "heterogeneous-out", gym_factory=_fake
    )
    assert [(pair.task_id, pair.n) for pair in manifest.pairs] == [
        ("lumen-nav-safe", 1),
        ("lumen-nav-safe-2", 2),
    ]
    assert manifest.observed_units == 3


def test_stage_counts_task_reported_scored_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Predictor:
        def predict(self, item: dict[str, object]) -> dict[str, object]:
            return {
                "id": item["id"],
                "release_audit_passed": True,
                "dias_prediction_count": 345,
                "cathaction_prediction_count": 5225,
                "sam_vit_b_mean_dice": 0.8,
                "sam_vit_l_mean_dice": 0.7,
                "medsam_vit_b_mean_dice": 0.6,
            }

    monkeypatch.setattr("or_audit.eval.runner.load_predictor_runtime", lambda *args: Predictor())
    job = tmp_path / "metric-stage"
    job.mkdir()
    (job / "job.toml").write_text(
        f"""format_version = "1"
id = "angiostress-stage"
n = 1
tasks = [{json.dumps(str(ANGIO_TASK))}]
agents = [{json.dumps(str(CATH_SEG))}]

[stage]
name = "pilot"
evaluation_unit = "scored DIAS prediction"
unit_source = "metric:dias_prediction_count"
target_units = 345
independent_case_unit = "benchmark release"
independent_case_key = "id"
independent_cases = 1
scenarios = ["angiostress-dias"]
operator_contexts = ["offline"]
stop_conditions = ["stop on release audit failure"]
prerequisites = ["integration-smoke"]
""",
        encoding="utf-8",
    )
    manifest = run_cartesian_job(resolve_job(job), out=tmp_path / "metric-stage-out")
    assert manifest.observed_units == 345


def test_job_missing_agent_path(tmp_path: Path) -> None:
    dest = tmp_path / "missing-agent"
    dest.mkdir()
    (dest / "job.toml").write_text(
        'format_version = "1"\nid = "missing-agent"\n'
        f"tasks = [{json.dumps(str(VIDEO_TASK))}]\n"
        'agents = ["nope/not-an-agent"]\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="does not exist"):
        resolve_job(dest)


def test_job_projection_must_match_task_declaration(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    verifier = task_dir / "verifier.toml"
    verifier.write_text(
        verifier.read_text(encoding="utf-8").replace('version = "0"', 'version = "2"'),
        encoding="utf-8",
    )
    job_dir = _write_job(tmp_path, task_dir)

    with pytest.raises(TaskContractError, match="projection does not match"):
        run_cartesian_job(
            resolve_job(job_dir),
            out=tmp_path / "mismatch",
            gym_factory=_fake,
        )


def test_cartesian_random_gym_export_unsafe_is_zero(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    job_dir = _write_job(tmp_path, task_dir, n=2)
    out = tmp_path / "jobs" / "lumen-nav-random"
    manifest = run_cartesian_job(resolve_job(job_dir), out=out, gym_factory=_fake)
    assert len(manifest.pairs) == 1
    pair = out / manifest.pairs[0].dir
    assert pair.name == pair_dir_name("lumen-nav-safe", "seldingermed/random")
    result = read_job_result(pair)
    raw1 = result.trials[1].vector.metric("raw_success")
    safe1 = result.trials[1].vector.metric("safe_success")
    assert raw1 is not None
    assert raw1.value is True
    assert safe1 is not None
    assert safe1.value is False

    rollouts = tmp_path / "rollouts.jsonl"
    n = export_rl(out, projection_id=ProjectionId.GATED_REACH_V0, out=rollouts)
    assert n == 2
    rows = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    by_seed = {row["seed"]: row for row in rows}
    assert by_seed[0]["projection"] == 1.0
    assert by_seed[1]["projection"] == 0.0
    assert by_seed[1]["episode_id"] == "lumen-nav-safe-1"
    assert by_seed[1]["projection_id"] == "gated_reach_v0"
    assert by_seed[0]["task_id"] == "lumen-nav-safe"
    assert by_seed[0]["projection_rule"]["source_metric"] == "raw_success"
    assert by_seed[0]["projection_rule"]["gate_failure"] == "zero"
    assert by_seed[0]["projection_digest"]

    trial0 = pair / "trial-lumen-nav-safe-0"
    recon = reconstitute_trial_vector(
        trial0,
        task=load_task(task_dir),
        task_dir=task_dir,
        agent_identity=result.agent_identity,
        seed=0,
        safety_max_pen=0.3,
    )
    assert recon == result.trials[0].vector


def test_cartesian_replay_matches_manifest(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    out = tmp_path / "jobs" / "lumen-nav-random"
    first = run_cartesian_job(
        resolve_job(_write_job(tmp_path, task_dir, n=2)),
        out=out,
        gym_factory=_fake,
    )
    replayed = replay_cartesian(
        out,
        load_task=load_task,
        load_agent=load_agent,
        gym_factory=_fake,
    )
    assert replayed.head == first.head


def test_pair_dir_collision_is_refused(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    job_dir = _write_job(
        tmp_path,
        task_dir,
        agents=["random", str(RANDOM_AGENT)],
    )
    with pytest.raises(TaskContractError, match="collides"):
        run_cartesian_job(resolve_job(job_dir), out=tmp_path / "out", gym_factory=_fake)


def test_cartesian_n_zero_is_refused(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    job_dir = _write_job(tmp_path, task_dir, n=2)
    with pytest.raises(TaskContractError, match="n must be >= 1"):
        run_cartesian_job(
            resolve_job(job_dir),
            out=tmp_path / "out",
            n=0,
            gym_factory=_fake,
        )


def test_export_rl_refuses_video_gated_reach(tmp_path: Path) -> None:
    out = tmp_path / "video-job"
    run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=out,
    )
    # A fresh video run has no runtime reporter, so its provenance is "unknown"
    # and export refuses it as unattested. Attest a real (physical) backend and
    # re-stamp the head so the projection-level refusal is what we exercise.
    result_path = out / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    we = payload["world_engine"]
    payload["world_engine"] = {
        "engine": we.get("engine", ""),
        "backend": "real",
        "backend_version": "",
        "world_pin": we.get("world_pin", ""),
    }
    payload["head"] = compute_head(JobResult.model_validate(payload))
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="diverged"):
        export_rl(
            out,
            projection_id=ProjectionId.GATED_REACH_V0,
            out=tmp_path / "nope.jsonl",
        )


def test_export_rl_refuses_stored_projection_disagreement(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    out = tmp_path / "job"
    run_cartesian_job(
        resolve_job(_write_job(tmp_path, task_dir, n=2)),
        out=out,
        gym_factory=_fake,
    )
    pair = out / read_manifest(out).pairs[0].dir
    result_path = pair / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["trials"][1]["projection"] = 1.0
    # Re-stamp a consistent head so the tampered projection is what the job
    # actually claims (head-verification passes); export must still refuse
    # because the stored projection disagrees with the recomputed score.
    payload["head"] = compute_head(JobResult.model_validate(payload))
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ScoreContractError, match="homemade float"):
        export_rl(pair, projection_id=ProjectionId.GATED_REACH_V0, out=tmp_path / "x.jsonl")


def test_export_rl_missing_job(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="neither a job"):
        export_rl(tmp_path, projection_id=ProjectionId.GATED_REACH_V0, out=tmp_path / "x.jsonl")


def test_trajectory_mismatch_fails_replay(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    out = tmp_path / "jobs" / "lumen-nav-random"
    run_cartesian_job(
        resolve_job(_write_job(tmp_path, task_dir, n=2)),
        out=out,
        gym_factory=_fake,
    )
    pair = out / read_manifest(out).pairs[0].dir
    traj_path = pair / "trial-lumen-nav-safe-0" / "trajectory.json"
    steps = json.loads(traj_path.read_text(encoding="utf-8"))
    steps[-1]["info"]["max_pen"] = 0.99
    steps[-1]["info"]["unsafe"] = True
    steps[-1]["info"]["safe_success"] = False
    traj_path.write_text(json.dumps(steps) + "\n", encoding="utf-8")
    with pytest.raises(ScoreContractError, match="reconstitutes a different vector"):
        replay_job(
            pair,
            load_task=load_task,
            load_agent=load_agent,
            gym_factory=_fake,
        )


def test_reconstitute_refuses_unknown_trajectory(tmp_path: Path) -> None:
    trial = tmp_path / "trial-x-0"
    trial.mkdir()
    (trial / "trajectory.json").write_text('[{"foo": 1}]\n', encoding="utf-8")
    with pytest.raises(TaskContractError, match="is not gym-policy"):
        reconstitute_trial_vector(
            trial,
            task=load_task(VIDEO_TASK),
            task_dir=VIDEO_TASK,
            agent_identity="example/video-predictor@0+none",
            seed=0,
        )


def test_cli_run_job_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("or_audit.eval.sim.gym_bridge.make_gym", _fake)
    task_dir = _pinned_lumen(tmp_path)
    job_dir = _write_job(tmp_path, task_dir, n=2)
    out = tmp_path / "cli-job"
    assert main(["run", "-c", str(job_dir), "--out", str(out)]) == 0
    captured = capsys.readouterr().out
    assert "ran: lumen-nav-random pairs=1" in captured
    manifest = read_manifest(out)
    rollouts = tmp_path / "rollouts.jsonl"
    assert (
        main(
            [
                "export-rl",
                str(out),
                "--projection",
                "gated_reach_v0",
                "--out",
                str(rollouts),
            ]
        )
        == 0
    )
    assert "exported: 2 episodes" in capsys.readouterr().out
    rows = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["projection"] == 0.0
    assert main(["replay", str(out), "--expect-head", manifest.head]) == 0


def test_cli_run_job_refuses_agent_flag(tmp_path: Path) -> None:
    assert (
        main(
            [
                "run",
                "-c",
                str(EXAMPLE_JOB),
                "-a",
                "random",
                "--out",
                str(tmp_path / "x"),
            ]
        )
        == 2
    )


def test_cli_run_job_and_task_together(tmp_path: Path) -> None:
    assert (
        main(
            [
                "run",
                "-c",
                str(EXAMPLE_JOB),
                "-t",
                str(VIDEO_TASK),
                "--out",
                str(tmp_path / "x"),
            ]
        )
        == 2
    )


def test_cli_run_task_requires_agent(tmp_path: Path) -> None:
    assert main(["run", "-t", str(VIDEO_TASK), "--out", str(tmp_path / "x")]) == 2


def test_job_refuses_duplicate_agents(tmp_path: Path) -> None:
    dest = tmp_path / "dup-agents"
    dest.mkdir()
    (dest / "job.toml").write_text(
        'format_version = "1"\nid = "dup-agents"\ntasks = ["a"]\nagents = ["random", "random"]\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="same agent twice"):
        load_job_config(dest)


def test_cli_export_rl_refuses_video_gated_reach(tmp_path: Path) -> None:
    out = tmp_path / "video-job"
    assert main(["run", "-t", str(VIDEO_TASK), "-a", str(VIDEO_AGENT), "--out", str(out)]) == 0
    assert (
        main(
            [
                "export-rl",
                str(out),
                "--projection",
                "gated_reach_v0",
                "--out",
                str(tmp_path / "nope.jsonl"),
            ]
        )
        == 1
    )


def test_cli_export_rl_refuses_undeclared_projection(tmp_path: Path) -> None:
    assert (
        main(
            [
                "export-rl",
                str(tmp_path),
                "--projection",
                "mean_reward",
                "--out",
                str(tmp_path / "x.jsonl"),
            ]
        )
        == 1
    )


def test_cli_n_overrides_job_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("or_audit.eval.sim.gym_bridge.make_gym", _fake)
    task_dir = _pinned_lumen(tmp_path)
    job_dir = _write_job(tmp_path, task_dir, n=30)
    out = tmp_path / "cli-n"
    assert main(["run", "-c", str(job_dir), "-n", "2", "--out", str(out)]) == 0
    pair = out / read_manifest(out).pairs[0].dir
    assert read_job_result(pair).n == 2
