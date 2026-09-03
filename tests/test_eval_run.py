"""P1 gym-policy and P2 video-predict runners."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import pytest

from or_audit.cli import main
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import PerturbationSpec
from or_audit.eval.gym_world import (
    assert_perturbations_applied,
    run_gym_episode,
    split_perturbations,
)
from or_audit.eval.job import TrialRecord, assemble_job_result, read_job_result
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import builtin_random_agent, replay_job, run_job
from or_audit.eval.sim.base import BACKEND_REAL
from or_audit.eval.vector import TrialVector

ROOT = Path(__file__).resolve().parents[1]
LUMEN_TASK = ROOT / "docs" / "examples" / "tasks" / "lumen-nav-safe"
VIDEO_TASK = ROOT / "docs" / "examples" / "tasks" / "video-nextstep"
ANGIO_TASK = ROOT / "docs" / "examples" / "tasks" / "angiostress-dias"
VIDEO_AGENT = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"
CATH_SEG = ROOT / "docs" / "examples" / "agents" / "seldingermed-cath-seg"
CATH_POLICY = ROOT / "docs" / "examples" / "agents" / "seldingermed-lumen-linear"


class _Box:
    low = np.array([-1.0, -1.0], dtype=np.float64)
    high = np.array([1.0, 1.0], dtype=np.float64)


class FakeLumenEnv:
    """Deterministic stand-in so CI does not import Newton."""

    safety_max_pen = 0.3
    action_space = _Box()

    def engine_provenance(self) -> dict[str, str]:
        # This fixture stands in for a real Lumen gym backend, so it attests
        # BACKEND_REAL with the full reporter contract.
        return {
            "engine": "lumen-nav-safe",
            "backend": BACKEND_REAL,
            "backend_version": "",
            "world_pin": "test-lumen-pin",
        }

    def __init__(self) -> None:
        self._seed = 0
        self._step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        self._seed = 0 if seed is None else seed
        self._step = 0
        return np.zeros(5, dtype=np.float32), {"or_audit": options["or_audit"]} if options else {}

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self._step += 1
        done = self._step >= 3
        if self._seed == 1:
            max_pen = 0.9
            success = done
        elif self._seed == 0:
            max_pen = 0.01
            success = done
        else:
            max_pen = 0.0
            success = False
        unsafe = max_pen > self.safety_max_pen
        info: dict[str, object] = {
            "success": success,
            "safe_success": bool(success and not unsafe),
            "unsafe": unsafe,
            "max_pen": max_pen,
            "diverged": False,
        }
        if self._seed == 0 and self._step == 1:
            info["or_audit"] = {"applied_perturbations": ["transient-wall-force"]}
        return np.zeros(5, dtype=np.float32), 0.0, done, False, info


def _fake(_task: object) -> FakeLumenEnv:
    return FakeLumenEnv()


def test_harness_applies_and_records_portable_interface_faults() -> None:
    class RecordingEnv:
        def __init__(self) -> None:
            self.actions: list[np.ndarray] = []
            self.step_index = 0

        def reset(
            self, *, seed: int | None = None, options: object = None
        ) -> tuple[np.ndarray, dict[str, object]]:
            del seed, options
            self.step_index = 0
            return np.array([2.0, 3.0]), {}

        def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            self.actions.append(np.asarray(action))
            self.step_index += 1
            return (
                np.array([2.0 + self.step_index, 3.0 + self.step_index]),
                0.0,
                self.step_index == 3,
                False,
                {},
            )

    events = (
        PerturbationSpec(id="drop", kind="harness-observation-zero", at_step=0),
        PerturbationSpec(
            id="noise",
            kind="harness-observation-gaussian-noise",
            at_step=1,
            parameters={"std": 0.1},
        ),
        PerturbationSpec(id="hold", kind="harness-action-hold", at_step=2),
    )

    def roll() -> tuple[RecordingEnv, list[np.ndarray], tuple[dict[str, Any], ...]]:
        env = RecordingEnv()
        observations: list[np.ndarray] = []

        def act(_env: object, observation: object, step: int) -> np.ndarray:
            observations.append(np.asarray(observation))
            return np.full(2, step + 1, dtype=float)

        _, steps = run_gym_episode(
            env,
            seed=7,
            action_fn=act,
            harness_perturbations=events,
            max_steps=3,
        )
        return env, observations, steps

    env, observations, steps = roll()
    _, repeated_observations, _ = roll()
    np.testing.assert_array_equal(observations[0], np.zeros(2))
    np.testing.assert_array_equal(observations[1], repeated_observations[1])
    assert not np.array_equal(observations[1], np.array([3.0, 4.0]))
    np.testing.assert_array_equal(env.actions[2], np.array([2.0, 2.0]))
    assert steps[0]["info"]["or_audit"]["applied_perturbations"][0]["id"] == "drop"
    assert steps[2]["applied_action"] == [2.0, 2.0]
    with pytest.raises(TaskContractError, match="episode ended before"):
        assert_perturbations_applied(
            steps,
            (PerturbationSpec(id="too-late", kind="harness-action-hold", at_step=3),),
        )
    with pytest.raises(TaskContractError, match="unsupported harness perturbation"):
        split_perturbations((PerturbationSpec(id="bad", kind="harness-invented"),))


def test_legacy_gym_box_restores_array_actions_from_json_plugins() -> None:
    class JsonBox:
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        dtype = np.dtype(np.float32)

        def contains(self, action: object) -> bool:
            return isinstance(action, np.ndarray) and action.shape == (2,)

        def from_jsonable(self, samples: list[object]) -> list[object]:
            return samples

    class ArrayActionEnv:
        action_space = JsonBox()

        def reset(
            self, *, seed: int | None = None, options: object = None
        ) -> tuple[int, dict[str, object]]:
            del seed, options
            return 0, {}

        def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
            assert isinstance(action, np.ndarray)
            np.testing.assert_array_equal(action, np.array([0.25, -0.5], dtype=np.float32))
            return 0, 0.0, True, False, {}

    _, steps = run_gym_episode(
        ArrayActionEnv(),
        seed=0,
        action_fn=lambda _env, _obs, _step: [0.25, -0.5],
        max_steps=1,
    )
    assert steps[0]["action"] == [0.25, -0.5]


def test_harness_observation_faults_transform_gym_dict_numeric_leaves() -> None:
    class DictObservationEnv:
        def __init__(self) -> None:
            self.step_index = 0

        def reset(
            self, *, seed: int | None = None, options: object = None
        ) -> tuple[object, dict[str, object]]:
            del seed, options
            self.step_index = 0
            return {"observation": np.array([2.0, 3.0]), "goal": np.array([4.0])}, {}

        def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, object]]:
            del action
            self.step_index += 1
            return (
                {
                    "observation": np.array([5.0]),
                    "goal": np.array([6.0]),
                },
                0.0,
                self.step_index == 2,
                False,
                {},
            )

    seen: list[Any] = []
    events = (
        PerturbationSpec(id="drop", kind="harness-observation-zero", at_step=0),
        PerturbationSpec(
            id="noise",
            kind="harness-observation-gaussian-noise",
            at_step=1,
            parameters={"std": 0.1},
        ),
    )

    def act(_env: object, observation: object, _step: int) -> np.ndarray:
        seen.append(observation)
        return np.zeros(1)

    run_gym_episode(
        DictObservationEnv(),
        seed=7,
        action_fn=act,
        harness_perturbations=events,
    )
    np.testing.assert_array_equal(seen[0]["observation"], np.zeros(2))
    np.testing.assert_array_equal(seen[0]["goal"], np.zeros(1))
    assert not np.array_equal(seen[1]["observation"], np.array([5.0]))
    assert not np.array_equal(seen[1]["goal"], np.array([6.0]))


def _pinned_lumen(tmp_path: Path) -> Path:
    dest = tmp_path / "lumen-task"
    shutil.copytree(LUMEN_TASK, dest)
    text = dest.joinpath("task.toml").read_text(encoding="utf-8")
    dest.joinpath("task.toml").write_text(
        text.replace('world_pin = ""', 'world_pin = "test-lumen-pin"'),
        encoding="utf-8",
    )
    return dest


def test_random_gym_job_emits_raw_and_safe(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    out = tmp_path / "job"
    result = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=30,
        gym_factory=_fake,
    )
    assert result.n == 30
    assert result.headline == "safe_success"
    assert result.unique_trajectories == 30
    assert result.duplicate_trajectories == 0
    assert result.gate_outcome == "failed"
    seed0 = result.trials[0].vector
    safe0 = seed0.metric("safe_success")
    raw0 = seed0.metric("raw_success")
    assert raw0 is not None
    assert safe0 is not None
    assert safe0.value is True
    seed1 = result.trials[1].vector
    raw1 = seed1.metric("raw_success")
    safe1 = seed1.metric("safe_success")
    assert raw1 is not None
    assert raw1.value is True
    assert safe1 is not None
    assert safe1.value is False
    assert seed1.any_gate_failed
    written = read_job_result(out)
    assert written.head == result.head
    assert (out / "result.json").is_file()
    assert (out / "trial-lumen-nav-safe-0" / "trajectory.json").is_file()
    assert (out / "trial-lumen-nav-safe-0" / "projection.json").is_file()


def test_gym_replay_matches_head(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    out = tmp_path / "job"

    first = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=5,
        gym_factory=_fake,
    )
    replayed = replay_job(
        out,
        load_task=load_task,
        load_agent=load_agent,
        gym_factory=_fake,
    )
    assert replayed.head == first.head


def test_world_engine_provenance_typed_and_head_covered(tmp_path: Path) -> None:
    from or_audit.eval.job import WorldEngineProvenance, verify_head

    task_dir = _pinned_lumen(tmp_path)
    out = tmp_path / "job"
    run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=2,
        gym_factory=_fake,
    )
    written = read_job_result(out)
    # provenance is stored as the typed model, not a bare dict
    assert isinstance(written.world_engine, WorldEngineProvenance)
    # the FakeLumenEnv attests BACKEND_REAL (not synthetic, not unknown)
    assert written.world_engine.backend == BACKEND_REAL
    assert written.world_engine.engine == "lumen-nav-safe"
    # provenance is bound into the head, so it survives replay verification
    assert verify_head(written)
    # a handed-in dict is coerced into the typed model
    coerced = WorldEngineProvenance(**{"backend": "real", "engine": "lumen-nav-safe"})
    assert coerced.backend == "real"


def test_world_engine_provenance_strict_model() -> None:
    from or_audit.eval.job import WorldEngineProvenance

    # only the three declared backend states validate
    for state in ("real", "synthetic-stub", "unknown"):
        WorldEngineProvenance(engine="e", backend=state)
    # a typo/ad-hoc backend value is rejected rather than passed through
    with pytest.raises(pydantic.ValidationError):
        WorldEngineProvenance(engine="e", backend="gym")
    # unknown reporter fields are rejected (no silent extra keys)
    with pytest.raises(pydantic.ValidationError):
        WorldEngineProvenance.model_validate({"engine": "e", "backend": "real", "rogue": "x"})


def test_unpinned_gym_is_not_runnable(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    task_file = task_dir / "task.toml"
    pin = load_task(task_dir).environment.world_pin
    task_file.write_text(
        task_file.read_text(encoding="utf-8").replace(
            f'world_pin = "{pin}"',
            'world_pin = ""',
        ),
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="world_pin"):
        run_job(
            task=load_task(task_dir),
            task_dir=task_dir,
            agent=builtin_random_agent(),
            agent_dir=None,
            out=tmp_path / "job",
            n=1,
            gym_factory=_fake,
        )


def test_policy_entrypoint_runs(tmp_path: Path) -> None:
    result = run_job(
        task=load_task(LUMEN_TASK),
        task_dir=LUMEN_TASK,
        agent=load_agent(CATH_POLICY),
        agent_dir=CATH_POLICY,
        out=tmp_path / "job",
        n=1,
        gym_factory=_fake,
    )
    assert result.n == 1
    assert result.trials[0].trajectory[0]["action"] == [0.0, 0.0]


def test_video_predictor_cannot_run_on_gym(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    with pytest.raises(TaskContractError, match="video-predict"):
        run_job(
            task=load_task(task_dir),
            task_dir=task_dir,
            agent=load_agent(VIDEO_AGENT),
            agent_dir=VIDEO_AGENT,
            out=tmp_path / "job",
            n=1,
            gym_factory=_fake,
        )


def test_publish_omitting_safe_success_metric_is_refused() -> None:
    task = load_task(LUMEN_TASK)
    raw_only = TrialVector(
        task_id=task.id,
        task_version=task.task_version,
        agent_identity="seldingermed/random@0+none",
        seed=0,
        gates=(),
        metrics=(
            {"id": "raw_success", "value": True, "headline": True},
            {"id": "diverged", "value": False},
        ),
    )
    with pytest.raises(TaskContractError, match="safe_success"):
        assemble_job_result(
            task=task,
            agent=builtin_random_agent(),
            trials=(TrialRecord(seed=0, vector=raw_only),),
            task_digest="task-digest",
            agent_digest="agent-digest",
        )


def test_video_nextstep_run(tmp_path: Path) -> None:
    out = tmp_path / "video-job"
    result = run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=out,
    )
    assert result.n == 3
    assert result.headline == "next_step_correct"
    assert result.trials[0].vector.headline.value is True
    assert result.trials[1].vector.any_gate_failed
    assert result.trials[2].vector.headline.value is None
    assert result.claim_footer == ""


def test_dataset_run_refuses_to_claim_more_trials_than_inputs(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="has 3 input items; cannot execute 10,000"):
        run_job(
            task=load_task(VIDEO_TASK),
            task_dir=VIDEO_TASK,
            agent=load_agent(VIDEO_AGENT),
            agent_dir=VIDEO_AGENT,
            out=tmp_path / "false-scale",
            n=10_000,
        )
    assert not (tmp_path / "false-scale").exists()


def test_angiostress_requires_claim_footer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Predictor:
        def predict(self, item: dict[str, object]) -> dict[str, object]:
            assert "label" not in item
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
    out = tmp_path / "angio-job"
    result = run_job(
        task=load_task(ANGIO_TASK),
        task_dir=ANGIO_TASK,
        agent=load_agent(CATH_SEG),
        agent_dir=CATH_SEG,
        out=out,
    )
    assert result.n == 1
    assert result.claim_footer
    assert "benchmark artifact" in result.claim_footer
    assert result.trials[0].vector.headline.value is True
    assert result.trials[0].vector.metric("dias_prediction_count") is not None


def test_angiostress_without_footer_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Predictor:
        def predict(self, item: dict[str, object]) -> dict[str, object]:
            return {"id": item["id"], "release_audit_passed": True}

    monkeypatch.setattr("or_audit.eval.runner.load_predictor_runtime", lambda *args: Predictor())
    task = load_task(ANGIO_TASK)
    result = run_job(
        task=task,
        task_dir=ANGIO_TASK,
        agent=load_agent(CATH_SEG),
        agent_dir=CATH_SEG,
        out=tmp_path / "angio-scratch",
        n=1,
    )
    with pytest.raises(TaskContractError, match="claim footer"):
        assemble_job_result(
            task=task,
            agent=load_agent(CATH_SEG),
            trials=(TrialRecord(seed=0, vector=result.trials[0].vector),),
            task_digest="task-digest",
            agent_digest="agent-digest",
            claim_footer="",
        )


def test_cli_run_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli-video"
    assert (
        main(
            [
                "run",
                "-t",
                str(VIDEO_TASK),
                "-a",
                str(VIDEO_AGENT),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "ran: video-nextstep" in capsys.readouterr().out
    head = read_job_result(out).head
    assert main(["replay", str(out), "--expect-head", head]) == 0


def test_cli_run_random_gym(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = _pinned_lumen(tmp_path)
    monkeypatch.setattr("or_audit.eval.sim.gym_bridge.make_gym", _fake)
    out = tmp_path / "cli-gym"
    assert main(["run", "-t", str(task_dir), "-a", "random", "-n", "30", "--out", str(out)]) == 0
    result = read_job_result(out)
    assert result.n == 30
    assert result.trials[0].vector.metric("safe_success") is not None


def test_cli_refuses_video_model_on_gym(tmp_path: Path) -> None:
    task_dir = _pinned_lumen(tmp_path)
    assert (
        main(
            [
                "run",
                "-t",
                str(task_dir),
                "-a",
                str(VIDEO_AGENT),
                "--out",
                str(tmp_path / "nope"),
            ]
        )
        == 1
    )


def test_cli_run_requires_task_or_dataset() -> None:
    assert main(["run", "-a", "random", "--out", "/tmp/x"]) == 2


def test_cli_run_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ds"
    assert (
        main(
            [
                "run",
                "-d",
                str(ROOT / "docs" / "examples" / "datasets" / "video-nextstep-v0"),
                "-a",
                str(VIDEO_AGENT),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "ran: video-nextstep" in capsys.readouterr().out
    assert (out / "video-nextstep" / "result.json").is_file()


def test_cli_replay_expect_head_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "cli-video"
    assert main(["run", "-t", str(VIDEO_TASK), "-a", str(VIDEO_AGENT), "--out", str(out)]) == 0
    assert main(["replay", str(out), "--expect-head", "deadbeef"]) == 1


def test_make_gym_without_lumen_explains_install() -> None:
    import importlib.util

    if importlib.util.find_spec("lumen") is not None:
        pytest.skip("lumen is installed in this environment")
    from or_audit.eval.gym_world import make_gym

    with pytest.raises(TaskContractError, match="Lumen"):
        make_gym(load_task(LUMEN_TASK))
