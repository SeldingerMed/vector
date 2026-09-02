from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.bind import assert_bind
from or_audit.eval.contracts import RuntimeDescriptor, RuntimeKind
from or_audit.eval.leaderboard import leaderboard_data
from or_audit.eval.loader import load_agent, load_task, load_taskset
from or_audit.eval.plugins import load_predictor_runtime, load_verifier_runtime
from or_audit.eval.runner import builtin_random_agent, replay_job, run_job
from or_audit.eval.scorecard import render_markdown, scorecard_data
from or_audit.eval.trace import ProceduralTrace

ROOT = Path(__file__).resolve().parents[1]
COUNTERFACTUAL_TASK = ROOT / "docs/examples/tasks/counterfactual-recovery"
COUNTERFACTUAL_AGENT = ROOT / "docs/examples/agents/example-counterfactual-world-model"
COUNTERFACTUAL_TASKSET = ROOT / "docs/examples/tasksets/counterfactual-recovery-v1"
VIDEO_TASK = ROOT / "docs/examples/tasks/video-nextstep"
VIDEO_AGENT = ROOT / "docs/examples/agents/example-video-predictor"
LUMEN_TASK = ROOT / "docs/examples/tasks/lumen-nav-safe"


class RecoveryLumenEnv:
    safety_max_pen = 0.3

    class ActionSpace:
        low = (-1.0, -1.0)
        high = (1.0, 1.0)

    action_space = ActionSpace()

    def __init__(self) -> None:
        self.step_index = 0
        self.reset_options: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        del seed
        self.step_index = 0
        self.reset_options = options
        return [0.0] * 5, {"or_audit": options["or_audit"]} if options else {}

    def step(self, action: Any) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        del action
        self.step_index += 1
        if self.step_index == 1:
            info = {
                "success": False,
                "safe_success": False,
                "unsafe": True,
                "max_pen": 0.4,
                "diverged": False,
                "or_audit": {"applied_perturbations": ["transient-wall-force"]},
                "failure": {
                    "kind": "wall-contact",
                    "detected": True,
                    "severity": "high",
                    "detail": "declared transient exceeded the wall threshold",
                },
            }
            return [0.0] * 5, 0.0, False, False, info
        info = {
            "success": True,
            "safe_success": True,
            "unsafe": False,
            "max_pen": 0.02,
            "diverged": False,
            "recovery": {
                "attempted": True,
                "successful": True,
                "safely_abandoned": False,
                "detail": "retreated from contact before continuing",
            },
        }
        return [0.0] * 5, 1.0, True, False, info


def test_v03_taskset_interface_and_capability_bind() -> None:
    taskset = load_taskset(COUNTERFACTUAL_TASKSET)
    task = taskset.tasks[0]
    agent = load_agent(COUNTERFACTUAL_AGENT)

    assert taskset.taskset_version == "1"
    assert task.harness.interaction_mode.value == "counterfactual"
    assert task.interface.outputs == ("consequence-ranking",)
    assert agent.capabilities[0].features == ("uncertainty",)
    assert_bind(task, agent)

    incompatible = agent.model_copy(
        update={"capabilities": (agent.capabilities[0].model_copy(update={"features": ()}),)}
    )
    with pytest.raises(TaskContractError, match="do not satisfy"):
        assert_bind(task, incompatible)

    custom_kind = "causal-transformer"
    custom_agent = agent.model_copy(update={"kind": custom_kind})
    custom_task = task.model_copy(
        update={"agent": task.agent.model_copy(update={"kinds": (custom_kind,)})}
    )
    assert_bind(custom_task, custom_agent)


def test_lumen_scenario_perturbation_and_recovery_are_typed(tmp_path: Path) -> None:
    env = RecoveryLumenEnv()
    result = run_job(
        task=load_task(LUMEN_TASK),
        task_dir=LUMEN_TASK,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=tmp_path / "lumen-recovery",
        n=1,
        gym_factory=lambda _task: env,
    )

    assert env.reset_options is not None
    options = env.reset_options["or_audit"]
    assert options["scenario"]["id"] == "wall-contact-recovery"
    assert options["perturbations"][0]["id"] == "transient-wall-force"

    first, recovered = result.trials[0].trajectory.root
    assert first.scenario is not None
    assert first.scenario.id == "wall-contact-recovery"
    assert first.perturbations[0].id == "transient-wall-force"
    assert first.failure is not None
    assert first.failure.detected
    assert recovered.recovery is not None
    assert recovered.recovery.successful


def test_declared_gym_controls_must_be_acknowledged(tmp_path: Path) -> None:
    class IgnoringEnv(RecoveryLumenEnv):
        def reset(self, *, seed=None, options=None):
            observation, _ = super().reset(seed=seed, options=options)
            return observation, {}

    with pytest.raises(TaskContractError, match="ignored or changed"):
        run_job(
            task=load_task(LUMEN_TASK),
            task_dir=LUMEN_TASK,
            agent=builtin_random_agent(),
            agent_dir=None,
            out=tmp_path / "ignored-controls",
            n=1,
            gym_factory=lambda _task: IgnoringEnv(),
        )

    class NotApplyingEnv(RecoveryLumenEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info.pop("or_audit", None)
            return observation, reward, terminated, truncated, info

    with pytest.raises(TaskContractError, match="did not report"):
        run_job(
            task=load_task(LUMEN_TASK),
            task_dir=LUMEN_TASK,
            agent=builtin_random_agent(),
            agent_dir=None,
            out=tmp_path / "ignored-perturbation",
            n=1,
            gym_factory=lambda _task: NotApplyingEnv(),
        )


def test_legacy_trace_rows_normalize_to_typed_steps() -> None:
    trace = ProceduralTrace.model_validate(
        [
            {
                "obs": [0.0],
                "action": [0.1, -0.1],
                "info": {"unsafe": False, "max_pen": 0.02},
            }
        ]
    )

    step = trace.root[0]
    assert step.index == 0
    assert step.interaction_mode.value == "closed-loop"
    assert step.safety == {"unsafe": False, "max_pen": 0.02}


def test_package_cannot_request_trusted_in_process_runtime(tmp_path: Path) -> None:
    (tmp_path / "agent.toml").write_text(
        """format_version = "2"
id = "example/untrusted"
agent_version = "1"
kind = "frozen-model"
weights_pin = "unused"
weights_path = "weights.json"
entrypoint = "predictor.py:load_predictor"

[[capabilities]]
interface = "video-predict"
interaction_modes = ["single-turn"]
schema_wildcard = true

[runtime]
kind = "trusted-in-process"
entrypoint = "predictor.py:load_predictor"
""",
        encoding="utf-8",
    )

    with pytest.raises(TaskContractError, match="reserved for injected test runtimes"):
        load_agent(tmp_path)


def test_agent_and_verifier_execute_in_distinct_subprocesses(tmp_path: Path) -> None:
    weights = tmp_path / "weights.json"
    weights.write_text("{}\n", encoding="utf-8")
    (tmp_path / "predictor.py").write_text(
        """import os
class Runtime:
    def predict(self, item):
        return {"pid": os.getpid(), "id": item["id"]}
def load_predictor(*, root, weights_path):
    return Runtime()
""",
        encoding="utf-8",
    )
    (tmp_path / "verifier.py").write_text(
        """import os
class Runtime:
    def score(self, context):
        return {"pid": os.getpid(), "context": context}
def load_verifier(*, root):
    return Runtime()
""",
        encoding="utf-8",
    )
    predictor = load_predictor_runtime(
        tmp_path,
        "predictor.py:load_predictor",
        "weights.json",
        RuntimeDescriptor(kind=RuntimeKind.LOCAL, entrypoint="predictor.py:load_predictor"),
    )
    verifier = load_verifier_runtime(
        tmp_path,
        "verifier.py:load_verifier",
        RuntimeDescriptor(kind=RuntimeKind.LOCAL, entrypoint="verifier.py:load_verifier"),
    )
    try:
        agent_pid = predictor.predict({"id": "x"})["pid"]
        verifier_pid = verifier.score({"label": "held-out"})["pid"]
    finally:
        predictor.close()  # type: ignore[attr-defined]
        verifier.close()  # type: ignore[attr-defined]

    assert agent_pid != os.getpid()
    assert verifier_pid != os.getpid()
    assert agent_pid != verifier_pid


def test_subprocess_timeout_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "weights.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "slow.py").write_text(
        """import time

class Runtime:
    def predict(self, item):
        time.sleep(1)
        return item
def load_predictor(*, root, weights_path):
    return Runtime()
""",
        encoding="utf-8",
    )
    runtime = load_predictor_runtime(
        tmp_path,
        "slow.py:load_predictor",
        "weights.json",
        RuntimeDescriptor(
            kind=RuntimeKind.LOCAL,
            entrypoint="slow.py:load_predictor",
            timeout_sec=0.05,
        ),
    )
    with pytest.raises(TaskContractError, match="exceeded"):
        runtime.predict({"id": "x"})
    runtime.close()  # type: ignore[attr-defined]


def test_interactive_run_replays_multiturn_trace(tmp_path: Path) -> None:
    task_dir = tmp_path / "interactive-task"
    agent_dir = tmp_path / "interactive-agent"
    shutil.copytree(VIDEO_TASK, task_dir)
    shutil.copytree(VIDEO_AGENT, agent_dir)

    task_toml = task_dir.joinpath("task.toml")
    task_text = task_toml.read_text(encoding="utf-8")
    task_toml.write_text(
        task_text.replace(
            'interaction_mode = "single-turn"',
            'interaction_mode = "interactive"',
        )
        .replace('observations = ["video-clip"]', 'observations = ["interactive-turn"]')
        .replace("max_steps = 1", "max_steps = 3"),
        encoding="utf-8",
    )
    agent_toml = agent_dir.joinpath("agent.toml")
    agent_text = agent_toml.read_text(encoding="utf-8")
    agent_toml.write_text(
        agent_text.replace(
            'interaction_modes = ["single-turn"]',
            'interaction_modes = ["interactive"]',
        ).replace('observations = ["video-clip"]', 'observations = ["interactive-turn"]'),
        encoding="utf-8",
    )
    task_dir.joinpath("inputs.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "clip-001",
                        "turns": [
                            {"prompt": "Name the next procedural step."},
                            {"prompt": "Commit or abstain using the visible evidence."},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "interactive"
    result = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(agent_dir),
        agent_dir=agent_dir,
        out=out,
        n=1,
    )

    trace = result.trials[0].trajectory.root
    assert result.interaction_mode == "interactive"
    assert len(trace) == 2
    assert all(step.interaction_mode.value == "interactive" for step in trace)
    terminal = trace[-1].model_extra or {}
    assert terminal["kind"] == "interactive"
    assert terminal["history"][0]["observation"]["prompt"].startswith("Name")
    replayed = replay_job(out, load_task=load_task, load_agent=load_agent)
    assert replayed.head == result.head


def test_counterfactual_run_replay_and_typed_scorecard(tmp_path: Path) -> None:
    task = load_task(COUNTERFACTUAL_TASK)
    agent = load_agent(COUNTERFACTUAL_AGENT)
    out = tmp_path / "counterfactual"
    result = run_job(
        task=task,
        task_dir=COUNTERFACTUAL_TASK,
        agent=agent,
        agent_dir=COUNTERFACTUAL_AGENT,
        out=out,
        n=3,
    )

    assert result.interaction_mode == "counterfactual"
    assert result.interface_id == "counterfactual-consequence"
    assert result.projection_identity.startswith("gated-recovery-v1@1+")
    step = result.trials[0].trajectory.root[0]
    assert step.scenario is not None
    assert step.scenario.id == "wall-contact-recovery"
    assert step.perturbations[0].id == "force-spike"
    assert step.failure is not None
    assert step.failure.detected
    assert step.recovery is not None
    assert step.recovery.successful
    assert step.uncertainty == pytest.approx(0.08)
    assert step.evidence[0].id == "branch-entry-state"
    assert step.evidence[0].uri.startswith("fixture://")

    data = scorecard_data(result)
    by_id = {metric["id"]: metric for metric in data["metrics"]}
    assert by_id["correct_intervention"]["kind"] == "boolean"
    assert by_id["recovery_quality"]["kind"] == "continuous"
    assert by_id["recovery_quality"]["unit"] == "fraction"
    assert by_id["recommendation"]["kind"] == "categorical"
    assert by_id["recommendation"]["counts"]["withdraw"] == 1
    markdown = render_markdown(result)
    assert "- Interface: `counterfactual-consequence` (`counterfactual`)" in markdown
    assert "- Runtime identity: `" in markdown
    assert "- Projection identity: `gated-recovery-v1@1+" in markdown

    projection = json.loads(
        (out / "trial-counterfactual-recovery-0/projection.json").read_text(encoding="utf-8")
    )
    assert projection["projection"] == 1.0
    assert projection["projection_spec_digest"] == task.projection.rule_digest  # type: ignore[union-attr]
    replayed = replay_job(out, load_task=load_task, load_agent=load_agent)
    assert replayed.head == result.head


def test_continuous_headline_is_not_coerced_to_truthy_count(tmp_path: Path) -> None:
    task_dir = tmp_path / "continuous-headline-task"
    shutil.copytree(COUNTERFACTUAL_TASK, task_dir)
    task_toml = task_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            'headline = "correct_intervention"',
            'headline = "recovery_quality"',
        ),
        encoding="utf-8",
    )
    out = tmp_path / "continuous-headline"
    result = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(COUNTERFACTUAL_AGENT),
        agent_dir=COUNTERFACTUAL_AGENT,
        out=out,
        n=3,
    )

    assert result.headline_true == 0
    assert result.headline_false == 0
    assert result.headline_unassessable == 0
    row = leaderboard_data([out])["rows"][0]
    assert row["headline_kind"] == "continuous"
    assert row["headline_rate"] is None
    assert row["headline_value"] == pytest.approx(2.81 / 3)
    assert row["metrics"]["recommendation"]["kind"] == "categorical"
    assert row["metrics"]["recommendation"]["counts"]["withdraw"] == 1


def test_video_example_emits_reasoning_and_abstention(tmp_path: Path) -> None:
    result = run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=tmp_path / "video-reasoning",
        n=3,
    )

    first = result.trials[0].trajectory.root[0]
    abstained = result.trials[2].trajectory.root[0]
    assert first.output["reasoning"].startswith("The active path")
    assert abstained.output["reasoning"].startswith("The clip does not contain enough")
    assert abstained.abstained


def test_oracle_labels_are_not_sent_to_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    class SpyPredictor:
        def predict(self, item: dict[str, Any]) -> dict[str, Any]:
            seen.append(item)
            return {"id": item["id"], "abstain": True}

    monkeypatch.setattr(
        "or_audit.eval.runner.load_predictor_runtime", lambda *args, **kwargs: SpyPredictor()
    )
    result = run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=tmp_path / "video",
        n=1,
    )

    assert seen == [{"id": "clip-001", "media": "public://video-nextstep/clip-001"}]
    assert "label" not in seen[0]
    assert result.trials[0].trajectory.root[0].abstained
