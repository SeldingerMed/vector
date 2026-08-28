"""Tests for the train-time surface: a task exported as a verifiers environment.

The contracts under test are the ones a training team could otherwise violate
without noticing: the reward is the task's declared projection recomputed from a
freshly scored vector, a failed hard gate projects to zero, no scalar escapes
without its projection digest and parent vector reference, and a task whose
world is a stand-in or has no safety instrumentation is not exportable at all.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from or_audit.commands.export_verifiers import register
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.export_verifiers import (
    DEFAULT_AGENT_IDENTITY,
    TASK_PACKAGE_DIR,
    assert_exportable,
    build_export,
    emit_reward_record,
    export_verifiers_environment,
    vector_reference,
)
from or_audit.eval.loader import load_task
from or_audit.eval.vector import project

REPO_ROOT = Path(__file__).resolve().parent.parent
LUMEN_TASK = REPO_ROOT / "docs/examples/tasks/lumen-nav-safe"
PREDICT_TASK = REPO_ROOT / "docs/examples/tasks/video-nextstep"
PROJECTION = "gated_reach_v0"
GENERATED = ("README.md", "env.toml", "load_environment.py", "pyproject.toml")


class StubLumenWorld:
    """Injected world: reports the safety state the task's gate binds to."""

    def __init__(self, *, max_pen: float, success: bool = True, diverged: bool = False) -> None:
        self.safety_max_pen = 0.3
        self._max_pen = max_pen
        self._success = success
        self._diverged = diverged
        self._step = 0
        self.actions: list[Any] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._step = 0
        return {"tip": 0.0}, {"seed": seed, "options": options}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self._step += 1
        terminated = self._step >= 3
        unsafe = self._max_pen > self.safety_max_pen
        return (
            {"tip": float(self._step)},
            # A world reward the export must ignore: if it ever leaked into the
            # projection, these tests would see 7.0 instead of 0.0 / 1.0.
            7.0,
            terminated,
            False,
            {
                "success": self._success,
                "safe_success": self._success and not unsafe and not self._diverged,
                "unsafe": unsafe,
                "max_pen": self._max_pen,
                "diverged": self._diverged,
            },
        )


def _load_generated(out: Path, name: str) -> ModuleType:
    """Import a generated load_environment.py the way a training host would."""
    spec = importlib.util.spec_from_file_location(name, out / "load_environment.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _copy_task(tmp_path: Path, name: str, source: Path = LUMEN_TASK) -> Path:
    target = tmp_path / name
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def _prediction_task(tmp_path: Path) -> Path:
    """video-nextstep with a projection declared: a single-turn export fixture.

    The bundled package declares no projection, which is itself correct — it is
    an eval task. Declaring one here exercises the prediction-mode export path
    without pretending the shipped package is a training environment.
    """
    target = _copy_task(tmp_path, "video-nextstep-projected", source=PREDICT_TASK)
    (target / "verifier.toml").write_text(
        "[projection]\n"
        f'id = "{PROJECTION}"\n'
        'version = "0"\n'
        'source_metric = "next_step_correct"\n'
        "require_false_metrics = []\n",
        encoding="utf-8",
    )
    return target


def _strip_gates(text: str) -> str:
    head = text.split("[[verifier.gates]]", 1)[0]
    tail = "[[verifier.metrics]]" + text.split("[[verifier.metrics]]", 1)[1]
    return head + tail


def test_export_writes_the_environment_with_its_projection_identity(tmp_path: Path):
    out = tmp_path / "env"
    export = export_verifiers_environment(LUMEN_TASK, out=out, projection_id=PROJECTION)
    task = load_task(LUMEN_TASK)
    assert task.projection is not None

    assert export.projection_identity == task.projection.identity
    assert export.projection_digest == task.projection.rule_digest
    assert export.projection_id == PROJECTION
    assert export.task_id == "lumen-nav-safe"
    assert export.world_kind == "lumen-gym"
    assert export.world_pin == task.environment.world_pin

    for name in GENERATED:
        assert (out / name).is_file()
    # The task package is vendored, so the environment carries its own world.
    assert (out / TASK_PACKAGE_DIR / "task.toml").is_file()
    assert (out / TASK_PACKAGE_DIR / "verifier.py").is_file()
    assert not (out / TASK_PACKAGE_DIR / "__pycache__").exists()

    env_toml = (out / "env.toml").read_text(encoding="utf-8")
    assert f'identity = "{task.projection.identity}"' in env_toml
    assert f'digest = "{export.task_digest}"' in env_toml
    assert f'pin = "{export.world_pin}"' in env_toml
    assert f'adapter_digest = "{export.adapter_digest}"' in env_toml
    assert len(export.adapter_digest) == 64

    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "A failed hard gate projects to 0" in readme
    assert "`reward.txt` is **not** an interface here." in readme


def test_two_exports_are_byte_identical(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    export = export_verifiers_environment(LUMEN_TASK, out=first, projection_id=PROJECTION)
    export_verifiers_environment(LUMEN_TASK, out=second, projection_id=PROJECTION)
    for relative in export.paths:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_generated_environment_rewards_the_gated_projection(tmp_path: Path):
    out = tmp_path / "env"
    export = export_verifiers_environment(LUMEN_TASK, out=out, projection_id=PROJECTION)
    module = _load_generated(out, "generated_lumen_safe")
    world = StubLumenWorld(max_pen=0.1)
    env = module.load_environment(gym_factory=lambda task: world, n=2)
    try:
        rollouts = env.evaluate()
    finally:
        env.close()

    task = load_task(LUMEN_TASK)
    assert task.projection is not None
    assert len(rollouts) == 2
    for rollout in rollouts:
        # The reward is the projection of the vector this rollout scored, not
        # the world's own 7.0 reward channel and not a stored float.
        assert rollout.reward == project(rollout.vector, task.projection)
        assert rollout.reward == 1.0
        assert [gate.status for gate in rollout.vector.gates] == ["pass"]
        record = rollout.record
        assert record.projection_digest == task.projection.rule_digest
        assert record.parent_vector_ref == vector_reference(rollout.vector)
        assert record.projection_id == PROJECTION
        assert record.task_digest == export.task_digest
        assert record.world_pin == export.world_pin
        assert record.agent_identity == DEFAULT_AGENT_IDENTITY
    assert [rollout.seed for rollout in rollouts] == [0, 1]
    assert world.actions, "the environment never stepped the injected world"
    # verifiers is not a dependency: the export loads and rewards without it.
    assert env.rubric is None


def test_generated_environment_zeroes_a_failed_hard_gate(tmp_path: Path):
    out = tmp_path / "env"
    export_verifiers_environment(LUMEN_TASK, out=out, projection_id=PROJECTION)
    module = _load_generated(out, "generated_lumen_unsafe")
    env = module.load_environment(gym_factory=lambda task: StubLumenWorld(max_pen=0.9))
    try:
        rollout = env.rollout(seed=0)
    finally:
        env.close()

    task = load_task(LUMEN_TASK)
    assert task.projection is not None
    assert [gate.status for gate in rollout.vector.gates] == ["fail"]
    # raw_success is True in the stub's info; the gate is what zeroes the reward.
    raw_success = rollout.vector.metric("raw_success")
    assert raw_success is not None
    assert raw_success.value is True
    assert rollout.reward == 0.0
    assert rollout.reward == project(rollout.vector, task.projection)
    assert rollout.record.projection_digest
    assert rollout.record.parent_vector_ref


def test_prediction_mode_task_exports_and_scores_its_own_oracle(tmp_path: Path):
    task_dir = _prediction_task(tmp_path)
    out = tmp_path / "predict-env"
    export = export_verifiers_environment(task_dir, out=out, projection_id=PROJECTION)
    assert export.interaction_mode == "single-turn"
    assert export.world_kind == "frame-source"
    assert "CLOSED_LOOP = False" in (out / "load_environment.py").read_text(encoding="utf-8")

    module = _load_generated(out, "generated_video_nextstep")
    env = module.load_environment()
    task = load_task(task_dir)
    assert task.projection is not None
    try:
        assert [item["id"] for item in env.items()] == ["clip-001", "clip-002", "clip-003"]

        def correct(observation: dict[str, Any], step: int) -> dict[str, Any]:
            del step
            answers = {"clip-001": "advance", "clip-002": "withdraw", "clip-003": "hold"}
            return {"next_step": answers[observation["id"]], "unsafe": False}

        def flagged_unsafe(observation: dict[str, Any], step: int) -> dict[str, Any]:
            payload = correct(observation, step)
            payload["unsafe"] = True
            return payload

        good = env.rollout(correct, seed=0)
        assert good.reward == 1.0
        assert good.reward == project(good.vector, task.projection)
        assert good.record.parent_vector_ref == vector_reference(good.vector)

        gated = env.rollout(flagged_unsafe, seed=2)
        assert [gate.status for gate in gated.vector.gates] == ["fail"]
        assert gated.reward == 0.0
        assert gated.record.seed == 2
        assert gated.record.projection_digest == task.projection.rule_digest

        # A prediction task has no world to sample actions from; a rollout
        # without a policy is refused rather than given a fabricated default.
        with pytest.raises(TaskContractError):
            env.rollout(seed=0)
    finally:
        env.close()


def test_generated_environment_rewards_carry_provenance_into_rubric_state(tmp_path: Path):
    out = tmp_path / "env"
    export_verifiers_environment(LUMEN_TASK, out=out, projection_id=PROJECTION)
    module = _load_generated(out, "generated_lumen_state")
    env = module.load_environment(gym_factory=lambda task: StubLumenWorld(max_pen=0.1))
    try:
        rollout = env.rollout(seed=0)
        state = rollout.to_state()
        assert env.reward_func(state=state) == rollout.reward
        assert state["reward_record"]["parent_vector_ref"] == vector_reference(rollout.vector)
        # A state whose record lost its provenance is refused, not silently read.
        stripped = dict(state)
        stripped["reward_record"] = {**state["reward_record"], "projection_digest": ""}
        with pytest.raises(ScoreContractError):
            env.reward_func(state=stripped)
        with pytest.raises(ScoreContractError):
            env.reward_func(state={"seed": 0})
    finally:
        env.close()


def test_generated_environment_refuses_an_edited_task_package(tmp_path: Path):
    out = tmp_path / "env"
    export_verifiers_environment(LUMEN_TASK, out=out, projection_id=PROJECTION)
    verifier = out / TASK_PACKAGE_DIR / "verifier.py"
    verifier.write_text(
        verifier.read_text(encoding="utf-8") + "\n# locally sweetened\n", encoding="utf-8"
    )
    module = _load_generated(out, "generated_lumen_edited")
    with pytest.raises(TaskContractError) as excinfo:
        module.load_environment()
    assert "does not match the exported pin" in str(excinfo.value)


def test_emit_reward_record_refuses_a_scalar_without_provenance():
    fields: dict[str, Any] = {
        "reward": 1.0,
        "projection_id": PROJECTION,
        "projection_version": "0",
        "projection_digest": "d" * 64,
        "parent_vector_ref": "v" * 64,
        "task_id": "lumen-nav-safe",
        "task_version": "0",
        "task_digest": "t" * 64,
        "world_pin": "pin",
        "agent_identity": DEFAULT_AGENT_IDENTITY,
        "seed": 0,
    }
    assert emit_reward_record(**fields).reward == 1.0

    with pytest.raises(ScoreContractError) as no_digest:
        emit_reward_record(**{**fields, "projection_digest": ""})
    assert "digest of the projection rule" in str(no_digest.value)

    with pytest.raises(ScoreContractError) as no_parent:
        emit_reward_record(**{**fields, "parent_vector_ref": ""})
    assert "parent trial" in str(no_parent.value)

    with pytest.raises(ScoreContractError):
        emit_reward_record(**{**fields, "reward": float("nan")})


def test_export_refuses_a_task_with_no_projection(tmp_path: Path):
    task_dir = _copy_task(tmp_path, "no-projection")
    (task_dir / "verifier.toml").unlink()
    with pytest.raises(TaskContractError) as excinfo:
        build_export(task_dir, projection_id=PROJECTION)
    message = str(excinfo.value)
    assert "has no declared projection" in message
    assert "verifier.toml" in message


def test_export_refuses_a_mismatched_projection_id(tmp_path: Path):
    with pytest.raises(TaskContractError) as excinfo:
        build_export(LUMEN_TASK, projection_id="ungated_reach_v9")
    assert "declares 'gated_reach_v0', not 'ungated_reach_v9'" in str(excinfo.value)


def test_export_refuses_a_synthetic_stub_task(tmp_path: Path):
    task_dir = _copy_task(tmp_path, "stubbed")
    toml_path = task_dir / "task.toml"
    toml_path.write_text(
        toml_path.read_text(encoding="utf-8").replace(
            '[environment]\nkind = "lumen-gym"',
            '[environment]\nkind = "lumen-gym"\nsynthetic_stub = true',
        ),
        encoding="utf-8",
    )
    assert load_task(task_dir).environment.synthetic_stub
    with pytest.raises(TaskContractError) as excinfo:
        build_export(task_dir, projection_id=PROJECTION)
    message = str(excinfo.value)
    assert "environment.synthetic_stub" in message
    assert "fabricated physics" in message


def test_export_refuses_a_metrics_only_task(tmp_path: Path):
    task_dir = _copy_task(tmp_path, "metrics-only")
    toml_path = task_dir / "task.toml"
    text = _strip_gates(toml_path.read_text(encoding="utf-8"))
    text = text.replace("safety_critical = true", "safety_critical = false")
    text = text.replace(
        '[environment]\nkind = "lumen-gym"',
        '[environment]\nkind = "lumen-gym"\nmetrics_only = true',
    )
    toml_path.write_text(text, encoding="utf-8")
    task = load_task(task_dir)
    assert task.environment.metrics_only
    assert task.verifier.gates == ()
    with pytest.raises(TaskContractError) as excinfo:
        build_export(task_dir, projection_id=PROJECTION)
    message = str(excinfo.value)
    assert "environment.metrics_only" in message
    assert "no safety instrumentation" in message


def test_export_refuses_safety_critical_without_hard_gates():
    task = load_task(LUMEN_TASK)
    # The loader already refuses this pairing, so the only way to reach the
    # export-side guard is to construct it without revalidation.
    gateless = task.model_copy(update={"verifier": task.verifier.model_copy(update={"gates": ()})})
    assert gateless.metadata.safety_critical
    with pytest.raises(TaskContractError) as excinfo:
        assert_exportable(gateless)
    assert "no hard gates" in str(excinfo.value)


def test_export_refuses_a_projection_that_does_not_zero_a_gate_failure(tmp_path: Path):
    """A training reward must be 0 on an unsafe episode, not an exception."""
    task_dir = _copy_task(tmp_path, "refuse-policy")
    verifier_toml = task_dir / "verifier.toml"
    text = verifier_toml.read_text(encoding="utf-8")
    # The example relies on the ZERO default; declare the refusing policy explicitly.
    verifier_toml.write_text(
        text.replace("[projection]", '[projection]\ngate_failure = "refuse"'), encoding="utf-8"
    )
    task = load_task(task_dir)
    assert task.projection is not None
    assert task.projection.gate_failure.value == "refuse"
    with pytest.raises(TaskContractError) as excinfo:
        build_export(task_dir, projection_id=PROJECTION)
    message = str(excinfo.value)
    assert "gate_failure='refuse'" in message
    assert "project a hard-gate failure to zero" in message
    # An abstained gate is a measurement gap, so refusing there stays legal.
    assert 'gate_unassessable may stay "refuse"' in message


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_cli_exports_and_warns_that_the_scalar_is_a_projection(tmp_path, capsys):
    out = tmp_path / "env"
    args = _parse(
        ["export-verifiers", str(LUMEN_TASK), "--projection", PROJECTION, "--out", str(out)]
    )
    assert args.func(args) == 0
    printed = capsys.readouterr().out
    assert "surgeval-env-lumen-nav-safe" in printed
    assert str(out / "load_environment.py") in printed
    projection = load_task(LUMEN_TASK).projection
    assert projection is not None
    assert f"projection {projection.identity}" in printed
    assert "WARNING: the exported scalar is a projection of a safety vector, not a score." in (
        printed.replace("\n", " ").replace("  ", " ")
    )
    assert (out / "load_environment.py").is_file()


def test_cli_refuses_on_stderr(tmp_path, capsys):
    args = _parse(
        [
            "export-verifiers",
            str(LUMEN_TASK),
            "--projection",
            "ungated_reach_v9",
            "--out",
            str(tmp_path / "env"),
        ]
    )
    assert args.func(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("REFUSED: ")
    assert "ungated_reach_v9" in captured.err
    assert not (tmp_path / "env" / "load_environment.py").exists()
