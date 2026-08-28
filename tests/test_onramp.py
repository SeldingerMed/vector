"""Tests for the 10-minute on-ramp: inferred capabilities, strict binding, scaffolding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import surgeval as se
from or_audit.commands.onramp import register
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import InteractionMode
from or_audit.eval.loader import load_agent
from surgeval.decorators import agent, capability_for, describe_agent

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_TASK = REPO_ROOT / "docs/examples/tasks/video-nextstep"


@agent(interface="video-predict", agent_id="declared-video")
class DeclaredVideoModel:
    observations = ("video-clip",)
    outputs = ("next-step",)
    features = ("reasoning", "abstention")

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"next_step": "advance", "outcome": "continue", "unsafe": False}


@agent(interface="gym-policy", agent_id="declared-policy")
class DeclaredPolicyModel:
    actions = ("catheter-delta",)

    def act(self, observation: Any, *, step: int = 0) -> list[float]:
        del observation, step
        return [0.0]


@agent(interface="video-predict", agent_id="undeclared-video")
class UndeclaredVideoModel:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"next_step": "advance", "outcome": "continue", "unsafe": False}


def test_predict_only_infers_single_turn_without_wildcard() -> None:
    capability = capability_for(DeclaredVideoModel)
    assert capability.interface == "video-predict"
    assert InteractionMode.SINGLE_TURN in capability.interaction_modes
    assert InteractionMode.CLOSED_LOOP not in capability.interaction_modes
    assert capability.schema_wildcard is False
    assert capability.observations == ("video-clip",)
    assert capability.outputs == ("next-step",)
    package = DeclaredVideoModel.to_agent_package()  # type: ignore[attr-defined]
    assert package.kind == "frozen-model"
    assert package.runtime is not None
    assert package.runtime.entrypoint == "runner.py:load_predictor"


def test_act_infers_closed_loop() -> None:
    capability = capability_for(DeclaredPolicyModel)
    assert capability.interaction_modes == (InteractionMode.CLOSED_LOOP,)
    package = DeclaredPolicyModel.to_agent_package()  # type: ignore[attr-defined]
    assert package.kind == "policy"
    assert package.runtime is not None
    assert package.runtime.entrypoint == "runner.py:load_policy"


def test_both_methods_declare_both_modes() -> None:
    @agent(interface="video-predict", agent_id="dual")
    class DualModel:
        def predict(self, item: dict[str, Any]) -> dict[str, Any]:
            del item
            return {}

        def act(self, observation: Any, *, step: int = 0) -> int:
            del observation, step
            return 0

    modes = capability_for(DualModel).interaction_modes
    assert InteractionMode.CLOSED_LOOP in modes
    assert InteractionMode.SINGLE_TURN in modes


def test_no_schemas_is_a_wildcard_and_describe_says_so() -> None:
    assert capability_for(UndeclaredVideoModel).schema_wildcard is True
    summary = describe_agent(UndeclaredVideoModel)
    assert "WILDCARD" in summary
    assert "declares no schemas" in summary
    assert "predict(item)" in summary
    assert "WILDCARD" not in describe_agent(DeclaredVideoModel)


def test_agent_without_predict_or_act_is_refused() -> None:
    with pytest.raises(TaskContractError, match="implements neither"):

        @agent(interface="video-predict")
        class Useless:
            pass


def test_declared_mode_must_match_implemented_methods() -> None:
    with pytest.raises(TaskContractError, match="closed-loop"):

        @agent(interface="video-predict", interaction_mode="closed-loop")
        class PredictOnly:
            def predict(self, item: dict[str, Any]) -> dict[str, Any]:
                del item
                return {}


def test_describe_agent_refuses_undecorated_class() -> None:
    class Plain:
        def predict(self, item: dict[str, Any]) -> dict[str, Any]:
            del item
            return {}

    with pytest.raises(TaskContractError, match="not a SurgEval agent"):
        describe_agent(Plain)


def test_default_evaluate_runs_and_records_wildcard(tmp_path: Path) -> None:
    out = tmp_path / "wildcard-run"
    result = se.evaluate(UndeclaredVideoModel(), VIDEO_TASK, out=out, n=2)
    assert result.n == 2
    config = json.loads((out / "config.json").read_text())
    assert config["binding_mode"] == "wildcard"


def test_strict_schemas_refuses_wildcard_binding(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="strict_schemas"):
        se.evaluate(
            UndeclaredVideoModel(),
            VIDEO_TASK,
            out=tmp_path / "strict-run",
            n=1,
            strict_schemas=True,
        )


def test_strict_schemas_accepts_a_declared_binding(tmp_path: Path) -> None:
    out = tmp_path / "declared-run"
    result = se.evaluate(DeclaredVideoModel(), VIDEO_TASK, out=out, n=2, strict_schemas=True)
    assert result.n == 2
    config = json.loads((out / "config.json").read_text())
    assert "binding_mode" not in config


@agent(
    interface="counterfactual-consequence",
    agent_id="declared/counterfactual",
    kind="world-model",
)
class DeclaredCounterfactualModel:
    observations = ("procedural-state", "candidate-interventions")
    outputs = ("consequence-ranking",)
    features = ("uncertainty",)

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {
            "selected": "withdraw-and-reorient",
            "uncertainty": 0.1,
            "ranking": ["withdraw-and-reorient"],
            "recovery_quality": 0.9,
        }


def test_predict_binds_the_counterfactual_task_shape(tmp_path: Path) -> None:
    """predict(item) is the predictor protocol, so it binds counterfactual tasks too."""
    task = REPO_ROOT / "docs/examples/tasks/counterfactual-recovery"
    result = se.evaluate(
        DeclaredCounterfactualModel(),
        task,
        out=tmp_path / "cf-run",
        n=2,
        strict_schemas=True,
    )
    assert result.n == 2
    assert result.trials[0].vector.gates[0].status == "pass"


def test_kind_override_reaches_a_narrower_task(tmp_path: Path) -> None:
    """A task that only accepts world-model agents needs the declared kind, not the shape."""

    @agent(interface="counterfactual-consequence", agent_id="wrong/kind")
    class FrozenKindModel(DeclaredCounterfactualModel):
        pass

    assert capability_for(FrozenKindModel) is not None
    with pytest.raises(TaskContractError, match="kind=frozen-model"):
        se.evaluate(
            FrozenKindModel(),
            REPO_ROOT / "docs/examples/tasks/counterfactual-recovery",
            out=tmp_path / "kind-run",
            n=1,
        )


MODEL_SOURCE = """
from typing import Any

import surgeval as se


@se.agent(interface="video-predict", agent_id="onramp/scaffolded", version="3")
class ScaffoldModel:
    observations = ["video-clip"]
    outputs = ["next-step"]
    features = ["reasoning", "abstention"]

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"next_step": "advance", "outcome": "continue", "unsafe": False}


class NotAnAgent:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {}
"""


@pytest.fixture
def onramp_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgeval")
    register(parser.add_subparsers(dest="command"))
    return parser


@pytest.fixture
def model_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, str]]:
    """Write a decorated model module and make it importable from a clean cwd."""
    name = f"onramp_model_{uuid4().hex[:12]}"
    (tmp_path / f"{name}.py").write_text(MODEL_SOURCE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    try:
        yield tmp_path, name
    finally:
        sys.modules.pop(name, None)


def test_init_agent_writes_a_loadable_package(
    model_module: tuple[Path, str],
    onramp_parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, name = model_module
    out = root / "pkg"
    args = onramp_parser.parse_args(["init-agent", f"{name}:ScaffoldModel", "--out", str(out)])
    assert args.func(args) == 0

    package = load_agent(out)
    assert package.id == "onramp/scaffolded"
    assert package.agent_version == "3"
    assert package.kind == "frozen-model"
    assert package.runtime is not None
    assert package.runtime.entrypoint == "runner.py:load_predictor"
    capability = package.capabilities[0]
    assert capability.schema_wildcard is False
    assert capability.observations == ("video-clip",)

    weights = out / package.weights_path
    assert package.weights_pin == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert (out / f"{name}.py").is_file(), "single-file model modules are vendored in"
    assert "load_policy" not in (out / "runner.py").read_text(encoding="utf-8")
    assert f"sha256:{package.weights_pin}" in capsys.readouterr().out


def test_init_agent_package_actually_runs(
    model_module: tuple[Path, str], onramp_parser: argparse.ArgumentParser
) -> None:
    root, name = model_module
    out = root / "pkg"
    args = onramp_parser.parse_args(["init-agent", f"{name}:ScaffoldModel", "--out", str(out)])
    assert args.func(args) == 0

    result = se.evaluate(out, VIDEO_TASK, out=root / "run", n=2, strict_schemas=True)
    assert result.n == 2
    assert result.trials[0].vector.metrics


def test_init_agent_pins_supplied_weights(
    model_module: tuple[Path, str], onramp_parser: argparse.ArgumentParser
) -> None:
    root, name = model_module
    weights = root / "real-weights.bin"
    weights.write_bytes(b"not-a-placeholder")
    out = root / "pkg"
    args = onramp_parser.parse_args(
        ["init-agent", f"{name}:ScaffoldModel", "--out", str(out), "--weights", str(weights)]
    )
    assert args.func(args) == 0
    package = load_agent(out)
    assert package.weights_path == "real-weights.bin"
    assert (out / "real-weights.bin").read_bytes() == b"not-a-placeholder"


def test_init_agent_refuses_undecorated_target(
    model_module: tuple[Path, str],
    onramp_parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, name = model_module
    args = onramp_parser.parse_args(
        ["init-agent", f"{name}:NotAnAgent", "--out", str(root / "pkg")]
    )
    assert args.func(args) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_describe_agent_command(
    model_module: tuple[Path, str],
    onramp_parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, name = model_module
    args = onramp_parser.parse_args(["describe-agent", f"{name}:ScaffoldModel"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "onramp/scaffolded@3" in out
    assert "video-predict" in out
    assert "verified against declared schemas" in out


def test_describe_agent_command_refuses_undecorated_target(
    model_module: tuple[Path, str],
    onramp_parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, name = model_module
    args = onramp_parser.parse_args(["describe-agent", f"{name}:NotAnAgent"])
    assert args.func(args) == 1
    assert "not a SurgEval agent" in capsys.readouterr().err


def test_describe_agent_command_refuses_bad_target(
    onramp_parser: argparse.ArgumentParser, capsys: pytest.CaptureFixture[str]
) -> None:
    args = onramp_parser.parse_args(["describe-agent", "no-colon-here"])
    assert args.func(args) == 1
    assert "module:Class" in capsys.readouterr().err
