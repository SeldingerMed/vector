"""Tests for surgeval developer SDK, decorators, model wrappers, and CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import surgeval as se
from or_audit.cli import build_parser, main
from or_audit.errors import TaskContractError
from or_audit.eval.bind import assert_bind
from or_audit.eval.contracts import InteractionMode
from or_audit.eval.loader import load_task
from surgeval.decorators import agent
from surgeval.integrations import wrap_gym_policy, wrap_hf, wrap_pytorch


def test_surgeval_module_exports() -> None:
    assert hasattr(se, "evaluate")
    assert hasattr(se, "load_task")
    assert hasattr(se, "load_taskset")
    assert hasattr(se, "load_agent")
    assert hasattr(se, "agent")
    assert hasattr(se, "wrap_pytorch")
    assert hasattr(se, "wrap_hf")
    assert hasattr(se, "wrap_gym_policy")
    assert hasattr(se, "ModalityKind")
    assert hasattr(se, "GateKind")
    assert hasattr(se, "__version__")


@agent(interface="video-predict", interaction_mode="single-turn", agent_id="cvs-detector")
class DummyCvsModel:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"cvs_achieved": True, "critical_structure": "cystic_duct"}


def test_agent_decorator_and_evaluate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs"

    out_dir = tmp_path / "sdk-eval-out"
    result = se.evaluate(DummyCvsModel(), task_dir, out=out_dir, n=3)
    assert result.n == 3
    assert len(result.trials) == 3
    assert result.trials[0].vector.gates[0].status == "pass"
    assert (out_dir / "scorecard.json").exists()


def test_pytorch_wrapper_lifecycle() -> None:
    class MockTorchModule:
        def __init__(self) -> None:
            self._reset_called = False

        def reset(self, seed: int | None = None) -> None:
            del seed
            self._reset_called = True

        def __call__(self, x: Any) -> Any:
            return [0.5, -0.5]

    module = MockTorchModule()
    wrapped = wrap_pytorch(module)
    wrapped.reset(seed=42)
    assert module._reset_called is True

    act = wrapped.act({"obs": [1.0, 2.0]})
    assert act == [0.5, -0.5]


def test_huggingface_wrapper_lifecycle() -> None:
    def mock_pipeline(item: dict[str, Any]) -> dict[str, Any]:
        return {"phase": "CalotTriangleDissection", "confidence": 0.95}

    hf_wrapped = wrap_hf(mock_pipeline)
    res = hf_wrapped.predict({"clip_id": "clip-01"})
    assert res["phase"] == "CalotTriangleDissection"
    assert res["confidence"] == 0.95


def test_gym_policy_wrapper_lifecycle() -> None:
    class MockSB3Policy:
        def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, None]:
            del obs, deterministic
            return [1.0, 0.0], None

    sb3_wrapped = wrap_gym_policy(MockSB3Policy())
    act = sb3_wrapped.act({"joint_pos": [0.0] * 7})
    assert act == [1.0, 0.0]


def test_cli_adapters_and_sim_list(capsys: pytest.CaptureFixture[str]) -> None:
    # Test `surgeval adapters list`
    status_adapters = main(["adapters", "list"])
    assert status_adapters == 0
    captured_adapters = capsys.readouterr()
    assert "Registered Modality Adapters" in captured_adapters.out
    assert "video-laparoscopic" in captured_adapters.out
    assert "airway-bronchoscopy" in captured_adapters.out
    assert "fluoroscopy-dsa" in captured_adapters.out
    assert "robotic-kinematics" in captured_adapters.out

    # Test `surgeval sim list`
    status_sim = main(["sim", "list"])
    assert status_sim == 0
    captured_sim = capsys.readouterr()
    assert "Registered Simulation Engines" in captured_sim.out
    assert "sofa" in captured_sim.out
    assert "warp" in captured_sim.out
    assert "gym" in captured_sim.out


def test_cli_prog_name() -> None:
    parser = build_parser(prog="surgeval")
    assert parser.prog == "surgeval"


def test_evaluate_temp_dir_and_task_dir() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs"
    task_obj = load_task(task_dir)
    # Evaluate with out=None (temp dir) and explicit task_dir
    result = se.evaluate(DummyCvsModel(), task_obj, task_dir=task_dir, n=2)
    assert result.n == 2
    assert len(result.trials) == 2
    assert result.trials[0].vector.gates[0].status == "pass"


def test_to_agent_package_contract() -> None:
    pkg = DummyCvsModel.to_agent_package()  # type: ignore[attr-defined]
    assert pkg.id == "custom/cvs-detector"
    assert pkg.weights_path == "weights.json"
    assert pkg.weights_pin != ""
    assert pkg.runtime is not None
    assert pkg.runtime.kind == "local"


def test_gym_policy_wrapper_non_sb3() -> None:
    # Model with 1-arg predict(obs)
    class CustomSimplePolicy:
        def predict(self, obs: Any) -> tuple[float, float]:
            del obs
            return 0.5, -0.5

    wrapped = wrap_gym_policy(CustomSimplePolicy())
    act = wrapped.act({"state": 1})
    assert act == 0.5


def test_error_exports() -> None:
    assert hasattr(se, "TaskContractError")
    assert hasattr(se, "ScoreContractError")
    assert hasattr(se, "AuditChainError")


class SdkModelA:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"cvs_achieved": True, "critical_structure": "cystic_duct"}


class SdkModelB:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"cvs_achieved": False, "critical_structure": "common_bile_duct"}


def test_sdk_identity_uses_real_weights_pin(tmp_path: Path) -> None:
    """Two different models must produce different agent ids."""
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs"

    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"
    se.evaluate(SdkModelA(), task_dir, out=out_a, n=1)
    se.evaluate(SdkModelB(), task_dir, out=out_b, n=1)

    import json

    cfg_a = json.loads((out_a / "config.json").read_text())
    cfg_b = json.loads((out_b / "config.json").read_text())
    assert cfg_a["agent_id"] != cfg_b["agent_id"], "different models must have different agent ids"
    assert cfg_a["binding_mode"] == "wildcard"
    assert cfg_b["binding_mode"] == "wildcard"


def test_sdk_bundle_has_no_host_paths(tmp_path: Path) -> None:
    """The generated runner.py must not embed absolute host paths."""
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs"
    out = tmp_path / "out-hostpath"
    se.evaluate(DummyCvsModel(), task_dir, out=out, n=1)
    runner_code = (out / "bundle" / "agent" / "runner.py").read_text()
    assert "/Users/" not in runner_code, "runner.py must not embed absolute host paths"


# --------------------------------------------------------------------------
# to_agent_package: one package, one runtime identity
# --------------------------------------------------------------------------


@agent(interface="video-predict", agent_id="sdk/dual")
class SdkDualModel:
    """Implements both wire protocols, so it has two runtime identities."""

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"cvs_achieved": True, "critical_structure": "cystic_duct"}

    def act(self, observation: Any, *, step: int = 0) -> list[float]:
        del observation, step
        return [0.0]


def test_to_agent_package_refuses_a_dual_mode_class_without_a_mode() -> None:
    """A package names one entrypoint, so it must not claim the other's modes."""
    with pytest.raises(TaskContractError, match="mode="):
        SdkDualModel.to_agent_package()  # type: ignore[attr-defined]


def test_to_agent_package_publishes_only_the_predictor_identity() -> None:
    pkg = SdkDualModel.to_agent_package(mode="single-turn")  # type: ignore[attr-defined]
    assert pkg.kind == "frozen-model"
    assert pkg.runtime is not None
    assert pkg.runtime.entrypoint == "runner.py:load_predictor"
    modes = pkg.capabilities[0].interaction_modes
    assert InteractionMode.CLOSED_LOOP not in modes
    assert InteractionMode.SINGLE_TURN in modes
    # The package a predictor task is handed must actually bind, kind included.
    repo_root = Path(__file__).resolve().parent.parent
    assert_bind(load_task(repo_root / "docs/examples/tasks/video-nextstep"), pkg)


def test_to_agent_package_publishes_only_the_policy_identity() -> None:
    pkg = SdkDualModel.to_agent_package(mode="closed-loop")  # type: ignore[attr-defined]
    assert pkg.kind == "policy"
    assert pkg.runtime is not None
    assert pkg.runtime.entrypoint == "runner.py:load_policy"
    assert pkg.capabilities[0].interaction_modes == (InteractionMode.CLOSED_LOOP,)


def test_to_agent_package_refuses_a_mode_the_class_cannot_drive() -> None:
    with pytest.raises(TaskContractError, match="not one of the modes"):
        DummyCvsModel.to_agent_package(mode="closed-loop")  # type: ignore[attr-defined]


def test_to_agent_package_keeps_the_interface_override_positional() -> None:
    """Single-identity classes need no mode: the ordinary path stays one call."""
    pkg = DummyCvsModel.to_agent_package("gym-policy")  # type: ignore[attr-defined]
    assert pkg.capabilities[0].interface == "gym-policy"
    assert pkg.kind == "frozen-model"
    assert pkg.runtime is not None
    assert pkg.runtime.entrypoint == "runner.py:load_predictor"


def test_dual_mode_class_still_evaluates_with_no_mode_argument(tmp_path: Path) -> None:
    """The trial path is unchanged: the task's required mode picks the identity.

    ``se.evaluate`` resolves kind and entrypoint from the mode the task asks for
    (``surgeval.client._resolve_synthesis``), so a dual-mode class needs no mode
    from the user. Only publication - which has no task in hand - has to choose.
    """
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs"
    result = se.evaluate(SdkDualModel(), task_dir, out=tmp_path / "dual-trial", n=2)
    assert result.n == 2
    assert result.trials[0].vector.gates[0].status == "pass"
