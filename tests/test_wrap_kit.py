"""Tests for the wrap kit (N3): scaffolded packages run, and dishonest ones refuse."""

from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
from pathlib import Path
from typing import Any

import pytest

from or_audit.commands.wrap import register
from or_audit.errors import TaskContractError
from or_audit.eval.export_rl import export_job_records
from or_audit.eval.loader import load_task
from or_audit.eval.runner import builtin_random_agent, run_job
from or_audit.eval.sim import reset_default_simulation_engines
from or_audit.eval.wrap import GateMapping, WrapRequest, scaffold_wrap

REPO_ROOT = Path(__file__).resolve().parent.parent
ORBIT_TASK = REPO_ROOT / "docs/examples/tasks/orbit-needle-lift"

FORCE_GATE = GateMapping(
    id="grasp-force",
    signal="contact_force_n",
    fail_when="contact_force_n > 1.5",
    threshold=1.5,
    unit="N",
    citation="dVRK needle-driver grasp force envelope v1",
)


def _request(**overrides: Any) -> WrapRequest:
    fields: dict[str, Any] = {
        "env_id": "SurRoL/NeedleReach-v0",
        "task_id": "surrol-needle-reach",
        "world_pin": "surrol-v2.0.1",
        "license": "MIT",
        "source_repo": "https://github.com/med-air/SurRoL",
        "max_steps": 4,
        "n_eval_episodes": 2,
        "gate_mappings": (FORCE_GATE,),
    }
    fields.update(overrides)
    return WrapRequest(**fields)


class ReportingGym:
    """Stand-in world that reports the signals a wrap declares."""

    def __init__(self, *, contact_force_n: float = 0.4, report_safety: bool = True) -> None:
        self._step = 0
        self._force = contact_force_n
        self._report_safety = report_safety

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del options
        self._step = 0
        return {"obs": [0.0]}, {"seed": seed}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        self._step += 1
        done = self._step >= 3
        info: dict[str, Any] = {"raw_success": done, "diverged": False}
        if self._report_safety:
            info["contact_force_n"] = self._force
        return {"obs": [0.1 * self._step]}, 1.0 if done else 0.0, done, False, info


def _run(task_dir: Path, out: Path, gym: ReportingGym | None = None) -> Any:
    task = load_task(task_dir)
    return run_job(
        task=task,
        task_dir=task_dir,
        agent=builtin_random_agent(task.interface.id),
        agent_dir=None,
        out=out,
        n=1,
        gym_factory=lambda _task: gym or ReportingGym(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgeval")
    register(parser.add_subparsers(dest="command"))
    return parser


def test_gated_wrap_loads_and_runs(tmp_path: Path) -> None:
    result = scaffold_wrap(_request(), tmp_path / "wrap")
    assert [path.name for path in result.files] == [
        "instruction.md",
        "task.toml",
        "verifier.py",
        "wrap.json",
    ]
    job = _run(result.root, tmp_path / "run")
    assert job.n == 1
    vector = job.trials[0].vector
    assert [(gate.id, gate.status) for gate in vector.gates] == [("grasp-force", "pass")]
    assert {metric.id: metric.value for metric in vector.metrics} == {
        "safe_success": True,
        "raw_success": True,
        "contact_force_n": 0.4,
        "diverged": False,
    }
    assert vector.metric("safe_success").headline is True


def test_gated_wrap_gate_bites_on_the_declared_signal(tmp_path: Path) -> None:
    result = scaffold_wrap(_request(), tmp_path / "wrap")
    job = _run(result.root, tmp_path / "run", ReportingGym(contact_force_n=2.0))
    vector = job.trials[0].vector
    assert vector.gates[0].status == "fail"
    # A breached gate is never laundered into the headline.
    assert vector.metric("safe_success").value is False


def test_unreported_signal_abstains_instead_of_passing(tmp_path: Path) -> None:
    """A wrapped world that reports nothing must not yield a safety verdict."""
    result = scaffold_wrap(_request(), tmp_path / "wrap")
    job = _run(result.root, tmp_path / "run", ReportingGym(report_safety=False))
    vector = job.trials[0].vector
    assert vector.gates[0].status == "not_assessable"
    assert vector.metric("safe_success").value is None
    assert vector.metric("contact_force_n").value is None


def test_metrics_only_wrap_is_not_safety_attested(tmp_path: Path) -> None:
    result = scaffold_wrap(_request(gate_mappings=(), metrics_only=True), tmp_path / "wrap")
    assert result.metrics_only is True
    task = load_task(result.root)
    assert task.verifier.gates == ()
    assert task.metadata.safety_critical is False
    assert task.environment.metrics_only is True
    assert task.verifier.headline == "raw_success"
    job = _run(result.root, tmp_path / "run")
    assert job.world_engine is not None
    assert job.world_engine.metrics_only is True
    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert config["world_engine"]["metrics_only"] is True
    instruction = (result.root / "instruction.md").read_text(encoding="utf-8")
    assert "explicitly **not safety-attested**" in instruction


@pytest.mark.parametrize("gated", [True, False])
def test_generated_verifier_is_importable_python_in_both_modes(gated: bool, tmp_path: Path):
    """The scaffold is source code, so both branches must compile and import.

    The gateless branch is the one that regressed: it renders `GATES = ()` with
    no gate functions above it, so the closing paren and the blank-line
    separator both have to disappear with them.
    """
    request = _request() if gated else _request(gate_mappings=(), metrics_only=True)
    result = scaffold_wrap(request, tmp_path / "wrap")
    path = result.root / "verifier.py"
    py_compile.compile(str(path), cfile=str(tmp_path / "v.pyc"), doraise=True)

    spec = importlib.util.spec_from_file_location("generated_verifier", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.GATES) == len(request.gate_mappings)

    # The prose must not promise gate handling a gateless package does not do.
    docstring = module.__doc__ or ""
    assert ("scalar-dsl" in docstring) is gated
    assert ("maps no hard gate" in docstring) is not gated


def test_no_gates_without_metrics_only_is_refused() -> None:
    with pytest.raises(TaskContractError, match="--metrics-only"):
        _request(gate_mappings=())


def test_metrics_only_with_gates_is_refused() -> None:
    with pytest.raises(TaskContractError, match="not safety-attested"):
        _request(metrics_only=True)


def test_threshold_without_a_basis_is_refused() -> None:
    with pytest.raises(TaskContractError, match="has no basis"):
        GateMapping(
            id="grasp-force",
            signal="contact_force_n",
            fail_when="contact_force_n > 1.5",
            threshold=1.5,
            unit="N",
        )


def test_threshold_without_a_unit_is_refused() -> None:
    with pytest.raises(TaskContractError, match="has no unit"):
        GateMapping(
            id="grasp-force",
            signal="contact_force_n",
            fail_when="contact_force_n > 1.5",
            threshold=1.5,
            citation="some envelope",
        )


def test_unpinned_wrap_is_refused() -> None:
    with pytest.raises(TaskContractError, match="world-pin"):
        _request(world_pin="")


def test_unknown_world_kind_without_capabilities_names_the_entry_point_group(
    tmp_path: Path,
) -> None:
    with pytest.raises(TaskContractError, match=r"or_audit\.world_kinds"):
        scaffold_wrap(_request(world_kind="vendor-sim"), tmp_path / "wrap")


def test_gate_may_not_reference_another_signal() -> None:
    with pytest.raises(TaskContractError, match="declares only the signal"):
        GateMapping(
            id="grasp-force",
            signal="contact_force_n",
            fail_when="contact_force_n > 1.5 or unsafe",
        )


def test_gate_may_not_bind_to_an_outcome_flag() -> None:
    with pytest.raises(TaskContractError, match="outcome flag"):
        GateMapping(id="won", signal="raw_success", fail_when="raw_success == false")


@pytest.mark.parametrize("expression", ['"contact_force_n > 1.5"', "true", "1 > 0"])
def test_gate_that_never_reads_its_signal_is_refused(expression: str) -> None:
    """A quoted expression is a string literal, and a constant verdict is not a gate."""
    with pytest.raises(TaskContractError, match="constant verdict"):
        GateMapping(id="grasp-force", signal="contact_force_n", fail_when=expression)


def test_scaffolding_twice_is_byte_identical(tmp_path: Path) -> None:
    request = _request()
    root = tmp_path / "wrap"
    first = {path.name: path.read_bytes() for path in scaffold_wrap(request, root).files}
    second = {path.name: path.read_bytes() for path in scaffold_wrap(request, root).files}
    assert first == second
    other = {
        path.name: path.read_bytes() for path in scaffold_wrap(request, tmp_path / "other").files
    }
    # Only the generated conformance command carries the output path.
    assert {name: blob for name, blob in first.items() if name != "wrap.json"} == {
        name: blob for name, blob in other.items() if name != "wrap.json"
    }


def test_generated_package_is_adapter_pinned(tmp_path: Path) -> None:
    """The pin must be the installed adapter's, so load_task verifies it rather than trust it."""
    reset_default_simulation_engines()
    result = scaffold_wrap(_request(), tmp_path / "wrap")
    task = load_task(result.root)
    assert task.environment.adapter == "or_audit.eval.sim.gym_bridge:make_gym_bridge"
    assert len(task.environment.adapter_digest) == 64
    assert int(task.environment.adapter_digest, 16) >= 0
    assert task.environment.adapter_digest == result.adapter_digest
    wrap_json = json.loads((result.root / "wrap.json").read_text(encoding="utf-8"))
    assert wrap_json["adapter_id"] == task.environment.adapter
    assert wrap_json["adapter_digest"] == task.environment.adapter_digest
    assert wrap_json["license"] == "MIT"
    assert "surgeval conformance" in wrap_json["next_steps"][0]


def test_broken_adapter_pin_is_refused(tmp_path: Path) -> None:
    result = scaffold_wrap(_request(), tmp_path / "wrap")
    task_toml = result.root / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(result.adapter_digest, "0" * 64),
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="content digest mismatch"):
        load_task(result.root)


def test_cli_scaffolds_and_prints_the_conformance_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "wrap"
    args = _parser().parse_args(
        [
            "wrap",
            "SurRoL/NeedleReach-v0",
            "--task-id",
            "surrol-needle-reach",
            "--world-pin",
            "surrol-v2.0.1",
            "--license",
            "MIT",
            "--out",
            str(out),
            "--max-steps",
            "4",
            "--gate",
            "grasp-force=contact_force_n:contact_force_n > 1.5@1.5:N:dVRK envelope v1, Table 2",
        ]
    )
    assert args.func(args) == 0
    printed = capsys.readouterr().out
    assert f"surgeval conformance {out}" in printed
    assert "grasp-force" in printed
    task = load_task(out)
    assert task.verifier.gates[0].threshold == 1.5
    basis = task.verifier.gates[0].threshold_basis
    assert basis is not None
    assert basis.citation == "dVRK envelope v1, Table 2"


def test_cli_refuses_a_wrap_with_no_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _parser().parse_args(
        [
            "wrap",
            "SurRoL/NeedleReach-v0",
            "--task-id",
            "surrol-needle-reach",
            "--world-pin",
            "surrol-v2.0.1",
            "--license",
            "MIT",
            "--out",
            str(tmp_path / "wrap"),
        ]
    )
    assert args.func(args) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("REFUSED: ")
    assert "--metrics-only" in captured.err
    assert not (tmp_path / "wrap").exists()


@pytest.mark.parametrize(
    "gate",
    [
        "no-equals-sign",
        "id=signal-without-expression",
        'id=contact_force_n:contact_force_n > 1.5@notanumber:N:"cite"',
        "id=contact_force_n:contact_force_n > 1.5@1.5:N",
        'id=contact_force_n:contact_force_n > 1.5@1.5::"cite"',
        "id=contact_force_n:contact_force_n > 1.5@1.5:N:",
    ],
)
def test_cli_refuses_a_malformed_gate(gate: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parser().parse_args(
            [
                "wrap",
                "SurRoL/NeedleReach-v0",
                "--task-id",
                "t",
                "--world-pin",
                "p",
                "--license",
                "MIT",
                "--out",
                str(tmp_path / "wrap"),
                "--gate",
                gate,
            ]
        )
    assert exit_info.value.code == 2


def test_orbit_example_is_an_honest_metrics_only_wrap(tmp_path: Path) -> None:
    """The N2/N4 worked wrap, as it survives contact with the real upstream.

    The earlier version of this example declared six physics gates with cited
    thresholds while running against a synthetic stand-in, so every gate
    resolved *pass* against invented numbers. Two rules now make that
    impossible - a stub reports no physical key, and a gate needs a citation -
    and the honest package that remains is metrics-only: it measures progress,
    declares no safety verdict, and says so in its own tags.
    """
    reset_default_simulation_engines()
    task = load_task(ORBIT_TASK)
    assert task.environment.kind_key == "isaac-lab"
    assert task.environment.world_pin
    assert task.environment.synthetic_stub is True
    assert task.environment.metrics_only is True
    # No gates at all: the claim boundary is the absence, not a passing gate.
    assert task.verifier.gates == ()
    assert task.metadata.safety_critical is False
    assert "metrics-only" in task.metadata.tags

    out = tmp_path / "orbit-run"
    job = run_job(
        task=task,
        task_dir=ORBIT_TASK,
        agent=builtin_random_agent(task.interface.id),
        agent_dir=None,
        out=out,
        n=1,
    )
    assert job.world_engine is not None
    assert job.world_engine.backend == "synthetic-stub"
    assert job.world_engine.engine == "isaac-lab"
    assert job.world_engine.adapter_id == "or_audit.eval.sim.isaac_bridge:make_isaac_bridge"
    assert "NOT PHYSICAL EVIDENCE" in (out / "scorecard.md").read_text(encoding="utf-8")
    with pytest.raises(TaskContractError, match="synthetic stand-in"):
        export_job_records(out, projection_id="gated_reach_v0")

    # The trajectory carries no physical safety key, because the stand-in has
    # no physics to report; this is what keeps the gates honestly absent.
    trial = next((out / "trial-orbit-needle-lift-0").glob("trajectory.json"))
    text = trial.read_text(encoding="utf-8")
    for invented in ("max_pen", "contact_force_n", "workspace_violation", "safe_success"):
        assert invented not in text
