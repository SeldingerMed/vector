"""Real-physics regression for lumen-nav-safe (Phase A2).

Skipped unless the pinned ``seldinger-lumen`` distribution and ``gymnasium``
are importable: CI has neither, so CI exercises the ``FakeLumenEnv`` paths
instead. Run locally with the pinned stack to prove the packaged task runs
unchanged against real physics and its evidence reconstitutes.

Both calls use the production construction path (registered
``LUMEN_GYM`` engine → ``make_gym_bridge``): pin enforcement at build and
bridge-reported provenance in the head-covered result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import builtin_random_agent, replay_job, run_job
from or_audit.eval.task import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
LUMEN_TASK = ROOT / "docs/examples/tasks/lumen-nav-safe"
PINNED_WORLD_PIN = "3c6bb39ecc916c73e81131b94353ad8deee62bd4"


def test_lumen_nav_safe_runs_against_real_physics(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("lumen.envs.registration")
    task = load_task(LUMEN_TASK)
    assert task.environment.world_pin == PINNED_WORLD_PIN
    assert task.scenarios == ()
    assert task.perturbations == ()
    out = tmp_path / "lumen-real"
    result = run_job(
        task=task,
        task_dir=LUMEN_TASK,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=1,
    )
    assert len(result.trials) == 1
    assert result.world_engine is not None
    assert result.world_engine.backend == "real"
    assert result.world_engine.world_pin == PINNED_WORLD_PIN
    gates = {gate.id: gate.status.value for gate in result.trials[0].vector.gates}
    assert gates.get("wall_penetration") in {"pass", "fail"}
    replayed = replay_job(out, load_task=load_task, load_agent=load_agent)
    assert replayed.head == result.head


def test_lumen_pin_mismatch_refuses_before_construction() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("lumen.envs.registration")
    from or_audit.errors import TaskContractError
    from or_audit.eval.sim.gym_bridge import make_gym_bridge

    task = load_task(LUMEN_TASK)
    wrong = task.model_copy(
        update={"environment": task.environment.model_copy(update={"world_pin": "0" * 40})}
    )
    with pytest.raises(TaskContractError, match="pin mismatch"):
        make_gym_bridge(wrong)

def _with_harness_faults(task: TaskSpec) -> TaskSpec:
    from or_audit.eval.contracts import PerturbationSpec
    return task.model_copy(
        update={
            "perturbations": (
                PerturbationSpec(
                    id="obs-noise",
                    version="1",
                    description="Gaussian observation noise applied by the harness.",
                    kind="harness-observation-gaussian-noise",
                    parameters={"std": 0.01},
                ),
                PerturbationSpec(
                    id="act-hold",
                    version="1",
                    description="Repeat the previous action, applied by the harness.",
                    kind="harness-action-hold",
                ),
            )
        }
    )
def test_lumen_harness_faults_complete_with_fake(tmp_path: Path) -> None:
    """Harness-applied faults need no world support: the runner applies and
    records them, so nominal-vs-fault robustness is measurable on any
    backend, including CI fakes."""
    from tests.test_eval_run import _fake

    task = _with_harness_faults(load_task(LUMEN_TASK))
    out = tmp_path / "lumen-faults"
    result = run_job(
        task=task,
        task_dir=LUMEN_TASK,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=2,
        gym_factory=_fake,
    )
    assert len(result.trials) == 2
    for trial in result.trials:
        gates = {gate.id: gate.status.value for gate in trial.vector.gates}
        assert gates.get("wall_penetration") in {"pass", "fail"}
        recorded = {p.id for step in trial.trajectory.root for p in step.perturbations}
        assert {"obs-noise", "act-hold"} <= recorded


def test_lumen_harness_faults_complete_on_real_physics(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("lumen.envs.registration")
    task = _with_harness_faults(load_task(LUMEN_TASK))
    out = tmp_path / "lumen-real-faults"
    result = run_job(
        task=task,
        task_dir=LUMEN_TASK,
        agent=builtin_random_agent(),
        agent_dir=None,
        out=out,
        n=1,
    )
    gates = {gate.id: gate.status.value for gate in result.trials[0].vector.gates}
    assert gates.get("wall_penetration") in {"pass", "fail"}
    recorded = {p.id for step in result.trials[0].trajectory.root for p in step.perturbations}
    assert {"obs-noise", "act-hold"} <= recorded
