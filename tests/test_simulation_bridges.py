"""Tests for simulation bridges (Gymnasium, SOFA Framework, NVIDIA Warp / Isaac Lab)."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.export_rl import export_job_records
from or_audit.eval.loader import load_task
from or_audit.eval.runner import builtin_random_agent, run_job
from or_audit.eval.sim import (
    BaseSimulationBridge,
    GymnasiumBridge,
    IsaacBridge,
    SimulationEngine,
    SofaBridge,
    WarpBridge,
    clear_simulation_registry,
    list_simulation_engines,
    make_gym_bridge,
    make_sofa_bridge,
    make_warp_bridge,
    register_simulation_engine,
    reset_default_simulation_engines,
)
from or_audit.eval.sim.sofa_bridge import _acquire_sofa_env
from or_audit.eval.sim.warp_bridge import _acquire_warp_env
from or_audit.eval.task import TaskSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
BRONCHO_TASK = REPO_ROOT / "docs/examples/tasks/broncho-airway-nav"


class FakePhysicsEnv:
    """Stand-in for a real backend object: records that it was actually driven."""

    def __init__(self) -> None:
        self.reset_seeds: list[int | None] = []
        self.actions: list[Any] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del options
        self.reset_seeds.append(seed)
        return {"from": "real-backend"}, {"real": True}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return {"from": "real-backend"}, 1.0, True, False, {"real": True}

    def get_state(self) -> dict[str, Any]:
        return {"from": "real-backend"}


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _sim_task(
    tmp_path: Path,
    *,
    kind: str,
    synthetic_stub: bool,
    gym_id: str = "",
    max_steps: int = 3,
) -> TaskSpec:
    """Copy the broncho package and repoint its world at a simulation engine."""
    task_dir = tmp_path / f"{kind}-task"
    shutil.copytree(BRONCHO_TASK, task_dir, ignore=shutil.ignore_patterns("__pycache__"))
    task_toml = task_dir / "task.toml"
    original = task_toml.read_text(encoding="utf-8")
    environment = "\n".join(
        [
            "[environment]",
            f'kind = "{kind}"',
            f'gym_id = "{gym_id}"',
            f'world_pin = "{kind}-pin-v1"',
            f"parameters = {{ max_steps = {max_steps} }}",
            f"synthetic_stub = {str(synthetic_stub).lower()}",
            "n_eval_episodes = 2",
            'seed_policy = "deterministic-eval-2"',
            "",
        ]
    )
    head, _, tail = original.partition("[environment]")
    _, _, rest = tail.partition("\n[interface]")
    task_toml.write_text(f"{head}{environment}\n[interface]{rest}", encoding="utf-8")
    return load_task(task_dir)


def test_simulation_engine_protocols() -> None:
    sofa = SofaBridge("TestScene", allow_synthetic=True)
    assert isinstance(sofa, SimulationEngine)
    assert isinstance(sofa, BaseSimulationBridge)

    warp = WarpBridge("TestWarp", allow_synthetic=True)
    assert isinstance(warp, SimulationEngine)
    assert isinstance(warp, BaseSimulationBridge)

    gym_bridge = GymnasiumBridge(FakePhysicsEnv(), world_pin="gym-pin-v1")
    assert isinstance(gym_bridge, SimulationEngine)
    assert isinstance(gym_bridge, BaseSimulationBridge)
    assert gym_bridge.unwrapped is not None


def test_sofa_bridge_reset_and_step_plumbing() -> None:
    sofa = SofaBridge(
        scene_name="AneurysmCoiling",
        parameters={"max_steps": 10},
        world_pin="sofa-pin-v1",
        allow_synthetic=True,
    )
    assert sofa.world_kind == WorldKind.SOFA

    obs, info = sofa.reset(seed=42)
    assert obs["scene"] == "AneurysmCoiling"
    assert info["scene_name"] == "AneurysmCoiling"
    assert info["seed"] == 42

    _obs, _reward, terminated, truncated, step_info = sofa.step({"insertion_step_mm": 0.0})
    assert not terminated
    assert not truncated
    assert step_info["step"] == 1
    # Every synthetic reading is labelled wherever a verifier can see it.
    assert info["backend"] == "synthetic-stub"
    assert step_info["backend"] == "synthetic-stub"
    assert sofa.get_state()["backend"] == "synthetic-stub"
    assert sofa.render() is None
    sofa.close()


def test_warp_bridge_reset_and_step_plumbing() -> None:
    warp = WarpBridge(
        env_name="SurgicalSuture-v0",
        parameters={"num_envs": 16, "device": "cuda:0"},
        world_pin="warp-pin-v1",
        allow_synthetic=True,
        max_steps=2,
    )
    assert warp.world_kind == WorldKind.WARP
    assert warp.num_envs == 16

    obs, info = warp.reset(seed=123)
    assert obs["num_envs"] == 16
    assert info["gpu_device"] == "cuda:0"
    assert info["backend"] == "synthetic-stub"

    warp.step([0.0] * 7)
    next_obs, _reward, terminated, _trunc, step_info = warp.step([0.0] * 7)
    assert len(next_obs["robot_joint_pos"]) == 7
    assert terminated
    assert step_info["backend"] == "synthetic-stub"
    assert warp.get_state()["backend"] == "synthetic-stub"
    warp.close()


def test_the_harness_owns_the_step_limit_not_environment_parameters() -> None:
    """``environment.parameters`` is forwarded verbatim to a real constructor.

    Putting the harness step limit in that dict is how the generator produced a
    task that ``gymnasium.make`` rejects with an unexpected keyword argument, so
    a stray ``max_steps`` there must have no effect on termination.
    """
    warp = WarpBridge(
        env_name="e",
        parameters={"max_steps": 1},
        allow_synthetic=True,
        max_steps=3,
    )
    warp.reset(seed=1)
    assert warp.step([0.0])[2] is False
    assert warp.step([0.0])[2] is False
    assert warp.step([0.0])[2] is True
    warp.close()


def test_a_stand_in_synthesizes_no_physical_safety_key() -> None:
    """A stub may report progress; it may never report physics.

    The earlier stand-ins emitted ``max_pen``, ``wall_force_n``,
    ``tissue_stress_kpa``, and ``haptic_overshoot_mm`` - numbers with physical
    units that a gate binds to and resolves *pass* against. That is the most
    convincing possible lie, so the invariant covers all three bridges.
    """
    forbidden = {
        "max_pen",
        "wall_force_n",
        "tissue_stress_kpa",
        "haptic_overshoot_mm",
        "contact_force_n",
        "safe_success",
        "workspace_violation",
    }
    bridges = (
        WarpBridge(env_name="e", parameters={}, allow_synthetic=True, max_steps=1),
        SofaBridge(scene_name="s", parameters={}, allow_synthetic=True, max_steps=1),
        IsaacBridge(env_id="e", parameters={}, allow_synthetic=True, max_steps=1),
    )
    for bridge in bridges:
        name = type(bridge).__name__
        _obs, info = bridge.reset(seed=1)
        assert not forbidden & set(info), f"{name} reset invented physics"
        _obs, _reward, _term, _trunc, step_info = bridge.step([0.0])
        leaked = forbidden & set(step_info)
        assert not leaked, f"{name} step invented {sorted(leaked)}"
        # It must still be honest about what it *is*.
        assert step_info["backend"] == "synthetic-stub"
        bridge.close()


def test_sofa_bridge_refuses_to_fabricate_physics() -> None:
    with pytest.raises(TaskContractError) as excinfo:
        SofaBridge("AneurysmCoiling")
    message = str(excinfo.value)
    assert "'sofa'" in message
    assert "environment.synthetic_stub = true" in message
    assert "'Sofa'" in message


def test_warp_bridge_refuses_to_fabricate_physics() -> None:
    with pytest.raises(TaskContractError) as excinfo:
        WarpBridge("SurgicalSuture-v0", world_kind=WorldKind.ISAAC_LAB)
    message = str(excinfo.value)
    assert "'isaac-lab'" in message
    assert "environment.synthetic_stub = true" in message
    assert "'warp'" in message


def test_sim_factories_refuse_a_task_without_the_stub_opt_in(tmp_path: Path) -> None:
    sofa_task = _sim_task(tmp_path, kind="sofa", synthetic_stub=False)
    with pytest.raises(TaskContractError, match="synthetic_stub"):
        make_sofa_bridge(sofa_task)

    warp_task = _sim_task(tmp_path, kind="warp", synthetic_stub=False)
    with pytest.raises(TaskContractError, match="synthetic_stub"):
        make_warp_bridge(warp_task)


def test_sim_factories_stamp_the_stub_when_the_task_opts_in(tmp_path: Path) -> None:
    sofa = make_sofa_bridge(_sim_task(tmp_path, kind="sofa", synthetic_stub=True))
    assert sofa.engine_provenance() == {
        "engine": "sofa",
        "backend": "synthetic-stub",
        "backend_version": "",
        "world_pin": "sofa-pin-v1",
    }

    warp = make_warp_bridge(_sim_task(tmp_path, kind="isaac-lab", synthetic_stub=True))
    assert warp.engine_provenance() == {
        "engine": "isaac-lab",
        "backend": "synthetic-stub",
        "backend_version": "",
        "world_pin": "isaac-lab-pin-v1",
    }


def test_sofa_factory_uses_a_real_backend_when_one_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _sim_task(
        tmp_path,
        kind="sofa",
        synthetic_stub=True,
        gym_id="sofa/AneurysmCoiling-v0",
    )
    real = FakePhysicsEnv()
    monkeypatch.setitem(sys.modules, "Sofa", _module("Sofa", __version__="24.06.00"))
    monkeypatch.setitem(sys.modules, "SofaRuntime", _module("SofaRuntime"))
    monkeypatch.setitem(sys.modules, "sofagym", _module("sofagym"))
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        _module("gymnasium", make=lambda env_id, **kwargs: real),
    )

    bridge = make_sofa_bridge(task)
    assert bridge.engine_provenance() == {
        "engine": "sofa",
        "backend": "real",
        "backend_version": "24.06.00",
        "world_pin": "sofa-pin-v1",
    }
    bridge.reset(seed=7)
    bridge.step({"insertion_step_mm": 1.0})
    assert real.reset_seeds == [7]
    assert real.actions == [{"insertion_step_mm": 1.0}]
    assert bridge.get_state() == {"from": "real-backend"}
    bridge.close()


def test_warp_factory_uses_a_real_backend_when_one_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _sim_task(
        tmp_path,
        kind="warp",
        synthetic_stub=False,
        gym_id="Isaac-Suture-v0",
    )
    real = FakePhysicsEnv()
    monkeypatch.setitem(sys.modules, "warp", _module("warp", __version__="1.9.0"))
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        _module("gymnasium", make=lambda env_id, **kwargs: real),
    )

    bridge = make_warp_bridge(task)
    provenance = bridge.engine_provenance()
    assert provenance["backend"] == "real"
    assert provenance["engine"] == "warp"
    obs, info = bridge.reset(seed=3)
    assert obs == {"from": "real-backend"}
    assert info == {"real": True}
    bridge.step([0.0] * 7)
    assert real.actions == [[0.0] * 7]
    bridge.close()


def test_engine_acquisition_refuses_a_backend_that_cannot_build_the_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _sim_task(tmp_path, kind="warp", synthetic_stub=False, gym_id="Isaac-Unknown-v0")

    def _boom(env_id: str, **kwargs: Any) -> Any:
        raise ValueError(f"unregistered env {env_id}")

    monkeypatch.setattr(
        "or_audit.eval.sim.warp_bridge.module_distribution_version",
        lambda module_name: "",
    )
    monkeypatch.setitem(sys.modules, "warp", _module("warp", __version__="1.9.0"))
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(sys.modules, "gymnasium", _module("gymnasium", make=_boom))

    assert _acquire_warp_env(task) == (None, "1.9.0")
    with pytest.raises(TaskContractError, match="synthetic_stub"):
        make_warp_bridge(task)


def test_engine_acquisition_is_empty_without_optional_bindings(tmp_path: Path) -> None:
    sofa_task = _sim_task(tmp_path, kind="sofa", synthetic_stub=True)
    assert _acquire_sofa_env(sofa_task)[0] is None
    warp_task = _sim_task(tmp_path, kind="warp", synthetic_stub=True)
    assert _acquire_warp_env(warp_task)[0] is None


def test_gym_bridge_reports_a_real_backend() -> None:
    bridge = GymnasiumBridge(
        FakePhysicsEnv(),
        world_kind=WorldKind.LUMEN_GYM,
        world_pin="lumen-pin-v1",
    )
    provenance = bridge.engine_provenance()
    assert provenance["engine"] == "lumen-gym"
    assert provenance["backend"] == "real"
    assert provenance["world_pin"] == "lumen-pin-v1"
    # A test double ships no distribution metadata; the field stays empty rather than guessed.
    assert provenance["backend_version"] == ""


def test_gym_factory_carries_the_task_world_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "or_audit.eval.sim.gym_bridge.make_gym",
        lambda spec: FakePhysicsEnv(),
    )
    bridge = make_gym_bridge(load_task(BRONCHO_TASK))
    assert bridge.engine_provenance() == {
        "engine": "gym",
        "backend": "real",
        "backend_version": "",
        "world_pin": "broncho-synthetic-v1",
    }


def test_base_bridge_default_provenance() -> None:
    class BareBridge(BaseSimulationBridge):
        world_kind = "custom-sim"

    assert BareBridge().engine_provenance() == {
        "engine": "custom-sim",
        "backend": "real",
        "backend_version": "",
        "world_pin": "",
    }


def test_simulation_registry() -> None:
    reset_default_simulation_engines()
    engines = list_simulation_engines()
    assert "sofa" in engines
    assert "warp" in engines
    assert "isaac-lab" in engines
    assert "gym" in engines

    class MockCustomSim(BaseSimulationBridge):
        world_kind = "custom-sim"

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            return {"custom": True}, {"seed": seed}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return {"custom": True}, 1.0, True, False, {"safe_success": True}

    register_simulation_engine("custom-sim", lambda task: MockCustomSim())
    assert "custom-sim" in list_simulation_engines()

    clear_simulation_registry()
    assert len(list_simulation_engines()) == 0
    reset_default_simulation_engines()


def test_run_closed_loop_records_world_engine_provenance(tmp_path: Path) -> None:
    reset_default_simulation_engines()
    broncho_task = load_task(BRONCHO_TASK)
    out_dir = tmp_path / "broncho-run"

    class MockBronchoGym:
        def __init__(self) -> None:
            self._step = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            self._step = 0
            return {"airway_id": "RB1"}, {"seed": seed}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            self._step += 1
            term = self._step >= 5
            return (
                {"airway_id": "RB1_sub"},
                1.0 if term else 0.0,
                term,
                False,
                {
                    "target_reached": term,
                    "max_contact_force_n": 0.5,
                    "safe_navigation": term,
                    "diverged": False,
                },
            )

    res = run_job(
        task=broncho_task,
        task_dir=BRONCHO_TASK,
        agent=builtin_random_agent("broncho-steering"),
        agent_dir=None,
        out=out_dir,
        n=2,
        gym_factory=lambda task: MockBronchoGym(),
    )

    assert res.n == 2
    assert res.trials[0].vector.gates[0].status == "pass"
    m_safe = res.trials[0].vector.metric("safe_navigation")
    assert m_safe is not None
    assert m_safe.value is True

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    # An injected factory exposes no reporter, so the backend is recorded as unattested,
    # but the adapter identity still comes from the world-kind registry.
    world_engine = config["world_engine"]
    assert world_engine["engine"] == "gym"
    assert world_engine["backend"] == "unknown"
    assert world_engine["backend_version"] == ""
    assert world_engine["world_pin"] == "broncho-synthetic-v1"
    assert world_engine["adapter_id"] == "or_audit.eval.sim.gym_bridge:make_gym_bridge"
    assert len(world_engine["adapter_digest"]) == 64
    assert res.world_engine is not None
    assert res.world_engine.adapter_digest == world_engine["adapter_digest"]
    scorecard = json.loads((out_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["world_engine"]["backend"] == "unknown"
    markdown = (out_dir / "scorecard.md").read_text(encoding="utf-8")
    assert "- World engine: `gym` (backend `unknown`)" in markdown
    assert "NOT PHYSICAL EVIDENCE" not in markdown


def test_stub_job_is_stamped_and_refused_by_export_rl(tmp_path: Path) -> None:
    reset_default_simulation_engines()
    task = _sim_task(tmp_path, kind="sofa", synthetic_stub=True)
    out_dir = tmp_path / "sofa-run"
    result = run_job(
        task=task,
        task_dir=tmp_path / "sofa-task",
        agent=builtin_random_agent("broncho-steering"),
        agent_dir=None,
        out=out_dir,
        n=1,
    )
    assert result.n == 1

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["world_engine"] == {
        "engine": "sofa",
        "backend": "synthetic-stub",
        "backend_version": "",
        "world_pin": "sofa-pin-v1",
        "adapter_id": "or_audit.eval.sim.sofa_bridge:make_sofa_bridge",
        "adapter_digest": config["world_engine"]["adapter_digest"],
        "metrics_only": False,
    }
    assert len(config["world_engine"]["adapter_digest"]) == 64

    markdown = (out_dir / "scorecard.md").read_text(encoding="utf-8")
    assert "NOT PHYSICAL EVIDENCE - SYNTHETIC STAND-IN" in markdown
    assert "not a physics backend" in markdown
    assert "- World engine: `sofa` (backend `synthetic-stub`)" in markdown
    html_card = (out_dir / "scorecard.html").read_text(encoding="utf-8")
    assert 'class="stub"' in html_card
    assert "NOT PHYSICAL EVIDENCE - SYNTHETIC STAND-IN" in html_card

    with pytest.raises(TaskContractError, match="synthetic stand-in"):
        export_job_records(out_dir, projection_id="gated_reach_v0")


def test_export_refuses_unknown_backend(tmp_path: Path) -> None:
    from or_audit.eval.job import JobResult, compute_head

    reset_default_simulation_engines()
    task = _sim_task(tmp_path, kind="sofa", synthetic_stub=True)
    out_dir = tmp_path / "sofa-run"
    run_job(
        task=task,
        task_dir=tmp_path / "sofa-task",
        agent=builtin_random_agent("broncho-steering"),
        agent_dir=None,
        out=out_dir,
        n=1,
    )
    # An environment with no reporting backend is "unknown": still unattested,
    # so the export guard must refuse rather than treat it as physical. Flip
    # the attested backend to unknown and re-stamp a consistent head so the
    # provenance refusal (not a head mismatch) is what we exercise.
    result_path = out_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["world_engine"]["backend"] = "unknown"
    payload["head"] = compute_head(JobResult.model_validate(payload))
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="unattested physics"):
        export_job_records(out_dir, projection_id="gated_reach_v0")
