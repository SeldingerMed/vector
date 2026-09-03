"""Tests for simulation bridges (Gymnasium, SOFA Framework, NVIDIA Warp / Isaac Lab)."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.export_rl import export_job_records
from or_audit.eval.gate_dsl import evaluate_gate
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
from or_audit.eval.sim.isaac_bridge import SAFETY_KEY_SOURCES, SAFETY_SIGNALS
from or_audit.eval.sim.sofa_bridge import _acquire_sofa_env
from or_audit.eval.sim.warp_bridge import _acquire_warp_env
from or_audit.eval.task import GateSpec, TaskSpec, ThresholdBasis
from or_audit.eval.worlds import (
    WorldCapabilities,
    WorldKindSpec,
    register_world_kind,
    world_kind_spec,
)

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
    """Plumbing only, at the one batch width this bridge can actually serve.

    This test used to pass ``num_envs=16`` and assert both ``warp.num_envs == 16``
    and ``obs["num_envs"] == 16``, which pinned the defect: the bridge never
    batched anything, so the same run reported one 7-element joint vector and one
    scalar reward while claiming sixteen environments. The batch width now has
    its own refusal test below; this one keeps the reset/step plumbing coverage
    it was actually written for.
    """
    warp = WarpBridge(
        env_name="SurgicalSuture-v0",
        parameters={"num_envs": 1, "device": "cuda:0"},
        world_pin="warp-pin-v1",
        allow_synthetic=True,
        max_steps=2,
    )
    assert warp.world_kind == WorldKind.WARP
    assert warp.num_envs == 1

    obs, info = warp.reset(seed=123)
    assert obs["num_envs"] == 1
    assert info["gpu_device"] == "cuda:0"
    assert info["backend"] == "synthetic-stub"

    warp.step([0.0] * 7)
    next_obs, _reward, terminated, _trunc, step_info = warp.step([0.0] * 7)
    assert len(next_obs["robot_joint_pos"]) == 7
    assert terminated
    assert step_info["backend"] == "synthetic-stub"
    assert warp.get_state()["backend"] == "synthetic-stub"
    warp.close()


def test_warp_bridge_refuses_a_batch_it_never_reduced() -> None:
    """``num_envs`` was decoration, and the real path leaked a batch silently.

    Nothing in the bridge ever split a batch: ``num_envs`` appeared only as a
    label in the stand-in's observation and state dicts. The real path was worse
    than a crash - ``step`` returned the engine's tuple verbatim through a
    ``-> tuple[Any, float, bool, bool, dict]`` signature, so a batched reward and
    a batched terminated flag reached :func:`run_gym_episode` unreduced, where
    ``bool([False, False])`` is ``True``: the episode ended after one step with a
    list recorded as the reward, and nothing raised.
    """
    with pytest.raises(TaskContractError) as excinfo:
        WarpBridge(
            env_name="SurgicalSuture-v0",
            parameters={"num_envs": 16},
            allow_synthetic=True,
            max_steps=2,
        )
    message = str(excinfo.value)
    assert "num_envs=16" in message
    assert "scalar contract" in message
    assert "n_eval_episodes" in message

    with pytest.raises(TaskContractError, match="must be an integer"):
        WarpBridge(env_name="e", parameters={"num_envs": "16"}, allow_synthetic=True)


def test_warp_bridge_refuses_a_batched_step_return() -> None:
    """A real engine that ignores ``num_envs=1`` must be caught at the seam."""

    class BatchedWarpEnv:
        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            del seed, options
            return {"obs": [[0.0], [0.0]]}, {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            del action
            return {"obs": [[0.1], [0.1]]}, [0.5, 1.5], [False, False], [False, False], {}

    bridge = WarpBridge(
        env_name="SurgicalSuture-v0",
        parameters={"num_envs": 1},
        warp_env=BatchedWarpEnv(),
        world_pin="warp-pin-v1",
    )
    with pytest.raises(TaskContractError) as excinfo:
        bridge.step([0.0] * 7)
    assert "returned 2 reward values for one step" in str(excinfo.value)


def test_warp_bridge_unwraps_a_single_env_batch_axis() -> None:
    """One entry out of a one-entry batch invents nothing, so it is unwrapped."""

    class SingleEnvTensorish:
        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            del action
            return {"obs": [[0.1]]}, [2.0], [True], [False], {"real": True}

    bridge = WarpBridge(env_name="e", warp_env=SingleEnvTensorish(), world_pin="p")
    obs, reward, terminated, truncated, info = bridge.step([0.0])
    assert reward == 2.0
    assert terminated is True
    assert truncated is False
    assert obs == {"obs": [[0.1]]}
    assert info == {"real": True}


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


class _GymNameNotFound(Exception):
    """Stands in for ``gymnasium.error.NameNotFound``, which is an optional import.

    Defined once at module level on purpose: the bridges classify a registration
    error by class identity, so building these classes per call would let a test
    raise one class while installing another and silently assert the wrong branch.
    """


#: The fake ``gymnasium.error`` surface, resolved by name in
#: :func:`or_audit.eval.sim.base.missing_world_errors`.
_GYM_ERRORS = _module("gymnasium.error", NameNotFound=_GymNameNotFound)


def test_engine_acquisition_treats_an_unregistered_world_as_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A world that is not registered here is an honest missing-backend condition.

    The fixture used to raise a bare ``ValueError`` labelled "unregistered env",
    which the bridge could not tell apart from a world that *is* registered and
    failed to build - the whole point of the narrowing below. It now raises the
    registration error the message always claimed, so this test covers the case
    it was named for and the misconfiguration case gets its own test.
    """
    task = _sim_task(tmp_path, kind="warp", synthetic_stub=False, gym_id="Isaac-Unknown-v0")

    def _unregistered(env_id: str, **kwargs: Any) -> Any:
        raise _GymNameNotFound(f"unregistered env {env_id}")

    monkeypatch.setattr(
        "or_audit.eval.sim.warp_bridge.module_distribution_version",
        lambda module_name: "",
    )
    monkeypatch.setitem(sys.modules, "warp", _module("warp", __version__="1.9.0"))
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(
        sys.modules, "gymnasium", _module("gymnasium", make=_unregistered, error=_GYM_ERRORS)
    )

    assert _acquire_warp_env(task) == (None, "1.9.0")
    with pytest.raises(TaskContractError, match="synthetic_stub"):
        make_warp_bridge(task)


def test_engine_acquisition_refuses_a_world_it_could_not_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered world that fails to build must not be reported as no backend.

    One bare ``except Exception`` conflated "the vendor runtime is absent" with
    "this task is misconfigured". With ``synthetic_stub = true`` the second case
    silently substituted a non-physical stand-in for the real world the task
    names, and the only trace was a backend field nobody reads.
    """
    task = _sim_task(tmp_path, kind="warp", synthetic_stub=True, gym_id="Isaac-Suture-v0")

    def _broken(env_id: str, **kwargs: Any) -> Any:
        del env_id, kwargs
        raise RuntimeError("no CUDA device")

    monkeypatch.setitem(sys.modules, "warp", _module("warp", __version__="1.9.0"))
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(
        sys.modules, "gymnasium", _module("gymnasium", make=_broken, error=_GYM_ERRORS)
    )

    with pytest.raises(TaskContractError) as excinfo:
        _acquire_warp_env(task)
    message = str(excinfo.value)
    assert "could not be built" in message
    assert "no CUDA device" in message
    # The stub opt-in must not launder a configuration failure into stub numbers.
    with pytest.raises(TaskContractError, match="could not be built"):
        make_warp_bridge(task)


def test_engine_acquisition_does_not_forward_the_harness_step_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_steps`` in ``parameters`` must not reach ``gymnasium.make``.

    Isaac's acquisition popped it; Warp's did not, so a task carrying it made
    ``gymnasium.make`` raise an unexpected-keyword error that the bare ``except``
    swallowed into "no backend" - a correctly-pinned real world silently
    downgraded to a stand-in.
    """
    task = _sim_task(tmp_path, kind="warp", synthetic_stub=True, gym_id="Isaac-Suture-v0")
    assert "max_steps" in task.environment.parameters
    seen: dict[str, Any] = {}

    def _strict_make(env_id: str, **kwargs: Any) -> Any:
        if "max_steps" in kwargs:
            raise TypeError("make() got an unexpected keyword argument 'max_steps'")
        seen.update(kwargs)
        return FakePhysicsEnv()

    monkeypatch.setitem(sys.modules, "warp", _module("warp", __version__="1.9.0"))
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        _module("gymnasium", make=_strict_make, error=_GYM_ERRORS),
    )

    env, version = _acquire_warp_env(task)
    assert isinstance(env, FakePhysicsEnv)
    assert version == "1.9.0"
    assert "max_steps" not in seen


def _sofa_modules(monkeypatch: pytest.MonkeyPatch, make: Any) -> None:
    """Install the optional SOFA import surface so acquisition reaches ``make``."""
    monkeypatch.setitem(sys.modules, "Sofa", _module("Sofa", __version__="24.06.00"))
    monkeypatch.setitem(sys.modules, "SofaRuntime", _module("SofaRuntime"))
    monkeypatch.setitem(sys.modules, "sofagym", _module("sofagym"))
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        _module("gymnasium", make=make, error=_GYM_ERRORS),
    )


def test_sofa_acquisition_treats_an_unregistered_scene_as_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SofaGym registers scenes as gymnasium envs, so an unknown id is a name error."""
    task = _sim_task(tmp_path, kind="sofa", synthetic_stub=False, gym_id="sofa/Unknown-v0")

    def _unregistered(env_id: str, **kwargs: Any) -> Any:
        raise _GymNameNotFound(f"unregistered scene {env_id}")

    _sofa_modules(monkeypatch, _unregistered)
    assert _acquire_sofa_env(task) == (None, "24.06.00")
    with pytest.raises(TaskContractError, match="synthetic_stub"):
        make_sofa_bridge(task)


def test_sofa_acquisition_refuses_a_scene_it_could_not_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SOFA is the wrap target that can host a real hard gate, so silence costs most.

    A registered scene that fails to build is a configuration failure, not an
    absent runtime. Reported as "no backend" under ``synthetic_stub = true`` it
    substituted stand-in numbers for the scene a hard gate was about to score.
    """
    task = _sim_task(tmp_path, kind="sofa", synthetic_stub=True, gym_id="sofa/AneurysmCoiling-v0")

    def _broken(env_id: str, **kwargs: Any) -> Any:
        del env_id, kwargs
        raise RuntimeError("SofaRuntime plugin SofaPython3 failed to load")

    _sofa_modules(monkeypatch, _broken)
    with pytest.raises(TaskContractError) as excinfo:
        _acquire_sofa_env(task)
    message = str(excinfo.value)
    assert "could not be built" in message
    assert "SofaPython3 failed to load" in message
    # The stub opt-in must not launder a configuration failure into stub numbers.
    with pytest.raises(TaskContractError, match="could not be built"):
        make_sofa_bridge(task)


def test_sofa_acquisition_does_not_forward_the_harness_step_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _sim_task(tmp_path, kind="sofa", synthetic_stub=True, gym_id="sofa/AneurysmCoiling-v0")
    assert "max_steps" in task.environment.parameters
    seen: dict[str, Any] = {}

    def _strict_make(env_id: str, **kwargs: Any) -> Any:
        if "max_steps" in kwargs:
            raise TypeError("make() got an unexpected keyword argument 'max_steps'")
        seen.update(kwargs)
        return FakePhysicsEnv()

    _sofa_modules(monkeypatch, _strict_make)
    env, version = _acquire_sofa_env(task)
    assert isinstance(env, FakePhysicsEnv)
    assert version == "24.06.00"
    assert "max_steps" not in seen


def test_sofa_bridge_refuses_a_non_scalar_step_return() -> None:
    """The last bridge returning the engine tuple verbatim through a scalar signature."""

    class BatchedSofaEnv:
        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            del action
            return {"beam_elements": 20}, [0.5, 1.5], [False, False], [False, False], {}

    bridge = SofaBridge(
        scene_name="AneurysmCoiling", sofa_env=BatchedSofaEnv(), world_pin="sofa-pin-v1"
    )
    with pytest.raises(TaskContractError) as excinfo:
        bridge.step({"insertion_step_mm": 1.0})
    assert "returned 2 reward values for one step" in str(excinfo.value)


def test_sofa_bridge_unwraps_a_single_element_step_return() -> None:
    """One entry out of a one-entry sequence invents nothing, so it is unwrapped."""

    class SingleValueSofaEnv:
        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            del action
            return {"beam_elements": 20}, [1.0], [True], [False], {"real": True}

    bridge = SofaBridge(scene_name="s", sofa_env=SingleValueSofaEnv(), world_pin="p")
    _obs, reward, terminated, truncated, info = bridge.step({"insertion_step_mm": 1.0})
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info == {"real": True}


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
    assert provenance["world_pin"] == ""
    # A test double ships no distribution metadata; the field stays empty rather than guessed.
    assert provenance["backend_version"] == ""


def test_gym_factory_does_not_self_attest_the_task_world_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "or_audit.eval.sim.gym_bridge.make_gym",
        lambda spec: FakePhysicsEnv(),
    )
    bridge = make_gym_bridge(load_task(BRONCHO_TASK))
    assert bridge.engine_provenance() == {
        "engine": "gym",
        "backend": "real",
        "backend_version": "",
        "world_pin": "",
    }


def test_pybullet_kind_uses_the_generic_gym_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _sim_task(
        tmp_path,
        kind="pybullet",
        synthetic_stub=False,
        gym_id="NeedleReach-v0",
    )
    monkeypatch.setattr(
        "or_audit.eval.gym_world._make_gymnasium",
        lambda gym_id, *, parameters: FakePhysicsEnv(),
    )
    bridge = make_gym_bridge(task)
    assert bridge.engine_provenance()["engine"] == "pybullet"
    assert bridge.engine_provenance()["world_pin"] == ""


def test_gym_bridge_reports_observed_vcs_pin_and_refuses_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "or_audit.eval.sim.gym_bridge.module_distribution_revision",
        lambda _module: "observed-commit",
    )
    bridge = GymnasiumBridge(FakePhysicsEnv(), world_pin="observed-commit")
    assert bridge.engine_provenance()["world_pin"] == "observed-commit"
    with pytest.raises(TaskContractError, match="world pin mismatch"):
        GymnasiumBridge(FakePhysicsEnv(), world_pin="different-commit")


def test_legacy_gym_prefix_normalizes_reset_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    class LegacyEnv:
        action_space = object()

        def __init__(self) -> None:
            self.seed_value: int | None = None

        def seed(self, seed: int) -> None:
            self.seed_value = seed

        def reset(self) -> dict[str, list[float]]:
            return {"observation": [1.0]}

        def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
            del action
            return {"observation": [2.0]}, 1.0, True, {"is_success": True}

    legacy = LegacyEnv()
    monkeypatch.setitem(sys.modules, "gym", _module("gym", make=lambda *_a, **_kw: legacy))
    from or_audit.eval.gym_world import _make_gymnasium

    env = _make_gymnasium("legacy:surrol.gym:NeedleReach-v0", parameters={})
    assert env.reset(seed=7) == ({"observation": [1.0]}, {})
    assert legacy.seed_value == 7
    assert env.step([0.0]) == (
        {"observation": [2.0]},
        1.0,
        True,
        False,
        {"is_success": True},
    )
    with pytest.raises(TaskContractError, match="cannot acknowledge reset options"):
        env.reset(seed=7, options={"or_audit": {}})


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

    # A stepping world must say so: an engine for an undeclared kind is refused.
    register_simulation_engine(
        "custom-sim",
        lambda task: MockCustomSim(),
        capabilities=WorldCapabilities(physics=True, closed_loop=True),
        provider="custom-sim-test",
    )
    assert "custom-sim" in list_simulation_engines()

    clear_simulation_registry()
    assert len(list_simulation_engines()) == 0
    reset_default_simulation_engines()


def test_a_bridge_with_no_reporter_records_no_observed_world_pin(tmp_path: Path) -> None:
    """An absent provenance reporter observes nothing, and says so in every field.

    This test previously asserted ``world_engine["world_pin"] ==
    "broncho-synthetic-v1"`` for a bridge exposing no ``engine_provenance``:
    ``_engine_provenance`` defaulted the field to ``task.environment.world_pin``,
    so the *declared* pin masqueraded as the *observed* one. It cannot, because
    ``surgeval conformance`` reads ``world_engine.world_pin`` as the evidence
    that the pinned revision is the revision that ran. A declaration is not
    evidence about itself, so an unobserved pin is ``""`` and conformance
    treats it as "cannot verify" rather than "matches".

    The adapter identity still comes from the world-kind registry, which is a
    fact about the installed kernel rather than a bridge's self-report.
    """
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
    # An injected factory exposes no reporter, so nothing about the engine was
    # observed: backend, version, and world pin are all empty rather than
    # optimistic.
    world_engine = config["world_engine"]
    assert world_engine["engine"] == "gym"
    assert world_engine["backend"] == "unknown"
    assert world_engine["backend_version"] == ""
    assert world_engine["world_pin"] == ""
    # The declaration is still on the row, just not as an observation.
    assert res.world_pin == "broncho-synthetic-v1"
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


class _SwappedWorld(BaseSimulationBridge):
    """A world nobody shipped: stands in for a patched or substituted adapter."""

    world_kind = "gym"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del options
        return {"swapped": True}, {"seed": seed}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        return {"swapped": True}, 0.0, True, False, {}


def _make_swapped_world(task: TaskSpec) -> _SwappedWorld:
    del task
    return _SwappedWorld()


def test_replacing_an_engine_factory_repins_the_adapter_identity() -> None:
    """The dispatcher and the attested identity must never disagree.

    Replacing ``gym``'s factory without restating capabilities used to leave the
    registered spec pinning ``make_gym_bridge``, so every job head attested an
    adapter the dispatcher no longer ran - the substitution the digest exists to
    detect, invisible.
    """
    reset_default_simulation_engines()
    before = world_kind_spec("gym")
    assert before is not None
    assert before.adapter_id == "or_audit.eval.sim.gym_bridge:make_gym_bridge"

    register_simulation_engine("gym", _make_swapped_world, override=True)
    after = world_kind_spec("gym")
    assert after is not None
    assert after.adapter_id == f"{__name__}:_make_swapped_world"
    assert after.adapter_digest != before.adapter_digest
    assert after.adapter_identity != before.adapter_identity
    # Capabilities carry over (the world did not change); provenance does not,
    # because "surgeval" no longer published what runs.
    assert after.capabilities == before.capabilities
    assert after.provider == ""

    # Resetting must land back on the shipped identity, not on the swap.
    reset_default_simulation_engines()
    restored = world_kind_spec("gym")
    assert restored is not None
    assert restored.adapter_identity == before.adapter_identity


def test_engine_factory_that_cannot_be_content_pinned_is_refused() -> None:
    """A factory whose behaviour cannot be digested must not be installed."""

    class CallableFactory:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        def __call__(self, task: TaskSpec) -> _SwappedWorld:
            del task
            return _SwappedWorld()

    reset_default_simulation_engines()
    before = list_simulation_engines()["gym"]
    with pytest.raises(TaskContractError, match="cannot be content-pinned"):
        register_simulation_engine("gym", CallableFactory("swap"), override=True)
    # Refused before the dispatcher was touched: the old factory still serves.
    assert list_simulation_engines()["gym"] == before
    reset_default_simulation_engines()


class _BatchedIsaacEnv:
    """A real-shaped Isaac env: every step return carries a leading env axis."""

    def __init__(self, num_envs: int, *, engine_info: dict[str, Any] | None = None) -> None:
        self.num_envs = num_envs
        self.engine_info = dict(engine_info or {})

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del seed, options
        return {"obs": [[0.0]] * self.num_envs}, dict(self.engine_info)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        del action
        return (
            {"obs": [[0.1]] * self.num_envs},
            [0.5 + index for index in range(self.num_envs)],
            [index > 0 for index in range(self.num_envs)],
            [False] * self.num_envs,
            dict(self.engine_info),
        )


def _contact_force_gate() -> GateSpec:
    """The ordinary force gate a task author writes against ``contact_force_n``."""
    return GateSpec(
        id="airway-wall-contact",
        inputs={"contact_force_n": "info.contact_force_n"},
        fail_when="contact_force_n > 1.5",
        maps_to="unsafe",
        kind="force-threshold",
        threshold=1.5,
        unit="N",
        threshold_basis=ThresholdBasis(
            value=1.5,
            unit="N",
            citation="Airway wall contact tolerance per task safety envelope v1",
        ),
    )


class TestIsaacBridgeIsSingleEnvironment:
    """``num_envs != 1`` is refused rather than scalar-coerced.

    The bridge accepted ``parameters['num_envs']``, forwarded it to the engine,
    and then called ``float(reward)`` / ``bool(terminated)`` on whatever came
    back. With ``num_envs=2`` that died on the first step with ``TypeError:
    float() argument must be a string or a real number, not 'list'``, and for a
    sliceable tensor it would have published env 0's step as the step every
    environment took.
    """

    def test_a_batched_env_is_refused_at_construction(self) -> None:
        with pytest.raises(TaskContractError) as excinfo:
            IsaacBridge(
                env_id="Isaac-Lift-Needle-PSM-v0",
                parameters={"num_envs": 2},
                isaac_env=_BatchedIsaacEnv(2),
                world_pin="isaac-pin-v1",
            )
        message = str(excinfo.value)
        assert "num_envs=2" in message
        assert "scalar contract" in message
        assert "n_eval_episodes" in message

    def test_num_envs_one_is_the_only_accepted_batch_width(self) -> None:
        bridge = IsaacBridge(
            env_id="e",
            parameters={"num_envs": 1},
            isaac_env=_BatchedIsaacEnv(1),
            world_pin="p",
        )
        assert bridge.num_envs == 1

    def test_a_non_integer_num_envs_is_refused(self) -> None:
        with pytest.raises(TaskContractError, match="must be an integer"):
            IsaacBridge(env_id="e", parameters={"num_envs": "2"}, allow_synthetic=True)

    def test_a_batched_step_return_is_refused_not_coerced(self) -> None:
        """An engine that ignores ``num_envs=1`` must not slip through.

        This is the reviewer's probe with the constructor satisfied: the refusal
        has to name the batch, not surface as a ``TypeError`` from ``float()``.
        """
        bridge = IsaacBridge(
            env_id="Isaac-Lift-Needle-PSM-v0",
            parameters={"num_envs": 1},
            isaac_env=_BatchedIsaacEnv(2),
            world_pin="isaac-pin-v1",
        )
        bridge.reset(seed=1)
        with pytest.raises(TaskContractError) as excinfo:
            bridge.step([0.0])
        message = str(excinfo.value)
        assert "returned 2 reward values for one step" in message
        assert "num_envs = 1" in message

    def test_a_single_env_batch_axis_is_unwrapped(self) -> None:
        """Isaac returns shape-``(1,)`` tensors even for one env; that unwraps."""
        bridge = IsaacBridge(env_id="e", isaac_env=_BatchedIsaacEnv(1), world_pin="p")
        _obs, reward, terminated, truncated, info = bridge.step([0.0])
        assert reward == 0.5
        assert terminated is False
        assert truncated is False
        assert info["raw_reward"] == 0.5


class TestIsaacBridgeReducesVectorSafetySignals:
    """Vector safety state is reduced to its declared scalar, never renamed.

    ``safety_projection`` copied each allowed alias verbatim onto its canonical
    scalar name, so Isaac's per-body ``net_contact_force`` array landed in
    ``contact_force_n`` as a nested list. A plain ``contact_force_n > 1.5`` gate
    then compared a list to a float, scored ``not_assessable``, and silently
    failed to report a 2 N contact - the exact failure mode a gate exists to
    prevent.
    """

    def test_per_body_contact_force_reduces_to_a_scalar_newton_value(self) -> None:
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection(
            {"net_contact_force": [[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]]}
        )
        assert projected["contact_force_n"] == pytest.approx(2.0)
        assert projected["safety_state_keys"] == ["contact_force_n"]
        # The published number is traceable to the key and rule behind it.
        assert projected["safety_state_reductions"] == {
            "contact_force_n": "net_contact_force:max-norm"
        }

    def test_the_reviewer_probe_now_fails_the_gate_instead_of_abstaining(self) -> None:
        """The whole point: a 2 N contact must score FAIL, not NOT_ASSESSABLE."""
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection(
            {"net_contact_force": [[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]]}
        )
        outcome = evaluate_gate(_contact_force_gate(), {"info": projected})
        assert outcome is not None
        assert outcome.status is GateStatus.FAIL

    def test_a_force_below_the_threshold_still_passes(self) -> None:
        """The reduction must not manufacture failures either."""
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection(
            {"net_contact_force": [[0.2, 0.0, 0.0], [0.0, 0.3, 0.4]]}
        )
        assert projected["contact_force_n"] == pytest.approx(0.5)
        outcome = evaluate_gate(_contact_force_gate(), {"info": projected})
        assert outcome is not None
        assert outcome.status is GateStatus.PASS

    def test_a_single_env_batch_axis_on_the_force_array_is_unwrapped(self) -> None:
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection(
            {"net_contact_force": [[[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]]]}
        )
        assert projected["contact_force_n"] == pytest.approx(2.0)

    def test_already_scalar_magnitudes_are_maxed_not_normed(self) -> None:
        """``contact_force_n`` declares newtons, so norming it would inflate it."""
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection({"contact_force_n": [1.0, 1.0, 1.0]})
        assert projected["contact_force_n"] == pytest.approx(1.0)
        assert projected["safety_state_reductions"] == {"contact_force_n": "contact_force_n:max"}

    def test_per_joint_violation_flags_reduce_to_any(self) -> None:
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection({"joint_limit_violation": [False, False, True]})
        assert projected["workspace_violation"] is True
        assert projected["safety_state_reductions"] == {
            "workspace_violation": "joint_limit_violation:any"
        }

    def test_per_contact_penetration_depths_reduce_to_the_max(self) -> None:
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection({"penetration_depth": [0.01, 0.04, 0.02]})
        assert projected["max_pen"] == pytest.approx(0.04)

    def test_a_scalar_engine_value_is_carried_through_unchanged(self) -> None:
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection({"unsafe": True, "max_pen": 0.03})
        assert projected["unsafe"] is True
        assert projected["max_pen"] == pytest.approx(0.03)

    def test_an_absent_signal_still_stays_absent(self) -> None:
        """Reduction must not become a reason to default a missing signal."""
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        projected = bridge.safety_projection({"raw_success": True})
        assert projected["safety_state_reported"] is False
        assert projected["safety_state_keys"] == []
        assert projected["safety_state_reductions"] == {}
        assert "contact_force_n" not in projected

    def test_the_stand_in_publishes_no_reduced_physics(self) -> None:
        """Scalarization must never give the stub a physical number to report.

        The reduction path only fires on keys the engine actually reported, and
        the stand-in reports none, so a stub run stays honestly unassessable
        rather than becoming a fabricated measurement.
        """
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=2)
        _obs, reset_info = bridge.reset(seed=1)
        _obs, _reward, _term, _trunc, step_info = bridge.step([0.0])
        for info in (reset_info, step_info):
            assert info["safety_state_reported"] is False
            assert info["safety_state_keys"] == []
            assert info["safety_state_reductions"] == {}
            assert not set(SAFETY_KEY_SOURCES) & set(info)

    @pytest.mark.parametrize(
        ("engine_info", "detail"),
        [
            ({"net_contact_force": [[0.2, 0.0], [2.0, 0.0]]}, "3-vector"),
            ({"net_contact_force": [0.5, 1.2]}, "ambiguous"),
            ({"net_contact_force": [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]}, "multi-environment"),
            ({"net_contact_force": []}, "empty"),
            ({"net_contact_force": "hard"}, "not a number"),
            ({"contact_force_n": [[0.5], [1.2]]}, "already-scalar magnitudes"),
            ({"joint_limit_violation": [[False], [True]]}, "multi-environment"),
        ],
    )
    def test_an_unsupported_shape_is_refused_not_published(
        self, engine_info: dict[str, Any], detail: str
    ) -> None:
        bridge = IsaacBridge(env_id="e", allow_synthetic=True, max_steps=1)
        with pytest.raises(TaskContractError) as excinfo:
            bridge.safety_projection(engine_info)
        message = str(excinfo.value)
        assert detail in message
        # The refusal has to say why silence would be worse than the error.
        assert "not_assessable" in message

    def test_the_alias_allowlist_is_unchanged_by_the_reduction_table(self) -> None:
        """Alias precedence is derived from one table; drift would be silent."""
        assert SAFETY_KEY_SOURCES == {
            "max_pen": ("max_pen", "penetration_depth", "max_penetration", "penetration"),
            "contact_force_n": (
                "contact_force_n",
                "contact_force",
                "max_contact_force",
                "net_contact_force",
            ),
            "wall_force_n": ("wall_force_n", "wall_force", "tissue_force_n"),
            "workspace_violation": (
                "workspace_violation",
                "out_of_workspace",
                "joint_limit_violation",
            ),
            "unsafe": ("unsafe", "safety_violation"),
            "diverged": ("diverged", "physics_diverged"),
        }
        assert set(SAFETY_SIGNALS) == set(SAFETY_KEY_SOURCES)


def _make_undeclared_world(task: TaskSpec) -> _SwappedWorld:
    del task
    return _SwappedWorld()


def test_engine_for_an_undeclared_world_kind_is_refused() -> None:
    """An adapter that declares nothing withholds everything.

    Installing a runnable engine for an undeclared kind used to leave
    ``resolve_world_capabilities`` with only the task's own word, so a package
    could hand itself ``physics=True`` (and a physics oracle with it) for a
    world no adapter had granted anything.
    """
    reset_default_simulation_engines()
    with pytest.raises(TaskContractError) as exc:
        register_simulation_engine("undeclared-sim", _make_undeclared_world, override=True)
    assert "has no declared capabilities" in str(exc.value)
    # Nothing lands: neither the dispatcher nor the world-kind registry.
    assert "undeclared-sim" not in list_simulation_engines()
    assert world_kind_spec("undeclared-sim") is None
    reset_default_simulation_engines()


def test_engine_registration_accepts_explicit_or_already_declared_capabilities() -> None:
    """The two honest paths: declare inline, or declare the kind first."""
    capabilities = WorldCapabilities(physics=True, closed_loop=True)

    reset_default_simulation_engines()
    register_simulation_engine(
        "inline-sim",
        _make_undeclared_world,
        capabilities=capabilities,
        provider="inline-test",
    )
    inline = world_kind_spec("inline-sim")
    assert inline is not None
    assert inline.capabilities == capabilities
    assert inline.adapter_id == f"{__name__}:_make_undeclared_world"
    assert inline.provider == "inline-test"

    reset_default_simulation_engines()
    register_world_kind(WorldKindSpec(kind="declared-sim", capabilities=capabilities))
    register_simulation_engine("declared-sim", _make_undeclared_world)
    inherited = world_kind_spec("declared-sim")
    assert inherited is not None
    # Capabilities are inherited from the standing declaration; the identity is
    # pinned to the factory that will actually run.
    assert inherited.capabilities == capabilities
    assert inherited.adapter_id == f"{__name__}:_make_undeclared_world"
    assert len(inherited.adapter_digest) == 64
    reset_default_simulation_engines()
