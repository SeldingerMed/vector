"""The open world-kind extension point (next.md N3 kernel prerequisite).

These tests pin the behaviour that lets a third-party world publish a task
without a kernel release, and the refusals that keep that opening from becoming
a way to self-grant eligibility or to run a swapped adapter unnoticed.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.loader import load_task
from or_audit.eval.runner import builtin_random_agent, run_job
from or_audit.eval.sim import (
    BACKEND_REAL,
    AdapterDiscovery,
    BaseSimulationBridge,
    WorldAdapter,
    register_world_adapter,
    reset_default_simulation_engines,
    world_kind_spec,
)
from or_audit.eval.sim.base import _load_adapter
from or_audit.eval.task import TaskSpec, WorldSpec
from or_audit.eval.worlds import (
    DeterminismClass,
    WorldCapabilities,
    WorldKindSpec,
    adapter_identity,
    determinism_at_least,
    list_world_kinds,
    register_world_kind,
    require_world_kind,
    reset_default_world_kinds,
    resolve_world_capabilities,
    world_kind_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BRONCHO_TASK = REPO_ROOT / "docs/examples/tasks/broncho-airway-nav"
VIDEO_TASK = REPO_ROOT / "docs/examples/tasks/video-nextstep"

THIRD_PARTY_KIND = "steve-sofa"


class _ThirdPartyWorld(BaseSimulationBridge):
    """Minimal third-party world: a guidewire world nobody in-tree knows about."""

    world_kind: WorldKind | str = THIRD_PARTY_KIND

    def __init__(self, *, world_pin: str = "", max_steps: int = 2) -> None:
        self.world_pin = world_pin
        self._max_steps = max_steps
        self._step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del options
        self._step = 0
        return {"tip": 0.0}, {"seed": seed}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        self._step += 1
        done = self._step >= self._max_steps
        info = {
            "target_reached": done,
            "max_contact_force_n": 0.4,
            "safe_navigation": done,
            "diverged": False,
        }
        return {"tip": float(self._step)}, 1.0 if done else 0.0, done, False, info

    def engine_provenance(self) -> dict[str, Any]:
        return {
            "engine": world_kind_key(self.world_kind),
            "backend": BACKEND_REAL,
            "backend_version": "0.9.1",
            "world_pin": self.world_pin,
        }


def _make_third_party(task: TaskSpec) -> _ThirdPartyWorld:
    return _ThirdPartyWorld(
        world_pin=task.environment.world_pin,
        max_steps=int(task.environment.parameters.get("max_steps", 2)),
    )


THIRD_PARTY_CAPABILITIES = WorldCapabilities(
    physics=True,
    closed_loop=True,
    requires_gym_id=True,
    requires_world_pin=True,
    determinism_class=DeterminismClass.TOLERANCE,
)


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    yield
    reset_default_simulation_engines()


def _third_party_task(
    tmp_path: Path,
    *,
    kind: str = THIRD_PARTY_KIND,
    capabilities: bool = False,
    adapter: tuple[str, str] | None = None,
    metrics_only: bool = False,
    drop_gates: bool = False,
) -> Path:
    """Copy the broncho package and repoint its world at a third-party kind."""
    task_dir = tmp_path / "third-party-task"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    shutil.copytree(BRONCHO_TASK, task_dir, ignore=shutil.ignore_patterns("__pycache__"))
    toml_path = task_dir / "task.toml"
    original = toml_path.read_text(encoding="utf-8")
    lines = [
        "[environment]",
        f'kind = "{kind}"',
        'gym_id = "stEVE/ArchVariety-v0"',
        'world_pin = "steve-pin-v1"',
        "parameters = { max_steps = 2 }",
        "n_eval_episodes = 2",
        'seed_policy = "deterministic-eval-2"',
    ]
    if metrics_only:
        lines.append("metrics_only = true")
    if adapter is not None:
        lines.append(f'adapter = "{adapter[0]}"')
        lines.append(f'adapter_digest = "{adapter[1]}"')
    if capabilities:
        lines.extend(
            [
                "[environment.capabilities]",
                "physics = true",
                "closed_loop = true",
                "requires_gym_id = true",
                "requires_world_pin = true",
            ]
        )
    head, _, tail = original.partition("[environment]")
    _, _, rest = tail.partition("\n[interface]")
    body = f"{head}{chr(10).join(lines)}\n\n[interface]{rest}"
    if metrics_only:
        body = body.replace("safety_critical = true", "safety_critical = false")
    if drop_gates or metrics_only:
        body = _strip_gates(body)
    toml_path.write_text(body, encoding="utf-8")
    return task_dir


def _strip_gates(body: str) -> str:
    """Remove every ``[[verifier.gates]]`` block, sub-tables included."""
    out: list[str] = []
    skipping = False
    for line in body.splitlines():
        if line.startswith("[[verifier.gates]]"):
            skipping = True
            continue
        if skipping:
            if line.startswith("[verifier.gates"):
                continue
            if line.startswith("["):
                skipping = False
            else:
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def test_builtin_world_kinds_declare_capabilities_instead_of_enum_sets() -> None:
    kinds = list_world_kinds()
    assert set(kinds) >= {kind.value for kind in WorldKind}
    gym = require_world_kind(WorldKind.GYM)
    assert gym.capabilities.physics is True
    assert gym.capabilities.closed_loop is True
    assert gym.capabilities.requires_gym_id is True
    assert gym.adapter_id == "or_audit.eval.sim.gym_bridge:make_gym_bridge"
    assert len(gym.adapter_digest) == 64
    assert gym.adapter_identity == f"{gym.adapter_id}+{gym.adapter_digest}"
    frame = require_world_kind(WorldKind.FRAME_SOURCE)
    assert frame.capabilities.physics is False
    assert frame.capabilities.closed_loop is False
    # Non-adapter kinds are declared but unattached: nothing steps a frame source.
    assert frame.adapter_identity == "unattached"


def test_isaac_lab_kind_is_served_by_the_isaac_adapter() -> None:
    spec = require_world_kind(WorldKind.ISAAC_LAB)
    assert spec.adapter_id == "or_audit.eval.sim.isaac_bridge:make_isaac_bridge"
    assert spec.provider == "surgeval"


def test_adapter_identity_tracks_module_content() -> None:
    from or_audit.eval.sim import gym_bridge, isaac_bridge

    gym_id, gym_digest = adapter_identity(gym_bridge.make_gym_bridge)
    isaac_id, isaac_digest = adapter_identity(isaac_bridge.make_isaac_bridge)
    assert gym_id != isaac_id
    assert gym_digest != isaac_digest
    assert adapter_identity(gym_bridge.make_gym_bridge) == (gym_id, gym_digest)


def test_third_party_world_publishes_without_a_kernel_release(tmp_path: Path) -> None:
    register_world_adapter(
        WorldAdapter(
            kind=THIRD_PARTY_KIND,
            capabilities=THIRD_PARTY_CAPABILITIES,
            factory=_make_third_party,
            provider="steve-adapter",
        )
    )
    task_dir = _third_party_task(tmp_path)
    task = load_task(task_dir)
    assert task.environment.kind == THIRD_PARTY_KIND
    assert task.environment.kind_key == THIRD_PARTY_KIND
    assert task.environment.resolved_capabilities.closed_loop is True

    result = run_job(
        task=task,
        task_dir=task_dir,
        agent=builtin_random_agent("broncho-steering"),
        agent_dir=None,
        out=tmp_path / "job",
        n=1,
    )
    assert result.world_engine is not None
    assert result.world_engine.engine == THIRD_PARTY_KIND
    assert result.world_engine.backend == BACKEND_REAL
    spec = require_world_kind(THIRD_PARTY_KIND)
    assert result.world_engine.adapter_id == spec.adapter_id
    assert result.world_engine.adapter_digest == spec.adapter_digest


def test_unregistered_world_kind_needs_declared_capabilities(tmp_path: Path) -> None:
    task_dir = _third_party_task(tmp_path)
    with pytest.raises(TaskContractError) as exc:
        load_task(task_dir)
    assert "no installed adapter" in str(exc.value)
    assert "[environment.capabilities]" in str(exc.value)

    declared_dir = _third_party_task(tmp_path, capabilities=True)
    task = load_task(declared_dir)
    assert task.environment.resolved_capabilities.physics is True


def test_task_cannot_grant_itself_eligibility_the_adapter_withholds(tmp_path: Path) -> None:
    register_world_adapter(
        WorldAdapter(
            kind=THIRD_PARTY_KIND,
            # A metrics-only wrap: the world does not support closed-loop stepping.
            capabilities=WorldCapabilities(requires_gym_id=True, requires_world_pin=True),
            factory=_make_third_party,
            provider="steve-adapter",
        )
    )
    task_dir = _third_party_task(tmp_path, capabilities=True)
    with pytest.raises(TaskContractError) as exc:
        load_task(task_dir)
    assert "disagrees with the installed adapter" in str(exc.value)


def test_closed_loop_requires_declared_closed_loop_capability(tmp_path: Path) -> None:
    register_world_adapter(
        WorldAdapter(
            kind=THIRD_PARTY_KIND,
            capabilities=WorldCapabilities(
                physics=True, requires_gym_id=True, requires_world_pin=True
            ),
            factory=_make_third_party,
            provider="steve-adapter",
        )
    )
    task_dir = _third_party_task(tmp_path)
    with pytest.raises(TaskContractError) as exc:
        load_task(task_dir)
    assert "closed-loop capability" in str(exc.value)


def test_declared_adapter_pin_is_verified_at_load(tmp_path: Path) -> None:
    register_world_adapter(
        WorldAdapter(
            kind=THIRD_PARTY_KIND,
            capabilities=THIRD_PARTY_CAPABILITIES,
            factory=_make_third_party,
            provider="steve-adapter",
        )
    )
    spec = require_world_kind(THIRD_PARTY_KIND)
    good = _third_party_task(tmp_path, adapter=(spec.adapter_id, spec.adapter_digest))
    assert load_task(good).environment.adapter == spec.adapter_id

    swapped = _third_party_task(tmp_path, adapter=(spec.adapter_id, "0" * 64))
    with pytest.raises(TaskContractError) as exc:
        load_task(swapped)
    assert "content digest mismatch" in str(exc.value)

    renamed = _third_party_task(tmp_path, adapter=("other.module:make", spec.adapter_digest))
    with pytest.raises(TaskContractError) as exc:
        load_task(renamed)
    assert "is served by" in str(exc.value)


def test_adapter_pin_without_installed_adapter_is_refused(tmp_path: Path) -> None:
    task_dir = _third_party_task(
        tmp_path,
        capabilities=True,
        adapter=("steve_adapter:make_world", "a" * 64),
    )
    with pytest.raises(TaskContractError) as exc:
        load_task(task_dir)
    assert "no adapter is registered" in str(exc.value)


def test_adapter_pin_must_be_complete() -> None:
    with pytest.raises(TaskContractError) as exc:
        WorldSpec(kind=WorldKind.FRAME_SOURCE, adapter="mod:make")
    assert "must be declared together" in str(exc.value)
    with pytest.raises(TaskContractError) as exc:
        WorldSpec(kind=WorldKind.FRAME_SOURCE, adapter="mod:make", adapter_digest="short")
    assert "64 lowercase hex" in str(exc.value)


def test_metrics_only_wrap_refuses_gates_and_is_stamped(tmp_path: Path) -> None:
    register_world_adapter(
        WorldAdapter(
            kind=THIRD_PARTY_KIND,
            capabilities=THIRD_PARTY_CAPABILITIES,
            factory=_make_third_party,
            provider="steve-adapter",
        )
    )
    gated = _third_party_task(tmp_path, metrics_only=False)
    gated_body = (gated / "task.toml").read_text(encoding="utf-8")
    (gated / "task.toml").write_text(
        gated_body.replace("n_eval_episodes = 2", "n_eval_episodes = 2\nmetrics_only = true"),
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError) as exc:
        load_task(gated)
    assert "must not declare hard gates" in str(exc.value)

    honest = _third_party_task(tmp_path, metrics_only=True)
    task = load_task(honest)
    assert task.environment.metrics_only is True
    assert not task.verifier.gates
    result = run_job(
        task=task,
        task_dir=honest,
        agent=builtin_random_agent("broncho-steering"),
        agent_dir=None,
        out=tmp_path / "metrics-only-job",
        n=1,
    )
    assert result.world_engine is not None
    assert result.world_engine.metrics_only is True
    scorecard = json.loads(
        (tmp_path / "metrics-only-job" / "scorecard.json").read_text(encoding="utf-8")
    )
    assert scorecard["metrics_only"] is True
    markdown = (tmp_path / "metrics-only-job" / "scorecard.md").read_text(encoding="utf-8")
    assert "METRICS-ONLY - NOT SAFETY-ATTESTED" in markdown


def test_metrics_only_cannot_stay_safety_critical(tmp_path: Path) -> None:
    task_dir = _third_party_task(tmp_path, capabilities=True, metrics_only=True)
    body = (task_dir / "task.toml").read_text(encoding="utf-8")
    (task_dir / "task.toml").write_text(
        body.replace("safety_critical = false", "safety_critical = true"), encoding="utf-8"
    )
    with pytest.raises(TaskContractError) as exc:
        load_task(task_dir)
    assert "safety_critical" in str(exc.value)


def test_plugin_entry_point_payload_must_be_a_world_adapter() -> None:
    adapter = WorldAdapter(
        kind=THIRD_PARTY_KIND,
        capabilities=THIRD_PARTY_CAPABILITIES,
        factory=_make_third_party,
    )
    assert _load_adapter(adapter, name="steve") is adapter
    assert _load_adapter(lambda: adapter, name="steve") is adapter
    with pytest.raises(TaskContractError) as exc:
        _load_adapter({"kind": THIRD_PARTY_KIND}, name="steve")
    assert "must resolve to a WorldAdapter" in str(exc.value)


def test_failed_plugin_discovery_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    from or_audit.eval.sim import base

    class _BrokenEntry:
        name = "broken-world"

        def load(self) -> Any:
            raise ImportError("no module named 'steve'")

    monkeypatch.setattr(base, "entry_points", lambda group: (_BrokenEntry(),))
    report = base.discover_world_adapters()
    assert report == (
        AdapterDiscovery(name="broken-world", ok=False, error="no module named 'steve'"),
    )
    task = load_task(VIDEO_TASK)
    with pytest.raises(TaskContractError) as exc:
        base.require_simulation_engine(task)
    assert "failed world-kind plugins: broken-world" in str(exc.value)


def test_working_plugin_discovery_registers_and_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_audit.eval.sim import base

    class _Entry:
        name = "steve"

        def load(self) -> Any:
            return WorldAdapter(
                kind=THIRD_PARTY_KIND,
                capabilities=THIRD_PARTY_CAPABILITIES,
                factory=_make_third_party,
                provider="steve-adapter",
            )

    monkeypatch.setattr(base, "entry_points", lambda group: (_Entry(),))
    report = base.discover_world_adapters(override=True)
    assert len(report) == 1
    discovery = report[0]
    assert discovery.ok is True
    assert discovery.kind == THIRD_PARTY_KIND
    assert discovery.provider == "steve-adapter"
    assert discovery.adapter_identity.endswith(require_world_kind(THIRD_PARTY_KIND).adapter_digest)


def test_capability_resolution_and_determinism_ordering() -> None:
    reset_default_world_kinds()
    declared = WorldCapabilities(physics=True, closed_loop=True)
    assert resolve_world_capabilities(WorldKind.FRAME_SOURCE, None).physics is False
    with pytest.raises(TaskContractError):
        resolve_world_capabilities(WorldKind.FRAME_SOURCE, declared)
    assert resolve_world_capabilities("brand-new-world", declared) is declared
    assert world_kind_spec("brand-new-world") is None

    assert determinism_at_least(DeterminismClass.BITWISE, DeterminismClass.TOLERANCE) is True
    assert determinism_at_least(DeterminismClass.TOLERANCE, DeterminismClass.BITWISE) is False
    assert (
        determinism_at_least(DeterminismClass.NONDETERMINISTIC, DeterminismClass.UNMEASURED)
        is False
    )
    assert determinism_at_least(DeterminismClass.UNMEASURED, DeterminismClass.UNMEASURED) is True


class _UnpinnableFactory:
    """A callable object: no importable qualname, no source file of its own.

    Two instances build different worlds but share a class, so no identity
    derived from names can tell them apart.
    """

    def __init__(self, world_pin: str) -> None:
        self.world_pin = world_pin

    def __call__(self, task: TaskSpec | None = None) -> _ThirdPartyWorld:
        del task
        return _ThirdPartyWorld(world_pin=self.world_pin)


def test_unidentifiable_adapter_factory_is_refused() -> None:
    """A factory that cannot be content-pinned must refuse, not mint a name hash.

    Hashing the fallback ``"anonymous"`` gave both of these the same identity,
    so swapping one for the other moved no job head and passed every adapter pin
    check - a digest that pins nothing is worse than no digest.
    """
    left, right = _UnpinnableFactory("pin-a"), _UnpinnableFactory("pin-b")
    # Same class, different behaviour: an identity built from names cannot
    # distinguish the two, so it must not claim to.
    assert left().world_pin != right().world_pin
    for factory in (left, right):
        with pytest.raises(TaskContractError) as exc:
            adapter_identity(factory)
        assert "cannot be content-pinned" in str(exc.value)
        assert "importable module:qualname" in str(exc.value)


def test_unpinnable_adapter_is_not_registered() -> None:
    """The refusal fires before the world kind exists, so nothing half-lands."""
    reset_default_world_kinds()
    with pytest.raises(TaskContractError, match="cannot be content-pinned"):
        register_world_adapter(
            WorldAdapter(
                kind=THIRD_PARTY_KIND,
                capabilities=THIRD_PARTY_CAPABILITIES,
                factory=_UnpinnableFactory("world-a"),
                provider="steve-adapter",
            )
        )
    assert world_kind_spec(THIRD_PARTY_KIND) is None


def test_unpinnable_plugin_is_a_failed_discovery_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_audit.eval.sim import base

    class _Entry:
        name = "steve"

        def load(self) -> Any:
            return WorldAdapter(
                kind=THIRD_PARTY_KIND,
                capabilities=THIRD_PARTY_CAPABILITIES,
                factory=_UnpinnableFactory("world-a"),
                provider="steve-adapter",
            )

    reset_default_world_kinds()
    monkeypatch.setattr(base, "entry_points", lambda group: (_Entry(),))
    report = base.discover_world_adapters(override=True)
    assert len(report) == 1
    assert report[0].ok is False
    assert "cannot be content-pinned" in report[0].error
    assert world_kind_spec(THIRD_PARTY_KIND) is None


def test_registry_and_task_normalize_a_kind_the_same_way() -> None:
    """A plugin kind must be reachable from the task that names it.

    ``WorldSpec`` folds ``steve_sofa`` to ``steve-sofa``; the registry used to
    store the author's spelling verbatim, so a registered kind resolved to
    ``None`` from its own task and both the capability gate and the engine
    dispatcher silently missed it.
    """
    reset_default_world_kinds()
    capabilities = WorldCapabilities(physics=True, closed_loop=True)
    spec = WorldKindSpec(kind="steve_sofa", capabilities=capabilities)
    assert spec.kind == "steve-sofa"
    register_world_kind(spec, override=True)

    world = WorldSpec(kind="steve_sofa", capabilities=capabilities)
    assert world.kind_key == "steve-sofa"
    resolved = world_kind_spec(world.kind)
    assert resolved is not None
    assert resolved.kind == "steve-sofa"
    # Either spelling reaches the same entry, from either direction.
    assert require_world_kind("steve_sofa") is resolved
    assert require_world_kind("steve-sofa") is resolved
    assert world_kind_key("steve_sofa") == world_kind_key("steve-sofa") == "steve-sofa"
    assert list(list_world_kinds()).count("steve_sofa") == 0
