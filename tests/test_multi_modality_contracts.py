"""Tests for multi-modality contracts, GateKind extensions, and ModalityAdapter."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.adapters.base import (
    BaseModalityAdapter,
    ModalityAdapter,
    adapter_revision,
    clear_registry,
    get_adapter,
    list_adapters,
    register_adapter,
    require_adapter,
    reset_default_adapters,
)
from or_audit.eval.contracts import (
    CapabilitySpec,
    InteractionMode,
    InterfaceSpec,
    StreamSpec,
)
from or_audit.eval.enums import GateKind, ModalityKind
from or_audit.eval.task import GateSpec, TaskMetadata, ThresholdBasis


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset_default_adapters()
    yield
    reset_default_adapters()


def test_modality_kind_enum_values() -> None:
    assert str(ModalityKind.VIDEO_LAPAROSCOPIC) == "video-laparoscopic"
    assert str(ModalityKind.VIDEO_ENDOSCOPIC) == "video-endoscopic"
    assert str(ModalityKind.AIRWAY_BRONCHOSCOPY) == "airway-bronchoscopy"
    assert str(ModalityKind.FLUOROSCOPY_DSA) == "fluoroscopy-dsa"
    assert str(ModalityKind.ORTHOPEDIC_POINTCLOUD) == "orthopedic-pointcloud"
    assert str(ModalityKind.ROBOTIC_KINEMATICS) == "robotic-kinematics"
    assert str(ModalityKind.ENDOVASCULAR_SIM) == "endovascular-sim"
    assert str(ModalityKind.SYNTHETIC_PROCEDURAL) == "synthetic-procedural"


def test_gate_kind_enum_values() -> None:
    assert str(GateKind.SPATIAL_EXCLUSION) == "spatial-exclusion"
    assert str(GateKind.FORCE_THRESHOLD) == "force-threshold"
    assert str(GateKind.PERFORATION_RISK) == "perforation-risk"
    assert str(GateKind.RADIATION_DOSE) == "radiation-dose"
    assert str(GateKind.TEMPORAL_BOUND) == "temporal-bound"
    assert str(GateKind.CUSTOM) == "custom"


def test_gate_spec_with_kind_and_threshold() -> None:
    gate = GateSpec(
        id="critical_view_safety",
        inputs={"distance_to_cbd": "oracle.distance_to_cbd"},
        fail_when="distance_to_cbd < 2.0",
        maps_to="cbd_injury_risk",
        kind=GateKind.SPATIAL_EXCLUSION,
        threshold=2.0,
        unit="mm",
        threshold_basis=ThresholdBasis(
            value=2.0, unit="mm", citation="CBD clearance normative bound v1"
        ),
    )
    assert gate.kind == GateKind.SPATIAL_EXCLUSION
    assert gate.threshold == 2.0
    assert gate.unit == "mm"


def test_gate_spec_defaults() -> None:
    gate = GateSpec(id="default_gate", source="info.unsafe", fail_when="unsafe == true")
    assert gate.kind == GateKind.CUSTOM
    assert gate.threshold is None
    assert gate.unit == ""


def test_interface_and_capability_modalities() -> None:
    stream = StreamSpec(
        id="laparoscopic-video",
        schema_id="stereo-rgb",
        adapter="video-laparoscopic",
        adapter_digest="a" * 64,
    )
    interface = InterfaceSpec(
        id="laparoscopic-action",
        interaction_mode=InteractionMode.CLOSED_LOOP,
        observations=("stereo-rgb",),
        actions=("tool-pose",),
        streams=(stream,),
    )
    assert interface.streams[0].adapter == "video-laparoscopic"
    assert interface.streams[0].schema_id == "stereo-rgb"

    matching_cap = CapabilitySpec(
        interface="laparoscopic-action",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("stereo-rgb",),
        actions=("tool-pose",),
        modalities=("video-laparoscopic",),
    )
    assert matching_cap.satisfies(interface)

    mismatched_cap = CapabilitySpec(
        interface="laparoscopic-action",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("stereo-rgb",),
        actions=("tool-pose",),
        modalities=(ModalityKind.FLUOROSCOPY_DSA.value,),
    )
    assert not mismatched_cap.satisfies(interface)

    wildcard_cap = CapabilitySpec(
        interface="laparoscopic-action",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        schema_wildcard=True,
    )
    assert wildcard_cap.satisfies(interface)


def test_interface_duplicate_stream_id_rejected() -> None:
    stream = StreamSpec(
        id="dup",
        schema_id="obs",
        adapter="video-laparoscopic",
        adapter_digest="a" * 64,
    )
    with pytest.raises(TaskContractError, match="duplicate stream id"):
        InterfaceSpec(
            id="i",
            interaction_mode=InteractionMode.SINGLE_TURN,
            observations=("obs",),
            outputs=("out",),
            streams=(stream, stream),
        )


def test_interface_stream_schema_must_be_declared() -> None:
    stream = StreamSpec(
        id="s",
        schema_id="phantom",
        adapter="video-laparoscopic",
        adapter_digest="a" * 64,
    )
    with pytest.raises(TaskContractError, match="not among the declared"):
        InterfaceSpec(
            id="i",
            interaction_mode=InteractionMode.SINGLE_TURN,
            observations=("obs",),
            outputs=("out",),
            streams=(stream,),
        )


def test_interface_interactive_streams_allowed(tmp_path: Path) -> None:
    # E5: the interactive agent route applies the same pinned stream
    # preprocessing as every other mode, so declaring streams is legal.
    # End-to-end behavior (agent sees composed turns, trace keeps raw
    # observations) is covered by test_interactive_turns_use_stream_pipeline.
    stream = StreamSpec(
        id="s",
        schema_id="obs",
        adapter="video-laparoscopic",
        adapter_digest="a" * 64,
    )
    interface = InterfaceSpec(
        id="i",
        interaction_mode=InteractionMode.INTERACTIVE,
        observations=("obs",),
        outputs=("out",),
        streams=(stream,),
    )
    assert interface.streams[0].adapter == "video-laparoscopic"


def test_task_metadata_modality() -> None:
    meta = TaskMetadata(
        title="Cholecystectomy Phase Recognition",
        modality=ModalityKind.VIDEO_LAPAROSCOPIC.value,
        tags=("laparoscopy", "phase"),
    )
    assert meta.modality == "video-laparoscopic"


class DummyBronchoAdapter(ModalityAdapter):
    modality: ModalityKind | str = ModalityKind.AIRWAY_BRONCHOSCOPY

    def validate_observation(self, observation: Any) -> bool:
        return isinstance(observation, dict) and "camera_frame" in observation


def test_adapter_registry() -> None:
    clear_registry()
    assert isinstance(DummyBronchoAdapter(), BaseModalityAdapter)
    register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, DummyBronchoAdapter)
    assert "airway-bronchoscopy" in list_adapters()

    adapter = get_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY)
    assert adapter is not None
    assert isinstance(adapter, DummyBronchoAdapter)
    assert adapter.validate_observation({"camera_frame": [1, 2, 3]})
    assert not adapter.validate_observation("invalid")

    # Duplicate registration raises without override
    with pytest.raises(TaskContractError, match="already registered"):
        register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, DummyBronchoAdapter)

    # Override works
    register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, DummyBronchoAdapter, override=True)

    # Require adapter succeeds when registered
    req = require_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY)
    assert isinstance(req, DummyBronchoAdapter)

    # Require adapter fails when missing
    with pytest.raises(TaskContractError, match="unknown modality"):
        require_adapter(ModalityKind.ORTHOPEDIC_POINTCLOUD)


def test_adapter_revision_returns_pin() -> None:
    assert adapter_revision("video-laparoscopic") == (
        "1181d4f61120cd7a78a1d4917dae3d75d68f653918122fe0adda86ea9ee41c2c"
    )
    # Unknown / unpinned plugins report an empty revision.
    assert adapter_revision("no-such-plugin") == ""
    clear_registry()
    assert adapter_revision("video-laparoscopic") == ""


def test_missing_stream_source_is_contract_error() -> None:
    from or_audit.eval.runner import _get_source

    with pytest.raises(TaskContractError, match="not present in observation"):
        _get_source({"camera": {}}, "/feed/frame")
    with pytest.raises(TaskContractError, match="not present in observation"):
        _get_source({"a": 1}, "deep.missing.key")


def test_multi_stream_preprocessing_composes_by_stream_id() -> None:
    import types

    from or_audit.eval.contracts import StreamSpec
    from or_audit.eval.runner import preprocess_observation

    class Vid(ModalityAdapter):
        def preprocess_observation(self, observation: Any) -> Any:
            return {"frames": observation.get("clip", [])}

    class Kin(ModalityAdapter):
        def preprocess_observation(self, observation: Any) -> Any:
            return observation

    clear_registry()
    register_adapter("my-video", Vid)
    register_adapter("my-kin", Kin)
    task = InterfaceSpec(
        id="multi",
        interaction_mode=InteractionMode.SINGLE_TURN,
        observations=("video-clip", "joint-obs"),
        outputs=("pred",),
        streams=(
            StreamSpec(
                id="cam",
                schema_id="video-clip",
                adapter="my-video",
                adapter_digest="a" * 64,
            ),
            StreamSpec(
                id="kin",
                schema_id="joint-obs",
                adapter="my-kin",
                adapter_digest="b" * 64,
                source="joint",
            ),
        ),
    )
    adapters = {"cam": Vid(), "kin": Kin()}
    from typing import cast

    from or_audit.eval.task import TaskSpec

    task_like = cast(TaskSpec, types.SimpleNamespace(interface=task))
    out = preprocess_observation(
        task_like, adapters, {"clip": [1, 2, 3], "joint": {"x": 1}, "other": 9}
    )
    # Each stream's processed slice lands under its own stream id, untouched
    # by the other stream's adapter.
    assert out["cam"] == {"frames": [1, 2, 3]}
    assert out["kin"] == {"x": 1}
    assert "other" not in out


def test_manifest_rejects_tampered_binding() -> None:
    """A plugin whose binding digest differs from its pin must be rejected."""
    from or_audit.eval.adapters.manifest import (
        BUNDLED_ADAPTER_PLUGINS,
        bootstrap_adapter_plugins,
    )

    entry = dict(BUNDLED_ADAPTER_PLUGINS[0])
    entry["sha256"] = "g" * 64  # tamper the declared pin
    clear_registry()
    with pytest.raises(TaskContractError, match="binding digest mismatch"):
        bootstrap_adapter_plugins((entry,))
    # The tampered plugin must not register.
    assert list_adapters() == {}


def test_stream_pin_verification_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import types
    from typing import cast

    from or_audit.eval import loader
    from or_audit.eval.task import TaskSpec

    task = cast(
        TaskSpec,
        types.SimpleNamespace(
            id="t",
            interface=InterfaceSpec(
                id="iface",
                interaction_mode=InteractionMode.SINGLE_TURN,
                observations=("obs",),
                outputs=("out",),
                streams=(
                    StreamSpec(
                        id="s",
                        schema_id="obs",
                        adapter="video-laparoscopic",
                        adapter_digest="0" * 64,
                    ),
                ),
            ),
        ),
    )
    # Registered adapter pin matches the stream pin -> loads clean.
    monkeypatch.setattr("or_audit.eval.adapters.adapter_revision", lambda plugin: "0" * 64)
    loader._verify_streams(task)

    # Registry pin differs -> content digest mismatch.
    monkeypatch.setattr("or_audit.eval.adapters.adapter_revision", lambda plugin: "1" * 64)
    with pytest.raises(TaskContractError, match="content digest mismatch"):
        loader._verify_streams(task)

    # Unpinned/unknown adapter -> contract error, never silent acceptance.
    monkeypatch.setattr("or_audit.eval.adapters.adapter_revision", lambda plugin: "")
    with pytest.raises(TaskContractError, match="unknown or unpinned"):
        loader._verify_streams(task)


def test_gate_spec_normalization() -> None:
    # Snake case gets normalized to kebab case enum
    gate = GateSpec(
        id="force_gate",
        kind="force_threshold",
        threshold=1.5,
        unit="N",
        inputs={"force": "info.force"},
        fail_when="force > 1.5",
        threshold_basis=ThresholdBasis(value=1.5, unit="N", citation="surgical force bound v1"),
    )
    assert gate.kind == GateKind.FORCE_THRESHOLD

    # Custom string remains custom slug
    custom_gate = GateSpec(
        id="my_gate",
        kind="my-custom-kind",
        source="info.x",
        fail_when="x == true",
    )
    assert custom_gate.kind == "my-custom-kind"


def test_adapter_safety_extraction_defensive() -> None:
    adapter = ModalityAdapter()
    # None or non-dict input returns empty dict
    assert adapter.extract_safety_state(None) == {}
    assert adapter.extract_safety_state({}) == {}
    # None values inside context are handled gracefully
    assert adapter.extract_safety_state({"info": None, "safety": None}) == {}
    # Extracts known keys from info into safety
    result = adapter.extract_safety_state(
        {"info": {"max_pen": 0.05, "other_metric": 123}, "safety": {"existing": True}}
    )
    assert result == {"existing": True, "max_pen": 0.05}


def test_adapter_instance_registration() -> None:
    instance = DummyBronchoAdapter()
    register_adapter("custom-broncho", instance)
    retrieved = get_adapter("custom-broncho")
    assert retrieved is instance


def test_eval_init_exports() -> None:
    import or_audit.eval as eval_module

    assert hasattr(eval_module, "ModalityKind")
    assert hasattr(eval_module, "GateKind")
    assert hasattr(eval_module, "GateSpec")
    assert hasattr(eval_module, "BaseModalityAdapter")
    assert hasattr(eval_module, "ModalityAdapter")
    assert hasattr(eval_module, "register_adapter")
    assert hasattr(eval_module, "get_adapter")


_ECHO_PREDICTOR = """
from pathlib import Path
from typing import Any

class Predictor:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"echo_turn": item.get("turn")}

def load_predictor(*, root: Path, weights_path: Path) -> Predictor:
    del root, weights_path
    return Predictor()
"""

_VIDEO_ADAPTER_DIGEST = "1181d4f61120cd7a78a1d4917dae3d75d68f653918122fe0adda86ea9ee41c2c"


def test_interactive_turns_use_stream_pipeline(tmp_path: Path) -> None:
    """E5: the agent sees the composed turn; history and trace keep raw."""
    import json
    import shutil
    from pathlib import Path as _Path

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = _Path(__file__).resolve().parents[1]
    task_dir = tmp_path / "stream-task"
    shutil.copytree(root / "docs/examples/tasks/video-nextstep", task_dir)
    text = (task_dir / "task.toml").read_text(encoding="utf-8")
    text = text.replace('interaction_mode = "single-turn"', 'interaction_mode = "interactive"')
    text = text.replace('observations = ["video-clip"]', 'observations = ["frame"]')
    text = text.replace('kinds = ["frozen-model", "vlm"]', 'kinds = ["policy"]')
    text += (
        '\n[[interface.streams]]\nid = "cam"\nschema_id = "frame"\n'
        'adapter = "video-laparoscopic"\nadapter_digest = "'
        + _VIDEO_ADAPTER_DIGEST
        + '"\nsource = "$"\n'
    )
    (task_dir / "task.toml").write_text(text, encoding="utf-8")
    (task_dir / "inputs.json").write_text(
        json.dumps({"items": [{"id": "clip-001", "turns": [{"frame_index": 7}]}]}),
        encoding="utf-8",
    )
    (task_dir / "labels.json").write_text(
        json.dumps(
            {"items": [{"id": "clip-001", "next_step": "x", "outcome": "y", "unsafe": False}]}
        ),
        encoding="utf-8",
    )
    agent_dir = tmp_path / "echo-agent"
    agent_dir.mkdir()
    (agent_dir / "agent.toml").write_text(
        'format_version = "2"\nid = "test/echo"\nagent_version = "0"\nkind = "policy"\n'
        'weights_pin = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"\n'
        'weights_path = "weights.json"\n'
        '[[capabilities]]\ninterface = "video-predict"\ninteraction_modes = ["interactive"]\n'
        'protocol_versions = ["1"]\nobservations = ["frame"]\noutputs = ["next-step"]\n'
        'modalities = ["video-laparoscopic"]\n'
        'features = ["reasoning", "abstention"]\n'
        '[runtime]\nkind = "local"\nprotocol_version = "1"\n'
        'entrypoint = "predictor.py:load_predictor"\ntimeout_sec = 60.0\n',
        encoding="utf-8",
    )
    (agent_dir / "predictor.py").write_text(_ECHO_PREDICTOR, encoding="utf-8")
    (agent_dir / "weights.json").write_text("{}", encoding="utf-8")
    result = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(agent_dir),
        agent_dir=agent_dir,
        out=tmp_path / "job",
        n=1,
    )
    [trial] = result.trials
    [step] = trial.trajectory.root
    seen = step.output["echo_turn"]
    assert seen["cam"]["frame_index"] == 7
    assert step.observation == {"frame_index": 7}


class _DropSecretAdapter(BaseModalityAdapter):
    modality = "test-drop"

    def preprocess_observation(self, observation: object) -> object:
        assert isinstance(observation, dict)
        return {k: v for k, v in observation.items() if k != "secret"}


_HISTORY_ECHO = """
from pathlib import Path
from typing import Any

class Predictor:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"turn": item.get("turn"), "history": list(item.get("history", []))}

def load_predictor(*, root: Path, weights_path: Path) -> Predictor:
    del root, weights_path
    return Predictor()
"""


def test_interactive_history_never_carries_raw_fields(tmp_path: Path) -> None:
    """Two-turn secret drop: raw `secret` reaches neither the current turn
    nor history, while trace evidence keeps it."""
    import json
    import shutil
    from pathlib import Path as _Path

    from or_audit.eval.adapters.base import register_adapter
    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    register_adapter("test-drop", _DropSecretAdapter, digest="12" * 32, override=True)
    root = _Path(__file__).resolve().parents[1]
    task_dir = tmp_path / "drop-task"
    shutil.copytree(root / "docs/examples/tasks/video-nextstep", task_dir)
    text = (task_dir / "task.toml").read_text(encoding="utf-8")
    text = text.replace('interaction_mode = "single-turn"', 'interaction_mode = "interactive"')
    text = text.replace('observations = ["video-clip"]', 'observations = ["frame"]')
    text = text.replace("max_steps = 1", "max_steps = 3")
    text = text.replace('kinds = ["frozen-model", "vlm"]', 'kinds = ["policy"]')
    text += (
        '\n[[interface.streams]]\nid = "cam"\nschema_id = "frame"\n'
        'adapter = "test-drop"\nadapter_digest = "' + "12" * 32 + '"\nsource = "$"\n'
    )
    (task_dir / "task.toml").write_text(text, encoding="utf-8")
    (task_dir / "inputs.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "clip-001",
                        "turns": [
                            {"frame_index": 7, "secret": "s1"},
                            {"frame_index": 8, "secret": "s2"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "labels.json").write_text(
        json.dumps(
            {"items": [{"id": "clip-001", "next_step": "x", "outcome": "y", "unsafe": False}]}
        ),
        encoding="utf-8",
    )
    agent_dir = tmp_path / "hist-agent"
    agent_dir.mkdir()
    (agent_dir / "agent.toml").write_text(
        'format_version = "2"\nid = "test/hist"\nagent_version = "0"\nkind = "policy"\n'
        'weights_pin = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"\n'
        'weights_path = "weights.json"\n'
        '[[capabilities]]\ninterface = "video-predict"\ninteraction_modes = ["interactive"]\n'
        'protocol_versions = ["1"]\nobservations = ["frame"]\noutputs = ["next-step"]\n'
        'modalities = ["test-drop"]\nfeatures = ["reasoning", "abstention"]\n'
        '[runtime]\nkind = "local"\nprotocol_version = "1"\n'
        'entrypoint = "predictor.py:load_predictor"\ntimeout_sec = 60.0\n',
        encoding="utf-8",
    )
    (agent_dir / "predictor.py").write_text(_HISTORY_ECHO, encoding="utf-8")
    (agent_dir / "weights.json").write_text("{}", encoding="utf-8")
    result = run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(agent_dir),
        agent_dir=agent_dir,
        out=tmp_path / "job",
        n=1,
    )
    [trial] = result.trials
    first, second = trial.trajectory.root
    for step in (first, second):
        assert "secret" not in json.dumps(step.output["turn"])
        assert "secret" in json.dumps(step.observation)
    hist_seen = second.output["history"][0]["observation"]
    assert "secret" not in json.dumps(hist_seen)
    assert hist_seen["cam"]["frame_index"] == 7


def test_semantic_stream_profile_mismatches_refuse_binding() -> None:
    base_stream = StreamSpec(
        id="kinematics-stream",
        schema_id="kinematic-telemetry",
        adapter="robotic-kinematics",
        adapter_digest="a" * 64,
        unit="mm",
        coordinate_frame="world",
        controller_id="delta_pos_v1",
        joint_order=("j1", "j2", "j3"),
        dtype="float32",
        shape=(3,),
    )
    interface = InterfaceSpec(
        id="kinematics-control",
        interaction_mode=InteractionMode.CLOSED_LOOP,
        observations=("kinematic-telemetry",),
        actions=("joint-cmd",),
        streams=(base_stream,),
    )

    # 1. Identical semantic profile -> binds
    matching_cap = CapabilitySpec(
        interface="kinematics-control",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("kinematic-telemetry",),
        actions=("joint-cmd",),
        modalities=("robotic-kinematics",),
        stream_profiles=(base_stream,),
    )
    assert matching_cap.satisfies(interface)

    # 2. Unit mismatch ("mm" vs "m") -> does not bind
    unit_mismatch = matching_cap.model_copy(
        update={"stream_profiles": (base_stream.model_copy(update={"unit": "m"}),)}
    )
    assert not unit_mismatch.satisfies(interface)

    # 3. Coordinate frame mismatch ("world" vs "tool_tip") -> does not bind
    frame_mismatch = matching_cap.model_copy(
        update={
            "stream_profiles": (base_stream.model_copy(update={"coordinate_frame": "tool_tip"}),)
        }
    )
    assert not frame_mismatch.satisfies(interface)

    # 4. Controller mismatch -> does not bind
    controller_mismatch = matching_cap.model_copy(
        update={
            "stream_profiles": (
                base_stream.model_copy(update={"controller_id": "absolute_pos_v1"}),
            )
        }
    )
    assert not controller_mismatch.satisfies(interface)

    # 5. Joint order mismatch -> does not bind
    joint_mismatch = matching_cap.model_copy(
        update={
            "stream_profiles": (base_stream.model_copy(update={"joint_order": ("j3", "j2", "j1")}),)
        }
    )
    assert not joint_mismatch.satisfies(interface)

    # 6. Privileged stream rejected unless accepts_privileged=True
    priv_stream = base_stream.model_copy(update={"privileged": True})
    priv_interface = interface.model_copy(update={"streams": (priv_stream,)})
    assert not matching_cap.satisfies(priv_interface)
    priv_cap = matching_cap.model_copy(update={"accepts_privileged": True})
    assert priv_cap.satisfies(priv_interface)

    # 7. Missing stream profile entirely when interface declares semantics -> does not bind
    no_profile_cap = matching_cap.model_copy(update={"stream_profiles": ()})
    assert not no_profile_cap.satisfies(interface)

    # 8. Empty string unit on capability profile when interface declares unit="mm" -> does not bind
    empty_unit_cap = matching_cap.model_copy(
        update={"stream_profiles": (base_stream.model_copy(update={"unit": ""}),)}
    )
    assert not empty_unit_cap.satisfies(interface)

    # 9. schema_wildcard=True cannot bypass semantic profile requirements
    wildcard_no_profile = CapabilitySpec(
        interface="kinematics-control",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        schema_wildcard=True,
        stream_profiles=(),
    )
    assert not wildcard_no_profile.satisfies(interface)

    # 10. schema_wildcard=True with matching semantic profile satisfies
    wildcard_matching_profile = CapabilitySpec(
        interface="kinematics-control",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        schema_wildcard=True,
        stream_profiles=(base_stream,),
    )
    assert wildcard_matching_profile.satisfies(interface)

    # 11. Schema-keyed lookup must not let a *different channel's* profile
    # vouch for this channel. Interface declares two channels whose stream ids
    # and schema names cross-reference each other (legal: both namespaces are
    # open slugs). Under a merged id/schema index, B's profile is reachable
    # through A's schema name, so A would bind on B's semantics.
    chan_a = StreamSpec(
        id="cam",
        schema_id="probe-schema",
        adapter="robotic-kinematics",
        adapter_digest="a" * 64,
        unit="mm",
        coordinate_frame="world",
    )
    chan_b = StreamSpec(
        id="probe",
        schema_id="cam",
        adapter="robotic-kinematics",
        adapter_digest="a" * 64,
        unit="mm",
        coordinate_frame="world",
    )
    crossed = InterfaceSpec(
        id="kinematics-control",
        interaction_mode=InteractionMode.CLOSED_LOOP,
        observations=("probe-schema", "cam"),
        actions=("joint-cmd",),
        streams=(chan_a, chan_b),
    )
    # Capability declares only B's profile (correct for B). It carries no
    # declaration for A at all, so A must refuse to bind.
    only_b = CapabilitySpec(
        interface="kinematics-control",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("probe-schema", "cam"),
        actions=("joint-cmd",),
        modalities=("robotic-kinematics",),
        stream_profiles=(chan_b,),
    )
    assert not only_b.satisfies(crossed)

    # 11b. Legitimate schema-keyed fallback: exactly one profile carries this
    # channel's schema and none of the interface's other channel ids collides
    # with it, so the profile is unambiguously about this channel.
    solo = InterfaceSpec(
        id="kinematics-control",
        interaction_mode=InteractionMode.CLOSED_LOOP,
        observations=("kinematic-telemetry",),
        actions=("joint-cmd",),
        streams=(base_stream.model_copy(update={"id": "an-alternate-channel-name"}),),
    )
    schema_keyed_cap = CapabilitySpec(
        interface="kinematics-control",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("kinematic-telemetry",),
        actions=("joint-cmd",),
        modalities=("robotic-kinematics",),
        stream_profiles=(base_stream,),
    )
    assert schema_keyed_cap.satisfies(solo)

    # 11c. Two profiles competing for one schema make the schema-keyed route
    # ambiguous; guessing either would bind on an unverified declaration.
    twin = base_stream.model_copy(update={"id": "other-channel-name"})
    assert not schema_keyed_cap.model_copy(
        update={"stream_profiles": (base_stream, twin)},
    ).satisfies(solo)
    assert schema_keyed_cap.model_copy(
        update={"stream_profiles": (twin,)},
    ).satisfies(solo)  # "other-channel-name" is not an interface channel; schema route ok

    # 12. camera_calibration mismatch refuses binding, matching satisfies
    calib_stream = base_stream.model_copy(update={"camera_calibration": {"focal_length": 50.0}})
    calib_interface = interface.model_copy(update={"streams": (calib_stream,)})
    calib_mismatch = matching_cap.model_copy(
        update={
            "stream_profiles": (
                base_stream.model_copy(update={"camera_calibration": {"focal_length": 35.0}}),
            )
        }
    )
    assert not calib_mismatch.satisfies(calib_interface)
    calib_match = matching_cap.model_copy(update={"stream_profiles": (calib_stream,)})
    assert calib_match.satisfies(calib_interface)

    # 13. Duplicate stream_profiles IDs on CapabilitySpec rejected with TaskContractError
    with pytest.raises(TaskContractError, match="duplicate stream_profile id"):
        CapabilitySpec(
            interface="kinematics-control",
            interaction_modes=(InteractionMode.CLOSED_LOOP,),
            observations=("kinematic-telemetry",),
            stream_profiles=(base_stream, base_stream),
        )


def _stream(**kw: Any) -> StreamSpec:
    base: dict[str, Any] = {
        "id": "cam",
        "schema_id": "video-obs",
        "adapter": "video-laparoscopic",
        "adapter_digest": "a" * 64,
    }
    base.update(kw)
    return StreamSpec(**base)


def _cap(profiles: tuple[StreamSpec, ...], **kw: Any) -> CapabilitySpec:
    base: dict[str, Any] = {
        "interface": "k",
        "interaction_modes": (InteractionMode.SINGLE_TURN,),
        "observations": ("video-obs",),
        "outputs": ("pred",),
        "modalities": ("video-laparoscopic",),
        "stream_profiles": profiles,
    }
    base.update(kw)
    return CapabilitySpec(**base)


def _iface(streams: tuple[StreamSpec, ...]) -> InterfaceSpec:
    return InterfaceSpec(
        id="k",
        interaction_mode=InteractionMode.SINGLE_TURN,
        observations=("video-obs",),
        outputs=("pred",),
        streams=streams,
    )


def test_stream_rejects_impossible_geometry() -> None:
    # A zero/negative axis is not a buffer anyone can allocate; a bool is not
    # a dimension. pydantic's tuple[int, ...] silently coerces True -> 1, so
    # the contract layer must reject it, not launder it.
    with pytest.raises(TaskContractError, match="non-positive shape dimension"):
        _stream(shape=(0,))
    with pytest.raises(TaskContractError, match="non-positive shape dimension"):
        _stream(shape=(-1,))
    with pytest.raises(TaskContractError, match="non-positive shape dimension"):
        _stream(shape=(True,))
    # NaN bounds make every comparison False: the range could never be
    # enforced, so "declared" would be a lie.
    with pytest.raises(TaskContractError, match="NaN"):
        _stream(valid_range=(float("nan"), 2.0))
    # Reversed and fully unbounded ranges assert nothing; they must be
    # omitted, not declared-vacuous.
    with pytest.raises(TaskContractError, match="reversed"):
        _stream(valid_range=(5.0, 1.0))
    with pytest.raises(TaskContractError, match="bounds neither side"):
        _stream(valid_range=(float("-inf"), float("inf")))
    # One-sided ranges are real physics (depth has no ceiling): allowed.
    assert _stream(valid_range=(0.0, float("inf"))).valid_range == (0.0, float("inf"))
    assert _stream(valid_range=(float("-inf"), 0.0)).valid_range == (float("-inf"), 0.0)
    # Fully finite ranges are the ordinary case; a guard that rejected them
    # would make valid_range unusable, so pin the happy path explicitly.
    assert _stream(valid_range=(0.0, 1.0)).valid_range == (0.0, 1.0)
    assert _stream(valid_range=(-5.0, 5.0)).valid_range == (-5.0, 5.0)
    assert _stream(valid_range=(2.0, 2.0)).valid_range == (2.0, 2.0)  # degenerate but sound


def test_camera_calibration_is_deeply_immutable() -> None:
    # A pinned calibration a caller can still edit is not a pin: digest the
    # stream, ship it, then mutate the dict and the refused binding holds.
    stream = _stream(camera_calibration={"intrinsics": {"fx": 50.0, "dist": [0.1, 0.2]}})
    # ``Any`` alias: the declared type is Mapping (read-only), but the whole
    # point of this test is that mutation attempts raise even when attempted
    # through the object the caller still holds.
    cal: Any = stream.camera_calibration
    assert isinstance(cal, dict)  # it really is a dict subclass, so these are live paths
    with pytest.raises(TypeError, match="deeply immutable"):
        cal["fx"] = 999.0
    with pytest.raises(TypeError, match="deeply immutable"):
        cal["intrinsics"]["fx"] = 999.0
    with pytest.raises(TypeError, match="deeply immutable"):
        cal.update({"fx": 1.0})
    with pytest.raises(TypeError, match="deeply immutable"):
        cal |= {"fx": 1.0}
    with pytest.raises(TypeError, match="deeply immutable"):
        del cal["intrinsics"]
    with pytest.raises(TypeError, match="deeply immutable"):
        cal.setdefault("fx", 1.0)
    with pytest.raises(TypeError, match="deeply immutable"):
        cal.pop("intrinsics")
    with pytest.raises(TypeError, match="deeply immutable"):
        cal.popitem()
    with pytest.raises(TypeError, match="deeply immutable"):
        cal.clear()
    # Nothing leaked through the failed mutations, and equality stays
    # content-based. The canonical form holds tuples where JSON has arrays
    # (that is what makes it deep-immutable), so compare against that.
    assert cal == {"intrinsics": {"fx": 50.0, "dist": (0.1, 0.2)}}
    assert isinstance(cal["intrinsics"]["dist"], tuple)
    # ``|`` without assignment stays available from dict and yields a fresh,
    # ordinary dict — the pin itself is untouched either way.
    merged = cal | {"extra": 1.0}
    assert merged["extra"] == 1.0
    assert "extra" not in cal


def test_camera_calibration_rejects_non_json_values() -> None:
    bad_values: list[Any] = [
        {"fx": float("nan")},
        {"fx": float("inf")},
        {1: "x"},
        {"ok": {"nested": {2.5: 1}}},
    ]
    for bad in bad_values:
        with pytest.raises(TaskContractError):
            _stream(camera_calibration=bad)


def test_zero_valued_calibration_counts_as_declared() -> None:
    # fx=0 / skew=False are real calibration values. Truthiness of the payload
    # would read them as "not declared" and let an undeclared profile bind.
    zeroed = _stream(camera_calibration={"fx": 0.0})
    iface = _iface((zeroed,))
    assert not _cap((_stream(),)).satisfies(iface)  # profile declares nothing
    assert _cap((_stream(camera_calibration={"fx": 0.0}),)).satisfies(iface)


def test_calibration_equality_ignores_key_order_and_uses_content() -> None:
    a = _stream(camera_calibration={"a": 1.0, "b": {"c": [1, 2]}})
    b = _stream(camera_calibration={"b": {"c": [1, 2]}, "a": 1.0})
    iface = _iface((a,))
    assert _cap((b,)).satisfies(iface)  # same content, different insertion order
    c = _stream(camera_calibration={"a": 1.0, "b": {"c": [1, 3]}})
    assert not _cap((c,)).satisfies(iface)  # deep content differs


def test_stream_identity_adapter_digest_and_source_are_enforced() -> None:
    plain = _stream()
    iface = _iface((plain,))
    # Same channel, same pinned plugin: binds.
    assert _cap((plain,)).satisfies(iface)
    # Different plugin build under the same channel is a different contract.
    assert not _cap((_stream(adapter_digest="b" * 64),)).satisfies(iface)
    assert not _cap((_stream(adapter="other-adapter"),)).satisfies(iface)
    # ``source`` selects the observation slice the channel carries: "$" vs a
    # pointer are different channels even under one id, and it is compared on
    # both sides so neither party can silently re-locate the data.
    assert not _cap((_stream(source="depth"),)).satisfies(iface)
    pointer_iface = _iface((_stream(source="depth"),))
    assert _cap((_stream(source="depth"),)).satisfies(pointer_iface)
    assert not _cap((_stream(),)).satisfies(pointer_iface)
