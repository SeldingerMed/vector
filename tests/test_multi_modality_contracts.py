"""Tests for multi-modality contracts, GateKind extensions, and ModalityAdapter."""

from collections.abc import Iterator
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


def test_interface_interactive_streams_rejected() -> None:
    # The interactive agent route does not apply the stream preprocessing
    # pipeline, so an interactive interface that declares a pinned stream would
    # bind while handing the agent a schema it did not declare. Reject the
    # combination at bind time rather than silently diverging.
    stream = StreamSpec(
        id="s",
        schema_id="obs",
        adapter="video-laparoscopic",
        adapter_digest="a" * 64,
    )
    with pytest.raises(TaskContractError, match="cannot declare streams"):
        InterfaceSpec(
            id="i",
            interaction_mode=InteractionMode.INTERACTIVE,
            observations=("obs",),
            outputs=("out",),
            streams=(stream,),
        )


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
        "ada92b5e4c9cbe363980f8e657ba08ebc7e63b32fda61006b588b74e52c14205"
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
