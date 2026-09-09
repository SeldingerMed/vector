"""Tests for modality adapters (Video, Endoluminal, Fluoroscopy, Kinematics)."""

from __future__ import annotations

from pathlib import Path

from or_audit.eval.adapters import (
    CatheterGuidewireAction,
    EndoluminalAction,
    EndoluminalAdapter,
    EndoluminalObservation,
    FluoroscopyAdapter,
    FluoroscopyObservation,
    KinematicAction,
    KinematicObservation,
    KinematicsAdapter,
    VideoAdapter,
    VideoFrameObservation,
    VideoToolAction,
    get_adapter,
    require_adapter,
)
from or_audit.eval.enums import ModalityKind
from or_audit.eval.loader import load_task


def test_video_adapter_lifecycle() -> None:
    adapter = require_adapter(ModalityKind.VIDEO_LAPAROSCOPIC)
    assert isinstance(adapter, VideoAdapter)

    # Valid observations
    obs_obj = VideoFrameObservation(frame_index=10, timestamp_ms=333.3)
    assert adapter.validate_observation(obs_obj)
    assert adapter.validate_observation({"frame_index": 10, "timestamp_ms": 333.3})
    assert not adapter.validate_observation(None)

    # Valid actions
    act_obj = VideoToolAction(tool_id="grasper_1", action_kind="grasp")
    assert adapter.validate_action(act_obj)
    assert adapter.validate_action({"tool_id": "grasper_1"})
    assert adapter.validate_action("cut")
    assert not adapter.validate_action(None)

    # Preprocessing / Postprocessing
    pre = adapter.preprocess_observation(
        {"frame_index": 5, "timestamp_ms": 166.6, "active_tools": ["bipolar"]}
    )
    assert isinstance(pre, VideoFrameObservation)
    assert pre.frame_index == 5
    assert pre.active_tools == ("bipolar",)

    post = adapter.postprocess_action(
        {"tool_id": "hook", "action_kind": "cauterize", "electrocautery_active": True}
    )
    assert isinstance(post, VideoToolAction)
    assert post.tool_id == "hook"
    assert post.electrocautery_active is True

    # Safety extraction
    safety = adapter.extract_safety_state(
        {
            "info": {
                "distance_to_critical_structure": 1.8,
                "unmonitored_cautery": False,
                "bleeding_detected": True,
            }
        }
    )
    assert safety["distance_to_critical_structure"] == 1.8
    assert safety["bleeding_detected"] is True


def test_endoluminal_adapter_lifecycle() -> None:
    adapter = require_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY)
    assert isinstance(adapter, EndoluminalAdapter)

    # Observations
    obs = EndoluminalObservation(frame_index=1, airway_id="RB1", target_distance_mm=12.5)
    assert adapter.validate_observation(obs)
    assert adapter.validate_observation({"airway_id": "LB2"})

    pre = adapter.preprocess_observation(
        {
            "frame_index": 2,
            "airway_id": "RB2",
            "em_sensor_pose": [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
            "target_distance_mm": 8.0,
        }
    )
    assert isinstance(pre, EndoluminalObservation)
    assert pre.airway_id == "RB2"
    assert pre.em_sensor_pose == (10.0, 20.0, 30.0, 0.0, 0.0, 0.0)

    # Actions
    act = EndoluminalAction(bend_angle_deg=45.0, insertion_mm=5.0, biopsy_deployed=True)
    assert adapter.validate_action(act)
    assert adapter.validate_action({"bend_angle_deg": 30.0})

    post = adapter.postprocess_action({"bend_angle_deg": 30.0, "insertion_mm": 2.0})
    assert isinstance(post, EndoluminalAction)
    assert post.bend_angle_deg == 30.0

    # Safety extraction
    safety = adapter.extract_safety_state(
        {"info": {"contact_force_n": 1.2, "wall_puncture": False, "off_target_biopsy": False}}
    )
    assert safety["contact_force_n"] == 1.2
    assert safety["wall_puncture"] is False


def test_fluoroscopy_adapter_lifecycle() -> None:
    adapter = require_adapter(ModalityKind.FLUOROSCOPY_DSA)
    assert isinstance(adapter, FluoroscopyAdapter)

    # Observations
    obs = FluoroscopyObservation(frame_index=1, dsa_contrast_active=True)
    assert adapter.validate_observation(obs)
    assert adapter.validate_observation({"carm_angles": [15.0, -10.0]})

    pre = adapter.preprocess_observation(
        {"frame_index": 4, "dsa_contrast_active": True, "carm_angles": [20.0, 0.0]}
    )
    assert isinstance(pre, FluoroscopyObservation)
    assert pre.carm_angles == (20.0, 0.0)

    # Actions
    act = CatheterGuidewireAction(insertion_step_mm=1.5, rotation_step_deg=45.0)
    assert adapter.validate_action(act)

    post = adapter.postprocess_action(
        {"insertion_step_mm": 2.0, "rotation_step_deg": 90.0, "balloon_inflation_psi": 6.0}
    )
    assert isinstance(post, CatheterGuidewireAction)
    assert post.balloon_inflation_psi == 6.0

    # Safety extraction
    safety = adapter.extract_safety_state(
        {
            "info": {
                "max_pen": 0.02,
                "radiation_dose_mgy": 14.5,
                "contrast_injected_ml": 5.0,
                "dissection_risk": False,
            }
        }
    )
    assert safety["max_pen"] == 0.02
    assert safety["radiation_dose_mgy"] == 14.5


def test_kinematics_adapter_lifecycle() -> None:
    adapter = require_adapter(ModalityKind.ROBOTIC_KINEMATICS)
    assert isinstance(adapter, KinematicsAdapter)

    # Observations
    obs = KinematicObservation(
        joint_positions=(0.1, 0.2, 0.3),
        ee_position_xyz=(100.0, 200.0, 50.0),
        haptic_boundary_distance_mm=0.25,
    )
    assert adapter.validate_observation(obs)

    pre = adapter.preprocess_observation(
        {
            "joint_positions": [0.1, 0.2],
            "ee_position_xyz": [10.0, 20.0, 30.0],
            "haptic_boundary_distance_mm": 0.4,
            "measured_force_n": 2.5,
        }
    )
    assert isinstance(pre, KinematicObservation)
    assert pre.haptic_boundary_distance_mm == 0.4

    # Actions
    act = KinematicAction(joint_velocity_cmd=(0.01, 0.02), cutting_rpm=15000.0)
    assert adapter.validate_action(act)

    post = adapter.postprocess_action(
        {"joint_velocity_cmd": [0.05, -0.05], "cutting_rpm": 20000.0, "gripper_state": 1.0}
    )
    assert isinstance(post, KinematicAction)
    assert post.cutting_rpm == 20000.0

    # Safety extraction
    safety = adapter.extract_safety_state(
        {
            "info": {
                "haptic_boundary_overshoot_mm": 0.1,
                "excessive_traction_force_n": 3.2,
                "collision_detected": False,
            }
        }
    )
    assert safety["haptic_boundary_overshoot_mm"] == 0.1


def test_endoscopic_and_orthopedic_registered_aliases() -> None:
    endo_adapter = get_adapter(ModalityKind.VIDEO_ENDOSCOPIC)
    assert endo_adapter is not None
    assert endo_adapter.modality == ModalityKind.VIDEO_ENDOSCOPIC

    ortho_adapter = get_adapter(ModalityKind.ORTHOPEDIC_POINTCLOUD)
    assert ortho_adapter is not None
    assert ortho_adapter.modality == ModalityKind.ORTHOPEDIC_POINTCLOUD


def test_reference_tasks_load_and_validate() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Laparoscopic Cholecystectomy CVS
    cholec_task = load_task(repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs")
    assert cholec_task.id == "laparoscopic-cholec-cvs"
    assert cholec_task.metadata.modality == "video-laparoscopic"
    assert cholec_task.verifier.gates[0].id == "critical_structure_misid"
    assert cholec_task.verifier.headline == "cvs_safe_accuracy"
    cholec_task.assert_runnable()

    # 2. Robotic Bronchoscopy
    broncho_task = load_task(repo_root / "docs/examples/tasks/broncho-airway-nav")
    assert broncho_task.id == "broncho-airway-nav"
    assert broncho_task.metadata.modality == "airway-bronchoscopy"
    assert broncho_task.verifier.gates[0].id == "airway_wall_puncture"
    assert broncho_task.verifier.gates[0].threshold == 1.5
    assert broncho_task.verifier.headline == "safe_navigation"

    # 3. Orthopedic Bone Milling
    ortho_task = load_task(repo_root / "docs/examples/tasks/ortho-burr-safe")
    assert ortho_task.id == "ortho-burr-safe"
    assert ortho_task.metadata.modality == "robotic-kinematics"
    assert ortho_task.verifier.gates[0].id == "haptic_boundary_overshoot"
    assert ortho_task.verifier.gates[0].threshold == 0.5
    assert ortho_task.verifier.headline == "safe_resection"


def test_reference_verifiers_end_to_end_scoring() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    from or_audit.eval.verifier import score_context

    # Laparoscopic Verifier
    cholec_task = load_task(repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs")
    res_pass = score_context(
        task=cholec_task,
        task_dir=repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs",
        agent_identity="test-vlm",
        seed=0,
        context={
            "prediction": {"cvs_achieved": True, "critical_structure": "cystic_duct"},
            "label": {"cvs_achieved": True, "critical_structure": "cystic_duct"},
        },
    )
    assert res_pass.gates[0].status == "pass"
    m_pass = res_pass.metric("cvs_safe_accuracy")
    assert m_pass is not None
    assert m_pass.value is True
    # Laparoscopic Verifier with critical violation (misidentified CBD)
    res_fail = score_context(
        task=cholec_task,
        task_dir=repo_root / "docs/examples/tasks/laparoscopic-cholec-cvs",
        agent_identity="test-vlm",
        seed=0,
        context={
            "prediction": {"cvs_achieved": True, "critical_structure": "cystic_duct"},
            "label": {"cvs_achieved": False, "critical_structure": "common_bile_duct"},
        },
    )
    assert res_fail.gates[0].status == "fail"
    m_fail = res_fail.metric("cvs_safe_accuracy")
    assert m_fail is not None
    assert m_fail.value is False
    # Broncho Verifier
    broncho_task = load_task(repo_root / "docs/examples/tasks/broncho-airway-nav")
    res_broncho = score_context(
        task=broncho_task,
        task_dir=repo_root / "docs/examples/tasks/broncho-airway-nav",
        agent_identity="test-policy",
        seed=0,
        context={"info": {"target_reached": True, "max_contact_force_n": 0.8, "diverged": False}},
    )
    assert res_broncho.gates[0].status == "pass"
    m_broncho = res_broncho.metric("safe_navigation")
    assert m_broncho is not None
    assert m_broncho.value is True
    # Ortho Verifier
    ortho_task = load_task(repo_root / "docs/examples/tasks/ortho-burr-safe")
    res_ortho = score_context(
        task=ortho_task,
        task_dir=repo_root / "docs/examples/tasks/ortho-burr-safe",
        agent_identity="test-policy",
        seed=0,
        context={"info": {"resection_complete": True, "max_overshoot_mm": 0.2, "diverged": False}},
    )
    assert res_ortho.gates[0].status == "pass"
    m_ortho = res_ortho.metric("safe_resection")
    assert m_ortho is not None
    assert m_ortho.value is True


def test_adapters_null_handling() -> None:
    # Test None and null handling in dicts across adapters
    v_adapter = require_adapter(ModalityKind.VIDEO_LAPAROSCOPIC)
    v_pre = v_adapter.preprocess_observation(
        {"frame_index": 1, "timestamp_ms": None, "optical_flow": None, "extra": None}
    )
    assert isinstance(v_pre, VideoFrameObservation)
    assert v_pre.optical_flow is None
    assert v_pre.timestamp_ms is None
    assert v_pre.image_uri is None
    assert v_pre.width is None
    assert v_pre.height is None

    b_adapter = require_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY)
    b_pre = b_adapter.preprocess_observation(
        {"frame_index": 1, "target_distance_mm": None, "em_sensor_pose": None}
    )
    assert isinstance(b_pre, EndoluminalObservation)
    assert b_pre.target_distance_mm is None

    f_adapter = require_adapter(ModalityKind.FLUOROSCOPY_DSA)
    f_pre = f_adapter.preprocess_observation(
        {"frame_index": 1, "target_distance_mm": None, "carm_angles": None}
    )
    assert isinstance(f_pre, FluoroscopyObservation)
    assert f_pre.target_distance_mm is None

    k_adapter = require_adapter(ModalityKind.ROBOTIC_KINEMATICS)
    k_pre = k_adapter.preprocess_observation(
        {"joint_positions": None, "haptic_boundary_distance_mm": None}
    )
    assert isinstance(k_pre, KinematicObservation)
    assert k_pre.haptic_boundary_distance_mm is None


def test_adapters_get_schema_spec() -> None:
    v_schema = require_adapter(ModalityKind.VIDEO_LAPAROSCOPIC).get_schema_spec()
    assert v_schema["observation_type"] == "VideoFrameObservation"

    b_schema = require_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY).get_schema_spec()
    assert b_schema["observation_type"] == "EndoluminalObservation"

    f_schema = require_adapter(ModalityKind.FLUOROSCOPY_DSA).get_schema_spec()
    assert f_schema["observation_type"] == "FluoroscopyObservation"

    k_schema = require_adapter(ModalityKind.ROBOTIC_KINEMATICS).get_schema_spec()
    assert k_schema["observation_type"] == "KinematicObservation"
